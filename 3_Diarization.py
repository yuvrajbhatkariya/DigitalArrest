import sounddevice as sd
import numpy as np
import queue
import threading
import webrtcvad
import time
import io
from collections import deque, defaultdict
from faster_whisper import WhisperModel
from resemblyzer import VoiceEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.io import wavfile

# ================ CONFIG (tune these) =================
SAMPLE_RATE = 16000
BLOCK_SIZE = 4000
CAPTURE_WINDOW = 2.0         # seconds to accumulate before processing
MIN_SEGMENT_DUR = 0.25       # ignore segments shorter than this (s)
NEW_SPKR_MIN_DUR = 0.9       # only consider long segments for new-speaker creation (s)
NEW_SPKR_CONSECUTIVE = 2     # how many consecutive low-match long segments before new speaker
SIM_ASSIGN = 0.82            # cosine similarity threshold to assign to existing speaker
SIM_NEW_CAND = 0.65          # if best similarity > this but < SIM_ASSIGN treat as candidate (not immediate new)
MERGE_THRESHOLD = 0.90       # if two profiles are > this, merge them
MAX_SPEAKERS = 4             # safety cap (set to 2 for strict two-party calls)
PROFILE_UPDATE_ALPHA = 0.85  # weight for running-average update
MERGE_RUN_INTERVAL = 30.0    # seconds between automatic profile merging checks
MIC_INDEX = 0

# ================ GLOBALS =================
audio_queue = queue.Queue()
transcription_queue = queue.Queue()
conversation_turns = deque(maxlen=200)

vad = webrtcvad.Vad(2)  # aggressiveness 0-3
encoder = VoiceEncoder()
whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")

speaker_profiles = {}  # { "User 1": embedding (np.array) }
speaker_count = 0
unknown_consecutive = 0
last_unknown_time = 0.0

# keep last few assignments for smoothing if needed
recent_assignments = deque(maxlen=10)

# ================ AUDIO CALLBACK =================
def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)
    audio_queue.put(indata.copy())

# ================ UTIL: VAD-based segmenter =================
def frame_generator(frame_ms, audio, sample_rate):
    """Yield frames (int16 numpy) of length frame_ms from int16 audio"""
    frame_size = int(sample_rate * frame_ms / 1000)
    offset = 0
    while offset + frame_size <= len(audio):
        yield audio[offset:offset+frame_size]
        offset += frame_size

def vad_segments(float_audio, sample_rate, frame_ms=20, padding_ms=200):
    """
    Return list of (start_sample, end_sample) speech segments (int indices into float_audio).
    Uses webrtcvad on 10/20/30ms frames and merges contiguous speech frames with padding.
    """
    int16_audio = (float_audio * 32768).astype(np.int16)
    frame_size = int(sample_rate * frame_ms / 1000)
    num_frames = len(int16_audio) // frame_size
    if num_frames == 0:
        return []

    speech_flags = []
    for i in range(num_frames):
        start = i * frame_size
        frame = int16_audio[start:start+frame_size]
        if len(frame) < frame_size:
            speech_flags.append(False)
        else:
            try:
                speech_flags.append(vad.is_speech(frame.tobytes(), sample_rate))
            except Exception:
                speech_flags.append(False)

    # Convert flags to segments (in samples), with padding
    segments = []
    start_frame = None
    for i, f in enumerate(speech_flags):
        if f and start_frame is None:
            start_frame = i
        elif not f and start_frame is not None:
            end_frame = i
            start_sample = max(0, int((start_frame * frame_ms - padding_ms) / 1000.0 * sample_rate))
            end_sample = min(len(int16_audio), int(((end_frame) * frame_ms + padding_ms) / 1000.0 * sample_rate))
            segments.append((start_sample, end_sample))
            start_frame = None
    if start_frame is not None:
        start_sample = max(0, int((start_frame * frame_ms - padding_ms) / 1000.0 * sample_rate))
        end_sample = len(int16_audio)
        segments.append((start_sample, end_sample))

    # merge very close segments
    merged = []
    for s,e in segments:
        if not merged:
            merged.append([s,e])
        else:
            if s <= merged[-1][1] + int(0.2 * sample_rate):
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s,e])
    return [(s,e) for s,e in merged]

# ================ SPEAKER LOGIC =================
def assign_or_create_speaker(embedding, seg_duration):
    """
    Returns assigned speaker name.
    Logic:
     - If best similarity >= SIM_ASSIGN -> assign and update profile
     - Else if best similarity >= SIM_NEW_CAND and seg_duration >= NEW_SPKR_MIN_DUR -> treat as candidate (increase unknown_consecutive)
     - Else if best similarity < SIM_NEW_CAND and seg_duration >= NEW_SPKR_MIN_DUR and unknown_consecutive >= NEW_SPKR_CONSECUTIVE -> create new speaker
     - Else assign to best existing if exists, or create if no profiles.
    """
    global speaker_profiles, speaker_count, unknown_consecutive, last_unknown_time

    if not speaker_profiles:
        speaker_count += 1
        name = f"User {speaker_count}"
        speaker_profiles[name] = embedding.copy()
        recent_assignments.append(name)
        return name

    # compute similarities
    best_name = None
    best_score = -1.0
    for name, prof in speaker_profiles.items():
        score = float(cosine_similarity(embedding.reshape(1, -1), prof.reshape(1, -1))[0][0])
        if score > best_score:
            best_score = score
            best_name = name

    # If strong match -> assign
    if best_score >= SIM_ASSIGN:
        # update profile (running average)
        speaker_profiles[best_name] = PROFILE_UPDATE_ALPHA * speaker_profiles[best_name] + (1.0 - PROFILE_UPDATE_ALPHA) * embedding
        unknown_consecutive = 0
        recent_assignments.append(best_name)
        return best_name

    # Candidate region: moderate similarity
    if best_score >= SIM_NEW_CAND and seg_duration >= NEW_SPKR_MIN_DUR:
        # treat as candidate but require consecutive occurrences to create new
        unknown_consecutive += 1
        last_unknown_time = time.time()
        # if reach threshold and we have capacity to create new speaker
        if unknown_consecutive >= NEW_SPKR_CONSECUTIVE and len(speaker_profiles) < MAX_SPEAKERS:
            speaker_count += 1
            new_name = f"User {speaker_count}"
            speaker_profiles[new_name] = embedding.copy()
            unknown_consecutive = 0
            recent_assignments.append(new_name)
            print(f"[INFO] New speaker created (candidate path): {new_name} (best_score={best_score:.2f})")
            return new_name
        else:
            # assign to best_name temporarily (so short utterances don't explode)
            speaker_profiles[best_name] = PROFILE_UPDATE_ALPHA * speaker_profiles[best_name] + (1.0 - PROFILE_UPDATE_ALPHA) * embedding
            recent_assignments.append(best_name)
            return best_name

    # Very low similarity: consider creating new only if segment is long and we saw consecutive unmatched segments
    if seg_duration >= NEW_SPKR_MIN_DUR:
        unknown_consecutive += 1
        last_unknown_time = time.time()
        if unknown_consecutive >= NEW_SPKR_CONSECUTIVE and len(speaker_profiles) < MAX_SPEAKERS:
            speaker_count += 1
            new_name = f"User {speaker_count}"
            speaker_profiles[new_name] = embedding.copy()
            unknown_consecutive = 0
            recent_assignments.append(new_name)
            print(f"[INFO] New speaker created (low-sim path): {new_name} (best_score={best_score:.2f})")
            return new_name

    # otherwise, fallback: assign to best_name (prevents noise proliferation)
    speaker_profiles[best_name] = PROFILE_UPDATE_ALPHA * speaker_profiles[best_name] + (1.0 - PROFILE_UPDATE_ALPHA) * embedding
    recent_assignments.append(best_name)
    return best_name

# ================ PROFILE MERGING (background) =================
def merge_similar_profiles():
    """
    Periodically merge profiles whose cosine similarity > MERGE_THRESHOLD.
    Merge by averaging embeddings and renaming to the lower-numbered user.
    """
    global speaker_profiles
    names = list(speaker_profiles.keys())
    merged = True
    while merged:
        merged = False
        n = len(names)
        for i in range(n):
            for j in range(i+1, n):
                a = names[i]
                b = names[j]
                if a not in speaker_profiles or b not in speaker_profiles:
                    continue
                sim = float(cosine_similarity(
                    speaker_profiles[a].reshape(1, -1),
                    speaker_profiles[b].reshape(1, -1)
                )[0][0])
                if sim >= MERGE_THRESHOLD:
                    # merge b into a
                    speaker_profiles[a] = (speaker_profiles[a] + speaker_profiles[b]) / 2.0
                    del speaker_profiles[b]
                    print(f"[INFO] Merged speaker profiles: {a} <- {b} (sim={sim:.3f})")
                    merged = True
                    names = list(speaker_profiles.keys())
                    break
            if merged:
                break

def profile_merger_worker():
    while True:
        time.sleep(MERGE_RUN_INTERVAL)
        try:
            merge_similar_profiles()
        except Exception as e:
            print("Merge error:", e)

# ================ DIARIZATION WORKER =================
def diarization_worker():
    buffer = np.zeros((0,), dtype=np.float32)
    while True:
        audio = audio_queue.get()
        buffer = np.concatenate((buffer, audio[:, 0]))

        if len(buffer) >= int(SAMPLE_RATE * CAPTURE_WINDOW):
            proc_buffer = buffer.copy()
            buffer = np.zeros((0,), dtype=np.float32)

            # find speech segments (sample indices)
            segs = vad_segments(proc_buffer, SAMPLE_RATE, frame_ms=20, padding_ms=150)
            if not segs:
                # no speech -> skip
                continue

            for (samp_s, samp_e) in segs:
                seg = proc_buffer[samp_s:samp_e]
                seg_dur = (samp_e - samp_s) / SAMPLE_RATE
                if seg_dur < MIN_SEGMENT_DUR:
                    continue  # ignore tiny blips

                # normalize audio to -0.99..0.99
                maxv = np.max(np.abs(seg)) if seg.size > 0 else 1.0
                if maxv > 0:
                    seg_n = seg / maxv * 0.99
                else:
                    seg_n = seg

                # compute embedding (catch exceptions)
                try:
                    emb = encoder.embed_utterance(seg_n)
                except Exception as e:
                    print("Embedding error:", e)
                    continue

                # assign speaker with robust logic
                speaker = assign_or_create_speaker(emb, seg_dur)

                # put segment for transcription (we use the actual segment)
                transcription_queue.put((speaker, seg_n.copy()))
                # small debug print
                print(f"[DBG] SegDur={seg_dur:.2f}s -> {speaker}")

# ================ TRANSCRIPTION WORKER =================
def transcription_worker():
    while True:
        speaker, audio_chunk = transcription_queue.get()
        try:
            segments, _ = whisper_model.transcribe(
                audio_chunk,
                language="en",
                vad_filter=True,
                beam_size=5
            )
            text = " ".join([seg.text.strip() for seg in segments]).strip()
            if text:
                conversation_turns.append((speaker, text))
                print(f"[{speaker}] {text}")
        except Exception as e:
            print("Whisper error:", e)

# ================ MAIN =================
def main():
    print("🎙️ Starting improved real-time diarization (stable profiles)")
    # start background merge worker
    threading.Thread(target=profile_merger_worker, daemon=True).start()
    threading.Thread(target=diarization_worker, daemon=True).start()
    threading.Thread(target=transcription_worker, daemon=True).start()

    with sd.InputStream(device=MIC_INDEX, samplerate=SAMPLE_RATE, channels=1,
                        blocksize=BLOCK_SIZE, callback=audio_callback):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping.")

if __name__ == "__main__":
    main()
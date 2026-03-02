import sounddevice as sd
import numpy as np
import queue
import threading
import webrtcvad
import time
from collections import deque
from faster_whisper import WhisperModel
from resemblyzer import VoiceEncoder
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIG =================
SAMPLE_RATE = 16000
BLOCK_SIZE = 4000
PROCESS_WINDOW = 2        # seconds for speaker detection
SIM_THRESHOLD = 0.75
MIC_INDEX = 0

# ================= GLOBALS =================
audio_queue = queue.Queue()
transcription_queue = queue.Queue()
conversation_turns = deque(maxlen=50)

speaker_profiles = {}
speaker_count = 0

vad = webrtcvad.Vad(2)  # aggressiveness 0-3
encoder = VoiceEncoder()
whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")

# ================= AUDIO CALLBACK =================
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

# ================= FRAME-BASED VAD =================
def has_speech(audio_float):
    """
    Proper WebRTC VAD usage:
    Must pass 10ms, 20ms or 30ms frames.
    """
    int16_audio = (audio_float * 32768).astype(np.int16)

    frame_duration = 20  # ms
    frame_size = int(SAMPLE_RATE * frame_duration / 1000)

    num_frames = len(int16_audio) // frame_size
    if num_frames == 0:
        return False

    speech_frames = 0

    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        frame = int16_audio[start:end]

        if len(frame) == frame_size:
            if vad.is_speech(frame.tobytes(), SAMPLE_RATE):
                speech_frames += 1

    return speech_frames > 0.4 * num_frames  # 40% rule

# ================= SPEAKER ASSIGNMENT =================
def assign_speaker(new_embedding):
    global speaker_profiles, speaker_count

    if not speaker_profiles:
        speaker_count += 1
        speaker_profiles[f"User {speaker_count}"] = new_embedding
        return f"User {speaker_count}"

    best_speaker = None
    best_score = 0

    for name, emb in speaker_profiles.items():
        score = cosine_similarity(
            new_embedding.reshape(1, -1),
            emb.reshape(1, -1)
        )[0][0]

        if score > best_score:
            best_score = score
            best_speaker = name

    if best_score > SIM_THRESHOLD:
        # Update running average (stabilizes identity)
        speaker_profiles[best_speaker] = (
            0.8 * speaker_profiles[best_speaker] +
            0.2 * new_embedding
        )
        return best_speaker
    else:
        speaker_count += 1
        speaker_profiles[f"User {speaker_count}"] = new_embedding
        return f"User {speaker_count}"

# ================= DIARIZATION THREAD =================
def diarization_worker():
    buffer = np.zeros((0,), dtype=np.float32)

    while True:
        audio = audio_queue.get()
        buffer = np.concatenate((buffer, audio[:, 0]))

        if len(buffer) >= SAMPLE_RATE * PROCESS_WINDOW:

            if has_speech(buffer):
                try:
                    embedding = encoder.embed_utterance(buffer)
                    speaker = assign_speaker(embedding)

                    # Send to transcription
                    transcription_queue.put((speaker, buffer.copy()))

                except Exception as e:
                    print("Embedding error:", e)

            buffer = np.zeros((0,), dtype=np.float32)

# ================= TRANSCRIPTION THREAD =================
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

            text = " ".join([seg.text.strip() for seg in segments])

            if text:
                conversation_turns.append((speaker, text))
                print(f"[{speaker}] {text}")

        except Exception as e:
            print("Whisper error:", e)

# ================= MAIN =================
def main():
    print("🎙️ Real-time diarization + Whisper medium started")

    threading.Thread(target=diarization_worker, daemon=True).start()
    threading.Thread(target=transcription_worker, daemon=True).start()

    with sd.InputStream(
        device=MIC_INDEX,
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=BLOCK_SIZE,
        callback=audio_callback,
    ):
        while True:
            time.sleep(0.1)

if __name__ == "__main__":
    main()
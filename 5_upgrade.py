# main.py

import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, VADIterator

from Prompts.claude_prompt import process_turn

# ============== AUDIO CONFIG ==============
SAMPLE_RATE = 16000
BLOCK_SIZE = 512
WHISPER_SIZE = "medium"
# ==========================================

audio_q = queue.Queue()

print("Loading models...")

whisper = WhisperModel(
    WHISPER_SIZE,
    device="cpu",
    compute_type="int8",
    cpu_threads=2
)

vad_model = load_silero_vad()
vad = VADIterator(
    vad_model,
    threshold=0.5,
    sampling_rate=SAMPLE_RATE,
    min_silence_duration_ms=600
)


def audio_callback(indata, frames, time_info, status):
    audio_q.put(indata.copy().flatten())


print("\n🛡 Guardian AI Running...\n")

stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype='float32',
    blocksize=BLOCK_SIZE,
    callback=audio_callback
)
stream.start()

buffer = np.array([], dtype=np.float32)
speech_active = False

try:
    while True:
        chunk = audio_q.get()
        buffer = np.concatenate((buffer, chunk))

        speech_dict = vad(chunk)

        if speech_dict:
            if speech_dict.get("start"):
                speech_active = True

            elif speech_dict.get("end") and speech_active:
                speech_active = False

                if len(buffer) > SAMPLE_RATE * 0.4:
                    segs, _ = whisper.transcribe(
                        buffer,
                        language="en",
                        beam_size=1,
                        best_of=1
                    )

                    text = " ".join(s.text.strip() for s in segs).strip()

                    if text:
                        print("\n🗣", text)

                        threading.Thread(
                            target=process_turn,
                            args=(text,),
                            daemon=True
                        ).start()

                buffer = np.array([], dtype=np.float32)

except KeyboardInterrupt:
    print("\nCall ended.")
finally:
    stream.stop()
    stream.close()
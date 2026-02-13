import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue
from collections import deque

SAMPLE_RATE = 16000
BLOCK_SIZE = 4000   # ~250 ms
MIC_INDEX = 0       

audio_queue = queue.Queue()
conversation_turns = deque(maxlen=10)


def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def main():
    model = WhisperModel(
        "medium",          #  "large-v3" 
        device="cpu",
        compute_type="int8"
    )

    print("🎙️ Speak clearly (Ctrl+C to stop)...")

    with sd.InputStream(
        device=MIC_INDEX,
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=BLOCK_SIZE,
        callback=audio_callback,
    ):
        buffer = np.zeros((0,), dtype=np.float32)

        while True:
            audio = audio_queue.get()
            buffer = np.concatenate((buffer, audio[:, 0]))

            # process every ~4 seconds
            if len(buffer) > SAMPLE_RATE * 4:
                segments, info = model.transcribe(
                    buffer,
                    language="en",
                    vad_filter=True,
                    beam_size=5,
                )

                for seg in segments:
                    turn_text = seg.text.strip().lower()
                    conversation_turns.append(turn_text)
                    print("🗣️", turn_text)

                buffer = np.zeros((0,), dtype=np.float32)

if __name__ == "__main__":
    main()

import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
MIC_INDEX = 0  # <-- CONFIRMED

def callback(indata, frames, time, status):
    volume = np.linalg.norm(indata) * 10
    print("Mic level:", volume)

with sd.InputStream(
    device=MIC_INDEX,
    channels=1,
    samplerate=SAMPLE_RATE,
    callback=callback,
):
    print("🎙️ Speak now (Ctrl+C to stop)...")
    while True:
        pass

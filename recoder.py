import sounddevice as sd
import numpy as np
import os
from scipy.io.wavfile import write

SAVE_FOLDER = "Recordings"  
DURATION = 5        
SAMPLE_RATE = 44100 
BASE_FILENAME = "audio"

os.makedirs(SAVE_FOLDER, exist_ok=True)

def get_next_filename(folder, base_name):
    existing = [
        f for f in os.listdir(folder)
        if f.startswith(base_name) and f.endswith(".wav")
    ]

    numbers = []
    for f in existing:
        try:
            num = int(f.replace(base_name, "").replace(".wav", "").replace("_", ""))
            numbers.append(num)
        except ValueError:
            pass

    next_num = max(numbers, default=0) + 1
    return f"{base_name}_{next_num:03d}.wav"

filename = get_next_filename(SAVE_FOLDER, BASE_FILENAME)
file_path = os.path.join(SAVE_FOLDER, filename)

print("🎙️ Recording started...")
audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype=np.int16
)
sd.wait()
print("✅ Recording finished")

write(file_path, SAMPLE_RATE, audio)
print(f"📁 Saved as: {file_path}")

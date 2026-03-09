# from transformers import pipeline

# classifier = pipeline("audio-classification",
#                       model="superb/wav2vec2-base-superb-er")

# result = classifier("Recordings/audio_003.wav")

# print(result)


# from transformers import pipeline

# classifier = pipeline("audio-classification",
#                       model="superb/wav2vec2-base-superb-er")

# result = classifier("Recordings/audio_003.wav")

# print(result)




import sounddevice as sd
import numpy as np
import librosa
from transformers import pipeline

# Load emotion classifier
classifier = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er"
)

# Emotion label mapping
emotion_map = {
    "hap": "Happy",
    "ang": "Angry",
    "sad": "Sad",
    "neu": "Neutral"
}

# Audio settings
duration = 3   # seconds per recording
sample_rate = 16000

print("Real-time emotion detection started...")

while True:
    print("\nListening...")

    # Record audio
    audio = sd.rec(int(duration * sample_rate),
                   samplerate=sample_rate,
                   channels=1)
    sd.wait()

    audio = audio.flatten()

    # Normalize audio
    audio = librosa.util.normalize(audio)

    # Run emotion prediction
    result = classifier(audio)

    emotion = result[0]["label"]
    score = result[0]["score"]

    emotion_name = emotion_map.get(emotion, emotion)

    print("Emotion:", emotion_name)
    print("Confidence:", round(score, 3)) 
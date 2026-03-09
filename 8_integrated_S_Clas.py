import sounddevice as sd
import numpy as np
import torch
import librosa
from speechbrain.pretrained import EncoderClassifier

# Load emotion model
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="tmp_emotion_model"
)

sample_rate = 16000
duration = 2

print("Real-time emotion detection started...")

while True:

    print("\nListening...")

    audio = sd.rec(int(duration * sample_rate),
                   samplerate=sample_rate,
                   channels=1)

    sd.wait()

    audio = audio.flatten()

    audio = librosa.util.normalize(audio)

    tensor = torch.tensor(audio).unsqueeze(0)

    out_prob, score, index, label = classifier.classify_batch(tensor)

    print("Emotion:", label[0])
    print("Confidence:", float(score[0]))
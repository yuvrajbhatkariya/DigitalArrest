import sounddevice as sd
import numpy as np
import queue
import json
import ollama
from collections import deque
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, VADIterator  
from Prompts.fraud_prompt import fraud_prompt

SAMPLE_RATE = 16000
BLOCK_SIZE = 512
LLM_MODEL = "phi3:mini"         
WHISPER_SIZE = "medium"       

print("Loading models...")
whisper = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
vad_model = load_silero_vad()
vad = VADIterator(vad_model, threshold=0.5, sampling_rate=SAMPLE_RATE, min_silence_duration_ms=800)

history = deque(maxlen=10)          # last 10 turns
audio_queue = queue.Queue()


def detect_fraud(current_turn: str) -> dict:
    history_text = "\n".join(list(history)[-8:])

    prompt = fraud_prompt(history_text, current_turn)

    try:
        resp = ollama.generate(
            model=LLM_MODEL,
            prompt=prompt,
            options={"temperature": 0.0}
        )
        return json.loads(resp["response"])
    except Exception as e:
        return {
            "risk_level": "low",
            "confidence": 0,
            "patterns": [],
            "reason": str(e),
            "advice": "Continue listening"
        }

def callback(indata, frames, time, status):
    if status: print(status)
    audio_queue.put(indata.copy().flatten())

print("\n🎤 Real-time Fraud Detector READY (VAD + LLM)")
print("Speak normally – it will detect natural turns\n")

stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                        blocksize=BLOCK_SIZE, callback=callback)
stream.start()

buffer = np.array([], dtype=np.float32)
speech_active = False

try:
    while True:
        chunk = audio_queue.get()
        buffer = np.concatenate((buffer, chunk))


        speech_dict = vad(chunk)
        if speech_dict is not None:
            if speech_dict.get("start") is not None:
                speech_active = True
            elif speech_dict.get("end") is not None and speech_active:
                speech_active = False
                
                if len(buffer) > SAMPLE_RATE * 0.5:   # at least 0.5s
                    segments, _ = whisper.transcribe(buffer, language="en", vad_filter=True)
                    text = " ".join(s.text.strip() for s in segments).strip()

                    if text:
                        print(f"\n🗣️ Turn: {text}")
                        history.append(text)

                        result = detect_fraud(text)
                        print(f"🚨 RISK: {result['risk_level'].upper()} ({result['confidence']}% confidence)")
                        print(f"Reason: {result['reason']}")
                        print(f"Advice: {result['advice']}\n")

                        if result["risk_level"] in ["high", "critical"]:
                            print("⚠️  IMMEDIATE ACTION: HANG UP NOW!\n")

                buffer = np.array([], dtype=np.float32)

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    stream.stop()
    stream.close()
    vad.reset_states()
import sounddevice as sd
import numpy as np
import queue
import json
import ollama
from collections import deque
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, VADIterator
from Prompts.fraud_prompt import fraud_prompt
from Prompts.summary_prompt import summary_prompt


# ================= CONFIG =================
SAMPLE_RATE = 16000
BLOCK_SIZE = 512
# LLM_MODEL = "phi3:mini"
LLM_MODEL = "mistral:instruct"
WHISPER_SIZE = "medium"

SUMMARY_THRESHOLD = 7
MAX_SUMMARIES = 5
USE_LAST_SUMMARIES = 3
# ==========================================


print("Loading models...")
whisper = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
vad_model = load_silero_vad()
vad = VADIterator(vad_model, threshold=0.5,
                  sampling_rate=SAMPLE_RATE,
                  min_silence_duration_ms=800)

audio_queue = queue.Queue()

# ===== MEMORY STRUCTURE =====
recent_turns = deque(maxlen=SUMMARY_THRESHOLD)
summaries = deque(maxlen=MAX_SUMMARIES)


# ===== LLM UTILS =====
def llm_generate(prompt, max_tokens=200, temperature=0.0):
    try:
        resp = ollama.generate(
            model=LLM_MODEL,
            prompt=prompt,
            options={"temperature": temperature, "max_tokens": max_tokens}
        )
        return resp.get("response", "")
    except Exception as e:
        print("LLM ERROR:", e)
        return ""


def summarize_block(turns):
    prompt = summary_prompt(turns)
    summary = llm_generate(prompt, max_tokens=120)
    return summary.strip()


def build_context():
    parts = []

    if summaries:
        last_summaries = list(summaries)[-USE_LAST_SUMMARIES:]
        summary_text = "\n".join(
            f"Summary {i+1}: {s}"
            for i, s in enumerate(last_summaries)
        )
        parts.append("Compressed history:\n" + summary_text)

    if recent_turns:
        turns_text = "\n".join(
            f"Turn {i+1}: {t}"
            for i, t in enumerate(recent_turns)
        )
        parts.append("Recent turns:\n" + turns_text)

    return "\n\n".join(parts) if parts else "No prior conversation."


def detect_fraud(current_turn):
    history_text = build_context()
    prompt = fraud_prompt(history_text, current_turn)

    try:
        resp = ollama.generate(
            model=LLM_MODEL,
            prompt=prompt,
            options={"temperature": 0.0, "max_tokens": 300}
        )
        raw = resp.get("response", "")
        return json.loads(raw)
    except Exception as e:
        return {
            "risk_level": "low",
            "confidence": 0,
            "patterns": [],
            "reason": str(e),
            "advice": "Continue listening"
        }


# ===== AUDIO CALLBACK =====
def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy().flatten())


print("\n🎤 Real-time Fraud Detector READY")
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype='float32',
    blocksize=BLOCK_SIZE,
    callback=callback
)
stream.start()

buffer = np.array([], dtype=np.float32)
speech_active = False

try:
    while True:
        chunk = audio_queue.get()
        buffer = np.concatenate((buffer, chunk))

        speech_dict = vad(chunk)
        if speech_dict is not None:
            if speech_dict.get("start"):
                speech_active = True
            elif speech_dict.get("end") and speech_active:
                speech_active = False

                if len(buffer) > SAMPLE_RATE * 0.5:
                    segments, _ = whisper.transcribe(buffer, language="en", vad_filter=True)
                    text = " ".join(s.text.strip() for s in segments).strip()

                    if text:
                        print(f"\n🗣️ {text}")

                        # Add to memory
                        recent_turns.append(text)

                        # Summarize if threshold reached
                        if len(recent_turns) >= SUMMARY_THRESHOLD:
                            summary = summarize_block(list(recent_turns))
                            summaries.append(summary)
                            recent_turns.clear()
                            print("📝 Summary created")

                        result = detect_fraud(text)

                        print(f"🚨 {result['risk_level'].upper()} ({result['confidence']}%)")
                        print("Reason:", result["reason"])
                        print("Advice:", result["advice"])

                        if result["risk_level"] in ["high", "critical"]:
                            print("⚠️ HANG UP IMMEDIATELY!")

                buffer = np.array([], dtype=np.float32)

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    stream.stop()
    stream.close()
    vad.reset_states()
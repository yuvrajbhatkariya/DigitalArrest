"""
Real-Time Fraud Detector  ─  Speed-Optimised
=============================================
Pipeline (3 parallel threads, zero blocking on main):

  [Thread 1] Audio capture + VAD   →  transcription_queue
  [Thread 2] Whisper transcription  →  llm_queue
  [Thread 3] LLM fraud detection   →  prints result
  [Thread 4] Async summarisation   (daemon, fires only when needed)

Speed choices:
  • Whisper  : "small" by default (~3x faster than medium, still accurate for Hinglish)
               On Apple Silicon → pip install mlx-whisper and set USE_MLX_WHISPER=True (~10x)
  • LLM      : llama3.2:3b-instruct  (fastest good-JSON model in ollama)
               Swap to mistral:7b-instruct for higher accuracy
  • num_predict capped at 250 (enough for JSON, cuts generation time)
  • LLM pre-warmed at startup — eliminates cold-start on first turn
  • Summarisation is a daemon thread — never blocks detection
"""

import json
import queue
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import ollama
import sounddevice as sd
from silero_vad import VADIterator, load_silero_vad

from Prompts.summary_prompt import (
    SYSTEM_PROMPT,
    build_detection_prompt,
    build_summary_prompt,
)

# ──────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────
SAMPLE_RATE       = 16000
BLOCK_SIZE        = 512
MIN_AUDIO_SECONDS = 0.4

# Whisper — "small" is 3x faster than "medium" with minimal accuracy loss
# Set USE_MLX_WHISPER = True on Apple Silicon (M1/M2/M3) for ~10x speedup
WHISPER_SIZE      = "medium"
USE_MLX_WHISPER   = False     # pip install mlx-whisper → then flip to True

# LLM — swap to "mistral:7b-instruct" for better accuracy at cost of speed
LLM_MODEL         = "mistral:7b-instruct"
LLM_MAX_TOKENS    = 250

# History
SUMMARY_THRESHOLD  = 6
RECENT_TURNS_LIMIT = 4

# ──────────────────────────────────────────────────────────
#  INTER-THREAD QUEUES  +  SHUTDOWN EVENT
# ──────────────────────────────────────────────────────────
audio_q         : queue.Queue = queue.Queue()
transcription_q : queue.Queue = queue.Queue()
llm_q           : queue.Queue = queue.Queue()
_shutdown                     = threading.Event()


# ──────────────────────────────────────────────────────────
#  WHISPER LOADER
# ──────────────────────────────────────────────────────────
def load_whisper():
    if USE_MLX_WHISPER:
        try:
            import mlx_whisper
            print(f"  ✅ MLX Whisper ({WHISPER_SIZE}) — Apple Silicon fast path")
            return mlx_whisper, "mlx"
        except ImportError:
            print("  ⚠️  mlx-whisper not installed, falling back to faster-whisper")
    from faster_whisper import WhisperModel
    model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
    print(f"  ✅ faster-whisper ({WHISPER_SIZE}  int8/CPU)")
    return model, "faster"


def transcribe(obj, kind: str, audio: np.ndarray) -> str:
    if kind == "mlx":
        result = obj.transcribe(
            audio,
            path_or_hf_repo=f"mlx-community/whisper-{WHISPER_SIZE}-mlx",
        )
        return result.get("text", "").strip()
    segments, _ = obj.transcribe(audio, language="en", vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


# ──────────────────────────────────────────────────────────
#  CONVERSATION MEMORY  (thread-safe)
# ──────────────────────────────────────────────────────────
@dataclass
class ConversationMemory:
    summary     : str   = ""
    recent_turns: deque = field(default_factory=lambda: deque(maxlen=RECENT_TURNS_LIMIT))
    _raw_buffer : list  = field(default_factory=list)
    _lock       : threading.Lock = field(default_factory=threading.Lock)

    def add_turn(self, text: str) -> bool:
        """Returns True when summarisation should be triggered."""
        with self._lock:
            self._raw_buffer.append(text)
            self.recent_turns.append(text)
            return len(self._raw_buffer) >= SUMMARY_THRESHOLD

    def snapshot(self) -> tuple:
        """Thread-safe (summary, recent_turns_minus_last) for the prompt."""
        with self._lock:
            return self.summary, list(self.recent_turns)[:-1]

    def compress_async(self) -> None:
        """Called in a daemon thread — summarises, never blocks detection."""
        with self._lock:
            to_summarize = self._raw_buffer[:SUMMARY_THRESHOLD]
            self._raw_buffer = self._raw_buffer[SUMMARY_THRESHOLD:]

        prompt = build_summary_prompt(to_summarize)
        try:
            resp = ollama.generate(
                model=LLM_MODEL,
                prompt=prompt,
                options={"temperature": 0.0, "num_predict": 180},
            )
            new_s = resp["response"].strip()
        except Exception as e:
            new_s = f"[summary error: {e}]"

        with self._lock:
            self.summary = (
                f"{self.summary}\n[Update]: {new_s}" if self.summary else new_s
            )
        print(f"\n  📋 Context compressed ({SUMMARY_THRESHOLD} turns → summary)\n")


# ──────────────────────────────────────────────────────────
#  LLM
# ──────────────────────────────────────────────────────────
def warmup_llm() -> None:
    try:
        ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            options={"num_predict": 1},
        )
        print(f"  ✅ LLM pre-warmed ({LLM_MODEL})")
    except Exception as e:
        print(f"  ⚠️  LLM warmup failed: {e}")


def detect_fraud(summary: str, recent: list, current_turn: str) -> dict:
    user_prompt = build_detection_prompt(summary, recent, current_turn)
    raw = ""
    try:
        resp = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            options={"temperature": 0.0, "num_predict": LLM_MAX_TOKENS},
            format="json",
        )
        raw = resp["message"]["content"].strip()
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            cleaned = raw.split("```json")[-1].split("```")[0].strip()
            return json.loads(cleaned)
        except Exception:
            pass
        return _fallback(f"JSON parse error: {raw[:60]}")
    except Exception as e:
        return _fallback(str(e))


def _fallback(reason: str) -> dict:
    return {
        "risk_level": "low", "confidence": 0,
        "patterns": [], "triggered_rules": [],
        "reason": reason, "prior_context_used": "",
        "advice": "Analysis failed — keep listening carefully.",
    }


# ──────────────────────────────────────────────────────────
#  OUTPUT
# ──────────────────────────────────────────────────────────
RISK_ICONS = {"low": "🟢", "medium": "🟡", "high": "🔴", "critical": "🚨"}
DIV = "─" * 62

def print_result(turn_text: str, result: dict, turn_num: int, elapsed: float) -> None:
    risk     = result.get("risk_level", "low").lower()
    conf     = result.get("confidence", 0)
    patterns = result.get("patterns", [])
    rules    = result.get("triggered_rules", [])
    reason   = result.get("reason", "")
    ctx      = result.get("prior_context_used", "")
    advice   = result.get("advice", "")
    icon     = RISK_ICONS.get(risk, "⚪")

    print(f"\n{DIV}")
    print(f"🎙️  Turn #{turn_num}  [{elapsed:.1f}s total]")
    print(f"    \"{turn_text}\"")
    print(DIV)
    print(f"{icon}  RISK : {risk.upper():<8}  Confidence: {conf}%")
    if patterns:
        print(f"🔍  Patterns       : {', '.join(patterns)}")
    if rules:
        print(f"⚖️   Rules triggered : {', '.join(str(r) for r in rules)}")
    print(f"💬  Reason         : {reason}")
    if ctx and ctx.strip().lower() not in ("n/a", "none", ""):
        print(f"📜  Prior context  : {ctx}")
    print(f"✅  Advice         : {advice}")

    if risk in ("high", "critical"):
        print()
        print("⚠️  ══════════════════════════════════════════════════════ ⚠️")
        print("⚠️      🚨  IMMEDIATE ACTION: HANG UP THE CALL NOW!  🚨      ⚠️")
        print("⚠️  ══════════════════════════════════════════════════════ ⚠️")
    print(DIV)


# ──────────────────────────────────────────────────────────
#  THREAD 1 — Audio capture + VAD
# ──────────────────────────────────────────────────────────
def audio_vad_thread(vad_model) -> None:
    vad = VADIterator(
        vad_model,
        threshold=0.5,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=700,   # slightly tighter → faster turn detection
    )
    buffer        = np.array([], dtype=np.float32)
    speech_active = False

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}")
        audio_q.put(indata.copy().flatten())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1,
        dtype="float32", blocksize=BLOCK_SIZE,
        callback=callback,
    )
    stream.start()
    try:
        while not _shutdown.is_set():
            try:
                chunk = audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            buffer = np.concatenate((buffer, chunk))
            ev = vad(chunk)
            if ev is None:
                continue

            if ev.get("start") is not None:
                speech_active = True
            elif ev.get("end") is not None and speech_active:
                speech_active = False
                if len(buffer) >= SAMPLE_RATE * MIN_AUDIO_SECONDS:
                    transcription_q.put(buffer.copy())
                buffer = np.array([], dtype=np.float32)
    finally:
        stream.stop()
        stream.close()
        vad.reset_states()


# ──────────────────────────────────────────────────────────
#  THREAD 2 — Whisper transcription
# ──────────────────────────────────────────────────────────
def transcription_thread(whisper_obj, whisper_type: str, memory: ConversationMemory) -> None:
    while not _shutdown.is_set():
        try:
            audio = transcription_q.get(timeout=0.2)
        except queue.Empty:
            continue

        t0   = time.time()
        text = transcribe(whisper_obj, whisper_type, audio)
        if not text:
            continue

        needs_compress = memory.add_turn(text)
        snap           = memory.snapshot()

        if needs_compress:
            threading.Thread(target=memory.compress_async, daemon=True).start()

        llm_q.put((text, snap, t0))
        print(f"\n  🎙️  Transcribed [{time.time()-t0:.1f}s]: {text}")


# ──────────────────────────────────────────────────────────
#  THREAD 3 — LLM fraud detection
# ──────────────────────────────────────────────────────────
def llm_thread() -> None:
    turn_num = 0
    while not _shutdown.is_set():
        try:
            text, (summary, recent), t0 = llm_q.get(timeout=0.2)
        except queue.Empty:
            continue

        turn_num += 1
        result    = detect_fraud(summary, recent, text)
        elapsed   = time.time() - t0
        print_result(text, result, turn_num, elapsed)


# ──────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────
def main() -> None:
    print(f"\n{'═'*62}")
    print("  🛡️  Real-Time Fraud Detector  (Speed-Optimised)")
    print(f"{'═'*62}")
    print(f"  Whisper : {WHISPER_SIZE}{'  [MLX — Apple Silicon]' if USE_MLX_WHISPER else '  [CPU int8]'}")
    print(f"  LLM     : {LLM_MODEL}")
    print(f"  History : {RECENT_TURNS_LIMIT} recent turns + auto-summary every {SUMMARY_THRESHOLD}")
    print(f"{'═'*62}\n")
    print("Loading models …")

    whisper_obj, whisper_type = load_whisper()
    vad_model                 = load_silero_vad()
    memory                    = ConversationMemory()

    print("Pre-warming LLM (eliminates cold-start on first turn) …")
    warmup_llm()

    print(f"\n  ✅ Ready.  Speak naturally.  Press Ctrl+C to stop.\n")

    threads = [
        threading.Thread(target=audio_vad_thread,
                         args=(vad_model,), daemon=True, name="AudioVAD"),
        threading.Thread(target=transcription_thread,
                         args=(whisper_obj, whisper_type, memory),
                         daemon=True, name="Whisper"),
        threading.Thread(target=llm_thread, daemon=True, name="LLM"),
    ]
    for t in threads:
        t.start()

    def _sigint(sig, frame):
        print("\n\n  Shutting down …")
        _shutdown.set()

    signal.signal(signal.SIGINT, _sigint)

    while not _shutdown.is_set():
        time.sleep(0.2)

    for t in threads:
        t.join(timeout=2)

    print("  Stopped. Goodbye.\n")


if __name__ == "__main__":
    main()
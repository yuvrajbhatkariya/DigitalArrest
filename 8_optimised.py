"""
Real-Time Fraud Detector  ─  Gemini + Sliding Window
=====================================================
Changes from Ollama version
───────────────────────────
• All LLM calls use google.generativeai (gemini-2.5-flash)
• Two separate Gemini clients: detection (JSON) + summary (plain text)
• Prompts trimmed to minimum viable tokens
• .env loaded once at startup via dotenv
"""

import json
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from silero_vad import VADIterator, load_silero_vad

import google.generativeai as genai

from Prompts.p2 import (
    SYSTEM_PROMPT,
    build_detection_prompt,
    build_summary_prompt,
)

# ──────────────────────────── CONFIG ────────────────────────────────────────

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

GEMINI_MODEL   = "models/gemini-2.5-flash"

SAMPLE_RATE    = 16_000
BLOCK_SIZE     = 512
WHISPER_SIZE   = "small"          # small = faster; medium = more accurate

WINDOW_SECONDS = 20               # audio each LLM call sees
SLIDE_SECONDS  = 15               # how often a new call fires
MIN_WORDS      = 6                # skip windows shorter than this

MAX_RECENT_TURNS  = 3             # raw turns kept in memory
COMPRESS_EVERY    = 5             # compress after N turns

DEDUP_THRESHOLD   = 0.85          # skip near-duplicate consecutive windows

DIVIDER = "─" * 62

# ──────────────────────────── GEMINI CLIENTS ────────────────────────────────

# Detection: enforces JSON output natively via response_mime_type
_detect_model = genai.GenerativeModel(
    GEMINI_MODEL,
    system_instruction=SYSTEM_PROMPT,
    generation_config=genai.GenerationConfig(
        temperature=0.0,
        max_output_tokens=300,
        response_mime_type="application/json",
    ),
)

# Summary: plain text, very short
_summary_model = genai.GenerativeModel(
    GEMINI_MODEL,
    generation_config=genai.GenerationConfig(
        temperature=0.0,
        max_output_tokens=180,
    ),
)

# ──────────────────────────── MEMORY ────────────────────────────────────────

@dataclass
class ConversationMemory:
    """
    Two-tier rolling memory
    ───────────────────────
    summary      : LLM-compressed text of older turns, risk levels baked in.
    recent_turns : Last MAX_RECENT_TURNS raw transcripts sent verbatim.
    _buffer      : Unsummarised annotated turns waiting for compression.
    """

    summary: str = ""
    recent_turns: deque = field(
        default_factory=lambda: deque(maxlen=MAX_RECENT_TURNS)
    )
    _buffer: list = field(default_factory=list)

    def add_turn(self, text: str, risk: str = "low", conf: int = 0) -> None:
        annotated = f"[{risk.upper()} {conf}%] {text}"
        self._buffer.append(annotated)
        self.recent_turns.append(text)
        if len(self._buffer) >= COMPRESS_EVERY:
            self._compress()

    def get_context(self) -> tuple[str, list[str]]:
        return self.summary, list(self.recent_turns)

    def _compress(self) -> None:
        batch        = self._buffer[:COMPRESS_EVERY]
        self._buffer = self._buffer[COMPRESS_EVERY:]
        try:
            resp      = _summary_model.generate_content(build_summary_prompt(batch))
            new_chunk = resp.text.strip()
        except Exception as exc:
            new_chunk = f"[Summary error: {exc}]"

        self.summary = (
            f"{self.summary}\n[+] {new_chunk}" if self.summary else new_chunk
        )
        print(f"\n📋 Memory compressed ({COMPRESS_EVERY} turns → summary)\n")


# ──────────────────────────── SLIDING WINDOW ────────────────────────────────

class SlidingAudioBuffer:
    """
    Accumulates raw audio; emits WINDOW_SECONDS slice every SLIDE_SECONDS.
    50 % overlap keeps cross-sentence patterns from falling between windows.
    """

    def __init__(self) -> None:
        self._win  = WINDOW_SECONDS * SAMPLE_RATE
        self._sld  = SLIDE_SECONDS  * SAMPLE_RATE
        self._buf  = np.array([], dtype=np.float32)
        self._last = 0.0

    def push(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        self._buf = np.concatenate((self._buf, chunk))
        if len(self._buf) > self._win * 2:
            self._buf = self._buf[-self._win * 2:]
        now = time.monotonic()
        if len(self._buf) >= self._win and (now - self._last) >= SLIDE_SECONDS:
            self._last = now
            return self._buf[-self._win:]
        return None


# ──────────────────────────── LLM CALLS ─────────────────────────────────────

def detect_fraud(memory: ConversationMemory, window_text: str) -> dict:
    summary, recent = memory.get_context()
    prompt          = build_detection_prompt(summary, recent, window_text)
    try:
        resp = _detect_model.generate_content(prompt)
        return json.loads(resp.text.strip())
    except json.JSONDecodeError:
        try:
            cleaned = resp.text.split("```json")[-1].split("```")[0].strip()
            return json.loads(cleaned)
        except Exception:
            pass
        return _fallback("JSON parse error")
    except Exception as exc:
        return _fallback(str(exc))


def _fallback(reason: str) -> dict:
    return {
        "risk_level": "low", "confidence": 0,
        "patterns": [], "triggered_rules": [],
        "reason": reason, "prior_context_used": "N/A",
        "advice": "Analysis failed — stay alert.",
    }


# ──────────────────────────── DEDUP ─────────────────────────────────────────

def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a.lower().split()), set(b.lower().split())
    union  = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


# ──────────────────────────── OUTPUT ────────────────────────────────────────

RISK_ICONS = {"low": "🟢", "medium": "🟡", "high": "🔴", "critical": "🚨"}

def print_result(text: str, result: dict, n: int) -> None:
    risk    = result.get("risk_level", "low").lower()
    conf    = result.get("confidence", 0)
    icon    = RISK_ICONS.get(risk, "⚪")
    preview = text if len(text) <= 110 else text[:107] + "…"

    print(f"\n{DIVIDER}")
    print(f"🎙️  Window #{n}  |  {preview}")
    print(DIVIDER)
    print(f"{icon}  RISK : {risk.upper():<8}  |  Confidence : {conf}%")

    for label, key in [("🔍 Patterns", "patterns"), ("⚖️  Rules", "triggered_rules")]:
        vals = result.get(key, [])
        if vals:
            print(f"{label:<18}: {', '.join(vals)}")

    print(f"💬 Reason   : {result.get('reason', '')}")
    ctx = result.get("prior_context_used", "")
    if ctx and ctx.strip().lower() not in ("n/a", "none", ""):
        print(f"📜 Context  : {ctx}")
    print(f"✅ Advice   : {result.get('advice', '')}")

    if risk in ("high", "critical"):
        print("\n⚠️  ══════════════════════════════════════════ ⚠️")
        print("⚠️       !! HANG UP THE CALL IMMEDIATELY !!     ⚠️")
        print("⚠️  ══════════════════════════════════════════ ⚠️")

    print(DIVIDER)


# ──────────────────────────── AUDIO + MAIN ──────────────────────────────────

audio_q: queue.Queue = queue.Queue()

def _audio_cb(indata, frames, time_info, status):
    if status:
        print(f"[audio] {status}", flush=True)
    audio_q.put(indata.copy().flatten())


def _worker(whisper, memory, counter, last_tx, window):
    segments, _ = whisper.transcribe(window, language="en", vad_filter=True)
    text = " ".join(s.text.strip() for s in segments).strip()

    if not text or len(text.split()) < MIN_WORDS:
        return

    if _similarity(text, last_tx[0]) >= DEDUP_THRESHOLD:
        print("\n[dedup] skipped (too similar to last window)")
        return
    last_tx[0] = text

    result = detect_fraud(memory, text)
    memory.add_turn(text, risk=result.get("risk_level", "low"),
                    conf=result.get("confidence", 0))
    counter[0] += 1
    print_result(text, result, counter[0])


def main() -> None:
    print("Loading models …")
    whisper   = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
    vad_model = load_silero_vad()
    vad       = VADIterator(vad_model, threshold=0.5, sampling_rate=SAMPLE_RATE,
                            min_silence_duration_ms=600)

    memory  = ConversationMemory()
    slide   = SlidingAudioBuffer()
    counter = [0]
    last_tx = [""]

    print(f"\n{'═'*62}")
    print(f"  🛡️  Fraud Detector  |  {GEMINI_MODEL}")
    print(f"  ⏱️  Window {WINDOW_SECONDS}s  •  Slide {SLIDE_SECONDS}s  •  Min words {MIN_WORDS}")
    print(f"{'═'*62}")
    print("  Speak normally.  First alert fires after ~30 s.")
    print("  Press Ctrl+C to stop.\n")

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=BLOCK_SIZE, callback=_audio_cb)
    stream.start()

    try:
        while True:
            chunk  = audio_q.get()
            window = slide.push(chunk)
            if window is None:
                continue
            threading.Thread(
                target=_worker,
                args=(whisper, memory, counter, last_tx, window),
                daemon=True,
            ).start()

    except KeyboardInterrupt:
        print("\n\nStopped. Goodbye.")
    finally:
        stream.stop()
        stream.close()
        vad.reset_states()


if __name__ == "__main__":
    main()
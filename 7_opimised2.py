import difflib
import json
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import ollama
import sounddevice as sd
from faster_whisper import WhisperModel


from Prompts.p2 import (
    SYSTEM_PROMPT,
    build_detection_prompt,
    build_summary_prompt,
)

# ─────────────────────────────── CONFIG ────────────────────────────────────

SAMPLE_RATE   = 16_000
BLOCK_SIZE    = 512
WHISPER_SIZE  = "medium"          # "small" = faster
LLM_MODEL     = "phi3:mini"

# Sliding window parameters (in seconds)
WINDOW_SECONDS  = 20          # how much audio each LLM call sees
SLIDE_SECONDS   = 15              # how often we fire a new LLM call
MIN_WORDS       = 4               # ignore windows with fewer words (noise)

# Memory parameters
MAX_RECENT_TURNS   = 3            # raw turns kept alongside compressed summary
COMPRESS_EVERY     = 5            # compress after every N turns

DEDUP_THRESHOLD    = 0.85         # Jaccard similarity above this = skip LLM

DIVIDER = "─" * 62

# ─────────────────────────────── MEMORY ────────────────────────────────────

@dataclass
class ConversationMemory:
    """
    Two-tier rolling memory
    ───────────────────────
    tier-1  summary      : LLM-compressed narrative of all older turns,
                           INCLUDING the risk levels that were detected.
                           Grows slowly; capped by summarisation.
    tier-2  recent_turns : Raw deque of last MAX_RECENT_TURNS entries.
                           Always sent verbatim so the LLM has exact wording.

    Why store risk in summary?
        The LLM must know "Turn 3 was already HIGH risk" without re-reading
        the whole conversation.  Baking it into the summary achieves this
        in ~1-2 sentences instead of N full turns.
    """

    summary: str = ""
    recent_turns: deque = field(
        default_factory=lambda: deque(maxlen=MAX_RECENT_TURNS)
    )
    _buffer: list = field(default_factory=list)   # unsummarised turns

    # ── public interface ────────────────────────────────────────────────────

    def add_turn(self, text: str, risk: str = "low", confidence: int = 0) -> None:
        """
        Record a new analysed turn.
        `risk` and `confidence` are woven into the summary so future
        LLM calls always know the historical threat level.
        """
        annotated = f"[{risk.upper()} {confidence}%] {text}"
        self._buffer.append(annotated)
        self.recent_turns.append(text)          # keep raw text in recent tier

        if len(self._buffer) >= COMPRESS_EVERY:
            self._compress()

    def get_context(self) -> tuple[str, list[str]]:
        """
        Returns (summary_str, recent_raw_turns_list).
        The caller feeds these directly into the detection prompt.
        """
        return self.summary, list(self.recent_turns)

    # ── internals ──────────────────────────────────────────────────────────

    def _compress(self) -> None:
        to_compress     = self._buffer[: COMPRESS_EVERY]
        self._buffer    = self._buffer[COMPRESS_EVERY :]

        prompt = build_summary_prompt(to_compress)
        try:
            resp = ollama.generate(
                model=LLM_MODEL,
                prompt=prompt,
                options={"temperature": 0.0, "num_predict": 220},
            )
            new_chunk = resp["response"].strip()
        except Exception as exc:
            new_chunk = f"[Summary error: {exc}]"

        # Append to running summary with a visual separator
        if self.summary:
            self.summary = f"{self.summary}\n[+] {new_chunk}"
        else:
            self.summary = new_chunk

        print(
            f"\n📋  Memory compressed  "
            f"({COMPRESS_EVERY} turns → summary, "
            f"summary ≈ {len(self.summary.split())} words)\n"
        )
        print("\n🧠 ================= SUMMARY UPDATE =================")
        print(self.summary)
        print("===================================================\n")


# ───────────────────────────── SLIDING WINDOW ──────────────────────────────

class SlidingAudioBuffer:
    """
    Maintains a ring-buffer of raw float32 audio.
    Produces windows of WINDOW_SECONDS every SLIDE_SECONDS.

    ┌──────────────────────── WINDOW_SECONDS ────────────────────────┐
    │  ░░░░░░░░░░░░░░░  (old half, overlap)  │  ████████████████████  │
    └────────────────────────────────────────┴───────────────────────┘
                                              ↑ SLIDE_SECONDS of new audio
    """

    def __init__(self) -> None:
        self._window_len = WINDOW_SECONDS * SAMPLE_RATE
        self._slide_len  = SLIDE_SECONDS  * SAMPLE_RATE
        self._buf        = np.array([], dtype=np.float32)
        self._last_fire  = 0.0   # timestamp of last window dispatch

    def push(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Append `chunk` and return a window array when enough new audio
        has accumulated; otherwise return None.
        """
        self._buf = np.concatenate((self._buf, chunk))

        # Keep buffer from growing forever (2 × window is plenty)
        max_keep = self._window_len * 2
        if len(self._buf) > max_keep:
            self._buf = self._buf[-max_keep :]

        now = time.monotonic()
        if (
            len(self._buf) >= self._window_len
            and (now - self._last_fire) >= SLIDE_SECONDS
        ):
            self._last_fire = now
            return self._buf[-self._window_len :]   # newest WINDOW_SECONDS

        return None


# ─────────────────────────── LLM FRAUD DETECTION ───────────────────────────

def detect_fraud(memory: ConversationMemory, window_text: str) -> dict:
    """
    Calls Ollama with system + user prompt.  Returns parsed result dict.
    Falls back gracefully on JSON/network errors.
    """
    summary, recent = memory.get_context()
    user_prompt     = build_detection_prompt(summary, recent, window_text)

    try:
        resp = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            options={"temperature": 0.0, "num_predict": 400},
            format="json",
        )
        raw = resp["message"]["content"].strip()
        return json.loads(raw)

    except json.JSONDecodeError:
        # Attempt to salvage JSON from markdown fences
        try:
            cleaned = raw.split("```json")[-1].split("```")[0].strip()
            return json.loads(cleaned)
        except Exception:
            pass
        return _fallback("JSON parse error")
    except Exception as exc:
        return _fallback(str(exc))


def _fallback(reason: str) -> dict:
    return {
        "risk_level":        "low",
        "confidence":        0,
        "patterns":          [],
        "triggered_rules":   [],
        "reason":            reason,
        "prior_context_used": "N/A",
        "advice":            "Analysis unavailable — stay vigilant.",
    }


# ──────────────────────────── DEDUP HELPER ─────────────────────────────────

def _similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity (fast, good enough for dedup)."""
    if not a or not b:
        return 0.0
    sa, sb = set(a.lower().split()), set(b.lower().split())
    inter  = sa & sb
    union  = sa | sb
    return len(inter) / len(union) if union else 0.0


# ────────────────────────── OUTPUT FORMATTING ──────────────────────────────

RISK_ICONS = {"low": "🟢", "medium": "🟡", "high": "🔴", "critical": "🚨"}

def print_result(window_text: str, result: dict, window_num: int) -> None:
    risk     = result.get("risk_level", "low").lower()
    conf     = result.get("confidence", 0)
    patterns = result.get("patterns", [])
    rules    = result.get("triggered_rules", [])
    reason   = result.get("reason", "")
    context  = result.get("prior_context_used", "")
    advice   = result.get("advice", "")
    icon     = RISK_ICONS.get(risk, "⚪")

    print(f"\n{DIVIDER}")
    print(f"🎙️  Window #{window_num}  ({WINDOW_SECONDS}s slide)")
    # Truncate long transcripts in the console header
    preview = window_text if len(window_text) <= 120 else window_text[:117] + "…"
    print(f"   {preview}")
    print(DIVIDER)
    print(f"{icon}  RISK : {risk.upper():<8}  |  Confidence : {conf}%")

    if patterns:
        print(f"🔍  Patterns      : {', '.join(patterns)}")
    if rules:
        print(f"⚖️   Rules Fired   : {', '.join(rules)}")

    print(f"💬  Reason        : {reason}")
    if context and context.strip().lower() not in ("n/a", "none", ""):
        print(f"📜  Prior context : {context}")
    print(f"✅  Advice        : {advice}")

    if risk in ("high", "critical"):
        print()
        print("⚠️  ══════════════════════════════════════════════════ ⚠️")
        print("⚠️        !! HANG UP THE CALL IMMEDIATELY !!            ⚠️")
        print("⚠️  ══════════════════════════════════════════════════ ⚠️")

    print(DIVIDER)


# ──────────────────────────────── MAIN ─────────────────────────────────────

audio_queue: queue.Queue = queue.Queue()

def _audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[audio] {status}", flush=True)
    audio_queue.put(indata.copy().flatten())


def _detection_worker(
    whisper,
    memory:          ConversationMemory,
    window_counter:  list,           # mutable int wrapper
    last_transcript: list,           # mutable str wrapper
    window:          np.ndarray,
) -> None:
    """
    Runs in a background thread.
    Transcribes `window`, skips duplicates, runs LLM, prints result.
    """
    segments, _ = whisper.transcribe(window, language="en", vad_filter=True)
    text = " ".join(s.text.strip() for s in segments).strip()

    if not text or len(text.split()) < MIN_WORDS:
        return   # too short / empty — skip silently

    # ── Dedup check ────────────────────────────────────────────────────────
    prev = last_transcript[0]
    sim  = _similarity(text, prev)
    if sim >= DEDUP_THRESHOLD:
        print(f"\n[dedup] Window skipped (similarity={sim:.2f})")
        return
    last_transcript[0] = text

    # ── LLM detection ──────────────────────────────────────────────────────
    result = detect_fraud(memory, text)

    risk = result.get("risk_level", "low")
    conf = result.get("confidence", 0)

    # Store annotated turn AFTER getting the verdict
    memory.add_turn(text, risk=risk, confidence=conf)

    window_counter[0] += 1
    print_result(text, result, window_counter[0])


def main() -> None:
    print("Loading models …")
    whisper   = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")

    memory          = ConversationMemory()
    slide_buf       = SlidingAudioBuffer()
    window_counter  = [0]          # list = mutable reference across threads
    last_transcript = [""]         # for dedup

    print(f"\n{'═'*62}")
    print(f"  🛡️  Real-Time Fraud Detector  |  Model : {LLM_MODEL}")
    print(f"  ⏱️  Window : {WINDOW_SECONDS}s   Slide : {SLIDE_SECONDS}s   "
          f"Min words : {MIN_WORDS}")
    print(f"{'═'*62}")
    print("  Speak normally.  First alert fires after ~30 s of audio.")
    print("  Press Ctrl+C to stop.\n")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=_audio_callback,
    )
    stream.start()

    try:
        while True:
            chunk  = audio_queue.get()
            window = slide_buf.push(chunk)

            if window is None:
                continue   # not enough audio yet — keep buffering

            # ── Fire detection in background (non-blocking) ────────────────
            t = threading.Thread(
                target=_detection_worker,
                args=(whisper, memory, window_counter, last_transcript, window),
                daemon=True,
            )
            t.start()

    except KeyboardInterrupt:
        print("\n\nStopped. Goodbye.")
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    main()
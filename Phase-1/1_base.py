

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
from silero_vad import VADIterator, load_silero_vad

from Prompts.new_prompt import (
    SYSTEM_PROMPT,
    build_detection_prompt,
    build_summary_prompt,
)

SAMPLE_RATE        = 16000
BLOCK_SIZE         = 512
WHISPER_SIZE       = "medium"       # "small" for speed, "medium" for accuracy

LLM_MODEL          = "phi3:mini"

SUMMARY_THRESHOLD  = 6   
RECENT_TURNS_LIMIT = 4   
MIN_AUDIO_SECONDS  = 0.5 

DIVIDER = "─" * 60

#  CONVERSATION MEMORY
@dataclass
class ConversationMemory:
    """
    Manages rolling conversation history with automatic LLM-based summarization.

    Layout:
        [compressed_summary]  +  [recent_turns (deque)]
        ↑ summarized older      ↑ last RECENT_TURNS_LIMIT raw turns
    """
    summary: str = ""
    recent_turns: deque = field(default_factory=lambda: deque(maxlen=RECENT_TURNS_LIMIT))
    _raw_buffer: list = field(default_factory=list)  

    def add_turn(self, text: str) -> None:
        """Add a new transcribed turn; trigger summarization when threshold is hit."""
        self._raw_buffer.append(text)
        self.recent_turns.append(text)

        if len(self._raw_buffer) >= SUMMARY_THRESHOLD:
            self._compress()

    def _compress(self) -> None:
        """Summarize the oldest turns and merge into the running summary."""
        to_summarize = self._raw_buffer[: SUMMARY_THRESHOLD]
        self._raw_buffer = self._raw_buffer[SUMMARY_THRESHOLD:]

        prompt = build_summary_prompt(to_summarize)
        try:
            resp = ollama.generate(
                model=LLM_MODEL,
                prompt=prompt,
                options={"temperature": 0.0, "num_predict": 200},
            )
            new_summary = resp["response"].strip()
        except Exception as e:
            new_summary = f"[Summary unavailable: {e}]"

        if self.summary:
            self.summary = f"{self.summary}\n[Update]: {new_summary}"
        else:
            self.summary = new_summary

        print(f"\n📋 History compressed into summary ({SUMMARY_THRESHOLD} turns → summary)")

    def get_context(self) -> tuple[str, list[str]]:
        """Returns (summary, list_of_recent_turns) for prompt construction."""
        return self.summary, list(self.recent_turns)[:-1]  # exclude current (last) turn

    def turn_count(self) -> int:
        return len(self._raw_buffer) + (
            len(self.summary) > 0
        ) * SUMMARY_THRESHOLD  

#  LLM FRAUD DETECTION

def detect_fraud(memory: ConversationMemory, current_turn: str) -> dict:
    """
    Sends context + current turn to the LLM and parses the JSON verdict.
    Uses ollama.chat() with a system prompt for better instruction adherence.
    """
    summary, recent = memory.get_context()
    user_prompt = build_detection_prompt(summary, recent, current_turn)

    try:
        resp = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            options={
                "temperature": 0.0,   
                "num_predict": 400,   # enough for JSON + reason fields
            },
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
        return _fallback_result(f"JSON parse error on: {raw[:80]}")
    except Exception as e:
        return _fallback_result(str(e))


def _fallback_result(reason: str) -> dict:
    return {
        "risk_level": "low",
        "confidence": 0,
        "patterns": [],
        "triggered_rules": [],
        "reason": reason,
        "prior_context_used": "N/A",
        "advice": "Could not analyse — continue listening carefully.",
    }



#  OUTPUT FORMATTING
RISK_ICONS = {
    "low":      "🟢",
    "medium":   "🟡",
    "high":     "🔴",
    "critical": "🚨",
}

def print_result(turn_text: str, result: dict, turn_number: int) -> None:
    risk      = result.get("risk_level", "low").lower()
    conf      = result.get("confidence", 0)
    patterns  = result.get("patterns", [])
    rules     = result.get("triggered_rules", [])
    reason    = result.get("reason", "")
    context   = result.get("prior_context_used", "")
    advice    = result.get("advice", "")
    icon      = RISK_ICONS.get(risk, "⚪")

    print(f"\n{DIVIDER}")
    print(f"🎙️  Turn #{turn_number}: {turn_text}")
    print(DIVIDER)
    print(f"{icon}  RISK LEVEL : {risk.upper()}   |   Confidence: {conf}%")

    if patterns:
        print(f"🔍  Patterns  : {', '.join(patterns)}")
    if rules:
        print(f"⚖️   Rules Hit : {', '.join(rules)}")

    print(f"💬  Reason    : {reason}")

    if context and context.strip().lower() not in ("n/a", "none", ""):
        print(f"📜  Prior ctx : {context}")

    print(f"✅  Advice    : {advice}")

    if risk in ("high", "critical"):
        print()
        print("⚠️  ══════════════════════════════════════════════ ⚠️")
        print("⚠️      IMMEDIATE ACTION: HANG UP THE CALL NOW!      ⚠️")
        print("⚠️  ══════════════════════════════════════════════ ⚠️")

    print(DIVIDER)


#  AUDIO PIPELINE
audio_queue: queue.Queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[audio] {status}")
    audio_queue.put(indata.copy().flatten())


def main() -> None:
    print("Loading models …")
    whisper   = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
    vad_model = load_silero_vad()
    vad       = VADIterator(
        vad_model,
        threshold=0.5,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=800,
    )

    memory     = ConversationMemory()
    turn_count = 0

    print(f"\n{'═'*60}")
    print("  🛡️  Real-Time Fraud Detector  |  Model: {LLM_MODEL}")
    print(f"{'═'*60}")
    print("  Speak normally — VAD detects natural turn boundaries.")
    print("  Press Ctrl+C to stop.\n")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=audio_callback,
    )
    stream.start()

    buffer       = np.array([], dtype=np.float32)
    speech_active = False

    try:
        while True:
            chunk  = audio_queue.get()
            buffer = np.concatenate((buffer, chunk))

            speech_dict = vad(chunk)
            if speech_dict is None:
                continue

            if speech_dict.get("start") is not None:
                speech_active = True

            elif speech_dict.get("end") is not None and speech_active:
                speech_active = False

                if len(buffer) < SAMPLE_RATE * MIN_AUDIO_SECONDS:
                    buffer = np.array([], dtype=np.float32)
                    continue

                # ── Transcribe ──
                segments, _ = whisper.transcribe(
                    buffer,
                    language="en",
                    vad_filter=True,
                )
                text = " ".join(s.text.strip() for s in segments).strip()
                buffer = np.array([], dtype=np.float32)

                if not text:
                    continue

                turn_count += 1
                memory.add_turn(text)

                # ── Detect (blocking; acceptable latency for fraud alert) ──
                result = detect_fraud(memory, text)
                print_result(text, result, turn_count)

    except KeyboardInterrupt:
        print("\n\nStopped. Goodbye.")
    finally:
        stream.stop()
        stream.close()
        vad.reset_states()


if __name__ == "__main__":
    main()
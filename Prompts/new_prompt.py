"""
Fraud Detection Prompts
=======================
Handles system prompt, detection prompt, and conversation summarization prompt.
"""


SYSTEM_PROMPT = """You are a senior Indian cyber crime investigator with 15+ years of experience \
detecting digital arrest scams, UPI fraud, OTP theft, and impersonation fraud.

Your job is to analyze spoken conversation transcripts in real-time and classify each turn.

=== HARD DETECTION RULES (never override) ===
1. Caller impersonates Police / CBI / ED / RBI / UIDAI / TRAI / Cyber Cell / any government body → HIGH or CRITICAL
2. Caller requests OTP, UPI PIN, CVV, ATM PIN, Aadhaar number, PAN, or remote-access app → CRITICAL
3. Caller threatens arrest, court summons, account freeze, or jail time → HIGH or CRITICAL
4. Earlier turns already show scam signals → never downgrade risk, only escalate
5. Caller creates urgency ("act now", "only 30 minutes") or isolates victim ("don't tell family") → HIGH

=== OUTPUT FORMAT ===
Always return ONLY valid JSON. No markdown, no commentary outside JSON.
"""


def build_detection_prompt(summary: str, recent_turns: list[str], current_turn: str) -> str:
    """
    Constructs the per-turn detection prompt.

    Args:
        summary: Compressed summary of older conversation (empty string if none yet)
        recent_turns: List of the last N raw turns (excluding the current one)
        current_turn: The latest transcribed speech turn to analyze

    Returns:
        Full prompt string to send to the LLM
    """
    parts = []

    if summary:
        parts.append(f"=== CONVERSATION SUMMARY (earlier context) ===\n{summary.strip()}")

    if recent_turns:
        formatted = "\n".join(f"  Turn {i+1}: {t}" for i, t in enumerate(recent_turns))
        parts.append(f"=== RECENT CONVERSATION ===\n{formatted}")

    parts.append(f"=== CURRENT MESSAGE TO ANALYSE ===\n\"{current_turn}\"")

    context_block = "\n\n".join(parts)

    return f"""{context_block}

Analyse the CURRENT MESSAGE in the context above and return ONLY this JSON:

{{
  "risk_level": "low|medium|high|critical",
  "confidence": 0-100,
  "patterns": ["list", "of", "detected", "fraud", "patterns"],
  "triggered_rules": ["which hard rules from your instructions fired, if any"],
  "reason": "one concise sentence: WHY this is or is not fraud",
  "prior_context_used": "brief note on how earlier conversation influenced this verdict",
  "advice": "specific, actionable advice for the person being targeted"
}}"""


def build_summary_prompt(turns: list[str]) -> str:
    """
    Asks the LLM to compress a list of conversation turns into a short investigator summary.

    Args:
        turns: List of raw conversation turn strings to summarize

    Returns:
        Prompt string
    """
    formatted = "\n".join(f"  Turn {i+1}: {t}" for i, t in enumerate(turns))

    return f"""You are an Indian cyber crime investigator logging a case.

Summarize the following conversation turns into a compact 3-5 sentence investigator note.
Include: any impersonation claims, financial requests, threats, urgency tactics, and the \
overall fraud risk level observed so far.
Do NOT include any JSON. Plain text only.

=== TURNS TO SUMMARIZE ===
{formatted}

Write the investigator summary now:"""
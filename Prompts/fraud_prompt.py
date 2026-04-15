"""
Fraud Detection Prompts  –  v2
==============================
Changes from v1
───────────────
• Detection prompt is more concise (fewer tokens → faster inference).
• Summary prompt now explicitly asks the LLM to preserve risk levels,
  so compressed history always carries threat-level context.
• build_detection_prompt receives the full annotated history (risk baked in).
"""


SYSTEM_PROMPT = """You are a senior Indian cyber crime investigator (15+ years) \
specialising in digital arrest scams, UPI/OTP fraud, and impersonation fraud.

Analyse spoken conversation transcripts in real-time.

=== HARD RULES (never override) ===
R1  Impersonates Police/CBI/ED/RBI/UIDAI/TRAI/Cyber Cell/any govt body → HIGH or CRITICAL
R2  Requests OTP, UPI PIN, CVV, ATM PIN, Aadhaar, PAN, remote-access app → CRITICAL
R3  Threatens arrest, summons, account freeze, or jail → HIGH or CRITICAL
R4  Prior turns already flagged HIGH/CRITICAL → never downgrade risk
R5  Creates urgency ("act now", "30 minutes") or asks victim to hide call → HIGH

=== OUTPUT ===
Return ONLY valid JSON — no markdown, no preamble.
"""


def build_detection_prompt(
    summary: str,
    recent_turns: list[str],
    current_window: str,
) -> str:
    """
    Build the per-window detection prompt.

    Parameters
    ----------
    summary        : Compressed history INCLUDING risk verdicts (may be empty).
    recent_turns   : Last MAX_RECENT_TURNS raw transcripts (no risk annotation).
    current_window : The 30-second transcript to analyse now.
    """
    parts: list[str] = []

    # ── Compressed history (short, includes risk context) ──────────────────
    if summary:
        parts.append(
            "=== CASE HISTORY (compressed, includes prior risk levels) ===\n"
            + summary.strip()
        )

    # ── Recent verbatim turns (exact wording matters for fraud signals) ─────
    if recent_turns:
        lines = "\n".join(f"  [{i+1}] {t}" for i, t in enumerate(recent_turns))
        parts.append(f"=== RECENT TURNS (verbatim) ===\n{lines}")

    # ── Current window ──────────────────────────────────────────────────────
    parts.append(
        f"=== CURRENT 30-SECOND WINDOW ===\n{current_window.strip()}"
    )

    context_block = "\n\n".join(parts)

    return (
        f"{context_block}\n\n"
        "Analyse the CURRENT WINDOW using the context above.\n"
        "Return ONLY this JSON:\n\n"
        "{\n"
        '  "risk_level": "low|medium|high|critical",\n'
        '  "confidence": 0-100,\n'
        '  "patterns": ["detected fraud patterns, empty list if none"],\n'
        '  "triggered_rules": ["R1"..."R5" labels that fired, empty if none"],\n'
        '  "reason": "one sentence: why fraud or not",\n'
        '  "prior_context_used": "how earlier turns influenced verdict, or N/A",\n'
        '  "advice": "specific actionable advice for the person being targeted"\n'
        "}"
    )


def build_summary_prompt(annotated_turns: list[str]) -> str:
    """
    Compress annotated turns (format: "[RISK CONF%] text") into a short
    investigator note.  The risk labels MUST survive into the summary so
    future detection calls always know the historical threat level.

    Parameters
    ----------
    annotated_turns : Turns already labelled with [RISK CONF%] prefix.
    """
    formatted = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(annotated_turns))

    return (
        "You are an Indian cyber crime investigator updating a case log.\n\n"
        "Compress the following turns into a 3-5 sentence investigator note.\n"
        "YOU MUST INCLUDE:\n"
        "  • Any impersonation claims or government body mentions\n"
        "  • Financial / OTP / credential requests\n"
        "  • Threats, urgency tactics, or isolation attempts\n"
        "  • The highest risk level reached so far (e.g. 'Risk escalated to HIGH')\n\n"
        "PLAIN TEXT ONLY — no JSON, no bullet points, no markdown.\n\n"
        f"=== TURNS ===\n{formatted}\n\n"
        "Write the investigator summary now:"
    )
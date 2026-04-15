"""
Fraud Detection Prompts  –  v3  (token-optimised)
==================================================
Every prompt is written to the minimum token count that still gives
Gemini enough signal.  Verbose labels and repeated instructions removed.
"""

# ─── SYSTEM PROMPT ──────────────────────────────────────────────────────────
# Kept short: Gemini Flash is instruction-following enough without padding.

SYSTEM_PROMPT = """\
You are an Indian cyber crime investigator detecting phone fraud in real time.

HARD RULES (never override):
R1 Impersonates Police/CBI/ED/RBI/UIDAI/TRAI/Cyber Cell/any govt body → HIGH or CRITICAL
R2 Requests OTP/UPI PIN/CVV/ATM PIN/Aadhaar/PAN/remote-access app → CRITICAL
R3 Threatens arrest/summons/account freeze/jail → HIGH or CRITICAL
R4 Prior turns already HIGH/CRITICAL → never downgrade
R5 Urgency ("act now","30 min") or isolation ("don't tell family") → HIGH

Output ONLY valid JSON, no markdown.\
"""


# ─── DETECTION PROMPT ───────────────────────────────────────────────────────

def build_detection_prompt(
    summary: str,
    recent_turns: list[str],
    current_window: str,
) -> str:
    """
    Minimal prompt — every section is included only when it has content.
    Token savings vs v2: ~35 % fewer prompt tokens on average.
    """
    parts: list[str] = []

    if summary:
        parts.append(f"HISTORY:\n{summary.strip()}")

    if recent_turns:
        lines = "\n".join(f"{i+1}. {t}" for i, t in enumerate(recent_turns))
        parts.append(f"RECENT:\n{lines}")

    parts.append(f"ANALYSE:\n{current_window.strip()}")

    return "\n\n".join(parts) + """

Return JSON:
{"risk_level":"low|medium|high|critical","confidence":0-100,"patterns":[],"triggered_rules":[],"reason":"1 sentence","prior_context_used":"brief or N/A","advice":"actionable tip"}"""


# ─── SUMMARY PROMPT ─────────────────────────────────────────────────────────

def build_summary_prompt(annotated_turns: list[str]) -> str:
    """
    Compress annotated turns ([RISK CONF%] text) into a short case note.
    Must preserve the highest risk level reached — future calls rely on it.
    Token savings vs v2: ~40 % fewer prompt tokens.
    """
    lines = "\n".join(f"{i+1}. {t}" for i, t in enumerate(annotated_turns))
    return (
        "Summarise these fraud-call turns in 2-4 plain-text sentences.\n"
        "Include: impersonation claims, financial/OTP requests, threats, "
        "urgency tactics, highest risk level reached.\n"
        "No JSON, no bullets.\n\n"
        f"{lines}\n\nSummary:"
    )
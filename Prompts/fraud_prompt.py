
def fraud_prompt(history_text: str, current_turn: str) -> str:
    return """
You are an expert Indian cyber-crime investigator.

Common scam patterns:
- Bank/UIDAI/SBI impersonation + urgency ("account blocked", "arrest warrant")
- Asking for OTP, UPI PIN, Aadhaar, remote access
- Never tells you to call official number yourself

Conversation so far (most recent first):
{history}

Latest turn: "{current_turn}"

Output ONLY valid JSON:
{{
  "risk_level": "low|medium|high|critical",
  "confidence": 0-100,
  "patterns": ["urgency", "otp_request"],
  "reason": "one short sentence",
  "advice": "Do NOT share OTP. Hang up and call your bank from official number."
}}
""".format(
        history=history_text,
        current_turn=current_turn
    )
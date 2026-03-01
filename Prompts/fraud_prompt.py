# prompts.py

def fraud_prompt(history_text: str, current_turn: str) -> str:
    return f"""
You are an expert Indian cyber-crime investigator.

Common scam patterns:
- Bank/UIDAI/SBI impersonation + urgency ("account blocked", "arrest warrant")
- Asking for OTP, UPI PIN, Aadhaar, remote access
- Pressure tactics
- Refusing official callback

Conversation context:
{history_text}

Latest turn:
"{current_turn}"

Output ONLY valid JSON:
{{
  "risk_level": "low|medium|high|critical",
  "confidence": 0-100,
  "patterns": ["urgency", "otp_request"],
  "reason": "one short sentence",
  "advice": "Clear action advice"
}}
"""

def summary_prompt(turns_block: list[str]) -> str:
    joined = "\n".join(f"- {t}" for t in turns_block)

    return f"""
You are summarizing a telecom call for fraud detection.

Summarize the following conversation very briefly.
Focus only on:
- Intent
- Sensitive info requests
- Urgency signals
- Scam indicators

Keep summary extremely short (1-2 sentences max).

Conversation:
{joined}

Summary:
"""
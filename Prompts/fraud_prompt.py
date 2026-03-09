# def fraud_prompt(history_text: str, current_turn: str) -> str:
#     return f"""
# You are a senior Indian cyber crime investigator.

# Important detection rules:

# 1. If the caller impersonates police, bank, RBI, UIDAI, cyber cell or government → risk must be HIGH or CRITICAL.
# 2. If the caller asks for OTP, UPI PIN, CVV, ATM PIN, Aadhaar, or remote access → risk must be CRITICAL.
# 3. If the caller pressures the victim or threatens arrest → risk must be HIGH or CRITICAL.
# 4. If earlier conversation already shows scam signals, never downgrade risk.

# Conversation history:
# {history_text}

# Current message:
# "{current_turn}"

# Return ONLY valid JSON:

# {{
#   "risk_level": "low|medium|high|critical",
#   "confidence": 0-100,
#   "patterns": ["authority_impersonation","financial_request"],
#   "reason": "one sentence explaining why it is fraud or not",
#   "advice": "clear action for the user"
# }}
# """

def fraud_prompt(conversation_summary: str, recent_history: str, current_turn: str) -> str:
    """
    Generate a prompt for fraud detection with conversation context.
    
    Args:
        conversation_summary: Summary of older conversation turns
        recent_history: Last 6 turns of conversation
        current_turn: Current speech turn to analyze
    
    Returns:
        Formatted prompt string for the LLM
    """
    return f"""
You are a senior Indian cyber crime investigator with expertise in identifying fraudulent communication patterns.

IMPORTANT DETECTION RULES:
1. If the caller impersonates police, bank, RBI, UIDAI, cyber cell or government → risk must be HIGH or CRITICAL.
2. If the caller asks for OTP, UPI PIN, CVV, ATM PIN, Aadhaar, or remote access → risk must be CRITICAL.
3. If the caller pressures the victim or threatens arrest → risk must be HIGH or CRITICAL.
4. If earlier conversation already shows scam signals, never downgrade risk.
5. If the caller creates urgency or claims limited time offers → increase risk level.
6. If the caller requests payment through unusual methods (gift cards, crypto) → risk must be HIGH or CRITICAL.

CONVERSATION SUMMARY:
{conversation_summary}

RECENT CONVERSATION (Last 6 turns):
{recent_history}

CURRENT MESSAGE:
"{current_turn}"

Analyze the current message in the context of the entire conversation. Consider:
- How the conversation has evolved
- Any escalation in urgency or threats
- Requests for sensitive information
- Attempts to establish authority

Return ONLY valid JSON:

{{
  "risk_level": "low|medium|high|critical",
  "confidence": 0-100,
  "patterns": ["authority_impersonation","financial_request","urgency_creation","threat","information_seeking"],
  "reason": "Detailed explanation of why this is considered fraud or not, referencing specific elements from the conversation",
  "advice": "Clear, actionable advice for the user based on the current risk level",
  "escalation_detected": true/false,
  "key_red_flags": ["list of specific red flags detected in this turn"]
}}
"""

def summarize_conversation_prompt(conversation_text: str) -> str:
    """
    Generate a prompt to summarize conversation history.
    
    Args:
        conversation_text: Text to summarize
    
    Returns:
        Formatted prompt string for the LLM
    """
    return f"""
Summarize the following conversation, focusing on:
- Key topics discussed
- Any requests for personal information
- Claims of authority
- Urgency or pressure tactics
- Financial requests or discussions

Keep the summary concise but preserve important context that would help identify fraud patterns.

CONVERSATION:
{conversation_text}

Return a concise summary (max 200 words):
"""
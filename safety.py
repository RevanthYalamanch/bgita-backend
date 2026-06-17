# backend/safety.py
"""Crisis (self-harm / suicide) detection for the chat endpoint.

This is a deterministic safety net that runs BEFORE the LLM. If a message
contains high-confidence crisis language, we skip the model entirely and return
a fixed, compassionate response with help resources. Rationale:

- Deterministic: doesn't depend on the model behaving correctly under load.
- Fast & free: no extra API call on the hot path.
- Fails safe: we deliberately lean toward over-triggering. Showing crisis
  resources to someone who didn't need them is a minor annoyance; missing
  someone who did is unacceptable.

This is a backstop, not a clinical tool. The system prompt also instructs the
model to handle risk gently for borderline phrasing this screen doesn't catch.
"""
import re

# Word-boundary patterns for explicit self-harm / suicidal statements. Phrases
# are intentionally multi-word to avoid idioms ("this is killing me", "dying to
# see you", "I could kill for a coffee").
_CRISIS_PATTERNS = [
    r"kill(?:ing)?\s+myself",
    r"end(?:ing)?\s+my\s+life",
    r"take\s+my\s+own\s+life",
    r"want(?:ing)?\s+to\s+die",
    r"wanna\s+die",
    r"don'?t\s+want\s+to\s+(?:live|be\s+alive)",
    r"do\s+not\s+want\s+to\s+(?:live|be\s+alive)",
    r"no\s+(?:reason|point)\s+(?:in\s+)?(?:to\s+)?living",
    r"better\s+off\s+dead",
    r"\bsuicid(?:e|al)\b",
    r"hurt(?:ing)?\s+myself",
    r"harm(?:ing)?\s+myself",
    r"cut(?:ting)?\s+myself",
    r"\bself[\s-]?harm",
    r"\bkms\b",  # common shorthand for "kill myself"
    r"end\s+it\s+all",
]

_CRISIS_RE = re.compile("|".join(_CRISIS_PATTERNS), re.IGNORECASE)


# Plain text (no markdown) — the chat UI renders content as pre-wrapped text.
CRISIS_RESPONSE = (
    "I'm really glad you told me this, and I want you to know you're not alone. "
    "It sounds like you're in a lot of pain right now, and that matters.\n\n"
    "I'm an AI and not able to keep you safe in a crisis, so please reach out to "
    "someone who can be there with you right now:\n\n"
    "- If you are in immediate danger, call your local emergency number "
    "(911 in the US).\n"
    "- US: call or text 988 (Suicide & Crisis Lifeline), available 24/7.\n"
    "- US: text HOME to 741741 (Crisis Text Line).\n"
    "- Anywhere: find a helpline near you at https://findahelpline.com\n\n"
    "If you can, please reach out to one of these now, or to someone you trust "
    "who can stay with you. I'm here to keep talking with you too — you don't "
    "have to go through this by yourself."
)


def detect_crisis(message: str) -> bool:
    """Return True if the message contains explicit self-harm / suicidal language."""
    if not message:
        return False
    return bool(_CRISIS_RE.search(message))

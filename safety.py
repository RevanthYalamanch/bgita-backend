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

# Multi-word patterns for explicit self-harm / suicidal statements. Phrasing is
# kept specific to avoid idioms ("this is killing me", "dying to see you", "I
# could kill for a coffee"). Coverage is validated by test_crisis_detection.py —
# add a case there before/after changing anything here.
#
# Known limitations (handled by the LLM + system-prompt safety clause, not this
# deterministic screen): adversarial obfuscation (leetspeak "k1ll", letter-
# spacing "k m s", censoring "k*ll") and purely implicit distress ("I'm a
# burden", "what's the point") that names no self-harm act.
_CRISIS_PATTERNS = [
    # --- Killing / ending oneself (incl. no-space "killmyself", typo "kil myself") ---
    r"kil{1,2}(?:ing)?\s*myself",
    r"end(?:ing)?\s+my\s+life",
    r"take\s+my\s+own\s+life",
    r"\boff\s+myself\b",                      # slang: "off myself"
    r"unaliv(?:e|ing)\s+myself",              # euphemism: "unalive myself"
    r"hang(?:ing)?\s+myself\b",
    r"slit(?:ting)?\s+my\s+wrists?",
    r"throw(?:ing)?\s+myself\s+(?:off|under|in\s+front)",
    r"jump(?:ing)?\s+off\s+(?:a\s+|the\s+)?(?:bridge|building|roof|balcony|cliff|ledge|overpass)",
    r"jump(?:ing)?\s+in\s+front\s+of\s+(?:a\s+|an\s+|the\s+)?(?:train|bus|car|truck|subway)",
    r"\boverdos(?:e|ing)\b",
    r"took\s+(?:a\s+whole\s+bottle\s+of|a\s+bottle\s+of|all\s+(?:my|the)|a\s+bunch\s+of|too\s+many)\s+(?:of\s+)?pills",

    # --- Wanting to die / not wanting to live / be here ---
    r"want(?:ing|ed|s)?\s+to\s+die",          # want/wanted/wanting/wants to die
    r"wanna\s+die",
    r"(?:just\s+)?(?:please\s+)?let\s+me\s+die",
    r"(?:just|please)\s+kill\s+me\b",
    r"don'?t\s+want\s+to\s+(?:live|be\s+alive|be\s+here|exist)",
    r"do\s+not\s+want\s+to\s+(?:live|be\s+alive|be\s+here|exist)",
    r"want(?:ing)?\s+it\s+(?:all\s+)?to\s+(?:be\s+over|end)\b",
    r"(?:ready|want(?:ing)?|going|need)\s+to\s+end\s+(?:things|it)\b",
    r"end\s+it\s+all",

    # --- "no reason to live" / "not worth living" / "better off ..." ---
    r"no\s+(?:reason|point)\s+(?:in\s+)?(?:to\s+)?(?:living|live)\b",
    r"(?:not|n'?t)\s+worth\s+living",
    r"better\s+off\s+dead",
    r"better\s+off\s+without\s+me",

    # --- "wish I were dead / never born / wouldn't wake up" ---
    r"wish\s+i\s+(?:was|were)\s+dead",
    r"wish\s+i\s+(?:had\s+)?(?:never\s+(?:been\s+)?born|was\s+never\s+born)",
    r"wish\s+i\s+(?:could\s+)?(?:just\s+)?(?:not|never)\s+wake\s+up",

    # --- Suicide / self-harm terms ---
    r"\bsuicid(?:e|al)\b",
    r"hurt(?:ing)?\s+myself",
    r"harm(?:ing)?\s+myself",
    r"cut(?:ting)?\s+myself",
    r"\bself[\s-]?harm",
    r"\bkms\b",  # common shorthand for "kill myself"
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


# ===========================================================================
# Harm-to-others (homicidal / mass-violence / sexual-violence intent) screen.
#
# This is a SEPARATE, higher-PRECISION screen from the self-harm one above, and
# the trade-off is deliberately inverted. Self-harm detection over-triggers on
# purpose. Here a false positive is itself harmful — wrongly treating a merely
# frustrated user ("I'm gonna kill my boss"), someone with Harm-OCD intrusive
# thoughts ("I'm terrified I might snap and hurt someone"), or — worst of all —
# a trauma SURVIVOR disclosing abuse ("he raped me", "my dad used to hit me") as
# a violent threat would be alienating and clinically damaging. So we fire ONLY
# on explicit statements of the speaker's own INTENT / PLAN to seriously harm
# others, we EXCLUDE fear-framed intrusive thoughts and victim disclosures, and
# we leave ambiguous single-target hyperbole ("I could kill him") to the LLM +
# system-prompt safety layer.
#
# IMPORTANT: a positive result only (a) shows the de-escalation message below and
# (b) lets the app surface the session to a human in the clinician portal. It
# MUST NOT trigger any automated report to third parties or authorities — any
# duty-to-warn action is a human clinical/legal judgement, not something this
# deterministic backstop should ever automate.
# ===========================================================================

# Fear / intrusive-thought / help-seeking framing that must NOT be read as a
# threat (Harm-OCD, "what if I lose control", explicit reluctance to act).
_HARM_OTHERS_EXCLUSIONS = re.compile(
    r"(?:afraid|scared|scares?\s+me|terrified|worried|anxious|nervous|fear(?:ful)?)"
    r"[^.!?\n]{0,40}?\b(?:hurt|harm|kill|murder|rape|hit|shoot|snap|lose\s+control|do\s+something)"
    r"|what\s+if\s+i\b"
    r"|intrusive\s+thought"
    r"|don'?t\s+want\s+to\s+(?:hurt|harm|kill|act|do)"
    r"|would\s+never\s+(?:actually|really|hurt|harm|do)",
    re.IGNORECASE,
)

# Intent markers: the speaker planning/wanting to act (not fear, not past tense
# by a third party). Kept as a fragment reused across patterns below.
_INTENT = (r"(?:want(?:ing|ed|s)?\s+to|going\s+to|gonna|planning\s+(?:to|a)|"
           r"plan\s+to|about\s+to|ready\s+to|going\s+to\s+finally)")

_HARM_OTHERS_PATTERNS = [
    # --- Mass violence: attacking a place or a group with a weapon ---
    r"shoot(?:ing)?\s+up\s+(?:the\s+|my\s+|a\s+|this\s+)?"
    r"(?:school|college|campus|church|mosque|synagogue|temple|office|workplace|"
    r"mall|store|building|class(?:room)?|place|crowd)",
    r"shoot(?:ing)?\s+(?:up\s+)?everyone",
    r"kill(?:ing)?\s+everyone\s+(?:at|in|here|there)\b",
    r"bring(?:ing)?\s+(?:a\s+|my\s+)?(?:gun|knife|weapon|firearm|rifle)\s+to\s+"
    r"(?:school|work|class|campus|the\s+office|church)",
    r"open\s+fire\s+on\b",
    r"plant(?:ing)?\s+a\s+bomb",
    r"blow\s+up\s+(?:the\s+|a\s+|this\s+)?(?:school|building|office|place|church|mall)",
    r"(?:commit|carry\s+out|do)\s+(?:a\s+)?(?:school|mass|workplace|church)\s+shooting",
    _INTENT + r"\s+(?:a\s+)?(?:school|mass|workplace|church)\s+shooting",
    # --- Explicit intent to murder / kill a person or people ---
    _INTENT + r"\s+murder\b",
    r"\bmurder\s+(?:him|her|them|someone|everybody|everyone|people|my\s+\w+)",
    _INTENT + r"\s+kill\s+(?:everyone|everybody|them\s+all|all\s+of\s+them|people|someone)\b",
    # --- Explicit sexual-violence intent (the speaker's own intent only) ---
    _INTENT + r"\s+(?:rape|sexually\s+assault|molest)\b",
]

_HARM_OTHERS_RE = re.compile("|".join(_HARM_OTHERS_PATTERNS), re.IGNORECASE)


# Plain text (no markdown) — mirrors CRISIS_RESPONSE. Non-judgemental and
# de-escalating: acknowledges the feeling, does not accuse, routes to real help
# (911 for imminent danger; 988 also supports thoughts of harming others).
HARM_TO_OTHERS_RESPONSE = (
    "I hear how much pain and anger you're carrying right now, and I'm taking what "
    "you said seriously. Thoughts about hurting someone else can be frightening and "
    "overwhelming, and telling someone about them takes real courage.\n\n"
    "I'm an AI and can't keep you or anyone else safe in a moment like this, so "
    "please reach out to someone who can, right now:\n\n"
    "- If you might act on this, or if anyone is in immediate danger, call 911 "
    "(or your local emergency number) now.\n"
    "- US: call or text 988 (Suicide & Crisis Lifeline) — they support people "
    "having thoughts of harming themselves OR someone else, 24/7 and confidential.\n"
    "- Anywhere: find a helpline near you at https://findahelpline.com\n\n"
    "You don't have to act on these thoughts, and you don't have to face them "
    "alone. Talking with a trained person right now can help you stay safe and "
    "get through this. I'm still here with you too."
)


def detect_harm_to_others(message: str) -> bool:
    """Return True if the message states explicit intent to seriously harm others.

    Deliberately precise (see the section notes above): fear-framed intrusive
    thoughts and victim/survivor disclosures are excluded so they are NOT
    misread as threats.
    """
    if not message:
        return False
    if _HARM_OTHERS_EXCLUSIONS.search(message):
        return False
    return bool(_HARM_OTHERS_RE.search(message))

"""
crisis_detector.py

Multi-tier, rule-based crisis detection and redirection matrix for MindCare.

This module is intentionally deterministic and rule-based (NOT model-based).
Crisis detection must be transparent, auditable, and fail toward caution --
a black-box classifier is not an acceptable gate for this decision. This
module is the FIRST thing every incoming message passes through, before any
intent classification or generative response is produced.

Design principles:
1. Fail toward caution: ambiguous but concerning language is treated as at
   least a "watch" tier rather than being silently ignored.
2. Never diagnose. This module only detects risk language patterns and
   routes to appropriate resources -- it does not attempt to assess
   clinical severity, intent credibility, or make any medical judgement.
3. Always resurface real crisis resources. The generative model is never
   used to write the crisis response; the response is a fixed, reviewed
   template so that its content can be trusted and audited.

Pattern provenance: the WATCH/ELEVATED/IMMINENT lists below were expanded
using real self-harm/suicide phrasings observed in a public mental-health
conversational dataset (Kaggle "Mental Health Conversational Data"), in
addition to hand-written coverage. Mining real phrasing this way is used
ONLY to widen this detector's recall -- none of that dataset's crisis-tagged
text is ever used to train the generator or produce a free-form reply.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List


class CrisisTier(Enum):
    NONE = "none"
    WATCH = "watch"          # Concerning language, not explicit intent/plan
    ELEVATED = "elevated"    # Explicit ideation without an active plan
    IMMINENT = "imminent"    # Explicit intent, plan, means, or timeframe


@dataclass
class CrisisAssessment:
    tier: CrisisTier
    matched_terms: List[str] = field(default_factory=list)
    matched_tier_name: str = "none"

    @property
    def is_crisis(self) -> bool:
        return self.tier != CrisisTier.NONE


# ---------------------------------------------------------------------------
# Tiered pattern matrix
#
# Patterns are simple regexes matched against a normalized (lowercased,
# punctuation-stripped) version of the message. They are deliberately broad
# rather than clever, because recall matters far more than precision here:
# a false positive costs the user a gentle check-in message; a false
# negative could cost far more.
# ---------------------------------------------------------------------------

IMMINENT_PATTERNS = [
    r"\bi(?:'?m| am) going to (?:kill myself|end my life|end it|commit suicide)\b",
    r"\bi have a plan to (?:kill myself|die|end my life)\b",
    r"\bi(?:'?ve| have) (?:got|bought|taken) (?:the pills|a gun|a rope)\b",
    r"\btonight i(?:'?m| am) going to (?:die|kill myself|end it)\b",
    r"\bi(?:'?m| am) about to (?:kill myself|jump|overdose)\b",
    r"\bgoodbye forever\b",
    r"\bthis is my last (?:message|text|goodbye)\b",
]

ELEVATED_PATTERNS = [
    r"\bi want to (?:kill myself|die|end my life)\b",
    r"\bi(?:'?m| am) going to end it\b",
    r"\bi don'?t want to (?:be alive|live) anymore\b",
    r"\bi wish i (?:was|were) dead\b",
    r"\bi want to hurt myself\b",
    r"\bi want to self ?harm\b",
    r"\bi'?m thinking about suicide\b",
    r"\bsuicidal thoughts\b",
    r"\bi keep thinking about ending my life\b",
    r"\blife (?:isn'?t|is not) worth living\b",
    r"\bi(?:'?ve| have) thought about (?:killing myself|ending my life|suicide)\b",
    r"\bthought about killing myself\b",
]

WATCH_PATTERNS = [
    r"\bi can'?t go on\b",
    r"\bi can'?t do this anymore\b",
    r"\bwhat'?s the point of (?:living|anything)\b",
    r"\bno reason to live\b",
    r"\beveryone would be better off without me\b",
    r"\bi(?:'?m| am) a burden\b",
    r"\bi feel like giving up\b",
    r"\bi want to disappear\b",
    r"\bi don'?t see a future for myself\b",
    r"\bi am good for nothing\b",
    r"\bi'?m good for nothing\b",
]

_COMPILED_MATRIX = {
    CrisisTier.IMMINENT: [re.compile(p) for p in IMMINENT_PATTERNS],
    CrisisTier.ELEVATED: [re.compile(p) for p in ELEVATED_PATTERNS],
    CrisisTier.WATCH: [re.compile(p) for p in WATCH_PATTERNS],
}

# Evaluated in order of severity -- the first (highest severity) match wins.
_TIER_ORDER = [CrisisTier.IMMINENT, CrisisTier.ELEVATED, CrisisTier.WATCH]


def _normalize(text: str) -> str:
    text = text.lower().strip()
    # Collapse repeated whitespace, strip most punctuation except apostrophes
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def assess_message(message: str) -> CrisisAssessment:
    """Run a message through the tiered crisis pattern matrix.

    Returns a CrisisAssessment describing the highest-severity tier matched,
    plus which literal pattern(s) triggered it (useful for audit logging).
    """
    normalized = _normalize(message)

    for tier in _TIER_ORDER:
        matches = [p.pattern for p in _COMPILED_MATRIX[tier] if p.search(normalized)]
        if matches:
            return CrisisAssessment(tier=tier, matched_terms=matches, matched_tier_name=tier.value)

    return CrisisAssessment(tier=CrisisTier.NONE)


# ---------------------------------------------------------------------------
# Fixed, reviewed response templates.
#
# These are NEVER generated by the LLM. They are static so that their
# clinical/legal accuracy can be reviewed and trusted.
# ---------------------------------------------------------------------------

CRISIS_RESOURCES_GLOBAL = (
    "If you are in immediate danger, please contact your local emergency "
    "number right away (for example 911 in the US, 999 in the UK, 999/112 "
    "in Malaysia)."
)

CRISIS_RESOURCES_BY_REGION = {
    "us": "988 Suicide & Crisis Lifeline (call or text 988) -- available 24/7 in the US.",
    "uk": "Samaritans -- call 116 123, free, 24/7 in the UK and Ireland.",
    "my": "Befrienders Malaysia -- call 03-7627 2929, or Talian Kasih 15999.",
    "global": "You can find a helpline in your country at befrienders.org or findahelpline.com.",
}

RESPONSES_BY_TIER = {
    CrisisTier.IMMINENT: (
        "I'm really worried about you right now, and I want to make sure you're safe. "
        "What you're describing sounds like an emergency. Please reach out right now to "
        "emergency services or a crisis line -- you don't have to be in this alone.\n\n"
        f"{CRISIS_RESOURCES_GLOBAL}\n\n"
        f"- {CRISIS_RESOURCES_BY_REGION['us']}\n"
        f"- {CRISIS_RESOURCES_BY_REGION['uk']}\n"
        f"- {CRISIS_RESOURCES_BY_REGION['my']}\n"
        f"- {CRISIS_RESOURCES_BY_REGION['global']}\n\n"
        "I'm an AI and I'm not able to provide the emergency support you need right now, "
        "but a real person on one of these lines can help immediately. Is there someone "
        "nearby -- a friend, family member, or neighbor -- who could stay with you right now?"
    ),
    CrisisTier.ELEVATED: (
        "Thank you for telling me that -- it takes courage to say it out loud, and I don't "
        "want you to go through this alone. I'm an AI, so I'm not able to provide the kind "
        "of support a trained crisis counselor can, but they are ready to listen right now.\n\n"
        f"- {CRISIS_RESOURCES_BY_REGION['us']}\n"
        f"- {CRISIS_RESOURCES_BY_REGION['uk']}\n"
        f"- {CRISIS_RESOURCES_BY_REGION['my']}\n"
        f"- {CRISIS_RESOURCES_BY_REGION['global']}\n\n"
        "Would you be willing to reach out to one of these right now, or to someone you trust?"
    ),
    CrisisTier.WATCH: (
        "It sounds like you're carrying something really heavy right now, and I want you to "
        "know that what you're feeling matters. If these feelings ever become thoughts of "
        "harming yourself, please know support is available immediately:\n\n"
        f"- {CRISIS_RESOURCES_BY_REGION['us']}\n"
        f"- {CRISIS_RESOURCES_BY_REGION['uk']}\n"
        f"- {CRISIS_RESOURCES_BY_REGION['my']}\n\n"
        "I'm here to keep listening -- can you tell me more about what's been going on?"
    ),
}

# Used by chatbot.py as a defense-in-depth fallback when the DistilBERT
# classifier alone predicts the "crisis" intent label but this rule-based
# detector did NOT independently flag the message. Deliberately gentler than
# the WATCH tier above (since the rule-based gate -- the more trustworthy
# signal -- did not fire), but still offers resources rather than free
# generation.
CLASSIFIER_ONLY_CRISIS_RESPONSE = (
    "I want to check in with you -- what you shared sounds like it might be really heavy. "
    "I'm not able to tell how serious things are from text alone, so I'd rather be careful. "
    "If you're ever having thoughts of harming yourself, support is available immediately:\n\n"
    f"- {CRISIS_RESOURCES_BY_REGION['us']}\n"
    f"- {CRISIS_RESOURCES_BY_REGION['uk']}\n"
    f"- {CRISIS_RESOURCES_BY_REGION['my']}\n\n"
    "Otherwise, I'm here and glad to keep listening -- can you tell me more about what's going on?"
)


def build_crisis_response(assessment: CrisisAssessment) -> str:
    """Return the fixed response template for the given assessment's tier."""
    if not assessment.is_crisis:
        raise ValueError("build_crisis_response called on a non-crisis assessment")
    return RESPONSES_BY_TIER[assessment.tier]


if __name__ == "__main__":
    # Lightweight self-test / demo when run directly.
    samples = [
        "hi, how are you?",
        "I feel like giving up lately",
        "I want to die, I don't want to live anymore",
        "I have a plan to end my life tonight",
        "I've thought about killing myself before",
        "my time has come, I am good for nothing",
    ]
    for s in samples:
        a = assess_message(s)
        print(f"[{a.tier.value:9s}] {s!r} -> matched: {a.matched_terms}")

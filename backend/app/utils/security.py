"""Context-aware safety and content moderation guardrails for ResearchPilot AI."""
import re
import logging
from typing import Tuple

logger = logging.getLogger("researchpilot.security")

# Explicit, inappropriate, or malicious query patterns
EXPLICIT_SAFETY_PATTERNS = [
    r"\b(explicit|adult\s*jokes?|erotic|porn|pornography|nude|nudity|nsfw|vulgar|profanity)\b",
    r"\b(sex|sexual|genital|genitals|penis|vagina)\b",
    r"\b(hate\s*speech|racist|slur|violence|self-harm|suicide|exploit|malware|hack)\b",
]

COMPILED_SAFETY = [re.compile(p, re.IGNORECASE) for p in EXPLICIT_SAFETY_PATTERNS]


def moderate_query(query: str) -> Tuple[bool, str]:
    """Evaluates a user query for safety violations, explicit content, and off-topic requests.
    Returns (is_safe, refusal_reason).
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return False, "Query cannot be empty."

    # Check explicit content and safety patterns
    for pattern in COMPILED_SAFETY:
        if pattern.search(cleaned_query):
            logger.warning(f"Security Guardrail Triggered: Explicit/Inappropriate query detected ('{cleaned_query[:30]}...')")
            return False, "🔒 Security Guardrail: Your query contains inappropriate, explicit, or non-academic content. ResearchPilot AI only processes academic, scientific, and educational research queries."

    return True, ""

"""Security and content moderation guardrails for ResearchPilot AI."""
import re
import logging
from typing import Tuple

logger = logging.getLogger("researchpilot.security")

# Prohibited topics and explicit query patterns
EXPLICIT_SAFETY_PATTERNS = [
    r"\b(genital|genitals|penis|vagina|sex|sexual|porn|nude|nudity|explicit|erotic)\b",
    r"\b(hate\s*speech|racist|slur|violence|self-harm|suicide|exploit|malware|hack)\b",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in EXPLICIT_SAFETY_PATTERNS]


def moderate_query(query: str) -> Tuple[bool, str]:
    """Evaluates a user query for domain scope and explicit safety rules.
    Returns (is_safe, refusal_reason).
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return False, "Query cannot be empty."

    # Check explicit content patterns
    for pattern in COMPILED_PATTERNS:
        if pattern.search(cleaned_query):
            logger.warning(f"Security Guardrail Triggered: Explicit/Inappropriate Query detected ('{cleaned_query[:30]}...')")
            return False, "🔒 Security Guardrail: Your query contains inappropriate, non-academic, or sexually explicit concepts. ResearchPilot AI only processes academic, scientific, and educational research queries."

    return True, ""

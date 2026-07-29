"""Context-aware safety and content moderation guardrails for ResearchPilot AI."""
import re
import logging
from typing import Tuple

logger = logging.getLogger("researchpilot.security")

# Severe malicious patterns (always block regardless of paper domain)
MALICIOUS_PATTERNS = [
    r"\b(porn|pornography|erotic|nude|nudity|nsfw)\b",
    r"\b(hate\s*speech|racist|slur|violence|self-harm|suicide|exploit|malware|hack)\b",
]

COMPILED_MALICIOUS = [re.compile(p, re.IGNORECASE) for p in MALICIOUS_PATTERNS]


def moderate_query(query: str) -> Tuple[bool, str]:
    """Evaluates a user query for severe safety violations (malware, hate speech, pornography).
    Medical and biological terms are permitted as long as they are grounded in scientific literature.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return False, "Query cannot be empty."

    # Check severe malicious patterns
    for pattern in COMPILED_MALICIOUS:
        if pattern.search(cleaned_query):
            logger.warning(f"Security Guardrail Triggered: Severe policy violation in query ('{cleaned_query[:30]}...')")
            return False, "🔒 Security Guardrail: Your query contains severe policy violations (hate speech, malware, or gratuitous explicit content). ResearchPilot AI only processes academic, scientific, and educational research queries."

    return True, ""

"""Shared confidence-scoring helpers.

Confidence bands (per project spec):
  0.90 - 1.00 -> High
  0.70 - 0.89 -> Medium
  below 0.70  -> Needs Review
  missing     -> 0.0
"""

CONFIDENCE_THRESHOLD = 0.70

EXACT_MATCH_CONFIDENCE = 0.90
NEARBY_MATCH_CONFIDENCE = 0.75
WEAK_MATCH_CONFIDENCE = 0.55
MISSING_CONFIDENCE = 0.0


def confidence_label(confidence: float) -> str:
    """Return a human-readable label for a confidence score."""
    if confidence >= 0.90:
        return "High"
    if confidence >= 0.70:
        return "Medium"
    return "Needs Review"


def needs_review(confidence: float, threshold: float = CONFIDENCE_THRESHOLD) -> bool:
    """A field/record needs human review if confidence is below threshold."""
    return confidence < threshold

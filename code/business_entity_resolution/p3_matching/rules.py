# -*- coding: utf-8 -*-
"""rules.py

Shared domain business rules for P3 matching model inference and validation.
Applies safe country pruning and domain boundary constraints consistently across
train holdout evaluation and production predict.py.
"""

from __future__ import annotations
from typing import Dict, Any, Set
from business_entity_resolution.preprocessing import COUNTRY_ALIASES

CANONICAL_COUNTRIES: Set[str] = set(COUNTRY_ALIASES.values())


def apply_business_rules(
    prob: float,
    rec_s1: Dict[str, Any],
    rec_cand: Dict[str, Any]
) -> float:
    """Apply high-precision post-model domain rules.

    1. Confirmed Country Pruning:
       If both records have known canonical country codes and they differ
       (e.g., 'us' vs 'uk', 'india' vs 'france'), force prob to 0.0.
       If either country is unmapped (OCR noise, unlisted territory),
       rely on model's soft country feature rather than hard pruning.
    """
    c_s1 = str(rec_s1.get("country_clean", "")).strip().casefold()
    c_cand = str(rec_cand.get("country_clean", "")).strip().casefold()

    if c_s1 and c_cand:
        # Only hard-prune when both are verified canonical country representations
        if c_s1 in CANONICAL_COUNTRIES and c_cand in CANONICAL_COUNTRIES:
            if c_s1 != c_cand:
                return 0.0

    return prob

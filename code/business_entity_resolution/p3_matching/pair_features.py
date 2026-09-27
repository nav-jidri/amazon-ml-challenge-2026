# -*- coding: utf-8 -*-
"""pair_features.py

Pairwise feature engineering for candidate pairs using P1 normalized representations.
Extracts name, address, country, token overlap, character n-gram, and source-level features.
"""

from __future__ import annotations
import math
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


FEATURE_NAMES = [
    # --- Name Features ---
    "name_exact_clean",              # Exact equality of name_clean
    "name_exact_compact",            # Exact equality of name_compact
    "name_core_exact",               # Exact equality of name_core (legal suffix normalized)
    "name_ascii_exact",              # Exact equality of diacritic-stripped name_ascii_compact
    "name_token_jaccard",            # Jaccard similarity of name token sets
    "name_token_overlap_count",      # Count of shared name tokens
    "name_sig_token_jaccard",        # Jaccard similarity of non-stopword significant tokens
    "name_token_containment_min",    # min(|A cap B| / |A|, |A cap B| / |B|)
    "name_token_containment_max",    # max(|A cap B| / |A|, |A cap B| / |B|)
    "name_token_sort_ratio",         # Character-level similarity on name_token_key
    "name_char_2gram_jaccard",       # Character 2-gram Jaccard on name_clean
    "name_char_3gram_jaccard",       # Character 3-gram Jaccard on name_clean
    "name_len_diff",                 # Absolute difference in clean name length
    "name_len_ratio",                # min(len_a, len_b) / max(len_a, len_b)

    # --- Address Features ---
    "addr_exact_clean",              # Exact equality of address_clean
    "addr_token_jaccard",            # Jaccard similarity of address token sets
    "addr_token_overlap_count",      # Count of shared address tokens
    "addr_num_overlap_count",        # Count of shared numeric tokens (house/PIN/zip)
    "addr_comp_exact",               # Exact match on address_component_key (house# + postal)
    "addr_postcode_exact",           # Exact match on postal_code
    "addr_token_containment_min",    # min(|A cap B| / |A|, |A cap B| / |B|)
    "addr_token_containment_max",    # max(|A cap B| / |A|, |A cap B| / |B|)
    "addr_char_3gram_jaccard",       # Character 3-gram Jaccard on address_clean
    "addr_len_diff",                 # Absolute difference in clean address length
    "addr_len_ratio",                # min(len_a, len_b) / max(len_a, len_b)

    # --- Country Features ---
    "country_match",                 # Exact match on country_clean
    "country_missing_either",        # 1 if either record country is empty
    "country_strict_mismatch",       # 1 if both records have country and they differ (critical negative)

    # --- Interaction & Meta Features ---
    "source_is_s2",                  # 1 if candidate is from S2
    "source_is_s3",                  # 1 if candidate is from S3
    "name_addr_joint_jaccard",       # name_token_jaccard * addr_token_jaccard
    "name_addr_geo_mean",            # sqrt(name_token_jaccard * addr_token_jaccard)
    "name_empty_either",             # 1 if either record name is missing/empty
    "addr_empty_either",             # 1 if either record address is missing/empty

    # --- Discriminative Negative Penalties ---
    "addr_street_key_match",         # Exact match on address_street_key
    "addr_house_num_match",          # Exact match on house_number
    "addr_house_num_mismatch",       # 1 if both have house numbers and they differ (critical negative)
    "addr_postcode_mismatch",        # 1 if both have postal codes and they differ (critical negative)
    "name_first_two_tokens_match",   # Exact match on name_first_two_tokens
    "name_first_two_tokens_overlap", # Count of shared leading tokens
    "name_low_addr_high_penalty",    # 1 if address matches high but name matches low (same-building false positive)
    "name_high_addr_low_penalty",    # 1 if name matches high but address matches low (different-location chain)
]


def _safe_str(val: Any) -> str:
    """Safely convert value to non-null string."""
    if val is None or pd.isna(val):
        return ""
    return str(val).strip()


def _get_char_ngrams(s: str, n: int) -> set:
    """Return set of character n-grams."""
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    union_len = len(set_a | set_b)
    if union_len == 0:
        return 0.0
    return len(set_a & set_b) / union_len


def _containment(set_a: set, set_b: set) -> Tuple[float, float]:
    """Calculate min and max containment between two sets."""
    if not set_a or not set_b:
        return 0.0, 0.0
    intersection_len = len(set_a & set_b)
    c_a = intersection_len / len(set_a)
    c_b = intersection_len / len(set_b)
    return min(c_a, c_b), max(c_a, c_b)


def _char_similarity(s1: str, s2: str) -> float:
    """Fast character-level dice similarity."""
    if not s1 or not s2:
        return 1.0 if (not s1 and not s2) else 0.0
    if s1 == s2:
        return 1.0
    ng1 = _get_char_ngrams(s1, 2)
    ng2 = _get_char_ngrams(s2, 2)
    return _jaccard_similarity(ng1, ng2)


def compute_pair_feature_vector(
    rec_a: Dict[str, Any],
    rec_b: Dict[str, Any],
    b_entity_id: str = "",
    precomputed_a: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """Compute pairwise feature vector for two preprocessed records."""
    # Names
    name_a = precomputed_a["name_a"] if precomputed_a else _safe_str(rec_a.get("name_clean", ""))
    name_b = _safe_str(rec_b.get("name_clean", ""))
    compact_a = _safe_str(rec_a.get("name_compact", ""))
    compact_b = _safe_str(rec_b.get("name_compact", ""))
    core_a = _safe_str(rec_a.get("name_core", ""))
    core_b = _safe_str(rec_b.get("name_core", ""))
    ascii_a = _safe_str(rec_a.get("name_ascii_compact", ""))
    ascii_b = _safe_str(rec_b.get("name_ascii_compact", ""))
    sig_a = _safe_str(rec_a.get("name_significant_token_key", ""))
    sig_b = _safe_str(rec_b.get("name_significant_token_key", ""))
    tkey_a = _safe_str(rec_a.get("name_token_key", ""))
    tkey_b = _safe_str(rec_b.get("name_token_key", ""))

    tokens_a = precomputed_a["tokens_a"] if precomputed_a else set(name_a.split())
    tokens_b = set(name_b.split())
    sig_toks_a = set(sig_a.split()) if sig_a else set()
    sig_toks_b = set(sig_b.split()) if sig_b else set()

    name_exact_clean = 1.0 if (name_a and name_a == name_b) else 0.0
    name_exact_compact = 1.0 if (compact_a and compact_a == compact_b) else 0.0
    name_core_exact = 1.0 if (core_a and core_a == core_b) else 0.0
    name_ascii_exact = 1.0 if (ascii_a and ascii_a == ascii_b) else 0.0
    name_token_jaccard = _jaccard_similarity(tokens_a, tokens_b)
    name_token_overlap_count = float(len(tokens_a & tokens_b))
    name_sig_token_jaccard = _jaccard_similarity(sig_toks_a, sig_toks_b)
    name_cont_min, name_cont_max = _containment(tokens_a, tokens_b)
    name_token_sort_ratio = _char_similarity(tkey_a, tkey_b)

    name_ng2_a = precomputed_a["name_ng2_a"] if precomputed_a else _get_char_ngrams(name_a, 2)
    name_ng2_b = _get_char_ngrams(name_b, 2)
    name_char_2gram_jaccard = _jaccard_similarity(name_ng2_a, name_ng2_b)

    name_ng3_a = precomputed_a["name_ng3_a"] if precomputed_a else _get_char_ngrams(name_a, 3)
    name_ng3_b = _get_char_ngrams(name_b, 3)
    name_char_3gram_jaccard = _jaccard_similarity(name_ng3_a, name_ng3_b)

    len_na = precomputed_a["len_na"] if precomputed_a else len(name_a)
    len_nb = len(name_b)
    name_len_diff = float(abs(len_na - len_nb))
    name_len_ratio = (min(len_na, len_nb) / max(len_na, len_nb)) if max(len_na, len_nb) > 0 else 1.0

    # Address
    addr_a = precomputed_a["addr_a"] if precomputed_a else _safe_str(rec_a.get("address_clean", ""))
    addr_b = _safe_str(rec_b.get("address_clean", ""))
    addr_toks_a = precomputed_a["addr_toks_a"] if precomputed_a else set(addr_a.split())
    addr_toks_b = set(addr_b.split())
    acomp_a = _safe_str(rec_a.get("address_component_key", ""))
    acomp_b = _safe_str(rec_b.get("address_component_key", ""))
    post_a = _safe_str(rec_a.get("address_postal_code", ""))
    post_b = _safe_str(rec_b.get("address_postal_code", ""))

    addr_exact_clean = 1.0 if (addr_a and addr_a == addr_b) else 0.0
    addr_token_jaccard = _jaccard_similarity(addr_toks_a, addr_toks_b)
    addr_token_overlap_count = float(len(addr_toks_a & addr_toks_b))
    addr_comp_exact = 1.0 if (acomp_a and acomp_a == acomp_b) else 0.0
    addr_postcode_exact = 1.0 if (post_a and post_a == post_b) else 0.0

    nums_a = precomputed_a["nums_a"] if precomputed_a else {t for t in addr_toks_a if t.isdigit()}
    nums_b = {t for t in addr_toks_b if t.isdigit()}
    addr_num_overlap_count = float(len(nums_a & nums_b))

    addr_cont_min, addr_cont_max = _containment(addr_toks_a, addr_toks_b)

    addr_ng3_a = precomputed_a["addr_ng3_a"] if precomputed_a else _get_char_ngrams(addr_a, 3)
    addr_ng3_b = _get_char_ngrams(addr_b, 3)
    addr_char_3gram_jaccard = _jaccard_similarity(addr_ng3_a, addr_ng3_b)

    len_aa = precomputed_a["len_aa"] if precomputed_a else len(addr_a)
    len_ab = len(addr_b)
    addr_len_diff = float(abs(len_aa - len_ab))
    addr_len_ratio = (min(len_aa, len_ab) / max(len_aa, len_ab)) if max(len_aa, len_ab) > 0 else 1.0

    # Country
    c_a = _safe_str(rec_a.get("country_clean", ""))
    c_b = _safe_str(rec_b.get("country_clean", ""))
    country_match = 1.0 if (c_a and c_b and c_a == c_b) else 0.0
    country_missing_either = 1.0 if (not c_a or not c_b) else 0.0
    country_strict_mismatch = 1.0 if (c_a and c_b and c_a != c_b) else 0.0

    # Interaction & Source meta
    b_id = b_entity_id or _safe_str(rec_b.get("entity_id", ""))
    source_is_s2 = 1.0 if b_id.startswith("S2-") else 0.0
    source_is_s3 = 1.0 if b_id.startswith("S3-") else 0.0
    name_addr_joint_jaccard = name_token_jaccard * addr_token_jaccard
    name_addr_geo_mean = math.sqrt(name_token_jaccard * addr_token_jaccard)
    name_empty_either = 1.0 if (not name_a or not name_b) else 0.0
    addr_empty_either = 1.0 if (not addr_a or not addr_b) else 0.0

    # Discriminative Street & House Number Features
    street_a = precomputed_a.get("street_a") if precomputed_a else _safe_str(rec_a.get("address_street_key", ""))
    street_b = _safe_str(rec_b.get("address_street_key", ""))
    house_a = precomputed_a.get("house_a") if precomputed_a else _safe_str(rec_a.get("address_house_number", ""))
    house_b = _safe_str(rec_b.get("address_house_number", ""))

    addr_street_key_match = 1.0 if (street_a and street_b and street_a == street_b) else 0.0
    addr_house_num_match = 1.0 if (house_a and house_b and house_a == house_b) else 0.0
    addr_house_num_mismatch = 1.0 if (house_a and house_b and house_a != house_b) else 0.0
    addr_postcode_mismatch = 1.0 if (post_a and post_b and post_a != post_b) else 0.0

    two_toks_a = precomputed_a.get("two_toks_a") if precomputed_a else _safe_str(rec_a.get("name_first_two_tokens", ""))
    two_toks_b = _safe_str(rec_b.get("name_first_two_tokens", ""))
    name_two_tokens_match = 1.0 if (two_toks_a and two_toks_b and two_toks_a == two_toks_b) else 0.0

    s_two_a = set(two_toks_a.split()) if two_toks_a else set()
    s_two_b = set(two_toks_b.split()) if two_toks_b else set()
    name_first_two_tokens_overlap = float(len(s_two_a & s_two_b))

    # Anti-FPR asymmetric penalties (gated on non-empty values so sparse data is not falsely penalized):
    # 1. High address match but near-zero name match (same-building / multi-tenant false positive)
    name_low_addr_high_penalty = 1.0 if (
        addr_token_jaccard >= 0.60 and name_token_jaccard < 0.15
        and name_core_exact == 0.0 and name_empty_either == 0.0
    ) else 0.0
    # 2. High name match but near-zero address match (chain brand in different city/branch)
    name_high_addr_low_penalty = 1.0 if (
        name_token_jaccard >= 0.70 and addr_token_jaccard < 0.15
        and addr_comp_exact == 0.0 and addr_empty_either == 0.0
    ) else 0.0

    return [
        name_exact_clean,
        name_exact_compact,
        name_core_exact,
        name_ascii_exact,
        name_token_jaccard,
        name_token_overlap_count,
        name_sig_token_jaccard,
        name_cont_min,
        name_cont_max,
        name_token_sort_ratio,
        name_char_2gram_jaccard,
        name_char_3gram_jaccard,
        name_len_diff,
        name_len_ratio,
        addr_exact_clean,
        addr_token_jaccard,
        addr_token_overlap_count,
        addr_num_overlap_count,
        addr_comp_exact,
        addr_postcode_exact,
        addr_cont_min,
        addr_cont_max,
        addr_char_3gram_jaccard,
        addr_len_diff,
        addr_len_ratio,
        country_match,
        country_missing_either,
        country_strict_mismatch,
        source_is_s2,
        source_is_s3,
        name_addr_joint_jaccard,
        name_addr_geo_mean,
        name_empty_either,
        addr_empty_either,
        addr_street_key_match,
        addr_house_num_match,
        addr_house_num_mismatch,
        addr_postcode_mismatch,
        name_two_tokens_match,
        name_first_two_tokens_overlap,
        name_low_addr_high_penalty,
        name_high_addr_low_penalty,
    ]


def build_feature_matrix(
    pair_rows: List[Tuple[str, str]],
    records_s1: Dict[str, Dict[str, Any]],
    records_s23: Dict[str, Dict[str, Any]]
) -> np.ndarray:
    """Generate feature matrix (N x M) for a batch of candidate pairs (s1_id, candidate_id).
    
    Caches S1-side token and character n-gram precomputations per batch to eliminate
    redundant work without storing heavy Python sets across millions of records.
    """
    rows = []
    dummy_rec = {
        "name_clean": "", "name_compact": "", "name_core": "", "name_token_key": "",
        "address_clean": "", "address_token_key": "", "country_clean": "",
        "address_street_key": "", "address_house_number": "", "name_first_two_tokens": "",
    }
    s1_cache: Dict[str, Dict[str, Any]] = {}

    for s1_id, cand_id in pair_rows:
        if s1_id not in s1_cache:
            rec_a = records_s1.get(s1_id, dummy_rec)
            name_a = _safe_str(rec_a.get("name_clean", ""))
            addr_a = _safe_str(rec_a.get("address_clean", ""))
            addr_toks_a = set(addr_a.split())
            s1_cache[s1_id] = {
                "name_a": name_a,
                "tokens_a": set(name_a.split()),
                "name_ng2_a": _get_char_ngrams(name_a, 2),
                "name_ng3_a": _get_char_ngrams(name_a, 3),
                "len_na": len(name_a),
                "addr_a": addr_a,
                "addr_toks_a": addr_toks_a,
                "nums_a": {t for t in addr_toks_a if t.isdigit()},
                "addr_ng3_a": _get_char_ngrams(addr_a, 3),
                "len_aa": len(addr_a),
                "street_a": _safe_str(rec_a.get("address_street_key", "")),
                "house_a": _safe_str(rec_a.get("address_house_number", "")),
                "two_toks_a": _safe_str(rec_a.get("name_first_two_tokens", "")),
            }

        rec_a = records_s1.get(s1_id, dummy_rec)
        rec_b = records_s23.get(cand_id, dummy_rec)
        feat = compute_pair_feature_vector(rec_a, rec_b, b_entity_id=cand_id, precomputed_a=s1_cache[s1_id])
        rows.append(feat)

    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


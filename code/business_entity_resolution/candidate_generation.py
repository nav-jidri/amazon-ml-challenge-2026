# -*- coding: utf-8 -*-
"""
candidate_generation.py

Implementation of the P2 candidate-generation pipeline.

The pipeline:

    Source 2 + Source 3
            ↓
      blocking indexes
            ↓
    Source 1 streaming
            ↓
    multiple blocking passes
            ↓
        union
            ↓
    candidate_pairs.tsv

Blocking passes:

1. name_token_key
2. name_compact
3. name_core
4. address_token_key
5. name_phonetic_key
6. character n-gram blocking

P2 only generates candidates. It does not perform pairwise scoring,
fuzzy matching, or final match decisions.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd

from .blocking import build_candidate_indexes
from .preprocessing import iter_preprocessed_file


def _write_header(fp) -> None:
    """Write the candidate-pairs TSV header."""

    fp.write(
        "source1_entity_id\tcandidate_entity_ids\n"
    )


def generate_candidate_pairs(
    source1_path: Path,
    source2_path: Path,
    source3_path: Path,
    output_path: Path,
    chunksize: int = 100_000,
) -> Dict[str, Any]:
    """
    Generate candidate pairs using multi-pass blocking.

    Parameters
    ----------
    source1_path : Path
        Source 1 TSV.

    source2_path : Path
        Source 2 TSV.

    source3_path : Path
        Source 3 TSV.

    output_path : Path
        Destination candidate-pairs TSV.

    chunksize : int, optional
        Chunk size used by the preprocessing iterator.

    Returns
    -------
    Dict[str, Any]
        Candidate-generation diagnostics.
    """

    # ------------------------------------------------------------------
    # 1. Build Source 2 / Source 3 indexes
    # ------------------------------------------------------------------

    indexes = build_candidate_indexes(
        source2_path,
        source3_path,
    )

    # ------------------------------------------------------------------
    # 2. Output setup
    # ------------------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_s1_rows = 0
    zero_cand_s1 = 0
    total_candidate_links = 0

    candidate_counts: List[int] = []

    # ------------------------------------------------------------------
    # Per-method diagnostics
    #
    # "candidate_counts" measures the number of candidate IDs returned
    # by that individual blocking pass.
    #
    # "method_s1_hits" measures how many S1 rows received at least one
    # candidate from that pass.
    # ------------------------------------------------------------------

    method_candidate_counts: Dict[str, int] = {
        "name_token": 0,
        "name_compact": 0,
        "name_core": 0,
        "name_ascii": 0,
        "address_street": 0,
        "address_house_postcode": 0,
        "address_component": 0,
        "significant_token": 0,
        "name_first_two_tokens": 0,
        "address_house_geo": 0,
        "address_token": 0,
        "phonetic": 0,
        "char_ngram": 0,
    }

    method_s1_hits: Dict[str, int] = {
        "name_token": 0,
        "name_compact": 0,
        "name_core": 0,
        "name_ascii": 0,
        "address_street": 0,
        "address_house_postcode": 0,
        "address_component": 0,
        "significant_token": 0,
        "name_first_two_tokens": 0,
        "address_house_geo": 0,
        "address_token": 0,
        "phonetic": 0,
        "char_ngram": 0,
    }

    # ------------------------------------------------------------------
    # Incremental contribution diagnostics
    #
    # This measures how many NEW candidate IDs each pass contributes
    # after the previous passes have already been considered.
    # ------------------------------------------------------------------

    incremental_candidate_counts: Dict[str, int] = {
        "name_token": 0,
        "name_compact": 0,
        "name_core": 0,
        "name_ascii": 0,
        "address_street": 0,
        "address_house_postcode": 0,
        "address_component": 0,
        "significant_token": 0,
        "name_first_two_tokens": 0,
        "address_house_geo": 0,
        "address_token": 0,
        "phonetic": 0,
        "char_ngram": 0,
    }

    incremental_s1_hits: Dict[str, int] = {
        "name_token": 0,
        "name_compact": 0,
        "name_core": 0,
        "name_ascii": 0,
        "address_street": 0,
        "address_house_postcode": 0,
        "address_component": 0,
        "significant_token": 0,
        "name_first_two_tokens": 0,
        "address_house_geo": 0,
        "address_token": 0,
        "phonetic": 0,
        "char_ngram": 0,
    }

    # ------------------------------------------------------------------
    # 3. Stream Source 1
    # ------------------------------------------------------------------

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as out_fp:

        _write_header(out_fp)

        for chunk in iter_preprocessed_file(
            source1_path,
            chunksize=chunksize,
        ):

            # Fast column access.
            col_eid = chunk["entity_id"]
            col_ntk = chunk["name_token_key"] if "name_token_key" in chunk else [""] * len(chunk)
            col_ncomp = chunk["name_compact"] if "name_compact" in chunk else [""] * len(chunk)
            col_ncore = chunk["name_core"] if "name_core" in chunk else [""] * len(chunk)
            col_nascii = chunk["name_ascii_compact"] if "name_ascii_compact" in chunk else [""] * len(chunk)
            col_nsig = chunk["name_significant_token_key"] if "name_significant_token_key" in chunk else [""] * len(chunk)
            col_ntwo = chunk["name_first_two_tokens"] if "name_first_two_tokens" in chunk else [""] * len(chunk)
            col_atk = chunk["address_token_key"] if "address_token_key" in chunk else [""] * len(chunk)
            col_acomp = chunk["address_component_key"] if "address_component_key" in chunk else [""] * len(chunk)
            col_astreet = chunk["address_street_key"] if "address_street_key" in chunk else [""] * len(chunk)
            col_ahouse_geo = chunk["address_house_geo_key"] if "address_house_geo_key" in chunk else [""] * len(chunk)
            col_apost = chunk["address_house_postcode_key"] if "address_house_postcode_key" in chunk else [""] * len(chunk)
            col_nphon = chunk["name_phonetic_key"] if "name_phonetic_key" in chunk else [""] * len(chunk)

            for (
                eid,
                ntk,
                ncomp,
                ncore,
                atk,
                nphon,
                nascii,
                nsig,
                ntwo,
                acomp,
                astreet,
                ahouse_geo,
                apost,
            ) in zip(
                col_eid,
                col_ntk,
                col_ncomp,
                col_ncore,
                col_atk,
                col_nphon,
                col_nascii,
                col_nsig,
                col_ntwo,
                col_acomp,
                col_astreet,
                col_ahouse_geo,
                col_apost,
            ):

                s1_id = (
                    str(eid).strip()
                    if eid is not None
                    and not pd.isna(eid)
                    else ""
                )

                if not s1_id:
                    continue

                total_s1_rows += 1

                ntk_str = str(ntk).strip() if ntk and not pd.isna(ntk) else ""
                ncomp_str = str(ncomp).strip() if ncomp and not pd.isna(ncomp) else ""
                ncore_str = str(ncore).strip() if ncore and not pd.isna(ncore) else ""
                atk_str = str(atk).strip() if atk and not pd.isna(atk) else ""
                nphon_str = str(nphon).strip() if nphon and not pd.isna(nphon) else ""
                nascii_str = str(nascii).strip() if nascii and not pd.isna(nascii) else ""
                nsig_str = str(nsig).strip() if nsig and not pd.isna(nsig) else ""
                ntwo_str = str(ntwo).strip() if ntwo and not pd.isna(ntwo) else ""
                acomp_str = str(acomp).strip() if acomp and not pd.isna(acomp) else ""
                astreet_str = str(astreet).strip() if astreet and not pd.isna(astreet) else ""
                ahouse_geo_str = str(ahouse_geo).strip() if ahouse_geo and not pd.isna(ahouse_geo) else ""
                apost_str = str(apost).strip() if apost and not pd.isna(apost) else ""

                # ------------------------------------------------------
                # Multi-Pass Candidate Retrieval
                # ------------------------------------------------------

                c_tok = indexes.retrieve_by_name_token(ntk_str) - {s1_id}
                c_comp = indexes.retrieve_by_name_compact(ncomp_str) - {s1_id}
                c_core = indexes.retrieve_by_name_core(ncore_str) - {s1_id}
                c_ascii = indexes.retrieve_by_name_ascii_compact(nascii_str) - {s1_id}
                c_astreet = indexes.retrieve_by_address_street(astreet_str) - {s1_id}
                c_apost = indexes.retrieve_by_address_house_postcode(apost_str) - {s1_id}
                c_acomp = indexes.retrieve_by_address_component(acomp_str) - {s1_id}
                c_sig = indexes.retrieve_by_significant_token(nsig_str) - {s1_id}
                c_ntwo = indexes.retrieve_by_name_first_two_tokens(ntwo_str) - {s1_id} if ntwo_str else set()
                c_ahouse_geo = indexes.retrieve_by_address_house_geo(ahouse_geo_str) - {s1_id} if ahouse_geo_str else set()
                c_addr = indexes.retrieve_by_address_token(atk_str) - {s1_id}
                c_phon = indexes.retrieve_by_phonetic(nphon_str) - {s1_id}
                c_char = indexes.retrieve_by_char_ngrams(ncomp_str) - {s1_id}

                # ------------------------------------------------------
                # Incremental Union with Volume Bounding
                # ------------------------------------------------------

                cumulative_candidates: Set[str] = set()

                ordered_methods = [
                    ("name_token", c_tok),
                    ("name_compact", c_comp),
                    ("name_core", c_core),
                    ("name_ascii", c_ascii),
                    ("address_street", c_astreet),
                    ("address_house_postcode", c_apost),
                    ("address_component", c_acomp),
                    ("significant_token", c_sig),
                    ("name_first_two_tokens", c_ntwo),
                    ("address_house_geo", c_ahouse_geo),
                    ("address_token", c_addr),
                    ("phonetic", c_phon),
                    ("char_ngram", c_char),
                ]

                for method_name, method_candidates in ordered_methods:
                    new_candidates = method_candidates - cumulative_candidates
                    if new_candidates:
                        method_candidate_counts[method_name] = method_candidate_counts.get(method_name, 0) + len(method_candidates)
                        method_s1_hits[method_name] = method_s1_hits.get(method_name, 0) + 1
                        incremental_candidate_counts[method_name] = incremental_candidate_counts.get(method_name, 0) + len(new_candidates)
                        incremental_s1_hits[method_name] = incremental_s1_hits.get(method_name, 0) + 1

                    # Bounded addition
                    for cand_id in new_candidates:
                        if len(cumulative_candidates) < 100:  # MAX_CANDIDATES_PER_S1
                            cumulative_candidates.add(cand_id)

                candidates = cumulative_candidates
                n_cand = len(candidates)
                candidate_counts.append(n_cand)
                total_candidate_links += n_cand

                if n_cand == 0:
                    zero_cand_s1 += 1

                sorted_candidates = sorted(candidates)
                candidate_string = ",".join(sorted_candidates)
                out_fp.write(f"{s1_id}\t{candidate_string}\n")

    # ------------------------------------------------------------------
    # 4. Aggregate diagnostics
    # ------------------------------------------------------------------

    average_candidates = (
        total_candidate_links / total_s1_rows
        if total_s1_rows > 0
        else 0.0
    )

    median_candidates = (
        float(statistics.median(candidate_counts))
        if candidate_counts
        else 0.0
    )

    max_candidates = (
        max(candidate_counts)
        if candidate_counts
        else 0
    )

    # ------------------------------------------------------------------
    # 5. Return diagnostics
    # ------------------------------------------------------------------

    diagnostics: Dict[str, Any] = {
        "total_s1_rows": total_s1_rows,

        "s1_rows_with_zero_candidates": zero_cand_s1,

        "total_candidate_links": total_candidate_links,

        "average_candidates_per_s1": round(
            average_candidates,
            4,
        ),

        "median_candidates_per_s1": median_candidates,

        "max_candidates_per_s1": max_candidates,

        "method_contributions": method_candidate_counts,

        "method_s1_hits": method_s1_hits,

        "incremental_method_contributions": (
            incremental_candidate_counts
        ),

        "incremental_method_s1_hits": (
            incremental_s1_hits
        ),

        "index_stats": indexes.stats(),
    }

    return diagnostics


__all__ = [
    "generate_candidate_pairs",
]
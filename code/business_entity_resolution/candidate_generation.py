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
        "address_token": 0,
        "phonetic": 0,
        "char_ngram": 0,
    }

    method_s1_hits: Dict[str, int] = {
        "name_token": 0,
        "name_compact": 0,
        "name_core": 0,
        "address_token": 0,
        "phonetic": 0,
        "char_ngram": 0,
    }

    # ------------------------------------------------------------------
    # Incremental contribution diagnostics
    #
    # This measures how many NEW candidate IDs each pass contributes
    # after the previous passes have already been considered.
    #
    # This is particularly important for evaluating the value of
    # character n-gram blocking.
    # ------------------------------------------------------------------

    incremental_candidate_counts: Dict[str, int] = {
        "name_token": 0,
        "name_compact": 0,
        "name_core": 0,
        "address_token": 0,
        "phonetic": 0,
        "char_ngram": 0,
    }

    incremental_s1_hits: Dict[str, int] = {
        "name_token": 0,
        "name_compact": 0,
        "name_core": 0,
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

            col_ntk = (
                chunk["name_token_key"]
                if "name_token_key" in chunk
                else [""] * len(chunk)
            )

            col_ncomp = (
                chunk["name_compact"]
                if "name_compact" in chunk
                else [""] * len(chunk)
            )

            col_ncore = (
                chunk["name_core"]
                if "name_core" in chunk
                else [""] * len(chunk)
            )

            col_atk = (
                chunk["address_token_key"]
                if "address_token_key" in chunk
                else [""] * len(chunk)
            )

            col_nphon = (
                chunk["name_phonetic_key"]
                if "name_phonetic_key" in chunk
                else [""] * len(chunk)
            )

            for (
                eid,
                ntk,
                ncomp,
                ncore,
                atk,
                nphon,
            ) in zip(
                col_eid,
                col_ntk,
                col_ncomp,
                col_ncore,
                col_atk,
                col_nphon,
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

                # ------------------------------------------------------
                # Normalize values coming from the preprocessing layer.
                # ------------------------------------------------------

                ntk_str = (
                    str(ntk).strip()
                    if ntk is not None
                    and not pd.isna(ntk)
                    else ""
                )

                ncomp_str = (
                    str(ncomp).strip()
                    if ncomp is not None
                    and not pd.isna(ncomp)
                    else ""
                )

                ncore_str = (
                    str(ncore).strip()
                    if ncore is not None
                    and not pd.isna(ncore)
                    else ""
                )

                atk_str = (
                    str(atk).strip()
                    if atk is not None
                    and not pd.isna(atk)
                    else ""
                )

                nphon_str = (
                    str(nphon).strip()
                    if nphon is not None
                    and not pd.isna(nphon)
                    else ""
                )

                # ------------------------------------------------------
                # Individual blocking passes
                # ------------------------------------------------------

                c_tok = indexes.retrieve_by_name_token(
                    ntk_str
                )

                c_comp = indexes.retrieve_by_name_compact(
                    ncomp_str
                )

                c_core = indexes.retrieve_by_name_core(
                    ncore_str
                )

                c_addr = indexes.retrieve_by_address_token(
                    atk_str
                )

                c_phon = indexes.retrieve_by_phonetic(
                    nphon_str
                )

                c_char = indexes.retrieve_by_char_ngrams(
                    ncomp_str
                )

                # ------------------------------------------------------
                # Remove self-match from individual diagnostics.
                # ------------------------------------------------------

                c_tok_clean = c_tok - {s1_id}
                c_comp_clean = c_comp - {s1_id}
                c_core_clean = c_core - {s1_id}
                c_addr_clean = c_addr - {s1_id}
                c_phon_clean = c_phon - {s1_id}
                c_char_clean = c_char - {s1_id}

                # ------------------------------------------------------
                # Individual method diagnostics
                # ------------------------------------------------------

                method_sets = {
                    "name_token": c_tok_clean,
                    "name_compact": c_comp_clean,
                    "name_core": c_core_clean,
                    "address_token": c_addr_clean,
                    "phonetic": c_phon_clean,
                    "char_ngram": c_char_clean,
                }

                for method_name, method_candidates in method_sets.items():

                    method_candidate_counts[
                        method_name
                    ] += len(method_candidates)

                    if method_candidates:
                        method_s1_hits[
                            method_name
                        ] += 1

                # ------------------------------------------------------
                # Incremental union diagnostics
                #
                # The order is deliberately fixed so we can measure the
                # additional value of each blocking pass.
                # ------------------------------------------------------

                cumulative_candidates: Set[str] = set()

                ordered_methods = [
                    ("name_token", c_tok_clean),
                    ("name_compact", c_comp_clean),
                    ("name_core", c_core_clean),
                    ("address_token", c_addr_clean),
                    ("phonetic", c_phon_clean),
                    ("char_ngram", c_char_clean),
                ]

                for method_name, method_candidates in ordered_methods:

                    new_candidates = (
                        method_candidates
                        - cumulative_candidates
                    )

                    if new_candidates:

                        incremental_candidate_counts[
                            method_name
                        ] += len(new_candidates)

                        incremental_s1_hits[
                            method_name
                        ] += 1

                    cumulative_candidates.update(
                        method_candidates
                    )

                # ------------------------------------------------------
                # Final candidate union
                # ------------------------------------------------------

                candidates = cumulative_candidates

                n_cand = len(candidates)

                candidate_counts.append(n_cand)

                total_candidate_links += n_cand

                if n_cand == 0:
                    zero_cand_s1 += 1

                # ------------------------------------------------------
                # Deterministic output
                # ------------------------------------------------------

                sorted_candidates = sorted(
                    candidates
                )

                candidate_string = ",".join(
                    sorted_candidates
                )

                out_fp.write(
                    f"{s1_id}\t{candidate_string}\n"
                )

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
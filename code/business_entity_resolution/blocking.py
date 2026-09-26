# -*- coding: utf-8 -*-
"""
blocking.py

Utility functions for building inverted indexes over the pre-processed
Source 2 and Source 3 records and retrieving candidate entity IDs for
Source 1 records.

P2 blocking methods:

1. name_token_key
2. name_compact
3. name_core
4. address_token_key
5. name_phonetic_key
6. character n-gram blocking

The module deliberately operates on representations produced by the P1
preprocessing layer. It does not perform raw business-name normalization,
fuzzy matching, pairwise scoring, or final match decisions.

Character n-gram blocking is implemented as a tolerant retrieval pass.
It is intended to recover candidates affected by small character-level
corruptions or spelling variations.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import pandas as pd

from .preprocessing import (
    char_ngrams,
    iter_preprocessed_file,
)


# ---------------------------------------------------------------------------
# Character n-gram configuration
# ---------------------------------------------------------------------------

CHAR_NGRAM_SIZE = 3

# Minimum number of shared character n-grams required for a candidate.
#
# This is deliberately conservative for the first experiment.
# We can tune this later based on recall and candidate-volume diagnostics.
MIN_SHARED_CHAR_NGRAMS = 2

# Do not use extremely common n-grams as retrieval keys.
#
# Example:
#     "ing", "com", etc.
#
# can occur in a very large number of businesses and can create
# pathological candidate sets.
MAX_NGRAM_POSTINGS = 5000


class Indexes:
    """
    Container for inverted indexes used by candidate generation.

    Existing blocking indexes
    --------------------------
    name_token_key
        Mapping name_token_key -> entity IDs.

    name_compact
        Mapping name_compact -> entity IDs.

    name_core
        Mapping name_core -> entity IDs.

    address_token_key
        Mapping address_token_key -> entity IDs.

    name_phonetic_key
        Mapping name_phonetic_key -> entity IDs.

    New character n-gram index
    --------------------------
    name_char_ngram
        Mapping character n-gram -> entity IDs.

    Character n-gram retrieval is performed using multiple shared grams.
    A candidate must share at least MIN_SHARED_CHAR_NGRAMS usable grams
    with the Source 1 record.

    Deduplication
    -------------
    Source 2 and Source 3 are deduplicated independently using:

        (name_compact, address_clean, country_clean)

    Rows where all three fields are empty are never deduplicated.
    """

    def __init__(self) -> None:
        # Existing indexes
        self.name_token_key: Dict[str, List[str]] = {}
        self.name_compact: Dict[str, List[str]] = {}
        self.name_core: Dict[str, List[str]] = {}
        self.address_token_key: Dict[str, List[str]] = {}
        self.name_phonetic_key: Dict[str, List[str]] = {}

        # New character n-gram index
        self.name_char_ngram: Dict[str, List[str]] = {}

        # Diagnostics
        self.dedup_stats: Dict[str, Dict[str, int]] = {}

        self.char_ngram_stats: Dict[str, int] = {
            "total_unique_ngrams_seen": 0,
            "indexed_ngrams": 0,
            "filtered_common_ngrams": 0,
        }

    # ------------------------------------------------------------------
    # Index insertion
    # ------------------------------------------------------------------

    def add_record(
        self,
        entity_id: str,
        name_token_key: str = "",
        name_compact: str = "",
        name_core: str = "",
        address_token_key: str = "",
        name_phonetic_key: str = "",
    ) -> None:
        """
        Insert an entity into the existing exact/phonetic indexes.

        Character n-gram indexing is handled separately because it requires
        a corpus-level frequency-control step.
        """

        if name_token_key:
            self.name_token_key.setdefault(
                name_token_key, []
            ).append(entity_id)

        if name_compact:
            self.name_compact.setdefault(
                name_compact, []
            ).append(entity_id)

        if name_core:
            self.name_core.setdefault(
                name_core, []
            ).append(entity_id)

        if address_token_key:
            self.address_token_key.setdefault(
                address_token_key, []
            ).append(entity_id)

        if name_phonetic_key:
            self.name_phonetic_key.setdefault(
                name_phonetic_key, []
            ).append(entity_id)

    def add(self, entity_id: str, row: Any) -> None:
        """
        Add a single pre-processed row to the indexes.

        Kept for backward compatibility.
        """

        get_val = (
            row.get
            if hasattr(row, "get")
            else lambda k, d="": getattr(row, k, d)
        )

        self.add_record(
            entity_id=entity_id,
            name_token_key=str(
                get_val("name_token_key", "") or ""
            ),
            name_compact=str(
                get_val("name_compact", "") or ""
            ),
            name_core=str(
                get_val("name_core", "") or ""
            ),
            address_token_key=str(
                get_val("address_token_key", "") or ""
            ),
            name_phonetic_key=str(
                get_val("name_phonetic_key", "") or ""
            ),
        )

    # ------------------------------------------------------------------
    # Character n-gram construction
    # ------------------------------------------------------------------

    def _collect_char_ngram_postings(
        self,
        entity_id: str,
        name_compact: str,
        ngram_postings: Dict[str, Set[str]],
    ) -> None:
        """
        Collect entity IDs for each unique character n-gram.

        We use name_compact from P1, so P2 does not duplicate
        normalization logic.
        """

        if not name_compact:
            return

        grams = set(
            char_ngrams(
                name_compact,
                n=CHAR_NGRAM_SIZE,
            )
        )

        self.char_ngram_stats["total_unique_ngrams_seen"] += len(grams)

        for gram in grams:
            ngram_postings[gram].add(entity_id)

    def _finalize_char_ngram_index(
        self,
        ngram_postings: Dict[str, Set[str]],
    ) -> None:
        """
        Convert temporary n-gram postings into the final index.

        Very common n-grams are discarded to prevent huge candidate
        expansions.
        """

        self.name_char_ngram.clear()

        for gram, entity_ids in ngram_postings.items():

            if len(entity_ids) > MAX_NGRAM_POSTINGS:
                self.char_ngram_stats[
                    "filtered_common_ngrams"
                ] += 1
                continue

            self.name_char_ngram[gram] = sorted(entity_ids)

        self.char_ngram_stats["indexed_ngrams"] = len(
            self.name_char_ngram
        )

    # ------------------------------------------------------------------
    # Build indexes
    # ------------------------------------------------------------------

    def build_from_paths(
        self,
        paths: Iterable[Path],
    ) -> None:
        """
        Populate indexes from Source 2 and Source 3 TSV files.

        Processing is streaming/chunked through the P1 preprocessing
        iterator.

        Existing indexes are populated immediately.

        Character n-gram postings are first collected temporarily so
        common n-grams can be filtered before becoming retrieval keys.
        """

        # Temporary postings for character n-grams.
        char_ngram_postings: Dict[str, Set[str]] = defaultdict(set)

        for p in paths:

            if not p.exists():
                raise FileNotFoundError(
                    f"Index source not found: {p}"
                )

            # Identify source label.
            p_name_lower = p.name.lower()

            if (
                "source2" in p_name_lower
                or "s2" in p_name_lower
            ):
                source_label = "S2"

            elif (
                "source3" in p_name_lower
                or "s3" in p_name_lower
            ):
                source_label = "S3"

            else:
                source_label = p.stem

            # Independent deduplication for each source.
            seen_dedup_keys: Set[
                Tuple[str, str, str]
            ] = set()

            input_rows = 0
            duplicate_rows_dropped = 0

            for chunk in iter_preprocessed_file(p):

                # Fast column access instead of iterrows().
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

                col_aclean = (
                    chunk["address_clean"]
                    if "address_clean" in chunk
                    else [""] * len(chunk)
                )

                col_cclean = (
                    chunk["country_clean"]
                    if "country_clean" in chunk
                    else [""] * len(chunk)
                )

                for (
                    eid,
                    ntk,
                    ncomp,
                    ncore,
                    atk,
                    nphon,
                    aclean,
                    cclean,
                ) in zip(
                    col_eid,
                    col_ntk,
                    col_ncomp,
                    col_ncore,
                    col_atk,
                    col_nphon,
                    col_aclean,
                    col_cclean,
                ):

                    eid_str = (
                        str(eid).strip()
                        if eid is not None
                        and not pd.isna(eid)
                        else ""
                    )

                    if not eid_str:
                        continue

                    input_rows += 1

                    k_comp = (
                        str(ncomp).strip()
                        if ncomp is not None
                        and not pd.isna(ncomp)
                        else ""
                    )

                    k_addr = (
                        str(aclean).strip()
                        if aclean is not None
                        and not pd.isna(aclean)
                        else ""
                    )

                    k_ctry = (
                        str(cclean).strip()
                        if cclean is not None
                        and not pd.isna(cclean)
                        else ""
                    )

                    # --------------------------------------------------
                    # Existing source-level deduplication
                    # --------------------------------------------------

                    if k_comp or k_addr or k_ctry:

                        dedup_key = (
                            k_comp,
                            k_addr,
                            k_ctry,
                        )

                        if dedup_key in seen_dedup_keys:
                            duplicate_rows_dropped += 1
                            continue

                        seen_dedup_keys.add(dedup_key)

                    # --------------------------------------------------
                    # Existing blocking indexes
                    # --------------------------------------------------

                    ntk_str = (
                        str(ntk).strip()
                        if ntk is not None
                        and not pd.isna(ntk)
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

                    self.add_record(
                        entity_id=eid_str,
                        name_token_key=ntk_str,
                        name_compact=k_comp,
                        name_core=ncore_str,
                        address_token_key=atk_str,
                        name_phonetic_key=nphon_str,
                    )

                    # --------------------------------------------------
                    # New character n-gram blocking index
                    # --------------------------------------------------

                    self._collect_char_ngram_postings(
                        entity_id=eid_str,
                        name_compact=k_comp,
                        ngram_postings=char_ngram_postings,
                    )

            self.dedup_stats[source_label] = {
                "input_rows": input_rows,
                "duplicate_rows_dropped": duplicate_rows_dropped,
            }

        # Finalize character n-gram index after both sources have
        # contributed their postings.
        self._finalize_char_ngram_index(
            char_ngram_postings
        )

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def retrieve_by_name_token(
        self,
        key: str,
    ) -> Set[str]:

        return (
            set(self.name_token_key.get(key, []))
            if key
            else set()
        )

    def retrieve_by_name_compact(
        self,
        key: str,
    ) -> Set[str]:

        return (
            set(self.name_compact.get(key, []))
            if key
            else set()
        )

    def retrieve_by_name_core(
        self,
        key: str,
    ) -> Set[str]:

        return (
            set(self.name_core.get(key, []))
            if key
            else set()
        )

    def retrieve_by_address_token(
        self,
        key: str,
    ) -> Set[str]:

        return (
            set(self.address_token_key.get(key, []))
            if key
            else set()
        )

    def retrieve_by_phonetic(
        self,
        key: str,
    ) -> Set[str]:

        return (
            set(self.name_phonetic_key.get(key, []))
            if key
            else set()
        )

    def retrieve_by_char_ngrams(
        self,
        name_compact: str,
        min_shared_ngrams: int = MIN_SHARED_CHAR_NGRAMS,
    ) -> Set[str]:
        """
        Retrieve candidates using character n-gram overlap.

        A candidate is returned only when it shares at least
        `min_shared_ngrams` indexed character n-grams with the
        Source 1 name.

        The counting is performed per entity ID, so repeated grams
        inside the same name do not artificially increase the score.
        """

        if not name_compact:
            return set()

        grams = set(
            char_ngrams(
                name_compact,
                n=CHAR_NGRAM_SIZE,
            )
        )

        if not grams:
            return set()

        shared_counts: Dict[str, int] = defaultdict(int)

        for gram in grams:

            postings = self.name_char_ngram.get(
                gram
            )

            if not postings:
                continue

            for entity_id in postings:
                shared_counts[entity_id] += 1

        return {
            entity_id
            for entity_id, count in shared_counts.items()
            if count >= min_shared_ngrams
        }

    # ------------------------------------------------------------------
    # Complete candidate retrieval
    # ------------------------------------------------------------------

    def retrieve_candidates_for_row(
        self,
        row: Any,
    ) -> Set[str]:
        """
        Union all P2 blocking passes for one Source 1 row.

        Existing passes:
            - name_token_key
            - name_compact
            - name_core
            - address_token_key
            - name_phonetic_key

        New pass:
            - character n-gram blocking
        """

        get_val = (
            row.get
            if hasattr(row, "get")
            else lambda k, d="": getattr(row, k, d)
        )

        candidates: Set[str] = set()

        # --------------------------------------------------------------
        # Existing five passes
        # --------------------------------------------------------------

        candidates.update(
            self.retrieve_by_name_token(
                str(
                    get_val(
                        "name_token_key",
                        "",
                    )
                    or ""
                )
            )
        )

        candidates.update(
            self.retrieve_by_name_compact(
                str(
                    get_val(
                        "name_compact",
                        "",
                    )
                    or ""
                )
            )
        )

        candidates.update(
            self.retrieve_by_name_core(
                str(
                    get_val(
                        "name_core",
                        "",
                    )
                    or ""
                )
            )
        )

        candidates.update(
            self.retrieve_by_address_token(
                str(
                    get_val(
                        "address_token_key",
                        "",
                    )
                    or ""
                )
            )
        )

        candidates.update(
            self.retrieve_by_phonetic(
                str(
                    get_val(
                        "name_phonetic_key",
                        "",
                    )
                    or ""
                )
            )
        )

        # --------------------------------------------------------------
        # New character n-gram pass
        # --------------------------------------------------------------

        name_compact = str(
            get_val(
                "name_compact",
                "",
            )
            or ""
        ).strip()

        candidates.update(
            self.retrieve_by_char_ngrams(
                name_compact
            )
        )

        return candidates

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """
        Summary statistics for index sizes, deduplication and
        character n-gram filtering.
        """

        return {
            "name_token_key_keys": len(
                self.name_token_key
            ),
            "name_compact_keys": len(
                self.name_compact
            ),
            "name_core_keys": len(
                self.name_core
            ),
            "address_token_key_keys": len(
                self.address_token_key
            ),
            "name_phonetic_key_keys": len(
                self.name_phonetic_key
            ),
            "name_char_ngram_keys": len(
                self.name_char_ngram
            ),
            "char_ngram_stats": self.char_ngram_stats,
            "dedup_stats": self.dedup_stats,
        }


def build_candidate_indexes(
    source2_path: Path,
    source3_path: Path,
) -> Indexes:
    """
    Convenience wrapper that builds indexes from Source 2 and Source 3.
    """

    idx = Indexes()

    idx.build_from_paths(
        [
            source2_path,
            source3_path,
        ]
    )

    return idx
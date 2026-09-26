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
MIN_SHARED_CHAR_NGRAMS = 2
MAX_NGRAM_POSTINGS = 5000

# Maximum postings allowed per blocking key to prevent explosive candidate sets
MAX_KEY_POSTINGS = 2000

# Maximum candidates per S1 entity to prevent unbounded memory usage
MAX_CANDIDATES_PER_S1 = 100


class Indexes:
    """
    Container for inverted indexes used by candidate generation.

    Blocking indexes:
    -----------------
    1. name_token_key: token-sorted name
    2. name_compact: whitespace-stripped name
    3. name_core: legal-suffix-stripped name
    4. name_ascii_compact: diacritic-stripped compact name (e.g. cafe -> cafe)
    5. name_significant_token_key: stopword-filtered core tokens
    6. address_token_key: token-sorted address
    7. address_component_key: house# + postal_code + country
    8. name_phonetic_key: token-sorted Soundex
    9. name_char_ngram: character 3-gram index with posting frequency filtering
    """

    def __init__(self) -> None:
        self.name_token_key: Dict[str, List[str]] = {}
        self.name_compact: Dict[str, List[str]] = {}
        self.name_core: Dict[str, List[str]] = {}
        self.name_ascii_compact: Dict[str, List[str]] = {}
        self.name_significant_token_key: Dict[str, List[str]] = {}
        self.address_token_key: Dict[str, List[str]] = {}
        self.address_component_key: Dict[str, List[str]] = {}
        self.name_phonetic_key: Dict[str, List[str]] = {}
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
        name_ascii_compact: str = "",
        name_significant_token_key: str = "",
        address_component_key: str = "",
    ) -> None:
        """Insert an entity into inverted indexes."""
        if name_token_key:
            self.name_token_key.setdefault(name_token_key, []).append(entity_id)

        if name_compact:
            self.name_compact.setdefault(name_compact, []).append(entity_id)

        if name_core:
            self.name_core.setdefault(name_core, []).append(entity_id)

        if name_ascii_compact:
            self.name_ascii_compact.setdefault(name_ascii_compact, []).append(entity_id)

        if name_significant_token_key:
            self.name_significant_token_key.setdefault(name_significant_token_key, []).append(entity_id)

        if address_token_key:
            self.address_token_key.setdefault(address_token_key, []).append(entity_id)

        if address_component_key:
            self.address_component_key.setdefault(address_component_key, []).append(entity_id)

        if name_phonetic_key:
            self.name_phonetic_key.setdefault(name_phonetic_key, []).append(entity_id)

    def add(self, entity_id: str, row: Any) -> None:
        """Add a single pre-processed row to the indexes."""
        get_val = (
            row.get
            if hasattr(row, "get")
            else lambda k, d="": getattr(row, k, d)
        )
        self.add_record(
            entity_id=entity_id,
            name_token_key=str(get_val("name_token_key", "") or ""),
            name_compact=str(get_val("name_compact", "") or ""),
            name_core=str(get_val("name_core", "") or ""),
            address_token_key=str(get_val("address_token_key", "") or ""),
            name_phonetic_key=str(get_val("name_phonetic_key", "") or ""),
            name_ascii_compact=str(get_val("name_ascii_compact", "") or ""),
            name_significant_token_key=str(get_val("name_significant_token_key", "") or ""),
            address_component_key=str(get_val("address_component_key", "") or ""),
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
                col_ntk = chunk["name_token_key"] if "name_token_key" in chunk else [""] * len(chunk)
                col_ncomp = chunk["name_compact"] if "name_compact" in chunk else [""] * len(chunk)
                col_ncore = chunk["name_core"] if "name_core" in chunk else [""] * len(chunk)
                col_nascii = chunk["name_ascii_compact"] if "name_ascii_compact" in chunk else [""] * len(chunk)
                col_nsig = chunk["name_significant_token_key"] if "name_significant_token_key" in chunk else [""] * len(chunk)
                col_atk = chunk["address_token_key"] if "address_token_key" in chunk else [""] * len(chunk)
                col_acomp = chunk["address_component_key"] if "address_component_key" in chunk else [""] * len(chunk)
                col_nphon = chunk["name_phonetic_key"] if "name_phonetic_key" in chunk else [""] * len(chunk)
                col_aclean = chunk["address_clean"] if "address_clean" in chunk else [""] * len(chunk)
                col_cclean = chunk["country_clean"] if "country_clean" in chunk else [""] * len(chunk)

                for (
                    eid,
                    ntk,
                    ncomp,
                    ncore,
                    atk,
                    nphon,
                    nascii,
                    nsig,
                    acomp,
                    aclean,
                    cclean,
                ) in zip(
                    col_eid,
                    col_ntk,
                    col_ncomp,
                    col_ncore,
                    col_atk,
                    col_nphon,
                    col_nascii,
                    col_nsig,
                    col_acomp,
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
                    # Source-level deduplication
                    # --------------------------------------------------

                    if k_comp or k_addr or k_ctry:
                        dedup_key = (k_comp, k_addr, k_ctry)
                        if dedup_key in seen_dedup_keys:
                            duplicate_rows_dropped += 1
                            continue
                        seen_dedup_keys.add(dedup_key)

                    # --------------------------------------------------
                    # Populate Inverted Indexes
                    # --------------------------------------------------

                    ntk_str = str(ntk).strip() if ntk and not pd.isna(ntk) else ""
                    ncore_str = str(ncore).strip() if ncore and not pd.isna(ncore) else ""
                    atk_str = str(atk).strip() if atk and not pd.isna(atk) else ""
                    nphon_str = str(nphon).strip() if nphon and not pd.isna(nphon) else ""
                    nascii_str = str(nascii).strip() if nascii and not pd.isna(nascii) else ""
                    nsig_str = str(nsig).strip() if nsig and not pd.isna(nsig) else ""
                    acomp_str = str(acomp).strip() if acomp and not pd.isna(acomp) else ""

                    self.add_record(
                        entity_id=eid_str,
                        name_token_key=ntk_str,
                        name_compact=k_comp,
                        name_core=ncore_str,
                        address_token_key=atk_str,
                        name_phonetic_key=nphon_str,
                        name_ascii_compact=nascii_str,
                        name_significant_token_key=nsig_str,
                        address_component_key=acomp_str,
                    )

                    # --------------------------------------------------
                    # Character n-gram postings collection
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

        # Finalize character n-gram index
        self._finalize_char_ngram_index(char_ngram_postings)

    # ------------------------------------------------------------------
    # Retrieval helpers (with frequency filtering)
    # ------------------------------------------------------------------

    def _safe_retrieve(self, index: Dict[str, List[str]], key: str) -> Set[str]:
        """Retrieve postings with frequency filtering to prevent explosive candidate sets."""
        if not key:
            return set()
        postings = index.get(key)
        if not postings:
            return set()
        if len(postings) > MAX_KEY_POSTINGS:
            return set()
        return set(postings)

    def retrieve_by_name_token(self, key: str) -> Set[str]:
        return self._safe_retrieve(self.name_token_key, key)

    def retrieve_by_name_compact(self, key: str) -> Set[str]:
        return self._safe_retrieve(self.name_compact, key)

    def retrieve_by_name_core(self, key: str) -> Set[str]:
        return self._safe_retrieve(self.name_core, key)

    def retrieve_by_name_ascii_compact(self, key: str) -> Set[str]:
        return self._safe_retrieve(self.name_ascii_compact, key)

    def retrieve_by_significant_token(self, key: str) -> Set[str]:
        return self._safe_retrieve(self.name_significant_token_key, key)

    def retrieve_by_address_token(self, key: str) -> Set[str]:
        return self._safe_retrieve(self.address_token_key, key)

    def retrieve_by_address_component(self, key: str) -> Set[str]:
        return self._safe_retrieve(self.address_component_key, key)

    def retrieve_by_phonetic(self, key: str) -> Set[str]:
        return self._safe_retrieve(self.name_phonetic_key, key)

    def retrieve_by_char_ngrams(
        self,
        name_compact: str,
        min_shared_ngrams: int = MIN_SHARED_CHAR_NGRAMS,
    ) -> Set[str]:
        """Retrieve candidates using character n-gram overlap."""
        if not name_compact or len(name_compact) < CHAR_NGRAM_SIZE:
            return set()

        grams = set(char_ngrams(name_compact, n=CHAR_NGRAM_SIZE))
        if not grams:
            return set()

        shared_counts: Dict[str, int] = defaultdict(int)
        for gram in grams:
            postings = self.name_char_ngram.get(gram)
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

    def retrieve_candidates_for_row(self, row: Any) -> Set[str]:
        """Union all P2 blocking passes for one Source 1 row with candidate bounding."""
        get_val = (
            row.get
            if hasattr(row, "get")
            else lambda k, d="": getattr(row, k, d)
        )

        candidates: Set[str] = set()

        # Pass 1: Exact Name Tokens & Compact
        ntk = str(get_val("name_token_key", "") or "")
        ncomp = str(get_val("name_compact", "") or "")
        candidates.update(self.retrieve_by_name_token(ntk))
        candidates.update(self.retrieve_by_name_compact(ncomp))

        # Pass 2: Name Core & ASCII Name
        ncore = str(get_val("name_core", "") or "")
        nascii = str(get_val("name_ascii_compact", "") or "")
        candidates.update(self.retrieve_by_name_core(ncore))
        candidates.update(self.retrieve_by_name_ascii_compact(nascii))

        # Pass 3: Significant Tokens
        nsig = str(get_val("name_significant_token_key", "") or "")
        candidates.update(self.retrieve_by_significant_token(nsig))

        # Pass 4: Address Token & Address Component Key
        atk = str(get_val("address_token_key", "") or "")
        acomp = str(get_val("address_component_key", "") or "")
        candidates.update(self.retrieve_by_address_token(atk))
        candidates.update(self.retrieve_by_address_component(acomp))

        # Pass 5: Phonetic Soundex (expand if candidate volume allows)
        if len(candidates) < MAX_CANDIDATES_PER_S1:
            nphon = str(get_val("name_phonetic_key", "") or "")
            phon_cands = self.retrieve_by_phonetic(nphon)
            candidates.update(phon_cands)

        # Pass 6: Character 3-Gram Tolerant Retrieval
        if len(candidates) < MAX_CANDIDATES_PER_S1:
            ngram_cands = self.retrieve_by_char_ngrams(ncomp)
            candidates.update(ngram_cands)

        return candidates

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Summary statistics for index sizes, deduplication and character n-gram filtering."""
        return {
            "name_token_key_keys": len(self.name_token_key),
            "name_compact_keys": len(self.name_compact),
            "name_core_keys": len(self.name_core),
            "name_ascii_compact_keys": len(self.name_ascii_compact),
            "name_significant_token_key_keys": len(self.name_significant_token_key),
            "address_token_key_keys": len(self.address_token_key),
            "address_component_key_keys": len(self.address_component_key),
            "name_phonetic_key_keys": len(self.name_phonetic_key),
            "name_char_ngram_keys": len(self.name_char_ngram),
            "char_ngram_stats": self.char_ngram_stats,
            "dedup_stats": self.dedup_stats,
        }


def build_candidate_indexes(
    source2_path: Path,
    source3_path: Path,
) -> Indexes:
    """Build Indexes from preprocessed Source 2 and Source 3 files."""
    indexes = Indexes()
    indexes.build_from_paths(
        [
            source2_path,
            source3_path,
        ]
    )
    return indexes
# -*- coding: utf-8 -*-
"""blocking.py

Utility functions for building inverted indexes over the pre‑processed source 2 & 3
records and retrieving candidate entity ids for a given source‑1 row.

The module is deliberately lightweight: it only uses dictionaries that map a
blocking key (e.g. ``name_token_key``) to a list of entity identifiers.  All
entity identifiers keep their original ``S2-`` or ``S3-`` prefix so later stages
can distinguish the source.

Memory usage:
- One dictionary per blocking key (three in total) is built once for the
  entire S2+S3 dataset.  Each entry stores a Python ``list`` of strings –
  suitable for the few‑million‑record scale of the challenge.
- Retrieval operates on a per‑row basis and returns a ``set`` of ids, avoiding
  duplicate work during the union of the three passes.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Set

import pandas as pd

# Local import – the public iterator that yields pre‑processed chunks.
from .preprocessing import iter_preprocessed_file


class Indexes:
    """Container for three inverted indexes used by the baseline blocking.

    Attributes
    ----------
    name_token_key : Dict[str, List[str]]
        Mapping ``name_token_key`` → list of entity ids (S2/S3).
    name_compact : Dict[str, List[str]]
        Mapping ``name_compact`` → list of entity ids.
    address_token_key : Dict[str, List[str]]
        Mapping ``address_token_key`` → list of entity ids.
    """

    def __init__(self) -> None:
        self.name_token_key: Dict[str, List[str]] = {}
        self.name_compact: Dict[str, List[str]] = {}
        self.address_token_key: Dict[str, List[str]] = {}

    def add(self, entity_id: str, row: pd.Series) -> None:
        """Add a single pre‑processed row to the three indexes.

        Parameters
        ----------
        entity_id : str
            The primary key of the record (already prefixed with ``S2-`` or
            ``S3-``).
        row : pandas.Series
            A pre‑processed row containing the six normalized columns.
        """
        def _insert(d: Dict[str, List[str]], key: str) -> None:
            if not key:
                return
            d.setdefault(key, []).append(entity_id)

        _insert(self.name_token_key, row.get("name_token_key", ""))
        _insert(self.name_compact, row.get("name_compact", ""))
        _insert(self.address_token_key, row.get("address_token_key", ""))

    def build_from_paths(self, paths: Iterable[Path]) -> None:
        """Populate the three indexes from all TSV files listed in *paths*.

        The function streams each file with ``iter_preprocessed_file`` to keep
        the memory footprint low while still constructing the full inverted
        indexes.
        """
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Index source not found: {p}")
            for chunk in iter_preprocessed_file(p):
                for _, row in chunk.iterrows():
                    entity_id = str(row.get("entity_id", "")).strip()
                    if not entity_id:
                        continue
                    self.add(entity_id, row)

    # Retrieval helpers ---------------------------------------------------
    def retrieve_by_name_token(self, key: str) -> Set[str]:
        return set(self.name_token_key.get(key, []))

    def retrieve_by_name_compact(self, key: str) -> Set[str]:
        return set(self.name_compact.get(key, []))

    def retrieve_by_address_token(self, key: str) -> Set[str]:
        return set(self.address_token_key.get(key, []))

    def retrieve_candidates_for_row(self, row: pd.Series) -> Set[str]:
        """Union of the three blocking passes for a single source‑1 row."""
        candidates: Set[str] = set()
        candidates.update(self.retrieve_by_name_token(row.get("name_token_key", "")))
        candidates.update(self.retrieve_by_name_compact(row.get("name_compact", "")))
        candidates.update(self.retrieve_by_address_token(row.get("address_token_key", "")))
        return candidates

    def stats(self) -> Dict[str, int]:
        """Simple size statistics for each index (number of distinct keys)."""
        return {
            "name_token_key_keys": len(self.name_token_key),
            "name_compact_keys": len(self.name_compact),
            "address_token_key_keys": len(self.address_token_key),
        }


def build_candidate_indexes(source2_path: Path, source3_path: Path) -> Indexes:
    """Convenience wrapper that builds indexes from the two source files.

    Parameters
    ----------
    source2_path, source3_path : Path
        Paths to the pre‑processed ``source2`` and ``source3`` TSVs.
    """
    idx = Indexes()
    idx.build_from_paths([source2_path, source3_path])
    return idx

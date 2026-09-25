# -*- coding: utf-8 -*-
"""candidate_generation.py

Implementation of the P2 candidate‑generation pipeline.
It consumes the preprocessing iterator for all three sources, builds
exact‑match inverted indexes (name_token_key, name_compact, address_token_key)
for source 2 and source 3, then streams source 1 and writes the candidate
pairs TSV.

The code is deliberately lightweight – it only uses standard library and
pandas, and avoids any heavy‑weight libraries or full‑dataset materialisation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

# Local imports
from .blocking import build_candidate_indexes
from .preprocessing import iter_preprocessed_file


def _write_header(fp) -> None:
    """Write the TSV header required by the validator.

    The validator expects the exact column names:
    ``source1_entity_id`` and ``candidate_entity_ids``.
    """
    fp.write("source1_entity_id\tcandidate_entity_ids\n")


def generate_candidate_pairs(
    source1_path: Path,
    source2_path: Path,
    source3_path: Path,
    output_path: Path,
    chunksize: int = 100_000,
) -> None:
    """Generate ``candidate_pairs.tsv``.

    Parameters
    ----------
    source1_path, source2_path, source3_path : Path
        Paths to the raw TSV files for the three sources.  They are fed
        through the existing ``iter_preprocessed_file`` iterator which
        yields pre‑processed pandas DataFrames.
    output_path : Path
        Destination file for the candidate pairs.  Parent directories are
        created automatically.
    chunksize : int, optional
        Chunk size passed to the preprocessing iterator.  The default of
        100 000 rows balances memory usage and IO.
    """
    # ---------------------------------------------------------------------
    # 1. Build inverted indexes from source 2 and source 3.
    # ---------------------------------------------------------------------
    indexes = build_candidate_indexes(source2_path, source3_path)

    # ---------------------------------------------------------------------
    # 2. Open the output file and write the header.
    # ---------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as out_fp:
        _write_header(out_fp)

        # -----------------------------------------------------------------
        # 3. Stream source 1, retrieve candidates, deduplicate and write.
        # -----------------------------------------------------------------
        for chunk in iter_preprocessed_file(source1_path, chunksize=chunksize):
            # ``chunk`` is a pandas DataFrame where each row corresponds to a
            # source‑1 entity that already contains the normalized columns.
            for _, row in chunk.iterrows():
                s1_id = str(row.get("entity_id", "")).strip()
                if not s1_id:
                    # Skip rows without an identifier – this should not happen
                    # but protects against malformed input.
                    continue

                # Retrieve the union of candidates from the three exact passes.
                candidates = indexes.retrieve_candidates_for_row(row)
                # Remove a potential self‑match (defensive).
                candidates.discard(s1_id)

                # Deterministic ordering: sorted list of IDs.
                sorted_candidates = sorted(candidates)
                cand_str = ",".join(sorted_candidates)
                out_fp.write(f"{s1_id}\t{cand_str}\n")

    # Optional: expose simple stats for the caller (not printed during import).
    return indexes.stats()

__all__ = ["generate_candidate_pairs"]

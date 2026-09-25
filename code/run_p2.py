# -*- coding: utf-8 -*-
"""run_p2.py

Command‑line entry point for the P2 candidate‑generation pipeline.
It wires together argument parsing, index construction and the streaming
candidate‑pair writer defined in ``candidate_generation.py``.

The script does **not** run automatically in this environment – it is only
written so that a teammate can execute ``python code/run_p2.py`` on the full
dataset later.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from business_entity_resolution.candidate_generation import generate_candidate_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate candidate_pairs.tsv for the Amazon ML Challenge 2026"
    )
    parser.add_argument(
        "--source1",
        required=True,
        help="Path to source1 TSV (train or test).",
    )
    parser.add_argument(
        "--source2",
        required=True,
        help="Path to source2 TSV (train or test).",
    )
    parser.add_argument(
        "--source3",
        required=True,
        help="Path to source3 TSV (train or test).",
    )
    parser.add_argument(
        "--output",
        default=Path("output") / "candidate_pairs.tsv",
        type=Path,
        help="Destination TSV file for candidate pairs.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Chunk size for the pre‑processing iterator.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source1 = Path(args.source1)
    source2 = Path(args.source2)
    source3 = Path(args.source3)
    output = Path(args.output)

    # Basic sanity checks – they are inexpensive and avoid obvious user errors.
    for p, name in [(source1, "source1"), (source2, "source2"), (source3, "source3")]:
        if not p.is_file():
            raise FileNotFoundError(f"{name} file not found: {p}")

    # Run the candidate‑generation pipeline.
    stats = generate_candidate_pairs(
        source1_path=source1,
        source2_path=source2,
        source3_path=source3,
        output_path=output,
        chunksize=args.chunksize,
    )

    # Print a tiny summary – this runs only when the script is executed.
    print("Candidate generation completed.")
    print("Index statistics:", stats)
    print(f"Output written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

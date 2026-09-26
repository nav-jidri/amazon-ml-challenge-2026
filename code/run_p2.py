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
import sys
from pathlib import Path

code_dir = str(Path(__file__).resolve().parent)
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

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

    # Print comprehensive diagnostic summary
    print("\n" + "=" * 60)
    print("P2 CANDIDATE GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total S1 rows evaluated:           {stats.get('total_s1_rows', 0):,}")
    print(f"S1 rows with zero candidates:      {stats.get('s1_rows_with_zero_candidates', 0):,}")
    print(f"Total candidate links generated:   {stats.get('total_candidate_links', 0):,}")
    print(f"Average candidates per S1:         {stats.get('average_candidates_per_s1', 0.0):.2f}")
    print(f"Median candidates per S1:          {stats.get('median_candidates_per_s1', 0.0):.1f}")
    print(f"Max candidates for any single S1:  {stats.get('max_candidates_per_s1', 0):,}")
    print("-" * 60)
    print("Candidate Contribution by Blocking Method:")
    contribs = stats.get("method_contributions", {})
    s1_hits = stats.get("method_s1_hits", {})
    for method, count in contribs.items():
        hits = s1_hits.get(method, 0)
        print(f"  - {method:<16}: {count:>10,} links generated (matched {hits:>8,} S1 rows)")
    print("-" * 60)
    print("Inverted Index Sizes (Distinct Keys):")
    print(f"  - name_token_key   : {stats.get('name_token_key_keys', 0):,}")
    print(f"  - name_compact     : {stats.get('name_compact_keys', 0):,}")
    print(f"  - name_core        : {stats.get('name_core_keys', 0):,}")
    print(f"  - address_token_key: {stats.get('address_token_key_keys', 0):,}")
    print(f"  - name_phonetic_key: {stats.get('name_phonetic_key_keys', 0):,}")
    print("-" * 60)
    print("Source Deduplication Summary:")
    dedup = stats.get("dedup_stats", {})
    for src, dstats in dedup.items():
        inp = dstats.get("input_rows", 0)
        dropped = dstats.get("duplicate_rows_dropped", 0)
        kept = inp - dropped
        rate = (dropped / inp * 100) if inp > 0 else 0.0
        print(f"  - {src:<4}: {inp:>10,} input rows, {dropped:>8,} dropped as duplicates ({rate:.2f}%), {kept:>10,} indexed")
    print("=" * 60)
    print(f"Output written to: {output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

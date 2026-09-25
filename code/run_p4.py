# -*- coding: utf-8 -*-
"""run_p4.py

Command-line entry point for Stage P4 (Evaluation + Integration)
for the Amazon ML Challenge 2026 Business Entity Resolution pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

code_dir = str(Path(__file__).resolve().parent)
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from business_entity_resolution.src.p4_evaluation import (
    load_ground_truth,
    load_scored_pairs,
    load_s1_entity_ids,
    select_matches,
    tune_threshold,
    write_matching_results,
    run_validator,
    error_analysis
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage P4 (Evaluation + Integration) pipeline"
    )
    parser.add_argument(
        "--scored-pairs",
        required=True,
        help="Path to candidate_pairs_scored.tsv"
    )
    parser.add_argument(
        "--output",
        default="output/matching_results.tsv",
        help="Path for matching_results.tsv"
    )
    parser.add_argument(
        "--test-source1",
        default="dataset/test/test_source1.tsv",
        help="Path to test_source1.tsv"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Decision threshold"
    )
    parser.add_argument(
        "--ground-truth",
        help="Path to ground truth TSV (optional, for tuning/eval)"
    )
    parser.add_argument(
        "--candidate-pairs",
        help="Path to candidate_pairs.tsv (optional, for validation)"
    )
    parser.add_argument(
        "--test-dir",
        default="dataset/test",
        help="Path to test directory"
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run threshold sweep"
    )
    parser.add_argument(
        "--error-analysis",
        action="store_true",
        help="Run error analysis"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run submission validator after writing"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scored_pairs_path = Path(args.scored_pairs)

    print(f"Loading scored pairs from {scored_pairs_path}...")
    scored_pairs = load_scored_pairs(scored_pairs_path)

    if args.sweep:
        if not args.ground_truth:
            print("Error: --ground-truth is required for --sweep")
            return 1
        print(f"Loading ground truth from {args.ground_truth}...")
        ground_truth = load_ground_truth(Path(args.ground_truth))
        print("Running threshold sweep...")
        best_threshold, best_f05 = tune_threshold(scored_pairs, ground_truth)
        print(f"Optimal threshold: {best_threshold:.4f} (F0.5 = {best_f05:.4f})")
        return 0

    if args.error_analysis:
        if not args.ground_truth:
            print("Error: --ground-truth is required for --error-analysis")
            return 1
        if args.threshold is None:
            print("Error: --threshold is required for --error-analysis")
            return 1
        print(f"Loading ground truth from {args.ground_truth}...")
        ground_truth = load_ground_truth(Path(args.ground_truth))
        print(f"Running error analysis with threshold {args.threshold}...")
        error_analysis(scored_pairs, ground_truth, args.threshold)
        return 0

    # Production mode
    threshold = args.threshold
    if threshold is None:
        if args.ground_truth:
            print(f"Loading ground truth from {args.ground_truth} for threshold tuning...")
            ground_truth = load_ground_truth(Path(args.ground_truth))
            print("Tuning threshold...")
            threshold, best_f05 = tune_threshold(scored_pairs, ground_truth)
            print(f"Optimal threshold found: {threshold:.4f} (F0.5 = {best_f05:.4f})")
        else:
            threshold = 0.50
            print(f"Warning: No threshold or ground truth provided. Using default threshold {threshold}.")

    # Resolve test directory and source1 path dynamically if default path doesn't exist
    test_dir_path = Path(args.test_dir)
    if not test_dir_path.exists():
        for alt in [Path("ml_dataset/data/test"), Path("ml_dataset/test")]:
            if alt.exists():
                test_dir_path = alt
                break

    test_source1_path = Path(args.test_source1)
    if not test_source1_path.exists():
        if (test_dir_path / "test_source1.tsv").exists():
            test_source1_path = test_dir_path / "test_source1.tsv"

    print(f"Selecting matches using threshold {threshold}...")
    matches = select_matches(scored_pairs, threshold)

    print(f"Loading all S1 entity IDs from {test_source1_path}...")
    s1_ids = load_s1_entity_ids(test_source1_path)

    output_path = Path(args.output)
    print(f"Writing matching results to {output_path}...")
    write_matching_results(matches, s1_ids, output_path)

    if args.validate:
        if not args.candidate_pairs:
            print("Error: --candidate-pairs is required for --validate")
            return 1
        print("Running validator...")
        exit_code, val_output = run_validator(
            matching_path=output_path,
            candidate_path=Path(args.candidate_pairs),
            test_dir=test_dir_path
        )
        print(val_output)
        if exit_code != 0:
            print("Validation failed!")
            return 1
        print("Validation successful!")

    print("Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

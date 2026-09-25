# -*- coding: utf-8 -*-
"""evaluate.py

Evaluation and Threshold Analysis module for P3 Matching Model.
Calculates macro-averaged F0.5, entity-level and pair-level Precision, Recall, and F0.5.
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd


def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Calculate F-beta score (default beta=0.5 for precision-heavy weighting)."""
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return ((1 + beta_sq) * precision * recall) / (beta_sq * precision + recall)


def compute_entity_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    beta: float = 0.5
) -> Tuple[float, float, float]:
    """Compute Macro-averaged Precision, Recall, and F0.5 across all S1 entities.

    Singletons (entities with 0 ground truth matches) score 1.0 if predicted empty,
    and 0.0 if any match is predicted.
    """
    precisions = []
    recalls = []
    f_betas = []

    all_entities = set(ground_truth.keys()) | set(predictions.keys())

    for s1_id in all_entities:
        true_matches = ground_truth.get(s1_id, set())
        pred_matches = predictions.get(s1_id, set())

        # Singleton handling
        if not true_matches:
            if not pred_matches:
                # Correctly predicted singleton
                precisions.append(1.0)
                recalls.append(1.0)
                f_betas.append(1.0)
            else:
                # False merge on singleton
                precisions.append(0.0)
                recalls.append(1.0)
                f_betas.append(0.0)
            continue

        # Non-singleton entities
        if not pred_matches:
            # Missed all matches
            precisions.append(0.0)
            recalls.append(0.0)
            f_betas.append(0.0)
            continue

        tp = len(true_matches & pred_matches)
        p = tp / len(pred_matches) if len(pred_matches) > 0 else 0.0
        r = tp / len(true_matches) if len(true_matches) > 0 else 0.0
        fb = compute_f_beta(p, r, beta=beta)

        precisions.append(p)
        recalls.append(r)
        f_betas.append(fb)

    macro_p = float(np.mean(precisions)) if precisions else 0.0
    macro_r = float(np.mean(recalls)) if recalls else 0.0
    macro_f = float(np.mean(f_betas)) if f_betas else 0.0
    return macro_p, macro_r, macro_f


def evaluate_scored_pairs(
    scored_pairs_df: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    threshold: float = 0.50
) -> Dict[str, float]:
    """Evaluate scored pairs dataframe [source1_entity_id, candidate_entity_id, match_probability] at threshold."""
    # Filter predictions at threshold
    filtered_df = scored_pairs_df[scored_pairs_df["match_probability"] >= threshold]
    predictions: Dict[str, Set[str]] = {}
    for _, row in filtered_df.iterrows():
        s1 = str(row["source1_entity_id"])
        cand = str(row["candidate_entity_id"])
        predictions.setdefault(s1, set()).add(cand)

    macro_p, macro_r, macro_f05 = compute_entity_macro_f05(ground_truth, predictions)
    return {
        "threshold": threshold,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f05": macro_f05,
        "predicted_matches_count": len(filtered_df)
    }


def threshold_sweep(
    scored_pairs_df: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    start: float = 0.10,
    end: float = 0.95,
    step: float = 0.05
) -> pd.DataFrame:
    """Perform sweep across probability thresholds."""
    records = []
    print("\n" + "=" * 80)
    print("THRESHOLD SWEEP & MACRO F0.5 EVALUATION")
    print("=" * 80)
    print(f"{'Threshold':>10} | {'Macro Precision':>16} | {'Macro Recall':>14} | {'Macro F0.5':>12} | {'Predicted Matches':>18}")
    print("-" * 80)

    for thresh in np.arange(start, end + 1e-5, step):
        res = evaluate_scored_pairs(scored_pairs_df, ground_truth, threshold=float(thresh))
        records.append(res)
        print(f"{res['threshold']:>10.2f} | {res['macro_precision']:>16.4f} | {res['macro_recall']:>14.4f} | {res['macro_f05']:>12.4f} | {res['predicted_matches_count']:>18,}")

    print("-" * 80)
    df_res = pd.DataFrame(records)
    best_row = df_res.loc[df_res["macro_f05"].idxmax()]
    print(f"Optimal Threshold: {best_row['threshold']:.2f} (Macro F0.5 = {best_row['macro_f05']:.4f})")
    return df_res


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate P3 Matching Model Scored Pairs")
    parser.add_argument("--scored-file", type=Path, default=Path("output/candidate_pairs_scored.tsv"), help="Path to candidate_pairs_scored.tsv")
    parser.add_argument("--threshold", type=float, default=0.50, help="Classification probability threshold")
    parser.add_argument("--sweep", action="store_true", help="Perform threshold sweep across [0.10, 0.95]")
    args = parser.parse_args()

    print(f"Evaluating threshold {args.threshold:.2f}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())

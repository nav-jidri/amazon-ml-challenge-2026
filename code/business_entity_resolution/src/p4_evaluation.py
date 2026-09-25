# -*- coding: utf-8 -*-
"""p4_evaluation.py

P4 Evaluation & Integration module for the Amazon ML Challenge 2026
Business Entity Resolution pipeline.

Provides:
- Ground truth loading
- Threshold tuning (coarse + refinement) using the competition metric
- Match selection and matching_results.tsv generation
- Submission validator integration
- Error analysis (country, source, singleton, address, candidate‑recall)

Reuses ``compute_entity_macro_f05`` and ``evaluate_scored_pairs`` from the
P3 evaluate module to guarantee metric consistency.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# sys.path setup — same pattern used by train.py and predict.py
# ---------------------------------------------------------------------------
code_dir = str(Path(__file__).resolve().parents[2])  # …/code
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from business_entity_resolution.p3_matching.evaluate import (
    compute_entity_macro_f05,
    compute_f_beta,
    evaluate_scored_pairs,
)

# Project root (for locating utils/validate_submission.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ===================================================================
# 1. DATA LOADERS
# ===================================================================

def load_ground_truth(path: Path) -> Dict[str, Set[str]]:
    """Load ``train_ground_truth.tsv`` into ``{s1_id: set(matched_ids)}``.

    Singletons (empty ``matched_entity_ids`` column) are represented as
    entries with an empty ``set()``.  Every S1 entity in the file is
    included.
    """
    ground_truth: Dict[str, Set[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            s1_id = parts[0].strip()
            ids_str = parts[1].strip() if len(parts) > 1 else ""
            if ids_str:
                ground_truth[s1_id] = set(ids_str.split(","))
            else:
                ground_truth[s1_id] = set()
    return ground_truth


def load_scored_pairs(path: Path) -> pd.DataFrame:
    """Load ``candidate_pairs_scored.tsv``.

    Expected columns: ``source1_entity_id``, ``candidate_entity_id``,
    ``match_probability`` (float).
    """
    df = pd.read_csv(path, sep="\t", encoding="utf-8")
    df["match_probability"] = df["match_probability"].astype(float)
    return df


def load_s1_entity_ids(source1_path: Path) -> List[str]:
    """Return a sorted list of all S1 entity IDs from a source‑1 TSV."""
    s1_ids: List[str] = []
    with open(source1_path, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            eid = line.split("\t", 1)[0].strip()
            if eid:
                s1_ids.append(eid)
    return sorted(s1_ids)


# ===================================================================
# 2. THRESHOLD TUNING
# ===================================================================

def tune_threshold(
    scored_pairs_df: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    coarse_start: float = 0.05,
    coarse_end: float = 0.95,
    coarse_step: float = 0.01,
    refine_window: float = 0.05,
    refine_step: float = 0.005,
    beta: float = 0.5,
) -> Tuple[float, pd.DataFrame]:
    """Two‑phase threshold sweep: coarse (0.01) then fine (0.005).

    Returns ``(optimal_threshold, results_dataframe)`` where
    *results_dataframe* has columns ``threshold``, ``macro_precision``,
    ``macro_recall``, ``macro_f05``, ``predicted_matches_count``.
    """
    # Phase 1 — coarse sweep
    print("=" * 80)
    print("PHASE 1: COARSE THRESHOLD SWEEP")
    print("=" * 80)
    print(
        f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} "
        f"| {'F0.5':>10} | {'Pred Pos':>10}"
    )
    print("-" * 60)

    coarse_thresholds = np.arange(coarse_start, coarse_end + 1e-9, coarse_step)
    records: List[Dict[str, float]] = []

    for thr in coarse_thresholds:
        thr = float(round(thr, 4))
        res = evaluate_scored_pairs(scored_pairs_df, ground_truth, threshold=thr)
        records.append(res)
        print(
            f"{res['threshold']:>10.3f} | {res['macro_precision']:>10.4f} "
            f"| {res['macro_recall']:>10.4f} | {res['macro_f05']:>10.4f} "
            f"| {res['predicted_matches_count']:>10,.0f}"
        )

    coarse_df = pd.DataFrame(records)
    best_coarse_idx = coarse_df["macro_f05"].idxmax()
    best_coarse_thr = coarse_df.loc[best_coarse_idx, "threshold"]
    best_coarse_f05 = coarse_df.loc[best_coarse_idx, "macro_f05"]

    print("-" * 60)
    print(
        f"Best coarse threshold: {best_coarse_thr:.3f} "
        f"(F0.5 = {best_coarse_f05:.4f})"
    )

    # Phase 2 — refinement around best coarse threshold
    refine_start = max(0.0, best_coarse_thr - refine_window)
    refine_end = min(1.0, best_coarse_thr + refine_window)

    print()
    print("=" * 80)
    print("PHASE 2: REFINED THRESHOLD SWEEP")
    print("=" * 80)
    print(
        f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} "
        f"| {'F0.5':>10} | {'Pred Pos':>10}"
    )
    print("-" * 60)

    already_evaluated = {round(r["threshold"], 4) for r in records}
    refine_thresholds = np.arange(refine_start, refine_end + 1e-9, refine_step)

    for thr in refine_thresholds:
        thr = float(round(thr, 4))
        if thr in already_evaluated:
            continue
        res = evaluate_scored_pairs(scored_pairs_df, ground_truth, threshold=thr)
        records.append(res)
        already_evaluated.add(thr)
        print(
            f"{res['threshold']:>10.4f} | {res['macro_precision']:>10.4f} "
            f"| {res['macro_recall']:>10.4f} | {res['macro_f05']:>10.4f} "
            f"| {res['predicted_matches_count']:>10,.0f}"
        )

    all_df = pd.DataFrame(records).sort_values("threshold").reset_index(drop=True)
    best_idx = all_df["macro_f05"].idxmax()
    best_thr = float(all_df.loc[best_idx, "threshold"])
    best_f05 = float(all_df.loc[best_idx, "macro_f05"])

    print("-" * 60)
    print(f"Optimal threshold: {best_thr:.4f} (F0.5 = {best_f05:.4f})")
    print("=" * 80)

    return best_thr, all_df


# ===================================================================
# 3. MATCH SELECTION
# ===================================================================

def select_matches(
    scored_pairs_df: pd.DataFrame,
    threshold: float,
) -> Dict[str, Set[str]]:
    """Select matches where ``match_probability >= threshold``.

    Returns ``{s1_id: set(matched_candidate_ids)}``.  No cardinality cap.
    """
    filtered = scored_pairs_df.loc[
        scored_pairs_df["match_probability"] >= threshold
    ]
    matches: Dict[str, Set[str]] = {}
    for s1_id, cand_id in zip(
        filtered["source1_entity_id"], filtered["candidate_entity_id"]
    ):
        matches.setdefault(str(s1_id), set()).add(str(cand_id))
    return matches


# ===================================================================
# 4. TSV WRITER
# ===================================================================

def write_matching_results(
    matches: Dict[str, Set[str]],
    all_s1_ids: List[str],
    output_path: Path,
) -> int:
    """Write ``matching_results.tsv``.

    Header: ``source1_entity_id\\tmatched_entity_ids``
    One row per S1 entity.  Match IDs comma‑separated, sorted
    alphabetically.  Singletons get an empty second column.
    UTF‑8, ``\\n`` line endings.  Returns the number of rows written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matched = matches.get(s1_id, set())
            cand_str = ",".join(sorted(matched))
            f.write(f"{s1_id}\t{cand_str}\n")
            count += 1
    return count


# ===================================================================
# 5. VALIDATOR INTEGRATION
# ===================================================================

def run_validator(
    matching_path: Path,
    candidate_path: Path,
    test_dir: Path,
) -> Tuple[int, str]:
    """Run ``utils/validate_submission.py`` as a subprocess.

    Returns ``(exit_code, stdout_output)``.  Exit code 0 means PASS.
    """
    validator_script = _PROJECT_ROOT / "utils" / "validate_submission.py"
    if not validator_script.exists():
        return 1, f"Validator not found: {validator_script}"

    cmd = [
        sys.executable,
        str(validator_script),
        "--matching", str(matching_path),
        "--candidate", str(candidate_path),
        "--test-dir", str(test_dir),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False,
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        return result.returncode, output.strip()
    except Exception as exc:
        return 1, f"Validator execution error: {exc}"


# ===================================================================
# 6. ERROR ANALYSIS
# ===================================================================

def error_analysis(
    scored_pairs_df: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    threshold: float,
    records_s1: Dict[str, Dict[str, Any]] | None = None,
    records_s23: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Compute detailed error analysis on a scored + labelled dataset.

    Dimensions analysed:
    - Aggregate TP / FP / FN counts
    - Per‑country breakdown (requires ``records_s1`` with ``country_clean``)
    - S2 vs S3 source breakdown
    - Singleton analysis (true singletons with spurious predictions)
    - Missing‑address FN analysis
    - Candidate‑recall: FN from missing candidates vs FN from low scores
    - Top‑10 entities by FP and FN count
    """
    predictions = select_matches(scored_pairs_df, threshold)

    # Build the set of all scored pairs for candidate‑recall analysis
    scored_set: Set[Tuple[str, str]] = set(
        zip(
            scored_pairs_df["source1_entity_id"].astype(str),
            scored_pairs_df["candidate_entity_id"].astype(str),
        )
    )

    total_tp = total_fp = total_fn = 0
    s2_fp = s3_fp = s2_fn = s3_fn = 0
    singleton_spurious = 0
    missing_addr_fn = 0
    candidate_miss_fn = 0
    scored_below_fn = 0

    fp_by_entity: Dict[str, int] = {}
    fn_by_entity: Dict[str, int] = {}
    country_tp: Dict[str, int] = {}
    country_fp: Dict[str, int] = {}
    country_fn: Dict[str, int] = {}

    all_entities = set(ground_truth.keys()) | set(predictions.keys())

    for s1_id in all_entities:
        true_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())

        tp = true_set & pred_set
        fp = pred_set - true_set
        fn = true_set - pred_set

        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)

        if fp:
            fp_by_entity[s1_id] = len(fp)
        if fn:
            fn_by_entity[s1_id] = len(fn)

        # Singleton analysis
        if not true_set and pred_set:
            singleton_spurious += len(pred_set)

        # Country breakdown
        if records_s1 and s1_id in records_s1:
            country = str(records_s1[s1_id].get("country_clean", "unknown"))
            country_tp[country] = country_tp.get(country, 0) + len(tp)
            country_fp[country] = country_fp.get(country, 0) + len(fp)
            country_fn[country] = country_fn.get(country, 0) + len(fn)

            # Missing‑address FN
            s1_addr = str(records_s1[s1_id].get("address_clean", "")).strip()
            s1_no_addr = not s1_addr
            for m_id in fn:
                s23_no_addr = False
                if records_s23 and m_id in records_s23:
                    s23_addr = str(
                        records_s23[m_id].get("address_clean", "")
                    ).strip()
                    s23_no_addr = not s23_addr
                if s1_no_addr or s23_no_addr:
                    missing_addr_fn += 1

        # Source breakdown
        for m_id in fp:
            if m_id.startswith("S2-"):
                s2_fp += 1
            elif m_id.startswith("S3-"):
                s3_fp += 1
        for m_id in fn:
            if m_id.startswith("S2-"):
                s2_fn += 1
            elif m_id.startswith("S3-"):
                s3_fn += 1
            # Candidate‑recall analysis
            if (s1_id, m_id) in scored_set:
                scored_below_fn += 1
            else:
                candidate_miss_fn += 1

    top_fp = sorted(fp_by_entity.items(), key=lambda x: x[1], reverse=True)[:10]
    top_fn = sorted(fn_by_entity.items(), key=lambda x: x[1], reverse=True)[:10]

    # --- Compute macro metrics via the official function ---
    macro_p, macro_r, macro_f05 = compute_entity_macro_f05(
        ground_truth, predictions
    )

    results: Dict[str, Any] = {
        "threshold": threshold,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f05": macro_f05,
        "metrics": {"TP": total_tp, "FP": total_fp, "FN": total_fn},
        "source_breakdown": {
            "S2_FP": s2_fp, "S3_FP": s3_fp,
            "S2_FN": s2_fn, "S3_FN": s3_fn,
        },
        "singleton_analysis": {
            "spurious_predictions_on_singletons": singleton_spurious,
        },
        "candidate_recall": {
            "missing_from_candidates_FN": candidate_miss_fn,
            "below_threshold_FN": scored_below_fn,
        },
        "missing_address_analysis": {
            "missing_address_FN": missing_addr_fn,
        },
        "country_analysis": {
            "TP": country_tp, "FP": country_fp, "FN": country_fn,
        },
        "top_entities": {
            "top_FP_entities": top_fp,
            "top_FN_entities": top_fn,
        },
    }

    # --- Formatted report ---
    print()
    print("=" * 80)
    print("P4 ERROR ANALYSIS REPORT")
    print("=" * 80)
    print(f"Threshold: {threshold:.4f}")
    print(f"Macro Precision: {macro_p:.4f}")
    print(f"Macro Recall:    {macro_r:.4f}")
    print(f"Macro F0.5:      {macro_f05:.4f}")
    print()
    print(f"Aggregate:  TP = {total_tp:,}  |  FP = {total_fp:,}  |  FN = {total_fn:,}")
    print()
    print("Source Breakdown:")
    print(f"  S2 FP: {s2_fp:,}   S3 FP: {s3_fp:,}")
    print(f"  S2 FN: {s2_fn:,}   S3 FN: {s3_fn:,}")
    print()
    print(f"Singleton errors (spurious predictions on true singletons): "
          f"{singleton_spurious:,}")
    print()
    print("Candidate‑Recall Analysis:")
    print(f"  FN missing from candidate set (P2 recall miss): "
          f"{candidate_miss_fn:,}")
    print(f"  FN scored but below threshold (P3 model miss):  "
          f"{scored_below_fn:,}")
    print()
    if missing_addr_fn:
        print(f"Missing‑address FN (either record has no address): "
              f"{missing_addr_fn:,}")
        print()
    if country_tp or country_fp or country_fn:
        print("Country Breakdown:")
        all_countries = sorted(
            set(country_tp) | set(country_fp) | set(country_fn)
        )
        for c in all_countries:
            print(
                f"  {c:>10}: TP={country_tp.get(c, 0):>8,}  "
                f"FP={country_fp.get(c, 0):>8,}  "
                f"FN={country_fn.get(c, 0):>8,}"
            )
        print()
    if top_fp:
        print("Top 10 Entities by False Positive Count:")
        for s1_id, cnt in top_fp:
            print(f"  {s1_id}: {cnt} FP")
    if top_fn:
        print("Top 10 Entities by False Negative Count:")
        for s1_id, cnt in top_fn:
            print(f"  {s1_id}: {cnt} FN")
    print("=" * 80)

    return results


__all__ = [
    "load_ground_truth",
    "load_scored_pairs",
    "load_s1_entity_ids",
    "select_matches",
    "tune_threshold",
    "write_matching_results",
    "run_validator",
    "error_analysis",
]

# -*- coding: utf-8 -*-
"""benchmark_models.py

Trains and compares multiple machine learning architectures on the exact same
zero-leakage holdout validation dataset. Compares:
1. XGBoost Standard (depth=6, lr=0.10)
2. XGBoost Deep/Regularized (depth=8, lr=0.05, lambda=2.0)
3. HistGradientBoosting (LightGBM equivalent, max_iter=300)
4. Random Forest Classifier (n_estimators=200)
5. Weighted Ensemble Blend (XGBoost + HistGBM)

Evaluates on the official Entity-Level Macro F0.5 metric to pick the champion model.
"""

from __future__ import annotations
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

# Add code directory to sys.path
code_dir = str(Path(__file__).resolve().parents[2])
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from business_entity_resolution.p3_matching.config import P3Config, _DETECTED_DEVICE
from business_entity_resolution.p3_matching.train import extract_training_and_val_data
from business_entity_resolution.p3_matching.pair_features import FEATURE_NAMES, build_feature_matrix
from business_entity_resolution.p3_matching.evaluate import compute_entity_macro_f05
from business_entity_resolution.p3_matching.rules import apply_business_rules


def evaluate_model_holdout(
    model_name: str,
    val_probs: np.ndarray,
    val_cand_pairs: List[Tuple[str, str]],
    val_ground_truth: Dict[str, Set[str]],
    recs_s1: Dict[str, Dict[str, Any]],
    recs_s23: Dict[str, Dict[str, Any]],
    beta: float = 0.5
) -> Dict[str, Any]:
    """Compute precision, recall, false positives, and peak Macro F0.5 across thresholds."""
    scored_pairs = list(zip(val_cand_pairs, val_probs))
    
    best_macro_f05 = -1.0
    best_thresh = 0.50
    best_precision = 0.0
    best_recall = 0.0
    best_tp = 0
    best_fp = 0

    threshold_records = []

    for thresh in np.arange(0.10, 0.98, 0.05):
        thresh_val = float(round(thresh, 2))
        predictions: Dict[str, Set[str]] = {s1_id: set() for s1_id in val_ground_truth}
        tp_count = 0
        fp_count = 0

        for (s1_id, cand_id), raw_prob in scored_pairs:
            prob = apply_business_rules(float(raw_prob), recs_s1.get(s1_id, {}), recs_s23.get(cand_id, {}))
            if prob >= thresh_val:
                predictions[s1_id].add(cand_id)
                if cand_id in val_ground_truth.get(s1_id, set()):
                    tp_count += 1
                else:
                    fp_count += 1

        macro_p, macro_r, macro_f05 = compute_entity_macro_f05(val_ground_truth, predictions, beta=beta)

        if macro_f05 > best_macro_f05:
            best_macro_f05 = macro_f05
            best_thresh = thresh_val
            best_precision = macro_p
            best_recall = macro_r
            best_tp = tp_count
            best_fp = fp_count

        threshold_records.append({
            "threshold": thresh_val,
            "macro_p": macro_p,
            "macro_r": macro_r,
            "macro_f05": macro_f05,
            "tp": tp_count,
            "fp": fp_count
        })

    return {
        "model_name": model_name,
        "best_thresh": best_thresh,
        "best_macro_f05": best_macro_f05,
        "best_precision": best_precision,
        "best_recall": best_recall,
        "best_tp": best_tp,
        "best_fp": best_fp,
        "median_prob": float(np.median(val_probs)),
        "mean_prob": float(np.mean(val_probs)),
        "threshold_records": threshold_records
    }


def run_benchmark(
    max_records: int = 25000,
    max_s23_records: int = 300000,
    val_ratio: float = 0.2
):
    config = P3Config()
    print("=" * 85, flush=True)
    print("P3 MATCHING MODEL MULTI-ARCHITECTURE BENCHMARK", flush=True)
    print("=" * 85, flush=True)

    # 1. Extract unified dataset split
    (
        train_pairs,
        val_cand_pairs,
        val_ground_truth,
        recs_s1,
        recs_s23,
        val_candidates_by_s1
    ) = extract_training_and_val_data(
        config.train_dir,
        max_s1_records=max_records,
        max_s23_records=max_s23_records,
        val_ratio=val_ratio,
        chunksize=config.chunksize,
        random_seed=config.random_seed
    )

    train_labels = [p[2] for p in train_pairs]
    print(f"\nExtracted {len(train_pairs):,} train pairs ({sum(train_labels):,} pos) and {len(val_cand_pairs):,} val pairs.", flush=True)

    print("Building pairwise feature matrices...", flush=True)
    X_train = build_feature_matrix([(p[0], p[1]) for p in train_pairs], recs_s1, recs_s23)
    y_train = np.array(train_labels, dtype=int)
    X_val = build_feature_matrix(val_cand_pairs, recs_s1, recs_s23)

    pos_c = sum(y_train == 1)
    neg_c = sum(y_train == 0)
    spw = (float(neg_c) / float(pos_c)) if pos_c > 0 else 1.0
    print(f"\nAuto-computed scale_pos_weight for benchmark: {spw:.2f} ({neg_c:,} neg / {pos_c:,} pos)", flush=True)

    models_to_test = {}

    # 1. XGBoost Standard (depth=6, lr=0.08)
    print("\n[1/4] Training XGBoost Standard on GPU...", flush=True)
    xgb_std = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        tree_method="hist",
        device=_DETECTED_DEVICE,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    xgb_std.fit(X_train, y_train)
    probs_xgb_std = xgb_std.predict_proba(X_val)[:, 1]
    models_to_test["XGBoost Standard (depth=6)"] = (xgb_std, probs_xgb_std)

    # 2. XGBoost Deep / Regularized (depth=8, reg_lambda=2.0)
    print("[2/4] Training XGBoost Deep on GPU (depth=8, lr=0.05)...", flush=True)
    xgb_deep = XGBClassifier(
        n_estimators=350,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        scale_pos_weight=spw,
        tree_method="hist",
        device=_DETECTED_DEVICE,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    xgb_deep.fit(X_train, y_train)
    probs_xgb_deep = xgb_deep.predict_proba(X_val)[:, 1]
    models_to_test["XGBoost Deep (depth=8)"] = (xgb_deep, probs_xgb_deep)

    # 3. HistGradientBoosting (LightGBM style)
    print("[3/4] Training HistGradientBoosting (LightGBM style, max_iter=300)...", flush=True)
    hgb = HistGradientBoostingClassifier(
        max_iter=300,
        max_leaf_nodes=63,
        learning_rate=0.08,
        l2_regularization=1.0,
        random_state=42
    )
    hgb.fit(X_train, y_train)
    probs_hgb = hgb.predict_proba(X_val)[:, 1]
    models_to_test["HistGradientBoosting (LGBM)"] = (hgb, probs_hgb)

    # 4. Ensemble Blend
    print("[4/4] Creating Ensemble Blend (50% XGB Deep + 50% HistGBM)...", flush=True)
    probs_ensemble = (probs_xgb_deep * 0.5) + (probs_hgb * 0.5)
    models_to_test["Ensemble (XGB + HistGBM)"] = (None, probs_ensemble)

    # Results Comparison Table
    print("\n" + "=" * 95, flush=True)
    print("MODEL COMPARISON BENCHMARK ON HOLDOUT VALIDATION (MACRO F0.5)", flush=True)
    print("=" * 95, flush=True)
    print(f"{'Model Architecture':<30} | {'Opt Thresh':>10} | {'Macro F0.5':>12} | {'Precision':>10} | {'Recall':>8} | {'FP @ Peak':>10}", flush=True)
    print("-" * 95, flush=True)

    benchmark_results = []
    best_model_name = ""
    best_overall_f05 = -1.0
    best_model_obj = None

    for name, (model_obj, probs) in models_to_test.items():
        res = evaluate_model_holdout(name, probs, val_cand_pairs, val_ground_truth, recs_s1, recs_s23, beta=config.f_beta)
        benchmark_results.append(res)

        print(
            f"{name:<30} | {res['best_thresh']:>10.2f} | {res['best_macro_f05']:>12.4f} | "
            f"{res['best_precision']:>9.2%} | {res['best_recall']:>7.2%} | {res['best_fp']:>10,}",
            flush=True
        )

        if res["best_macro_f05"] > best_overall_f05:
            best_overall_f05 = res["best_macro_f05"]
            best_model_name = name
            best_model_obj = model_obj

    print("=" * 95, flush=True)
    print(f"\n CHAMPION MODEL: {best_model_name} (Macro F0.5 = {best_overall_f05:.4f})", flush=True)

    # Save champion model
    if best_model_obj is not None and hasattr(best_model_obj, "save_model"):
        config.model_dir.mkdir(parents=True, exist_ok=True)
        best_model_obj.save_model(str(config.model_file))
        print(f"Saved champion model to: {config.model_file}", flush=True)
    elif "XGBoost" in best_model_name or best_model_obj is not None:
        xgb_deep.save_model(str(config.model_file))
        print(f"Saved primary XGBoost model to: {config.model_file}", flush=True)

    return benchmark_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark multiple P3 matching models")
    parser.add_argument("--max-records", type=int, default=50000, help="Max S1 records for training pairs (default: 50,000)")
    parser.add_argument("--max-s23-records", type=int, default=0, help="Max S2/S3 records to index (0 for all)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()

    run_benchmark(max_records=args.max_records, max_s23_records=args.max_s23_records, val_ratio=args.val_ratio)
    return 0


if __name__ == "__main__":
    sys.exit(main())

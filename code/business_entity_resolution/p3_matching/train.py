# -*- coding: utf-8 -*-
"""train.py

Training script for P3 Matching Model (XGBoost).
Generates training pairs from training sources, creates pairwise features,
splits train/validation without entity leakage, trains XGBClassifier, and saves artifacts.
"""

from __future__ import annotations
import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

# Add code directory to sys.path
code_dir = str(Path(__file__).resolve().parents[2])
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from business_entity_resolution.preprocessing import iter_preprocessed_file
from business_entity_resolution.blocking import Indexes
from business_entity_resolution.p3_matching.config import P3Config
from business_entity_resolution.p3_matching.pair_features import FEATURE_NAMES, build_feature_matrix, compute_pair_feature_vector


def load_ground_truth(gt_path: Path) -> Dict[str, Set[str]]:
    """Load ground truth mappings from train_ground_truth.tsv into {s1_id: set(matched_ids)}."""
    ground_truth: Dict[str, Set[str]] = {}
    if not gt_path.exists():
        print(f"  Warning: Ground truth file not found at {gt_path}")
        return ground_truth

    with open(gt_path, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            s1_id = parts[0].strip()
            if not s1_id:
                continue
            matched_str = parts[1].strip() if len(parts) > 1 else ""
            if matched_str:
                ground_truth[s1_id] = {m.strip() for m in matched_str.split(",") if m.strip()}
            else:
                ground_truth[s1_id] = set()
    return ground_truth


def extract_training_and_val_data(
    train_dir: Path,
    max_s1_records: int = 25000,
    val_ratio: float = 0.2,
    chunksize: int = 50000,
    random_seed: int = 42
) -> Tuple[
    List[Tuple[str, str, int]],                      # train_labeled_pairs
    List[Tuple[str, str]],                           # val_candidate_pairs
    Dict[str, Set[str]],                             # val_ground_truth
    Dict[str, Dict[str, Any]],                       # records_s1
    Dict[str, Dict[str, Any]],                       # records_s23
    Dict[str, Set[str]]                              # val_candidates_by_s1
]:
    """Generate training pairs with true labels and realistic validation holdout sets.

    Train pairs:
      - Contains candidates retrieved by P2 blocking for train_s1_ids with true GT labels.
      - Augments true GT matches so the classifier learns positive patterns.

    Validation holdout:
      - S1 entities in val_s1_ids are held out completely (zero leakage).
      - Candidates are ONLY what P2 blocking actually retrieved (no artificial GT injection).
      - Ground truth covers ALL val_s1_ids (including singletons and 0-candidate entities).
    """
    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"
    gt_path = train_dir / "train_ground_truth.tsv"

    print(f"Loading training records from {train_dir}...")
    
    # 1. Load official ground truth
    print(f"  Loading official ground truth from {gt_path.name}...")
    ground_truth = load_ground_truth(gt_path)
    print(f"  Loaded ground truth for {len(ground_truth):,} Source 1 entities.")

    # 2. Store preprocessed record metadata and build blocking indexes from S2/S3
    records_s1: Dict[str, Dict[str, Any]] = {}
    records_s23: Dict[str, Dict[str, Any]] = {}

    indexes = Indexes()
    for src_path in [s2_path, s3_path]:
        print(f"  Indexing {src_path.name}...")
        for chunk in iter_preprocessed_file(src_path, chunksize=chunksize):
            col_core = chunk["name_core"] if "name_core" in chunk else [""] * len(chunk)
            col_phon = chunk["name_phonetic_key"] if "name_phonetic_key" in chunk else [""] * len(chunk)
            col_nascii = chunk["name_ascii_compact"] if "name_ascii_compact" in chunk else [""] * len(chunk)
            col_nsig = chunk["name_significant_token_key"] if "name_significant_token_key" in chunk else [""] * len(chunk)
            col_acomp = chunk["address_component_key"] if "address_component_key" in chunk else [""] * len(chunk)

            for eid, nclean, ncomp, ncore, ntk, nphon, nascii, nsig, acomp, aclean, atk, cclean in zip(
                chunk["entity_id"],
                chunk["name_clean"],
                chunk["name_compact"],
                col_core,
                chunk["name_token_key"],
                col_phon,
                col_nascii,
                col_nsig,
                col_acomp,
                chunk["address_clean"],
                chunk["address_token_key"],
                chunk["country_clean"]
            ):
                eid_str = str(eid).strip()
                if not eid_str:
                    continue
                records_s23[eid_str] = {
                    "name_clean": nclean,
                    "name_compact": ncomp,
                    "name_core": ncore,
                    "name_token_key": ntk,
                    "address_clean": aclean,
                    "address_token_key": atk,
                    "country_clean": cclean,
                }
                indexes.add_record(
                    entity_id=eid_str,
                    name_token_key=str(ntk).strip() if ntk else "",
                    name_compact=str(ncomp).strip() if ncomp else "",
                    name_core=str(ncore).strip() if ncore else "",
                    address_token_key=str(atk).strip() if atk else "",
                    name_phonetic_key=str(nphon).strip() if nphon else "",
                    name_ascii_compact=str(nascii).strip() if nascii else "",
                    name_significant_token_key=str(nsig).strip() if nsig else "",
                    address_component_key=str(acomp).strip() if acomp else "",
                )

    print(f"Total S2/S3 indexed records: {len(records_s23):,}")

    # 3. Stream S1 and collect candidates
    print(f"Streaming S1 records (up to {max_s1_records:,})...")
    s1_ids_ordered: List[str] = []
    s1_candidates_map: Dict[str, Set[str]] = {}

    s1_count = 0
    for chunk in iter_preprocessed_file(s1_path, chunksize=chunksize):
        for _, row in chunk.iterrows():
            s1_id = str(row.get("entity_id", "")).strip()
            if not s1_id:
                continue

            records_s1[s1_id] = {
                "name_clean": row.get("name_clean", ""),
                "name_compact": row.get("name_compact", ""),
                "name_core": row.get("name_core", ""),
                "name_token_key": row.get("name_token_key", ""),
                "address_clean": row.get("address_clean", ""),
                "address_token_key": row.get("address_token_key", ""),
                "country_clean": row.get("country_clean", ""),
            }

            candidates = indexes.retrieve_candidates_for_row(row)
            candidates.discard(s1_id)

            s1_ids_ordered.append(s1_id)
            s1_candidates_map[s1_id] = candidates

            s1_count += 1
            if s1_count >= max_s1_records:
                break
        if s1_count >= max_s1_records:
            break

    print(f"Processed {len(s1_ids_ordered):,} S1 entities.")

    # 4. Split S1 entities into Train and Validation holdout sets
    rng = np.random.RandomState(random_seed)
    shuffled_s1 = s1_ids_ordered.copy()
    rng.shuffle(shuffled_s1)

    n_val = int(len(shuffled_s1) * val_ratio)
    val_s1_set = set(shuffled_s1[:n_val])
    train_s1_set = set(shuffled_s1[n_val:])

    print(f"  Train S1 entities: {len(train_s1_set):,}")
    print(f"  Holdout Val S1 entities: {len(val_s1_set):,}")

    # 5. Build Training Pairs
    train_labeled_pairs: List[Tuple[str, str, int]] = []
    for s1_id in train_s1_set:
        cands = s1_candidates_map.get(s1_id, set())
        true_set = ground_truth.get(s1_id, set())

        for cand_id in cands:
            if cand_id in records_s23:
                label = 1 if cand_id in true_set else 0
                train_labeled_pairs.append((s1_id, cand_id, label))

        # Augment true matches in training set so model learns positive matching signals
        for true_id in true_set:
            if true_id in records_s23 and true_id not in cands:
                train_labeled_pairs.append((s1_id, true_id, 1))

    # 6. Build Validation Holdout Candidate Pairs & Ground Truth
    val_candidate_pairs: List[Tuple[str, str]] = []
    val_candidates_by_s1: Dict[str, Set[str]] = {}
    val_ground_truth: Dict[str, Set[str]] = {}

    for s1_id in val_s1_set:
        # Every validation S1 entity is in val_ground_truth (including singletons!)
        val_ground_truth[s1_id] = ground_truth.get(s1_id, set())

        cands = s1_candidates_map.get(s1_id, set())
        val_candidates_by_s1[s1_id] = set()

        for cand_id in cands:
            if cand_id in records_s23:
                val_candidate_pairs.append((s1_id, cand_id))
                val_candidates_by_s1[s1_id].add(cand_id)

    return (
        train_labeled_pairs,
        val_candidate_pairs,
        val_ground_truth,
        records_s1,
        records_s23,
        val_candidates_by_s1
    )


def train_p3_model(
    config: P3Config,
    max_train_records: int = 25000,
    val_ratio: float = 0.2
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """Train XGBoost matching model with genuine entity-level holdout evaluation."""
    print("=" * 80)
    print("P3 MATCHING MODEL TRAINING & HOLDOUT VALIDATION")
    print("=" * 80)

    (
        train_pairs,
        val_cand_pairs,
        val_ground_truth,
        recs_s1,
        recs_s23,
        val_candidates_by_s1
    ) = extract_training_and_val_data(
        config.train_dir,
        max_s1_records=max_train_records,
        val_ratio=val_ratio,
        chunksize=config.chunksize,
        random_seed=config.random_seed
    )

    if not train_pairs:
        raise ValueError("No training pairs generated.")

    train_labels = [p[2] for p in train_pairs]
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    pos_ratio = (pos_count / len(train_labels)) * 100

    print("\nTraining Dataset Diagnostics:")
    print(f"  Total train pairs:    {len(train_pairs):,}")
    print(f"  Positive matches:     {pos_count:,} ({pos_ratio:.2f}%)")
    print(f"  Negative pairs:       {neg_count:,} ({100 - pos_ratio:.2f}%)")

    # Feature extraction
    print("\nExtracting pairwise features for training...")
    X_train = build_feature_matrix([(p[0], p[1]) for p in train_pairs], recs_s1, recs_s23)
    y_train = np.array(train_labels, dtype=int)

    print("Extracting pairwise features for holdout validation candidate pairs...")
    X_val = build_feature_matrix(val_cand_pairs, recs_s1, recs_s23)
    
    # Ground truth labels for validation candidate pairs (for loss monitoring)
    val_cand_labels = np.array([
        1 if cand_id in val_ground_truth.get(s1_id, set()) else 0
        for s1_id, cand_id in val_cand_pairs
    ], dtype=int)

    print(f"  Feature matrix shapes: X_train = {X_train.shape}, X_val = {X_val.shape}")

    # Calculate scale_pos_weight from training set
    train_pos = max(1, pos_count)
    scale_pos_weight = neg_count / train_pos
    print(f"\nCalculated scale_pos_weight from training split: {scale_pos_weight:.2f}")

    params = config.xgb_params.copy()
    params["scale_pos_weight"] = scale_pos_weight

    print("\nTraining XGBoost Classifier...")
    model = XGBClassifier(**params)
    eval_set = [(X_train, y_train)]
    if len(X_val) > 0:
        eval_set.append((X_val, val_cand_labels))

    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=50
    )

    # Validation Holdout Evaluation
    print("\n" + "=" * 80)
    print("GENUINE HOLDOUT VALIDATION EVALUATION (OFFICIAL MACRO F0.5)")
    print("=" * 80)

    # 1. Measure P2 Retrieval Ceiling on Holdout Split
    total_val_gt_links = sum(len(m) for m in val_ground_truth.values())
    recovered_val_links = sum(
        len(val_ground_truth[s1] & val_candidates_by_s1.get(s1, set()))
        for s1 in val_ground_truth
    )
    val_singletons = sum(1 for m in val_ground_truth.values() if len(m) == 0)
    val_zero_candidates = sum(1 for s1 in val_ground_truth if len(val_candidates_by_s1.get(s1, set())) == 0)

    p2_recall = (recovered_val_links / total_val_gt_links) if total_val_gt_links > 0 else 1.0
    print(f"Holdout S1 Entities:          {len(val_ground_truth):,}")
    print(f"  - True singletons:          {val_singletons:,} ({val_singletons/len(val_ground_truth):.2%})")
    print(f"  - Zero P2 candidates:       {val_zero_candidates:,} ({val_zero_candidates/len(val_ground_truth):.2%})")
    print(f"  - Total true GT links:      {total_val_gt_links:,}")
    print(f"  - P2 recovered GT links:    {recovered_val_links:,}")
    print(f"  - P2 Candidate Recall:      {p2_recall:.4%}")
    print("-" * 80)

    val_probs = model.predict_proba(X_val)[:, 1] if len(X_val) > 0 else np.array([])

    # Group candidate pair predictions
    scored_pairs = list(zip(val_cand_pairs, val_probs))

    from business_entity_resolution.p3_matching.evaluate import compute_entity_macro_f05, compute_f_beta

    print(f"{'Threshold':>10} | {'Macro Precision':>16} | {'Macro Recall':>14} | {'Macro F0.5':>12} | {'Pred Matches':>14}")
    print("-" * 80)

    best_macro_f05 = -1.0
    best_thresh = 0.50

    for thresh in np.arange(0.10, 0.95, 0.05):
        thresh_val = float(round(thresh, 2))
        predictions: Dict[str, Set[str]] = {s1_id: set() for s1_id in val_ground_truth}
        pred_match_count = 0

        for (s1_id, cand_id), prob in scored_pairs:
            if prob >= thresh_val:
                predictions[s1_id].add(cand_id)
                pred_match_count += 1

        macro_p, macro_r, macro_f05 = compute_entity_macro_f05(val_ground_truth, predictions, beta=config.f_beta)

        if macro_f05 > best_macro_f05:
            best_macro_f05 = macro_f05
            best_thresh = thresh_val

        print(f"{thresh_val:>10.2f} | {macro_p:>16.4f} | {macro_r:>14.4f} | {macro_f05:>12.4f} | {pred_match_count:>14,}")

    print("-" * 80)
    print(f"Optimal Entity Macro F0.5 on Holdout: {best_thresh:.2f} (Macro F0.5 = {best_macro_f05:.4f})")
    print("=" * 80)

    # Save model artifact
    config.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.model_file
    model.save_model(str(model_path))
    print(f"\nTrained model saved to: {model_path}")

    # Feature importances
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTop 10 Important Features:")
    for rank, idx in enumerate(sorted_idx[:10], 1):
        print(f"  {rank:2d}. {FEATURE_NAMES[idx]:<30} {importances[idx]:.4f}")

    stats = {
        "train_samples": len(train_pairs),
        "val_entities": len(val_ground_truth),
        "val_candidate_pairs": len(val_cand_pairs),
        "p2_candidate_recall": float(p2_recall),
        "scale_pos_weight": float(scale_pos_weight),
        "best_thresh": float(best_thresh),
        "best_macro_f05": float(best_macro_f05)
    }
    return model, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Train P3 Matching Model")
    parser.add_argument("--max-records", type=int, default=25000, help="Max S1 records for training pairs")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()

    config = P3Config()
    train_p3_model(config, max_train_records=args.max_records, val_ratio=args.val_ratio)
    return 0


if __name__ == "__main__":
    sys.exit(main())


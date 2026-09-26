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


def extract_training_pairs(
    train_dir: Path,
    max_s1_records: int = 25000,
    chunksize: int = 50000
) -> Tuple[List[Tuple[str, str, int]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Generate labeled candidate pairs (s1_id, candidate_id, label) from training data.

    Positive pairs are true matching business entities across sources.
    Negative pairs are hard negatives retrieved via blocking passes that do not match.
    """
    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"

    print(f"Loading training records from {train_dir}...")
    
    # Store preprocessed record metadata
    records_s1: Dict[str, Dict[str, Any]] = {}
    records_s23: Dict[str, Dict[str, Any]] = {}

    # Build blocking indexes from S2 and S3
    indexes = Indexes()
    for src_path in [s2_path, s3_path]:
        print(f"  Indexing {src_path.name}...")
        for chunk in iter_preprocessed_file(src_path, chunksize=chunksize):
            col_core = chunk["name_core"] if "name_core" in chunk else [""] * len(chunk)
            col_phon = chunk["name_phonetic_key"] if "name_phonetic_key" in chunk else [""] * len(chunk)
            for eid, nclean, ncomp, ncore, ntk, nphon, aclean, atk, cclean in zip(
                chunk["entity_id"],
                chunk["name_clean"],
                chunk["name_compact"],
                col_core,
                chunk["name_token_key"],
                col_phon,
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
                )

    print(f"Total S2/S3 indexed records: {len(records_s23):,}")

    # Stream S1 and retrieve candidates
    print(f"Retrieving candidate pairs for S1 training records (up to {max_s1_records:,})...")
    s1_count = 0
    labeled_pairs: List[Tuple[str, str, int]] = []

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

            rec_a = records_s1[s1_id]
            toks_name_a = set(str(rec_a["name_clean"]).split())
            toks_addr_a = set(str(rec_a["address_clean"]).split())

            for cand_id in candidates:
                rec_b = records_s23.get(cand_id)
                if not rec_b:
                    continue

                # Determine label: matching entity criteria
                toks_name_b = set(str(rec_b["name_clean"]).split())
                toks_addr_b = set(str(rec_b["address_clean"]).split())

                name_overlap = len(toks_name_a & toks_name_b) / max(1, len(toks_name_a | toks_name_b))
                addr_overlap = len(toks_addr_a & toks_addr_b) / max(1, len(toks_addr_a | toks_addr_b))
                country_same = (rec_a["country_clean"] == rec_b["country_clean"]) and (rec_a["country_clean"] != "")

                is_match = 0
                # True positive match conditions:
                if country_same:
                    if (rec_a["name_compact"] == rec_b["name_compact"] and rec_a["name_compact"] != "") and (addr_overlap >= 0.3 or rec_a["address_clean"] == rec_b["address_clean"]):
                        is_match = 1
                    elif name_overlap >= 0.6 and addr_overlap >= 0.35:
                        is_match = 1
                    elif rec_a["name_token_key"] == rec_b["name_token_key"] and (addr_overlap >= 0.25 or rec_a["address_token_key"] == rec_b["address_token_key"]):
                        is_match = 1

                labeled_pairs.append((s1_id, cand_id, is_match))

            s1_count += 1
            if s1_count >= max_s1_records:
                break
        if s1_count >= max_s1_records:
            break

    print(f"Generated {len(labeled_pairs):,} labeled candidate pairs from {s1_count:,} S1 records.")
    return labeled_pairs, records_s1, records_s23


def train_p3_model(
    config: P3Config,
    max_train_records: int = 25000,
    val_ratio: float = 0.2
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """Train XGBoost matching model with grouped validation."""
    print("=" * 80)
    print("P3 MATCHING MODEL TRAINING")
    print("=" * 80)

    pairs, recs_s1, recs_s23 = extract_training_pairs(
        config.train_dir,
        max_s1_records=max_train_records,
        chunksize=config.chunksize
    )

    if not pairs:
        raise ValueError("No candidate pairs generated for training.")

    df_pairs = pd.DataFrame(pairs, columns=["s1_id", "cand_id", "label"])
    pos_count = int(df_pairs["label"].sum())
    neg_count = len(df_pairs) - pos_count
    pos_ratio = (pos_count / len(df_pairs)) * 100

    print("\nDataset Diagnostics:")
    print(f"  Total candidate pairs: {len(df_pairs):,}")
    print(f"  Positive matches:     {pos_count:,} ({pos_ratio:.2f}%)")
    print(f"  Negative pairs:       {neg_count:,} ({100 - pos_ratio:.2f}%)")

    # Grouped train/validation split by S1 entity
    print("\nSplitting Train/Validation grouped by S1 entity to prevent leakage...")
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=config.random_seed)
    train_idx, val_idx = next(gss.split(df_pairs, groups=df_pairs["s1_id"]))

    train_df = df_pairs.iloc[train_idx].reset_index(drop=True)
    val_df = df_pairs.iloc[val_idx].reset_index(drop=True)

    print(f"  Train set: {len(train_df):,} pairs across {train_df['s1_id'].nunique():,} S1 entities (Pos: {train_df['label'].sum():,})")
    print(f"  Val set:   {len(val_df):,} pairs across {val_df['s1_id'].nunique():,} S1 entities (Pos: {val_df['label'].sum():,})")

    # Feature extraction
    print("\nExtracting pairwise features...")
    X_train = build_feature_matrix(list(zip(train_df["s1_id"], train_df["cand_id"])), recs_s1, recs_s23)
    y_train = train_df["label"].values.astype(int)

    X_val = build_feature_matrix(list(zip(val_df["s1_id"], val_df["cand_id"])), recs_s1, recs_s23)
    y_val = val_df["label"].values.astype(int)

    print(f"  Feature matrix shape: X_train = {X_train.shape}, X_val = {X_val.shape}")
    print(f"  Features ({len(FEATURE_NAMES)}): {', '.join(FEATURE_NAMES[:6])}...")

    # Calculate scale_pos_weight from training set only
    train_pos = max(1, int(y_train.sum()))
    train_neg = int(len(y_train) - train_pos)
    scale_pos_weight = train_neg / train_pos
    print(f"\nCalculated scale_pos_weight from training split: {scale_pos_weight:.2f}")

    params = config.xgb_params.copy()
    params["scale_pos_weight"] = scale_pos_weight

    print("\nTraining XGBoost Classifier...")
    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50
    )

    # Validation evaluation
    val_probs = model.predict_proba(X_val)[:, 1]
    
    val_df["prob"] = val_probs
    val_df.to_dict("records")

    print("\n" + "=" * 80)
    print("VALIDATION THRESHOLD SWEEP (P3 Probabilities)")
    print("=" * 80)
    print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F0.5':>10} | {'Pred Pos':>10}")
    print("-" * 60)

    best_f05 = -1.0
    best_thresh = 0.5

    for thresh in np.arange(0.10, 0.95, 0.05):
        preds = (val_probs >= thresh).astype(int)
        tp = np.sum((preds == 1) & (y_val == 1))
        fp = np.sum((preds == 1) & (y_val == 0))
        fn = np.sum((preds == 0) & (y_val == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        beta_sq = 0.5 ** 2
        f05 = ((1 + beta_sq) * precision * recall) / (beta_sq * precision + recall) if (precision + recall) > 0 else 0.0

        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh

        print(f"{thresh:>10.2f} | {precision:>10.4f} | {recall:>10.4f} | {f05:>10.4f} | {np.sum(preds):>10,}")

    print("-" * 60)
    print(f"Optimal F0.5 Threshold on Validation: {best_thresh:.2f} (F0.5 = {best_f05:.4f})")

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
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "n_features": len(FEATURE_NAMES),
        "train_pos": int(train_pos),
        "train_neg": int(train_neg),
        "scale_pos_weight": float(scale_pos_weight),
        "best_thresh": float(best_thresh),
        "best_f05": float(best_f05)
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

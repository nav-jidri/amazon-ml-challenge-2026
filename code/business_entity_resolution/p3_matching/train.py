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
from business_entity_resolution.p3_matching.rules import apply_business_rules


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
    max_s1_records: int = 100000,
    max_s23_records: int = 0,
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

    # 2. Stream S1 entities first (up to max_s1_records)
    print(f"Streaming S1 records (up to {max_s1_records:,})...")
    records_s1: Dict[str, Dict[str, Any]] = {}
    s1_ids_ordered: List[str] = []
    s1_rows: List[Dict[str, Any]] = []

    s1_count = 0
    for chunk in iter_preprocessed_file(s1_path, chunksize=chunksize):
        for _, row in chunk.iterrows():
            s1_id = str(row.get("entity_id", "")).strip()
            if not s1_id:
                continue

            row_dict = {
                "entity_id": s1_id,
                "name_clean": row.get("name_clean", ""),
                "name_compact": row.get("name_compact", ""),
                "name_core": row.get("name_core", ""),
                "name_ascii_compact": row.get("name_ascii_compact", ""),
                "name_significant_token_key": row.get("name_significant_token_key", ""),
                "name_first_two_tokens": row.get("name_first_two_tokens", ""),
                "name_token_key": row.get("name_token_key", ""),
                "address_clean": row.get("address_clean", ""),
                "address_token_key": row.get("address_token_key", ""),
                "address_component_key": row.get("address_component_key", ""),
                "address_street_key": row.get("address_street_key", ""),
                "address_house_number": row.get("address_house_number", ""),
                "address_postal_code": row.get("address_postal_code", ""),
                "country_clean": row.get("country_clean", ""),
            }
            records_s1[s1_id] = row_dict
            s1_rows.append(row_dict)
            s1_ids_ordered.append(s1_id)

            s1_count += 1
            if s1_count >= max_s1_records:
                break
        if s1_count >= max_s1_records:
            break

    print(f"Processed {len(s1_ids_ordered):,} S1 entities.")

    # Collect all true match IDs for these S1 records
    target_s23_ids: Set[str] = set()
    for s1_id in s1_ids_ordered:
        target_s23_ids.update(ground_truth.get(s1_id, set()))
    print(f"  Target true ground truth matches to guarantee: {len(target_s23_ids):,}")

    # 3. Build blocking inverted indexes from S2 and S3 (Pass 1: Inverted Index construction)
    indexes = Indexes()
    total_indexed = 0

    for src_path in [s2_path, s3_path]:
        print(f"  Indexing {src_path.name} for candidate retrieval...", flush=True)
        for chunk in iter_preprocessed_file(src_path, chunksize=chunksize):
            col_core = chunk["name_core"] if "name_core" in chunk else [""] * len(chunk)
            col_phon = chunk["name_phonetic_key"] if "name_phonetic_key" in chunk else [""] * len(chunk)
            col_nascii = chunk["name_ascii_compact"] if "name_ascii_compact" in chunk else [""] * len(chunk)
            col_nsig = chunk["name_significant_token_key"] if "name_significant_token_key" in chunk else [""] * len(chunk)
            col_ntwo = chunk["name_first_two_tokens"] if "name_first_two_tokens" in chunk else [""] * len(chunk)
            col_acomp = chunk["address_component_key"] if "address_component_key" in chunk else [""] * len(chunk)
            col_astreet = chunk["address_street_key"] if "address_street_key" in chunk else [""] * len(chunk)
            col_ahouse = chunk["address_house_number"] if "address_house_number" in chunk else [""] * len(chunk)
            col_ahouse_geo = chunk["address_house_geo_key"] if "address_house_geo_key" in chunk else [""] * len(chunk)
            col_apost = chunk["address_house_postcode_key"] if "address_house_postcode_key" in chunk else [""] * len(chunk)
            col_post = chunk["address_postal_code"] if "address_postal_code" in chunk else [""] * len(chunk)

            for eid, nclean, ncomp, ncore, ntk, nphon, nascii, nsig, ntwo, acomp, astreet, ahouse, ahouse_geo, apost, post, aclean, atk, cclean in zip(
                chunk["entity_id"],
                chunk["name_clean"],
                chunk["name_compact"],
                col_core,
                chunk["name_token_key"],
                col_phon,
                col_nascii,
                col_nsig,
                col_ntwo,
                col_acomp,
                col_astreet,
                col_ahouse,
                col_ahouse_geo,
                col_apost,
                col_post,
                chunk["address_clean"],
                chunk["address_token_key"],
                chunk["country_clean"]
            ):
                eid_str = str(eid).strip()
                if not eid_str:
                    continue

                indexes.add_record(
                    entity_id=eid_str,
                    name_token_key=str(ntk).strip() if ntk else "",
                    name_compact=str(ncomp).strip() if ncomp else "",
                    name_core=str(ncore).strip() if ncore else "",
                    address_token_key=str(atk).strip() if atk else "",
                    name_phonetic_key=str(nphon).strip() if nphon else "",
                    name_ascii_compact=str(nascii).strip() if nascii else "",
                    name_significant_token_key=str(nsig).strip() if nsig else "",
                    name_first_two_tokens=str(ntwo).strip() if ntwo else "",
                    address_component_key=str(acomp).strip() if acomp else "",
                    address_street_key=str(astreet).strip() if astreet else "",
                    address_house_geo_key=str(ahouse_geo).strip() if ahouse_geo else "",
                    address_house_postcode_key=str(apost).strip() if apost else "",
                )
                total_indexed += 1

            if max_s23_records > 0 and total_indexed >= max_s23_records:
                break

    print(f"Total S2/S3 indexed records in blocking tables: {total_indexed:,}", flush=True)

    # 4. Stream S1 and collect candidate sets
    print(f"Retrieving candidate pairs for {len(s1_rows):,} S1 entities...", flush=True)
    s1_candidates_map: Dict[str, Set[str]] = {}
    needed_s23_ids: Set[str] = set(target_s23_ids)

    for row_dict in s1_rows:
        s1_id = row_dict["entity_id"]
        candidates = indexes.retrieve_candidates_for_row(row_dict)
        candidates.discard(s1_id)
        s1_candidates_map[s1_id] = candidates
        needed_s23_ids.update(candidates)

    print(f"Target candidate entities to load representation for: {len(needed_s23_ids):,} (out of {total_indexed:,})", flush=True)

    # 4b. Load representation dictionaries ONLY for required candidates (Pass 2)
    records_s23: Dict[str, Dict[str, Any]] = {}
    for src_path in [s2_path, s3_path]:
        print(f"  Streaming {src_path.name} representations...", flush=True)
        for chunk in iter_preprocessed_file(src_path, chunksize=chunksize):
            col_core = chunk["name_core"] if "name_core" in chunk else [""] * len(chunk)
            col_nascii = chunk["name_ascii_compact"] if "name_ascii_compact" in chunk else [""] * len(chunk)
            col_nsig = chunk["name_significant_token_key"] if "name_significant_token_key" in chunk else [""] * len(chunk)
            col_ntwo = chunk["name_first_two_tokens"] if "name_first_two_tokens" in chunk else [""] * len(chunk)
            col_acomp = chunk["address_component_key"] if "address_component_key" in chunk else [""] * len(chunk)
            col_astreet = chunk["address_street_key"] if "address_street_key" in chunk else [""] * len(chunk)
            col_ahouse = chunk["address_house_number"] if "address_house_number" in chunk else [""] * len(chunk)
            col_post = chunk["address_postal_code"] if "address_postal_code" in chunk else [""] * len(chunk)

            for eid, nclean, ncomp, ncore, nascii, nsig, ntwo, ntk, aclean, atk, acomp, astreet, ahouse, post, cclean in zip(
                chunk["entity_id"],
                chunk["name_clean"],
                chunk["name_compact"],
                col_core,
                col_nascii,
                col_nsig,
                col_ntwo,
                chunk["name_token_key"],
                chunk["address_clean"],
                chunk["address_token_key"],
                col_acomp,
                col_astreet,
                col_ahouse,
                col_post,
                chunk["country_clean"]
            ):
                eid_str = str(eid).strip()
                if eid_str in needed_s23_ids:
                    records_s23[eid_str] = {
                        "name_clean": nclean,
                        "name_compact": ncomp,
                        "name_core": ncore,
                        "name_ascii_compact": nascii,
                        "name_significant_token_key": nsig,
                        "name_first_two_tokens": ntwo,
                        "name_token_key": ntk,
                        "address_clean": aclean,
                        "address_token_key": atk,
                        "address_component_key": acomp,
                        "address_street_key": astreet,
                        "address_house_number": ahouse,
                        "address_postal_code": post,
                        "country_clean": cclean,
                    }
        if len(records_s23) >= len(needed_s23_ids):
            break

    print(f"Loaded {len(records_s23):,} S2/S3 candidate representations into memory (Memory footprint: ~{len(records_s23)*1.5/1024:.1f} MB)", flush=True)

    # 5. Split S1 entities into Train and Validation holdout sets
    rng = np.random.RandomState(random_seed)
    shuffled_s1 = s1_ids_ordered.copy()
    rng.shuffle(shuffled_s1)

    n_val = int(len(shuffled_s1) * val_ratio)
    val_s1_set = set(shuffled_s1[:n_val])
    train_s1_set = set(shuffled_s1[n_val:])

    print(f"  Train S1 entities: {len(train_s1_set):,}", flush=True)
    print(f"  Holdout Val S1 entities: {len(val_s1_set):,}", flush=True)

    # 6. Build Training Pairs (with Structural Hard-Negative Mining)
    train_labeled_pairs: List[Tuple[str, str, int]] = []
    max_negs_per_s1 = 25  # Optimal ratio (~15-20% positives) prioritizing hard collisions

    for s1_id in train_s1_set:
        cands = s1_candidates_map.get(s1_id, set())
        true_set = ground_truth.get(s1_id, set())
        s1_rec = records_s1.get(s1_id, {})
        s1_ncore = s1_rec.get("name_core", "")
        s1_astreet = s1_rec.get("address_street_key", "")
        s1_apost = s1_rec.get("address_postal_code", "")

        # 1. Positive matches (always include all GT matches)
        for true_id in true_set:
            if true_id in records_s23:
                train_labeled_pairs.append((s1_id, true_id, 1))

        # 2. Hard-negative prioritization (prioritize near-misses on street, postal code, or brand root)
        neg_cands = [cid for cid in cands if cid not in true_set and cid in records_s23]
        if len(neg_cands) > max_negs_per_s1:
            hard_negs = []
            regular_negs = []
            for cid in neg_cands:
                cand_rec = records_s23.get(cid, {})
                is_hard = False
                if s1_astreet and cand_rec.get("address_street_key", "") == s1_astreet:
                    is_hard = True
                elif s1_ncore and cand_rec.get("name_core", "") == s1_ncore:
                    is_hard = True
                elif s1_apost and cand_rec.get("address_postal_code", "") == s1_apost:
                    is_hard = True

                if is_hard:
                    hard_negs.append(cid)
                else:
                    regular_negs.append(cid)

            if len(hard_negs) >= max_negs_per_s1:
                sampled_negs = rng.choice(hard_negs, size=max_negs_per_s1, replace=False).tolist()
            else:
                needed_rem = max_negs_per_s1 - len(hard_negs)
                if len(regular_negs) > needed_rem:
                    sampled_regular = rng.choice(regular_negs, size=needed_rem, replace=False).tolist()
                else:
                    sampled_regular = regular_negs
                sampled_negs = hard_negs + sampled_regular
        else:
            sampled_negs = neg_cands

        for cand_id in sampled_negs:
            train_labeled_pairs.append((s1_id, cand_id, 0))

    # 7. Build Validation Holdout Candidate Pairs & Ground Truth (No negative subsampling in validation)
    val_candidate_pairs: List[Tuple[str, str]] = []
    val_candidates_by_s1: Dict[str, Set[str]] = {}
    val_ground_truth: Dict[str, Set[str]] = {}

    for s1_id in val_s1_set:
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
    max_train_records: int = 100000,
    max_s23_records: int = 0,
    val_ratio: float = 0.2,
    scale_pos_weight: Optional[float] = None,
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
        max_s23_records=max_s23_records,
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

    print("\nTraining Dataset Diagnostics:", flush=True)
    print(f"  Total train pairs:    {len(train_pairs):,}", flush=True)
    print(f"  Positive matches:     {pos_count:,} ({pos_ratio:.2f}%)", flush=True)
    print(f"  Negative pairs:       {neg_count:,} ({100 - pos_ratio:.2f}%)", flush=True)

    # Feature extraction
    print("\nExtracting pairwise features for training...", flush=True)
    X_train = build_feature_matrix([(p[0], p[1]) for p in train_pairs], recs_s1, recs_s23)
    y_train = np.array(train_labels, dtype=int)

    print("Extracting pairwise features for holdout validation candidate pairs...", flush=True)
    X_val = build_feature_matrix(val_cand_pairs, recs_s1, recs_s23)
    
    # Ground truth labels for validation candidate pairs (for loss monitoring)
    val_cand_labels = np.array([
        1 if cand_id in val_ground_truth.get(s1_id, set()) else 0
        for s1_id, cand_id in val_cand_pairs
    ], dtype=int)

    print(f"  Feature matrix shapes: X_train = {X_train.shape}, X_val = {X_val.shape}", flush=True)

    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    if scale_pos_weight is None or scale_pos_weight <= 0:
        effective_spw = float(neg_count) / float(pos_count) if pos_count > 0 else 1.0
        print(f"\nAuto-computed scale_pos_weight from training balance: {effective_spw:.2f} ({neg_count:,} neg / {pos_count:,} pos)", flush=True)
    else:
        effective_spw = scale_pos_weight
        print(f"\nUsing manual scale_pos_weight: {effective_spw:.2f}", flush=True)

    params = config.xgb_params.copy()
    params["scale_pos_weight"] = effective_spw
    if len(X_val) > 0:
        params["early_stopping_rounds"] = 25

    print("\nTraining XGBoost Classifier...", flush=True)
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
    print("\n" + "=" * 80, flush=True)
    print("GENUINE HOLDOUT VALIDATION EVALUATION (OFFICIAL MACRO F0.5)", flush=True)
    print("=" * 80, flush=True)

    # 1. Measure P2 Retrieval Ceiling on Holdout Split
    total_val_gt_links = sum(len(m) for m in val_ground_truth.values())
    recovered_val_links = sum(
        len(val_ground_truth[s1] & val_candidates_by_s1.get(s1, set()))
        for s1 in val_ground_truth
    )
    val_singletons = sum(1 for m in val_ground_truth.values() if len(m) == 0)
    val_zero_candidates = sum(1 for s1 in val_ground_truth if len(val_candidates_by_s1.get(s1, set())) == 0)

    p2_recall = (recovered_val_links / total_val_gt_links) if total_val_gt_links > 0 else 1.0
    print(f"Holdout S1 Entities:          {len(val_ground_truth):,}", flush=True)
    print(f"  - True singletons:          {val_singletons:,} ({val_singletons/len(val_ground_truth):.2%})", flush=True)
    print(f"  - Zero P2 candidates:       {val_zero_candidates:,} ({val_zero_candidates/len(val_ground_truth):.2%})", flush=True)
    print(f"  - Total true GT links:      {total_val_gt_links:,}", flush=True)
    print(f"  - P2 recovered GT links:    {recovered_val_links:,}", flush=True)
    print(f"  - P2 Candidate Recall:      {p2_recall:.4%}", flush=True)
    print("-" * 80, flush=True)

    val_probs = model.predict_proba(X_val)[:, 1] if len(X_val) > 0 else np.array([])
    if len(val_probs) > 0:
        med_prob = np.median(val_probs)
        mean_prob = np.mean(val_probs)
        print(f"Predicted Probability Distribution on Val Candidates:", flush=True)
        print(f"  - Mean:   {mean_prob:.4f}", flush=True)
        print(f"  - Median: {med_prob:.4f}", flush=True)
        print(f"  - Min:    {np.min(val_probs):.4f}, Max: {np.max(val_probs):.4f}", flush=True)
        print("-" * 80, flush=True)

    # Group candidate pair predictions
    scored_pairs = list(zip(val_cand_pairs, val_probs))

    from business_entity_resolution.p3_matching.evaluate import compute_entity_macro_f05, compute_f_beta

    print(f"{'Threshold':>10} | {'Macro Precision':>16} | {'Macro Recall':>14} | {'Macro F0.5':>12} | {'True Pos':>10} | {'False Pos':>10}", flush=True)
    print("-" * 90, flush=True)

    best_macro_f05 = -1.0
    best_thresh = 0.50

    for thresh in np.arange(0.10, 0.98, 0.05):
        thresh_val = float(round(thresh, 2))
        predictions: Dict[str, Set[str]] = {s1_id: set() for s1_id in val_ground_truth}
        pred_match_count = 0
        tp_count = 0
        fp_count = 0

        for (s1_id, cand_id), raw_prob in scored_pairs:
            prob = apply_business_rules(float(raw_prob), recs_s1.get(s1_id, {}), recs_s23.get(cand_id, {}))
            if prob >= thresh_val:
                predictions[s1_id].add(cand_id)
                pred_match_count += 1
                if cand_id in val_ground_truth.get(s1_id, set()):
                    tp_count += 1
                else:
                    fp_count += 1

        macro_p, macro_r, macro_f05 = compute_entity_macro_f05(val_ground_truth, predictions, beta=config.f_beta)

        if macro_f05 > best_macro_f05:
            best_macro_f05 = macro_f05
            best_thresh = thresh_val

        print(f"{thresh_val:>10.2f} | {macro_p:>16.4f} | {macro_r:>14.4f} | {macro_f05:>12.4f} | {tp_count:>10,} | {fp_count:>10,}", flush=True)

    print("-" * 90, flush=True)
    print(f"Optimal Entity Macro F0.5 on Holdout: {best_thresh:.2f} (Macro F0.5 = {best_macro_f05:.4f})", flush=True)
    print("=" * 80, flush=True)

    # Save model artifact
    config.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.model_file
    model.save_model(str(model_path))
    print(f"\nTrained model saved to: {model_path}", flush=True)

    # Save validation artifacts for P4 evaluation & threshold tuning
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    val_s1_list = sorted(list(val_ground_truth.keys()))
    val_s1_path = output_dir / "val_s1_ids.json"
    with open(val_s1_path, "w", encoding="utf-8") as f:
        json.dump(val_s1_list, f, indent=2)
    print(f"Saved {len(val_s1_list):,} validation S1 IDs to: {val_s1_path}", flush=True)

    val_gt_serializable = {k: sorted(list(v)) for k, v in val_ground_truth.items()}
    val_gt_path = output_dir / "val_ground_truth.json"
    with open(val_gt_path, "w", encoding="utf-8") as f:
        json.dump(val_gt_serializable, f, indent=2)
    print(f"Saved validation ground truth to: {val_gt_path}", flush=True)

    val_cands_path = output_dir / "val_candidate_pairs.tsv"
    with open(val_cands_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in val_s1_list:
            cands = sorted(list(val_candidates_by_s1.get(s1_id, set())))
            f.write(f"{s1_id}\t{','.join(cands)}\n")
    print(f"Saved validation candidate pairs to: {val_cands_path}", flush=True)

    val_scored_path = output_dir / "val_candidates_scored.tsv"
    with open(val_scored_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_id\tmatch_probability\n")
        for (s1_id, cand_id), prob in scored_pairs:
            final_prob = apply_business_rules(float(prob), recs_s1.get(s1_id, {}), recs_s23.get(cand_id, {}))
            f.write(f"{s1_id}\t{cand_id}\t{final_prob:.6f}\n")
    print(f"Saved {len(scored_pairs):,} scored validation candidate pairs to: {val_scored_path}", flush=True)

    # Feature importances
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTop 15 Important Features:", flush=True)
    for rank, idx in enumerate(sorted_idx[:15], 1):
        print(f"  {rank:2d}. {FEATURE_NAMES[idx]:<30} {importances[idx]:.4f}", flush=True)

    stats = {
        "train_samples": len(train_pairs),
        "val_entities": len(val_ground_truth),
        "val_candidate_pairs": len(val_cand_pairs),
        "p2_candidate_recall": float(p2_recall),
        "scale_pos_weight": float(effective_spw),
        "best_thresh": float(best_thresh),
        "best_macro_f05": float(best_macro_f05)
    }
    return model, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Train P3 Matching Model")
    parser.add_argument("--max-records", type=int, default=100000, help="Max S1 records for training pairs (default: 100,000)")
    parser.add_argument("--max-s23-records", type=int, default=0, help="Max S2/S3 records to index (0 for all)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--scale-pos-weight", type=float, default=None, help="XGBoost scale_pos_weight (default: auto-computed from split balance)")
    args = parser.parse_args()

    config = P3Config()
    train_p3_model(
        config,
        max_train_records=args.max_records,
        max_s23_records=args.max_s23_records,
        val_ratio=args.val_ratio,
        scale_pos_weight=args.scale_pos_weight
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


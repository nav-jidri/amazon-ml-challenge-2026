# -*- coding: utf-8 -*-
"""export_val_for_p4.py

Exports genuine holdout validation S1 IDs, ground truth mappings, and scored candidate pairs
so the P4 team can run end-to-end evaluation and threshold optimization.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from xgboost import XGBClassifier

from business_entity_resolution.p3_matching.config import P3Config
from business_entity_resolution.p3_matching.train import extract_training_and_val_data
from business_entity_resolution.p3_matching.pair_features import build_feature_matrix
from business_entity_resolution.p3_matching.rules import apply_business_rules


def export_validation_artifacts(max_records: int = 50000, val_ratio: float = 0.2):
    config = P3Config()
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting validation data split (seed=42, 50k S1, 0.2 val)...", flush=True)
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
        val_ratio=val_ratio,
        chunksize=config.chunksize,
        random_seed=config.random_seed
    )

    # Save val_s1_ids.json
    val_s1_list = sorted(list(val_ground_truth.keys()))
    val_s1_path = output_dir / "val_s1_ids.json"
    with open(val_s1_path, "w", encoding="utf-8") as f:
        json.dump(val_s1_list, f, indent=2)
    print(f"Exported {len(val_s1_list):,} validation S1 IDs to: {val_s1_path}", flush=True)

    # Save val_ground_truth.json
    val_gt_serializable = {k: sorted(list(v)) for k, v in val_ground_truth.items()}
    val_gt_path = output_dir / "val_ground_truth.json"
    with open(val_gt_path, "w", encoding="utf-8") as f:
        json.dump(val_gt_serializable, f, indent=2)
    print(f"Exported validation ground truth to: {val_gt_path}", flush=True)

    # Save val_candidate_pairs.tsv (unscored)
    val_cands_path = output_dir / "val_candidate_pairs.tsv"
    with open(val_cands_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in val_s1_list:
            cands = sorted(list(val_candidates_by_s1.get(s1_id, set())))
            f.write(f"{s1_id}\t{','.join(cands)}\n")
    print(f"Exported validation candidate pairs to: {val_cands_path}", flush=True)

    # Load model and score val candidate pairs
    if config.model_file.exists():
        print(f"Scoring {len(val_cand_pairs):,} validation pairs with trained model...", flush=True)
        model = XGBClassifier()
        model.load_model(str(config.model_file))
        X_val = build_feature_matrix(val_cand_pairs, recs_s1, recs_s23)
        val_probs = model.predict_proba(X_val)[:, 1]

        val_scored_path = output_dir / "val_candidates_scored.tsv"
        with open(val_scored_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_id\tmatch_probability\n")
            for (s1_id, cand_id), prob in zip(val_cand_pairs, val_probs):
                final_prob = apply_business_rules(float(prob), recs_s1.get(s1_id, {}), recs_s23.get(cand_id, {}))
                f.write(f"{s1_id}\t{cand_id}\t{final_prob:.6f}\n")
        print(f"Exported scored validation pairs to: {val_scored_path}", flush=True)


if __name__ == "__main__":
    export_validation_artifacts()

# -*- coding: utf-8 -*-
"""predict.py

Inference script for P3 Matching Model.
Loads trained XGBoost model, reads P2 candidate_pairs.tsv and normalized records,
computes pairwise matching probabilities, and outputs candidate_pairs_scored.tsv.
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# Add code directory to sys.path
code_dir = str(Path(__file__).resolve().parents[2])
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from business_entity_resolution.preprocessing import iter_preprocessed_file
from business_entity_resolution.p3_matching.config import P3Config
from business_entity_resolution.p3_matching.pair_features import build_feature_matrix



def load_normalized_records(
    source1_path: Path,
    source2_path: Path,
    source3_path: Path,
    chunksize: int = 100_000
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Load normalized representation dictionaries for S1 and S2/S3."""
    print("Loading normalized representations for inference...")
    records_s1: Dict[str, Dict[str, Any]] = {}
    records_s23: Dict[str, Dict[str, Any]] = {}

    # Load S2 and S3 records
    for path in [source2_path, source3_path]:
        print(f"  Streaming {path.name}...")
        for chunk in iter_preprocessed_file(path, chunksize=chunksize):
            col_core = chunk["name_core"] if "name_core" in chunk else [""] * len(chunk)
            for eid, nclean, ncomp, ncore, ntk, aclean, atk, cclean in zip(
                chunk["entity_id"],
                chunk["name_clean"],
                chunk["name_compact"],
                col_core,
                chunk["name_token_key"],
                chunk["address_clean"],
                chunk["address_token_key"],
                chunk["country_clean"]
            ):
                eid_str = str(eid).strip()
                if eid_str:
                    records_s23[eid_str] = {
                        "name_clean": nclean,
                        "name_compact": ncomp,
                        "name_core": ncore,
                        "name_token_key": ntk,
                        "address_clean": aclean,
                        "address_token_key": atk,
                        "country_clean": cclean,
                    }

    print(f"  Loaded {len(records_s23):,} S2/S3 records.")

    # Load S1 records
    print(f"  Streaming {source1_path.name}...")
    for chunk in iter_preprocessed_file(source1_path, chunksize=chunksize):
        col_core = chunk["name_core"] if "name_core" in chunk else [""] * len(chunk)
        for eid, nclean, ncomp, ncore, ntk, aclean, atk, cclean in zip(
            chunk["entity_id"],
            chunk["name_clean"],
            chunk["name_compact"],
            col_core,
            chunk["name_token_key"],
            chunk["address_clean"],
            chunk["address_token_key"],
            chunk["country_clean"]
        ):
            eid_str = str(eid).strip()
            if eid_str:
                records_s1[eid_str] = {
                    "name_clean": nclean,
                    "name_compact": ncomp,
                    "name_core": ncore,
                    "name_token_key": ntk,
                    "address_clean": aclean,
                    "address_token_key": atk,
                    "country_clean": cclean,
                }

    print(f"  Loaded {len(records_s1):,} S1 records.")
    return records_s1, records_s23


def score_candidate_pairs(
    candidate_pairs_path: Path,
    output_path: Path,
    model_path: Path,
    records_s1: Dict[str, Dict[str, Any]],
    records_s23: Dict[str, Dict[str, Any]],
    batch_size: int = 50_000
) -> int:
    """Stream candidate pairs, generate pairwise features, score probabilities with XGBoost, and write TSV."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}. Run train.py first.")

    print(f"\nLoading trained model from {model_path}...")
    model = XGBClassifier()
    model.load_model(str(model_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Scoring candidate pairs from {candidate_pairs_path} -> {output_path}...")

    total_pairs_scored = 0
    buffer_pairs: List[Tuple[str, str]] = []

    with open(candidate_pairs_path, "r", encoding="utf-8") as in_fp, \
         open(output_path, "w", encoding="utf-8", newline="\n") as out_fp:

        # Header
        out_fp.write("source1_entity_id\tcandidate_entity_id\tmatch_probability\n")

        # Skip header of candidate_pairs.tsv
        header = in_fp.readline()

        for line in in_fp:
            parts = line.strip().split("\t")
            if not parts or len(parts) < 1:
                continue
            s1_id = parts[0].strip()
            if not s1_id:
                continue

            cands_str = parts[1].strip() if len(parts) > 1 else ""
            if not cands_str:
                continue

            for cand_id in cands_str.split(","):
                cand_id = cand_id.strip()
                if cand_id:
                    buffer_pairs.append((s1_id, cand_id))

            if len(buffer_pairs) >= batch_size:
                X_batch = build_feature_matrix(buffer_pairs, records_s1, records_s23)
                probs = model.predict_proba(X_batch)[:, 1]

                for (s1, cand), prob in zip(buffer_pairs, probs):
                    out_fp.write(f"{s1}\t{cand}\t{prob:.6f}\n")

                total_pairs_scored += len(buffer_pairs)
                buffer_pairs.clear()

        # Flush remaining buffer
        if buffer_pairs:
            X_batch = build_feature_matrix(buffer_pairs, records_s1, records_s23)
            probs = model.predict_proba(X_batch)[:, 1]
            for (s1, cand), prob in zip(buffer_pairs, probs):
                out_fp.write(f"{s1}\t{cand}\t{prob:.6f}\n")
            total_pairs_scored += len(buffer_pairs)
            buffer_pairs.clear()

    print(f"Scoring completed. Total pairs scored: {total_pairs_scored:,}")
    print(f"Output saved to: {output_path}")
    return total_pairs_scored


def main() -> int:
    parser = argparse.ArgumentParser(description="Score candidate pairs with P3 XGBoost model")
    parser.add_argument("--candidate-pairs", type=Path, default=None, help="Path to candidate_pairs.tsv")
    parser.add_argument("--output", type=Path, default=None, help="Path for scored output TSV")
    parser.add_argument("--source1", type=Path, default=None, help="Path to source1 TSV")
    parser.add_argument("--source2", type=Path, default=None, help="Path to source2 TSV")
    parser.add_argument("--source3", type=Path, default=None, help="Path to source3 TSV")
    parser.add_argument("--model-file", type=Path, default=None, help="Path to trained XGBoost model")
    args = parser.parse_args()

    config = P3Config()
    cand_path = args.candidate_pairs or config.candidate_pairs_path
    out_path = args.output or config.scored_pairs_path
    model_path = args.model_file or config.model_file

    s1_path = args.source1 or (config.test_dir / "test_source1.tsv")
    s2_path = args.source2 or (config.test_dir / "test_source2.tsv")
    s3_path = args.source3 or (config.test_dir / "test_source3.tsv")

    recs_s1, recs_s23 = load_normalized_records(s1_path, s2_path, s3_path, chunksize=config.chunksize)
    score_candidate_pairs(cand_path, out_path, model_path, recs_s1, recs_s23)
    return 0


if __name__ == "__main__":
    sys.exit(main())

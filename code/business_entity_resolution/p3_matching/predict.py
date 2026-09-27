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
from business_entity_resolution.p3_matching.rules import apply_business_rules



def load_normalized_records(
    source1_path: Path,
    source2_path: Path,
    source3_path: Path,
    candidate_pairs_path: Optional[Path] = None,
    chunksize: int = 100_000
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Load normalized representation dictionaries for S1 and S2/S3."""
    needed_s1: Optional[Set[str]] = None
    needed_s23: Optional[Set[str]] = None

    if candidate_pairs_path and candidate_pairs_path.exists():
        print(f"Scanning {candidate_pairs_path.name} to identify required entity representations...")
        needed_s1 = set()
        needed_s23 = set()
        with open(candidate_pairs_path, "r", encoding="utf-8") as f:
            next(f, None)  # header
            for line in f:
                parts = line.strip().split("\t")
                if not parts:
                    continue
                s1_id = parts[0].strip()
                if s1_id:
                    needed_s1.add(s1_id)
                if len(parts) > 1 and parts[1].strip():
                    for cand in parts[1].split(","):
                        cand = cand.strip()
                        if cand:
                            needed_s23.add(cand)
        print(f"  Target required entities: {len(needed_s1):,} S1, {len(needed_s23):,} S2/S3 candidates.")

    print("Loading normalized representations for inference...")
    records_s1: Dict[str, Dict[str, Any]] = {}
    records_s23: Dict[str, Dict[str, Any]] = {}

    # Load S2 and S3 records
    for path in [source2_path, source3_path]:
        print(f"  Streaming {path.name}...")
        for chunk in iter_preprocessed_file(path, chunksize=chunksize):
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
                if not eid_str:
                    continue
                if needed_s23 is not None and eid_str not in needed_s23:
                    continue
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

    print(f"  Loaded {len(records_s23):,} S2/S3 records.")

    # Load S1 records
    print(f"  Streaming {source1_path.name}...")
    for chunk in iter_preprocessed_file(source1_path, chunksize=chunksize):
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
            if not eid_str:
                continue
            if needed_s1 is not None and eid_str not in needed_s1:
                continue
            records_s1[eid_str] = {
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

    print(f"  Loaded {len(records_s1):,} S1 records.")
    return records_s1, records_s23


def score_candidate_pairs(
    candidate_pairs_path: Path,
    output_path: Path,
    model_path: Path,
    records_s1: Dict[str, Dict[str, Any]],
    records_s23: Dict[str, Dict[str, Any]],
    batch_size: int = 200_000
) -> int:
    """Stream candidate pairs, generate pairwise features, score probabilities with XGBoost, and write TSV."""
    import time
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}. Run train.py first.")

    print(f"\nLoading trained model from {model_path}...", flush=True)
    model = XGBClassifier()
    model.load_model(str(model_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Scoring candidate pairs from {candidate_pairs_path.name} -> {output_path.name} (Batch size: {batch_size:,})...", flush=True)

    total_pairs_scored = 0
    buffer_pairs: List[Tuple[str, str]] = []
    start_time = time.time()
    last_log_time = start_time

    # 16MB write buffer for high-throughput disk writing
    with open(candidate_pairs_path, "r", encoding="utf-8") as in_fp, \
         open(output_path, "w", encoding="utf-8", newline="\n", buffering=16 * 1024 * 1024) as out_fp:

        # Header
        out_fp.write("source1_entity_id\tcandidate_entity_id\tmatch_probability\n")

        # Skip header of candidate_pairs.tsv
        _ = in_fp.readline()

        def _process_batch(pairs: List[Tuple[str, str]]) -> None:
            nonlocal total_pairs_scored, last_log_time
            if not pairs:
                return
            X_batch = build_feature_matrix(pairs, records_s1, records_s23)
            probs = model.predict_proba(X_batch)[:, 1]

            # Vectorized high-precision country strict mismatch zeroing (feature index 27)
            if X_batch.shape[1] > 27:
                probs[X_batch[:, 27] == 1.0] = 0.0

            # Fast chunk string join and buffered write
            chunk_str = "".join(f"{s}\t{c}\t{p:.4f}\n" for (s, c), p in zip(pairs, probs))
            out_fp.write(chunk_str)

            total_pairs_scored += len(pairs)
            now = time.time()
            if now - last_log_time >= 5.0 or total_pairs_scored % 1_000_000 == 0:
                elapsed = now - start_time
                rate = total_pairs_scored / elapsed if elapsed > 0 else 0
                print(f"  Scored {total_pairs_scored:11,d} pairs | {rate:6,.0f} pairs/sec | Elapsed: {elapsed/60:4.1f} min", flush=True)
                last_log_time = now

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
                _process_batch(buffer_pairs)
                buffer_pairs.clear()

        # Flush remaining buffer
        if buffer_pairs:
            _process_batch(buffer_pairs)
            buffer_pairs.clear()

    total_time = time.time() - start_time
    avg_rate = total_pairs_scored / total_time if total_time > 0 else 0
    print(f"\nScoring completed in {total_time/60:.2f} min! Total pairs: {total_pairs_scored:,} (Avg: {avg_rate:,.0f} pairs/sec)", flush=True)
    print(f"Output saved to: {output_path}", flush=True)
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

    recs_s1, recs_s23 = load_normalized_records(
        s1_path, s2_path, s3_path, candidate_pairs_path=cand_path, chunksize=config.chunksize
    )
    score_candidate_pairs(cand_path, out_path, model_path, recs_s1, recs_s23)
    return 0


if __name__ == "__main__":
    sys.exit(main())

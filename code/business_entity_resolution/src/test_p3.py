# -*- coding: utf-8 -*-
"""test_p3.py

Automated tests for P3 Matching Model training, feature extraction,
and genuine entity-level holdout validation.
"""

from __future__ import annotations
import sys
import shutil
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

code_dir = str(Path(__file__).resolve().parents[2])
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from business_entity_resolution.p3_matching.config import P3Config
from business_entity_resolution.p3_matching.evaluate import compute_entity_macro_f05, compute_f_beta
from business_entity_resolution.p3_matching.pair_features import FEATURE_NAMES, build_feature_matrix
from business_entity_resolution.p3_matching.train import (
    load_ground_truth,
    extract_training_and_val_data,
    train_p3_model,
)


@pytest.fixture
def synthetic_data_dir(tmp_path: Path) -> Path:
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True)

    # train_source1.tsv: 10 entities
    # S1-1 to S1-6: have true matches and shared tokens with non-matches (hard negatives)
    # S1-7 to S1-10: singletons (no matches)
    s1_rows = [
        ("entity_id", "business_name", "business_address", "country"),
        ("S1-1", "Apple Inc", "1 Infinite Loop Cupertino CA", "US"),
        ("S1-2", "Google LLC", "1600 Amphitheatre Pkwy Mountain View", "US"),
        ("S1-3", "Microsoft Corp", "One Microsoft Way Redmond", "US"),
        ("S1-4", "Amazon Com", "410 Terry Ave N Seattle", "US"),
        ("S1-5", "Meta Platforms", "1 Hacker Way Menlo Park", "US"),
        ("S1-6", "Tesla Inc", "3500 Deer Creek Rd Palo Alto", "US"),
        ("S1-7", "Solo Cafe", "123 Main St Austin TX", "US"),
        ("S1-8", "Unique Bakery", "456 Oak St Denver CO", "US"),
        ("S1-9", "Rare Books", "789 Pine St Boston MA", "US"),
        ("S1-10", "Lone Star Plumbing", "101 Elm St Dallas TX", "US"),
    ]
    with open(train_dir / "train_source1.tsv", "w", encoding="utf-8", newline="\n") as f:
        for row in s1_rows:
            f.write("\t".join(row) + "\n")

    # train_source2.tsv
    s2_rows = [
        ("entity_id", "business_name", "business_address", "country"),
        ("S2-1", "Apple Incorporated", "1 Infinite Loop Cupertino CA", "United States"),
        ("S2-2", "Google", "1600 Amphitheatre Parkway Mountain View", "USA"),
        ("S2-3", "Microsoft Corporation", "1 Microsoft Way Redmond WA", "US"),
        ("S2-10", "Apple Holdings LLC", "999 Random Lane Cupertino CA", "US"),  # Matches name_core 'apple' -> Negative for S1-1
        ("S2-20", "Google Ventures", "1600 Amphitheatre Pkwy Mountain View", "US"),  # Matches address -> Negative for S1-2
        ("S2-30", "Microsoft Retail Store LLC", "1 Microsoft Way Redmond WA", "US"),  # Matches address -> Negative for S1-3
        ("S2-50", "Meta Reality Labs LLC", "1 Hacker Way Menlo Park", "US"),  # Negative for S1-5
        ("S2-60", "Tesla Energy Corp", "3500 Deer Creek Rd Palo Alto", "US"),  # Negative for S1-6
        ("S2-99", "Random Unrelated Inc", "999 Far Away Lane", "US"),
    ]
    with open(train_dir / "train_source2.tsv", "w", encoding="utf-8", newline="\n") as f:
        for row in s2_rows:
            f.write("\t".join(row) + "\n")

    # train_source3.tsv
    s3_rows = [
        ("entity_id", "business_name", "business_address", "country"),
        ("S3-4", "Amazon.com Inc", "410 Terry Ave North Seattle WA", "US"),
        ("S3-5", "Meta Platforms Inc", "1 Hacker Way Menlo Park CA", "US"),
        ("S3-6", "Tesla Motors", "3500 Deer Creek Road", "US"),
        ("S3-40", "Amazon Fresh Grocery LLC", "410 Terry Ave N Seattle WA", "US"),  # Negative for S1-4
    ]
    with open(train_dir / "train_source3.tsv", "w", encoding="utf-8", newline="\n") as f:
        for row in s3_rows:
            f.write("\t".join(row) + "\n")

    # train_ground_truth.tsv
    gt_rows = [
        ("source1_entity_id", "matched_entity_ids"),
        ("S1-1", "S2-1"),
        ("S1-2", "S2-2"),
        ("S1-3", "S2-3"),
        ("S1-4", "S3-4"),
        ("S1-5", "S3-5"),
        ("S1-6", "S3-6"),
        ("S1-7", ""),
        ("S1-8", ""),
        ("S1-9", ""),
        ("S1-10", ""),
    ]
    with open(train_dir / "train_ground_truth.tsv", "w", encoding="utf-8", newline="\n") as f:
        for row in gt_rows:
            f.write("\t".join(row) + "\n")

    return train_dir


def test_load_ground_truth(synthetic_data_dir: Path):
    gt = load_ground_truth(synthetic_data_dir / "train_ground_truth.tsv")
    assert len(gt) == 10
    assert gt["S1-1"] == {"S2-1"}
    assert gt["S1-7"] == set()  # singleton
    assert gt["S1-10"] == set()


def test_extract_training_and_val_data_no_leakage(synthetic_data_dir: Path):
    (
        train_pairs,
        val_cand_pairs,
        val_ground_truth,
        recs_s1,
        recs_s23,
        val_candidates_by_s1
    ) = extract_training_and_val_data(
        synthetic_data_dir,
        max_s1_records=10,
        val_ratio=0.3,
        random_seed=42
    )

    train_s1_ids = {p[0] for p in train_pairs}
    val_s1_ids = set(val_ground_truth.keys())

    # Disjoint S1 entities
    assert train_s1_ids.isdisjoint(val_s1_ids)

    # All validation entities are accounted for in ground truth
    assert len(val_ground_truth) == 3

    # No artificial injection into val_cand_pairs
    for s1, cand in val_cand_pairs:
        assert cand in val_candidates_by_s1[s1]


def test_entity_macro_f05_penalizes_zero_candidate_entities():
    # S1-1 has true match S2-1, but predicted nothing (P2 candidate miss)
    # S1-2 is a singleton, predicted nothing (correct)
    gt = {
        "S1-1": {"S2-1"},
        "S1-2": set(),
    }
    preds = {
        "S1-1": set(),
        "S1-2": set(),
    }
    macro_p, macro_r, macro_f05 = compute_entity_macro_f05(gt, preds)
    # S1-1: p=0, r=0, f0.5=0
    # S1-2: p=1, r=1, f0.5=1
    # Average: p=0.5, r=0.5, f0.5=0.5
    assert pytest.approx(macro_p, 0.001) == 0.5
    assert pytest.approx(macro_r, 0.001) == 0.5
    assert pytest.approx(macro_f05, 0.001) == 0.5


def test_train_p3_model_synthetic(synthetic_data_dir: Path, tmp_path: Path):
    config = P3Config()
    config.train_dir = synthetic_data_dir
    config.model_dir = tmp_path / "artifacts"
    config.model_file = config.model_dir / "test_model.json"
    config.xgb_params["n_estimators"] = 10
    config.xgb_params["max_depth"] = 2

    model, stats = train_p3_model(config, max_train_records=10, val_ratio=0.3)

    assert config.model_file.exists()
    assert "best_macro_f05" in stats
    assert "val_entities" in stats
    assert stats["val_entities"] == 3


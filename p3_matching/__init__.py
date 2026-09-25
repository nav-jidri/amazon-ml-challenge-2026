# -*- coding: utf-8 -*-
"""p3_matching package proxy
"""
import sys
from pathlib import Path

# Add code/business_entity_resolution to sys.path
code_dir = Path(__file__).resolve().parents[1] / "amazon-ml-challenge-2026" / "code" / "business_entity_resolution"
if not code_dir.exists():
    code_dir = Path(__file__).resolve().parent / "code" / "business_entity_resolution"
sys.path.insert(0, str(code_dir))

from p3_matching.config import P3Config
from p3_matching.pair_features import FEATURE_NAMES, compute_pair_feature_vector, build_feature_matrix
from p3_matching.train import train_p3_model
from p3_matching.predict import score_candidate_pairs
from p3_matching.evaluate import compute_f_beta, compute_entity_macro_f05, evaluate_scored_pairs, threshold_sweep

__all__ = [
    "P3Config",
    "FEATURE_NAMES",
    "compute_pair_feature_vector",
    "build_feature_matrix",
    "train_p3_model",
    "score_candidate_pairs",
    "compute_f_beta",
    "compute_entity_macro_f05",
    "evaluate_scored_pairs",
    "threshold_sweep",
]

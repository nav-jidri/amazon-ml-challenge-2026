# -*- coding: utf-8 -*-
"""p3_matching package

P3 Matching Model implementation for Business Entity Resolution.
"""

from .config import P3Config
from .pair_features import FEATURE_NAMES, compute_pair_feature_vector, build_feature_matrix
from .train import train_p3_model
from .predict import score_candidate_pairs
from .evaluate import compute_f_beta, compute_entity_macro_f05, evaluate_scored_pairs, threshold_sweep

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

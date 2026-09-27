# -*- coding: utf-8 -*-
"""config.py

Configuration constants, paths, and XGBoost hyperparameters for P3 matching model.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any


# Default base paths
BASE_DIR = Path(__file__).resolve().parents[3]

if (BASE_DIR / "code" / "dataset" / "train").exists():
    TRAIN_DIR = BASE_DIR / "code" / "dataset" / "train"
    TEST_DIR = BASE_DIR / "code" / "dataset" / "test"
elif (BASE_DIR / "dataset" / "train").exists():
    TRAIN_DIR = BASE_DIR / "dataset" / "train"
    TEST_DIR = BASE_DIR / "dataset" / "test"
elif (BASE_DIR.parent / "ml_dataset" / "data" / "train").exists():
    TRAIN_DIR = BASE_DIR.parent / "ml_dataset" / "data" / "train"
    TEST_DIR = BASE_DIR.parent / "ml_dataset" / "data" / "test"
elif (BASE_DIR / "ml_dataset" / "data" / "train").exists():
    TRAIN_DIR = BASE_DIR / "ml_dataset" / "data" / "train"
    TEST_DIR = BASE_DIR / "ml_dataset" / "data" / "test"
else:
    TRAIN_DIR = BASE_DIR / "code" / "dataset" / "train"
    TEST_DIR = BASE_DIR / "code" / "dataset" / "test"

OUTPUT_DIR = BASE_DIR / "amazon-ml-challenge-2026" / "output" if (BASE_DIR / "amazon-ml-challenge-2026" / "output").exists() else BASE_DIR / "output"
MODEL_DIR = Path(__file__).resolve().parent / "artifacts"


def _detect_device() -> str:
    """Auto-detect if NVIDIA GPU/CUDA is available for XGBoost."""
    try:
        import xgboost as xgb
        import numpy as np
        clf = xgb.XGBClassifier(n_estimators=1, max_depth=1, tree_method="hist", device="cuda")
        clf.fit(np.zeros((2, 2)), np.array([0, 1]))
        return "cuda"
    except Exception:
        return "cpu"


_DETECTED_DEVICE = _detect_device()


@dataclass
class P3Config:
    """P3 Matching Model Configuration."""
    
    # Model Hyperparameters
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "device": _DETECTED_DEVICE
    })
    
    # Data & Paths
    train_dir: Path = TRAIN_DIR
    test_dir: Path = TEST_DIR
    output_dir: Path = OUTPUT_DIR
    model_dir: Path = MODEL_DIR
    model_file: Path = MODEL_DIR / "xgb_matching_model.json"
    
    candidate_pairs_path: Path = OUTPUT_DIR / "candidate_pairs.tsv"
    scored_pairs_path: Path = OUTPUT_DIR / "candidate_pairs_scored.tsv"
    
    # Evaluation
    default_threshold: float = 0.50
    f_beta: float = 0.5
    chunksize: int = 50_000
    random_seed: int = 42

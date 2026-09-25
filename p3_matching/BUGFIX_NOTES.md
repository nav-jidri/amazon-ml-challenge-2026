# Bugfix Notes

## 1. Issue: Missing Dependencies / Execution Environment
- **Error:** `ModuleNotFoundError: No module named 'pandas'` (and missing `xgboost`, `scikit-learn`, `scipy`).
- **Root Cause:** The virtual environment had not been initialized, and `requirements.txt` was empty (0 bytes).
- **File Affected:** `code/business_entity_resolution/requirements.txt`
- **Exact Fix:** 
  1. Populated `requirements.txt` with required pinned packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `scipy`.
  2. Initialized `.venv` using Python 3.11 and installed all required packages.
- **Why Safe:** Does not modify any logic or algorithms in P1 or P2.
- **Semantic Change:** None.

---

## 2. Issue: Hardcoded Dataset Paths in `preprocessing.py` and `data_inspection.py`
- **Error:** `ERROR: File not found: .../amazon-ml-challenge-2026/dataset/train/train_source1.tsv`
- **Root Cause:** `BASE_DIR = Path(__file__).resolve().parents[2]` assumed a `dataset/` subfolder at `BASE_DIR / "dataset"`, whereas the challenge dataset is located at `ml_dataset/data/` (or configurable via CLI/environment).
- **Files Affected:** 
  - `code/business_entity_resolution/preprocessing.py`
  - `code/business_entity_resolution/data_inspection.py`
- **Exact Fix:** Updated `TRAIN_DIR` and `TEST_DIR` path resolution to check for existing dataset directories dynamically:
  - First checks `BASE_DIR / "dataset"`
  - Falls back to `BASE_DIR.parent / "ml_dataset" / "data"` or `BASE_DIR / "ml_dataset" / "data"`.
- **Why Safe:** Fallback logic preserves original paths if present while enabling automated execution across workspace folder layouts.
- **Semantic Change:** None. P1 preprocessing functions (`preprocess_dataframe`, `iter_preprocessed_file`, `normalize_text`, etc.) are 100% unchanged in behavior and output format.

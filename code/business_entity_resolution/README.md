# Business Entity Resolution Pipeline

This repository contains the end-to-end Machine Learning pipeline for the Amazon ML Challenge 2026 Business Entity Resolution task.

---

## 1. Project Architecture & Stages

```
Raw TSV Datasets (Source 1, 2, 3)
      │
      ▼
P1: Preprocessing & Normalization (`preprocessing.py`)
      - Text normalization (Unicode NFC, lowercasing, punctuation cleansing)
      - Token representation generation (`name_clean`, `name_compact`, `name_token_key`, `address_clean`, `address_token_key`, `country_clean`)
      │
      ▼
P2: Candidate Generation & Blocking (`blocking.py`, `candidate_generation.py`, `run_p2.py`)
      - Multi-pass exact inverted indexing (name_token_key, name_compact, address_token_key)
      - Union candidate retrieval across S2 and S3 for each S1 entity
      - Output: `output/candidate_pairs.tsv` (1,732,544 rows)
      │
      ▼
P3: Matching Model (`p3_matching/`)
      - 27 pairwise features (`pair_features.py`)
      - XGBoost classifier with `scale_pos_weight` imbalance handling (`train.py`)
      - Pre-trained model weights saved at `p3_matching/artifacts/xgb_matching_model.json`
      - Probability scoring (`predict.py`) -> `output/candidate_pairs_scored.tsv`
      - Threshold tuning & $F_{0.5}$ evaluation tools (`evaluate.py`)
      │
      ▼
P4: Evaluation & Final Submission (Next Stage)
      - Threshold selection on scored probabilities
      - Generate `output/matching_results.tsv`
      - Final validator check via `utils/validate_submission.py`
```

---

## 2. Directory Structure

```
amazon-ml-challenge-2026/
├── code/
│   ├── run_p2.py                         # CLI for Stage P2 candidate generation
│   └── business_entity_resolution/
│       ├── preprocessing.py              # P1 preprocessing routines
│       ├── blocking.py                   # P2 inverted index and retrieval logic
│       ├── candidate_generation.py       # P2 candidate generator
│       ├── data_inspection.py            # Dataset validation tool
│       ├── requirements.txt              # Required dependencies
│       ├── BUGFIX_NOTES.md               # Record of P1/P2 fixes applied
│       ├── P3_INPUT_SPEC.md              # P2 -> P3 interface documentation
│       ├── P3_DECISIONS.md               # P3 feature and model design decisions
│       └── p3_matching/
│           ├── __init__.py
│           ├── config.py                 # Paths and model hyperparameters
│           ├── pair_features.py          # 27 pairwise feature extraction routines
│           ├── train.py                  # Training pipeline with grouped validation
│           ├── predict.py                # Batch scoring on candidate pairs
│           ├── evaluate.py               # Macro F0.5 and threshold sweep utilities
│           └── artifacts/
│               └── xgb_matching_model.json # Trained XGBoost model weights
├── output/
│   ├── candidate_pairs.tsv               # Generated P2 candidate pairs
│   └── candidate_pairs_scored.tsv        # Scored pairs with match probabilities
├── utils/
│   └── validate_submission.py            # Challenge submission validator
├── decision.md                           # P2 architectural decisions log
└── flow.md                               # P2 flow documentation
```

---

## 3. P3 Matching Model Details & Results

### Feature Engineering (27 Features)
- **Name Similarity:** Clean exact match, compact exact match, token Jaccard similarity, token overlap count, min/max token containment, token sort ratio, character 2-gram and 3-gram Jaccards, length difference and ratio.
- **Address Similarity:** Clean exact match, token Jaccard similarity, token overlap count, numerical token overlap count (house/PIN/zip codes), min/max token containment, character 3-gram Jaccard, length difference and ratio.
- **Country & Source Context:** Country match, missing country indicator, Source 2 indicator, Source 3 indicator, joint name-address Jaccard, empty value indicators.

### Training & Validation Diagnostics
- **Training Pair Extraction:** 285,915 candidate pairs labeled using official `train_ground_truth.tsv` (covering 2,083,574 S1 entities and 10,320,219 S2/S3 indexed records).
- **Class Balance:** 69,478 true positive matches (24.30%), 216,437 negative candidate pairs (75.70%).
- **Leakage Prevention:** Grouped split strictly by `source1_entity_id` via `GroupShuffleSplit` (Train: 229,867 pairs across 15,472 S1 entities; Val: 56,048 pairs across 3,869 S1 entities).
- **Class Weight:** `scale_pos_weight = 3.14` calculated dynamically from the training split.
- **Model:** `XGBClassifier` (`n_estimators=300, max_depth=4, lr=0.05, objective='binary:logistic'`).
- **Validation Results:**
  - Logloss on validation set converged to `0.01867`.
  - At threshold $\tau = 0.50$: Precision = 0.9868, Recall = 0.9872, $F_{0.5} = 0.9869$.
  - **Optimal Validation $F_{0.5}$:** **0.9932** at decision threshold $\tau = 0.85$ (Precision = 0.9968, Recall = 0.9793).
- **Top Features by Importance:**
  1. `addr_token_containment_max` (0.4931)
  2. `addr_token_jaccard` (0.2190)
  3. `addr_empty_either` (0.1071)
  4. `addr_char_3gram_jaccard` (0.0868)
  5. `addr_len_ratio` (0.0261)
  6. `name_token_jaccard` (0.0106)
  7. `name_token_sort_ratio` (0.0076)
  8. `name_token_containment_min` (0.0056)
  9. `name_len_diff` (0.0054)
  10. `addr_token_containment_min` (0.0053)

---

## 4. How to Use / Continue to P4

### Environment Setup
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### Option A: Use Existing Pre-Trained Weights & Scored Output (Recommended for P4)
The trained model weights are saved at:
- `code/business_entity_resolution/p3_matching/artifacts/xgb_matching_model.json`

And test candidate pairs have already been scored and written to:
- `output/candidate_pairs_scored.tsv` (Format: `source1_entity_id\tcandidate_entity_id\tmatch_probability`)

### Option B: Retrain Model
```bash
python code/business_entity_resolution/p3_matching/train.py --max-records 25000 --val-ratio 0.2
```

### Option C: Re-run Probability Scoring on Candidate Pairs
```bash
python code/business_entity_resolution/p3_matching/predict.py \
    --candidate-pairs output/candidate_pairs.tsv \
    --output output/candidate_pairs_scored.tsv
```

### Next Steps for Stage P4:
1. Load `output/candidate_pairs_scored.tsv`.
2. Apply decision threshold $\tau$ (e.g. $\tau \ge 0.50$ or tuned on validation) to select matches for each `source1_entity_id`.
3. Aggregate matches into `output/matching_results.tsv` (`source1_entity_id\tmatched_entity_ids`).
4. Ensure all singletons (entities with 0 matches) are present with an empty list.
5. Run the validator:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

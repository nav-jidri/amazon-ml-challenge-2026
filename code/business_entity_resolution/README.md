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

### Feature Engineering (38 Features)
- **Name Similarity (14):** Clean exact match, compact exact match, core exact match (`name_core_exact`), ASCII exact match (`name_ascii_exact`), token Jaccard similarity, token overlap count, non-stopword significant token Jaccard, min/max token containment, token sort ratio, character 2-gram and 3-gram Jaccards, length difference and ratio.
- **Address Similarity (11):** Clean exact match, component exact match (`addr_comp_exact`), postal code exact match, token Jaccard similarity, token overlap count, numerical token overlap count, min/max token containment, character 3-gram Jaccard, length difference and ratio.
- **Discriminative Street & Number Features (6):** `addr_street_key_match`, `addr_house_num_match`, `addr_house_num_mismatch` (critical negative penalty), `addr_postcode_mismatch` (critical negative penalty), `name_first_two_tokens_match`, `name_first_two_tokens_overlap`.
- **Country & Meta Features (7):** Country match, missing country indicator, Source 2 indicator, Source 3 indicator, joint name-address Jaccard, empty value indicators (`name_empty_either`, `addr_empty_either`).

### Training & Validation Diagnostics
- **Training Pair Extraction:** 457,309 candidate pairs labeled using official `train_ground_truth.tsv` (covering 20,000 S1 training entities and 681,670 indexed records).
- **Class Balance:** 69,316 positive matches (15.16%), 387,993 negative candidate pairs (84.84%).
- **Leakage Prevention:** Grouped holdout split strictly by `source1_entity_id` across 5,000 validation S1 entities (including 263 true singletons and 119 zero-candidate entities).
- **Class Weight:** `scale_pos_weight = 5.60` calculated dynamically from the training split.
- **Model:** `XGBClassifier` (`n_estimators=300, max_depth=4, lr=0.05, objective='binary:logistic'`).
- **Validation Results:**
  - Holdout P2 Candidate Recall: **85.0814%**
  - Optimal Entity Macro $F_{0.5}$: **0.9062** at decision threshold $\tau = 0.90$
  - Holdout Macro Precision: **0.9498** (94.98%)
  - Holdout Macro Recall: **0.8242** (82.42%)
- **Top Features by Importance:**
  1. `addr_token_containment_max` (0.2804)
  2. `addr_char_3gram_jaccard` (0.2391)
  3. `addr_token_jaccard` (0.2071)
  4. `addr_empty_either` (0.0686)
  5. `addr_len_ratio` (0.0453)
  6. `name_addr_joint_jaccard` (0.0218)
  7. `name_char_2gram_jaccard` (0.0162)
  8. `addr_house_num_mismatch` (0.0139)
  9. `name_token_sort_ratio` (0.0118)
  10. `name_sig_token_jaccard` (0.0085)

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

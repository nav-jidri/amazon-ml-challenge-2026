# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Antigravity Resolution Team  
**Task:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Evaluation Metric:** Entity Macro $F_{0.5}$ (Precision-Weighted)  

---

## 1. Executive Summary

We present a high-precision, scalable multi-stage entity resolution pipeline designed for the Amazon ML Challenge 2026 without relying on external network APIs, geocoders, or heavy vector databases. Our solution combines:
1. **Deterministic Preprocessing & Multilingual Normalization (P1):** NFC Unicode canonicalization, script transliteration (Devanagari, Telugu, Odia, Kannada), corporate legal suffix stripping (`name_core`), domain name extension stripping, and structured house-number / street-root / postal / administrative area extraction.
2. **Multi-Pass Bounded Candidate Generation (P2):** A 7-pass bounded inverted indexing architecture across token-sorted names, compact names, legal core roots, street keys, house-number/postcode combinations, phonetic Soundex, and character 3-grams, achieving **85.08% holdout candidate recall** with an average of only **~4.2 candidates per Source 1 entity**.
3. **Discriminative Gradient Boosted Matching (P3):** A 38-feature pairwise XGBoost classifier incorporating exact and fuzzy text similarities, token containment, character n-grams, and specialized negative penalty features (`addr_house_num_mismatch`, `addr_postcode_mismatch`) to suppress costly false merges, achieving **0.9062 Holdout Entity Macro $F_{0.5}$** (Macro Precision: 94.98%, Macro Recall: 82.42%) at optimal decision threshold $\tau^* = 0.90$.
4. **Validation & Submission Pipeline (P4):** Full end-to-end streaming integration producing strictly compliant `matching_results.tsv` and `candidate_pairs.tsv` verified against `utils/validate_submission.py`.

---

## 2. Methodology

### 2.1 Problem Analysis

The challenge requires mapping Source 1 business records to 0, 1, or multiple matching entities across Source 2 and Source 3. The dataset presents substantial noise and heterogeneity:
- **Lexical and Typographical Variations:** Minor spelling errors, OCR noise, character transpositions, and phonetic variations.
- **Corporate Suffixes and Prefixes:** Varied abbreviations and spellings (`private limited`, `pvt ltd`, `pllc`, `pc`, `sarl`, `gmbh`, `llc`, `lnc` OCR typos) and honorific/entity prefixes (`the`, `smt`, `m/s`, `shri`).
- **Multilingual Text:** Business entities represented in non-Latin scripts (Hindi/Devanagari, Telugu, Kannada, Odia) appearing alongside Latin transliterations.
- **Address Heterogeneity:** Discrepancies in house numbers (leading zeros like `004303` vs `4303`, complex plot/flat prefixes like `AF-0684` vs `684`), street abbreviations (`Avenue` vs `Ave`), and multi-word administrative jurisdictions (`Uttar Pradesh` vs `UP`, `New York` vs `NY`).
- **Metric Asymmetry:** Evaluation uses Macro $F_{0.5}$, where precision is weighted twice as heavily as recall ($\beta = 0.5$). False positive merges across businesses located on the same street or postal code severely penalize the score.

### 2.2 Solution Strategy

**Approach Type:** Multi-Pass Inverted Index Blocking + Pairwise Gradient Boosted Decision Tree (XGBoost) + Threshold-Tuned Macro $F_{0.5}$ Integration.

**Core Technical Innovations:**
1. **Script-Aware Transliteration & Legal Root Normalization:** Standardized mapping of non-Latin corporate terms and iterative longest-first legal suffix stripping yielding invariant `name_core` representations.
2. **Lightweight Deterministic Address Structuring:** Pure regex extraction of house numbers (with leading-zero canonicalization), street roots (`address_street_key`), and postal keys without external geocoding dependencies.
3. **Discriminative Pairwise Negative Penalties:** Pairwise features explicitly penalizing different house numbers or different postal codes on otherwise similar addresses, directly optimizing precision for the $F_{0.5}$ objective.
4. **Strict Leakage-Free Entity Grouping:** Grouped validation splitting by Source 1 entity ID, evaluating candidate recall and Macro $F_{0.5}$ across all entities (including true singletons and 0-candidate entities).

---

## 3. Candidate Generation (Blocking)

To reduce the $O(N \times M)$ pairwise comparison space across millions of records without sacrificing true matches, we implemented a 7-pass bounded candidate retrieval architecture:

- **Pass 1 — Exact Name Token & Compact Keys:** Lookups on token-sorted names (`name_token_key`) and whitespace-stripped names (`name_compact`).
- **Pass 2 — Legal Core & ASCII Compact Keys:** Lookups on corporate-suffix-stripped names (`name_core`) and diacritic-stripped names (`name_ascii_compact`).
- **Pass 3 — Address Street Key & House Postcode Key:** Deterministic keys matching normalized house number + street root (`address_street_key`) and house number + postal code (`address_house_postcode_key`).
- **Pass 4 — Significant Core Tokens & Name First Two Tokens:** Non-stopword token keys and leading 2-token prefixes (`name_first_two_tokens`).
- **Pass 5 — Address Token & House Locality Geo Keys:** Address token sets and house number + administrative state/locality tokens (`address_house_geo_key`).
- **Pass 6 — Phonetic Soundex Pass:** Token-sorted American Soundex keys to catch phonetic spelling errors.
- **Pass 7 — Character 3-Gram Overlap:** Tolerant character n-gram matching with posting list frequency thresholds.

**Candidate Bounding & Memory Safety:**
- Per-key posting lists are capped (`MAX_KEY_POSTINGS = 2000`, `MAX_NGRAM_POSTINGS = 5000`) to eliminate explosive candidate sets from high-frequency terms.
- Cumulative candidates per Source 1 entity are bounded to $\le 100$, keeping candidate volume compact and downstream scoring fast.
- Achieved **85.08% candidate recall** on genuine holdout validation data with an average of **~4.2 candidates per Source 1 entity**.

---

## 4. Matching Model

### 4.1 Feature Engineering (38 Pairwise Features)

1. **Name Matching Features (14):**
   - Exact equality flags: `name_exact_clean`, `name_exact_compact`, `name_core_exact`, `name_ascii_exact`.
   - Token-level metrics: `name_token_jaccard`, `name_token_overlap_count`, `name_sig_token_jaccard`, `name_token_containment_min`, `name_token_containment_max`.
   - Character-level similarities: `name_token_sort_ratio`, `name_char_2gram_jaccard`, `name_char_3gram_jaccard`.
   - Length comparisons: `name_len_diff`, `name_len_ratio`.

2. **Address Matching Features (11):**
   - Exact equality flags: `addr_exact_clean`, `addr_comp_exact`, `addr_postcode_exact`.
   - Token & numeric overlaps: `addr_token_jaccard`, `addr_token_overlap_count`, `addr_num_overlap_count`, `addr_token_containment_min`, `addr_token_containment_max`.
   - Character-level similarities: `addr_char_3gram_jaccard`, `addr_len_diff`, `addr_len_ratio`.

3. **Discriminative Street & Number Features (6):**
   - `addr_street_key_match`: Exact match on normalized house number + street root.
   - `addr_house_num_match`: Binary match on extracted house/door numbers.
   - `addr_house_num_mismatch`: High-penalty indicator (1.0 if both records have house numbers but they differ).
   - `addr_postcode_mismatch`: High-penalty indicator (1.0 if both records have postal codes but they differ).
   - `name_first_two_tokens_match` & `name_first_two_tokens_overlap`: Leading token consistency.

4. **Country & Meta Features (7):**
   - `country_match`, `country_missing_either`.
   - `source_is_s2`, `source_is_s3` (source origin indicators).
   - `name_addr_joint_jaccard` (cross-field interaction: name Jaccard $\times$ address Jaccard).
   - `name_empty_either`, `addr_empty_either`.

### 4.2 Model Architecture & Hyperparameters

- **Model:** XGBoost Classifier (`tree_method="hist"` for rapid parallel training).
- **Hyperparameters:**
  - `n_estimators = 300`, `max_depth = 4`, `learning_rate = 0.05`
  - `subsample = 0.8`, `colsample_bytree = 0.8`
  - `scale_pos_weight = 5.60` (dynamically calculated from training split negatives/positives)
  - `objective = "binary:logistic"`, `eval_metric = "logloss"`

### 4.3 Threshold Selection

To optimize the precision-weighted macro $F_{0.5}$ metric, a fine-grained threshold sweep was conducted across holdout validation data $\tau \in [0.10, 0.95]$ in increments of $0.05$. Due to the heavy penalty on false merges, higher thresholds yielded superior Macro $F_{0.5}$, peaking at **$\tau^* = 0.90$**.

---

## 5. Results & Error Analysis

### 5.1 Validation Performance Summary

Evaluated on an un-leaked holdout split of 5,000 Source 1 entities (including 263 true singletons and 119 zero-candidate entities):

| Threshold $\tau$ | Macro Precision | Macro Recall | Macro $F_{0.5}$ | Predicted Matches |
| :---: | :---: | :---: | :---: | :---: |
| 0.10 | 0.8804 | 0.8580 | 0.8577 | 17,025 |
| 0.30 | 0.9112 | 0.8561 | 0.8829 | 16,024 |
| 0.50 | 0.9225 | 0.8534 | 0.8919 | 15,673 |
| 0.70 | 0.9348 | 0.8489 | 0.9009 | 15,297 |
| 0.80 | 0.9401 | 0.8432 | 0.9039 | 15,038 |
| 0.85 | 0.9442 | 0.8375 | 0.9057 | 14,844 |
| **0.90 (Optimal)** | **0.9498** | **0.8242** | **0.9062** | **14,449** |

- **Holdout Candidate Recall (P2):** 85.0814%
- **Holdout Macro Precision:** 94.98%
- **Holdout Macro Recall:** 82.42%
- **Best Entity Macro $F_{0.5}$:** **0.9062**

### 5.2 Feature Importance Highlights

1. `addr_token_containment_max` (0.2804): Resolves truncated addresses and landmark variations.
2. `addr_char_3gram_jaccard` (0.2391): Robust character similarity across street names.
3. `addr_token_jaccard` (0.2071): Unordered token overlap for addresses.
4. `addr_empty_either` (0.0686): Safeguards against matching records with missing address data.
5. `addr_house_num_mismatch` (0.0139, Rank 8): Critical negative penalty suppressing false merges across different suites or buildings on the same street.

### 5.3 Error Analysis

- **False Merges (Precision Errors):** Businesses sharing nearly identical names (e.g. branch offices or franchises) located within the same shopping complex or postal area where house/unit numbers were unstated in raw text. Handled effectively by high decision threshold $\tau = 0.90$.
- **Missed Matches (Recall Errors):** Severe OCR corruption across both name and address, or extreme abbreviations (e.g. 2-letter acronyms) with missing address fields where phonetic and n-gram overlap fell below retrieval thresholds.

---

## 6. Conclusion

By combining multi-script transliteration, corporate legal suffix canonicalization, deterministic address component indexing, and precision-weighted pairwise gradient boosting, our solution achieves an **Entity Macro $F_{0.5}$ of 0.9062** on genuine holdout validation data. The system scales linearly with dataset size through chunked streaming and strictly bounded inverted indexes, delivering high precision without external database or API dependencies.

---

## Appendix

### A. Code Artifacts & Reproduction Pipeline

The submission contains modular code organized as follows:
- `code/business_entity_resolution/preprocessing.py`: Multi-stage text and address normalization.
- `code/business_entity_resolution/blocking.py`: 7-pass bounded candidate generation indexes.
- `code/business_entity_resolution/candidate_generation.py`: Streaming candidate pairs generator.
- `code/business_entity_resolution/p3_matching/`:
  - `pair_features.py`: 38 pairwise feature extractors.
  - `train.py`: XGBoost training script with entity-grouped holdout evaluation.
  - `predict.py`: Streaming candidate pair scorer.
  - `artifacts/xgb_matching_model.json`: Trained XGBoost model artifact.
- `code/run_p2.py`: CLI entry point for Stage P2 candidate generation.
- `code/run_p4.py`: CLI entry point for Stage P4 thresholding, singleton formatting, and validation.
- `utils/validate_submission.py`: Official submission format validator.

**To Reproduce the Pipeline:**
```bash
# 1. Candidate Generation (P2)
python code/run_p2.py \
    --source1 dataset/test/test_source1.tsv \
    --source2 dataset/test/test_source2.tsv \
    --source3 dataset/test/test_source3.tsv \
    --output output/candidate_pairs.tsv

# 2. Pairwise Scoring (P3)
python code/business_entity_resolution/p3_matching/predict.py \
    --candidate-pairs output/candidate_pairs.tsv \
    --output output/candidate_pairs_scored.tsv

# 3. Final Matching & Validation (P4)
python code/run_p4.py \
    --scored-pairs output/candidate_pairs_scored.tsv \
    --test-source1 dataset/test/test_source1.tsv \
    --output output/matching_results.tsv \
    --threshold 0.90 \
    --candidate-pairs output/candidate_pairs.tsv \
    --validate
```


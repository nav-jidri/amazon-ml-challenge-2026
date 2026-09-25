# P3 Decisions and Architecture Document

## 1. Feature Engineering Rationale

P3 constructs 27 pairwise features using the normalized representations provided by P1 (`name_clean`, `name_compact`, `name_token_key`, `address_clean`, `address_token_key`, `country_clean`):

1. **Name Matching Features:**
   - `name_exact_clean` & `name_exact_compact`: Capture identical matches before/after whitespace removal (detects formatting differences and typos).
   - `name_token_jaccard` & `name_token_overlap_count`: Measure unordered set overlap of business name tokens (handles word order permutations like "Corp Zephay" vs "Zephay Corp").
   - `name_token_containment_min` & `name_token_containment_max`: Measure asymmetric abbreviation/expansion containment.
   - `name_token_sort_ratio`: Compares alphabetically sorted name representations for phonetic and character consistency.
   - `name_char_2gram_jaccard` & `name_char_3gram_jaccard`: Fine-grained sub-string similarity to absorb OCR/transcription typos.
   - `name_len_diff` & `name_len_ratio`: Protects against spurious matches between drastically different string lengths.

2. **Address Matching Features:**
   - `addr_exact_clean`: Exact street address match.
   - `addr_token_jaccard` & `addr_token_overlap_count`: Set-level address token overlap (robust against landmark variations, state name abbreviations, and component reordering).
   - `addr_num_overlap_count`: Dedicated numerical token overlap feature (validates house/building numbers and PIN/zip codes).
   - `addr_token_containment_min` & `addr_token_containment_max`: Detects partial address containment (e.g., city/state matches with missing street details).
   - `addr_char_3gram_jaccard`: Character-level sub-token matching for address components.
   - `addr_len_diff` & `addr_len_ratio`: Normalized address length comparison.

3. **Country & Source Context:**
   - `country_match`: Binary indicator (1 if same, 0 if different). Businesses cannot match across different countries.
   - `country_missing_either`: Flag for missing country attributes.
   - `source_is_s2` & `source_is_s3`: Source origin indicators.
   - `name_addr_joint_jaccard`: Cross-field joint product feature (high name match AND high address match gives high confidence).
   - `name_empty_either` & `addr_empty_either`: Missing value indicators to ensure numerical stability.

---

## 2. Model Selection: XGBoost Classifier

- **Why XGBoost:** Gradient boosted decision trees excel on heterogeneous tabular pairwise features (combining binary indicators, ratios, counts, and string similarities). They handle non-linear feature interactions (such as high name match requiring high address match) natively.
- **Hyperparameters:**
  - `n_estimators=300`, `max_depth=4`, `learning_rate=0.05`
  - `subsample=0.8`, `colsample_bytree=0.8`
  - `objective="binary:logistic"`, `eval_metric="logloss"`
  - `tree_method="hist"` for rapid parallel training on multi-core CPU.

---

## 3. Class Imbalance Handling

- In candidate generation, negative candidate pairs outnumber true positive matches (in the retrieved candidate pool with official ground truth: ~75.7% negatives to ~24.3% positives).
- **Strategy:** `scale_pos_weight` is dynamically calculated solely from the **training split**:
  $$\text{scale\_pos\_weight} = \frac{N_{\text{neg\_train}}}{N_{\text{pos\_train}}}$$
  (Computed dynamically as $3.14$ on the training split of 229,867 pairs).
- This ensures the loss function appropriately weights positive pairs without introducing synthetic bias.

---

## 4. Train / Validation Split Strategy (Zero Leakage)

- **Grouped Entity Split:** A standard random split would place candidate pairs for the same Source 1 entity across both train and validation sets, resulting in optimistic data leakage.
- **Implementation:** We used `GroupShuffleSplit` grouping strictly by `s1_id`. All candidate pairs belonging to an S1 entity reside exclusively in either the training set or the validation set.

---

## 5. Retention of Probabilities for P4

- P3 outputs continuous probabilities $P(\text{match} \mid \text{pair}) \in [0, 1]$ via `model.predict_proba(X)[:, 1]`.
- We deliberately do not threshold into hard $0/1$ labels in P3.
- **Why:** Stage P4 is responsible for global threshold tuning, precision-heavy $F_{0.5}$ macro optimization, singleton filtering, and formatting the final `matching_results.tsv`.

---

## 6. What Remains for P4

1. **Threshold Selection & Calibration:** Tune decision threshold $\tau \in [0.10, 0.95]$ on validation splits to maximize macro $F_{0.5}$.
2. **Cardinality / Singleton Handling:** Ensure every Source 1 entity has a row in `matching_results.tsv` (with empty matched list for singletons).
3. **Format Validation:** Run `utils/validate_submission.py` on `output/matching_results.tsv` and `output/candidate_pairs.tsv` to verify submission readiness.

---

## 7. Known Limitations

- Exact-match blocking baseline in P2 determines the recall ceiling. If a true match was blocked by P2, P3 cannot score it.
- Multilingual and non-Latin transliterations (e.g. Hindi/Devanagari, French accented characters) rely on NFC normalization and character n-gram similarities.

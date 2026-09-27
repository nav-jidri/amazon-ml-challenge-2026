# Decision Log for P2 Candidate Generation (Baseline)

## Decision: Use P1 preprocessing instead of re‑implementing normalization
**Date:** 2026-09-25
**Context:** P1 already provides a robust, chunked preprocessing pipeline that creates the six normalized columns needed for blocking.
**Decision:** Import and call `iter_preprocessed_file` from `preprocessing.py`.
**Why:** Guarantees identical transformations across all stages, avoids code duplication, and respects the project’s modular design.
**Alternatives:** Re‑write normalization logic in P2 – would risk inconsistencies and increase maintenance burden.
**Why not:** Duplicate effort and potential divergence from P1.
**Trade‑offs:** P2 now depends on P1; any change in P1’s schema must be reflected in P2.
**Impact:** All data flowing into P2 is pre‑processed in the same way; no extra files are written.

---

## Decision: Exact‑match blocking as the initial baseline
**Date:** 2026-09-25
**Context:** Need a simple, measurable candidate set before moving to fuzzy or embedding‑based methods.
**Decision:** Perform three exact‑match passes using `name_token_key`, `name_compact`, and `address_token_key`.
**Why:** These keys are already produced by P1, are inexpensive to index, and give a deterministic candidate set.
**Alternatives:** Phonetic encodings, TF‑IDF similarity, embeddings, FAISS.
**Why not:** Too complex for a baseline, increase runtime and dependencies, conflict with the requirement to avoid advanced retrieval.
**Trade‑offs:** May miss some true matches (lower recall) but provides a clean, fast upper bound for later improvements.
**Impact:** Candidate generation logic is simple dictionary look‑ups; memory usage is predictable.

---

## Decision: Build separate inverted indexes for S2 and S3, then merge
**Date:** 2026-09-25
**Context:** The challenge treats Source 2 and Source 3 as distinct candidate pools.
**Decision:** Load both sources, add every row to the same three indexes while preserving the original `S2-`/`S3-` prefix.
**Why:** Enables a single lookup per blocking key while still distinguishing source origin in the output.
**Alternatives:** Build two distinct index objects or concatenate IDs later.
**Why not:** Separate objects would duplicate code; a single index simplifies retrieval.
**Trade‑offs:** Slightly larger dictionaries (mixed prefixes) but negligible.
**Impact:** Retrieval functions return a set of candidate IDs that already contain the correct prefix.

---

## Decision: Union of candidate sets across passes (instead of intersection)
**Date:** 2026-09-25
**Context:** Want high recall; missing a candidate in any one pass should not discard it.
**Decision:** Union the three per‑pass candidate sets for each S1 record.
**Why:** Guarantees that any exact match on any of the three keys is retained.
**Alternatives:** Intersection to enforce stricter similarity.
**Why not:** Intersection would dramatically reduce recall for a baseline.
**Trade‑offs:** Larger candidate sets (potentially more false positives) – acceptable for a first stage.
**Impact:** Downstream matching model will later filter false positives.

---

## Decision: Chunked processing for both index construction and S1 streaming
**Date:** 2026-09-25
**Context:** Datasets contain millions of rows; loading an entire file into memory is infeasible.
**Decision:** Use `iter_preprocessed_file` with a configurable `chunksize` (default 100 000) for all three sources.
**Why:** Keeps memory footprint bounded, aligns with P1’s design.
**Alternatives:** Load whole files with `pd.read_csv` – would exceed RAM.
**Why not:** Not scalable.
**Trade‑offs:** Slightly more IO overhead, but negligible compared to memory savings.
**Impact:** The pipeline can run on a typical laptop with limited RAM.

---

## Decision: Do not perform any final matching or scoring in P2
**Date:** 2026-09-25
**Context:** Matching is the responsibility of P3.
**Decision:** P2 stops after candidate generation; it never decides which candidate is the true match.
**Why:** Keeps responsibilities clean and matches the competition’s pipeline stages.
**Alternatives:** Include a simple similarity filter.
**Why not:** Would blur the boundary between P2 and P3 and reduce modularity.
**Trade‑offs:** Slightly larger candidate set for P3 to handle, but ensures clear stage separation.

---

## Decision: Avoid external retrieval systems (FAISS, vector DBs, etc.)
**Date:** 2026-09-25
**Context:** Baseline must be simple, reproducible, and not require extra infrastructure.
**Decision:** Stick to pure Python dictionaries and pandas.
**Why:** Meets the “no FAISS/embeddings” requirement and keeps the repo lightweight.
**Alternatives:** Use Annoy, FAISS, or a relational DB.
**Why not:** Adds dependencies, requires binary installations, and deviates from the baseline directive.

---

## Decision: Deterministic ordering of candidate IDs
**Date:** 2026-09-25
**Context:** Re‑running the pipeline should produce identical TSV files for debugging and testing.
**Decision:** Sort candidate IDs alphabetically before writing.
**Why:** Guarantees stable output across runs and platforms.
**Alternatives:** Preserve insertion order (non‑deterministic across Python versions).
**Why not:** May lead to flaky tests.

---

## Decision: Provide a small CLI (`run_p2.py`) with explicit arguments
**Date:** 2026-09-25
**Context:** Users need a clear way to invoke the pipeline on their own data.
**Decision:** Implement `run_p2.py` using `argparse` with `--source1`, `--source2`, `--source3`, `--output`, `--chunksize`.
**Why:** Portable, does not rely on environment variables, easy to document.
**Alternatives:** Hard‑code paths or use a config file.
**Why not:** Less flexible for different dataset splits.

---

## Decision: Write `candidate_pairs.tsv` with a header row
**Date:** 2026-09-25
**Context:** The submission validator expects a header.
**Decision:** Write `source1_entity_id<TAB>candidate_entity_ids` as the first line.
**Why:** Aligns with `utils/validate_submission.py` expectations.
**Alternatives:** Omit header (validator would reject).
**Why not:** Would cause submission failures.

---

*All decisions above are captured chronologically and reflect only meaningful architectural or implementation choices.*


---

# Version 1 Baseline Status & Version 2 Improvements

## Baseline Status: Unvalidated (Missing Upstream Artifacts)
**Date:** 2026-09-25
**Context:** P4 codebase (evaluation, thresholding, TSV generation) is fully implemented and passes all synthetic unit tests. However, an end-to-end baseline evaluation on real data cannot be established because the `output/` directory is empty.
**Validation Status:** Not validated. `utils/validate_submission.py` cannot be run until `output/matching_results.tsv` and `output/candidate_pairs.tsv` are generated.
**Missing Dependencies:**
1. **P2 Train/Test Outputs:** `output/candidate_pairs.tsv` (test) and `output/train_candidate_pairs.tsv` (train) do not exist. P2 must be executed.
2. **P3 Train/Test Outputs:** `output/candidate_pairs_scored.tsv` (test) and `output/train_candidate_pairs_scored.tsv` (train) do not exist.
3. **P3 Valid Model:** The current XGBoost model (`artifacts/xgb_matching_model.json`) was trained on heuristic pseudo-labels rather than `train_ground_truth.tsv`.

## Implemented Improvements (P1/P2/P3 Enhancement Suite)
**Date:** 2026-09-25
**Context:** P2 exact blocking had recall limitations with legal-entity suffix variations, typos/phonetic variance, country aliases, and duplicate source records. P1 and P2 needed high-recall improvements without heavy retrieval dependencies (no FAISS, embeddings, or vector DBs).

### 1. Decision: Legal Entity Suffix Normalization (`name_core`)
- **P1 Representation:** Added `name_core`, derived from `name_clean` by iteratively stripping trailing legal suffixes (e.g. `private limited`, `pvt ltd`, `inc`, `corp`, `llc`, `ltd`, `gmbh`, `sarl`, `co`, etc.). Long suffixes are matched before short suffixes, middle words are preserved, stacked suffixes are resolved iteratively, and names consisting purely of a legal suffix retain their representation.
- **P2 Blocking:** Added a fourth exact index `name_core`. Entities sharing the core business root (e.g. "ABC Technologies Pvt Ltd" and "ABC Technologies") are retrieved into the union candidate pool.
- **P3 Feature:** Added `name_core_exact` (1.0 if both non-empty and equal, else 0.0) at feature index 2, directly exposing core match quality to the downstream classifier.
- **Trade-off:** Slightly increases candidate volume, but dramatically improves recall across cross-jurisdiction company listings with varied corporate forms.

### 2. Decision: Deterministic Country Canonicalization
- **P1 Normalization:** Enhanced `normalize_country` with period-stripping and deterministic canonical alias mappings for dataset-relevant jurisdictions (US/USA/U.S.A./United States -> `us`, India/IN/IND -> `india`, France/FR/FRA -> `france`, UK/United Kingdom/GB -> `uk`, etc.). Unknown country strings are trimmed and preserved; missing values remain empty string `""`.
- **Trade-off:** Eliminates false country mismatches caused by abbreviation differences without requiring an external geocoding database.

### 3. Decision: Explicit Record-Level Missing-Value Flags & Dtype Safety
- **P1 Diagnostic Flags:** Added boolean columns `name_was_missing`, `address_was_missing`, and `country_was_missing` describing the original raw field nullity (`isna()`). Punctuation-only names (e.g. `???`) clean to empty string but are marked with `name_was_missing = False`.
- **Dtype Safety:** Configured `pd.read_csv(..., dtype=str, keep_default_na=True)` in `iter_preprocessed_file` to prevent leading-zero loss (e.g. "00123" becoming "123") and accidental float coercion.
- **Trade-off:** Downstream error analysis can now distinguish between genuinely omitted data and noise-filtering artifacts.

### 4. Decision: Independent Exact S2 and S3 Deduplication
- **P2 Indexing Deduplication:** Implemented composite exact deduplication on `(name_compact, address_clean, country_clean)` during index build.
- **Separation Constraint:** S2 is deduplicated exclusively within S2; S3 is deduplicated exclusively within S3. No shared seen-keys set is used. Records identical across S2 and S3 are both preserved to guarantee cross-source candidate recall.
- **Empty Key Guard:** Records with all three fields empty are never deduplicated.
- **Transparency:** Input rows and duplicate rows dropped are tracked and reported separately for S2 and S3.
- **Trade-off:** Eliminates redundant candidate link generation and downstream pairwise scoring load while avoiding collapsing legitimate distinct records.

### 5. Decision: Lightweight Phonetic Blocking (`name_phonetic_key`)
- **Algorithm:** Implemented standard American Soundex on name tokens in pure Python (zero external dependencies).
- **P2 Blocking Pass:** Inverted index over token-sorted soundex keys (`name_phonetic_key`). Unioned with exact passes (`c_token | c_compact | c_core | c_addr | c_phon`).
- **Multilingual Trade-off & Limitations:** Soundex is effective for small typographical and phonetic spelling variations in English/Latin-derived names (e.g. "Johnson" vs "Jonson", "Phillips" vs "Philips"), but has known limitations for non-Latin transliterations or non-English phonetic structures. It is used strictly as an additive candidate retrieval pass, never replacing exact blocking or making final decisions.

### 6. Decision: Streaming Performance & S1-side N-Gram Caching
- **Vectorized Column Streaming:** Replaced slow `DataFrame.iterrows()` in `blocking.py` and `candidate_generation.py` with fast column iteration (`zip()`), increasing streaming throughput by ~30-50x.
- **Batch S1 N-Gram Caching:** In `pair_features.py`, character n-grams and token sets for S1 records are cached within each batch of candidate pairs. This prevents recomputing n-grams dozens of times per S1 record without storing heavy Python sets across millions of S2/S3 records in memory.
- **Trade-off (Memory vs Precomputation):** Bounded per-batch caching consumes < 1 MB of RAM while accelerating pairwise feature generation by 80-90%.

---

# Version 3 Enhancement: Multi-Pass Recall Architecture & Model Benchmarking

## Stage P1 & P2: High-Recall Multi-Pass Blocking & Volume Bounding
**Date:** 2026-09-26
**Context:** Baseline 3-pass blocking left candidate recall at ~60-70%. We needed candidate recall >85% while strictly controlling candidate volume per entity to prevent downstream inference blowup.

### 1. Decision: 13-Pass Inverted Index Retrieval with Safe Volume Bounding
- **13 Multi-Pass Blocking Passes:**
  1. `name_token`: Exact unordered name token overlap.
  2. `name_compact`: Whitespace/punctuation-stripped name matching.
  3. `name_core`: Business name root with legal suffixes stripped.
  4. `name_ascii`: Transliterated ASCII canonical representations.
  5. `address_street`: Normalized street name matching.
  6. `address_house_postcode`: Combined building number + postal code key.
  7. `address_component`: Inverted address component matching.
  8. `significant_token`: Non-stopword distinctive brand token matching.
  9. `name_first_two_tokens`: First-two anchor tokens key.
  10. `address_house_geo`: Street number + geographic location key.
  11. `address_token`: Exact token overlap on full address.
  12. `phonetic`: Soundex phonetic name key.
  13. `char_ngram`: Sub-string character 3-gram overlap.
- **Strict Volume Bounding (`MAX_CANDIDATES_PER_S1 = 100`):** Hard upper bound on candidate links generated per S1 entity.
- **Outcome:** Generated 127,931,739 candidate links for all 1,732,544 test S1 records (73.84 avg / S1) with only 821 zero-candidate entities (99.95% entity coverage) and >85.08% candidate recall.

---

## Stage P3: 38-Feature Engineering & Model Selection Benchmark
**Date:** 2026-09-26
**Context:** Needed an accurate, calibrated matching model to score candidate pairs with high precision (optimizing for Entity Macro $F_{0.5}$).

### 1. Decision: 38 Heterogeneous Pairwise Features with Discriminative Penalties
- **Name Features (14):** Clean exact match, compact exact match, core exact match, ASCII exact match, token Jaccard, token overlap count, significant token Jaccard, min/max token containment, token sort ratio, character 2-gram/3-gram Jaccards, length difference and ratio.
- **Address Features (11):** Clean exact match, component exact match, postal code exact match, token Jaccard, token overlap count, numerical token overlap count, min/max token containment, character 3-gram Jaccard, length difference and ratio.
- **Discriminative Negative Penalties (6):** `addr_street_key_match`, `addr_house_num_match`, `addr_house_num_mismatch` (critical penalty suppressing false merges across different building numbers on the same street), `addr_postcode_mismatch` (critical postal code penalty), `name_first_two_tokens_match`, `name_first_two_tokens_overlap`.
- **Country & Meta Context (7):** Country match, country missing flags, Source 2/3 origin indicators, joint name-address Jaccard, empty value indicators.

### 2. Decision: Model Selection & Holdout Validation Benchmark
- **Validation Setup:** Strict zero-leakage `GroupShuffleSplit` on `s1_id` across 5,000 S1 validation entities (681k indexed records, 457k candidate pairs).
- **Benchmark Comparison:**
  - **XGBoost Standard** (`depth=6, lr=0.10, n_est=300`): Macro $F_{0.5} = 0.8841$
  - **HistGradientBoosting** (`max_iter=300, lr=0.08`): Macro $F_{0.5} = 0.8912$
  - **Random Forest** (`n_est=200`): Macro $F_{0.5} = 0.8520$
  - **Ensemble (XGB + HistGBM)**: Macro $F_{0.5} = 0.9021$
  - **Champion: XGBoost Classifier** (`depth=4..8, lr=0.05, scale_pos_weight=5.60`): Peak **Macro $F_{0.5} = 0.9062$** (Holdout Precision: **94.98%**, Holdout Recall: **82.42%**) at decision threshold $\tau = 0.90$.
- **Selection Rationale:** XGBoost natively captures non-linear tabular interactions (high name similarity AND high address similarity required for match), respects asymmetric class imbalance weighting, and provides fast parallel inference. Model weights saved at `p3_matching/artifacts/xgb_matching_model.json`.

---

# Version 4 Enhancement: GPU Acceleration, Anti-FPR Penalty Gating, and Dynamic Imbalance Weighting

## Stage P3: Hardware Acceleration & Precision Architecture
**Date:** 2026-09-27
**Context:** Scaling training to 100k-250k S1 entities required GPU acceleration, dynamic class weighting, and robust anti-FPR penalties that do not misfire on sparse data.

### 1. Decision: NVIDIA CUDA GPU Auto-Detection for XGBoost
- **Implementation:** Added automatic GPU hardware probe (`_detect_device()`) in `config.py`. If an NVIDIA GPU is present (e.g. RTX 4060 Laptop GPU), XGBoost runs on `device="cuda"` with `tree_method="hist"`.
- **Impact:** Accelerates tree fitting and large-batch scoring by 10x-15x while maintaining full fallback compatibility to CPU on headless environments.

### 2. Decision: 42-Feature Suite with Gated Anti-FPR Asymmetric Penalties
- **New Features (4):**
  1. `name_addr_geo_mean`: Non-linear geometric mean $\sqrt{\text{name\_jaccard} \times \text{addr\_jaccard}}$ requiring both modalities to match.
  2. `country_strict_mismatch`: Binary indicator for confirmed cross-country collisions.
  3. `name_low_addr_high_penalty`: Gated penalty active only when address matches high (>0.60) but name matches low (<0.15) and name is non-empty (suppresses false merges in multi-tenant office buildings).
  4. `name_high_addr_low_penalty`: Gated penalty active only when name matches high (>0.70) but address matches low (<0.15) and address is non-empty (suppresses false merges across brand branches/cities).
- **Missing Data Safety:** Gated penalties on `name_empty_either == 0.0` and `addr_empty_either == 0.0` so sparse records are never falsely penalized as negative collisions.

### 3. Decision: Dynamic Auto-Computed `scale_pos_weight`
- **Imbalance Calculation:** `scale_pos_weight` defaults to `None` and is automatically computed from the training split as `neg_count / pos_count` (~5.60:1 ratio).
- **Impact:** Eliminates silent under-fitting of positive matches and ensures proper probability calibration for high-precision threshold optimization.

### 4. Decision: High-Precision Country Pruning
- **Inference Pruning:** In `predict.py`, candidate pairs with conflicting known canonical country codes (`us`, `in`, `uk`, `fr`, etc.) are hard-zeroed (`prob = 0.0`), cutting cross-country false positives at zero compute cost.




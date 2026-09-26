# Flow Documentation for P2 Candidate Generation (Baseline)

## Overview
The pipeline transforms raw source TSV files into a `candidate_pairs.tsv` file that lists, for each **Source‑1** entity, a set of plausible **Source‑2** / **Source‑3** candidate IDs.
It is **not** executed in this development environment – the code is only written.
The full run is triggered by a teammate with:

```bash
python code/run_p2.py \
    --source1 dataset/test/test_source1.tsv \
    --source2 dataset/test/test_source2.tsv \
    --source3 dataset/test/test_source3.tsv \
    --output output/candidate_pairs.tsv
```

Below is a step‑by‑step description of what happens when the command is executed.

---

## 1. Entry Point (`run_p2.py`)
```
main()
    ↓ parse_args()
    ↓ sanity‑check file existence
    ↓ generate_candidate_pairs(...)
```
* `argparse` parses the CLI arguments (`--source1`, `--source2`, `--source3`, `--output`, optional `--chunksize`).
* Minimal validation ensures each input path exists before proceeding.

---

## 2. Candidate Generation (`candidate_generation.generate_candidate_pairs`)
```
generate_candidate_pairs()
    ↓ build_candidate_indexes(source2_path, source3_path)
        ↓ Indexes.build_from_paths([source2_path, source3_path])
            for each TSV (S2, then S3 independently):
                initialize per-source seen_dedup_keys: Set[Tuple[name_compact, address_clean, country_clean]]
                for each pre-processed chunk via iter_preprocessed_file():
                    for each record in chunk (via fast column zip):
                        check dedup key (non-empty); if seen, increment duplicate_rows_dropped and skip
                        Indexes.add_record(entity_id, ntk, ncomp, ncore, atk, nphon)
    ↓ open output file, write header
    ↓ for each pre-processed chunk of source1 (iter_preprocessed_file(source1_path)):
        for each record in chunk (via fast column zip):
            s1_id = record["entity_id"]
            candidates = indexes.retrieve_candidates_for_row(record)
                ↓ retrieve_by_name_token(name_token_key)
                ↓ retrieve_by_name_compact(name_compact)
                ↓ retrieve_by_name_core(name_core)
                ↓ retrieve_by_address_token(address_token_key)
                ↓ retrieve_by_phonetic(name_phonetic_key)
                union of five candidate sets
            candidates.discard(s1_id)   # defensive self-match removal
            sorted_candidates = sorted(candidates)   # deterministic order
            write "s1_id\tcomma_separated_ids\n" to output
            track diagnostics: zero candidates, link counts, and method contributions
    ↓ return comprehensive diagnostics dictionary (stats, percentiles, method contributions, index counts)
```
* **Index construction**: `Indexes` holds five dictionaries mapping blocking keys → list of entity IDs (preserving `S2-`/`S3-` prefixes). Independent deduplication on `(name_compact, address_clean, country_clean)` ensures duplicate records within S2 or within S3 are removed, while identical records across S2 and S3 are both retained.
* **Chunked processing**: Both index building and S1 streaming use `iter_preprocessed_file`, which reads TSVs with `dtype=str, keep_default_na=True` in user-defined `chunksize` (default 100,000) and applies P1 normalizations on-the-fly.
* **Candidate retrieval**: For each S1 row, five look-ups are performed:
  1. `name_token_key` (exact token-sorted match)
  2. `name_compact` (whitespace-stripped match)
  3. `name_core` (trailing legal entity suffix stripped match)
  4. `address_token_key` (token-sorted address match)
  5. `name_phonetic_key` (token-sorted American Soundex match)
  The resulting ID sets are **unioned** to form `C(S1)`.
* **Deduplication & ordering**: Candidate set union guarantees deduplication. Candidate IDs are sorted alphabetically before writing for deterministic output across runs.
* **Output**: Standard TSV header (`source1_entity_id\tcandidate_entity_ids`), followed by one line per S1 entity. Empty candidate lists produce `s1_id\t\n`.

---

## 3. Data Flow Summary
```
Raw TSV (S2 / S3) ──► iter_preprocessed_file() ──► Indexes (5 dicts, S2/S3 deduped independently)
                                                             │
                                                             ▼
Raw TSV (S1) ──► iter_preprocessed_file() ──► fast column zip
                                                             │
                                                             ▼
indexes.retrieve_candidates_for_row(...) ──► 5-pass union set of candidate IDs
                                                             │
                                                             ▼
sorted, joined, written to candidate_pairs.tsv + diagnostics collected
```
* All three sources go through **the same preprocessing iterator**, guaranteeing identical normalized columns:
  - `name_clean`, `name_compact`, `name_token_key`, `name_core`, `name_phonetic_key`
  - `address_clean`, `address_token_key`
  - `country_clean` (deterministic canonical aliases)
  - `name_was_missing`, `address_was_missing`, `country_was_missing` (original raw nullity flags)
* Indexes are built **once** and kept in memory; they are queried **per-row** for S1 using fast vector zipped streaming.
* Memory usage is bounded by the five dictionaries of ID lists. No full DataFrames are retained after each chunk.

---

## 4. Where Memory Is Used
| Phase | Data Structure | Approx. Size (depends on dataset) |
|-------|----------------|-----------------------------------|
| Index build | `Indexes` 5 dictionaries (keys → list of IDs) | O(#unique keys + #records) – modest memory footprint for millions of records, easily fits within RAM. |
| Deduplication | `seen_dedup_keys` set of 3-tuples (cleared between S2 and S3) | Temporary per source during index build; freed after indexing. |
| S1 streaming | One chunk (default 100 k rows) as a pandas DataFrame | O(chunksize) – controlled by the `--chunksize` argument. |
| Candidate set per row | `set[str]` of candidate IDs | Typically a few dozen IDs; negligible. |
| S1 batch n-gram cache (P3) | `s1_cache` in `pair_features.py` | Cached only for unique S1 records in active batch (~2k records, < 1 MB). |

---

## 5. What Happens for an S1 Row with No Candidates
* All five look-ups return empty sets → `candidates` remains empty.
* After `discard(s1_id)` the set is still empty.
* `sorted_candidates` becomes an empty list; `cand_str` is `""`.
* The output line written is:
```
S1-123\t\n
```
* This satisfies the validator requirement that every S1 entity must appear exactly once.
* Recorded in diagnostics under `s1_rows_with_zero_candidates`.

---

## 6. Execution Note
* The code **does not** run automatically on full datasets in this environment. It is only written.
* A teammate can invoke the pipeline later with the CLI shown in section 1.
* No heavy-weight libraries (FAISS, vector databases, neural embeddings) are used; only Python standard library and pandas.

---

## 7. Extensibility & Diagnostics
* P2 provides detailed candidate generation statistics:
  - `total_s1_rows`
  - `s1_rows_with_zero_candidates`
  - `total_candidate_links`
  - `average_candidates_per_s1`, `median_candidates_per_s1`, `max_candidates_per_s1`
  - Per-method candidate link contributions (`name_token`, `name_compact`, `name_core`, `address_token`, `phonetic`)
  - S2 and S3 deduplication counts (`input_rows`, `duplicate_rows_dropped`)
* P3 consumes candidates and computes 28 features (including `name_core_exact`) with batch-level S1 n-gram caching.


---

*End of flow documentation.*


# P4 Integration Flow

## 8. P4 Entry Point (`run_p4.py`)
```bash
python code/run_p4.py \
    --scored-pairs output/candidate_pairs_scored.tsv \
    --test-source1 dataset/test/test_source1.tsv \
    --output output/matching_results.tsv \
    --threshold 0.50 \
    --candidate-pairs output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --validate
```
* Supports three modes via CLI flags: `--sweep` (tuning), `--error-analysis` (diagnostics), and default production mode (generating `matching_results.tsv`).
* Imports functions from `code/business_entity_resolution/src/p4_evaluation.py`.

---

## 9. P4 Threshold Tuning Flow (Training Data Only)
```
tune_threshold(scored_pairs_df, ground_truth)
    ↓ Phase 1: Coarse sweep
        for thr in 0.05 to 0.95 (step 0.01):
            evaluate_scored_pairs(..., threshold=thr)
            track macro F0.5
    ↓ Identify best coarse threshold
    ↓ Phase 2: Refined sweep
        for thr in [best_coarse - 0.05] to [best_coarse + 0.05] (step 0.005):
            evaluate_scored_pairs(..., threshold=thr)
    ↓ Return absolute best threshold and full results dataframe
```
* **Required Inputs:** `output/train_candidate_pairs_scored.tsv` (from P3 on training data) and `dataset/train/train_ground_truth.tsv`.
* Reuses `compute_entity_macro_f05` from P3's `evaluate.py` for metric consistency.
* Evaluates ~110 distinct thresholds to find the optimum F0.5 without requiring a full model retrain.

---

## 10. P4 Match Selection & TSV Generation Flow
```
select_matches(scored_pairs_df, threshold)
    ↓ filter DF where match_probability >= threshold
    ↓ group by source1_entity_id into sets of candidate_entity_ids

write_matching_results(matches, all_s1_ids, output_path)
    ↓ load_s1_entity_ids(test_source1.tsv)  # guarantees all entities are present
    ↓ write header
    ↓ for each s1_id in all_s1_ids:
        write s1_id \t comma_separated_sorted_matches
```
* **Output:** `output/matching_results.tsv` (the final scored submission file).
* **Validation:** Optionally runs `utils/validate_submission.py` as a subprocess to verify formatting, ID prefixes, and test set completeness before submission.

---

## 11. P4 Error Analysis Flow
```
error_analysis(scored_pairs_df, ground_truth, threshold)
    ↓ select_matches() using threshold
    ↓ for each S1 entity:
        compute True Positives (TP), False Positives (FP), False Negatives (FN)
    ↓ aggregate metrics:
        - Overall TP, FP, FN
        - S2 vs S3 source breakdown
        - Spurious matches on singletons
        - Candidate-recall issues (FN missing from P2 vs FN scored low by P3)
        - Missing-address FN (either record missing address)
        - Country-level breakdown (if records provided)
        - Top 10 entities with highest FP/FN counts
```
* Provides actionable insights for P2 (blocking recall) and P3 (model precision).

---

## 12. P4 Complete Data Flow Summary
```text
[P1: preprocessing]
       │
       ▼
[P2: candidate_pairs.tsv] ──► [P3: candidate_pairs_scored.tsv]
                                       │
                                       ▼
[test_source1.tsv] ─────────► (select_matches + threshold)
                                       │
                                       ▼
                            [write_matching_results]
                                       │
                                       ▼
                            output/matching_results.tsv ──► [validator]
```

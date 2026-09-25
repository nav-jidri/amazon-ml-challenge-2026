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

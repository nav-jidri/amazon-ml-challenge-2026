# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
*Provide a brief 2-3 sentence overview of your approach and key innovations.*

---

## 2. Methodology
### 2.1 Problem Analysis

The datasets contain noisy business entity information across multiple sources. The same business can appear with differences in capitalization, punctuation, word order, abbreviations, legal suffixes, spelling variations, multilingual text, and address formatting.

Data inspection also showed missing business names and addresses in some source records. The test data additionally contains France records, while the training data contains US and India records, so country handling was kept general rather than hard-coded to a fixed set of countries.

The preprocessing stage was designed to preserve the original data while creating normalized representations for downstream candidate generation and matching.

### 2.2 Solution Strategy
*Outline your high-level approach.*

**Approach Type:** [Blocking + Classifier / End-to-End / Graph-Based / Hybrid, etc]  
**Core Innovation:** [Brief description of your main technical contribution]

---

## 3. Candidate Generation (Blocking)
*Describe how you reduced the comparison space to a manageable candidate set.*

- **Blocking keys used:** [e.g., PIN code, phonetic name encoding, TF-IDF, etc.]
- **Candidate pairs generated:** [total]
- **How you ensured true matches were not lost:**

---

## 4. Matching Model

**Features used:**
- Name features: [e.g., Jaccard, Levenshtein, phonetic encoding]
- Address features: [e.g., token overlap, edit distance, PIN code matching]
- Other: []

**Model type:** [e.g., XGBoost, Siamese Network, Transformer, etc.]  
**Threshold selection method:** [e.g., F_0.5 optimization on validation set]

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [your best validation score]
- **Common false positives (wrong merges):** [brief description]
- **Common false negatives (missed matches):** [brief description]

---

## 6. Conclusion
*Summarize your approach, key achievements, and lessons learned in 2-3 sentences.*

---

## Appendix

### A. Code Artefacts
*Your complete, runnable code ships in the submission zip under
`code/business_entity_resolution/` (all source in `src/`, with a `README.md` and
`requirements.txt`). Summarise its structure and the entry point(s) to reproduce
`output/matching_results.tsv` and `output/candidate_pairs.tsv` here.*

### B. Additional Results
*Include any additional charts, graphs, or detailed results.*

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.

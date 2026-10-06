# Leakage Report

Train rows: 539  
Validation rows: 132  
Test rows: 139

| Pair | Exact overlap | Normalized overlap (not exact) | Semantic candidates (≥ 0.90) |
|---|---:|---:|---:|
| train-validate | 1 | 0 | 48 |
| train-test | 2 | 4 | 39 |
| validate-test | 1 | 0 | 15 |

## Definitions

- **Exact**: identical question string (after trimming).
- **Normalized**: identical after Unicode/HTML/case/punctuation/whitespace normalization.
- **Semantic**: question-embedding cosine similarity above the threshold (excluding exact/normalized matches).

## Action taken

- Exact + normalized overlaps of validate/test questions with train: **EXCLUDED from evaluation (EXCLUDE_LEAKED_EVAL_QUESTIONS=true). No file was modified.**
- Semantic candidates: **reported only**. In this benchmark every eval question is *supposed* to have an answering document in train, so a semantically similar train question is expected and is not, on its own, leakage. Review `leakage_semantic_candidates.csv` if you need a stricter policy.
- The vector indexes are built from train only; validate/test are never indexed.

### train-validate overlapping validate question_ids

- exact: `validate-29b3b3b49f58-0`

### train-test overlapping test question_ids

- exact: `test-29b3b3b49f58-0`
- exact: `test-65160e8e43c2-0`
- normalized: `test-073600e40913-0`
- normalized: `test-0769b7b62cfb-0`
- normalized: `test-510d4e38c12e-0`
- normalized: `test-56e11b503634-0`

### validate-test overlapping test question_ids

- exact: `test-29b3b3b49f58-0`

## Top semantic candidates

| Pair | Eval question | Nearest question | Similarity |
|---|---|---|---:|
| train-validate | What are the symptoms of iron-deficiency anemia? | What are the signs of iron-deficiency anemia? | 0.986 |
| train-test | What are the signs of celiac disease? | What are the symptoms of celiac disease? | 0.983 |
| train-validate | What are the signs of type 1 diabetes? | What are the symptoms of type 1 diabetes? | 0.980 |
| train-validate | What are the signs of rheumatoid arthritis? | What are the symptoms of rheumatoid arthritis? | 0.979 |
| train-test | What are the signs of hyperthyroidism? | What are the symptoms of hyperthyroidism? | 0.979 |
| validate-test | What are the signs of coronary artery disease? | What are the symptoms of coronary artery disease? | 0.979 |
| train-validate | What are the signs of psoriasis? | What are the symptoms of psoriasis? | 0.978 |
| train-test | What are the signs of hypothyroidism? | What are the symptoms of hypothyroidism? | 0.976 |
| train-validate | What are the symptoms of osteoporosis? | What are the signs of osteoporosis? | 0.976 |
| train-test | What are the signs of generalized anxiety disorder? | What are the symptoms of generalized anxiety disorder? | 0.974 |

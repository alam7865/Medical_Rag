# Medical RAG: Raw Data vs Cleaned Data

A retrieval experiment that answers one question: **does cleaning the knowledge corpus improve RAG retrieval?**

Two retrieval pipelines are built from the same `train.csv`. One indexes the data as-is (**Raw RAG**) and the other indexes a cleaned copy (**Cleaned RAG**). Both are scored on the same held-out questions with the same metrics. The two pipelines share every setting, so the corpus is the only thing that differs.

> ⚠️ **The data shipped in `data/raw/` is synthetic** (`scripts/generate_synthetic_data.py`), generated to make the project runnable and to exercise every cleaning path. It is educational placeholder text, **not medical advice**. Drop your own `train.csv / validate.csv / test.csv` in its place and re-run (see [Using your own data](#using-your-own-data)).

---

## 1. Problem statement

Real medical QA corpora are messy. Scraped records carry HTML, site boilerplate, Unicode artefacts, empty or placeholder answers and many duplicates. People usually assume cleaning helps RAG, but it should be measured. This project does that with a controlled experiment, and it also explains *why* the metrics change.

## 2. Architecture

```text
                         ┌───────────────┐
                         │  Medical Data │
                         │ train/val/test│
                         └───────┬───────┘
                                 │  (train only → corpus)
                    ┌────────────┴────────────┐
                    ▼                         ▼
              ┌───────────┐             ┌───────────┐
              │ Raw Data  │             │Clean Data │  ← audit + cleaning + dedup
              └─────┬─────┘             └─────┬─────┘
                    ▼                         ▼
               Chunking                  Chunking        same CHUNK_SIZE / OVERLAP
                    ▼                         ▼
              Embeddings                 Embeddings      same model (all-MiniLM-L6-v2)
                    ▼                         ▼
          ChromaDB medical_rag_raw   ChromaDB medical_rag_cleaned   (cosine)
                    └──────────┬──────────────┘
                               ▼
                 Same eval questions (validate / test)
                               ▼
                   Retrieval (TOP_K, unique records)
                               ▼
              Recall@K · Precision@K · MRR · NDCG@K
                               ▼
              Raw vs Cleaned comparison + error analysis
```

| Layer | Code |
|---|---|
| Config (single source of all settings) | `src/config.py`, `.env.example` |
| Load + schema auto-detection + stable IDs | `src/data/loader.py` |
| Normalization (cleaning vs matching) | `src/data/normalization.py` |
| Cleaning + audit log | `src/data/cleaner.py` |
| Duplicate detection | `src/data/deduplicator.py` |
| Leakage detection | `src/data/leakage.py` |
| Embeddings | `src/embeddings/embedding_model.py` |
| Vector store | `src/vectorstore/chroma_store.py` |
| Chunking / retrieval / index build | `src/rag/chunker.py`, `retriever.py`, `pipeline.py` |
| Metrics / evaluator / error analysis | `src/evaluation/` |
| Optional API | `src/api/app.py` |

## 3. Dataset

| Split | Role | Rows (synthetic) |
|---|---|---:|
| `train.csv` | **Knowledge corpus** (the only data that is indexed) | 539 |
| `validate.csv` | Development / tuning | 132 |
| `test.csv` | Final, untouched evaluation | 139 |

The synthetic set covers 25 conditions × 5 aspects (symptoms, causes, diagnosis, treatment, prevention), plus 6 long "overview" records that need several chunks. Each fact has 5 question phrasings: one goes to test, one to validate and three to train. Eval questions are therefore paraphrases (often lay terms like *"brittle bones"*) that never appear verbatim in train.

Train noise injected on purpose: exact and re-formatted duplicates, HTML tags/entities, web boilerplate, NBSP / zero-width / full-width characters, empty-answer rows whose questions look real, null questions, junk rows (`test`, `N/A`), too-short answers, conflicting answers, plus a few deliberate train→test overlaps for the leakage check to find.

## 4. Raw RAG

`train.csv` → load (missing values become `""`) → chunk → embed → `medical_rag_raw`. There is no cleaning: duplicates, markup, boilerplate and empty answers all stay. Only rows with no text at all are skipped, because there is nothing to embed.

## 5. Cleaned RAG

`train.csv` → `clean_dataset.py` → `data/processed/cleaned_train.csv` → the **same** chunk → embed → `medical_rag_cleaned`.

## 6. Why no training is required

This is a **retrieval evaluation**. The embedding model is a pre-trained Sentence Transformers model used as-is, and there is no LLM. Training anything would add a second variable, which would make it impossible to attribute changes to the data. Neither the LLM nor the embedding model is trained or fine-tuned, and no API keys are needed.

## 7. Data cleaning

| Step | What it does | Action |
|---|---|---|
| Invalid rows | null / empty question or answer (including "empty after removing markup"), placeholders (`test`, `N/A`, `???`), no letters, below `MIN_QUESTION_CHARS` / `MIN_ANSWER_CHARS` | removed, reason logged |
| Normalization | Unicode NFKC, zero-width/control chars, smart quotes → ASCII, HTML tags + entities, known web boilerplate, whitespace | applied, operations logged per row |
| Exact duplicate Q+A | identical raw question and answer | removed (first kept) |
| Normalized duplicate Q+A | identical after case/punctuation/markup normalization | removed (first kept) |
| Same question, different answer | possibly complementary, or outdated | **kept**, flagged |
| Semantic near-duplicate questions | cosine ≥ `SEMANTIC_DUP_THRESHOLD` | **kept**, flagged (type 1 vs type 2 diabetes look alike but are different) |

Guard-rails against over-cleaning: punctuation, numbers, units (`mg/dL`, `130/80 mmHg`), `<7` thresholds and medical terms are preserved, and lowercasing is **off** by default because case can carry meaning (`HbA1c`, `IgA`). Every input row appears in `reports/cleaning_log.csv` with `kept / kept_flagged / removed`, the reason and `duplicate_of`. Nothing is removed silently, and the raw file is never modified.

**Result on the synthetic data:** 539 → **399** rows (46 invalid, 65 exact duplicates, 29 normalized duplicates). 12 conflicting-answer rows and 9 semantic near-duplicate pairs were flagged and kept. Details are in `reports/data_cleaning_report.md` / `.json` / `.csv`.

## 8. Leakage detection

`audit_dataset.py` compares train↔validate, train↔test and validate↔test at three levels: **exact**, **normalized** and **semantic** (embedding cosine ≥ `SEMANTIC_LEAK_THRESHOLD`).

| Pair | Exact | Normalized | Semantic candidates (≥ 0.90) |
|---|---:|---:|---:|
| train-validate | 1 | 0 | 48 |
| train-test | 2 | 4 | 39 |
| validate-test | 1 | 0 | 15 |

Actions taken:
- **Exact and normalized overlaps** of eval questions with train are *excluded from evaluation* (`EXCLUDE_LEAKED_EVAL_QUESTIONS=true`). No file is modified, and the count appears in every report.
- **Semantic candidates are reported only.** In this benchmark every eval question is *meant* to have an answering document in train, so a similar train question is expected and is not leakage on its own terms.
- Indexes are built from train only. Validate and test are never indexed, and test is never used for tuning.

## 9. Embeddings

`sentence-transformers/all-MiniLM-L6-v2` by default (set `EMBEDDING_MODEL` to change it). Embeddings are L2-normalized, so dot product equals cosine similarity. We compute the embeddings ourselves and pass them to Chroma, which guarantees both collections use the identical model. The model name is stored in the collection metadata, and the retriever refuses to query an index that was built with a different model.

## 10. ChromaDB

There are two persistent collections, `medical_rag_raw` (`chroma/raw/`) and `medical_rag_cleaned` (`chroma/cleaned/`), both using cosine HNSW. Chunk metadata holds `question_id, chunk_index, n_chunks, dataset_split, source, question, answer, answer_key, pipeline`. Collection metadata stores the experiment settings and the dataset version (a content hash of the CSV).

## 11. Retrieval and chunking

A record is rendered as `Question: …\nAnswer: …`. If it fits in `CHUNK_SIZE` (1000 chars), it becomes one chunk. Otherwise the answer is split into overlapping windows (`CHUNK_OVERLAP` = 200) at sentence or word boundaries, and each chunk is prefixed with the question. At query time we fetch `TOP_K × CHUNK_FETCH_MULTIPLIER` chunks and collapse them to unique records, keeping each record's best chunk. All metrics are computed at **record level**.

## 12. Evaluation metrics

**Ground truth:** a corpus record is relevant to an eval question when its answer has the same `answer_key`, a hash of the answer after aggressive normalization (case, punctuation, markup, Unicode and boilerplate are ignored). This needs no hand-labelled relevance file, works on any QA dataset, and is applied identically to both corpora. Eval questions with no matching answer in the raw corpus are excluded from both runs and counted in the report.

| Metric | Meaning |
|---|---|
| **Recall@K** (1, 3, 5, 10) | a correct record appears in the top K (hit rate) |
| **Precision@K** (1, 3, 5, 10) | fraction of the top K that is correct |
| **MRR** | mean of 1 / rank of the first correct record |
| **NDCG@K** (5, 10) | ranking quality vs. the ideal ranking for that corpus |
| **Semantic Similarity** (optional) | cosine between the expected answer and the retrieved answer embeddings. This is **not** accuracy. |

## 13. Results (synthetic data, actual measured run)

Test split, 131 questions evaluated (139 rows − 1 empty − 6 leaked − 1 answer not in corpus). Same 131 questions for both pipelines:

| Metric | Raw RAG | Cleaned RAG | Improvement |
|---|---:|---:|---:|
| Recall@1 | 0.4885 | 0.6718 | +37.5% |
| Recall@3 | 0.8168 | 0.8779 | +7.5% |
| Recall@5 | 0.9008 | 0.9313 | +3.4% |
| Recall@10 | 0.9771 | 0.9847 | +0.8% |
| Precision@5 | 0.4183 | 0.3878 | **−7.3%** |
| MRR | 0.6653 | 0.7845 | +17.9% |
| NDCG@5 | 0.5573 | 0.6541 | +17.4% |
| NDCG@10 | 0.6549 | 0.7521 | +14.8% |
| Semantic Similarity (top-1) | 0.6570 | 0.8955 | +36.3% |

Improvement % = (Cleaned − Raw) / Raw × 100 (reported as n/a when Raw = 0). The validation split shows the same pattern (Recall@1 0.550 → 0.687, MRR 0.698 → 0.800).

**Reading the results**
- Cleaning helps most **at the top of the ranking** (Recall@1, MRR, NDCG). At K=10 both pipelines usually find the answer somewhere, so the gap closes.
- **Precision@5 and @10 get worse.** In the raw corpus up to 8 records share the same answer (3 question phrasings plus duplicates), against at most 3 after cleaning. Every copy counts as relevant, so raw can fill the list with duplicates. After de-duplication there are fewer correct copies left to retrieve. The user gets no extra information from duplicates, which is why Precision@K has to be read alongside Recall and MRR.
- Semantic Similarity (top-1) rises sharply because the raw top-1 is often an empty or "See a doctor." record.

These numbers come from **synthetic** data with injected noise. They show the method works but say nothing about real-world effect sizes. Re-run on your own data.

## 14. Error analysis

Each question falls into one of four buckets: *raw fail / cleaned success*, *raw success / cleaned fail*, *both failed* or *both succeeded*. The default success definition is a correct document at rank 1, and the report also shows the buckets at K=5. For every raw ranking, the documents above the correct answer are looked up in the cleaning log.

Test split at K=1: **26** raw-fail/clean-success, **2** raw-success/clean-fail, 41 both failed, 62 both succeeded. For **24 of the 26** cleaned-only wins, the raw ranking had at least one record that cleaning removed (empty answer, too-short answer, junk) ranked above the correct answer. That is the concrete mechanism by which noise hurts. About two-thirds of the 41 "both failed" questions use lay terms (*"adult-onset diabetes"*, *"overactive thyroid"*, *"low iron"*) that the small embedding model struggles to link to the clinical name. That points to a model limitation, which cleaning cannot fix.

See `reports/error_analysis.md` for examples and `reports/evaluation_results.csv` for per-question side-by-side data.

## 15. How to run

```bash
cd medical-rag
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # optional - defaults work

# (only if data/raw is empty) generate the synthetic placeholder data
python scripts/generate_synthetic_data.py

python scripts/audit_dataset.py          # data audit + leakage report
python scripts/clean_dataset.py          # cleaned_train.csv + cleaning reports
python scripts/build_raw_index.py        # Chroma: medical_rag_raw
python scripts/build_clean_index.py      # Chroma: medical_rag_cleaned
python scripts/evaluate_raw.py           # --split validate for development
python scripts/evaluate_cleaned.py       # default --split test (final)
python scripts/compare_results.py        # comparison, error analysis, figures

# or everything at once
./run_all.sh test                        # PYTHON=.venv/bin/python ./run_all.sh

pytest                                   # 58 unit tests
uvicorn src.api.app:app --reload         # web UI → http://127.0.0.1:8000/  (API docs at /docs)
```

**Recommended workflow:** iterate on cleaning or chunking settings with `--split validate`. Run `--split test` once at the end.

### Web UI

```bash
uvicorn src.api.app:app --reload   # then open http://127.0.0.1:8000/
```

A single-page frontend (`src/api/static/index.html`, plain HTML/CSS/JS with no build step), served by the API:

- **Side-by-side comparison**: Raw and Cleaned results for the same question, with similarity score bars.
- **Noise badges on raw results**: `Empty answer`, `Duplicate of #N`, and `Removed by cleaning: <reason>`, looked up in `reports/cleaning_log.csv`.
- **Summary tiles**: useful results, junk records, duplicates, and whether the top-1 result is junk.
- **Offline evaluation dashboard**: Raw vs Cleaned metric bars, the cleaning funnel, and the error-analysis summary.
- Top-K switch (3/5/10), example questions drawn from the *validation* split (never test), shareable links (`/?q=...&k=5`), light/dark theme, and a mobile layout. Press `/` to focus the search box.

### Live demo on Netlify (static, no server)

The same UI also runs **entirely in the browser**, so it can be hosted on Netlify for free:

- `scripts/export_static_site.py` exports both ChromaDB indexes (chunk embeddings + records, ~2.4 MB), the metrics and the example questions into `web/`.
- In the browser, [transformers.js](https://github.com/huggingface/transformers.js) embeds the question with `Xenova/all-MiniLM-L6-v2`, the ONNX build of the same model. `rag-engine.js` then does exact cosine search and the same chunk-to-record collapsing as `retriever.py`.
- **Parity check:** on all 131 test questions the fp32 in-browser model gives the same top-1 as Python 131/131 times, with identical Recall@1 and MRR for both pipelines. The first visit downloads the model once (~90 MB, then cached). For a ~23 MB download, set `"js_dtype": "q8"` in `export_static_site.py`, at the cost of ~1% metric drift.
- The page finds out which mode it's in by fetching `data/manifest.json`, which `{"mode": "static"}` on Netlify and `{"mode": "api"}` from FastAPI.

Deploy:
```bash
python scripts/export_static_site.py     # regenerate web/ after any pipeline change
git add web && git commit -m "Update static site" && git push
```
In Netlify: **Add new site → Import an existing project → GitHub → this repo**. `netlify.toml` already sets the publish directory to `web/` and needs no build command, so each push redeploys. Test locally with `python -m http.server -d web 8899`.

### API

| Endpoint | Body | Returns |
|---|---|---|
| `GET /` | — | the web UI |
| `GET /health` | — | status, model, chunk counts per index |
| `GET /metrics` | — | evaluation summary + cleaning report (for the dashboard) |
| `GET /examples?n=6` | — | sample questions from the validation split |
| `GET /data/manifest.json` | — | `{"mode": "api"}` (tells the shared UI it is talking to the server) |
| `POST /retrieve/raw` | `{"question": "...", "top_k": 5}` | ranked raw results |
| `POST /retrieve/cleaned` | same | ranked cleaned results |
| `POST /compare` | same | `raw_results`, `cleaned_results`, `comparison` (top-1 scores, empty/duplicate answers in raw, shared IDs) |

### Outputs

```text
data/processed/cleaned_train.csv
reports/data_audit.md
reports/leakage_report.md|json, leakage_semantic_candidates.csv
reports/data_cleaning_report.md|json|csv, cleaning_log.csv, conflicting_questions.csv, semantic_duplicates.csv
reports/results/{raw,cleaned}_{validate,test}_{per_question.csv,summary.json}, index_{raw,cleaned}.json
reports/evaluation_results.csv, evaluation_summary.json, evaluation_report.md, error_analysis.md
reports/figures/recall_at_k.png, precision_at_k.png, mrr.png, ndcg.png
```

`evaluation_report.md`, `evaluation_results.csv` and the figures describe whichever split `compare_results.py` last ran on. The per-split files in `reports/results/` are kept for both splits.

## Using your own data

1. Put `train.csv`, `validate.csv` and `test.csv` in `data/raw/`, replacing the synthetic files.
2. Columns are auto-detected (`question/Question/query/input/prompt…`, `answer/Answer/response/output…`, optional `id/qid/question_id`, `source`). If yours differ, set `QUESTION_COLUMN`, `ANSWER_COLUMN`, `ID_COLUMN` and `SOURCE_COLUMN` in `.env`. Without an ID column, IDs are generated deterministically from content (`<split>-<sha1(q‖a)>-<n>`), so they do not depend on row position.
3. Run `python scripts/audit_dataset.py` first and read `reports/data_audit.md` to check that the detected schema, sample records and quality profile look right.
4. Check `reports/data_cleaning_report.md`. Tune `MIN_*_CHARS` or the boilerplate patterns in `src/data/normalization.py` for your source, **using the validation split only**.
5. Run the rest of the pipeline.

Note on ground truth: relevance is defined by answer matching, which assumes eval answers also exist in train (as in datasets where each answer has several question phrasings). If your test answers never occur in train, the eval set will be empty and the report will say so. In that case you need a relevance mapping, such as a shared document or answer ID column. `answer_key` in `src/rag/pipeline.py` is the single place to change.

## Reproducibility

Every summary JSON records `embedding_model, chunk_size, chunk_overlap, top_k, chunk_fetch_multiplier, distance_metric, lowercase_documents, eval K values, leakage policy` and content hashes of the corpus and eval files. `compare_results.py` **refuses to compare** runs whose settings or eval file differ. Synthetic data generation is seeded (`RANDOM_SEED`), and reruns produce identical metrics.

## Next step (optional): generation

The retriever is self-contained. A generation layer (`Retriever → context → LLM → answer`) can be added on top later. It is intentionally not a dependency of the retrieval evaluation.

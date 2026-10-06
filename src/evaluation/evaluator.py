"""Retrieval evaluation.

Ground truth
------------
An eval question's ground truth is its answer. A corpus document is relevant
if its answer has the same `answer_key` (canonical answer text: case-,
punctuation-, markup- and Unicode-insensitive). This works for any QA dataset
without a hand-made relevance file, and it is applied identically to both
corpora. Because duplicates of the same answer all share one key, a corpus may
contain several relevant documents for one question.

Eval set
--------
`build_eval_set` produces the SAME question list for both pipelines:
  - drops eval rows with an empty question or answer (cannot be evaluated)
  - optionally excludes questions that leak from train (exact/normalized)
  - excludes questions whose answer does not exist in the RAW train corpus
    (unanswerable by construction; the raw corpus is the superset)
Every exclusion is counted and reported.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import Config, get_config
from src.data.deduplicator import Embedder
from src.data.leakage import leaked_eval_ids
from src.data.loader import load_cleaned_train, load_split
from src.evaluation import metrics as M
from src.rag.pipeline import answer_key, get_retriever
from src.utils.helpers import file_fingerprint, get_logger, save_json

logger = get_logger(__name__)


@dataclass
class EvalSet:
    split: str
    questions: pd.DataFrame  # question_id, question, answer, answer_key
    exclusions: dict[str, int] = field(default_factory=dict)


def corpus_for(variant: str, config: Config) -> pd.DataFrame:
    return load_split("train", config) if variant == "raw" else load_cleaned_train(config)


def build_eval_set(split: str, config: Config | None = None) -> EvalSet:
    if split not in ("validate", "test"):
        raise ValueError("Evaluation split must be 'validate' or 'test' (train is the corpus).")
    config = config or get_config()
    eval_df = load_split(split, config)
    raw_train = load_split("train", config)

    exclusions: dict[str, int] = {"total_rows": len(eval_df)}
    eval_df = eval_df.assign(question=eval_df["question"].str.strip(), answer_key=eval_df["answer"].map(answer_key))

    unusable = (eval_df["question"] == "") | (eval_df["answer_key"] == "")
    exclusions["empty_question_or_answer"] = int(unusable.sum())
    eval_df = eval_df[~unusable]

    leaked = leaked_eval_ids(raw_train, eval_df)
    exclusions["leaked_from_train_exact_or_normalized"] = len(leaked)
    if config.exclude_leaked_eval_questions:
        eval_df = eval_df[~eval_df["question_id"].isin(leaked)]
    else:
        exclusions["leaked_from_train_exact_or_normalized"] = 0  # reported but kept

    raw_keys = set(raw_train["answer"].map(answer_key)) - {""}
    no_answer = ~eval_df["answer_key"].isin(raw_keys)
    exclusions["answer_not_in_corpus"] = int(no_answer.sum())
    eval_df = eval_df[~no_answer]

    exclusions["evaluated"] = len(eval_df)
    logger.info("Eval set (%s): %s", split, exclusions)
    return EvalSet(split, eval_df[["question_id", "question", "answer", "answer_key"]].reset_index(drop=True), exclusions)


def _semantic_similarity(embedder: Embedder, expected: list[str], retrieved: list[list[str]], k: int) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity between expected answer and (top-1, best of top-k) retrieved answers."""
    exp = embedder.encode(expected)
    flat = [a for docs in retrieved for a in (docs[:k] or [""])]
    ret = embedder.encode(flat)
    top1 = np.zeros(len(expected))
    best = np.zeros(len(expected))
    pos = 0
    for i, docs in enumerate(retrieved):
        n = max(1, min(k, len(docs)))
        sims = ret[pos : pos + n] @ exp[i]
        top1[i], best[i] = sims[0], sims.max()
        pos += n
    return top1, best


def evaluate(variant: str, split: str, embedder: Embedder, config: Config | None = None, semantic: bool = True) -> dict:
    config = config or get_config()
    eval_set = build_eval_set(split, config)
    q = eval_set.questions

    corpus = corpus_for(variant, config)
    relevant_counts = Counter(k for k in corpus["answer"].map(answer_key) if k)

    retriever = get_retriever(variant, embedder, config)
    results = retriever.retrieve_batch(q["question"].tolist(), config.top_k)

    rows = []
    for (_, row), docs in zip(q.iterrows(), results):
        rel = [1 if d.answer_key == row["answer_key"] else 0 for d in docs]
        n_rel = relevant_counts.get(row["answer_key"], 0)
        rec: dict = {
            "question_id": row["question_id"],
            "question": row["question"],
            "ground_truth": row["answer"],
            "n_relevant_in_corpus": n_rel,
            "rank": M.first_relevant_rank(rel) or "",
            "reciprocal_rank": M.reciprocal_rank(rel),
        }
        for k in config.eval_k_values:
            rec[f"recall@{k}"] = M.recall_at_k(rel, k)
            rec[f"precision@{k}"] = M.precision_at_k(rel, k)
        for k in config.ndcg_k_values:
            rec[f"ndcg@{k}"] = M.ndcg_at_k(rel, k, n_rel)
        for k in (1, 3, 5):
            rec[f"top_{k}_ids"] = "|".join(d.question_id for d in docs[:k])
        rec["all_ids"] = "|".join(d.question_id for d in docs)
        rec["top_1_question"] = docs[0].question if docs else ""
        rec["top_1_answer"] = docs[0].answer if docs else ""
        rec["top_1_score"] = docs[0].score if docs else None
        rec["top_5_relevance"] = "".join(str(r) for r in rel[:5])
        rows.append(rec)
    per_q = pd.DataFrame(rows)

    if semantic and len(per_q):
        top1, best5 = _semantic_similarity(
            embedder, q["answer"].tolist(), [[d.answer for d in docs] for docs in results], 5
        )
        per_q["semantic_similarity_top1"] = top1.round(4)
        per_q["semantic_similarity_best_top5"] = best5.round(4)

    metric_cols = (
        [f"recall@{k}" for k in config.eval_k_values]
        + [f"precision@{k}" for k in config.eval_k_values]
        + ["reciprocal_rank"]
        + [f"ndcg@{k}" for k in config.ndcg_k_values]
    )
    if semantic:
        metric_cols += ["semantic_similarity_top1", "semantic_similarity_best_top5"]
    aggregate = {("mrr" if c == "reciprocal_rank" else c): round(float(per_q[c].mean()), 6) for c in metric_cols}

    corpus_path = config.split_path("train") if variant == "raw" else config.cleaned_train_path
    summary = {
        "pipeline": variant,
        "split": split,
        "n_questions": int(len(per_q)),
        "metrics": aggregate,
        "eval_set": eval_set.exclusions,
        "corpus_records": int(len(corpus)),
        "index_chunks": retriever.store.count(),
        "dataset_version": {
            "corpus": file_fingerprint(corpus_path),
            "eval": file_fingerprint(config.split_path(split)),
        },
        "experiment_settings": config.experiment_settings(),
    }

    out_dir = config.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    per_q.to_csv(out_dir / f"{variant}_{split}_per_question.csv", index=False)
    save_json(summary, out_dir / f"{variant}_{split}_summary.json")
    logger.info("[%s/%s] %s", variant, split, aggregate)
    return summary

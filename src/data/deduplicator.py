"""Duplicate detection.

Policy (see README "Data cleaning"):

| Type                                         | Action  |
|----------------------------------------------|---------|
| Exact duplicate question+answer              | removed |
| Normalized duplicate question+answer         | removed |
| Same (normalized) question, different answer | kept, flagged as conflicting |
| Semantic near-duplicate questions            | kept, flagged for review |

Semantic duplicates are never removed automatically: "What is type 1
diabetes?" and "What is type 2 diabetes?" are near-identical in embedding
space but medically distinct.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from src.data.normalization import matching_key


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


def first_occurrence_map(keys: pd.Series) -> pd.Series:
    """For each row, the index label of the first row with the same key, or None.

    The first occurrence itself maps to None (it is the one that is kept).
    """
    first: dict[str, object] = {}
    out: list[object] = []
    for idx, key in keys.items():
        if key in first:
            out.append(first[key])
        else:
            first[key] = idx
            out.append(None)
    return pd.Series(out, index=keys.index, dtype=object)


def exact_qa_key(df: pd.DataFrame) -> pd.Series:
    return df["question"].astype(str) + "␟" + df["answer"].astype(str)


def normalized_qa_key(df: pd.DataFrame) -> pd.Series:
    return df["question"].map(matching_key) + "␟" + df["answer"].map(matching_key)


def count_duplicated(keys: pd.Series) -> int:
    """Number of rows that are a repeat of an earlier row (first copy not counted)."""
    return int(keys.duplicated(keep="first").sum())


def find_conflicting_questions(df: pd.DataFrame) -> pd.DataFrame:
    """Groups of rows sharing a normalized question but having different answers."""
    q_key = df["question"].map(matching_key)
    a_key = df["answer"].map(matching_key)
    tmp = pd.DataFrame({"q_key": q_key, "a_key": a_key, "question_id": df["question_id"]})
    n_answers = tmp.groupby("q_key")["a_key"].transform("nunique")
    conflicting = tmp[n_answers > 1]
    return conflicting.sort_values("q_key")


def max_similarity(
    a: np.ndarray, b: np.ndarray, exclude_self: bool = False, batch_size: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    """For each row in `a`, the most similar row in `b` (cosine, embeddings L2-normalized).

    Computed in batches so memory stays O(batch_size * len(b)).
    Returns (best_index, best_score).
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), dtype=int), np.zeros(len(a), dtype=float)
    best_idx = np.empty(len(a), dtype=int)
    best_score = np.empty(len(a), dtype=float)
    for start in range(0, len(a), batch_size):
        sims = a[start : start + batch_size] @ b.T
        if exclude_self:
            rows = np.arange(sims.shape[0])
            sims[rows, rows + start] = -np.inf
        best_idx[start : start + batch_size] = sims.argmax(axis=1)
        best_score[start : start + batch_size] = sims.max(axis=1)
    return best_idx, best_score


def find_semantic_duplicates(
    df: pd.DataFrame, embedder: Embedder, threshold: float
) -> pd.DataFrame:
    """Pairs of rows whose questions are semantically near-identical (flag only).

    Pairs that are already normalized duplicates are skipped - they are
    handled by the deterministic rules.
    """
    if len(df) < 2:
        return pd.DataFrame(columns=["question_id", "similar_to", "similarity", "question", "similar_question"])
    questions = df["question"].tolist()
    emb = embedder.encode(questions)
    idx, score = max_similarity(emb, emb, exclude_self=True)
    keys = df["question"].map(matching_key).tolist()
    ids = df["question_id"].tolist()
    rows = []
    seen_pairs: set[tuple[str, str]] = set()
    for i, (j, s) in enumerate(zip(idx, score)):
        if s < threshold or keys[i] == keys[j]:
            continue
        pair = tuple(sorted((ids[i], ids[j])))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        rows.append(
            {
                "question_id": ids[i],
                "similar_to": ids[j],
                "similarity": round(float(s), 4),
                "question": questions[i],
                "similar_question": questions[j],
            }
        )
    return pd.DataFrame(rows)

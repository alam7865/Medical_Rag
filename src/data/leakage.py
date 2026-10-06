"""Leakage detection between train / validate / test.

Three levels, each reported separately (a pair is counted at the strictest
level it matches):

* exact      - identical question string (after trimming)
* normalized - identical `matching_key` (case/punctuation/markup-insensitive)
* semantic   - cosine similarity of question embeddings >= threshold

Action policy (also written into the report):
* exact + normalized overlaps of eval questions with train are EXCLUDED from
  evaluation when `EXCLUDE_LEAKED_EVAL_QUESTIONS=true` (the default) - the
  answer would otherwise be found by trivial string matching. They are never
  deleted from any file.
* semantic candidates are reported only. In a retrieval benchmark the corpus is
  *supposed* to contain a document that answers each eval question, so a
  similar train question is expected and is not, by itself, leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.deduplicator import Embedder, max_similarity
from src.data.normalization import matching_key


@dataclass
class OverlapResult:
    left: str
    right: str
    exact_ids: set[str]
    normalized_ids: set[str]
    semantic: pd.DataFrame  # right-side rows with their nearest left-side question

    def summary(self) -> dict:
        return {
            "pair": f"{self.left}-{self.right}",
            "exact_overlap": len(self.exact_ids),
            "normalized_overlap": len(self.normalized_ids),
            "semantic_candidates": int(len(self.semantic)),
        }


def find_overlap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
    embedder: Embedder | None = None,
    semantic_threshold: float = 0.9,
) -> OverlapResult:
    """Which rows of `right` (usually an eval split) overlap with `left` (usually train)."""
    left_exact = set(left["question"].str.strip()) - {""}
    left_norm = set(left["question"].map(matching_key)) - {""}

    exact_ids: set[str] = set()
    norm_ids: set[str] = set()
    for qid, q in zip(right["question_id"], right["question"]):
        q_strip = q.strip()
        if not q_strip:
            continue
        if q_strip in left_exact:
            exact_ids.add(qid)
        elif matching_key(q) in left_norm:
            norm_ids.add(qid)

    semantic = pd.DataFrame(columns=["question_id", "question", "nearest_left_id", "nearest_left_question", "similarity"])
    if embedder is not None:
        l = left[left["question"].str.strip() != ""]
        r = right[(right["question"].str.strip() != "") & ~right["question_id"].isin(exact_ids | norm_ids)]
        if len(l) and len(r):
            l_emb = embedder.encode(l["question"].tolist())
            r_emb = embedder.encode(r["question"].tolist())
            idx, score = max_similarity(r_emb, l_emb)
            mask = score >= semantic_threshold
            semantic = pd.DataFrame(
                {
                    "question_id": r["question_id"].to_numpy()[mask],
                    "question": r["question"].to_numpy()[mask],
                    "nearest_left_id": l["question_id"].to_numpy()[idx[mask]],
                    "nearest_left_question": l["question"].to_numpy()[idx[mask]],
                    "similarity": score[mask].round(4),
                }
            ).sort_values("similarity", ascending=False)

    return OverlapResult(left_name, right_name, exact_ids, norm_ids, semantic)


def leakage_report(
    splits: dict[str, pd.DataFrame], embedder: Embedder | None, semantic_threshold: float
) -> dict[str, OverlapResult]:
    pairs = [("train", "validate"), ("train", "test"), ("validate", "test")]
    return {
        f"{a}-{b}": find_overlap(splits[a], splits[b], a, b, embedder, semantic_threshold)
        for a, b in pairs
        if a in splits and b in splits
    }


def leaked_eval_ids(train: pd.DataFrame, eval_df: pd.DataFrame) -> set[str]:
    """Eval question_ids whose question appears in train (exact or normalized)."""
    res = find_overlap(train, eval_df, "train", "eval")
    return res.exact_ids | res.normalized_ids

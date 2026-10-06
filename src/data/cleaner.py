"""Cleaning pipeline for the knowledge corpus (train split).

Every row ends up in an audit log with the action taken and why, so nothing
is deleted silently. The original raw file is never modified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from src.config import Config, get_config
from src.data import deduplicator as dd
from src.data.normalization import (
    is_missing,
    normalize_text,
    normalize_unicode,
    normalize_whitespace,
    remove_boilerplate,
    strip_html,
)
from src.utils.helpers import get_logger

logger = get_logger(__name__)

# Placeholder / junk values that are not real medical content.
PLACEHOLDERS = {
    "n/a", "na", "none", "null", "nan", "test", "testing", "asdf", "tbd", "todo",
    "lorem ipsum", "xxx", "?", "??", "???", "-", "...", "no answer", "answer", "question",
}
_HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


@dataclass
class CleaningResult:
    cleaned: pd.DataFrame
    log: pd.DataFrame
    report: dict
    conflicting: pd.DataFrame = field(default_factory=pd.DataFrame)
    semantic_duplicates: pd.DataFrame = field(default_factory=pd.DataFrame)


def _invalid_reason(raw_q: str, raw_a: str, q: str, a: str, config: Config) -> str | None:
    """First reason a row is unusable, or None if valid. Checked in a fixed order."""
    if is_missing(raw_q) and is_missing(raw_a):
        return "null_question_and_answer"
    if is_missing(raw_q):
        return "null_question"
    if is_missing(raw_a):
        return "null_answer"
    if not q:
        return "empty_question"  # had content, but it was only markup/boilerplate
    if not a:
        return "empty_answer"
    if q.lower().strip(" .?!") in PLACEHOLDERS or not _HAS_LETTER.search(q):
        return "malformed_question"
    if a.lower().strip(" .?!") in PLACEHOLDERS or not _HAS_LETTER.search(a):
        return "malformed_answer"
    if len(q) < config.min_question_chars:
        return "too_short_question"
    if len(a) < config.min_answer_chars:
        return "too_short_answer"
    return None


def _operations_applied(text: str) -> list[str]:
    """Which normalization steps actually changed this text (for the audit log)."""
    ops = []
    if strip_html(text) != text:
        ops.append("html_removed")
    if normalize_unicode(text) != text:
        ops.append("unicode_normalized")
    if remove_boilerplate(text) != text:
        ops.append("boilerplate_removed")
    if normalize_whitespace(text) != text:
        ops.append("whitespace_normalized")
    return ops


def clean_dataframe(df: pd.DataFrame, config: Config | None = None, embedder: dd.Embedder | None = None) -> CleaningResult:
    """Clean a standardized QA DataFrame (see loader.STANDARD_COLUMNS)."""
    config = config or get_config()
    df = df.reset_index(drop=True).copy()
    n_total = len(df)

    # 1) Normalize text --------------------------------------------------------
    df["raw_question"] = df["question"]
    df["raw_answer"] = df["answer"]
    df["question"] = df["raw_question"].map(lambda v: normalize_text(v, config.lowercase_documents))
    df["answer"] = df["raw_answer"].map(lambda v: normalize_text(v, config.lowercase_documents))

    log = pd.DataFrame(
        {
            "question_id": df["question_id"],
            "action": "kept",
            "reason": "",
            "duplicate_of": "",
            "operations": [
                ";".join(sorted(set(_operations_applied(q) + _operations_applied(a))))
                for q, a in zip(df["raw_question"], df["raw_answer"])
            ],
        }
    )

    # 2) Invalid rows ----------------------------------------------------------
    reasons = [
        _invalid_reason(rq, ra, q, a, config)
        for rq, ra, q, a in zip(df["raw_question"], df["raw_answer"], df["question"], df["answer"])
    ]
    invalid_mask = pd.Series([r is not None for r in reasons], index=df.index)
    log.loc[invalid_mask, "action"] = "removed"
    log.loc[invalid_mask, "reason"] = [r for r in reasons if r is not None]
    valid = df[~invalid_mask]

    # 3) Duplicates (on valid rows only) ---------------------------------------
    # Detection counts (informational).
    exact_dup_questions = dd.count_duplicated(valid["raw_question"])
    normalized_dup_questions = dd.count_duplicated(valid["question"].map(dd.matching_key))

    # Removal: exact QA pairs first, then normalized QA pairs.
    exact_of = dd.first_occurrence_map(dd.exact_qa_key(valid.assign(question=valid["raw_question"], answer=valid["raw_answer"])))
    exact_mask = exact_of.notna()
    norm_of = dd.first_occurrence_map(dd.normalized_qa_key(valid))
    norm_mask = norm_of.notna() & ~exact_mask

    id_of = df["question_id"]
    for mask, ref, reason in ((exact_mask, exact_of, "exact_duplicate_qa_pair"), (norm_mask, norm_of, "normalized_duplicate_qa_pair")):
        idx = mask[mask].index
        log.loc[idx, "action"] = "removed"
        log.loc[idx, "reason"] = reason
        log.loc[idx, "duplicate_of"] = [id_of[ref[i]] for i in idx]

    kept = valid[~(exact_mask | norm_mask)]

    # 4) Flags (kept, but reported) -------------------------------------------
    conflicting = dd.find_conflicting_questions(kept)
    conflict_ids = set(conflicting["question_id"])
    flag_idx = log.index[log["question_id"].isin(conflict_ids)]
    log.loc[flag_idx, "action"] = "kept_flagged"
    log.loc[flag_idx, "reason"] = "same_question_different_answer"

    semantic = pd.DataFrame()
    if embedder is not None:
        semantic = dd.find_semantic_duplicates(kept, embedder, config.semantic_dup_threshold)
        sem_ids = set(semantic["question_id"]) | set(semantic.get("similar_to", []))
        sem_idx = log.index[log["question_id"].isin(sem_ids) & (log["action"] == "kept")]
        log.loc[sem_idx, "action"] = "kept_flagged"
        log.loc[sem_idx, "reason"] = "semantic_near_duplicate_question"

    cleaned = kept[["question_id", "question", "answer", "source", "dataset_split"]].reset_index(drop=True)

    reason_counts = log.loc[log["action"] == "removed", "reason"].value_counts().to_dict()
    op_counts: dict[str, int] = {}
    for ops in log["operations"]:
        for op in filter(None, ops.split(";")):
            op_counts[op] = op_counts.get(op, 0) + 1

    null_rows = sum(reason_counts.get(r, 0) for r in ("null_question_and_answer", "null_question", "null_answer"))
    report = {
        "total_rows": n_total,
        "null_rows": null_rows,
        "empty_questions": reason_counts.get("empty_question", 0) + reason_counts.get("null_question", 0)
        + reason_counts.get("null_question_and_answer", 0),
        "empty_answers": reason_counts.get("empty_answer", 0) + reason_counts.get("null_answer", 0),
        "malformed_rows": reason_counts.get("malformed_question", 0) + reason_counts.get("malformed_answer", 0),
        "too_short_rows": reason_counts.get("too_short_question", 0) + reason_counts.get("too_short_answer", 0),
        "invalid_rows": int(invalid_mask.sum()),
        "exact_duplicate_questions_detected": exact_dup_questions,
        "normalized_duplicate_questions_detected": normalized_dup_questions,
        "exact_duplicates": int(exact_mask.sum()),
        "normalized_duplicates": int(norm_mask.sum()),
        "conflicting_answer_rows_flagged": len(conflict_ids),
        "semantic_duplicate_pairs_flagged": int(len(semantic)),
        "final_rows": len(cleaned),
        "removed_by_reason": reason_counts,
        "normalization_operations": op_counts,
        "settings": {
            "min_question_chars": config.min_question_chars,
            "min_answer_chars": config.min_answer_chars,
            "lowercase_documents": config.lowercase_documents,
            "semantic_dup_threshold": config.semantic_dup_threshold,
        },
    }
    assert report["final_rows"] == n_total - report["invalid_rows"] - report["exact_duplicates"] - report["normalized_duplicates"]
    logger.info(
        "Cleaning: %d -> %d rows (invalid=%d, exact dup=%d, normalized dup=%d)",
        n_total, len(cleaned), report["invalid_rows"], report["exact_duplicates"], report["normalized_duplicates"],
    )
    return CleaningResult(cleaned=cleaned, log=log, report=report, conflicting=conflicting, semantic_duplicates=semantic)

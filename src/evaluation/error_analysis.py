"""Per-question error analysis: WHY did the metrics move?

Each eval question is put in one of four buckets based on success@K
(a correct document within the top K) for the two pipelines:

    raw_fail_clean_success | raw_success_clean_fail | both_failed | both_succeeded

For raw failures we also look up the raw top-1 document in the cleaning audit
log: if it was a row that cleaning removed (empty answer, duplicate, junk...),
that is direct evidence of how noise hurt the raw pipeline.
"""

from __future__ import annotations

import pandas as pd

CATEGORIES = ("raw_fail_clean_success", "raw_success_clean_fail", "both_failed", "both_succeeded")


def merge_results(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side per-question table (same questions, prefixed columns)."""
    if set(raw["question_id"]) != set(cleaned["question_id"]):
        raise ValueError("Raw and cleaned results were computed on different question sets.")
    shared = ["question_id", "question", "ground_truth"]
    r = raw.rename(columns={c: f"raw_{c}" for c in raw.columns if c not in shared})
    c = cleaned.drop(columns=["question", "ground_truth"]).rename(
        columns={col: f"cleaned_{col}" for col in cleaned.columns if col not in shared}
    )
    return r.merge(c, on="question_id", how="inner")


def categorize(merged: pd.DataFrame, k: int) -> pd.Series:
    raw_ok = merged[f"raw_recall@{k}"] > 0
    clean_ok = merged[f"cleaned_recall@{k}"] > 0
    cat = pd.Series("both_failed", index=merged.index)
    cat[raw_ok & clean_ok] = "both_succeeded"
    cat[~raw_ok & clean_ok] = "raw_fail_clean_success"
    cat[raw_ok & ~clean_ok] = "raw_success_clean_fail"
    return cat


def attach_cleaning_status(merged: pd.DataFrame, cleaning_log: pd.DataFrame | None) -> pd.DataFrame:
    """What did cleaning do to the raw pipeline's documents?

    raw_top1_cleaning_status   - status of the raw top-1 document
    raw_removed_above_correct  - removal reasons of raw documents ranked ABOVE the
                                 first correct one (or in the whole list if none was correct)
    """
    merged = merged.copy()
    if cleaning_log is None or cleaning_log.empty:
        merged["raw_top1_cleaning_status"] = "unknown"
        merged["raw_removed_above_correct"] = ""
        return merged
    status = cleaning_log.set_index("question_id").apply(
        lambda r: r["action"] if r["action"] != "removed" else f"removed:{r['reason']}", axis=1
    ).to_dict()
    top1 = merged["raw_top_1_ids"].fillna("").astype(str).str.split("|").str[0]
    merged["raw_top1_cleaning_status"] = top1.map(status).fillna("unknown")

    def removed_above(row: pd.Series) -> str:
        ids = [i for i in str(row.get("raw_all_ids", "") or "").split("|") if i]
        rank = row.get("raw_rank")
        cutoff = int(rank) - 1 if pd.notna(rank) and str(rank) != "" else len(ids)
        reasons = [status.get(i, "") for i in ids[:cutoff]]
        return ";".join(r.split(":", 1)[1] for r in reasons if r.startswith("removed:"))

    merged["raw_removed_above_correct"] = merged.apply(removed_above, axis=1)
    return merged


def explain(row: pd.Series) -> str:
    cat = row["category"]
    status = row.get("raw_top1_cleaning_status", "unknown")
    removed = [r for r in str(row.get("raw_removed_above_correct", "")).split(";") if r]
    if cat == "raw_fail_clean_success":
        if removed:
            counts = pd.Series(removed).value_counts()
            detail = ", ".join(f"{n}x {r}" for r, n in counts.items())
            return f"{len(removed)} raw document(s) ranked above the correct one were noise that cleaning removed ({detail})."
        return "No removed records above the answer: normalization (HTML/boilerplate/Unicode stripping) changed the embeddings enough to re-rank it."
    if cat == "raw_success_clean_fail":
        if row.get("cleaned_n_relevant_in_corpus", 1) == 0:
            return "Cleaning removed every copy of the correct answer (over-cleaning)."
        return "Cleaned ranking placed other documents above the correct one."
    if cat == "both_failed":
        return "Neither corpus ranks the answer in top-K: likely an embedding/paraphrase limitation, not data quality."
    return ""


def run_error_analysis(
    raw: pd.DataFrame, cleaned: pd.DataFrame, cleaning_log: pd.DataFrame | None, k: int = 5
) -> tuple[pd.DataFrame, dict]:
    merged = merge_results(raw, cleaned)
    merged["category"] = categorize(merged, k)
    merged = attach_cleaning_status(merged, cleaning_log)
    merged["explanation"] = merged.apply(explain, axis=1)

    counts = {c: int((merged["category"] == c).sum()) for c in CATEGORIES}
    raw_fail = merged[merged[f"raw_recall@{k}"] == 0]
    summary = {
        "success_definition": f"correct document within top {k}",
        "category_counts": counts,
        "category_counts_by_k": {
            kk: categorize(merged, kk).value_counts().reindex(CATEGORIES, fill_value=0).astype(int).to_dict()
            for kk in sorted({1, 5, k})
            if f"raw_recall@{kk}" in merged
        },
        "raw_fail_clean_success_with_removed_noise_above": int(
            ((merged["category"] == "raw_fail_clean_success") & (merged["raw_removed_above_correct"] != "")).sum()
        ),
        "raw_failures_top1_cleaning_status": raw_fail["raw_top1_cleaning_status"].value_counts().to_dict(),
        "all_raw_top1_cleaning_status": merged["raw_top1_cleaning_status"].value_counts().to_dict(),
    }
    return merged, summary


def examples(merged: pd.DataFrame, category: str, n: int = 5) -> pd.DataFrame:
    sub = merged[merged["category"] == category]
    if category in ("raw_fail_clean_success", "raw_success_clean_fail"):
        # Most informative first: biggest rank difference.
        diff = (sub["raw_reciprocal_rank"] - sub["cleaned_reciprocal_rank"]).abs()
        sub = sub.assign(_d=diff).sort_values("_d", ascending=False).drop(columns="_d")
    return sub.head(n)

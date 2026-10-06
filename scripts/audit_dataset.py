"""Step 1 - Audit the raw splits and detect train/validate/test leakage.

Outputs:
  reports/data_audit.md
  reports/leakage_report.md / leakage_report.json
  reports/leakage_semantic_candidates.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import get_config  # noqa: E402
from src.data.deduplicator import count_duplicated  # noqa: E402
from src.data.leakage import leakage_report  # noqa: E402
from src.data.loader import detect_columns, load_all_splits, read_csv  # noqa: E402
from src.data.normalization import matching_key, normalize_text  # noqa: E402
from src.embeddings.embedding_model import get_embedding_model  # noqa: E402
from src.utils.helpers import file_fingerprint, get_logger, save_json, truncate  # noqa: E402

logger = get_logger("audit")


def profile(df: pd.DataFrame) -> dict:
    q_empty = df["question"].str.strip() == ""
    a_empty = df["answer"].str.strip() == ""
    a_clean_empty = df["answer"].map(normalize_text) == ""
    lengths = df["answer"].str.len()
    return {
        "rows": len(df),
        "empty_or_null_questions": int(q_empty.sum()),
        "empty_or_null_answers": int(a_empty.sum()),
        "answers_empty_after_markup_removal": int((a_clean_empty & ~a_empty).sum()),
        "exact_duplicate_questions": count_duplicated(df["question"]),
        "exact_duplicate_qa_pairs": count_duplicated(df["question"] + "␟" + df["answer"]),
        "normalized_duplicate_questions": count_duplicated(df["question"].map(matching_key)),
        "answers_with_html": int(df["answer"].str.contains(r"<[a-zA-Z/][^>]*>|&[a-z]+;|&#\d+;", regex=True).sum()),
        "answers_with_non_ascii": int(df["answer"].map(lambda s: any(ord(c) > 127 for c in s)).sum()),
        "answer_len_mean": round(float(lengths.mean()), 1) if len(df) else 0,
        "answer_len_max": int(lengths.max()) if len(df) else 0,
        "unique_sources": int(df["source"].nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-semantic", action="store_true", help="skip embedding-based leakage check")
    args = parser.parse_args()
    cfg = get_config()
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    splits = load_all_splits(cfg)
    schema = {}
    for s in splits:
        raw_df = read_csv(cfg.split_path(s))
        m = detect_columns(raw_df, cfg)
        schema[s] = {"columns": list(raw_df.columns), "question": m.question, "answer": m.answer, "id": m.id,
                     "source": m.source, "sha256_16": file_fingerprint(cfg.split_path(s))}
    profiles = {s: profile(df) for s, df in splits.items()}

    # ---- data_audit.md --------------------------------------------------------
    lines = ["# Data Audit", "", "Raw splits as loaded (no cleaning applied).", "", "## Schema", ""]
    for s, info in schema.items():
        lines.append(f"- **{s}**: columns={info['columns']} → question=`{info['question']}`, answer=`{info['answer']}`, "
                     f"id=`{info['id'] or '(generated)'}`, source=`{info['source'] or '(none)'}`, version=`{info['sha256_16']}`")
    lines += ["", "## Quality profile", "", "| Check | " + " | ".join(profiles) + " |", "|---|" + "---:|" * len(profiles)]
    for key in next(iter(profiles.values())):
        lines.append(f"| {key} | " + " | ".join(str(p[key]) for p in profiles.values()) + " |")
    lines += ["", "## Sample records (train)", ""]
    for _, r in splits["train"].head(5).iterrows():
        lines.append(f"- `{r['question_id']}` **Q:** {truncate(r['question'], 100)!r}  **A:** {truncate(r['answer'], 140)!r}")
    (cfg.reports_dir / "data_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- leakage ----------------------------------------------------------------
    embedder = None if args.no_semantic else get_embedding_model(cfg)
    results = leakage_report(splits, embedder, cfg.semantic_leak_threshold)
    summary = {k: v.summary() for k, v in results.items()}
    sem_frames = []
    for k, v in results.items():
        if len(v.semantic):
            sem_frames.append(v.semantic.assign(pair=k))
    sem_all = pd.concat(sem_frames) if sem_frames else pd.DataFrame()
    sem_all.to_csv(cfg.reports_dir / "leakage_semantic_candidates.csv", index=False)

    action = (
        "EXCLUDED from evaluation (EXCLUDE_LEAKED_EVAL_QUESTIONS=true). No file was modified."
        if cfg.exclude_leaked_eval_questions
        else "KEPT in evaluation (EXCLUDE_LEAKED_EVAL_QUESTIONS=false). Metrics may be optimistic."
    )
    save_json({"sizes": {s: len(d) for s, d in splits.items()}, "overlaps": summary,
               "semantic_threshold": cfg.semantic_leak_threshold, "semantic_checked": embedder is not None,
               "action_exact_normalized": action, "action_semantic": "reported only",
               "overlap_ids": {k: {"exact": sorted(v.exact_ids), "normalized": sorted(v.normalized_ids)} for k, v in results.items()}},
              cfg.reports_dir / "leakage_report.json")

    lines = ["# Leakage Report", "",
             f"Train rows: {len(splits['train'])}  ", f"Validation rows: {len(splits['validate'])}  ", f"Test rows: {len(splits['test'])}", "",
             "| Pair | Exact overlap | Normalized overlap (not exact) | Semantic candidates (≥ %.2f) |" % cfg.semantic_leak_threshold,
             "|---|---:|---:|---:|"]
    for v in summary.values():
        sem = v["semantic_candidates"] if embedder is not None else "not checked"
        lines.append(f"| {v['pair']} | {v['exact_overlap']} | {v['normalized_overlap']} | {sem} |")
    lines += [
        "", "## Definitions", "",
        "- **Exact**: identical question string (after trimming).",
        "- **Normalized**: identical after Unicode/HTML/case/punctuation/whitespace normalization.",
        "- **Semantic**: question-embedding cosine similarity above the threshold (excluding exact/normalized matches).",
        "", "## Action taken", "",
        f"- Exact + normalized overlaps of validate/test questions with train: **{action}**",
        "- Semantic candidates: **reported only**. In this benchmark every eval question is *supposed* to have an "
        "answering document in train, so a semantically similar train question is expected and is not, on its own, leakage. "
        "Review `leakage_semantic_candidates.csv` if you need a stricter policy.",
        "- The vector indexes are built from train only; validate/test are never indexed.",
    ]
    for k, v in results.items():
        if v.exact_ids or v.normalized_ids:
            lines += ["", f"### {k} overlapping {v.right} question_ids", ""]
            lines += [f"- exact: `{i}`" for i in sorted(v.exact_ids)] + [f"- normalized: `{i}`" for i in sorted(v.normalized_ids)]
    if len(sem_all):
        lines += ["", "## Top semantic candidates", "", "| Pair | Eval question | Nearest question | Similarity |", "|---|---|---|---:|"]
        for _, r in sem_all.sort_values("similarity", ascending=False).head(10).iterrows():
            lines.append(f"| {r['pair']} | {truncate(r['question'], 70)} | {truncate(r['nearest_left_question'], 70)} | {r['similarity']:.3f} |")
    (cfg.reports_dir / "leakage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for v in summary.values():
        logger.info("%s", v)
    logger.info("Wrote data_audit.md and leakage_report.md to %s", cfg.reports_dir)


if __name__ == "__main__":
    main()

"""Step 2 - Clean train.csv into data/processed/cleaned_train.csv.

The raw file is never modified. Outputs:
  data/processed/cleaned_train.csv
  reports/data_cleaning_report.json / .csv / .md
  reports/cleaning_log.csv              (one row per input record: kept/removed + reason)
  reports/conflicting_questions.csv     (same question, different answers - kept)
  reports/semantic_duplicates.csv       (near-duplicate questions - kept, flagged)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import get_config  # noqa: E402
from src.data.cleaner import clean_dataframe  # noqa: E402
from src.data.loader import load_split  # noqa: E402
from src.embeddings.embedding_model import get_embedding_model  # noqa: E402
from src.utils.helpers import file_fingerprint, get_logger, save_json, truncate  # noqa: E402

logger = get_logger("clean")


def write_markdown(report: dict, log: pd.DataFrame, raw: pd.DataFrame, semantic: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Data Cleaning Report", "",
        "Cleaning applied to the knowledge corpus (`train.csv`) only. The raw file is unchanged; every decision is in `cleaning_log.csv`.",
        "", "## Summary", "", "| Step | Rows |", "|---|---:|",
        f"| Original rows | {report['total_rows']} |",
        f"| Invalid rows removed | {report['invalid_rows']} |",
        f"| &nbsp;&nbsp;null rows | {report['null_rows']} |",
        f"| &nbsp;&nbsp;empty questions (incl. null) | {report['empty_questions']} |",
        f"| &nbsp;&nbsp;empty answers (incl. null) | {report['empty_answers']} |",
        f"| &nbsp;&nbsp;malformed / placeholder | {report['malformed_rows']} |",
        f"| &nbsp;&nbsp;too short | {report['too_short_rows']} |",
        f"| Exact duplicate QA pairs removed | {report['exact_duplicates']} |",
        f"| Normalized duplicate QA pairs removed | {report['normalized_duplicates']} |",
        f"| **Final rows** | **{report['final_rows']}** |",
        "", "## Detected but not removed", "",
        f"- Exact duplicate questions (any answer): {report['exact_duplicate_questions_detected']}",
        f"- Normalized duplicate questions (any answer): {report['normalized_duplicate_questions_detected']}",
        f"- Rows sharing a question but with a *different* answer: {report['conflicting_answer_rows_flagged']} "
        "(kept - they may be complementary or need expert review; see `conflicting_questions.csv`)",
        f"- Semantic near-duplicate question pairs (≥ {report['settings']['semantic_dup_threshold']}): "
        f"{report['semantic_duplicate_pairs_flagged']} (kept - e.g. type 1 vs type 2 diabetes are similar but medically distinct)",
        "", "## Removed rows by reason", "", "| Reason | Rows |", "|---|---:|",
    ]
    lines += [f"| {k} | {v} |" for k, v in sorted(report["removed_by_reason"].items(), key=lambda x: -x[1])]
    lines += ["", "## Normalization applied (rows changed, kept or removed)", "", "| Operation | Rows |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(report["normalization_operations"].items())]
    lines += [
        "", "Normalization is conservative: Unicode NFKC, HTML tag/entity removal, known web boilerplate, whitespace. "
        "Punctuation, numbers, units (mg/dL, mmHg) and medical terminology are preserved; "
        f"lowercasing is {'ON' if report['settings']['lowercase_documents'] else 'OFF'} (case can carry meaning, e.g. `HbA1c`, `IgA`).",
        "", "## Examples", "",
    ]
    raw_by_id = raw.set_index("question_id")
    for reason in log.loc[log["action"] == "removed", "reason"].unique():
        ex = log[(log["action"] == "removed") & (log["reason"] == reason)].head(2)
        for _, r in ex.iterrows():
            src = raw_by_id.loc[r["question_id"]]
            dup = f" (duplicate of `{r['duplicate_of']}`)" if r["duplicate_of"] else ""
            lines.append(f"- **{reason}**{dup}: Q={truncate(src['question'], 70)!r} A={truncate(src['answer'], 90)!r}")
    changed = log[(log["action"] != "removed") & (log["operations"] != "")].head(3)
    for _, r in changed.iterrows():
        lines.append(f"- **normalized** ({r['operations']}): A={truncate(raw_by_id.loc[r['question_id'], 'answer'], 110)!r}")
    if len(semantic):
        lines += ["", "## Semantic near-duplicate examples (kept)", ""]
        for _, r in semantic.head(5).iterrows():
            lines.append(f"- {r['similarity']:.3f}: {truncate(r['question'], 70)!r} ↔ {truncate(r['similar_question'], 70)!r}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-semantic", action="store_true", help="skip semantic near-duplicate flagging")
    args = parser.parse_args()
    cfg = get_config()

    raw = load_split("train", cfg)
    embedder = None if args.no_semantic else get_embedding_model(cfg)
    result = clean_dataframe(raw, cfg, embedder)

    cfg.processed_data_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    result.cleaned.to_csv(cfg.cleaned_train_path, index=False)

    report = {**result.report, "input_file": str(cfg.split_path("train").name),
              "input_version": file_fingerprint(cfg.split_path("train")),
              "output_version": file_fingerprint(cfg.cleaned_train_path)}
    save_json(report, cfg.reports_dir / "data_cleaning_report.json")
    flat = {k: v for k, v in report.items() if not isinstance(v, dict)}
    pd.DataFrame([flat]).T.rename(columns={0: "value"}).rename_axis("metric").to_csv(cfg.reports_dir / "data_cleaning_report.csv")
    result.log.to_csv(cfg.reports_dir / "cleaning_log.csv", index=False)
    result.conflicting.to_csv(cfg.reports_dir / "conflicting_questions.csv", index=False)
    result.semantic_duplicates.to_csv(cfg.reports_dir / "semantic_duplicates.csv", index=False)
    write_markdown(report, result.log, raw, result.semantic_duplicates, cfg.reports_dir / "data_cleaning_report.md")
    logger.info("Wrote %s (%d rows)", cfg.cleaned_train_path, len(result.cleaned))


if __name__ == "__main__":
    main()

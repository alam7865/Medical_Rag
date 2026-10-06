"""Step 5 - Compare Raw vs Cleaned results, run error analysis, write reports and charts.

Usage: python scripts/compare_results.py [--split test|validate] [--success-k 5]

Outputs:
  reports/evaluation_results.csv     per-question, side-by-side, with error category
  reports/evaluation_summary.json
  reports/evaluation_report.md
  reports/error_analysis.md
  reports/figures/*.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import get_config  # noqa: E402
from src.evaluation.error_analysis import CATEGORIES, examples, run_error_analysis  # noqa: E402
from src.utils.helpers import load_json, safe_improvement_pct, save_json, truncate  # noqa: E402

# Validated categorical pair (slot 1 / slot 2 of the reference palette) + chart chrome.
COLORS = {"raw": "#2a78d6", "cleaned": "#eb6834"}
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

TABLE_METRICS = ["recall@1", "recall@3", "recall@5", "recall@10", "precision@1", "precision@3", "precision@5",
                 "precision@10", "mrr", "ndcg@5", "ndcg@10", "semantic_similarity_top1", "semantic_similarity_best_top5"]
LABELS = {"semantic_similarity_top1": "Semantic Similarity (top-1)",
          "semantic_similarity_best_top5": "Semantic Similarity (best of top-5)"}


def label(m: str) -> str:
    return LABELS.get(m, m.replace("recall", "Recall").replace("precision", "Precision").replace("ndcg", "NDCG").replace("mrr", "MRR"))


def fmt_pct(v: float | None) -> str:
    return "n/a (raw = 0)" if v is None else f"{v:+.1f}%"


# ---------------------------------------------------------------- figures
def bar_chart(metrics: list[str], raw: dict, cleaned: dict, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(4.2, 1.5 * len(metrics) + 1.8), 3.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = range(len(metrics))
    w = 0.36
    for off, (name, vals) in zip((-w / 2, w / 2), (("raw", raw), ("cleaned", cleaned))):
        ys = [vals[m] for m in metrics]
        bars = ax.bar([i + off for i in x], ys, width=w - 0.03, color=COLORS[name],
                      label="Raw RAG" if name == "raw" else "Cleaned RAG", zorder=3)
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, y + 0.015, f"{y:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK_2)
    ax.set_xticks(list(x), [label(m) for m in metrics], color=INK_2, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis="y", colors=MUTED, labelsize=8)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=22)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.1), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK_2,
              handlelength=1, borderaxespad=0)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def make_figures(raw: dict, cleaned: dict, k_values: list[int], ndcg_k: list[int], fig_dir: Path, split: str) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("recall_at_k.png", [f"recall@{k}" for k in k_values], f"Recall@K - Raw vs Cleaned ({split})"),
        ("precision_at_k.png", [f"precision@{k}" for k in k_values], f"Precision@K - Raw vs Cleaned ({split})"),
        ("mrr.png", ["mrr"], f"MRR ({split})"),
        ("ndcg.png", [f"ndcg@{k}" for k in ndcg_k], f"NDCG@K ({split})"),
    ]
    for fname, metrics, title in specs:
        bar_chart(metrics, raw, cleaned, title, fig_dir / fname)
    return [s[0] for s in specs]


# ---------------------------------------------------------------- report
def comparison_rows(raw: dict, cleaned: dict) -> list[dict]:
    rows = []
    for m in TABLE_METRICS:
        if m in raw and m in cleaned:
            rows.append({"metric": m, "raw": raw[m], "cleaned": cleaned[m], "abs_diff": cleaned[m] - raw[m],
                         "improvement_pct": safe_improvement_pct(raw[m], cleaned[m])})
    return rows


def fmt_rank(rank: object, top_k: int) -> str:
    return f">{top_k}" if pd.isna(rank) or rank == "" else str(int(float(rank)))


def example_block(merged: pd.DataFrame, cat: str, cfg_top_k: int, n: int = 4) -> list[str]:
    ex = examples(merged, cat, n)
    if ex.empty:
        return ["_No questions in this category._", ""]
    out = []
    for _, r in ex.iterrows():
        out += [
            f"- **Q:** {truncate(r['question'], 110)}  ",
            f"  **Expected:** {truncate(r['ground_truth'], 120)}  ",
            f"  **Raw** rank={fmt_rank(r['raw_rank'], cfg_top_k)} · top-1: {truncate(r['raw_top_1_question'], 60)!r} → "
            f"{truncate(r['raw_top_1_answer'], 80)!r} _(cleaning status: {r['raw_top1_cleaning_status']})_  ",
            f"  **Cleaned** rank={fmt_rank(r['cleaned_rank'], cfg_top_k)} · top-1: {truncate(r['cleaned_top_1_question'], 60)!r} → "
            f"{truncate(r['cleaned_top_1_answer'], 80)!r}  ",
            f"  _Why:_ {r['explanation']}",
        ]
    return out + [""]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["validate", "test"], default="test")
    parser.add_argument("--success-k", type=int, default=1, help="K used to define success in error analysis")
    args = parser.parse_args()
    cfg = get_config()
    rd = cfg.results_dir

    try:
        s_raw = load_json(rd / f"raw_{args.split}_summary.json")
        s_cln = load_json(rd / f"cleaned_{args.split}_summary.json")
        pq_raw = pd.read_csv(rd / f"raw_{args.split}_per_question.csv", keep_default_na=False, na_values=[""])
        pq_cln = pd.read_csv(rd / f"cleaned_{args.split}_per_question.csv", keep_default_na=False, na_values=[""])
    except FileNotFoundError as e:
        sys.exit(f"Missing results ({e.filename}). Run evaluate_raw.py and evaluate_cleaned.py --split {args.split} first.")
    for df in (pq_raw, pq_cln):
        for c in ("question", "ground_truth", "top_1_question", "top_1_answer", "top_1_ids"):
            if c in df:
                df[c] = df[c].fillna("")

    # ---- experimental-control checks -------------------------------------------
    if s_raw["experiment_settings"] != s_cln["experiment_settings"]:
        sys.exit(f"Settings differ between runs - comparison is invalid:\n{s_raw['experiment_settings']}\n{s_cln['experiment_settings']}")
    if s_raw["dataset_version"]["eval"] != s_cln["dataset_version"]["eval"]:
        sys.exit("Raw and cleaned were evaluated on different versions of the eval file.")

    log_path = cfg.reports_dir / "cleaning_log.csv"
    cleaning_log = pd.read_csv(log_path, keep_default_na=False) if log_path.exists() else None
    merged, ea = run_error_analysis(pq_raw, pq_cln, cleaning_log, args.success_k)

    raw_m, cln_m = s_raw["metrics"], s_cln["metrics"]
    rows = comparison_rows(raw_m, cln_m)
    better = [r["metric"] for r in rows if r["abs_diff"] > 1e-9]
    worse = [r["metric"] for r in rows if r["abs_diff"] < -1e-9]
    same = [r["metric"] for r in rows if abs(r["abs_diff"]) <= 1e-9]
    core = ["recall@1", "recall@5", "mrr", "ndcg@10"]
    core_wins = sum(cln_m[m] > raw_m[m] for m in core)
    winner = "Cleaned RAG" if core_wins > len(core) / 2 else ("Raw RAG" if core_wins < len(core) / 2 else "Tie")

    figures = make_figures(raw_m, cln_m, list(cfg.eval_k_values), list(cfg.ndcg_k_values), cfg.figures_dir, args.split)

    # ---- files ---------------------------------------------------------------------
    merged.to_csv(cfg.reports_dir / "evaluation_results.csv", index=False)
    cleaning = load_json(cfg.reports_dir / "data_cleaning_report.json") if (cfg.reports_dir / "data_cleaning_report.json").exists() else {}
    summary = {
        "split": args.split,
        "winner_on_core_metrics": winner,
        "core_metrics": core,
        "comparison": rows,
        "improved_metrics": better,
        "worse_metrics": worse,
        "unchanged_metrics": same,
        "error_analysis": ea,
        "dataset": {
            "train_rows": s_raw["corpus_records"],
            "cleaned_train_rows": s_cln["corpus_records"],
            "raw_index_chunks": s_raw["index_chunks"],
            "cleaned_index_chunks": s_cln["index_chunks"],
            "eval_set": s_raw["eval_set"],
            "dataset_version": {"raw": s_raw["dataset_version"], "cleaned": s_cln["dataset_version"]},
        },
        "experiment_settings": s_raw["experiment_settings"],
    }
    save_json(summary, cfg.reports_dir / "evaluation_summary.json")

    sizes = {}
    for s in ("train", "validate", "test"):
        try:
            sizes[s] = len(pd.read_csv(cfg.split_path(s), dtype=str))
        except FileNotFoundError:
            sizes[s] = "missing"
    es = s_raw["eval_set"]
    settings = s_raw["experiment_settings"]

    L = [f"# Evaluation Report - Raw RAG vs Cleaned RAG ({args.split} split)", "",
         "All numbers below were produced by running the pipeline; nothing is hand-entered.", "",
         "## Verdict", ""]
    r1 = next(r for r in rows if r["metric"] == "recall@1")
    mrr = next(r for r in rows if r["metric"] == "mrr")
    L += [f"**{winner}** performs better on the core retrieval metrics ({core_wins}/{len(core)} of {', '.join(core)} improved).", "",
          f"- Recall@1: {r1['raw']:.4f} → {r1['cleaned']:.4f} ({r1['abs_diff']:+.4f} absolute, {fmt_pct(r1['improvement_pct'])})",
          f"- MRR: {mrr['raw']:.4f} → {mrr['cleaned']:.4f} ({mrr['abs_diff']:+.4f} absolute, {fmt_pct(mrr['improvement_pct'])})",
          f"- Improved: {', '.join(better) or 'none'}",
          f"- Worse: {', '.join(worse) or 'none'}",
          f"- Unchanged: {', '.join(same) or 'none'}", ""]

    L += ["## Dataset statistics", "", "| Item | Value |", "|---|---:|",
          f"| Train size (raw corpus) | {sizes['train']} |", f"| Validation size | {sizes['validate']} |", f"| Test size | {sizes['test']} |",
          f"| Cleaned train size | {s_cln['corpus_records']} |",
          f"| Raw index chunks | {s_raw['index_chunks']} |", f"| Cleaned index chunks | {s_cln['index_chunks']} |",
          f"| {args.split} rows | {es['total_rows']} |",
          f"| &nbsp;&nbsp;excluded: empty question/answer | {es['empty_question_or_answer']} |",
          f"| &nbsp;&nbsp;excluded: leaked from train (exact/normalized) | {es['leaked_from_train_exact_or_normalized']} |",
          f"| &nbsp;&nbsp;excluded: answer not in corpus | {es['answer_not_in_corpus']} |",
          f"| **Questions evaluated (identical for both)** | **{es['evaluated']}** |", ""]
    if cleaning:
        L += [f"Cleaning removed {cleaning['invalid_rows']} invalid rows, {cleaning['exact_duplicates']} exact duplicates and "
              f"{cleaning['normalized_duplicates']} normalized duplicates (see `data_cleaning_report.md`).", ""]

    L += ["## Retrieval metrics", "", "| Metric | Raw RAG | Cleaned RAG | Abs. diff | Improvement |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {label(r['metric'])} | {r['raw']:.4f} | {r['cleaned']:.4f} | {r['abs_diff']:+.4f} | {fmt_pct(r['improvement_pct'])} |")
    L += ["", "Improvement % = (Cleaned − Raw) / Raw × 100 (n/a when Raw = 0). "
          "Semantic Similarity is the cosine similarity between the expected answer and retrieved answer embeddings; it is **not** accuracy.", ""]

    L += ["## Interpreting the metrics", "",
          "- **Recall@K** = share of questions with a correct document anywhere in the top K (hit rate).",
          "- **Precision@K** = share of the top K that is correct. Note: the raw corpus contains several *copies* of many answers "
          "(duplicates), and each copy counts as a separate relevant document. That lets raw fill more of the top K with correct "
          "copies, which can raise raw Precision@K at larger K without giving the user any extra information. "
          "Read Precision@K together with Recall@K and MRR.",
          "- **MRR** rewards placing the first correct document high. **NDCG@K** rewards good ordering of all correct documents, "
          "normalized by the ideal ranking *for that corpus*.", ""]
    if any(m.startswith("precision") for m in worse):
        L += [f"In this run Precision@K got worse for: {', '.join(m for m in worse if m.startswith('precision'))}. "
              "This is consistent with the duplicate effect above: after de-duplication there are fewer correct copies to fill the list.", ""]

    L += ["## Figures", ""] + [f"![{f}](figures/{f})" for f in figures] + [""]

    cc = ea["category_counts"]
    L += ["## Error analysis", "", f"Success = {ea['success_definition']}.", "",
          "| Category | Questions |", "|---|---:|"] + [f"| {c} | {cc[c]} |" for c in CATEGORIES]
    L += ["", "Same breakdown at other K:", "", "| K | " + " | ".join(CATEGORIES) + " |", "|---|" + "---:|" * len(CATEGORIES)]
    for kk, counts in ea["category_counts_by_k"].items():
        L.append(f"| {kk} | " + " | ".join(str(counts[c]) for c in CATEGORIES) + " |")
    L += ["", f"Of the {cc['raw_fail_clean_success']} questions that only the cleaned pipeline answered, "
          f"**{ea['raw_fail_clean_success_with_removed_noise_above']}** had at least one raw document ranked above the correct "
          "answer that cleaning later removed (empty answer, junk, too short, duplicate).", "",
          "**What was the raw pipeline's top-1 document?** (looked up in the cleaning log)", "",
          "| Raw top-1 status | All questions |", "|---|---:|"]
    for st, n in sorted(ea["all_raw_top1_cleaning_status"].items(), key=lambda x: -x[1]):
        L.append(f"| {st} | {n} |")
    L += ["", "Rows marked `removed:*` are records the cleaner deleted (duplicates, empty answers, junk) that the raw retriever "
          "still ranked first - direct evidence of how noise affects the raw pipeline. A `removed:exact_duplicate_qa_pair` top-1 "
          "can still be a *correct* answer (it is a copy), so this table explains rankings, not failures by itself.", "",
          "Full examples per category: `error_analysis.md`. Per-question data: `evaluation_results.csv`.", ""]

    L += ["## Reproducibility", "", "| Setting | Value (identical for both pipelines) |", "|---|---|"]
    L += [f"| {k} | `{v}` |" for k, v in settings.items()]
    L += [f"| corpus version (raw / cleaned) | `{s_raw['dataset_version']['corpus']}` / `{s_cln['dataset_version']['corpus']}` |",
          f"| eval file version | `{s_raw['dataset_version']['eval']}` |", ""]
    (cfg.reports_dir / "evaluation_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    E = [f"# Error Analysis ({args.split} split)", "", f"Success = {ea['success_definition']}. Up to 4 examples per category, "
         "most informative (largest rank change) first.", ""]
    for cat in CATEGORIES:
        E += [f"## {cat} ({cc[cat]})", ""] + example_block(merged, cat, cfg.top_k)
    (cfg.reports_dir / "error_analysis.md").write_text("\n".join(E) + "\n", encoding="utf-8")

    print(f"\nRaw vs Cleaned ({args.split}, {es['evaluated']} questions)\n")
    print(f"{'Metric':32s} {'Raw':>8s} {'Cleaned':>8s} {'Improvement':>14s}")
    for r in rows:
        print(f"{label(r['metric']):32s} {r['raw']:8.4f} {r['cleaned']:8.4f} {fmt_pct(r['improvement_pct']):>14s}")
    print(f"\nWinner on core metrics: {winner}. Error categories (success@{args.success_k}): {cc}")
    print(f"Reports written to {cfg.reports_dir}")


if __name__ == "__main__":
    main()

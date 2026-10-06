"""Shared entry points so the raw and cleaned scripts run the *identical* procedure.

The only argument that differs between the two pipelines is `variant`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config  # noqa: E402
from src.data.loader import load_cleaned_train, load_split  # noqa: E402
from src.embeddings.embedding_model import get_embedding_model  # noqa: E402
from src.evaluation.evaluator import evaluate  # noqa: E402
from src.rag.pipeline import build_index  # noqa: E402
from src.utils.helpers import file_fingerprint, save_json  # noqa: E402


def build(variant: str) -> None:
    cfg = get_config()
    if variant == "raw":
        corpus, path = load_split("train", cfg), cfg.split_path("train")
    else:
        corpus, path = load_cleaned_train(cfg), cfg.cleaned_train_path
    stats = build_index(variant, corpus, get_embedding_model(cfg), cfg, dataset_version=file_fingerprint(path))
    save_json(stats, cfg.results_dir / f"index_{variant}.json")
    print(f"[{variant}] indexed {stats['records_indexed']} records as {stats['chunks']} chunks "
          f"into '{stats['collection']}' ({cfg.chroma_path(variant)})")


def run_eval(variant: str) -> None:
    parser = argparse.ArgumentParser(description=f"Evaluate the {variant} RAG retriever.")
    parser.add_argument("--split", choices=["validate", "test"], default="test",
                        help="'validate' for development/tuning, 'test' for the final untouched evaluation")
    parser.add_argument("--no-semantic", action="store_true", help="skip semantic-similarity scoring")
    args = parser.parse_args()
    cfg = get_config()
    summary = evaluate(variant, args.split, get_embedding_model(cfg), cfg, semantic=not args.no_semantic)
    print(f"\n[{variant}] {args.split}: {summary['n_questions']} questions")
    for k, v in summary["metrics"].items():
        print(f"  {k:32s} {v:.4f}")

"""Ask a question and see what the Raw and Cleaned retrievers return, side by side.

Usage:
  python scripts/ask.py "What are the symptoms of diabetes?"
  python scripts/ask.py "How is gout treated?" --top-k 5 --pipeline cleaned
  python scripts/ask.py            # interactive mode (empty line or Ctrl+C to quit)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config  # noqa: E402
from src.embeddings.embedding_model import get_embedding_model  # noqa: E402
from src.rag.pipeline import get_retriever  # noqa: E402
from src.utils.helpers import truncate  # noqa: E402


def show(question: str, retrievers: dict, top_k: int) -> None:
    for name, retriever in retrievers.items():
        print(f"\n=== {name.upper()} RAG ===")
        for d in retriever.retrieve(question, top_k):
            answer = d.answer.strip() or "<EMPTY ANSWER>"
            print(f"{d.rank}. score={d.score:.3f}  Q: {truncate(d.question, 80)}")
            print(f"   A: {truncate(answer, 220)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?", help="question to ask (omit for interactive mode)")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--pipeline", choices=["both", "raw", "cleaned"], default="both")
    args = parser.parse_args()

    cfg = get_config()
    embedder = get_embedding_model(cfg)
    variants = ["raw", "cleaned"] if args.pipeline == "both" else [args.pipeline]
    retrievers = {v: get_retriever(v, embedder, cfg) for v in variants}

    if args.question:
        show(args.question, retrievers, args.top_k)
        return
    try:
        while question := input("\nAsk a medical question (empty to quit): ").strip():
            show(question, retrievers, args.top_k)
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()

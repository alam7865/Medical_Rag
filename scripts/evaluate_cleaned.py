"""Evaluate cleaned RAG retrieval. Usage: python scripts/evaluate_cleaned.py [--split validate|test]"""

from _runner import run_eval

if __name__ == "__main__":
    run_eval("cleaned")

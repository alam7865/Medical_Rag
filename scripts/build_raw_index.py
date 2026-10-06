"""Build the ChromaDB index for the raw pipeline (collection: medical_rag_raw)."""

from _runner import build

if __name__ == "__main__":
    build("raw")

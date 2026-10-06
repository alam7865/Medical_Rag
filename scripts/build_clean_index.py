"""Build the ChromaDB index for the cleaned pipeline (collection: medical_rag_cleaned)."""

from _runner import build

if __name__ == "__main__":
    build("cleaned")

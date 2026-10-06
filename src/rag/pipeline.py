"""Index building and retriever construction for the two pipelines.

`build_index` is the ONLY place documents enter Chroma, and it is called with
identical settings for "raw" and "cleaned" - the corpus DataFrame is the only
thing that differs.
"""

from __future__ import annotations

import pandas as pd

from src.config import Config, get_config
from src.data.deduplicator import Embedder
from src.data.normalization import matching_key
from src.rag.chunker import chunk_dataframe
from src.rag.retriever import Retriever
from src.utils.helpers import get_logger, sha1_text
from src.vectorstore.chroma_store import ChromaStore

logger = get_logger(__name__)

VARIANTS = ("raw", "cleaned")


def answer_key(answer: str) -> str:
    """Relevance key used for ground truth. Empty answers get an empty key (never relevant)."""
    key = matching_key(answer)
    return sha1_text(key, 16) if key else ""


def get_store(variant: str, config: Config | None = None) -> ChromaStore:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    config = config or get_config()
    return ChromaStore(config.chroma_path(variant), config.collection_name(variant))


def build_index(
    variant: str,
    corpus: pd.DataFrame,
    embedder: Embedder,
    config: Config | None = None,
    dataset_version: str = "",
) -> dict:
    config = config or get_config()
    chunks = chunk_dataframe(corpus, config.chunk_size, config.chunk_overlap)
    if not chunks:
        raise ValueError("Corpus produced no chunks - is it empty?")

    by_id = corpus.set_index("question_id")
    metadatas = []
    for c in chunks:
        row = by_id.loc[c.question_id]
        metadatas.append(
            {
                "question_id": c.question_id,
                "chunk_index": c.chunk_index,
                "n_chunks": c.n_chunks,
                "dataset_split": row["dataset_split"],
                "source": row["source"],
                "question": row["question"],
                "answer": row["answer"],
                "answer_key": answer_key(row["answer"]),
                "pipeline": variant,
            }
        )

    logger.info("[%s] embedding %d chunks from %d records", variant, len(chunks), len(corpus))
    embeddings = embedder.encode([c.text for c in chunks])

    store = get_store(variant, config)
    store.recreate(
        config.distance_metric,
        {**config.experiment_settings(), "eval_k_values": str(list(config.eval_k_values)),
         "ndcg_k_values": str(list(config.ndcg_k_values)), "pipeline": variant, "dataset_version": dataset_version},
    )
    store.add([c.chunk_id for c in chunks], embeddings, [c.text for c in chunks], metadatas)

    stats = {
        "pipeline": variant,
        "collection": config.collection_name(variant),
        "records": int(len(corpus)),
        "records_indexed": len({c.question_id for c in chunks}),
        "chunks": len(chunks),
        "multi_chunk_records": len({c.question_id for c in chunks if c.n_chunks > 1}),
        "dataset_version": dataset_version,
        **config.experiment_settings(),
    }
    logger.info("[%s] index built: %s", variant, stats)
    return stats


def get_retriever(variant: str, embedder: Embedder, config: Config | None = None) -> Retriever:
    config = config or get_config()
    store = get_store(variant, config)
    stored_model = store.settings.get("embedding_model")
    if stored_model and stored_model != config.embedding_model:
        raise RuntimeError(
            f"Index '{variant}' was built with {stored_model} but config uses {config.embedding_model}. Rebuild it."
        )
    return Retriever(store, embedder, config.top_k, config.chunk_fetch_multiplier, config.distance_metric)

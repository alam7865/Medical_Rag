"""Retriever: question -> embedding -> Chroma -> top-K unique documents.

Chroma returns *chunks*. A long record can produce several chunks, so we fetch
TOP_K * CHUNK_FETCH_MULTIPLIER chunks and collapse them to unique parent
records (keeping each record's best-scoring chunk). Metrics are computed on
these record-level rankings for both pipelines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.data.deduplicator import Embedder
from src.vectorstore.chroma_store import ChromaStore


@dataclass(frozen=True)
class RetrievedDoc:
    rank: int
    question_id: str
    question: str
    answer: str
    answer_key: str
    score: float  # similarity (1 - cosine distance)
    chunk_id: str
    chunk_text: str

    def to_dict(self) -> dict:
        return asdict(self)


def distance_to_similarity(distance: float, metric: str) -> float:
    if metric == "cosine":
        return 1.0 - distance
    if metric == "ip":
        return 1.0 - distance  # Chroma reports 1 - dot for "ip"
    return -distance  # l2: smaller is better


def collapse_to_documents(hits: list[dict], top_k: int, metric: str) -> list[RetrievedDoc]:
    docs: list[RetrievedDoc] = []
    seen: set[str] = set()
    for hit in hits:
        meta = hit["metadata"]
        qid = meta["question_id"]
        if qid in seen:
            continue
        seen.add(qid)
        docs.append(
            RetrievedDoc(
                rank=len(docs) + 1,
                question_id=qid,
                question=meta.get("question", ""),
                answer=meta.get("answer", ""),
                answer_key=meta.get("answer_key", ""),
                score=round(distance_to_similarity(hit["distance"], metric), 6),
                chunk_id=hit["id"],
                chunk_text=hit["document"],
            )
        )
        if len(docs) == top_k:
            break
    return docs


class Retriever:
    def __init__(self, store: ChromaStore, embedder: Embedder, top_k: int, fetch_multiplier: int, metric: str) -> None:
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.fetch_multiplier = fetch_multiplier
        self.metric = metric

    def retrieve_batch(self, questions: list[str], top_k: int | None = None, batch_size: int = 256) -> list[list[RetrievedDoc]]:
        k = top_k or self.top_k
        n_fetch = k * self.fetch_multiplier
        results: list[list[RetrievedDoc]] = []
        for s in range(0, len(questions), batch_size):
            batch = questions[s : s + batch_size]
            emb = self.embedder.encode(batch)
            for hits in self.store.query(emb, n_fetch):
                results.append(collapse_to_documents(hits, k, self.metric))
        return results

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedDoc]:
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")
        return self.retrieve_batch([question], top_k)[0]

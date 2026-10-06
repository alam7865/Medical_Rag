"""ChromaDB persistence layer.

We compute embeddings ourselves (src/embeddings) and hand them to Chroma, so
both collections are guaranteed to use exactly the same model. The experiment
settings are written into the collection metadata and verified at query time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings

from src.utils.helpers import get_logger

logger = get_logger(__name__)

_ADD_BATCH = 2000  # stay well below Chroma's max batch size


def _client(path: Path) -> chromadb.ClientAPI:
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


def _sanitize(meta: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata values must be str/int/float/bool (no None)."""
    clean: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            clean[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean[k] = float(v)
        else:
            clean[k] = str(v)
    return clean


class ChromaStore:
    def __init__(self, path: Path, collection_name: str) -> None:
        self.path = path
        self.collection_name = collection_name
        self._client = _client(path)
        self._collection = None

    # -- build -------------------------------------------------------------------
    def recreate(self, distance_metric: str, settings: dict[str, Any]) -> None:
        """Drop and recreate the collection so every build starts from scratch."""
        try:
            self._client.delete_collection(self.collection_name)
            logger.info("Deleted existing collection %s", self.collection_name)
        except Exception:
            pass
        self._collection = self._client.create_collection(
            name=self.collection_name,
            configuration={"hnsw": {"space": distance_metric, "ef_construction": 200, "ef_search": 200}},
            metadata=_sanitize(settings),
            embedding_function=None,
        )

    def add(self, ids: list[str], embeddings: np.ndarray, documents: list[str], metadatas: list[dict]) -> None:
        if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
            raise ValueError("ids, embeddings, documents and metadatas must have the same length")
        col = self.collection
        for s in range(0, len(ids), _ADD_BATCH):
            e = s + _ADD_BATCH
            col.add(
                ids=ids[s:e],
                embeddings=embeddings[s:e].tolist(),
                documents=documents[s:e],
                metadatas=[_sanitize(m) for m in metadatas[s:e]],
            )

    # -- query -------------------------------------------------------------------
    @property
    def collection(self):
        if self._collection is None:
            try:
                self._collection = self._client.get_collection(self.collection_name, embedding_function=None)
            except Exception as exc:
                raise RuntimeError(
                    f"Collection '{self.collection_name}' not found in {self.path}. Build the index first."
                ) from exc
        return self._collection

    @property
    def settings(self) -> dict[str, Any]:
        return dict(self.collection.metadata or {})

    def count(self) -> int:
        return self.collection.count()

    def query(self, query_embeddings: np.ndarray, n_results: int) -> list[list[dict[str, Any]]]:
        """Returns, per query, a ranked list of {id, document, metadata, distance}."""
        n_results = max(1, min(n_results, self.count()))
        res = self.collection.query(
            query_embeddings=query_embeddings.tolist(),
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for ids, docs, metas, dists in zip(res["ids"], res["documents"], res["metadatas"], res["distances"]):
            out.append(
                [
                    {"id": i, "document": d, "metadata": m, "distance": float(dist)}
                    for i, d, m, dist in zip(ids, docs, metas, dists)
                ]
            )
        return out

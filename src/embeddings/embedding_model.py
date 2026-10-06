"""Thin wrapper around a pre-trained Sentence Transformers model.

The model is used as-is (no training / fine-tuning). Embeddings are
L2-normalized, so dot product == cosine similarity.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from src.config import Config, get_config
from src.utils.helpers import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    def __init__(self, model_name: str, batch_size: int = 64, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.model_name = model_name
        self.batch_size = batch_size
        logger.info("Loading embedding model %s", model_name)
        self._model = SentenceTransformer(model_name, device=device)
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dimension = int(get_dim())

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        emb = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return emb.astype(np.float32)


@lru_cache(maxsize=4)
def _cached(model_name: str, batch_size: int, device: str | None) -> EmbeddingModel:
    return EmbeddingModel(model_name, batch_size, device)


def get_embedding_model(config: Config | None = None) -> EmbeddingModel:
    """One shared instance per (model, batch size, device)."""
    config = config or get_config()
    return _cached(config.embedding_model, config.embedding_batch_size, config.embedding_device)

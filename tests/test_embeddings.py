"""Real sentence-transformers model. Skipped when the model is not available locally/offline."""

import numpy as np
import pytest

from src.config import get_config


@pytest.fixture(scope="module")
def model():
    try:
        from src.embeddings.embedding_model import EmbeddingModel
        return EmbeddingModel(get_config().embedding_model, batch_size=8)
    except Exception as exc:  # no network / not cached
        pytest.skip(f"embedding model unavailable: {exc}")


def test_shape_and_normalization(model):
    emb = model.encode(["What causes gout?", "How is asthma treated?"])
    assert emb.shape == (2, model.dimension)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)


def test_empty_input(model):
    assert model.encode([]).shape == (0, model.dimension)


def test_paraphrase_is_closer_than_unrelated(model):
    q, para, other = model.encode(["What are the symptoms of the flu?", "What does influenza feel like?", "How are kidney stones removed?"])
    assert q @ para > q @ other

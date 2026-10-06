"""Shared fixtures. A deterministic bag-of-words embedder keeps tests fast and offline."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config  # noqa: E402


class FakeEmbedder:
    """Hashing bag-of-words embedder: texts sharing words have higher cosine similarity."""

    dimension = 256
    model_name = "fake-bow"

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", str(t).lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dimension] += 1.0
            n = np.linalg.norm(out[i])
            out[i] = out[i] / n if n else out[i]
        return out


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    (tmp_path / "raw").mkdir()
    return Config(
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        chroma_dir=tmp_path / "chroma",
        reports_dir=tmp_path / "reports",
        question_column=None,
        answer_column=None,
        id_column=None,
        source_column=None,
        embedding_model="fake-bow",
        chunk_size=300,
        chunk_overlap=50,
        top_k=10,
        min_question_chars=5,
        min_answer_chars=10,
        exclude_leaked_eval_questions=True,
    )


DIABETES = "Type 2 diabetes is diagnosed when HbA1c is 6.5% or higher or fasting glucose is 126 mg/dL or higher."
ASTHMA = "Asthma is treated with inhaled corticosteroids and short-acting bronchodilators such as albuterol."
GOUT = "Gout causes sudden intense pain and swelling in a joint, often the big toe."


@pytest.fixture
def tiny_dataset(cfg: Config) -> Config:
    """Writes a tiny noisy train/validate/test set into cfg.raw_data_dir."""
    train = pd.DataFrame(
        {
            "Question": [
                "How is type 2 diabetes diagnosed?",
                "How is type 2 diabetes diagnosed?",  # exact duplicate
                "HOW IS TYPE 2 DIABETES DIAGNOSED ?",  # normalized duplicate
                "How is asthma treated?",
                "What does gout feel like?",
                "How do doctors diagnose type 2 diabetes?",  # empty-answer distractor
                None,
                "test",
            ],
            "Answer": [DIABETES, DIABETES, f"<p>{DIABETES}</p>", ASTHMA + " Click here to subscribe to our newsletter.",
                       GOUT, "", "Orphan answer without a question here.", "test"],
            "source": ["s"] * 8,
        }
    )
    validate = pd.DataFrame({"Question": ["Which medicines treat asthma?"], "Answer": [ASTHMA], "source": ["s"]})
    test = pd.DataFrame(
        {
            "Question": ["Which test diagnoses type 2 diabetes?", "What are gout symptoms?", "How is asthma treated?", "Unknown?"],
            "Answer": [DIABETES, GOUT, ASTHMA, "An answer that does not exist in the corpus at all."],
            "source": ["s"] * 4,
        }
    )
    train.to_csv(cfg.raw_data_dir / "train.csv", index=False)
    validate.to_csv(cfg.raw_data_dir / "validate.csv", index=False)
    test.to_csv(cfg.raw_data_dir / "test.csv", index=False)
    return cfg

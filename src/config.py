"""Central configuration.

Every experiment parameter lives here so that the Raw and Cleaned pipelines are
guaranteed to share the same settings. Values can be overridden through
environment variables or a `.env` file at the project root.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is optional at import time
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_opt(name: str) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else None


def _env_int(name: str, default: int) -> int:
    return int(_env_str(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env_str(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return _env_str(name, str(default)).strip().lower() in {"1", "true", "yes", "y"}


def _path(name: str, default: str) -> Path:
    p = Path(_env_str(name, default))
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class Config:
    # --- Data -----------------------------------------------------------------
    raw_data_dir: Path = field(default_factory=lambda: _path("RAW_DATA_DIR", "data/raw"))
    processed_data_dir: Path = field(default_factory=lambda: _path("PROCESSED_DATA_DIR", "data/processed"))
    train_file: str = field(default_factory=lambda: _env_str("TRAIN_FILE", "train.csv"))
    validate_file: str = field(default_factory=lambda: _env_str("VALIDATE_FILE", "validate.csv"))
    test_file: str = field(default_factory=lambda: _env_str("TEST_FILE", "test.csv"))
    cleaned_train_file: str = field(default_factory=lambda: _env_str("CLEANED_TRAIN_FILE", "cleaned_train.csv"))

    # Column names. None => auto-detect from common aliases (see data/loader.py).
    question_column: str | None = field(default_factory=lambda: _env_opt("QUESTION_COLUMN"))
    answer_column: str | None = field(default_factory=lambda: _env_opt("ANSWER_COLUMN"))
    id_column: str | None = field(default_factory=lambda: _env_opt("ID_COLUMN"))
    source_column: str | None = field(default_factory=lambda: _env_opt("SOURCE_COLUMN"))

    # --- Embeddings / vector store ---------------------------------------------
    embedding_model: str = field(
        default_factory=lambda: _env_str("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    embedding_batch_size: int = field(default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 64))
    embedding_device: str | None = field(default_factory=lambda: _env_opt("EMBEDDING_DEVICE"))
    distance_metric: str = field(default_factory=lambda: _env_str("DISTANCE_METRIC", "cosine"))
    chroma_dir: Path = field(default_factory=lambda: _path("CHROMA_DIR", "chroma"))
    raw_collection: str = field(default_factory=lambda: _env_str("RAW_COLLECTION", "medical_rag_raw"))
    cleaned_collection: str = field(default_factory=lambda: _env_str("CLEANED_COLLECTION", "medical_rag_cleaned"))

    # --- Chunking / retrieval (shared by both pipelines) -----------------------
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 200))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 10))
    # Chunks fetched per query = top_k * multiplier, then collapsed to unique documents.
    chunk_fetch_multiplier: int = field(default_factory=lambda: _env_int("CHUNK_FETCH_MULTIPLIER", 3))

    # --- Cleaning ---------------------------------------------------------------
    min_question_chars: int = field(default_factory=lambda: _env_int("MIN_QUESTION_CHARS", 10))
    min_answer_chars: int = field(default_factory=lambda: _env_int("MIN_ANSWER_CHARS", 20))
    lowercase_documents: bool = field(default_factory=lambda: _env_bool("LOWERCASE_DOCUMENTS", False))

    # --- Leakage / evaluation ---------------------------------------------------
    semantic_dup_threshold: float = field(default_factory=lambda: _env_float("SEMANTIC_DUP_THRESHOLD", 0.95))
    semantic_leak_threshold: float = field(default_factory=lambda: _env_float("SEMANTIC_LEAK_THRESHOLD", 0.90))
    exclude_leaked_eval_questions: bool = field(
        default_factory=lambda: _env_bool("EXCLUDE_LEAKED_EVAL_QUESTIONS", True)
    )
    eval_k_values: tuple[int, ...] = (1, 3, 5, 10)
    ndcg_k_values: tuple[int, ...] = (5, 10)
    random_seed: int = field(default_factory=lambda: _env_int("RANDOM_SEED", 42))

    # --- Output -----------------------------------------------------------------
    reports_dir: Path = field(default_factory=lambda: _path("REPORTS_DIR", "reports"))

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.top_k < max(self.eval_k_values):
            raise ValueError(f"TOP_K ({self.top_k}) must be >= the largest eval K ({max(self.eval_k_values)})")
        if self.distance_metric not in {"cosine", "l2", "ip"}:
            raise ValueError("DISTANCE_METRIC must be one of: cosine, l2, ip")

    # Derived paths ----------------------------------------------------------------
    def split_path(self, split: str) -> Path:
        files = {"train": self.train_file, "validate": self.validate_file, "test": self.test_file}
        if split not in files:
            raise ValueError(f"Unknown split '{split}'. Expected one of {list(files)}")
        return self.raw_data_dir / files[split]

    @property
    def cleaned_train_path(self) -> Path:
        return self.processed_data_dir / self.cleaned_train_file

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def results_dir(self) -> Path:
        return self.reports_dir / "results"

    def chroma_path(self, variant: str) -> Path:
        return self.chroma_dir / variant

    def collection_name(self, variant: str) -> str:
        return {"raw": self.raw_collection, "cleaned": self.cleaned_collection}[variant]

    def experiment_settings(self) -> dict:
        """The settings that MUST be identical between Raw and Cleaned runs."""
        return {
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "chunk_fetch_multiplier": self.chunk_fetch_multiplier,
            "distance_metric": self.distance_metric,
            "lowercase_documents": self.lowercase_documents,
            "eval_k_values": list(self.eval_k_values),
            "ndcg_k_values": list(self.ndcg_k_values),
            "exclude_leaked_eval_questions": self.exclude_leaked_eval_questions,
        }

    def to_dict(self) -> dict:
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}


def get_config() -> Config:
    return Config()

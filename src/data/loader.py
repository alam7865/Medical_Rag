"""Load dataset splits into one standard schema.

Every split is converted to the columns:

    question_id | question | answer | source | dataset_split

Column names in the CSV are auto-detected from common aliases unless set
explicitly in the config, so a different dataset can be dropped in without
code changes. Values are NOT cleaned here - that is the cleaner's job - only
missing values are turned into empty strings so the data is usable.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import Config, get_config
from src.data.normalization import to_text
from src.utils.helpers import get_logger, sha1_text

logger = get_logger(__name__)

STANDARD_COLUMNS = ["question_id", "question", "answer", "source", "dataset_split"]

QUESTION_ALIASES = ("question", "questions", "query", "input", "prompt", "instruction", "q", "patient_question")
ANSWER_ALIASES = ("answer", "answers", "response", "output", "completion", "a", "doctor_answer", "text")
ID_ALIASES = ("question_id", "qid", "id", "uid", "record_id")
SOURCE_ALIASES = ("source", "dataset", "origin", "document_source")


class SchemaError(ValueError):
    """Raised when required columns cannot be found."""


@dataclass(frozen=True)
class ColumnMapping:
    question: str
    answer: str
    id: str | None
    source: str | None


def _find_column(columns: list[str], explicit: str | None, aliases: tuple[str, ...], required: bool) -> str | None:
    if explicit:
        if explicit not in columns:
            raise SchemaError(f"Configured column '{explicit}' not found. Available: {columns}")
        return explicit
    lowered = {c.lower().strip(): c for c in columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    if required:
        raise SchemaError(
            f"Could not auto-detect a column from aliases {aliases}. Available: {columns}. "
            "Set QUESTION_COLUMN / ANSWER_COLUMN in .env."
        )
    return None


def detect_columns(df: pd.DataFrame, config: Config | None = None) -> ColumnMapping:
    config = config or get_config()
    cols = list(df.columns)
    question = _find_column(cols, config.question_column, QUESTION_ALIASES, required=True)
    answer = _find_column(
        [c for c in cols if c != question], config.answer_column, ANSWER_ALIASES, required=True
    )
    id_col = _find_column(cols, config.id_column, ID_ALIASES, required=False)
    source = _find_column(cols, config.source_column, SOURCE_ALIASES, required=False)
    return ColumnMapping(question=question, answer=answer, id=id_col, source=source)


def generate_ids(questions: list[str], answers: list[str], split: str) -> list[str]:
    """Deterministic, content-based IDs (not row-position based).

    id = "<split>-<sha1(question || answer)>-<n>", where n disambiguates exact
    duplicate rows. Re-ordering non-identical rows never changes an ID.
    """
    seen: dict[str, int] = defaultdict(int)
    ids: list[str] = []
    for q, a in zip(questions, answers):
        h = sha1_text(f"{q}␟{a}")
        ids.append(f"{split}-{h}-{seen[h]}")
        seen[h] += 1
    return ids


def standardize(df: pd.DataFrame, split: str, config: Config | None = None) -> pd.DataFrame:
    """Map an arbitrary QA DataFrame into the standard schema."""
    config = config or get_config()
    mapping = detect_columns(df, config)

    questions = [to_text(v) for v in df[mapping.question].tolist()]
    answers = [to_text(v) for v in df[mapping.answer].tolist()]

    if mapping.id is not None:
        ids = [to_text(v) for v in df[mapping.id].tolist()]
        # Fall back to generated IDs for missing ones; make duplicates unique.
        generated = generate_ids(questions, answers, split)
        counts: dict[str, int] = defaultdict(int)
        fixed = []
        for raw_id, gen_id in zip(ids, generated):
            base = f"{split}-{raw_id}" if raw_id else gen_id
            fixed.append(base if counts[base] == 0 else f"{base}-dup{counts[base]}")
            counts[base] += 1
        ids = fixed
    else:
        ids = generate_ids(questions, answers, split)

    sources = (
        [to_text(v) or "unknown" for v in df[mapping.source].tolist()]
        if mapping.source
        else ["unknown"] * len(df)
    )

    out = pd.DataFrame(
        {
            "question_id": ids,
            "question": questions,
            "answer": answers,
            "source": sources,
            "dataset_split": split,
        }
    )
    assert out["question_id"].is_unique, "question_id must be unique"
    return out


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    try:
        # keep_default_na=True so 'NA', 'null', '' become NaN -> treated as missing.
        return pd.read_csv(path, dtype=str, keep_default_na=True, on_bad_lines="warn", encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed for %s, retrying with latin-1", path)
        return pd.read_csv(path, dtype=str, keep_default_na=True, on_bad_lines="warn", encoding="latin-1")


def load_split(split: str, config: Config | None = None) -> pd.DataFrame:
    config = config or get_config()
    path = config.split_path(split)
    df = standardize(read_csv(path), split, config)
    logger.info("Loaded %s: %d rows from %s", split, len(df), path.name)
    return df


def load_all_splits(config: Config | None = None) -> dict[str, pd.DataFrame]:
    config = config or get_config()
    return {s: load_split(s, config) for s in ("train", "validate", "test")}


def load_cleaned_train(config: Config | None = None) -> pd.DataFrame:
    config = config or get_config()
    path = config.cleaned_train_path
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/clean_dataset.py first.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(STANDARD_COLUMNS) - set(df.columns)
    if missing:
        raise SchemaError(f"Cleaned file is missing columns: {missing}")
    return df

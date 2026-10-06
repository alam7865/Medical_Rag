"""Turn QA records into retrieval chunks.

A record is rendered as "Question: ...\\nAnswer: ...". If that fits in
CHUNK_SIZE characters it becomes one chunk. Otherwise the *answer* is split
into overlapping windows (preferring sentence / word boundaries) and every
chunk is prefixed with the question so it stays self-contained.

The same function is used for both pipelines; only the input differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

_BOUNDARIES = ("\n", ". ", "? ", "! ", "; ", ", ", " ")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    question_id: str
    chunk_index: int
    n_chunks: int
    text: str


def format_document(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into windows of at most `chunk_size` chars with ~`chunk_overlap` overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # Back off to the last natural boundary in the second half of the window.
            window = text[start:end]
            for sep in _BOUNDARIES:
                cut = window.rfind(sep)
                if cut > chunk_size // 2:
                    end = start + cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        next_start = end - chunk_overlap
        # Snap the overlap start forward to a word boundary so chunks don't begin mid-word.
        space = text.find(" ", next_start, end)
        if space != -1:
            next_start = space + 1
        start = max(next_start, start + 1)  # always make progress
    return chunks


def chunk_record(question_id: str, question: str, answer: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    full = format_document(question, answer)
    if len(full) <= chunk_size:
        pieces = [full]
    else:
        prefix = f"Question: {question}\nAnswer: "
        # Budget for the answer text; guarantee room even for very long questions.
        budget = max(chunk_size - len(prefix), chunk_size // 2)
        overlap = min(chunk_overlap, budget - 1)
        pieces = [prefix + p for p in split_text(answer, budget, overlap)] or [full]
    return [
        Chunk(chunk_id=f"{question_id}::c{i}", question_id=question_id, chunk_index=i, n_chunks=len(pieces), text=p)
        for i, p in enumerate(pieces)
    ]


def chunk_dataframe(df: pd.DataFrame, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for qid, q, a in zip(df["question_id"], df["question"], df["answer"]):
        if not (str(q).strip() or str(a).strip()):
            continue  # nothing to embed at all
        chunks.extend(chunk_record(str(qid), str(q), str(a), chunk_size, chunk_overlap))
    return chunks

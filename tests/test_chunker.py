import pandas as pd
import pytest

from src.rag.chunker import chunk_dataframe, chunk_record, split_text


def test_short_record_is_one_chunk():
    chunks = chunk_record("id1", "What is gout?", "Gout is arthritis.", 300, 50)
    assert len(chunks) == 1
    assert chunks[0].text == "Question: What is gout?\nAnswer: Gout is arthritis."
    assert chunks[0].chunk_id == "id1::c0"


def test_long_record_is_split_with_question_prefix():
    answer = " ".join(f"Sentence number {i} about osteoporosis." for i in range(80))
    chunks = chunk_record("id2", "What is osteoporosis?", answer, 300, 50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)
    assert all(c.text.startswith("Question: What is osteoporosis?\nAnswer: ") for c in chunks)
    assert {c.n_chunks for c in chunks} == {len(chunks)}


def test_split_text_overlap_and_coverage():
    text = " ".join(f"w{i}" for i in range(400))
    parts = split_text(text, 200, 50)
    assert all(len(p) <= 200 for p in parts)
    assert parts[0].split()[0] == "w0" and parts[-1].split()[-1] == "w399"
    # consecutive chunks share some words
    assert set(parts[0].split()) & set(parts[1].split())


def test_split_text_never_loops_on_unbreakable_text():
    parts = split_text("x" * 1000, 100, 20)
    assert all(len(p) <= 100 for p in parts) and len(parts) >= 10


def test_invalid_params():
    with pytest.raises(ValueError):
        split_text("abc", 100, 100)
    with pytest.raises(ValueError):
        split_text("abc", 0, 0)


def test_chunk_dataframe_skips_fully_empty_rows():
    df = pd.DataFrame({"question_id": ["a", "b"], "question": ["", "Q"], "answer": ["", "A"]})
    assert [c.question_id for c in chunk_dataframe(df, 300, 50)] == ["b"]

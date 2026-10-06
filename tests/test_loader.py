import pandas as pd
import pytest

from src.data.loader import SchemaError, detect_columns, generate_ids, load_split, standardize


def test_autodetects_capitalized_columns(cfg):
    df = pd.DataFrame({"Question": ["q"], "Answer": ["a"], "Source": ["x"]})
    m = detect_columns(df, cfg)
    assert (m.question, m.answer, m.source, m.id) == ("Question", "Answer", "Source", None)


def test_autodetects_alternative_names(cfg):
    m = detect_columns(pd.DataFrame({"input": ["q"], "output": ["a"], "id": [1]}), cfg)
    assert (m.question, m.answer, m.id) == ("input", "output", "id")


def test_missing_columns_raise(cfg):
    with pytest.raises(SchemaError):
        detect_columns(pd.DataFrame({"foo": [1], "bar": [2]}), cfg)


def test_ids_are_deterministic_and_unique():
    q, a = ["q1", "q2", "q1"], ["a1", "a2", "a1"]
    ids = generate_ids(q, a, "train")
    assert len(set(ids)) == 3
    assert ids == generate_ids(q, a, "train")
    # Not position-based: reordering distinct rows keeps their IDs.
    assert generate_ids(["q2", "q1"], ["a2", "a1"], "train") == [ids[1], ids[0]]


def test_missing_values_become_empty_strings(cfg):
    df = standardize(pd.DataFrame({"question": [None, "q"], "answer": ["a", float("nan")]}), "train", cfg)
    assert df["question"].tolist() == ["", "q"]
    assert df["answer"].tolist() == ["a", ""]
    assert df["source"].tolist() == ["unknown", "unknown"]


def test_existing_id_column_is_used_and_made_unique(cfg):
    df = standardize(pd.DataFrame({"qid": ["7", "7"], "question": ["a", "b"], "answer": ["x", "y"]}), "test", cfg)
    assert df["question_id"].tolist() == ["test-7", "test-7-dup1"]


def test_load_split_reads_csv(tiny_dataset):
    df = load_split("train", tiny_dataset)
    assert len(df) == 8 and df["dataset_split"].eq("train").all()


def test_load_split_missing_file(cfg):
    with pytest.raises(FileNotFoundError):
        load_split("train", cfg)

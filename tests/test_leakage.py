import pandas as pd

from src.data.leakage import find_overlap, leaked_eval_ids


def df(qs, split):
    return pd.DataFrame({"question_id": [f"{split}-{i}" for i in range(len(qs))], "question": qs})


def test_exact_normalized_and_semantic_levels(embedder):
    train = df(["What is gout?", "How is asthma treated?", "What causes kidney stones?"], "train")
    test = df(["What is gout?", "  HOW IS ASTHMA TREATED  ", "What causes kidney stones in adults?", "Unrelated xyz"], "test")
    res = find_overlap(train, test, "train", "test", embedder, semantic_threshold=0.8)
    assert res.exact_ids == {"test-0"}
    assert res.normalized_ids == {"test-1"}
    assert list(res.semantic["question_id"]) == ["test-2"]


def test_empty_questions_never_overlap():
    assert leaked_eval_ids(df(["", "a b c"], "train"), df(["", "  "], "test")) == set()

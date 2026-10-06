"""Index building, retrieval and end-to-end evaluation on a tiny dataset (fake embedder, temp Chroma)."""

import pandas as pd
import pytest

from src.data.cleaner import clean_dataframe
from src.data.loader import load_split
from src.evaluation.error_analysis import run_error_analysis
from src.evaluation.evaluator import build_eval_set, evaluate
from src.rag.pipeline import answer_key, build_index, get_retriever
from src.rag.retriever import collapse_to_documents


@pytest.fixture
def built(tiny_dataset, embedder):
    cfg = tiny_dataset
    raw = load_split("train", cfg)
    res = clean_dataframe(raw, cfg)
    cfg.processed_data_dir.mkdir(parents=True, exist_ok=True)
    res.cleaned.to_csv(cfg.cleaned_train_path, index=False)
    build_index("raw", raw, embedder, cfg)
    build_index("cleaned", res.cleaned, embedder, cfg)
    return cfg, res


def test_answer_key_ignores_formatting_and_empty_is_blank():
    assert answer_key("<p>Gout   hurts.</p>") == answer_key("gout hurts")
    assert answer_key("") == "" and answer_key("<p></p>") == ""


def test_indexes_have_expected_sizes(built, embedder):
    cfg, res = built
    raw_r = get_retriever("raw", embedder, cfg)
    clean_r = get_retriever("cleaned", embedder, cfg)
    assert raw_r.store.count() == 8  # every non-empty raw row is indexed
    assert clean_r.store.count() == len(res.cleaned) == 3


def test_retrieve_returns_ranked_unique_docs(built, embedder):
    cfg, _ = built
    docs = get_retriever("cleaned", embedder, cfg).retrieve("How is asthma treated?", top_k=3)
    assert docs[0].question == "How is asthma treated?"
    assert [d.rank for d in docs] == list(range(1, len(docs) + 1))
    assert len({d.question_id for d in docs}) == len(docs)
    assert "Click here" not in docs[0].answer  # boilerplate cleaned


def test_retrieve_rejects_blank_question(built, embedder):
    cfg, _ = built
    with pytest.raises(ValueError):
        get_retriever("raw", embedder, cfg).retrieve("   ")


def test_collapse_keeps_best_chunk_per_record():
    hits = [
        {"id": "a::c1", "document": "x", "distance": 0.1, "metadata": {"question_id": "a"}},
        {"id": "a::c0", "document": "y", "distance": 0.2, "metadata": {"question_id": "a"}},
        {"id": "b::c0", "document": "z", "distance": 0.3, "metadata": {"question_id": "b"}},
    ]
    docs = collapse_to_documents(hits, top_k=5, metric="cosine")
    assert [(d.question_id, d.chunk_id) for d in docs] == [("a", "a::c1"), ("b", "b::c0")]
    assert docs[0].score == pytest.approx(0.9)


def test_eval_set_is_identical_and_exclusions_reported(built):
    cfg, _ = built
    es = build_eval_set("test", cfg)
    assert es.exclusions["leaked_from_train_exact_or_normalized"] == 1  # "How is asthma treated?"
    assert es.exclusions["answer_not_in_corpus"] == 1
    assert es.exclusions["evaluated"] == 2
    with pytest.raises(ValueError):
        build_eval_set("train", cfg)


def test_end_to_end_evaluation_and_error_analysis(built, embedder):
    cfg, res = built
    s_raw = evaluate("raw", "test", embedder, cfg, semantic=True)
    s_cln = evaluate("cleaned", "test", embedder, cfg, semantic=True)
    assert s_raw["n_questions"] == s_cln["n_questions"] == 2
    assert s_raw["experiment_settings"] == s_cln["experiment_settings"]
    for s in (s_raw, s_cln):
        m = s["metrics"]
        assert 0 <= m["recall@1"] <= m["recall@3"] <= m["recall@5"] <= m["recall@10"] <= 1
        assert 0 <= m["mrr"] <= 1 and 0 <= m["ndcg@10"] <= 1
    raw_pq = pd.read_csv(cfg.results_dir / "raw_test_per_question.csv")
    cln_pq = pd.read_csv(cfg.results_dir / "cleaned_test_per_question.csv")
    merged, summary = run_error_analysis(raw_pq, cln_pq, res.log, k=1)
    assert sum(summary["category_counts"].values()) == 2
    assert {"category", "explanation", "raw_top1_cleaning_status"} <= set(merged.columns)

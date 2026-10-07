import pytest
from fastapi.testclient import TestClient

from src.api import app as api
from src.data.cleaner import clean_dataframe
from src.data.loader import load_split
from src.rag.pipeline import build_index


@pytest.fixture
def client(tiny_dataset, embedder, monkeypatch):
    cfg = tiny_dataset
    raw = load_split("train", cfg)
    build_index("raw", raw, embedder, cfg)
    build_index("cleaned", clean_dataframe(raw, cfg).cleaned, embedder, cfg)
    monkeypatch.setattr(api, "get_config", lambda: cfg)
    monkeypatch.setattr(api, "get_embedding_model", lambda c=None: embedder)
    api._retriever.cache_clear()
    yield TestClient(api.app)
    api._retriever.cache_clear()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["indexes"]["raw"]["chunks"] == 8


def test_retrieve_endpoints(client):
    for path in ("/retrieve/raw", "/retrieve/cleaned"):
        r = client.post(path, json={"question": "How is asthma treated?", "top_k": 2})
        assert r.status_code == 200
        assert len(r.json()["results"]) <= 2


def test_compare(client):
    body = client.post("/compare", json={"question": "How do doctors diagnose type 2 diabetes?", "top_k": 3}).json()
    assert set(body) == {"raw_results", "cleaned_results", "comparison"}
    assert body["comparison"]["raw_empty_answers_in_results"] >= 1  # raw still contains the empty-answer row


def test_validation_errors(client):
    assert client.post("/compare", json={"question": "", "top_k": 3}).status_code == 422
    assert client.post("/compare", json={"question": "   ", "top_k": 3}).status_code == 422
    assert client.post("/compare", json={"question": "x", "top_k": 0}).status_code == 422


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Raw vs Clean RAG" in r.text


def test_metrics_and_examples(client):
    assert set(client.get("/metrics").json()) == {"evaluation", "cleaning"}
    qs = client.get("/examples?n=1").json()["questions"]
    assert qs == ["Which medicines treat asthma?"]  # from validate, never test


def test_raw_results_carry_cleaning_status(client, tiny_dataset):
    from src.api import app as api
    from src.data.cleaner import clean_dataframe
    from src.data.loader import load_split
    log = clean_dataframe(load_split("train", tiny_dataset), tiny_dataset).log
    tiny_dataset.reports_dir.mkdir(parents=True, exist_ok=True)
    log.to_csv(tiny_dataset.reports_dir / "cleaning_log.csv", index=False)
    api._cleaning_status.cache_clear()
    body = client.post("/compare", json={"question": "How do doctors diagnose type 2 diabetes?", "top_k": 3}).json()
    api._cleaning_status.cache_clear()
    statuses = [d["cleaning_status"] for d in body["raw_results"]]
    assert any(s and s.startswith("removed:") for s in statuses)
    assert all(d["cleaning_status"] is None for d in body["cleaned_results"])


def test_concurrent_first_requests_do_not_crash(client):
    """Regression: parallel cold-start requests used to race inside ChromaDB's client registry."""
    from concurrent.futures import ThreadPoolExecutor

    from src.api import app as api
    api._retriever.cache_clear()
    calls = [lambda: client.get("/health"), lambda: client.post("/compare", json={"question": "How is asthma treated?", "top_k": 2})] * 4
    with ThreadPoolExecutor(max_workers=8) as pool:
        codes = [f.result().status_code for f in [pool.submit(c) for c in calls]]
    assert all(c == 200 for c in codes)

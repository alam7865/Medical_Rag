"""Optional FastAPI retrieval service (separate from the offline evaluation).

Run:  uvicorn src.api.app:app --reload
UI:   http://127.0.0.1:8000/
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config import get_config
from src.embeddings.embedding_model import get_embedding_model
from src.rag.pipeline import VARIANTS, get_retriever
from src.rag.retriever import Retriever
from src.utils.helpers import load_json

app = FastAPI(title="Medical RAG - Raw vs Cleaned retrieval", version="1.0.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What are the symptoms of diabetes?"])
    top_k: int = Field(5, ge=1, le=50)


class Doc(BaseModel):
    rank: int
    question_id: str
    question: str
    answer: str
    score: float
    cleaning_status: str | None = None  # raw docs: kept / kept_flagged / removed:<reason>


class RetrieveResponse(BaseModel):
    pipeline: str
    results: list[Doc]


class CompareResponse(BaseModel):
    raw_results: list[Doc]
    cleaned_results: list[Doc]
    comparison: dict


@lru_cache(maxsize=2)
def _retriever(variant: str) -> Retriever:
    cfg = get_config()
    return get_retriever(variant, get_embedding_model(cfg), cfg)


@lru_cache(maxsize=1)
def _cleaning_status() -> dict[str, str]:
    """question_id -> what the cleaner did with that raw record (empty if not cleaned yet)."""
    path = get_config().reports_dir / "cleaning_log.csv"
    if not path.exists():
        return {}
    log = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {
        qid: (action if action != "removed" else f"removed:{reason}")
        for qid, action, reason in zip(log["question_id"], log["action"], log["reason"])
    }


def _retrieve(variant: str, req: RetrieveRequest) -> list[Doc]:
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be blank")
    try:
        docs = _retriever(variant).retrieve(req.question, req.top_k)
    except RuntimeError as exc:  # index missing / model mismatch
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    status = _cleaning_status() if variant == "raw" else {}
    return [
        Doc(rank=d.rank, question_id=d.question_id, question=d.question, answer=d.answer, score=d.score,
            cleaning_status=status.get(d.question_id))
        for d in docs
    ]


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/metrics")
def metrics() -> dict:
    """Offline evaluation summary + cleaning report, for the UI dashboard."""
    cfg = get_config()
    out: dict = {}
    for name, fname in (("evaluation", "evaluation_summary.json"), ("cleaning", "data_cleaning_report.json")):
        path = cfg.reports_dir / fname
        out[name] = load_json(path) if path.exists() else None
    return out


@app.get("/examples")
def example_questions(n: int = 8) -> dict:
    """Sample questions from the validation split (never test) for the UI's suggestion chips."""
    try:
        df = pd.read_csv(get_config().split_path("validate"), dtype=str, keep_default_na=False)
        col = next(c for c in df.columns if c.lower() in ("question", "query", "input", "prompt"))
        qs = [q.strip() for q in df[col] if q.strip()]
    except (FileNotFoundError, StopIteration):
        qs = []
    return {"questions": random.sample(qs, min(n, len(qs)))}


@app.get("/health")
def health() -> dict:
    cfg = get_config()
    status = {}
    for v in VARIANTS:
        try:
            status[v] = {"collection": cfg.collection_name(v), "chunks": _retriever(v).store.count()}
        except Exception as exc:  # report, don't crash the health check
            status[v] = {"error": str(exc)}
    ok = all("error" not in s for s in status.values())
    return {"status": "ok" if ok else "degraded", "embedding_model": cfg.embedding_model, "indexes": status}


@app.post("/retrieve/raw", response_model=RetrieveResponse)
def retrieve_raw(req: RetrieveRequest) -> RetrieveResponse:
    return RetrieveResponse(pipeline="raw", results=_retrieve("raw", req))


@app.post("/retrieve/cleaned", response_model=RetrieveResponse)
def retrieve_cleaned(req: RetrieveRequest) -> RetrieveResponse:
    return RetrieveResponse(pipeline="cleaned", results=_retrieve("cleaned", req))


@app.post("/compare", response_model=CompareResponse)
def compare(req: RetrieveRequest) -> CompareResponse:
    raw, cleaned = _retrieve("raw", req), _retrieve("cleaned", req)
    raw_answers = [d.answer.strip() for d in raw]
    return CompareResponse(
        raw_results=raw,
        cleaned_results=cleaned,
        comparison={
            "raw_top1_score": raw[0].score if raw else None,
            "cleaned_top1_score": cleaned[0].score if cleaned else None,
            "raw_empty_answers_in_results": sum(1 for a in raw_answers if not a),
            "raw_duplicate_answers_in_results": len(raw_answers) - len(set(raw_answers)),
            "shared_question_ids": sorted({d.question_id for d in raw} & {d.question_id for d in cleaned}),
        },
    )

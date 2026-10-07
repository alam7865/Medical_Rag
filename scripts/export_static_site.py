"""Export the two ChromaDB indexes + reports into a static site (web/) for Netlify.

The static site runs retrieval entirely in the browser:
  * questions are embedded with transformers.js (ONNX build of the same MiniLM model)
  * search is brute-force cosine similarity over the exported chunk embeddings
  * chunks are collapsed to unique records exactly like src/rag/retriever.py

Run after the pipeline (indexes and reports must exist):
  python scripts/export_static_site.py

Outputs:
  web/index.html, web/rag-engine.js   (copied from src/api/static/ - one UI for both modes)
  web/data/manifest.json  (settings + file list; its presence switches the UI to static mode)
  web/data/raw.json, web/data/cleaned.json   (records, chunk -> record map, base64 float32 embeddings)
  web/data/metrics.json, web/data/examples.json
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import PROJECT_ROOT, get_config  # noqa: E402
from src.rag.pipeline import VARIANTS, get_store  # noqa: E402
from src.utils.helpers import get_logger, load_json, save_json  # noqa: E402

logger = get_logger("export")

# Hugging Face ONNX port of sentence-transformers/all-MiniLM-L6-v2 used by transformers.js.
JS_MODELS = {"sentence-transformers/all-MiniLM-L6-v2": "Xenova/all-MiniLM-L6-v2"}


def cleaning_status_map(reports_dir: Path) -> dict[str, str]:
    path = reports_dir / "cleaning_log.csv"
    if not path.exists():
        return {}
    log = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {
        qid: (action if action != "removed" else f"removed:{reason}")
        for qid, action, reason in zip(log["question_id"], log["action"], log["reason"])
    }


def export_variant(variant: str, cfg, status: dict[str, str]) -> dict:
    col = get_store(variant, cfg).collection
    got = col.get(include=["embeddings", "metadatas"])
    # Sort by chunk id so the export is deterministic regardless of Chroma's internal order.
    order = sorted(range(len(got["ids"])), key=lambda i: got["ids"][i])

    docs: list[dict] = []
    doc_index: dict[str, int] = {}
    chunk_doc: list[int] = []
    emb = np.asarray(got["embeddings"], dtype=np.float32)[order]
    for i in order:
        m = got["metadatas"][i]
        qid = m["question_id"]
        if qid not in doc_index:
            doc_index[qid] = len(docs)
            docs.append({
                "question_id": qid,
                "question": m.get("question", ""),
                "answer": m.get("answer", ""),
                "answer_key": m.get("answer_key", ""),
                "cleaning_status": status.get(qid) if variant == "raw" else None,
            })
        chunk_doc.append(doc_index[qid])

    return {
        "variant": variant,
        "collection": cfg.collection_name(variant),
        "dim": int(emb.shape[1]),
        "n_chunks": int(emb.shape[0]),
        "docs": docs,
        "chunk_doc": chunk_doc,
        "embeddings_b64": base64.b64encode(emb.astype("<f4").tobytes()).decode("ascii"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "web")
    args = parser.parse_args()
    cfg = get_config()

    js_model = JS_MODELS.get(cfg.embedding_model)
    if js_model is None:
        sys.exit(f"No transformers.js model mapped for {cfg.embedding_model}. Add it to JS_MODELS.")
    if cfg.distance_metric != "cosine":
        sys.exit("The static site implements cosine similarity only.")

    data_dir = args.out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("index.html", "rag-engine.js"):
        shutil.copyfile(PROJECT_ROOT / "src/api/static" / fname, args.out / fname)

    status = cleaning_status_map(cfg.reports_dir)
    sizes = {}
    for v in VARIANTS:
        payload = export_variant(v, cfg, status)
        path = data_dir / f"{v}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        sizes[v] = payload["n_chunks"]
        logger.info("[%s] %d chunks / %d records -> %s (%.1f KB)", v, payload["n_chunks"], len(payload["docs"]),
                    path.name, path.stat().st_size / 1024)

    metrics = {}
    for name, fname in (("evaluation", "evaluation_summary.json"), ("cleaning", "data_cleaning_report.json")):
        p = cfg.reports_dir / fname
        metrics[name] = load_json(p) if p.exists() else None
    save_json(metrics, data_dir / "metrics.json")

    validate = pd.read_csv(cfg.split_path("validate"), dtype=str, keep_default_na=False)
    qcol = next(c for c in validate.columns if c.lower() in ("question", "query", "input", "prompt"))
    save_json({"questions": sorted({q.strip() for q in validate[qcol] if q.strip()})}, data_dir / "examples.json")

    save_json({
        "mode": "static",
        "embedding_model": cfg.embedding_model,
        "js_model": js_model,
        # fp32 reproduces the Python embeddings exactly (verified 131/131 top-1 on test);
        # "q8" is a ~4x smaller download with ~1% metric drift.
        "js_dtype": "fp32",
        "pooling": "mean",
        "normalize": True,
        "top_k_fetch_multiplier": cfg.chunk_fetch_multiplier,
        "indexes": {v: {"collection": cfg.collection_name(v), "chunks": sizes[v], "file": f"{v}.json"} for v in VARIANTS},
    }, data_dir / "manifest.json")
    logger.info("Static site written to %s", args.out)


if __name__ == "__main__":
    main()

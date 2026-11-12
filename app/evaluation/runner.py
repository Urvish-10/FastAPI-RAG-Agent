"""
In-process RAG eval runner.

Runs every dataset question through the real graph nodes (rewrite is a no-op for
single-turn questions) one stage at a time, so we can record:
  - raw vector-search chunks   (before reranking)
  - reranked chunks            (what the LLM actually sees, after RERANK_MIN_SCORE)
  - the generated answer
  - per-stage latency

Results are written to eval_results/<timestamp>/ as results.json (raw records),
metrics.json and report.md. Retrieval metrics are deterministic (no LLM).

Prerequisite: `uv run python -m app.evaluation.ingest` (once).

Usage:
    uv run python -m app.evaluation.runner
    uv run python -m app.evaluation.runner --no-generate      # retrieval only (no chat LLM calls)
    uv run python -m app.evaluation.runner --limit 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from app.agent import graph as g
from app.core.config import settings
from app.core.logging import get_logger
from app.evaluation.metrics import compute_metrics, render_markdown

logger = get_logger(__name__)

THIS_DIR = Path(__file__).parent
DEFAULT_DATASET = THIS_DIR / "datasets" / "policy_docs_qa.json"
RESULTS_ROOT = THIS_DIR.parent.parent / "eval_results"


def _slim(chunk: Dict[str, Any], score_key: str) -> Dict[str, Any]:
    meta = chunk.get("metadata", {})
    return {
        "id": chunk.get("id"),
        "document": meta.get("filename"),
        # PyPDF page metadata is 0-based; dataset pages are 1-based
        "page": int(meta.get("page", 0)) + 1,
        score_key: chunk.get(score_key, chunk.get("score")),
        "preview": chunk.get("text", "")[:160],
    }


async def run_question(item: Dict[str, Any], generate: bool) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "messages": [HumanMessage(content=item["question"])],
        "user_id": settings.EVAL_NAMESPACE,   # Pinecone namespace == user_id in the app
        "query": item["question"],
        "document_ids": None,
        "raw_chunks": [],
        "reranked_chunks": [],
        "sources": [],
        "context": "",
    }
    latency: Dict[str, float] = {}
    error: Optional[str] = None
    answer = None

    try:
        t = time.perf_counter()
        state.update(await g.retrieve_node(state))
        latency["retrieve_s"] = round(time.perf_counter() - t, 3)

        t = time.perf_counter()
        state.update(await g.rerank_node(state))
        latency["rerank_s"] = round(time.perf_counter() - t, 3)

        if generate:
            t = time.perf_counter()
            out = await g.generate_node(state)
            latency["generate_s"] = round(time.perf_counter() - t, 3)
            answer = _text(out["messages"][-1].content)
    except Exception as e:  # keep going; record the failure
        logger.error(f"{item['id']} failed: {e}", exc_info=True)
        error = f"{type(e).__name__}: {e}"

    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "expected_answer": item["expected_answer"],
        "evidence": item["evidence"],
        # unique (document, 1-based page) pairs that contain the answer
        "expected": sorted({(e["document"], e["page"]) for e in item["evidence"]}),
        "raw_chunks": [_slim(c, "score") for c in state["raw_chunks"]],
        # rerank_node already dropped chunks below RERANK_MIN_SCORE, so the
        # threshold curve in metrics.py is computed from the unfiltered scores
        # only when RERANK_MIN_SCORE == 0 (the default). Keep it at 0 when evaluating.
        "reranked_chunks": [_slim(c, "rerank_score") for c in state["reranked_chunks"]],
        "context_sources": state["sources"],
        "context": state["context"],   # exact text the LLM saw (needed for faithfulness judging)
        "answer": answer,
        "latency": latency,
        "error": error,
    }


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b if isinstance(b, str) else b.get("text", "")
            for b in content
            if isinstance(b, str) or (isinstance(b, dict) and b.get("type") == "text")
        )
    return ""


async def run(dataset_path: Path, generate: bool, limit: Optional[int], concurrency: int) -> Path:
    dataset = json.loads(dataset_path.read_text())
    if limit:
        dataset = dataset[:limit]

    sem = asyncio.Semaphore(concurrency)

    async def guarded(item):
        async with sem:
            rec = await run_question(item, generate)
            print(f"  {rec['id']:8s} {rec['category']:12s} "
                  f"{'ERR ' + rec['error'] if rec['error'] else 'ok'}")
            return rec

    print(f"Running {len(dataset)} questions (namespace='{settings.EVAL_NAMESPACE}', "
          f"generate={generate}, RERANK_MIN_SCORE={settings.RERANK_MIN_SCORE})")
    records: List[Dict[str, Any]] = await asyncio.gather(*(guarded(i) for i in dataset))

    failed = [r for r in records if r["error"]]
    scored = [r for r in records if not r["error"]]

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset_path.name,
        "chat_model": settings.GEMINI_CHAT_MODEL,
        "embedding_model": f"{settings.GEMINI_EMBEDDING_MODEL} ({settings.GEMINI_EMBEDDING_DIMENSIONS}d)",
        "rerank_model": settings.PINECONE_RERANK_MODEL,
        "retrieval_top_k": settings.RETRIEVAL_TOP_K,
        "rerank_top_n": settings.RERANK_TOP_N,
        "rerank_min_score": settings.RERANK_MIN_SCORE,
        "chunk_size/overlap": f"{settings.CHUNK_SIZE}/{settings.CHUNK_OVERLAP}",
        "failed_questions": len(failed),
    }
    metrics = compute_metrics(scored)

    out_dir = RESULTS_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({"meta": meta, "records": records}, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps({"meta": meta, "metrics": metrics}, indent=2))
    (out_dir / "report.md").write_text(render_markdown(metrics, meta))
    print(f"\nWrote results to {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    out = asyncio.run(run(args.dataset, not args.no_generate, args.limit, args.concurrency))
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()

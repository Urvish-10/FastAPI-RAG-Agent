"""Run a repeatable RAGAS baseline against the live RAG API.

Each golden question uses a new chat session so earlier cases cannot influence
later answers through conversation history.

Run ``python -m app.evaluation.setup_eval_data`` first, then:

    uv run --group eval python -m app.evaluation.ragas_runner
"""

from __future__ import annotations

import asyncio
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig

from app.core.config import settings

BASE_URL = "http://localhost:8000/api/v1"
EVAL_EMAIL = "testevals@yopmail.com"
EVAL_PASSWORD = "Test105*"

THIS_DIR = Path(__file__).parent
DATASETS_DIR = THIS_DIR / "datasets"
MANIFEST_PATH = THIS_DIR / "eval_context.json"
RESULTS_DIR = THIS_DIR / "results"
RESULTS_PATH = RESULTS_DIR / "baseline_results.json"

# The application calls Gemini once per case. Pace the live API requests so a
# baseline run remains usable with conservative provider request-per-minute
# limits, and retry transient provider failures surfaced as API 5xx responses.
REQUEST_INTERVAL_SECONDS = 4
CHAT_RETRY_ATTEMPTS = 3
CHAT_RETRY_DELAY_SECONDS = 60


def load_cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Load only cases whose source fixture was provisioned successfully."""
    provisioned_documents = set(manifest.get("documents", {}))
    cases: list[dict[str, Any]] = []
    for dataset_path in sorted(DATASETS_DIR.glob("*_qa.json")):
        with dataset_path.open(encoding="utf-8") as dataset_file:
            for case in json.load(dataset_file):
                if case["source_document"] in provisioned_documents:
                    cases.append(case)
                else:
                    print(
                        f"[eval] Skipping {case['id']}: fixture "
                        f"{case['source_document']} was not provisioned"
                    )
    if not cases:
        raise RuntimeError("No evaluation cases match the provisioned fixture documents.")
    return cases


async def login(client: httpx.AsyncClient) -> dict[str, str]:
    response = await client.post(
        "/auth/login", json={"email": EVAL_EMAIL, "password": EVAL_PASSWORD}
    )
    response.raise_for_status()
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


async def ask_case(
    client: httpx.AsyncClient, headers: dict[str, str], case: dict[str, Any]
) -> dict[str, Any]:
    """Run one case in an isolated conversation and retain its API evidence."""
    session_response = await client.post(
        "/sessions", headers=headers, json={"title": f"eval-{case['id']}"}
    )
    session_response.raise_for_status()
    session_id = session_response.json()["id"]

    request_payload = {
        "message": case["question"],
        "stream": False,
        "include_retrieved_contexts": True,
    }
    for attempt in range(1, CHAT_RETRY_ATTEMPTS + 1):
        response = await client.post(
            f"/sessions/{session_id}/chat",
            headers=headers,
            json=request_payload,
        )
        if response.is_success:
            break
        is_transient = response.status_code in {429, 500, 502, 503, 504}
        if is_transient and attempt < CHAT_RETRY_ATTEMPTS:
            print(
                f"[eval] {case['id']} received HTTP {response.status_code}; "
                f"retrying in {CHAT_RETRY_DELAY_SECONDS}s "
                f"({attempt}/{CHAT_RETRY_ATTEMPTS})"
            )
            await asyncio.sleep(CHAT_RETRY_DELAY_SECONDS)
            continue
        raise RuntimeError(
            f"Chat failed for {case['id']} with HTTP {response.status_code}: "
            f"{response.text}"
        )
    payload = response.json()
    sources = payload.get("sources") or []
    return {
        "case": case,
        "session_id": session_id,
        "answer": payload["answer"],
        "sources": sources,
        "contexts": payload.get("retrieved_contexts") or [],
    }


def build_evaluator() -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    """Use Gemini for judging, avoiding a second provider or API key."""
    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_CHAT_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0,
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.GEMINI_EMBEDDING_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
    )
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def finite_mean(values: list[Any]) -> float | None:
    numeric_values = [
        float(value)
        for value in values
        if value is not None and not math.isnan(float(value))
    ]
    return round(sum(numeric_values) / len(numeric_values), 4) if numeric_values else None


def summarise(records: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    """Produce overall and category metrics suitable for charts and CI gates."""
    overall = {
        metric: finite_mean([record.get(metric) for record in records])
        for metric in metric_names
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["category"]].append(record)
    return {
        "overall": overall,
        "by_category": {
            category: {
                "case_count": len(category_records),
                "metrics": {
                    metric: finite_mean([record.get(metric) for record in category_records])
                    for metric in metric_names
                },
            }
            for category, category_records in sorted(grouped.items())
        },
    }


async def main() -> None:
    if not MANIFEST_PATH.exists():
        raise RuntimeError(
            "Evaluation manifest is missing. Run `python -m app.evaluation.setup_eval_data` first."
        )
    with MANIFEST_PATH.open(encoding="utf-8") as manifest_file:
        cases = load_cases(json.load(manifest_file))

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120.0) as client:
        headers = await login(client)
        api_results = []
        for index, case in enumerate(cases, start=1):
            print(f"[eval] Running {index}/{len(cases)}: {case['id']}")
            api_results.append(await ask_case(client, headers, case))
            if index < len(cases):
                await asyncio.sleep(REQUEST_INTERVAL_SECONDS)

    dataset_rows = [
        {
            "user_input": result["case"]["question"],
            "retrieved_contexts": result["contexts"],
            "response": result["answer"],
            "reference": result["case"]["expected_answer"],
        }
        for result in api_results
    ]
    evaluator_llm, evaluator_embeddings = build_evaluator()
    metrics = [
        LLMContextPrecisionWithReference(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    ]
    metric_names = [metric.name for metric in metrics]
    evaluation_result = evaluate(
        dataset=EvaluationDataset.from_list(dataset_rows),
        metrics=metrics,
        # Avoid a burst of judge-model calls on small/free provider quotas.
        run_config=RunConfig(max_workers=1),
    )
    metric_records = evaluation_result.to_pandas().to_dict(orient="records")

    records = []
    for api_result, metric_record in zip(api_results, metric_records, strict=True):
        case = api_result["case"]
        records.append(
            {
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "expected_pages": case["expected_pages"],
                "answer": api_result["answer"],
                "sources": api_result["sources"],
                **{metric: metric_record.get(metric) for metric in metric_names},
            }
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "generated_at": datetime.now(UTC).isoformat(),
        "case_count": len(records),
        "metrics": summarise(records, metric_names),
        "cases": records,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[eval] Wrote baseline results to {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())

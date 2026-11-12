"""
LLM judge for the generation side of the RAG evals.

For each record in a run's results.json (produced by the runner WITH generation),
one Gemini call (settings.EVAL_JUDGE_MODEL, temperature 0, structured output)
decides:
  - abstained:   did the answer decline / say the information isn't available?
  - correctness: correct | partial | incorrect vs. the reference answer
                 (not_applicable for unanswerable questions)
  - faithful:    is every factual claim in the answer supported by the retrieved
                 context? (world knowledge does not count as support)

Aggregation (pure, no LLM) is in aggregate_generation().

Usage:
    uv run python -m app.evaluation.judge                          # latest run, default judge
    uv run python -m app.evaluation.judge eval_results/20261112-190859
    uv run python -m app.evaluation.judge --model gemini-2.5-flash # pick the judge model

Results are stored per judge model (judged_<model>.json, report_<model>.md), so a
judge whose daily quota runs out can be swapped, and judges can be compared. Runs
are resumable: re-running with the same model only judges what is still missing.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Literal, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import get_logger
from app.evaluation.metrics import compute_metrics, render_markdown
from app.evaluation.runner import RESULTS_ROOT

logger = get_logger(__name__)

CORRECTNESS_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


class Judgment(BaseModel):
    abstained: bool = Field(description="True if the answer declines to answer or says the information is not available/covered in the documents.")
    correctness: Literal["correct", "partial", "incorrect", "not_applicable"] = Field(
        description="Compared with the reference answer. 'not_applicable' only for unanswerable questions."
    )
    faithful: bool = Field(description="True only if every factual claim in the answer is supported by the provided context. An answer that only declines to answer is faithful.")
    unsupported_claims: List[str] = Field(default_factory=list, description="Claims in the answer not supported by the context (empty if faithful).")
    reasoning: str = Field(description="One or two sentences justifying the verdicts.")


JUDGE_PROMPT = """You are a strict, impartial evaluator of a document question-answering system.

You are given a QUESTION, the CONTEXT the system retrieved, the system's ANSWER, and a REFERENCE
answer written by a human who read the source documents. ANSWERABLE says whether the documents
contain the information.

Decide:
1. abstained - the ANSWER declines to answer or states the information is not available/covered.
2. correctness - how well the ANSWER matches the REFERENCE on the facts that matter:
   correct = all key facts right (extra correct detail is fine);
   partial = some key facts right, but something important is missing or wrong;
   incorrect = key facts wrong, missing, or the answer abstains although the reference has the answer.
   If ANSWERABLE is false, use not_applicable.
3. faithful - every factual claim in the ANSWER is supported by the CONTEXT. Judge ONLY against the
   CONTEXT: facts that are true in the real world but absent from the CONTEXT are NOT supported.
   Citation markers and polite phrasing are not claims. A pure refusal is faithful.

For an unanswerable question, an answer that invents specifics, or answers from general knowledge
instead of declining, is NOT an abstention (abstained=false) and is unfaithful.

QUESTION:
{question}

ANSWERABLE: {answerable}

REFERENCE ANSWER:
{reference}

CONTEXT:
{context}

ANSWER:
{answer}
"""


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model)


def _judge_llm(model: str):
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0,
    ).with_structured_output(Judgment)


class QuotaExhausted(RuntimeError):
    """The API says to wait too long for retrying to be sensible (e.g. daily quota)."""


MAX_WAIT_S = 120.0  # longer server-requested waits are treated as quota exhaustion


def parse_retry_delay(msg: str) -> Optional[float]:
    """Parse Gemini retry hints: 'retry in 17h38m9.37s', 'retry in 30.8s' or "'retryDelay': '63489s'"."""
    m = re.search(r"retry in ((?:\d+(?:\.\d+)?[hms])+)", msg)
    if m:
        total = 0.0
        for value, unit in re.findall(r"(\d+(?:\.\d+)?)([hms])", m.group(1)):
            total += float(value) * {"h": 3600, "m": 60, "s": 1}[unit]
        return total
    m = re.search(r"retryDelay'?\W+(\d+(?:\.\d+)?)s", msg)
    return float(m.group(1)) if m else None


def _is_transient(msg: str) -> bool:
    return any(t in msg for t in ("503", "UNAVAILABLE", "500", "INTERNAL", "504", "DEADLINE"))


async def judge_record(llm, record: Dict[str, Any], max_retries: int = 5) -> Dict[str, Any]:
    prompt = JUDGE_PROMPT.format(
        question=record["question"],
        answerable="true" if record["expected"] else "false",
        reference=record["expected_answer"],
        context=record.get("context") or "(no context was retrieved)",
        answer=record["answer"],
    )
    for attempt in range(max_retries + 1):
        try:
            result: Judgment = await llm.ainvoke(prompt)
            return result.model_dump()
        except Exception as e:
            msg = str(e)
            rate_limited = "429" in msg or "RESOURCE_EXHAUSTED" in msg
            if not (rate_limited or _is_transient(msg)) or attempt == max_retries:
                raise
            delay = parse_retry_delay(msg) if rate_limited else None
            if delay is not None and delay > MAX_WAIT_S:
                raise QuotaExhausted(f"API asked to wait {delay / 3600:.1f}h: quota exhausted") from e
            await asyncio.sleep(delay + 1 if delay is not None else min(5 * 2 ** attempt, 60))


def _rate(values: List[bool]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def aggregate_generation(records: List[Dict[str, Any]], judge_model: Optional[str] = None) -> Dict[str, Any]:
    """Aggregate judged records (each with a 'judgment' dict). Pure function."""
    judged = [r for r in records if r.get("judgment")]
    answerable = [r for r in judged if r["expected"]]
    unanswerable = [r for r in judged if not r["expected"]]

    def correctness(rs):
        scores = [CORRECTNESS_SCORE[r["judgment"]["correctness"]] for r in rs
                  if r["judgment"]["correctness"] in CORRECTNESS_SCORE]
        return round(mean(scores), 4) if scores else None

    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for r in answerable:
        by_cat[r["category"]].append(r)

    return {
        "n_judged": len(judged),
        "judge_model": judge_model or settings.EVAL_JUDGE_MODEL,
        "answerable": {
            "n": len(answerable),
            "correctness_score": correctness(answerable),  # correct=1, partial=.5, incorrect=0
            "fully_correct_rate": _rate([r["judgment"]["correctness"] == "correct" for r in answerable]),
            "faithfulness_rate": _rate([r["judgment"]["faithful"] for r in answerable]),
            "false_refusal_rate": _rate([r["judgment"]["abstained"] for r in answerable]),
        },
        "unanswerable": {
            "n": len(unanswerable),
            "abstention_rate": _rate([r["judgment"]["abstained"] for r in unanswerable]),
            "hallucination_rate": _rate([not r["judgment"]["abstained"] for r in unanswerable]),
            "faithfulness_rate": _rate([r["judgment"]["faithful"] for r in unanswerable]),
        },
        "by_category": {
            c: {"n": len(rs), "correctness_score": correctness(rs),
                "faithfulness_rate": _rate([r["judgment"]["faithful"] for r in rs])}
            for c, rs in sorted(by_cat.items())
        },
        "latency_s": {
            stage: round(mean(r["latency"][stage] for r in records if stage in r.get("latency", {})), 3)
            for stage in ("retrieve_s", "rerank_s", "generate_s")
            if any(stage in r.get("latency", {}) for r in records)
        },
    }


def render_generation_markdown(g: Dict[str, Any]) -> str:
    a, u = g["answerable"], g["unanswerable"]
    lines = [
        "## Generation (LLM-judged)", "",
        f"Judge model: `{g['judge_model']}` · {g['n_judged']} answers judged"
        + (f" · **{g['n_pending']} still pending (partial results)**" if g.get("n_pending") else ""), "",
        f"### Answerable questions (n={a['n']})", "",
        "| metric | value |", "|---|---|",
        f"| correctness score (correct=1, partial=0.5) | {a['correctness_score']} |",
        f"| fully correct | {a['fully_correct_rate']} |",
        f"| faithfulness (all claims supported by context) | {a['faithfulness_rate']} |",
        f"| false refusals | {a['false_refusal_rate']} |", "",
        f"### Unanswerable questions (n={u['n']})", "",
        "| metric | value |", "|---|---|",
        f"| correctly abstained | {u['abstention_rate']} |",
        f"| hallucinated an answer | {u['hallucination_rate']} |", "",
        "### Answerable by category", "",
        "| category | n | correctness | faithfulness |", "|---|---|---|---|",
    ]
    for c, b in g["by_category"].items():
        lines.append(f"| {c} | {b['n']} | {b['correctness_score']} | {b['faithfulness_rate']} |")
    if g.get("latency_s"):
        lines += ["", "Mean latency (s): " + ", ".join(f"{k} {v}" for k, v in g["latency_s"].items())]
    lines.append("")
    return "\n".join(lines)


def _save_judged(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)  # atomic: never leaves a half-written file


async def judge_run(run_dir: Path, model: Optional[str] = None, concurrency: int = 3) -> Dict[str, Any]:
    """Judge every generated answer; resumable (keeps judgments from earlier runs)."""
    model = model or settings.EVAL_JUDGE_MODEL
    slug = _slug(model)
    results = json.loads((run_dir / "results.json").read_text())
    data = {"meta": results["meta"], "records": results["records"]}
    records = data["records"]

    judged_path = run_dir / f"judged_{slug}.json"
    if judged_path.exists():  # resume: reuse earlier successful judgments
        prior = {r["id"]: r for r in json.loads(judged_path.read_text())["records"]}
        for r in records:
            old = prior.get(r["id"])
            if old and old.get("judgment"):
                r["judgment"] = old["judgment"]

    candidates = [r for r in records if not r["error"] and r.get("answer")]
    if not candidates:
        raise SystemExit("No generated answers in results.json; run the runner without --no-generate.")
    todo = [r for r in candidates if not r.get("judgment")]
    print(f"{len(candidates) - len(todo)} already judged, {len(todo)} to judge "
          f"with {model}")

    llm = _judge_llm(model)
    sem = asyncio.Semaphore(concurrency)
    quota_hit = asyncio.Event()

    async def one(r):
        async with sem:
            if quota_hit.is_set():
                return
            try:
                r["judgment"] = await judge_record(llm, r)
                r.pop("judge_error", None)
                tag = "abstain" if r["judgment"]["abstained"] else r["judgment"]["correctness"]
                print(f"  {r['id']:6s} {r['category']:12s} {tag:10s} faithful={r['judgment']['faithful']}")
            except QuotaExhausted as e:
                quota_hit.set()
                r["judge_error"] = str(e)
                print(f"  {r['id']:6s} STOP: {e}")
            except Exception as e:
                r["judge_error"] = f"{type(e).__name__}: {str(e)[:300]}"
                print(f"  {r['id']:6s} JUDGE ERROR {r['judge_error'][:120]}")
            data["meta"]["judge_model"] = model
            _save_judged(judged_path, data)  # incremental: nothing is lost if interrupted

    await asyncio.gather(*(one(r) for r in todo))

    pending = [r["id"] for r in candidates if not r.get("judgment")]
    gen = aggregate_generation(records, model)
    gen["n_pending"] = len(pending)
    data["meta"]["judge_model"] = model
    _save_judged(judged_path, data)
    (run_dir / f"generation_metrics_{slug}.json").write_text(json.dumps(gen, indent=2))

    scored = [r for r in records if not r["error"]]
    report = render_markdown(compute_metrics(scored), data["meta"]) + "\n" + render_generation_markdown(gen)
    report_path = run_dir / f"report_{slug}.md"
    report_path.write_text(report)
    if pending:
        print(f"\n{len(pending)} answers still unjudged ({', '.join(pending[:8])}{'...' if len(pending) > 8 else ''}). "
              f"Re-run the same command later to resume, or use --model to switch judge.")
    print(f"Wrote {judged_path.name}, generation_metrics_{slug}.json, {report_path.name} to {run_dir}")
    return gen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--model", default=None, help="judge model (default: EVAL_JUDGE_MODEL)")
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    run_dir = args.run_dir or sorted(p for p in RESULTS_ROOT.iterdir() if p.is_dir())[-1]
    model = args.model or settings.EVAL_JUDGE_MODEL
    asyncio.run(judge_run(run_dir, model, args.concurrency))
    print((run_dir / f"report_{_slug(model)}.md").read_text())


if __name__ == "__main__":
    main()

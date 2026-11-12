"""
Deterministic retrieval metrics for RAG evals (no LLM involved).

Input is a list of per-question *records* (see runner.py). Each record has:
    id, category,
    expected:        [[document, page], ...]  1-based pages that contain the answer
                     (empty for unanswerable questions)
    raw_chunks:      [{"document", "page", "score", ...}]  vector-search order
    reranked_chunks: [{"document", "page", "rerank_score", ...}]  rerank order

A chunk counts as relevant only if BOTH its document and page match, since page
numbers repeat across documents. Pages are 1-based (the runner converts PyPDF's
0-based metadata).
"""
from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence

RAW_KS = (1, 3, 5, 10, 20)
RERANKED_KS = (1, 3, 5)
THRESHOLDS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7)


def hit_at_k(pages: Sequence[int], expected: Iterable[int], k: int) -> float:
    return 1.0 if set(pages[:k]) & set(expected) else 0.0


def recall_at_k(pages: Sequence[int], expected: Iterable[int], k: int) -> float:
    expected = set(expected)
    return len(set(pages[:k]) & expected) / len(expected) if expected else 0.0


def reciprocal_rank(pages: Sequence[int], expected: Iterable[int]) -> float:
    expected = set(expected)
    for i, p in enumerate(pages, start=1):
        if p in expected:
            return 1.0 / i
    return 0.0


def _keys(chunks: List[Dict[str, Any]]) -> List[tuple]:
    return [(c["document"], c["page"]) for c in chunks]


def _docs(chunks: List[Dict[str, Any]]) -> List[str]:
    return [c["document"] for c in chunks]


def _expected(record: Dict[str, Any]) -> set:
    return {tuple(x) for x in record["expected"]}


def _expected_docs(record: Dict[str, Any]) -> set:
    return {d for d, _ in _expected(record)}


def _avg(values: List[float]) -> float | None:
    return round(mean(values), 4) if values else None


def _retrieval_block(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Metrics over answerable records (those with expected_pages)."""
    out: Dict[str, Any] = {"n": len(records)}
    for label, key, ks in (
        ("raw", "raw_chunks", RAW_KS),
        ("reranked", "reranked_chunks", RERANKED_KS),
    ):
        block: Dict[str, Any] = {}
        for k in ks:
            block[f"hit@{k}"] = _avg(
                [hit_at_k(_keys(r[key]), _expected(r), k) for r in records]
            )
            block[f"recall@{k}"] = _avg(
                [recall_at_k(_keys(r[key]), _expected(r), k) for r in records]
            )
        block["mrr"] = _avg(
            [reciprocal_rank(_keys(r[key]), _expected(r)) for r in records]
        )
        # Looser: did we at least retrieve the right document(s)?
        block["doc_hit@5"] = _avg(
            [hit_at_k(_docs(r[key]), _expected_docs(r), 5) for r in records]
        )
        out[label] = block

    # Rerank lift: reranked top-k vs. the same top-k straight from vector search
    def lift(metric: str):
        a, b = out["raw"].get(metric), out["reranked"].get(metric)
        return round(b - a, 4) if a is not None and b is not None else None

    out["rerank_lift"] = {m: lift(m) for m in ("hit@1", "hit@3", "hit@5", "mrr")}
    return out


def _top_rerank_score(record: Dict[str, Any]) -> float:
    scores = [c.get("rerank_score", 0.0) for c in record["reranked_chunks"]]
    return max(scores) if scores else 0.0


def _threshold_curve(answerable: List[Dict], unanswerable: List[Dict]) -> List[Dict[str, Any]]:
    """
    For each candidate RERANK_MIN_SCORE, how often would we keep context for
    answerable questions vs. correctly drop all context for unanswerable ones?
    """
    rows = []
    for t in THRESHOLDS:
        kept = [1.0 if _top_rerank_score(r) >= t else 0.0 for r in answerable]
        # also require the kept context to still contain a correct page
        useful = [
            1.0
            if any(c.get("rerank_score", 0.0) >= t and (c["document"], c["page"]) in _expected(r)
                   for c in r["reranked_chunks"])
            else 0.0
            for r in answerable
        ]
        dropped = [1.0 if _top_rerank_score(r) < t else 0.0 for r in unanswerable]
        rows.append({
            "min_score": t,
            "answerable_keep_context": _avg(kept),
            "answerable_keep_correct_page": _avg(useful),
            "unanswerable_no_context": _avg(dropped),
        })
    return rows


def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    answerable = [r for r in records if r["expected"]]
    unanswerable = [r for r in records if not r["expected"]]

    by_category: Dict[str, List[Dict]] = defaultdict(list)
    for r in answerable:
        by_category[r["category"]].append(r)

    return {
        "n_total": len(records),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "retrieval": _retrieval_block(answerable) if answerable else {},
        "retrieval_by_category": {c: _retrieval_block(rs) for c, rs in sorted(by_category.items())},
        "top_rerank_score": {
            "answerable_mean": _avg([_top_rerank_score(r) for r in answerable]),
            "unanswerable_mean": _avg([_top_rerank_score(r) for r in unanswerable]),
            "unanswerable_max": round(max((_top_rerank_score(r) for r in unanswerable), default=0.0), 4),
        },
        "threshold_curve": _threshold_curve(answerable, unanswerable) if records else [],
    }


def render_markdown(metrics: Dict[str, Any], meta: Dict[str, Any] | None = None) -> str:
    lines = ["# RAG eval report", ""]
    if meta:
        for k, v in meta.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
    lines.append(
        f"Questions: {metrics['n_total']} "
        f"({metrics['n_answerable']} answerable, {metrics['n_unanswerable']} unanswerable)"
    )
    lines.append("")

    ret = metrics.get("retrieval") or {}
    if ret:
        lines += ["## Retrieval (answerable questions, page-level)", "",
                  "| stage | " + " | ".join(
                      [f"hit@{k}" for k in RAW_KS] + ["mrr", "recall@5"]) + " |",
                  "|---|" + "---|" * (len(RAW_KS) + 2)]
        raw = ret["raw"]
        lines.append("| vector (raw) | " + " | ".join(
            [str(raw.get(f"hit@{k}")) for k in RAW_KS]
            + [str(raw["mrr"]), str(raw["recall@5"])]) + " |")
        rr = ret["reranked"]
        lines.append("| reranked | " + " | ".join(
            [str(rr.get(f"hit@{k}", "–")) for k in RAW_KS]
            + [str(rr["mrr"]), str(rr["recall@5"])]) + " |")
        lift = ret["rerank_lift"]
        lines += ["", "Rerank lift (reranked − vector): "
                  + ", ".join(f"{m} **{v:+}**" for m, v in lift.items() if v is not None),
                  f"Right-document hit@5: vector {raw['doc_hit@5']}, reranked {rr['doc_hit@5']}", ""]

        lines += ["### By category", "", "| category | n | raw hit@5 | reranked hit@1 | reranked hit@5 | raw recall@5 | reranked recall@5 | reranked mrr |", "|---|---|---|---|---|---|---|---|"]
        for cat, b in metrics["retrieval_by_category"].items():
            lines.append(f"| {cat} | {b['n']} | {b['raw']['hit@5']} | {b['reranked']['hit@1']} | "
                         f"{b['reranked']['hit@5']} | {b['raw']['recall@5']} | {b['reranked']['recall@5']} | "
                         f"{b['reranked']['mrr']} |")
        lines.append("")

    t = metrics["top_rerank_score"]
    lines += ["## Rerank score separation", "",
              f"- mean top score, answerable: {t['answerable_mean']}",
              f"- mean top score, unanswerable: {t['unanswerable_mean']}",
              f"- max top score, unanswerable: {t['unanswerable_max']}", "",
              "### Candidate `RERANK_MIN_SCORE` values", "",
              "| min_score | answerable keeps context | …keeps a correct page | unanswerable gets no context |",
              "|---|---|---|---|"]
    for row in metrics["threshold_curve"]:
        lines.append(f"| {row['min_score']} | {row['answerable_keep_context']} | "
                     f"{row['answerable_keep_correct_page']} | {row['unanswerable_no_context']} |")
    lines.append("")
    return "\n".join(lines)

from app.evaluation.metrics import (
    compute_metrics, hit_at_k, recall_at_k, reciprocal_rank, render_markdown,
)


def test_primitives():
    pages = [7, 3, 9, 3]
    assert hit_at_k(pages, [3], 1) == 0.0
    assert hit_at_k(pages, [3], 2) == 1.0
    assert recall_at_k(pages, [3, 9, 50], 3) == 2 / 3
    assert reciprocal_rank(pages, [9]) == 1 / 3
    assert reciprocal_rank(pages, [99]) == 0.0
    assert recall_at_k(pages, [], 3) == 0.0


D = "a.pdf"


def _rec(id, cat, expected, raw_pages, rr, doc=D):
    """expected: list of pages in document `doc` (empty = unanswerable)."""
    return {
        "id": id, "category": cat,
        "expected": [[doc, p] for p in expected],
        "raw_chunks": [{"document": doc, "page": p, "score": 1.0} for p in raw_pages],
        "reranked_chunks": [{"document": doc, "page": p, "rerank_score": s} for p, s in rr],
    }


def test_compute_metrics_and_rerank_lift():
    records = [
        # relevant page 4 is rank 3 in vector search, rank 1 after rerank
        _rec("a", "direct", [4], [1, 2, 4, 5, 6], [(4, 0.9), (1, 0.1)]),
        # never retrieved
        _rec("b", "direct", [8], [1, 2, 3, 5, 6], [(1, 0.2)]),
        # unanswerable: low rerank scores
        _rec("c", "unanswerable", [], [1, 2], [(1, 0.02)]),
    ]
    m = compute_metrics(records)
    assert (m["n_total"], m["n_answerable"], m["n_unanswerable"]) == (3, 2, 1)
    assert m["retrieval"]["raw"]["hit@5"] == 0.5
    assert m["retrieval"]["raw"]["hit@1"] == 0.0
    assert m["retrieval"]["reranked"]["hit@1"] == 0.5
    assert m["retrieval"]["reranked"]["mrr"] == 0.5
    assert m["retrieval"]["rerank_lift"]["hit@5"] == 0.0  # raw top-5 already had it
    assert m["retrieval"]["rerank_lift"]["hit@1"] == 0.5
    assert m["retrieval"]["raw"]["doc_hit@5"] == 1.0
    assert m["top_rerank_score"]["unanswerable_max"] == 0.02

    row = {r["min_score"]: r for r in m["threshold_curve"]}[0.1]
    assert row["unanswerable_no_context"] == 1.0
    assert row["answerable_keep_correct_page"] == 0.5


def test_render_markdown_smoke():
    m = compute_metrics([_rec("a", "direct", [4], [4], [(4, 0.9)])])
    md = render_markdown(m, {"chat_model": "x"})
    assert "# RAG eval report" in md and "hit@5" in md


def test_embed_retry_on_429(monkeypatch):
    from app.services import vector_store as vs

    monkeypatch.setattr(vs.time, "sleep", lambda s: None)
    calls = {"n": 0}

    class Flaky:
        def embed_documents(self, batch):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("429 RESOURCE_EXHAUSTED ... Please retry in 5.0s.")
            return [[0.0]] * len(batch)

    assert vs._embed_batch_with_retry(Flaky(), ["a", "b"]) == [[0.0], [0.0]]
    assert calls["n"] == 3
    assert vs._retry_delay_s("Please retry in 30.8s") == 31.8


def test_same_page_in_other_document_does_not_count():
    # Page 4 exists in both documents; only b.pdf p4 is correct.
    rec = {
        "id": "x", "category": "direct", "expected": [["b.pdf", 4]],
        "raw_chunks": [{"document": "a.pdf", "page": 4, "score": 1.0},
                       {"document": "b.pdf", "page": 4, "score": 0.9}],
        "reranked_chunks": [{"document": "a.pdf", "page": 4, "rerank_score": 0.9}],
    }
    m = compute_metrics([rec])
    assert m["retrieval"]["raw"]["hit@1"] == 0.0
    assert m["retrieval"]["raw"]["mrr"] == 0.5
    assert m["retrieval"]["reranked"]["hit@5"] == 0.0
    assert m["retrieval"]["reranked"]["doc_hit@5"] == 0.0


def test_verify_dataset_catches_wrong_page(tmp_path, monkeypatch):
    import json
    from app.evaluation import verify_dataset as vd

    (tmp_path / "a.pdf").write_bytes(b"")
    monkeypatch.setattr(vd, "_pages", lambda path: (vd._norm("page one text"), vd._norm("the answer is 42")))
    data = [{"id": "q1", "category": "direct", "question": "?", "expected_answer": "42",
             "evidence": [{"document": "a.pdf", "page": 1, "contains": "the answer is 42"}]},
            {"id": "q2", "category": "direct", "question": "?", "expected_answer": "42",
             "evidence": [{"document": "a.pdf", "page": 2, "contains": "The  Answer is 42"}]},
            {"id": "q3", "category": "unanswerable", "question": "?", "expected_answer": "",
             "evidence": []}]
    path = tmp_path / "d.json"
    path.write_text(json.dumps(data))
    errors = vd.verify(path, tmp_path)
    assert len(errors) == 1 and "q1" in errors[0] and "found on pages [2]" in errors[0]


def test_aggregate_generation():
    from app.evaluation.judge import aggregate_generation

    def rec(cat, expected, **j):
        base = {"abstained": False, "correctness": "correct", "faithful": True}
        return {"id": "x", "category": cat, "expected": expected, "latency": {},
                "judgment": {**base, **j}}

    recs = [
        rec("direct", [["a.pdf", 1]]),
        rec("direct", [["a.pdf", 1]], correctness="partial"),
        rec("direct", [["a.pdf", 1]], correctness="incorrect", abstained=True),
        rec("synthesis", [["a.pdf", 1]], faithful=False),
        rec("unanswerable", [], abstained=True, correctness="not_applicable"),
        rec("unanswerable", [], abstained=False, correctness="not_applicable", faithful=False),
    ]
    g = aggregate_generation(recs)
    assert g["answerable"]["n"] == 4 and g["unanswerable"]["n"] == 2
    assert g["answerable"]["correctness_score"] == round((1 + 0.5 + 0 + 1) / 4, 4)
    assert g["answerable"]["fully_correct_rate"] == 0.5
    assert g["answerable"]["faithfulness_rate"] == 0.75
    assert g["answerable"]["false_refusal_rate"] == 0.25
    assert g["unanswerable"]["abstention_rate"] == 0.5
    assert g["unanswerable"]["hallucination_rate"] == 0.5
    assert g["by_category"]["synthesis"]["faithfulness_rate"] == 0.0


def test_parse_retry_delay():
    from app.evaluation.judge import parse_retry_delay

    assert parse_retry_delay("Please retry in 17h38m9.37s.") == 17 * 3600 + 38 * 60 + 9.37
    assert parse_retry_delay("Please retry in 30.8s.") == 30.8
    assert parse_retry_delay("... 'retryDelay': '63489s'}") == 63489
    assert parse_retry_delay("boom") is None

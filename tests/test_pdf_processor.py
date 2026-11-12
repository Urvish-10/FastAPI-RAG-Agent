from app.services.pdf_processor import sanitize_text, build_context_string


def test_sanitize_strips_control_chars_and_collapses_whitespace():
    assert sanitize_text("a\x00b\x07  c\n\n d ") == "ab c d"


def test_sanitize_empty():
    assert sanitize_text("") == ""
    assert sanitize_text(None) == ""


def _chunk(i, text="x" * 100, score=0.9):
    return {
        "text": text,
        "rerank_score": score,
        "metadata": {"filename": "f.pdf", "page": i, "chunk_index": i, "document_id": "d"},
    }


def test_build_context_respects_budget():
    chunks = [_chunk(i) for i in range(10)]
    context, sources = build_context_string(chunks, max_tokens=80)  # ~320 chars
    assert 0 < len(sources) < 10
    assert len(sources) == context.count("[Source:")


def test_build_context_source_fields():
    _, sources = build_context_string([_chunk(3, score=0.12345)])
    assert sources[0]["page"] == 3
    assert sources[0]["relevance_score"] == 0.1235

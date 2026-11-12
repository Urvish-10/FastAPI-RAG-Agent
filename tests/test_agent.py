"""Run the real graph with an in-memory checkpointer and fake LLM / retrieval."""
import json

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent import graph as graph_mod
from app.agent import runner

CHUNKS = [
    {"id": "d_0", "score": 0.9, "text": "Python lists are mutable.",
     "metadata": {"document_id": "d", "filename": "f.pdf", "page": 1, "chunk_index": 0}},
]


@pytest.fixture
def env(monkeypatch):
    seen = {"retrieve_queries": [], "doc_filters": [], "prompts": []}

    def fake_query_chunks(query, namespace, top_k, document_ids=None):
        seen["retrieve_queries"].append(query)
        seen["doc_filters"].append(document_ids)
        return CHUNKS

    def fake_rerank(query, chunks, top_n):
        return [{**c, "rerank_score": 0.8} for c in chunks]

    monkeypatch.setattr(graph_mod, "query_chunks", fake_query_chunks)
    monkeypatch.setattr(graph_mod, "rerank_chunks", fake_rerank)

    replies = iter([])

    class Recording(GenericFakeChatModel):
        async def _agenerate(self, messages, *a, **kw):
            seen["prompts"].append(messages)
            return await super()._agenerate(messages, *a, **kw)

    def set_replies(*texts):
        nonlocal replies
        replies = iter(AIMessage(content=t) for t in texts)

    monkeypatch.setattr(
        graph_mod, "_get_llm", lambda streaming=False: Recording(messages=replies)
    )
    monkeypatch.setattr(graph_mod, "_compiled_graph", graph_mod.build_graph(MemorySaver()))
    return seen, set_replies


async def test_non_streaming_answer_and_sources(env):
    seen, set_replies = env
    set_replies("Lists are mutable.")
    answer, sources = await runner.run_rag_query("u1", "s1", "Are lists mutable?")
    assert answer == "Lists are mutable."
    assert sources[0]["page"] == 1


async def test_history_not_duplicated_and_followup_rewritten(env):
    seen, set_replies = env
    set_replies("Lists are mutable.", "what about tuples", "Tuples are immutable.")

    await runner.run_rag_query("u1", "s2", "Are lists mutable?")
    await runner.run_rag_query("u1", "s2", "And tuples?")

    # Turn 2 retrieval used the rewritten standalone query
    assert seen["retrieve_queries"] == ["Are lists mutable?", "what about tuples"]

    # Final generate prompt: system + exactly 3 history msgs (H, A, H), no repeats
    final_prompt = seen["prompts"][-1]
    humans = [m.content for m in final_prompt if isinstance(m, HumanMessage)]
    assert humans == ["Are lists mutable?", "And tuples?"]


async def test_streaming_emits_tokens_sources_done(env):
    seen, set_replies = env
    set_replies("Lists are mutable.")
    events = []
    async for sse in runner.stream_rag_query("u1", "s3", "Are lists mutable?", ["d"]):
        events.append(json.loads(sse[6:]))

    types = [e["type"] for e in events]
    assert types[-2:] == ["sources", "done"]
    assert "".join(e["content"] for e in events if e["type"] == "token").strip() == "Lists are mutable."
    assert events[-2]["sources"][0]["document_id"] == "d"
    assert seen["doc_filters"] == [["d"]]


async def test_min_score_filters_context(env, monkeypatch):
    seen, set_replies = env
    set_replies("I don't know.")
    monkeypatch.setattr(graph_mod.settings, "RERANK_MIN_SCORE", 0.95)
    _, sources = await runner.run_rag_query("u1", "s4", "q")
    assert sources == []

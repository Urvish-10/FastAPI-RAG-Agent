"""
Agent runner - called by API routes.

Both the streaming and non-streaming paths run the same compiled LangGraph.
Conversation history lives in the Postgres checkpointer (thread_id = session_id),
so only the NEW user message is passed in on each call.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Tuple

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from app.agent.graph import get_graph, RAGState
from app.core.logging import get_logger
from app.services.pdf_processor import sanitize_text

logger = get_logger(__name__)


def _text_of(content: Any) -> str:
    """Normalise LangChain message content (str or list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def _initial_state(user_id: str, query: str) -> RAGState:
    safe_query = sanitize_text(query)
    return {
        # The checkpointer supplies prior turns; add_messages appends this one.
        "messages": [HumanMessage(content=safe_query)],
        "user_id": user_id,
        # Per-request fields are reset every turn
        "query": safe_query,
        "raw_chunks": [],
        "reranked_chunks": [],
        "sources": [],
        "context": "",
    }


def _config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


async def run_rag_query(
    user_id: str,
    session_id: str,
    query: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Non-streaming query. Returns (answer_text, sources)."""
    graph = get_graph()
    final_state = await graph.ainvoke(
        _initial_state(user_id, query), config=_config(session_id)
    )

    ai_messages = [m for m in final_state["messages"] if isinstance(m, AIMessage)]
    answer = _text_of(ai_messages[-1].content) if ai_messages else "No answer generated."
    sources = final_state.get("sources", [])

    logger.info(
        f"Session={session_id} | Sources={len(sources)} | "
        f"Answer length={len(answer)} chars"
    )
    return answer, sources


async def stream_rag_query(
    user_id: str,
    session_id: str,
    query: str,
) -> AsyncIterator[str]:
    """
    Streaming query through the compiled graph.

    Yields Server-Sent Events strings:
      - data: {"type": "token", "content": "..."}
      - data: {"type": "sources", "sources": [...]}
      - data: {"type": "done"}
    """
    graph = get_graph()
    sources: List[Dict[str, Any]] = []
    token_count = 0

    async for mode, payload in graph.astream(
        _initial_state(user_id, query),
        config=_config(session_id),
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            chunk, metadata = payload
            if (
                metadata.get("langgraph_node") == "generate"
                and isinstance(chunk, AIMessageChunk)
            ):
                text = _text_of(chunk.content)
                if text:
                    token_count += 1
                    yield f"data: {json.dumps({'type': 'token', 'content': text})}\n\n"
        elif mode == "updates":
            rerank_update = payload.get("rerank")
            if rerank_update:
                sources = rerank_update.get("sources", [])

    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

    logger.info(
        f"Stream complete: session={session_id} | "
        f"chunks={token_count} | sources={len(sources)}"
    )

"""
Agent runner - called by API routes.

Handles:
- Building RAGState from session history + new message
- Invoking the graph (non-streaming) or running retrieval + streaming generation
- Returning final answer + sources
"""
from __future__ import annotations

import json
from typing import List, Dict, Any, AsyncIterator, Tuple, Optional

from langchain_core.messages import HumanMessage, AIMessage

from app.agent.graph import (
    get_graph,
    generate_node_streaming,
    retrieve_node,
    rerank_node,
    RAGState,
)
from app.core.logging import get_logger
from app.services.pdf_processor import sanitize_text

logger = get_logger(__name__)


def _build_lc_messages(db_messages: List[Any]) -> list:
    """Convert DB ChatMessage ORM objects → LangChain message objects."""
    lc = []
    for m in db_messages:
        if m.role == "user":
            lc.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            lc.append(AIMessage(content=m.content))
    return lc


async def run_rag_query(
    user_id: str,
    session_id: str,
    query: str,
    history_messages: List[Any],  # ORM ChatMessage objects
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Non-streaming query. Returns (answer_text, sources).
    """

    safe_query = sanitize_text(query)


    graph = get_graph()
    lc_messages = _build_lc_messages(history_messages)
    lc_messages.append(HumanMessage(content=safe_query))

    config = {"configurable": {"thread_id": session_id}}

    initial_state: RAGState = {
        "messages": lc_messages,
        "user_id": user_id,
        "query": query,
        "raw_chunks": [],
        "reranked_chunks": [],
        "sources": [],
        "context": "",
    }

    final_state = await graph.ainvoke(initial_state, config=config)

    # Extract assistant answer
    assistant_messages = [
        m for m in final_state["messages"] if isinstance(m, AIMessage)
    ]
    answer = assistant_messages[-1].content if assistant_messages else "No answer generated."
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
    history_messages: List[Any],
) -> AsyncIterator[str]:
    """
    Streaming query.

    Yields Server-Sent Events strings:
      - data: {"type": "token", "content": "..."}
      - data: {"type": "sources", "sources": [...]}
      - data: {"type": "done"}
    """

    safe_query = sanitize_text(query)

    lc_messages = _build_lc_messages(history_messages)
    lc_messages.append(HumanMessage(content=safe_query))

    # Run retrieval + reranking synchronously first (these are fast)
    partial_state: RAGState = {
        "messages": lc_messages,
        "user_id": user_id,
        "query": query,
        "raw_chunks": [],
        "reranked_chunks": [],
        "sources": [],
        "context": "",
    }

    # Retrieve
    retrieve_result = retrieve_node(partial_state)
    partial_state.update(retrieve_result)

    # Rerank
    rerank_result = rerank_node(partial_state)
    partial_state.update(rerank_result)

    sources = partial_state.get("sources", [])

    # Stream generation
    full_response = []
    async for token in generate_node_streaming(partial_state):
        full_response.append(token)
        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

    # Emit sources after generation completes
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

    logger.info(
        f"Stream complete: session={session_id} | "
        f"tokens={len(full_response)} | sources={len(sources)}"
    )

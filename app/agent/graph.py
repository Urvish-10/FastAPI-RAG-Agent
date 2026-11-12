"""
LangGraph RAG Agent

Graph flow:
  START → retrieve → rerank → generate → END

The graph is compiled once at startup.
Each conversation uses a unique thread_id (session_id) so
PostgresSaver checkpoints isolate history per session.
"""
from __future__ import annotations

import asyncio
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.core.config import settings
from app.core.logging import get_logger
from app.services.vector_store import query_chunks
from app.services.reranker import rerank_chunks
from app.services.pdf_processor import build_context_string

logger = get_logger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    # Conversation messages (add_messages merges instead of replacing)
    messages: Annotated[List[BaseMessage], add_messages]
    # Set per request, not persisted in checkpoint
    user_id: str
    query: str
    # Filled by retrieve node
    raw_chunks: List[Dict[str, Any]]
    # Filled by rerank node
    reranked_chunks: List[Dict[str, Any]]
    # Filled by rerank node, passed to caller for citation display
    sources: List[Dict[str, Any]]
    # Context string built from chunks, injected into LLM prompt
    context: str


# ── LLM ───────────────────────────────────────────────────────────────────────

def _get_llm(streaming: bool = False) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_CHAT_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.3,
        max_tokens=2048,
        streaming=streaming,
    )


SYSTEM_PROMPT = """You are a precise and helpful document assistant.
You answer questions based ONLY on the provided document context.

Rules:
- Ground every answer in the provided context. Do not invent facts.
- If the context doesn't contain enough information, say so clearly.
- Cite the source document and page number when referencing specific information.
- For multi-part questions, structure your answer clearly.
- Keep answers concise but complete.

If no context is provided, politely inform the user that no documents have been indexed yet."""


# ── Nodes ─────────────────────────────────────────────────────────────────────

# Max prior messages sent to the LLM. The checkpointer stores the full thread;
# we only window what goes into the prompt.
MAX_HISTORY_MESSAGES = 20


def _build_system_message(context: str) -> SystemMessage:
    if context:
        return SystemMessage(
            content=(
                f"{SYSTEM_PROMPT}\n\n"
                f"=== DOCUMENT CONTEXT ===\n{context}\n=== END CONTEXT ==="
            )
        )
    return SystemMessage(content=SYSTEM_PROMPT)


def _window_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Keep the most recent messages, starting on a human turn."""
    window = messages[-MAX_HISTORY_MESSAGES:]
    for i, m in enumerate(window):
        if isinstance(m, HumanMessage):
            return window[i:]
    return window


async def retrieve_node(state: RAGState) -> dict:
    """Query Pinecone for top-K chunks (sync SDK calls run in a worker thread)."""
    query = state["query"]
    user_id = state["user_id"]

    logger.info(f"Retrieving chunks for user={user_id}")

    raw_chunks = await asyncio.to_thread(
        query_chunks,
        query=query,
        namespace=user_id,
        top_k=settings.RETRIEVAL_TOP_K,
    )

    return {"raw_chunks": raw_chunks}


async def rerank_node(state: RAGState) -> dict:
    """Rerank retrieved chunks with Pinecone Inference API, then build context string."""
    query = state["query"]
    raw_chunks = state.get("raw_chunks", [])

    if not raw_chunks:
        logger.warning("No chunks to rerank — skipping")
        return {"reranked_chunks": [], "context": "", "sources": []}

    reranked = await asyncio.to_thread(
        rerank_chunks,
        query=query,
        chunks=raw_chunks,
        top_n=settings.RERANK_TOP_N,
    )

    context, sources = build_context_string(reranked)

    return {
        "reranked_chunks": reranked,
        "context": context,
        "sources": sources,
    }


async def generate_node(state: RAGState) -> dict:
    """
    Generate the answer with Gemini. Uses ainvoke; when the graph is run with
    stream_mode="messages", LangGraph surfaces the LLM tokens as they arrive.
    """
    messages = _window_messages(state.get("messages", []))
    llm = _get_llm(streaming=True)

    logger.info(f"Generating answer. Prompt history: {len(messages)} messages")

    response = await llm.ainvoke([_build_system_message(state.get("context", ""))] + messages)
    return {"messages": [response]}


# ── Graph compilation ─────────────────────────────────────────────────────────

def build_graph(checkpointer: AsyncPostgresSaver) -> StateGraph:
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("RAG graph compiled successfully")
    return compiled


# ── Singleton compiled graph (set during app startup) ─────────────────────────

_compiled_graph = None
_checkpointer: Optional[AsyncPostgresSaver] = None
_checkpointer_ctx = None   # holds the async context manager for clean shutdown


def get_graph():
    if _compiled_graph is None:
        raise RuntimeError("Graph not initialized. Call init_graph() at startup.")
    return _compiled_graph


def get_checkpointer() -> AsyncPostgresSaver:
    if _checkpointer is None:
        raise RuntimeError("Checkpointer not initialized.")
    return _checkpointer


async def init_graph(database_url: str):
    """
    Called once at application startup via FastAPI lifespan.

    AsyncPostgresSaver.from_conn_string() is an async context manager.
    We enter it here and store both the live checkpointer and the ctx
    so teardown() can cleanly close the connection.
    """
    global _compiled_graph, _checkpointer, _checkpointer_ctx

    _checkpointer_ctx = AsyncPostgresSaver.from_conn_string(database_url)
    _checkpointer = await _checkpointer_ctx.__aenter__()

    await _checkpointer.setup()  # Creates LangGraph checkpoint tables if needed

    _compiled_graph = build_graph(_checkpointer)
    logger.info("Graph initialized with PostgresSaver checkpointer")


async def teardown_graph():
    """Called on application shutdown to cleanly close the Postgres connection."""
    global _checkpointer_ctx
    if _checkpointer_ctx is not None:
        try:
            await _checkpointer_ctx.__aexit__(None, None, None)
            logger.info("PostgresSaver connection closed")
        except Exception as e:
            logger.warning(f"Error closing checkpointer: {e}")
from typing import List
import json

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db
from app.models.models import ChatSession, ChatMessage, Document, DocumentStatus
from app.schemas.schemas import (
    SessionCreate, SessionOut, ChatRequest,
    ChatMessageOut, ChatHistoryResponse,
)
from app.core.security import get_current_user
from app.core.logging import get_logger
from app.agent.runner import run_rag_query, stream_rag_query

logger = get_logger(__name__)
router = APIRouter(prefix="/sessions", tags=["sessions"])


# ── Session management ────────────────────────────────────────────────────────

@router.post("", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def create_session(
    payload: SessionCreate,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session = ChatSession(
        user_id=current_user["sub"],
        title=payload.title or "New conversation",
    )
    db.add(session)
    await db.flush()
    return session


@router.get("", response_model=List[SessionOut])
async def list_sessions(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 50,
):
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == current_user["sub"])
        .order_by(ChatSession.updated_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()


@router.get("/{session_id}", response_model=SessionOut)
async def get_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _get_user_session(session_id, current_user["sub"], db)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session = await _get_user_session(session_id, current_user["sub"], db)
    await db.delete(session)


# ── Chat history ──────────────────────────────────────────────────────────────

@router.get("/{session_id}/history", response_model=ChatHistoryResponse)
async def get_history(
    session_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session = await _get_user_session(session_id, current_user["sub"], db)

    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
    )
    messages = result.scalars().all()

    return ChatHistoryResponse(
        session_id=session_id,
        messages=list(messages),
    )


# ── Chat endpoint ─────────────────────────────────────────────────────────────

@router.post("/{session_id}/chat")
async def chat(
    session_id: str,
    payload: ChatRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Send a message to the RAG agent.

    - If payload.stream=True (default), returns a Server-Sent Events stream.
    - If payload.stream=False, returns a JSON response with the full answer.

    The agent:
    1. Retrieves top-20 chunks from Pinecone (user namespace)
    2. Reranks to top-5 with Pinecone Inference API (bge-reranker-v2-m3)
    3. Generates answer with Gemini using conversation history + context
    """
    user_id = current_user["sub"]
    session = await _get_user_session(session_id, user_id, db)

    # Verify user has at least one ready document
    doc_result = await db.execute(
        select(Document).where(
            Document.user_id == user_id,
            Document.status == DocumentStatus.READY,
        ).limit(1)
    )
    has_docs = doc_result.scalar_one_or_none() is not None

    # Load existing history (last 20 messages to cap context)
    history_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .limit(20)
    )
    history = history_result.scalars().all()

    # Save user message immediately
    user_msg = ChatMessage(
        session_id=session_id,
        role="user",
        content=payload.message,
    )
    db.add(user_msg)
    await db.flush()

    if payload.stream:
        # --- Streaming path ---
        async def event_generator():
            full_answer = []
            final_sources = []

            try:
                async for sse_chunk in stream_rag_query(
                    user_id=user_id,
                    session_id=session_id,
                    query=payload.message,
                    history_messages=list(history),
                ):
                    # Parse internally to capture answer + sources for DB save
                    if sse_chunk.startswith("data: "):
                        data = json.loads(sse_chunk[6:])
                        if data["type"] == "token":
                            full_answer.append(data["content"])
                        elif data["type"] == "sources":
                            final_sources = data["sources"]

                    yield sse_chunk

                # Save assistant message to DB after stream completes
                assistant_msg = ChatMessage(
                    session_id=session_id,
                    role="assistant",
                    content="".join(full_answer),
                    sources=final_sources,
                )
                async with db.begin_nested():
                    db.add(assistant_msg)

            except Exception as e:
                logger.error(f"Streaming error for session {session_id}: {e}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",   # Disable nginx buffering
            },
        )

    else:
        # --- Non-streaming path ---
        answer, sources = await run_rag_query(
            user_id=user_id,
            session_id=session_id,
            query=payload.message,
            history_messages=list(history),
        )

        assistant_msg = ChatMessage(
            session_id=session_id,
            role="assistant",
            content=answer,
            sources=sources,
        )
        db.add(assistant_msg)
        await db.flush()

        return {
            "session_id": session_id,
            "answer": answer,
            "sources": sources,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_user_session(
    session_id: str, user_id: str, db: AsyncSession
) -> ChatSession:
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

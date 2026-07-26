from typing import List, Dict, Any, Optional
from pinecone import Pinecone

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_pinecone_client: Optional[Pinecone] = None


def get_pinecone_client() -> Pinecone:
    """
    Reuse the same Pinecone client singleton.
    No new API key needed — reranking uses PINECONE_API_KEY.
    """
    global _pinecone_client
    if _pinecone_client is None:
        _pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)
    return _pinecone_client


def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_n: int = None,
) -> List[Dict[str, Any]]:
    """
    Rerank retrieved chunks using Pinecone's native Inference Rerank API
    (bge-reranker-v2-m3 — multilingual cross-encoder, no extra API key).

    Constraints from Pinecone docs:
      - Max query tokens:    256
      - Max doc tokens:      1024
      - Max documents:       100

    Args:
        query:  The user's question.
        chunks: List of chunk dicts from Pinecone with at least {"id", "text", ...}
        top_n:  How many results to return. Defaults to settings.RERANK_TOP_N.

    Returns:
        Reranked and trimmed list of chunks with added "rerank_score" key.
    """
    if not chunks:
        return []

    top_n = top_n or settings.RERANK_TOP_N
    pc = get_pinecone_client()

    # Pinecone rerank expects list of dicts with "id" and "text"
    documents = [
        {"id": c.get("id", str(i)), "text": c["text"]}
        for i, c in enumerate(chunks)
    ]

    logger.info(
        f"Reranking {len(documents)} chunks → top {top_n} "
        f"via Pinecone [{settings.PINECONE_RERANK_MODEL}]"
    )

    result = pc.inference.rerank(
        model=settings.PINECONE_RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=top_n,
        return_documents=False,  # we already have the full text in `chunks`
    )

    reranked = []
    for r in result.data:
        chunk = chunks[r.index].copy()
        chunk["rerank_score"] = r.score
        reranked.append(chunk)

    logger.info(
        f"Reranked scores: {[round(c['rerank_score'], 4) for c in reranked]}"
    )
    return reranked
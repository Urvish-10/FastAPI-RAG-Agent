import re
import threading
import time
from collections import deque
from typing import Deque, List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings
from app.core.logging import get_logger
from app.services.pdf_processor import sanitize_text

logger = get_logger(__name__)

_pinecone_client: Optional[Pinecone] = None
_embeddings: Optional[GoogleGenerativeAIEmbeddings] = None

# Gemini free tier allows 100 embedContent requests/minute, and each text in a
# batch counts as one request. We keep a sliding window of recent request
# timestamps (shared across all documents in this process) and wait whenever
# the next batch would exceed settings.EMBED_REQUESTS_PER_MIN.
EMBED_BATCH_SIZE = 80
_EMBED_WINDOW_S = 60.0
_embed_times: Deque[float] = deque()
_embed_lock = threading.Lock()


def _wait_for_embed_budget(n: int) -> None:
    """Block until `n` more embedding requests fit in the per-minute budget."""
    limit = settings.EMBED_REQUESTS_PER_MIN
    if limit <= 0:
        return
    n = min(n, limit)
    with _embed_lock:
        while True:
            now = time.monotonic()
            while _embed_times and now - _embed_times[0] >= _EMBED_WINDOW_S:
                _embed_times.popleft()
            if len(_embed_times) + n <= limit:
                _embed_times.extend([now] * n)
                return
            wait = _EMBED_WINDOW_S - (now - _embed_times[0]) + 0.5
            logger.info(f"Embedding rate limit: waiting {wait:.0f}s")
            time.sleep(max(wait, 0.5))


def get_pinecone_client() -> Pinecone:
    global _pinecone_client
    if _pinecone_client is None:
        _pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)
        _ensure_index_exists(_pinecone_client)
    return _pinecone_client


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.GEMINI_EMBEDDING_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            output_dimensionality=settings.GEMINI_EMBEDDING_DIMENSIONS,
        )
    return _embeddings


def _ensure_index_exists(pc: Pinecone) -> None:
    existing = [idx.name for idx in pc.list_indexes()]
    if settings.PINECONE_INDEX_NAME not in existing:
        logger.info(f"Creating Pinecone index: {settings.PINECONE_INDEX_NAME}")
        pc.create_index(
            name=settings.PINECONE_INDEX_NAME,
            dimension=settings.GEMINI_EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=settings.PINECONE_ENVIRONMENT),
        )
        logger.info("Pinecone index created.")
    else:
        logger.info(f"Pinecone index '{settings.PINECONE_INDEX_NAME}' already exists.")


def _retry_delay_s(message: str) -> float:
    """Parse 'Please retry in 30.8s' from a Gemini 429; default to a full window."""
    m = re.search(r"retry in ([0-9.]+)s", message)
    return float(m.group(1)) + 1.0 if m else _EMBED_WINDOW_S + 1.0


def _embed_batch_with_retry(embedder, batch: List[str], max_retries: int = 5):
    """embed_documents with retry on 429s (quota used by other processes, etc.)."""
    for attempt in range(max_retries + 1):
        try:
            return embedder.embed_documents(batch)
        except Exception as e:
            msg = str(e)
            if attempt == max_retries or not ("RESOURCE_EXHAUSTED" in msg or "429" in msg):
                raise
            delay = _retry_delay_s(msg)
            logger.warning(f"Gemini embedding quota hit; retrying in {delay:.0f}s "
                           f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)


def _embed_with_rate_limit(texts: List[str]) -> List[List[float]]:
    """
    Embed texts in batches of EMBED_BATCH_SIZE, waiting as needed to stay under
    settings.EMBED_REQUESTS_PER_MIN. On a paid tier raise that setting (or set 0
    to disable throttling).
    """
    embedder = get_embeddings()
    all_embeddings: List[List[float]] = []
    total_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i: i + EMBED_BATCH_SIZE]
        _wait_for_embed_budget(len(batch))
        logger.info(
            f"Embedding batch {i // EMBED_BATCH_SIZE + 1}/{total_batches} "
            f"({len(batch)} chunks, total so far: {i + len(batch)}/{len(texts)})"
        )
        all_embeddings.extend(_embed_batch_with_retry(embedder, batch))

    return all_embeddings


def upsert_chunks(
    chunks: List[Dict[str, Any]],
    namespace: str,
    pinecone_batch_size: int = 100,
) -> int:
    """
    Embed and upsert a list of chunks into Pinecone.

    Embedding is throttled by a sliding window (settings.EMBED_REQUESTS_PER_MIN).
    Pinecone upsert is batched separately (max 100 vectors per upsert call).

    Each chunk dict: {
        "id": str,           # unique vector id
        "text": str,         # raw text to embed
        "metadata": dict     # stored as Pinecone metadata
    }

    Returns total vectors upserted.
    """
    pc = get_pinecone_client()
    index = pc.Index(settings.PINECONE_INDEX_NAME)

    texts = [c["text"] for c in chunks]
    logger.info(
        f"Starting embedding for {len(texts)} chunks "
        f"(namespace={namespace}, batches of {EMBED_BATCH_SIZE})"
    )

    # Embed with rate limiting
    embeddings = _embed_with_rate_limit(texts)

    # Build Pinecone vectors
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id": chunk["id"],
            "values": embedding,
            "metadata": {**chunk["metadata"], "text": chunk["text"]},
        })

    # Upsert to Pinecone in batches
    total_upserted = 0
    total_pinecone_batches = (len(vectors) + pinecone_batch_size - 1) // pinecone_batch_size

    for i in range(0, len(vectors), pinecone_batch_size):
        batch = vectors[i: i + pinecone_batch_size]
        batch_num = i // pinecone_batch_size + 1
        index.upsert(vectors=batch, namespace=namespace)
        total_upserted += len(batch)
        logger.info(
            f"Pinecone upsert batch {batch_num}/{total_pinecone_batches} "
            f"({len(batch)} vectors)"
        )

    logger.info(f"Upsert complete: {total_upserted} vectors in namespace={namespace}")
    return total_upserted


def query_chunks(
    query: str,
    namespace: str,
    top_k: int = 20,
    document_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Embed query and retrieve top_k similar chunks from the user's namespace,
    optionally restricted to specific documents.
    Returns list of {id, score, text, metadata} dicts.
    """
    pc = get_pinecone_client()
    index = pc.Index(settings.PINECONE_INDEX_NAME)
    embedder = get_embeddings()

    query = sanitize_text(query)
    query_embedding = embedder.embed_query(query)

    result = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        filter={"document_id": {"$in": document_ids}} if document_ids else None,
    )

    matches = []
    for match in result.matches:
        matches.append({
            "id": match.id,
            "score": match.score,
            "text": match.metadata.get("text", ""),
            "metadata": {k: v for k, v in match.metadata.items() if k != "text"},
        })

    logger.info(f"Retrieved {len(matches)} chunks from Pinecone for namespace={namespace}")
    return matches


def delete_document_chunks(document_id: str, namespace: str) -> None:
    """Delete all vectors for a given document from the namespace."""
    pc = get_pinecone_client()
    index = pc.Index(settings.PINECONE_INDEX_NAME)
    index.delete(
        filter={"document_id": {"$eq": document_id}},
        namespace=namespace,
    )
    logger.info(f"Deleted vectors for document_id={document_id} from namespace={namespace}")


def delete_namespace(namespace: str) -> None:
    """Delete all vectors for a user namespace (e.g. on account deletion)."""
    pc = get_pinecone_client()
    index = pc.Index(settings.PINECONE_INDEX_NAME)
    index.delete(delete_all=True, namespace=namespace)
    logger.info(f"Deleted entire namespace={namespace}")
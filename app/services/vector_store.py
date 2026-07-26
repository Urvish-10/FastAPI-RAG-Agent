import time
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings
from app.core.logging import get_logger
from app.services.pdf_processor import sanitize_text

logger = get_logger(__name__)

_pinecone_client: Optional[Pinecone] = None
_embeddings: Optional[GoogleGenerativeAIEmbeddings] = None

# Gemini free tier: 100 embed requests/minute
# We embed in batches of this size and pause between batches
EMBED_BATCH_SIZE = 80          # stay safely under the 100/min limit
EMBED_RATE_LIMIT_PAUSE = 62   # seconds to wait between batches (62s > 60s window)


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


def _embed_with_rate_limit(texts: List[str]) -> List[List[float]]:
    """
    Embed texts in batches, pausing between batches to respect
    the Gemini free tier rate limit (100 requests/minute).

    Each call to embed_documents() counts as 1 request regardless
    of how many texts are in the batch — so we batch up to
    EMBED_BATCH_SIZE texts per call and pause between calls.

    On paid tier you can raise EMBED_BATCH_SIZE to 500+ and set
    EMBED_RATE_LIMIT_PAUSE to 0.
    """
    embedder = get_embeddings()
    all_embeddings: List[List[float]] = []
    total_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i: i + EMBED_BATCH_SIZE]
        batch_num = i // EMBED_BATCH_SIZE + 1

        logger.info(
            f"Embedding batch {batch_num}/{total_batches} "
            f"({len(batch)} chunks, total so far: {i + len(batch)}/{len(texts)})"
        )

        batch_embeddings = embedder.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)

        # Pause between batches (not after the last one)
        if i + EMBED_BATCH_SIZE < len(texts):
            logger.info(
                f"Rate limit pause: waiting {EMBED_RATE_LIMIT_PAUSE}s "
                f"before next embedding batch..."
            )
            time.sleep(EMBED_RATE_LIMIT_PAUSE)

    return all_embeddings


def upsert_chunks(
    chunks: List[Dict[str, Any]],
    namespace: str,
    pinecone_batch_size: int = 100,
) -> int:
    """
    Embed and upsert a list of chunks into Pinecone.

    Embedding is rate-limited to stay within Gemini free tier (100 req/min).
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
        f"(namespace={namespace}, "
        f"batches of {EMBED_BATCH_SIZE} with {EMBED_RATE_LIMIT_PAUSE}s pause)"
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
) -> List[Dict[str, Any]]:
    """
    Embed query and retrieve top_k similar chunks from the user's namespace.
    Returns list of {id, score, text, metadata} dicts.
    """
    pc = get_pinecone_client()
    index = pc.Index(settings.PINECONE_INDEX_NAME)
    embedder = get_embeddings()

    # query_embedding = embedder.embed_query(query)
    query = sanitize_text(query)
    query_embedding = embedder.embed_query(query)

    result = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
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
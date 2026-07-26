import os
import uuid
from typing import List, Dict, Any, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logging import get_logger

import re
import unicodedata

logger = get_logger(__name__)

# Remove problematic control characters
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def sanitize_text(text: str) -> str:
    """
    Clean malformed PDF text before embedding/storage.

    Removes:
    - NULL bytes
    - ASCII control chars
    - invalid unicode normalization artifacts
    - excessive whitespace

    Safe for:
    - PostgreSQL
    - JSON serialization
    - Gemini embeddings
    - LangGraph checkpoints
    """

    if not text:
        return ""

    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove NULL bytes explicitly
    text = text.replace("\x00", "")

    # Remove remaining control chars
    text = CONTROL_CHAR_RE.sub("", text)

    # Remove weird repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_and_chunk_pdf(
    file_path: str,
    document_id: str,
    user_id: str,
    original_filename: str,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Load a PDF, split into chunks, and return a list of dicts ready for Pinecone upsert.

    Returns:
        (chunks, page_count)

    Each chunk dict: {
        "id": str,       # unique vector id
        "text": str,
        "metadata": {
            "document_id": str,
            "user_id": str,
            "filename": str,
            "page": int,
            "chunk_index": int,
            "total_chunks": int,   # filled in after splitting
        }
    }
    """
    logger.info(f"Loading PDF: {file_path}")

    loader = PyPDFLoader(file_path)
    pages = loader.load()
    page_count = len(pages)
    logger.info(f"Loaded {page_count} pages from {original_filename}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks = splitter.split_documents(pages)
    total_chunks = len(all_chunks)
    logger.info(f"Split into {total_chunks} chunks")

    result = []
    for idx, chunk in enumerate(all_chunks):
        # Use page number from metadata if available
        page_num = chunk.metadata.get("page", 0)
        cleaned_text = sanitize_text(chunk.page_content)

        if len(cleaned_text) < 20:
            continue

        result.append({
            "id": f"{document_id}_{idx}",
            "text": cleaned_text, # chunk.page_content.strip(),
            "metadata": {
                "document_id": document_id,
                "user_id": user_id,
                "filename": original_filename,
                "page": page_num,
                "chunk_index": idx,
                "total_chunks": total_chunks,
            },
        })

    # Filter out empty chunks
    # result = [c for c in result if len(c["text"]) > 20]
    logger.info(f"Final chunk count after filtering: {len(result)}")

    return result, page_count


def build_context_string(
    chunks: List[Dict[str, Any]],
    max_tokens: int = None,
) -> Tuple[str, List[Dict]]:
    """
    Combine reranked chunks into a context string for the LLM prompt,
    respecting max token budget (rough estimate: 1 token ≈ 4 chars).

    Returns:
        (context_string, source_list_for_citations)
    """
    max_chars = (max_tokens or settings.MAX_CONTEXT_TOKENS) * 4
    context_parts = []
    sources = []
    current_chars = 0

    for chunk in chunks:
        text =  sanitize_text(chunk["text"])
        meta = chunk.get("metadata", {})
        score = chunk.get("rerank_score", chunk.get("score", 0.0))

        filename = sanitize_text(meta.get("filename", "unknown"))
        header = (
            f"[Source: {filename} | "
            f"Page {meta.get('page', '?')} | "
            f"Chunk {meta.get('chunk_index', '?')}]"
        )
        block = f"{header}\n{text}"

        if current_chars + len(block) > max_chars:
            logger.info("Context budget reached, stopping chunk inclusion")
            break

        context_parts.append(block)
        current_chars += len(block)

        sources.append({
            "document_id": meta.get("document_id", ""),
            "filename": meta.get("filename", ""),
            "page": meta.get("page"),
            "chunk_index": meta.get("chunk_index", 0),
            "relevance_score": round(score, 4),
            "text_preview": text[:200],
        })

    return "\n\n---\n\n".join(context_parts), sources
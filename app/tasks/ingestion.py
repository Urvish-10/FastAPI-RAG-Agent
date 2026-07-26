import os
from celery import Task
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.tasks.celery_app import celery_app
from app.core.config import settings
from app.core.logging import get_logger
from app.models.models import Document, DocumentStatus
from app.services.pdf_processor import load_and_chunk_pdf
from app.services.vector_store import upsert_chunks

logger = get_logger(__name__)

# Sync engine for Celery (Celery workers are sync by nature)
sync_engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SyncSession = sessionmaker(bind=sync_engine)


def _cleanup_file(file_path: str) -> None:
    """Delete temp file from disk, ignoring errors."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temp file: {file_path}")
    except OSError as e:
        logger.warning(f"Could not delete temp file {file_path}: {e}")


class BaseTask(Task):
    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """
        Called by Celery only when all retries are exhausted.
        Safe to delete the file here — no more retries will need it.
        """
        logger.error(f"Task {task_id} permanently failed: {exc}")
        file_path = kwargs.get("file_path") or (args[1] if len(args) > 1 else None)
        if file_path:
            _cleanup_file(file_path)


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="app.tasks.ingestion.process_pdf",
    max_retries=3,
    default_retry_delay=30,
)
def process_pdf(
    self,
    document_id: str,
    file_path: str,
    user_id: str,
    original_filename: str,
) -> dict:
    """
    Celery task: load PDF → chunk → embed → upsert to Pinecone.
    Updates Document status in PostgreSQL throughout.

    File cleanup rules:
    - SUCCESS → delete file immediately after upsert
    - RETRY   → keep file on disk (next retry attempt needs it)
    - FINAL FAILURE (all retries exhausted) → on_failure() deletes it
    """
    db = SyncSession()
    try:
        # Mark as PROCESSING
        doc = db.query(Document).filter_by(id=document_id).first()
        if not doc:
            raise ValueError(f"Document {document_id} not found in DB")

        doc.status = DocumentStatus.PROCESSING.value
        db.commit()
        logger.info(f"Processing document {document_id}: {original_filename}")

        # Verify file still exists before attempting to load
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"PDF file not found at {file_path}. "
                f"It may have been deleted by a previous failed attempt."
            )

        # Load and chunk
        chunks, page_count = load_and_chunk_pdf(
            file_path=file_path,
            document_id=document_id,
            user_id=user_id,
            original_filename=original_filename,
        )

        # Upsert into Pinecone (namespace = user_id for isolation)
        total_upserted = upsert_chunks(
            chunks=chunks,
            namespace=user_id,
        )

        # Mark as READY
        doc.status = DocumentStatus.READY.value
        doc.chunk_count = total_upserted
        doc.page_count = page_count
        doc.pinecone_namespace = user_id
        db.commit()

        logger.info(
            f"Document {document_id} ingested successfully: "
            f"{page_count} pages, {total_upserted} chunks upserted"
        )

        # SUCCESS — safe to delete the file now
        _cleanup_file(file_path)

        return {
            "document_id": document_id,
            "page_count": page_count,
            "chunk_count": total_upserted,
        }

    except Exception as exc:
        logger.error(f"Error processing document {document_id}: {exc}", exc_info=True)

        # Update DB status only if retries are exhausted
        if self.request.retries >= self.max_retries:
            try:
                doc = db.query(Document).filter_by(id=document_id).first()
                if doc:
                    doc.status = DocumentStatus.FAILED.value
                    doc.error_message = str(exc)[:500]
                    db.commit()
            except Exception:
                pass

        # Retry with exponential backoff — file is kept on disk for next attempt
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 30)

    finally:
        db.close()
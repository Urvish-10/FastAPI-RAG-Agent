import asyncio
import os
import uuid
import aiofiles
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.database import get_db
from app.models.models import Document, DocumentStatus
from app.schemas.schemas import DocumentOut, DocumentListResponse, TaskStatusResponse
from app.core.security import get_current_user
from app.core.config import settings
from app.core.logging import get_logger
from app.tasks.ingestion import process_pdf
from app.tasks.celery_app import celery_app
from app.services.vector_store import delete_document_chunks

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_CONTENT_TYPES = {"application/pdf"}
MAX_FILE_SIZE = settings.MAX_FILE_SIZE_MB * 1024 * 1024  # bytes


@router.post("/upload", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a PDF document. Returns immediately with status=pending.
    Processing happens asynchronously via Celery.
    """
    # Validate content type (client-supplied; real check is the magic bytes below)
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    user_id = current_user["sub"]
    document_id = str(uuid.uuid4())

    # Stream to disk in chunks, enforcing the size cap without buffering the file
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    safe_filename = f"{document_id}.pdf"
    file_path = os.path.join(settings.UPLOAD_DIR, safe_filename)

    size = 0
    try:
        async with aiofiles.open(file_path, "wb") as f:
            header = await file.read(5)
            if header != b"%PDF-":
                raise HTTPException(status_code=400, detail="File is not a valid PDF")
            size = len(header)
            await f.write(header)
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {settings.MAX_FILE_SIZE_MB}MB limit",
                    )
                await f.write(chunk)
    except BaseException:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise

    # Create DB record
    doc = Document(
        id=document_id,
        user_id=user_id,
        filename=safe_filename,
        original_filename=file.filename,
        file_size_bytes=size,
        status=DocumentStatus.PENDING,
    )
    db.add(doc)
    await db.flush()

    # Dispatch Celery task
    task = process_pdf.delay(
        document_id=document_id,
        file_path=file_path,
        user_id=user_id,
        original_filename=file.filename,
    )

    doc.celery_task_id = task.id
    logger.info(f"Dispatched ingestion task {task.id} for document {document_id}")

    return doc


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    """List all documents for the authenticated user."""
    user_id = current_user["sub"]

    result = await db.execute(
        select(Document)
        .where(Document.user_id == user_id)
        .order_by(Document.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    docs = result.scalars().all()

    total_result = await db.execute(
        select(func.count()).select_from(Document).where(Document.user_id == user_id)
    )
    total = total_result.scalar()

    return DocumentListResponse(documents=list(docs), total=total)


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    doc = await _get_user_doc(document_id, current_user["sub"], db)
    return doc


@router.get("/{document_id}/task-status", response_model=TaskStatusResponse)
async def get_task_status(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Poll Celery task status for a document ingestion job."""
    doc = await _get_user_doc(document_id, current_user["sub"], db)

    if not doc.celery_task_id:
        return TaskStatusResponse(task_id="", status="no_task")

    result = celery_app.AsyncResult(doc.celery_task_id)
    return TaskStatusResponse(
        task_id=doc.celery_task_id,
        status=result.status,
        result=result.result if result.ready() else None,
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a document and remove its vectors from Pinecone."""
    user_id = current_user["sub"]
    doc = await _get_user_doc(document_id, user_id, db)

    # An in-flight ingestion task would re-upsert vectors after we delete them
    if doc.status in (DocumentStatus.PENDING.value, DocumentStatus.PROCESSING.value):
        raise HTTPException(
            status_code=409,
            detail="Document is still being processed; try again once it is ready or failed",
        )

    # Remove vectors first; if that fails keep the DB row so the delete can be retried
    try:
        await asyncio.to_thread(
            delete_document_chunks, document_id=document_id, namespace=user_id
        )
    except Exception as e:
        logger.error(f"Failed to delete Pinecone vectors for {document_id}: {e}")
        raise HTTPException(status_code=502, detail="Failed to delete document vectors")

    await db.delete(doc)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_user_doc(document_id: str, user_id: str, db: AsyncSession) -> Document:
    result = await db.execute(
        select(Document).where(Document.id == document_id, Document.user_id == user_id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

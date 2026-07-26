import os
import uuid
import aiofiles
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
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
    # Validate content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Read and validate file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.MAX_FILE_SIZE_MB}MB limit",
        )

    user_id = current_user["sub"]
    document_id = str(uuid.uuid4())

    # Save file to disk for Celery worker
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    safe_filename = f"{document_id}.pdf"
    file_path = os.path.join(settings.UPLOAD_DIR, safe_filename)

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    # Create DB record
    doc = Document(
        id=document_id,
        user_id=user_id,
        filename=safe_filename,
        original_filename=file.filename,
        file_size_bytes=len(content),
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
    skip: int = 0,
    limit: int = 50,
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

    # Remove from Pinecone
    try:
        delete_document_chunks(document_id=document_id, namespace=user_id)
    except Exception as e:
        logger.warning(f"Failed to delete Pinecone vectors for {document_id}: {e}")

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

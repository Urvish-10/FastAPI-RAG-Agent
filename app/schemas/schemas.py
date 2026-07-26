from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Any
from datetime import datetime
from app.models.models import DocumentStatus


# ── Auth ──────────────────────────────────────────────────────────────────────

class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: str
    email: str
    created_at: datetime

    class Config:
        from_attributes = True


# ── Documents ─────────────────────────────────────────────────────────────────

class DocumentOut(BaseModel):
    id: str
    filename: str
    original_filename: str
    file_size_bytes: Optional[int]
    status: DocumentStatus
    error_message: Optional[str]
    chunk_count: Optional[int]
    page_count: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    documents: List[DocumentOut]
    total: int


# ── Sessions ──────────────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    title: Optional[str] = None


class SessionOut(BaseModel):
    id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    stream: bool = True


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    page: Optional[int]
    chunk_index: int
    relevance_score: float
    text_preview: str   # first 200 chars


class ChatMessageOut(BaseModel):
    id: str
    role: str
    content: str
    sources: Optional[List[SourceChunk]]
    created_at: datetime

    class Config:
        from_attributes = True


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[ChatMessageOut]


# ── Celery task status ────────────────────────────────────────────────────────

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Any] = None

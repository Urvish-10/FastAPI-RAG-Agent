from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # App
    APP_NAME: str = "RAG Agent API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # PostgreSQL
    DATABASE_URL: str
    # Async version used by SQLAlchemy / app
    DATABASE_URL_ASYNC: str

    # Redis (Celery broker + backend)
    REDIS_URL: str = "redis://localhost:6379/0"

    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "rag-agent"
    PINECONE_ENVIRONMENT: str = "us-east-1"

    # Google Gemini
    GEMINI_API_KEY: str
    GEMINI_CHAT_MODEL: str = "gemini-2.0-flash"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
    GEMINI_EMBEDDING_DIMENSIONS: int = 768

    # Pinecone Rerank (uses same PINECONE_API_KEY, no extra key needed)
    PINECONE_RERANK_MODEL: str = "bge-reranker-v2-m3"

    # RAG parameters
    RETRIEVAL_TOP_K: int = 20       # Fetch from Pinecone
    RERANK_TOP_N: int = 5           # Keep after reranking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_CONTEXT_TOKENS: int = 8000  # Limit context sent to LLM

    # File storage
    UPLOAD_DIR: str = "/tmp/rag_uploads"
    MAX_FILE_SIZE_MB: int = 50

    # JWT (simple API-key auth for now)
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
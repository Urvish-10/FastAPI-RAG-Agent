import os

# Settings are read at import time; provide dummies before any app import.
os.environ.setdefault("DATABASE_URL", "postgresql://u:p@localhost/db")
os.environ.setdefault("DATABASE_URL_ASYNC", "postgresql+asyncpg://u:p@localhost/db")
os.environ.setdefault("PINECONE_API_KEY", "test")
os.environ.setdefault("GEMINI_API_KEY", "test")
os.environ.setdefault("SECRET_KEY", "x" * 40)

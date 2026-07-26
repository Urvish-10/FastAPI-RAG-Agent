# RAG Agent API

Production-grade Retrieval-Augmented Generation API with multi-turn conversations, built with **LangGraph**, **Google Gemini**, **Pinecone**, and **Pinecone Inference reranking**.

---

## Architecture

### Ingestion pipeline

```
POST /documents/upload
  → Save PDF to disk
  → Create DB record (status: pending)
  → Dispatch Celery task
      → PyPDFLoader → RecursiveCharacterTextSplitter
      → Gemini text-embedding-004
      → Pinecone upsert (namespace = user_id)
  → Update DB record (status: ready, chunk_count, page_count)
```

### Query pipeline

```
POST /sessions/{id}/chat
  → Load conversation history from PostgreSQL (via LangGraph PostgresSaver)
  → retrieve node: Pinecone query → top-20 chunks (user namespace)
  → rerank node:   Pinecone Inference rerank → top-5 chunks
  → generate node: Gemini Flash with history + context
  → Stream SSE tokens back to client
  → Save assistant message + sources to DB
```

---

## Tech stack

| Component        | Technology                          |
|-----------------|--------------------------------------|
| API framework    | FastAPI + Uvicorn                   |
| Agent orchestration | LangGraph StateGraph             |
| LLM              | Google Gemini 2.0 Flash             |
| Embeddings       | Gemini text-embedding-004 (768 dim) |
| Vector store     | Pinecone (serverless)               |
| Reranking        | Pinecone Inference API (bge-reranker-v2-m3) |
| Memory           | LangGraph PostgresSaver             |
| Task queue       | Celery + Redis                      |
| Database         | PostgreSQL (SQLAlchemy async)       |
| Migrations       | Alembic                             |

---

## Setup

### 1. Prerequisites

- Python 3.11+
- Docker + Docker Compose (for PostgreSQL and Redis)
- Pinecone account (free tier works)
- Google AI Studio API key (Gemini)
# No Cohere needed — reranking is handled by Pinecone Inference API

### 2. Configure environment

```bash
cp .env.example .env
# Fill in: DATABASE_URL, DATABASE_URL_ASYNC, REDIS_URL,
#          PINECONE_API_KEY, GEMINI_API_KEY, SECRET_KEY
```

### 3. Start infrastructure

```bash
docker-compose up postgres redis -d
```

### 4. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5. Run database migrations

```bash
alembic upgrade head
```

### 6. Start the API server

```bash
uvicorn app.main:app --reload --port 8000
```

### 7. Start the Celery worker (separate terminal)

```bash
uv run celery -A app.tasks.celery_app worker --pool=threads --concurrency=4 --loglevel=info
```

Or run everything with Docker Compose:

```bash
docker-compose up --build
```

---

## API reference

Interactive docs: `http://localhost:8000/docs`

### Auth

| Method | Endpoint              | Description        |
|--------|-----------------------|--------------------|
| POST   | `/api/v1/auth/register` | Register a user  |
| POST   | `/api/v1/auth/login`    | Get access token |
| GET    | `/api/v1/auth/me`       | Current user     |

### Documents

| Method | Endpoint                                | Description                  |
|--------|-----------------------------------------|------------------------------|
| POST   | `/api/v1/documents/upload`              | Upload a PDF (async)         |
| GET    | `/api/v1/documents`                     | List user's documents        |
| GET    | `/api/v1/documents/{id}`                | Get document status          |
| GET    | `/api/v1/documents/{id}/task-status`    | Poll Celery ingestion status |
| DELETE | `/api/v1/documents/{id}`                | Delete document + vectors    |

### Sessions & Chat

| Method | Endpoint                             | Description                      |
|--------|--------------------------------------|----------------------------------|
| POST   | `/api/v1/sessions`                   | Create chat session              |
| GET    | `/api/v1/sessions`                   | List sessions                    |
| GET    | `/api/v1/sessions/{id}/history`      | Get full chat history            |
| POST   | `/api/v1/sessions/{id}/chat`         | Send message (streaming or JSON) |
| DELETE | `/api/v1/sessions/{id}`              | Delete session                   |

### Chat streaming

Send `"stream": true` (default) to get an SSE stream:

```
data: {"type": "token", "content": "The "}
data: {"type": "token", "content": "answer "}
...
data: {"type": "sources", "sources": [{...}, ...]}
data: {"type": "done"}
```

Send `"stream": false` for a plain JSON response:

```json
{
  "session_id": "...",
  "answer": "The contract states...",
  "sources": [
    {
      "document_id": "...",
      "filename": "contract.pdf",
      "page": 4,
      "chunk_index": 12,
      "relevance_score": 0.9821,
      "text_preview": "Section 3.2 states that..."
    }
  ]
}
```

---

## Example usage

```bash
# 1. Register
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123"}'

# 2. Login
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123"}' \
  | jq -r .access_token)

# 3. Upload a PDF
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/document.pdf"

# 4. Create a session
SESSION=$(curl -s -X POST http://localhost:8000/api/v1/sessions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title": "My first chat"}' | jq -r .id)

# 5. Chat (streaming)
curl -N -X POST http://localhost:8000/api/v1/sessions/$SESSION/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the key findings in this document?", "stream": true}'
```

---

## Project structure

```
rag-agent/
├── app/
│   ├── main.py               # FastAPI app + lifespan
│   ├── core/
│   │   ├── config.py         # Settings (pydantic-settings)
│   │   ├── security.py       # JWT auth
│   │   └── logging.py        # Logging setup
│   ├── db/
│   │   └── database.py       # Async SQLAlchemy engine + session
│   ├── models/
│   │   └── models.py         # ORM: User, Document, ChatSession, ChatMessage
│   ├── schemas/
│   │   └── schemas.py        # Pydantic request/response models
│   ├── services/
│   │   ├── vector_store.py   # Pinecone: upsert, query, delete
│   │   ├── reranker.py       # Pinecone Inference reranking
│   │   └── pdf_processor.py  # PDF loading, chunking, context building
│   ├── agent/
│   │   ├── graph.py          # LangGraph StateGraph, nodes, compilation
│   │   └── runner.py         # Agent invocation + streaming adapter
│   ├── tasks/
│   │   ├── celery_app.py     # Celery app config
│   │   └── ingestion.py      # PDF ingestion Celery task
│   └── api/routes/
│       ├── auth.py           # /auth endpoints
│       ├── documents.py      # /documents endpoints
│       └── chat.py           # /sessions + /chat endpoints
├── alembic/                  # DB migrations
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Key design decisions

**Per-user Pinecone namespaces** — every user's chunks are isolated in `namespace=user_id`. Queries never touch another user's data.

**Pinecone-native reranking** — reranking uses Pinecone's built-in Inference Rerank API (`bge-reranker-v2-m3`) instead of a separate Cohere dependency. Same `PINECONE_API_KEY`, one fewer SDK, and the multilingual BGE cross-encoder handles most document types well. Constraints: max 256 query tokens, max 1024 doc tokens, max 100 documents per call — all within our `RETRIEVAL_TOP_K=20` budget.

**PostgresSaver for multi-turn memory** — LangGraph's `AsyncPostgresSaver` checkpoints the full message history per `thread_id` (= `session_id`) in PostgreSQL. Conversations survive restarts.

**Two embedding task types** — Gemini embeddings accept a `task_type` parameter. Documents are embedded with `retrieval_document`, queries with `retrieval_query`. This is important — using the wrong task type degrades retrieval quality.

**Async-first** — FastAPI routes, SQLAlchemy, and LangGraph graph invocation are all async. Celery workers use sync SQLAlchemy (Celery's own event loop).

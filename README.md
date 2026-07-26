# 🤖 FastAPI RAG Agent

> A production-grade Retrieval-Augmented Generation API with multi-turn conversations, async document ingestion, and per-user vector isolation — built on **LangGraph**, **Google Gemini**, **Pinecone**, **Celery**, and **PostgreSQL**.

---

## What This Is

This project is a fully working RAG backend that lets users upload PDF documents and have contextual, multi-turn conversations about them. It is not a prototype — it handles async ingestion, streaming responses, per-user data isolation, and persistent conversation memory, all wired together in a clean, async-first FastAPI application.

**Core idea:** Upload a PDF → it gets chunked and embedded in the background → ask questions about it → get streamed answers backed by cited source chunks.

---

## Architecture

### Ingestion Pipeline (async, background)

```
POST /documents/upload
  → Save PDF to disk
  → Create DB record (status: pending)
  → Dispatch Celery task via Redis
      → PyPDFLoader → RecursiveCharacterTextSplitter
      → Gemini text-embedding-004 (task_type: retrieval_document)
      → Pinecone upsert (namespace = user_id)
  → Update DB record (status: ready, chunk_count, page_count)
```

### Query Pipeline (LangGraph StateGraph)

```
POST /sessions/{id}/chat
  → Load conversation history from PostgreSQL (LangGraph AsyncPostgresSaver)
  → [retrieve node]  Pinecone query → top-20 chunks (user's namespace only)
  → [rerank node]    Pinecone Inference API (bge-reranker-v2-m3) → top-5 chunks
  → [generate node]  Gemini 2.0 Flash with history + reranked context
  → Stream SSE tokens back to client
  → Persist assistant message + sources to DB
```

### System Overview

```
┌──────────────┐     HTTP/SSE      ┌─────────────────────────────────┐
│   Client     │ ◄────────────────► │         FastAPI (async)          │
└──────────────┘                   └───────┬─────────────┬────────────┘
                                           │             │
                              ┌────────────▼──┐   ┌──────▼──────────┐
                              │  LangGraph    │   │  Celery Worker  │
                              │  StateGraph   │   │  (ingestion)    │
                              └──┬─────────┬─┘   └──────┬──────────┘
                                 │         │            │
                        ┌────────▼─┐  ┌────▼──────┐  ┌─▼──────┐
                        │ Pinecone │  │ PostgreSQL │  │ Redis  │
                        │ (vectors)│  │ (history)  │  │(queue) │
                        └──────────┘  └───────────┘  └────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| Agent Orchestration | LangGraph `StateGraph` |
| LLM | Google Gemini 2.0 Flash |
| Embeddings | Gemini `text-embedding-004` (768-dim) |
| Vector Store | Pinecone (serverless) |
| Reranking | Pinecone Inference API (`bge-reranker-v2-m3`) |
| Conversation Memory | LangGraph `AsyncPostgresSaver` |
| Background Tasks | Celery + Redis |
| Database | PostgreSQL (SQLAlchemy async) |
| Migrations | Alembic |
| Package Manager | uv |
| Containerisation | Docker + Docker Compose |

---

## Key Design Decisions

**Per-user Pinecone namespaces** — every user's chunks are isolated under `namespace=user_id`. Queries are strictly scoped; one user can never retrieve another user's documents.

**Pinecone-native reranking** — uses Pinecone's built-in Inference Rerank API (`bge-reranker-v2-m3`) instead of a separate Cohere dependency. Same `PINECONE_API_KEY`, one fewer SDK. The multilingual BGE cross-encoder handles most document types well. Constraints respected: max 256 query tokens, max 1024 doc tokens, max 100 documents per call — comfortably within the `RETRIEVAL_TOP_K=20` budget.

**Task-typed embeddings** — Gemini's embedding model accepts a `task_type` parameter. Documents are embedded with `retrieval_document`; queries use `retrieval_query`. Using the wrong task type measurably degrades retrieval quality — this is intentional, not incidental.

**AsyncPostgresSaver for multi-turn memory** — LangGraph checkpoints the full message history per `thread_id` (= `session_id`) directly in PostgreSQL. Conversations survive server restarts; no separate memory store needed.

**Async-first throughout** — FastAPI routes, SQLAlchemy sessions, and LangGraph graph invocation are all async. Celery workers use synchronous SQLAlchemy since Celery manages its own event loop.

**Two-stage retrieval (retrieve → rerank)** — fetching top-20 then reranking to top-5 is a deliberate trade-off: broad recall at the vector stage, high precision at the generation stage. This pattern comes directly from the RAG literature and avoids stuffing the context window with marginally relevant chunks.

---

## Project Structure

```
rag-agent/
├── app/
│   ├── main.py                   # FastAPI app, lifespan, middleware
│   ├── core/
│   │   ├── config.py             # Settings via pydantic-settings
│   │   ├── security.py           # JWT auth (issue + verify)
│   │   └── logging.py            # Structured logging setup
│   ├── db/
│   │   └── database.py           # Async SQLAlchemy engine + session factory
│   ├── models/
│   │   └── models.py             # ORM: User, Document, ChatSession, ChatMessage
│   ├── schemas/
│   │   └── schemas.py            # Pydantic request / response models
│   ├── services/
│   │   ├── vector_store.py       # Pinecone: upsert, query, delete
│   │   ├── reranker.py           # Pinecone Inference reranking
│   │   └── pdf_processor.py      # PDF loading, chunking, context building
│   ├── agent/
│   │   ├── graph.py              # LangGraph StateGraph definition + nodes
│   │   └── runner.py             # Agent invocation + SSE streaming adapter
│   ├── tasks/
│   │   ├── celery_app.py         # Celery app config + Redis broker
│   │   └── ingestion.py          # PDF ingestion task (embed + upsert)
│   └── api/routes/
│       ├── auth.py               # /auth endpoints
│       ├── documents.py          # /documents endpoints
│       └── chat.py               # /sessions + /chat endpoints
├── alembic/                      # DB migration scripts
├── docker-compose.yml            # PostgreSQL + Redis + app + worker
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── .env.example
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- [Pinecone account](https://www.pinecone.io/) (free tier works)
- [Google AI Studio API key](https://aistudio.google.com/) (Gemini)

### 1. Clone and configure

```bash
git clone https://github.com/Urvish-10/FastAPI-RAG-Agent.git
cd FastAPI-RAG-Agent
cp .env.example .env
# Fill in: DATABASE_URL, DATABASE_URL_ASYNC, REDIS_URL,
#          PINECONE_API_KEY, GEMINI_API_KEY, SECRET_KEY
```

### 2. Run locally

```bash
# Start infrastructure (PostgreSQL + Redis)
docker-compose up postgres redis -d

# Install dependencies
pip install uv
uv sync

# Run migrations
alembic upgrade head

# Start API server
uvicorn app.main:app --reload --port 8000

# Start Celery worker (separate terminal)
uv run celery -A app.tasks.celery_app worker --pool=threads --concurrency=4 --loglevel=info
```

API docs available at: `http://localhost:8000/docs`

---

## API Reference

### Auth

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/auth/register` | Register a new user |
| `POST` | `/api/v1/auth/login` | Get JWT access token |
| `GET` | `/api/v1/auth/me` | Get current user info |

### Documents

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/documents/upload` | Upload a PDF (triggers async ingestion) |
| `GET` | `/api/v1/documents` | List user's documents |
| `GET` | `/api/v1/documents/{id}` | Get document + status |
| `GET` | `/api/v1/documents/{id}/task-status` | Poll Celery ingestion progress |
| `DELETE` | `/api/v1/documents/{id}` | Delete document + Pinecone vectors |

### Sessions & Chat

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/sessions` | Create a new chat session |
| `GET` | `/api/v1/sessions` | List all sessions |
| `GET` | `/api/v1/sessions/{id}/history` | Get full conversation history |
| `POST` | `/api/v1/sessions/{id}/chat` | Send a message (streaming or JSON) |
| `DELETE` | `/api/v1/sessions/{id}` | Delete session |

### Streaming Response Format (SSE)

Send `"stream": true` (default):

```
data: {"type": "token", "content": "The "}
data: {"type": "token", "content": "contract "}
...
data: {"type": "sources", "sources": [{...}]}
data: {"type": "done"}
```

Send `"stream": false` for plain JSON:

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

## Example Usage (curl)

```bash
# 1. Register
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123"}'

# 2. Login and grab token
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

# 5. Chat with streaming
curl -N -X POST http://localhost:8000/api/v1/sessions/$SESSION/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the key findings?", "stream": true}'
```

---

## Papers Behind This Project

The architecture is a practical implementation of ideas from:

- **RAG** — Lewis et al., 2020 — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* ([arXiv:2005.11401](https://arxiv.org/abs/2005.11401)) — the two-stage retrieve + generate pattern
- **ReAct** — Yao et al., 2023 — *Synergizing Reasoning and Acting in Language Models* ([arXiv:2210.03629](https://arxiv.org/abs/2210.03629)) — the LangGraph node/edge model maps directly to the ReAct loop

---

> 🐳 **Docker support** is planned for a future release. For now, use the local setup instructions.

---

## Author

**Urvish Bhatt** — Software Engineer (Python · FastAPI · Django · AI Agents · R&D)

- 📧 urvishh.bhatt@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/urvish-bhatt)
- 🐙 [GitHub](https://github.com/Urvish-10)
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "rag_agent",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.tasks.ingestion"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,   # Process one task at a time per worker (PDF is heavy)
    result_expires=3600,            # Task results expire after 1 hour
)

"""
Celery Application
------------------
Defines the Celery app that powers background pipeline execution.

Why Celery over raw ThreadPoolExecutor?
  - Workers run in SEPARATE PROCESSES (bypass Python GIL for CPU-bound work)
  - Distributed: workers can run on different machines / containers
  - Retry logic, task routing, and priority queues built in
  - Task state is persisted in Redis (result backend)
  - Integrates with monitoring tools (Flower, Prometheus)

Broker:  Redis  (job queue)
Backend: Redis  (task result / state storage)

To start a local worker (after `pip install celery redis`):
    celery -A app.workers.celery_app worker --loglevel=info --concurrency=2

To monitor tasks via Flower:
    celery -A app.workers.celery_app flower
"""
from __future__ import annotations

from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "telecom_vision",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.workers.tasks"],       # auto-discover tasks
)

celery_app.conf.update(
    # ── Serialisation ────────────────────────────────────────────────
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # ── Reliability ──────────────────────────────────────────────────
    task_acks_late=True,                 # ack only after task completes (no message loss on crash)
    task_reject_on_worker_lost=True,     # re-queue if worker dies mid-task

    # ── Timeouts ─────────────────────────────────────────────────────
    task_soft_time_limit=int(settings.JOB_TIMEOUT_SECONDS),      # raises SoftTimeLimitExceeded
    task_time_limit=int(settings.JOB_TIMEOUT_SECONDS) + 60,      # hard kill after that

    # ── Result expiry ────────────────────────────────────────────────
    result_expires=int(settings.JOB_RETENTION_HOURS * 3600),

    # ── Worker behaviour ────────────────────────────────────────────────
    worker_prefetch_multiplier=1,        # one task at a time per worker process
    task_track_started=True,             # enables STARTED state for polling

    # ── Timezone ─────────────────────────────────────────────────────
    timezone="UTC",
    enable_utc=True,

    # ── Connection Fail-Fast (Dev & Fallback) ─────────────────────────
    broker_connection_retry_on_startup=False,
    broker_connection_max_retries=5,
    broker_connection_timeout=2.0,
    result_backend_max_retries=5,
)

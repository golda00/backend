"""
Celery Tasks
------------
Wraps run_pipeline_sync as a Celery task so it executes in a
dedicated worker process (not in the FastAPI process).

Why this matters:
  - Celery workers are separate processes → true parallelism (no GIL)
  - FastAPI stays responsive while workers grind through GPU inference
  - Tasks can be retried automatically on failure
  - Worker count is controlled independently (scale workers ≠ scale API)

Usage from jobs.py:
    task = run_pipeline_task.delay(job_id)
    # or with explicit routing:
    run_pipeline_task.apply_async(args=[job_id], queue="gpu")

The task uses a SYNCHRONOUS Redis client (redis-py, not asyncio)
because Celery tasks run in regular (non-async) worker processes.
"""
from __future__ import annotations

import json
import logging
import time

import redis

from app.core.config import get_settings
from app.models.schemas import JobStatus
from app.workers.celery_app import celery_app
from app.workers.pipeline import run_pipeline_sync, run_fiber_overview_pipeline
from app.services.fiber_after import run_fiber_after_pipeline


logger = logging.getLogger(__name__)


@celery_app.task(
    name="run_fiber_overview",
    bind=True,
    max_retries=1,
    default_retry_delay=10,
)
def run_fiber_overview_task(self, job_id: str) -> dict:
    """
    Celery task entry point for Fiber Overview pipeline.
    """
    settings = get_settings()
    r = redis.from_url(settings.REDIS_URL, decode_responses=True)

    class _RedisSyncProxy(dict):
        def __init__(self, initial: dict):
            super().__init__(initial)
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
        def update(self, other=None, **kwargs):
            if other: super().update(other, **kwargs)
            else: super().update(**kwargs)
            _sync_update_job(r, job_id, dict(self))

    raw = r.get(f"telecom_job:{job_id}")
    if raw is None:
        logger.error(f"[task] Job {job_id!r} not found in Redis")
        return {"error": "Job not found"}

    initial_data = json.loads(raw)
    proxy_store = {job_id: _RedisSyncProxy(initial_data)}

    try:
        run_fiber_overview_pipeline(
            job_id=job_id,
            job_store=proxy_store,
            settings=settings,
            processor=None,
        )
    except Exception as exc:
        logger.exception(f"[task] Fiber pipeline failed: {exc}")
        _sync_update_job(r, job_id, {
            "status": JobStatus.FAILED,
            "message": "Fiber pipeline failed unexpectedly.",
            "error": str(exc),
        })
        try: raise self.retry(exc=exc)
        except self.MaxRetriesExceededError: pass

    return {"job_id": job_id, "status": "completed"}


@celery_app.task(
    name="run_fiber_before",
    bind=True,
    max_retries=1,
    default_retry_delay=10,
)
def run_fiber_before_task(self, job_id: str) -> dict:
    """
    Celery task entry point for Fiber Before map workflow.
    """
    settings = get_settings()
    r = redis.from_url(settings.REDIS_URL, decode_responses=True)

    class _RedisSyncProxy(dict):
        def __init__(self, initial: dict):
            super().__init__(initial)
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
        def update(self, other=None, **kwargs):
            if other: super().update(other, **kwargs)
            else: super().update(**kwargs)
            _sync_update_job(r, job_id, dict(self))

    raw = r.get(f"telecom_job:{job_id}")
    if raw is None:
        logger.error(f"[task] Job {job_id!r} not found in Redis")
        return {"error": "Job not found"}

    initial_data = json.loads(raw)
    proxy_store = {job_id: _RedisSyncProxy(initial_data)}

    from app.services.fiber_before import run_fiber_before_pipeline

    try:
        run_fiber_before_pipeline(
            job_id=job_id,
            job_store=proxy_store,
            settings=settings,
        )
    except Exception as exc:
        logger.exception(f"[task] Fiber Before pipeline failed: {exc}")
        _sync_update_job(r, job_id, {
            "status": JobStatus.FAILED,
            "message": "Fiber Before pipeline failed unexpectedly.",
            "error": str(exc),
        })
        try: raise self.retry(exc=exc)
        except self.MaxRetriesExceededError: pass

    return {"job_id": job_id, "status": "completed"}


@celery_app.task(
    name="run_coax_before",
    bind=True,
    max_retries=1,
    default_retry_delay=10,
)
def run_coax_before_task(self, job_id: str) -> dict:
    """
    Celery task entry point for Coax Before map workflow.
    """
    settings = get_settings()
    r = redis.from_url(settings.REDIS_URL, decode_responses=True)

    class _RedisSyncProxy(dict):
        def __init__(self, initial: dict):
            super().__init__(initial)
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
        def update(self, other=None, **kwargs):
            if other: super().update(other, **kwargs)
            else: super().update(**kwargs)
            _sync_update_job(r, job_id, dict(self))

    raw = r.get(f"telecom_job:{job_id}")
    if raw is None:
        logger.error(f"[task] Job {job_id!r} not found in Redis")
        return {"error": "Job not found"}

    initial_data = json.loads(raw)
    proxy_store = {job_id: _RedisSyncProxy(initial_data)}

    from app.services.coax_before import run_coax_before_pipeline

    try:
        run_coax_before_pipeline(
            job_id=job_id,
            job_store=proxy_store,
            settings=settings,
        )
    except Exception as exc:
        logger.exception(f"[task] Coax Before pipeline failed: {exc}")
        _sync_update_job(r, job_id, {
            "status": JobStatus.FAILED,
            "message": "Coax Before pipeline failed unexpectedly.",
            "error": str(exc),
        })
        try: raise self.retry(exc=exc)
        except self.MaxRetriesExceededError: pass

    return {"job_id": job_id, "status": "completed"}
@celery_app.task(
    name="run_fiber_after",
    bind=True,
    max_retries=1,
    default_retry_delay=10,
)
def run_fiber_after_task(self, job_id: str) -> dict:
    """
    Celery task entry point for the Fiber After ML pipeline.
    """
    settings = get_settings()
    r = redis.from_url(settings.REDIS_URL, decode_responses=True)

    class _RedisSyncProxy(dict):
        def __init__(self, initial: dict):
            super().__init__(initial)
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
        def update(self, other=None, **kwargs):
            if other: super().update(other, **kwargs)
            else: super().update(**kwargs)
            _sync_update_job(r, job_id, dict(self))

    raw = r.get(f"telecom_job:{job_id}")
    if raw is None:
        logger.error(f"[task] Job {job_id!r} not found in Redis")
        return {"error": "Job not found"}

    initial_data = json.loads(raw)
    proxy_store = {job_id: _RedisSyncProxy(initial_data)}

    try:
        run_fiber_after_pipeline(
            job_id=job_id,
            store=proxy_store,
            settings=settings,
        )
    except Exception as exc:
        logger.exception(f"[task] Fiber After pipeline failed: {exc}")
        _sync_update_job(r, job_id, {
            "status": JobStatus.FAILED,
            "message": "Fiber After pipeline failed unexpectedly.",
            "error": str(exc),
        })
        try: raise self.retry(exc=exc)
        except self.MaxRetriesExceededError: pass

    return {"job_id": job_id, "status": "completed"}


def _sync_update_job(r: redis.Redis, job_id: str, updates: dict, ttl: int = 86400) -> None:
    """
    Synchronously merge updates into a Redis job record.
    Used by the Celery task (no asyncio event loop in worker processes).
    """
    raw = r.get(f"telecom_job:{job_id}")
    if raw is None:
        logger.warning(f"[task] Job {job_id!r} not found in Redis for update.")
        return

    def _parse_status(val):
        if isinstance(val, str):
            try:
                return JobStatus(val)
            except ValueError:
                return val
        return val

    data = json.loads(raw)
    if "status" in data:
        data["status"] = _parse_status(data["status"])

    data.update(updates)

    # Re-serialise (convert enums back to strings)
    def _default(obj):
        if isinstance(obj, JobStatus):
            return obj.value
        raise TypeError(f"Not serialisable: {type(obj)}")

    r.set(f"telecom_job:{job_id}", json.dumps(data, default=_default), ex=ttl)


@celery_app.task(
    name="run_pipeline",
    bind=True,
    max_retries=1,          # retry once on unexpected failure
    default_retry_delay=10,
)
def run_pipeline_task(self, job_id: str) -> dict:
    """
    Celery task entry point for the full pipeline.

    Uses a synchronous Redis client (redis-py) to write job state because
    Celery worker processes don't have an asyncio event loop.

    The pipeline's run_pipeline_sync() is called with a simple dict-like
    proxy that writes updates straight to Redis.
    """
    settings = get_settings()

    # ── Connect to Redis (sync client inside Celery worker) ──────────
    r = redis.from_url(settings.REDIS_URL, decode_responses=True)

    # ── Build a sync dict-proxy so run_pipeline_sync can "update" it ─
    # run_pipeline_sync was designed to update a dict in-place; we wrap
    # a Redis write behind the same interface.
    class _RedisSyncProxy(dict):
        """
        Looks like a dict to run_pipeline_sync but persists every write
        to Redis. Reads are served from the local cache (for speed).
        """
        def __init__(self, initial: dict):
            super().__init__(initial)

        def __setitem__(self, key, value):
            super().__setitem__(key, value)

        def update(self, other=None, **kwargs):
            if other:
                super().update(other, **kwargs)
            else:
                super().update(**kwargs)
            # Flush full record to Redis after every update
            _sync_update_job(r, job_id, dict(self))

    raw = r.get(f"telecom_job:{job_id}")
    if raw is None:
        logger.error(f"[task] Job {job_id!r} not found in Redis — aborting.")
        return {"error": "Job not found"}

    initial_data = json.loads(raw)
    proxy_store = {job_id: _RedisSyncProxy(initial_data)}

    try:
        run_pipeline_sync(
            job_id=job_id,
            job_store=proxy_store,
            settings=settings,
            detector=None,      # Celery workers load models once at process init (see below)
        )
    except Exception as exc:
        logger.exception(f"[task] Pipeline failed for job {job_id}: {exc}")
        _sync_update_job(r, job_id, {
            "status": JobStatus.FAILED,
            "message": "Pipeline failed unexpectedly.",
            "error": str(exc),
        })
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            pass

    return {"job_id": job_id, "status": "completed"}

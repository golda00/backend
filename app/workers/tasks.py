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
from contextlib import contextmanager

from app.core.config import get_settings
from app.models.schemas import JobStatus
from app.workers.celery_app import celery_app
from app.workers.pipeline import run_pipeline_sync, run_fiber_overview_pipeline
from app.services.fiber_after import run_fiber_after_pipeline


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helper: load job with Redis → jobs.json fallback
# ─────────────────────────────────────────────────────────────────────────────

def _load_job_with_fallback(r: redis.Redis, job_id: str, settings) -> dict | None:
    """
    Try Redis first. If the job is not there (e.g. API used PersistentJobStore),
    fall back to reading storage/outputs/jobs.json written by PersistentJobStore.
    If found in the fallback, also write it into Redis so future lookups succeed.
    """
    raw = r.get(f"telecom_job:{job_id}")
    if raw is not None:
        return json.loads(raw)

    # ── Fallback: read from PersistentJobStore flat file ─────────────────────
    fallback_path = settings.OUTPUTS_DIR / "jobs.json"
    if fallback_path.exists():
        try:
            with open(fallback_path, "r", encoding="utf-8") as f:
                all_jobs = json.load(f)
            if job_id in all_jobs:
                job_data = all_jobs[job_id]
                # Backfill Redis so subsequent lookups work
                try:
                    r.set(f"telecom_job:{job_id}", json.dumps(job_data), ex=86400)
                    logger.info(f"[fallback] Backfilled job {job_id!r} from jobs.json into Redis.")
                except Exception:
                    pass
                return job_data
        except Exception as e:
            logger.warning(f"[fallback] Could not read jobs.json: {e}")

    return None

# ── Global AI Engine Singletons ──────────────────────────────────
_SHARED_DETECTOR = None
_SHARED_OVERVIEW_PROCESSOR = None

def _get_detector(settings):
    """Worker-level singleton: Loads main models once per process."""
    global _SHARED_DETECTOR
    if _SHARED_DETECTOR is None:
        from app.services.vision import TelecomDetector
        logger.info("Initializing Shared AI Engine (this takes ~3 mins)...")
        _SHARED_DETECTOR = TelecomDetector(
            main_model_path=settings.MAIN_MODEL_PATH,
            ps_model_path=settings.PS_MODEL_PATH,
            node_model_path=settings.NODE_MODEL_PATH,
            internal_model_path=settings.INTERNAL_MODEL_PATH,
            use_gpu=settings.USE_GPU,
            dpi=settings.PDF_DPI,
        )
        logger.info("✅ Shared AI Engine ready.")
    return _SHARED_DETECTOR

def _get_overview_processor(settings):
    """Worker-level singleton: Loads Fiber Node model once."""
    global _SHARED_OVERVIEW_PROCESSOR
    if _SHARED_OVERVIEW_PROCESSOR is None:
        from app.services.fiber_overview import FiberOverviewProcessor
        logger.info("Initializing Fiber Overview Processor...")
        _SHARED_OVERVIEW_PROCESSOR = FiberOverviewProcessor(
            model_path=settings.FIBER_NODE_MODEL_PATH
        )
        logger.info("✅ Fiber Overview Processor ready.")
    return _SHARED_OVERVIEW_PROCESSOR


@celery_app.task(
    name="run_fiber_overview",
    bind=True,
    max_retries=1,
    default_retry_delay=10,
)
def run_fiber_overview_task(self, job_id: str, job_data: dict | None = None) -> dict:
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

    # Try provided data first, then fallback
    initial_data = job_data or _load_job_with_fallback(r, job_id, settings)
    
    if initial_data is None:
        logger.error(f"[task] Job {job_id!r} not found (no job_data, Redis, or jobs.json)")
        return {"error": "Job not found"}

    # If we got data from the message but it's missing from Redis, backfill now
    if job_data and not r.exists(f"telecom_job:{job_id}"):
        try:
            r.set(f"telecom_job:{job_id}", json.dumps(job_data), ex=86400)
            logger.info(f"[task] Backfilled Redis for {job_id!r} using job_data from message.")
        except Exception: pass

    proxy_store = {job_id: _RedisSyncProxy(initial_data)}

    logger.info(f"[{job_id}] 🚀 Fiber Overview task started.")
    try:
        run_fiber_overview_pipeline(
            job_id=job_id,
            job_store=proxy_store,
            settings=settings,
            processor=_get_overview_processor(settings),
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
def run_fiber_before_task(self, job_id: str, job_data: dict | None = None) -> dict:
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

    initial_data = job_data or _load_job_with_fallback(r, job_id, settings)
    if initial_data is None:
        logger.error(f"[task] Job {job_id!r} not found")
        return {"error": "Job not found"}

    if job_data and not r.exists(f"telecom_job:{job_id}"):
        try:
            r.set(f"telecom_job:{job_id}", json.dumps(job_data), ex=86400)
        except Exception: pass

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
def run_coax_before_task(self, job_id: str, job_data: dict | None = None) -> dict:
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

    initial_data = job_data or _load_job_with_fallback(r, job_id, settings)
    if initial_data is None:
        logger.error(f"[task] Job {job_id!r} not found")
        return {"error": "Job not found"}

    if job_data and not r.exists(f"telecom_job:{job_id}"):
        try:
            r.set(f"telecom_job:{job_id}", json.dumps(job_data), ex=86400)
        except Exception: pass

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
def run_fiber_after_task(self, job_id: str, job_data: dict | None = None) -> dict:
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

    initial_data = job_data or _load_job_with_fallback(r, job_id, settings)
    if initial_data is None:
        logger.error(f"[task] Job {job_id!r} not found")
        return {"error": "Job not found"}

    if job_data and not r.exists(f"telecom_job:{job_id}"):
        try:
            r.set(f"telecom_job:{job_id}", json.dumps(job_data), ex=86400)
        except Exception: pass

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
    Falls back to reading from jobs.json if the key isn't in Redis yet.
    """
    raw = r.get(f"telecom_job:{job_id}")
    if raw is None:
        # Try to backfill from jobs.json fallback
        settings = get_settings()
        fallback_path = settings.OUTPUTS_DIR / "jobs.json"
        if fallback_path.exists():
            try:
                with open(fallback_path, "r", encoding="utf-8") as f:
                    all_jobs = json.load(f)
                if job_id in all_jobs:
                    raw = json.dumps(all_jobs[job_id])
                    logger.info(f"[_sync_update_job] Loaded {job_id!r} from jobs.json fallback.")
            except Exception as e:
                logger.warning(f"[_sync_update_job] Could not read jobs.json: {e}")
        if raw is None:
            logger.warning(f"[task] Job {job_id!r} not found in Redis or jobs.json for update.")
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
def run_pipeline_task(self, job_id: str, job_data: dict | None = None) -> dict:
    """
    Celery task entry point for the full pipeline.
    """
    settings = get_settings()

    # ── Connect to Redis (sync client inside Celery worker) ──────────
    r = redis.from_url(settings.REDIS_URL, decode_responses=True)

    class _RedisSyncProxy(dict):
        def __init__(self, initial: dict):
            super().__init__(initial)

        def __setitem__(self, key, value):
            super().__setitem__(key, value)

        def update(self, other=None, **kwargs):
            if other:
                super().update(other, **kwargs)
            else:
                super().update(**kwargs)
            _sync_update_job(r, job_id, dict(self))

    initial_data = job_data or _load_job_with_fallback(r, job_id, settings)
    if initial_data is None:
        logger.error(f"[task] Job {job_id!r} not found — aborting.")
        return {"error": "Job not found"}

    if job_data and not r.exists(f"telecom_job:{job_id}"):
        try:
            r.set(f"telecom_job:{job_id}", json.dumps(job_data), ex=86400)
        except Exception: pass

    proxy_store = {job_id: _RedisSyncProxy(initial_data)}

    logger.info(f"[{job_id}] 🚀 Main Coax/Fiber task started.")
    try:
        run_pipeline_sync(
            job_id=job_id,
            job_store=proxy_store,
            settings=settings,
            detector=_get_detector(settings), # Use the Warm Engine
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

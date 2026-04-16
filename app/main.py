"""
FastAPI Application Entry Point
--------------------------------
Creates the FastAPI app, wires up middleware, registers routers,
and manages the lifespan (model pre-loading on startup).

Batch 1 Optimization additions:
  - Bounded ThreadPoolExecutor for pipeline jobs (avoids starving I/O pool)
  - Upload size limit middleware (DoS protection)
  - Auto-cleanup background task (prevents disk exhaustion)
"""
from __future__ import annotations

import asyncio
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.api.v1 import jobs, health

# Initialise logging before anything else
setup_logging()
logger = get_logger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────
#  Upload size limiter middleware
# ─────────────────────────────────────────────

class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    """
    OPT: Rejects oversized requests before they hit the route handler.
    Without this a user could upload a multi-GB file and exhaust disk/RAM.
    Limit is configurable via MAX_UPLOAD_BYTES in .env (default 200 MB).
    """
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.MAX_UPLOAD_BYTES:
            mb = settings.MAX_UPLOAD_BYTES // (1024 * 1024)
            return Response(
                content=f"Upload too large. Max allowed: {mb} MB.",
                status_code=413,
            )
        return await call_next(request)


# ─────────────────────────────────────────────
#  Background cleanup task
# ─────────────────────────────────────────────

async def _cleanup_old_jobs(app, retention_hours: float) -> None:
    """
    OPT: Runs every hour and deletes disk files for expired jobs.
    When Redis is the store, TTL handles record expiry automatically.
    For InMemoryJobStore, also prunes the in-memory dict.
    """
    while True:
        await asyncio.sleep(3600)
        store = getattr(app.state, "job_store", None)
        if store is None:
            continue

        cutoff = time.time() - (retention_hours * 3600)

        # InMemoryJobStore exposes ._data; Redis store relies solely on TTL
        jobs_to_clean = {}
        if hasattr(store, "_data"):
            jobs_to_clean = {
                jid: j for jid, j in list(store._data.items())
                if j.get("created_at", float("inf")) < cutoff
            }

        purged = 0
        for job_id, job in jobs_to_clean.items():
            out = job.get("output_dir")
            if out:
                shutil.rmtree(out, ignore_errors=True)
            bp = job.get("before_path")
            if bp:
                shutil.rmtree(str(Path(bp).parent), ignore_errors=True)
            await store.delete(job_id)
            purged += 1

        if purged:
            logger.info(f"[cleanup] Removed {purged} expired job(s).")




# ─────────────────────────────────────────────
#  Lifespan
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager — runs setup before first request and teardown on shutdown.
    Models are loaded here once so every request handler shares the same warm instance.
    """
    logger.info("🚀 Starting up Telecom Vision API...")

    # Ensure storage directories exist
    settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Batch 3: Redis connection + job store ────────────────────────
    redis_client = None
    try:
        import redis.asyncio as aioredis
        from app.core.store import RedisJobStore
        redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
        await redis_client.ping()   # verify connection
        app.state.job_store = RedisJobStore(
            redis_client,
            ttl=int(settings.JOB_RETENTION_HOURS * 3600),
        )
        logger.info(f"✅ Redis connected at {settings.REDIS_URL} — using RedisJobStore.")
    except Exception as redis_err:
        from app.core.store import PersistentJobStore
        app.state.job_store = PersistentJobStore(settings.STORAGE_DIR / "jobs.json")
        logger.warning(
            f"⚠️  Redis unavailable ({redis_err}). "
            "Falling back to PersistentJobStore — progress will be saved in jobs.json."
        )

    # OPT: Dedicated bounded thread pool for pipeline jobs.
    # Keeps pipeline threads separate from FastAPI's built-in I/O thread pool.
    # PIPELINE_WORKERS defaults to 4; tune based on CPU/GPU count.
    logger.info(f"Creating pipeline thread pool ({settings.PIPELINE_WORKERS} workers)...")
    app.state.pipeline_pool = ThreadPoolExecutor(
        max_workers=settings.PIPELINE_WORKERS,
        thread_name_prefix="pipeline",
    )

    # Pre-load models if all weight files exist
    models_exist = all(
        p.exists() for p in [
            settings.MAIN_MODEL_PATH,
            settings.PS_MODEL_PATH,
            settings.NODE_MODEL_PATH,
            settings.INTERNAL_MODEL_PATH,
        ]
    )

    if models_exist:
        from app.services.vision import TelecomDetector
        from app.services.fiber_overview import FiberOverviewProcessor
        
        logger.info("Loading ML models into memory (this is done ONCE at startup)...")
        t0 = time.perf_counter()
        
        app.state.detector = TelecomDetector(
            main_model_path=settings.MAIN_MODEL_PATH,
            ps_model_path=settings.PS_MODEL_PATH,
            node_model_path=settings.NODE_MODEL_PATH,
            internal_model_path=settings.INTERNAL_MODEL_PATH,
            use_gpu=settings.USE_GPU,
            dpi=settings.PDF_DPI,
        )
        
        # Load Fiber Overview Processor
        app.state.fiber_processor = FiberOverviewProcessor(
            model_path=settings.FIBER_NODE_MODEL_PATH
        )
        
        load_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"✅ All models loaded in {load_ms:.0f} ms. Will be reused for every job.")
    else:
        logger.warning(
            "⚠️  One or more model weight files not found in model_weights/. "
            "Place your .pt files there and restart the server."
        )
        app.state.detector = None

    # OPT: Start background cleanup loop
    cleanup_task = asyncio.create_task(
        _cleanup_old_jobs(app, settings.JOB_RETENTION_HOURS)
    )
    logger.info(f"🧹 Auto-cleanup running (retention: {settings.JOB_RETENTION_HOURS}h).")

    yield  # ← Server is running here

    logger.info("🛑 Shutting down Telecom Vision API.")
    cleanup_task.cancel()
    app.state.pipeline_pool.shutdown(wait=False)
    app.state.detector = None
    if redis_client:
        await redis_client.aclose()
        logger.info("Redis connection closed.")


# ─────────────────────────────────────────────
#  App factory
# ─────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Production-ready REST API for automated telecom network map analysis. "
            "Accepts BEFORE/AFTER PDF map pairs and returns annotated vector PDF reports."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # S-7: Attach rate limiter state so @limiter.limit() works across the app
    from app.api.v1.jobs import limiter as jobs_limiter
    app.state.limiter = jobs_limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # OPT: Upload size guard (before any route processing)
    app.add_middleware(LimitUploadSizeMiddleware)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health.router, prefix=settings.API_PREFIX)
    app.include_router(jobs.router,   prefix=settings.API_PREFIX)

    # Static assets (Tiles, Reports)
    app.mount("/outputs", StaticFiles(directory=str(settings.OUTPUTS_DIR)), name="outputs")

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/docs")

    return app


app = create_app()

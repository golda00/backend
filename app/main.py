"""
FastAPI Application Entry Point (Stable Version)
"""
from __future__ import annotations
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.api.v1 import jobs, health

setup_logging()
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure storage exists
    settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Use RedisJobStore if available, otherwise fallback to PersistentJobStore.
    # This ensures the API can see progress updates written by Celery workers to Redis.
    from app.core.store import RedisJobStore, PersistentJobStore
    import redis.asyncio as aioredis
    
    # Try multiple common Redis URLs (localhost vs 127.0.0.1) to avoid IPv6 issues on Windows
    redis_urls = [settings.REDIS_URL, "redis://127.0.0.1:6379/0"]
    connected = False
    
    for url in redis_urls:
        try:
            logger.info(f"Attempting to connect to Redis at {url}...")
            redis_client = aioredis.from_url(url, encoding="utf-8", decode_responses=True)
            # Quick ping to check if Redis is actually up
            await asyncio.wait_for(redis_client.ping(), timeout=1.0)
            app.state.job_store = RedisJobStore(redis_client)
            app.state.job_store_type = "redis"
            logger.info(f"🚀 [SUCCESS] Connected to Redis at {url}")
            connected = True
            break
        except Exception as e:
            logger.warning(f"Failed to connect to Redis at {url}: {e}")
            
    if not connected:
        logger.warning("⚠️ All Redis connection attempts failed. Falling back to PersistentJobStore (local disk).")
        logger.warning("   Note: Progress updates from Celery workers will NOT be visible in the API if using local disk fallback.")
        app.state.job_store = PersistentJobStore(settings.OUTPUTS_DIR / "jobs.json")
        app.state.job_store_type = "disk"
    
    # NO MODEL LOADING IN API FOR STABILITY
    app.state.detector = None
    app.state.pipeline_pool = ThreadPoolExecutor(max_workers=settings.PIPELINE_WORKERS)
    
    yield
    app.state.pipeline_pool.shutdown()

def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(health.router, prefix=settings.API_PREFIX)
    app.include_router(jobs.router, prefix=settings.API_PREFIX)
    app.mount("/outputs", StaticFiles(directory=str(settings.OUTPUTS_DIR)), name="outputs")
    return app

app = create_app()

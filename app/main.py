"""
FastAPI Application Entry Point (Stable Version)
"""
from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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
    
    # Use PersistentJobStore — writes jobs.json reliably.
    # Celery workers read via Redis first, then fall back to jobs.json.
    from app.core.store import PersistentJobStore
    app.state.job_store = PersistentJobStore(settings.OUTPUTS_DIR / "jobs.json")
    
    # NO MODEL LOADING IN API FOR STABILITY
    app.state.detector = None
    app.state.pipeline_pool = ThreadPoolExecutor(max_workers=4)
    
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

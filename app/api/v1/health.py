"""
FastAPI Router — Health
/api/v1/health

Simple liveness + readiness check endpoint for load balancers and k8s probes.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.core.config import get_settings, Settings
from app.models.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse, summary="Service health check")
async def health_check(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    # Check if the detector was successfully loaded at startup
    detector_loaded = hasattr(request.app.state, "detector") and request.app.state.detector is not None
    job_store_type = getattr(request.app.state, "job_store_type", "unknown")
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        models_loaded=detector_loaded,
        job_store_type=job_store_type,
    )

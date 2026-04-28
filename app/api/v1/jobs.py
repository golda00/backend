"""
FastAPI Router — Jobs
/api/v1/jobs

Endpoints:
  POST   /jobs                  → Submit a new analysis job (upload 2 PDFs)
  GET    /jobs/{job_id}         → Poll job status + progress
  GET    /jobs/{job_id}/result  → Retrieve completed results + timing stats
  GET    /jobs/{job_id}/download → Download the annotated output PDF

Batch 3 changes:
  - _JOB_STORE dict replaced with Redis-backed RedisJobStore
    (with InMemoryJobStore fallback when Redis is unavailable)
  - asyncio.ensure_future / run_in_executor replaced with Celery task dispatch
    (run_pipeline_task.delay → Celery worker picks it up)
"""
from __future__ import annotations

import logging
import secrets
import shutil
import time
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import get_settings, Settings
from app.models.schemas import (
    JobCreatedResponse, JobResultResponse, JobStatus, JobStatusResponse, JobActionRequest
)

logger = logging.getLogger(__name__)

# S-7: Rate limiter keyed on caller IP
limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/jobs", tags=["Jobs"])


def _get_store(request: Request):
    """Returns the job store attached to app.state (Redis or in-memory fallback)."""
    return request.app.state.job_store


from app.workers.tasks import (
    run_pipeline_task, run_fiber_overview_task, run_fiber_before_task, 
    run_coax_before_task, run_fiber_after_task
)


# ─────────────────────────────────────────────
#  POST /jobs/coax-before
# ─────────────────────────────────────────────

@router.post(
    "/coax-before",
    response_model=JobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a new Coax BEFORE map job",
    description=(
        "Upload a single Coax BEFORE map (PDF). "
        "Stamps the Survey Info and Title Box using smart whitest-corner placement. "
        "Does not invoke any object detection or ML pipelines."
    ),
)
@limiter.limit("5/minute")
async def submit_coax_before_job(
    request: Request,
    before_pdf: Annotated[UploadFile, File(description="Coax BEFORE PDF map")],
    survey_image: Annotated[UploadFile | None, File(description="Optional Survey Info screenshot")] = None,
    prism_id: Annotated[str, Form(description="Prism ID")] = "",
    map_type: Annotated[str, Form(description="Map Type (e.g. BEFORE PRINT)")] = "BEFORE PRINT",
    node_name: Annotated[str, Form(description="Node Name")] = "",
    instance: Annotated[str, Form(description="Instance (e.g. 1 OF 1)")] = "",
    dpi: Annotated[int, Form(description="Rendering DPI")] = 300,
    settings: Settings = Depends(get_settings),
) -> JobCreatedResponse:
    _validate_pdf(before_pdf)

    job_id = str(uuid.uuid4())
    job_dir = settings.UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = job_dir / "before.pdf"
    with pdf_path.open("wb") as f:
        shutil.copyfileobj(before_pdf.file, f)

    job_token = secrets.token_urlsafe(32)
    store = _get_store(request)

    survey_img_path = None
    if survey_image and survey_image.filename:
        survey_img_path = job_dir / survey_image.filename
        with survey_img_path.open("wb") as f:
            shutil.copyfileobj(survey_image.file, f)

    job_record = {
        "status": JobStatus.QUEUED,
        "progress": 0.0,
        "message": "Job queued. Waiting for Celery worker.",
        "pdf_path": str(pdf_path),
        "output_dir": str(settings.OUTPUTS_DIR / job_id),
        "dpi": dpi,
        "title_box": {
            "prism_id": prism_id,
            "node_name": node_name,
            "instance": instance,
            "map_type": map_type,
        },
        "survey_image_path": str(survey_img_path) if survey_img_path else None,
        "token": job_token,
        "callouts": [],
        "report_path": None,
        "error": None,
        "created_at": time.time(),
        "stage_times": {},
        "pipeline_type": "coax_before",
    }
    await store.set(job_id, job_record)

    # Dispatch Celery task with local fallback for coax before
    try:
        run_coax_before_task.apply_async(args=[job_id], kwargs={"job_data": _serialize_for_celery(job_record)})
    except Exception as celery_err:
        import asyncio
        from app.services.coax_before import run_coax_before_pipeline
        loop = asyncio.get_running_loop()
        pipeline_pool = getattr(request.app.state, "pipeline_pool", None)
        _stub: dict[str, dict] = {job_id: dict(job_record)}
        
        async def _run_and_sync():
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(pipeline_pool, run_coax_before_pipeline, job_id, _stub, settings),
                    timeout=settings.JOB_TIMEOUT_SECONDS,
                )
                await store.set(job_id, _stub[job_id])
            except asyncio.TimeoutError:
                await store.update(job_id, {"status": JobStatus.FAILED, "message": "Job timed out.", "error": "TimeoutError"})
        asyncio.ensure_future(_run_and_sync())


    return JobCreatedResponse(job_id=job_id, job_token=job_token)

@router.post(
    "/fiber-overview",
    response_model=JobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a new fiber overview map analysis job",
    description=(
        "Upload a single fiber overview map (PDF). "
        "Supply metadata like **prism_id** and **node_name**.\n\n"
        "**Business Logic:**\n"
        "- If **is_connected** is True: Provide **hub_name** and **port_name**.\n"
        "- If **is_connected** is False: Provide **splice_can_name**."
    ),
)
@limiter.limit("5/minute")
async def submit_fiber_overview_job(
    request: Request,
    file: Annotated[UploadFile, File(description="Fiber Overview PDF map")],
    survey_image: Annotated[UploadFile | None, File(description="Optional Survey Info (Top Right)")] = None,
    prism_id: Annotated[str, Form(description="Prism ID")] = "",
    node_name: Annotated[str, Form(description="Node Name (for the detection callout)")] = "",
    instance: Annotated[str, Form(description="Instance (e.g. 1 OF 1)")] = "",
    is_connected: Annotated[bool, Form(description="Whether the node is connected to a hub")] = True,
    hub_name: Annotated[str, Form(description="Hub Name (if connected)")] = "",
    port_name: Annotated[str, Form(description="Port/Panel Name (if connected)")] = "",
    splice_can_name: Annotated[str, Form(description="Splice Can Name (if not connected)")] = "",
    dpi: Annotated[int, Form(description="Rendering DPI")] = 300,
    settings: Settings = Depends(get_settings),
) -> JobCreatedResponse:
    _validate_pdf(file)
    
    job_id = str(uuid.uuid4())
    job_dir = settings.UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = job_dir / "overview.pdf"
    with pdf_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
        
    job_token = secrets.token_urlsafe(32)
    store = _get_store(request)
    
    survey_img_path = None
    if survey_image:
        survey_img_path = job_dir / survey_image.filename
        with survey_img_path.open("wb") as f:
            shutil.copyfileobj(survey_image.file, f)
            
    job_record = {
        "status": JobStatus.QUEUED,
        "progress": 0.0,
        "message": "Job queued. Waiting for Celery worker.",
        "pdf_path": str(pdf_path),
        "output_dir": str(settings.OUTPUTS_DIR / job_id),
        "dpi": dpi,
        "title_box": {
            "prism_id": prism_id,
            "node_name": node_name,
            "instance": instance,
        },
        "survey_image_path": str(survey_img_path) if survey_img_path else None,
        "is_connected": is_connected,
        "hub_name": hub_name,
        "port_name": port_name,
        "splice_can_name": splice_can_name,
        "token": job_token,
        "callouts": [],
        "report_path": None,
        "error": None,
        "created_at": time.time(),
        "stage_times": {},
        "pipeline_type": "fiber_overview",
    }
    # Fix: store.set_job -> store.set
    await store.set(job_id, job_record)
    # Dispatch Celery task with local thread pool fallback
    try:
        run_fiber_overview_task.apply_async(args=[job_id], kwargs={"job_data": _serialize_for_celery(job_record)})
        logger.info(f"[{job_id}] Dispatched Fiber Overview to Celery worker.")
    except Exception as celery_err:
        logger.warning(f"[{job_id}] Celery unavailable ({celery_err}). Falling back to local thread pool executor.")
        import asyncio
        from app.workers.pipeline import run_fiber_overview_pipeline
        
        loop = asyncio.get_running_loop()
        pipeline_pool = getattr(request.app.state, "pipeline_pool", None)
        from app.core.store import InMemoryJobStore
        _stub: dict[str, dict] = {job_id: dict(job_record)}
        
        async def _run_and_sync():
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(
                        pipeline_pool,
                        run_fiber_overview_pipeline,
                        job_id,
                        _stub,
                        settings,
                        None
                    ),
                    timeout=settings.JOB_TIMEOUT_SECONDS,
                )
                await store.set(job_id, _stub[job_id])
            except asyncio.TimeoutError:
                await store.update(job_id, {"status": JobStatus.FAILED, "message": "Job timed out.", "error": "TimeoutError"})
        asyncio.ensure_future(_run_and_sync())
    
    return JobCreatedResponse(job_id=job_id, job_token=job_token)


# ─────────────────────────────────────────────
#  POST /jobs/fiber-overview-before
# ─────────────────────────────────────────────

@router.post(
    "/fiber-overview-before",
    response_model=JobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a new fiber overview BEFORE map job",
    description=(
        "Upload a single fiber overview BEFORE map (PDF). "
        "Stamps the Survey Info and Title Box natively into the top right corner. "
        "Does not invoke object detection or ML pipelines."
    ),
)
@limiter.limit("5/minute")
async def submit_fiber_overview_before_job(
    request: Request,
    before_pdf: Annotated[UploadFile, File(description="Fiber Overview BEFORE PDF map")],
    survey_image: Annotated[UploadFile | None, File(description="Optional Survey Info (Top Right)")] = None,
    prism_id: Annotated[str, Form(description="Prism ID")] = "",
    map_type: Annotated[str, Form(description="Map Type (e.g. BEFORE)")] = "BEFORE",
    node_name: Annotated[str, Form(description="Node Name")] = "",
    instance: Annotated[str, Form(description="Instance (e.g. 1 OF 1)")] = "",
    dpi: Annotated[int, Form(description="Rendering DPI")] = 300,
    settings: Settings = Depends(get_settings),
) -> JobCreatedResponse:
    _validate_pdf(before_pdf)
    
    job_id = str(uuid.uuid4())
    job_dir = settings.UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = job_dir / "before.pdf"
    with pdf_path.open("wb") as f:
        shutil.copyfileobj(before_pdf.file, f)
        
    job_token = secrets.token_urlsafe(32)
    store = _get_store(request)
    
    survey_img_path = None
    if survey_image:
        survey_img_path = job_dir / survey_image.filename
        with survey_img_path.open("wb") as f:
            shutil.copyfileobj(survey_image.file, f)
            
    job_record = {
        "status": JobStatus.QUEUED,
        "progress": 0.0,
        "message": "Job queued. Waiting for Celery worker.",
        "pdf_path": str(pdf_path),
        "output_dir": str(settings.OUTPUTS_DIR / job_id),
        "dpi": dpi,
        "title_box": {
            "prism_id": prism_id,
            "node_name": node_name,
            "instance": instance,
            "map_type": map_type,
        },
        "survey_image_path": str(survey_img_path) if survey_img_path else None,
        "token": job_token,
        "callouts": [],
        "report_path": None,
        "error": None,
        "created_at": time.time(),
        "stage_times": {},
        "pipeline_type": "fiber_before",
    }
    await store.set(job_id, job_record)
    
    # Dispatch Celery task with local thread pool fallback
    try:
        run_fiber_before_task.apply_async(args=[job_id], kwargs={"job_data": _serialize_for_celery(job_record)})
        logger.info(f"[{job_id}] Dispatched Fiber Before to Celery worker.")
    except Exception as celery_err:
        logger.warning(f"[{job_id}] Celery unavailable ({celery_err}). Falling back to local threads.")
        import asyncio
        from app.services.fiber_before import run_fiber_before_pipeline
        
        loop = asyncio.get_running_loop()
        pipeline_pool = getattr(request.app.state, "pipeline_pool", None)
        _stub: dict[str, dict] = {job_id: dict(job_record)}
        
        async def _run_and_sync():
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(pipeline_pool, run_fiber_before_pipeline, job_id, _stub, settings),
                    timeout=settings.JOB_TIMEOUT_SECONDS,
                )
                await store.set(job_id, _stub[job_id])
            except asyncio.TimeoutError:
                await store.update(job_id, {"status": JobStatus.FAILED, "message": "Job timed out.", "error": "TimeoutError"})
        asyncio.ensure_future(_run_and_sync())
    
    return JobCreatedResponse(job_id=job_id, job_token=job_token)


# ─────────────────────────────────────────────
#  POST /jobs/fiber-after
# ─────────────────────────────────────────────

@router.post(
    "/fiber-after",
    response_model=JobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a new fiber AFTER map analysis job (ML)",
    description=(
        "Upload a single fiber AFTER map (PDF). "
        "Supply metadata and selection DPI (50, 70, or 90). "
        "Uses YOLOv8 for symbol detection and Rule Engine for automated callouts."
    ),
)
@limiter.limit("5/minute")
async def submit_fiber_after_job(
    request: Request,
    file: Annotated[UploadFile, File(description="Fiber AFTER PDF map")],
    survey_image: Annotated[UploadFile | None, File(description="Optional Survey Info screenshot")] = None,
    prism_id: Annotated[str, Form(description="Prism ID")] = "",
    node_name: Annotated[str, Form(description="Node Name")] = "",
    instance: Annotated[str, Form(description="Instance (e.g. 1 OF 1)")] = "",
    hub: Annotated[str, Form(description="Hub Name")] = "",
    port_panel: Annotated[str, Form(description="Port/Panel Detail")] = "",
    dpi: Annotated[int, Form(description="Rendering DPI (50, 70, or 90)")] = 50,
    include_mux: Annotated[bool, Form(description="Whether to include MUX LOCATION callout in output PDF")] = True,
    settings: Settings = Depends(get_settings),
) -> JobCreatedResponse:
    _validate_pdf(file)
    
    job_id = str(uuid.uuid4())
    job_dir = settings.UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = job_dir / "after.pdf"
    with pdf_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
        
    job_token = secrets.token_urlsafe(32)
    store = _get_store(request)
    
    survey_img_path = None
    if survey_image and survey_image.filename:
        survey_img_path = job_dir / survey_image.filename
        with survey_img_path.open("wb") as f:
            shutil.copyfileobj(survey_image.file, f)
            
    job_record = {
        "status": JobStatus.QUEUED,
        "progress": 0.0,
        "message": "Job queued. Waiting for Celery worker.",
        "pdf_path": str(pdf_path),
        "output_dir": str(settings.OUTPUTS_DIR / job_id),
        "dpi": dpi,
        "title_box": {
            "prism_id": prism_id,
            "node_name": node_name,
            "instance": instance,
            "hub": hub,
            "port_panel": port_panel,
        },
        "survey_image_path": str(survey_img_path) if survey_img_path else None,
        "token": job_token,
        "callouts": [],
        "report_path": None,
        "error": None,
        "created_at": time.time(),
        "stage_times": {},
        "pipeline_type": "fiber_after",
        "include_mux": include_mux,
    }
    await store.set(job_id, job_record)
    
    try:
        run_fiber_after_task.apply_async(args=[job_id], kwargs={"job_data": _serialize_for_celery(job_record)})
        logger.info(f"[{job_id}] Dispatched Fiber After to Celery worker.")
    except Exception as celery_err:
        logger.warning(f"[{job_id}] Celery unavailable. Falling back to local thread pool.")
        import asyncio
        from app.services.fiber_after import run_fiber_after_pipeline
        loop = asyncio.get_running_loop()
        pipeline_pool = getattr(request.app.state, "pipeline_pool", None)
        _stub = {job_id: dict(job_record)}
        
        async def _run_async():
             try:
                 await loop.run_in_executor(pipeline_pool, run_fiber_after_pipeline, job_id, _stub, settings)
                 await store.set(job_id, _stub[job_id])
             except Exception as e:
                 await store.update(job_id, {"status": JobStatus.FAILED, "error": str(e)})
        asyncio.ensure_future(_run_async())

    return JobCreatedResponse(job_id=job_id, job_token=job_token)



# ─────────────────────────────────────────────
#  POST /jobs
# ─────────────────────────────────────────────

@router.post(
    "",
    response_model=JobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a new map analysis job",
    description=(
        "Upload a BEFORE and AFTER telecom network PDF map. "
        "Supply an optional **dpi** value (72–1200, default 300) to control "
        "rendering resolution. Higher DPI = more detail but much more RAM and time. "
        "Recommended: 300 for speed, 600 for balanced quality, 800 for max detail.\n\n"
        "ℹ️ The response includes a **job_token** — store it securely. "
        "You must send it as the `X-Job-Token` header on all subsequent requests."
    ),
)
@limiter.limit("10/minute")   # S-7: at most 10 new jobs per IP per minute
async def submit_job(
    request:    Request,
    before_pdf: Annotated[UploadFile, File(description="PDF map BEFORE changes")],
    after_pdf:  Annotated[UploadFile, File(description="PDF map AFTER changes")],
    dpi: Annotated[int, Form(description="Rendering DPI (72–1200). Higher = more RAM & time. Default: 300")] = 300,
    survey_image: Annotated[UploadFile | None, File(description="Optional Survey Info screenshot")] = None,
    prism_id: Annotated[str, Form(description="Prism ID (e.g. 4147677_4147697)")] = "",
    map_type: Annotated[str, Form(description="Map Type (e.g. AFTER)")] = "",
    node_name: Annotated[str, Form(description="Node Name (e.g. OX003A_OX003B)")] = "",
    instance: Annotated[str, Form(description="Instance (e.g. PG 1 OF 4)")] = "",
    before_node_type: Annotated[str, Form(description="Node type before replacement (3x3, 4x4)")] = "",
    before_node_names: Annotated[str, Form(description="Comma-separated node names before replacement")] = "",
    after_node_type: Annotated[str, Form(description="Node type after replacement (2x2)")] = "",
    after_node_names: Annotated[str, Form(description="Comma-separated node names after replacement")] = "",
    settings:   Settings = Depends(get_settings),
) -> JobCreatedResponse:
    _validate_pdf(before_pdf)
    _validate_pdf(after_pdf)
    if not (72 <= dpi <= 1200):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"dpi must be between 72 and 1200, got {dpi}.",
        )

    job_id = str(uuid.uuid4())
    job_dir = settings.UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    before_path = job_dir / "before.pdf"
    after_path  = job_dir / "after.pdf"

    with before_path.open("wb") as f:
        shutil.copyfileobj(before_pdf.file, f)

    with after_path.open("wb") as f:
        shutil.copyfileobj(after_pdf.file, f)

    survey_image_path = None
    if survey_image and survey_image.filename:
        survey_image_path = str(job_dir / "survey_info.png")
        with open(survey_image_path, "wb") as f:
            shutil.copyfileobj(survey_image.file, f)

    # S-8: Generate a cryptographically random secret token for this job.
    job_token = secrets.token_urlsafe(32)

    # ── Persist job record to Redis (Batch 3) ────────────────────────
    store = _get_store(request)
    job_record = {
        "status":       JobStatus.QUEUED,
        "progress":     0.0,
        "message":      "Job queued. Waiting for Celery worker.",
        "before_path":  str(before_path),
        "after_path":   str(after_path),
        "output_dir":   str(settings.OUTPUTS_DIR / job_id),
        "dpi":          dpi,
        "survey_image_path": survey_image_path,
        "title_box": {
            "prism_id": prism_id,
            "map_type": map_type,
            "node_name": node_name,
            "instance": instance,
        },
        "before_node_type":  before_node_type or None,
        "before_node_names": [n.strip() for n in before_node_names.split(",") if n.strip()] if before_node_names else None,
        "after_node_type":   after_node_type or None,
        "after_node_names":  [n.strip() for n in after_node_names.split(",") if n.strip()] if after_node_names else None,
        "token":        job_token,
        "callouts":     [],
        "report_path":  None,
        "error":        None,
        "created_at":   time.time(),
        "stage_times":  {},
        "pipeline_type": "coax",
    }
    await store.set(job_id, job_record)

    logger.info(f"[{job_id}] Job persisted to Redis at {dpi} DPI.")

    # ── Dispatch to Celery worker (Batch 3) ──────────────────────────
    # If Redis / Celery is unavailable, fall back to thread pool executor
    try:
        from app.workers.tasks import run_pipeline_task
        run_pipeline_task.apply_async(args=[job_id], kwargs={"job_data": _serialize_for_celery(job_record)})
        logger.info(f"[{job_id}] Dispatched to Celery worker.")
    except Exception as celery_err:
        logger.warning(
            f"[{job_id}] Celery unavailable ({celery_err}). "
            "Falling back to local thread pool executor."
        )
        import asyncio
        from app.workers.pipeline import run_pipeline_sync

        detector = getattr(request.app.state, "detector", None)
        loop = asyncio.get_running_loop()
        pipeline_pool = getattr(request.app.state, "pipeline_pool", None)

        # Build an in-memory stub so run_pipeline_sync can update job state
        # and mirror those updates back to Redis
        from app.core.store import InMemoryJobStore
        _stub: dict[str, dict] = {job_id: dict(job_record)}

        async def _run_and_sync():
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(
                        pipeline_pool,
                        run_pipeline_sync,
                        job_id,
                        _stub,
                        settings,
                        detector,
                    ),
                    timeout=settings.JOB_TIMEOUT_SECONDS,
                )
                # Mirror final state to Redis
                await store.set(job_id, _stub[job_id])
            except asyncio.TimeoutError:
                await store.update(job_id, {
                    "status":  JobStatus.FAILED,
                    "message": f"Job timed out after {settings.JOB_TIMEOUT_SECONDS}s.",
                    "error":   "TimeoutError",
                })

        asyncio.ensure_future(_run_and_sync())

    return JobCreatedResponse(job_id=job_id, job_token=job_token)


# ─────────────────────────────────────────────
#  GET /jobs/{job_id}
# ─────────────────────────────────────────────

@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll job status",
)
async def get_job_status(
    job_id: str,
    request: Request,
    x_job_token: Annotated[str | None, Header(description="Job secret token (X-Job-Token)")] = None,
) -> JobStatusResponse:
    job = await _get_job_or_404(request, job_id)
    _verify_token(job, x_job_token)
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress_pct=job["progress"],
        message=job["message"],
        error=job["error"],
        sample_tiles=job.get("sample_tiles"),
        flagged_tiles=job.get("flagged_tiles"),
    )


# ─────────────────────────────────────────────
#  GET /jobs/{job_id}/result
# ─────────────────────────────────────────────

@router.get(
    "/{job_id}/result",
    response_model=JobResultResponse,
    summary="Retrieve job results",
)
async def get_job_result(
    job_id: str,
    request: Request,
    x_job_token: Annotated[str | None, Header(description="Job secret token (X-Job-Token)")] = None,
) -> JobResultResponse:
    job = await _get_job_or_404(request, job_id)
    _verify_token(job, x_job_token)

    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is not yet completed. Current status: {job['status']}",
        )

    return JobResultResponse(
        job_id=job_id,
        status=job["status"],
        callouts=job["callouts"],
        report_url=f"/api/v1/jobs/{job_id}/download",
        stats=job.get("stats"),
    )


# ─────────────────────────────────────────────
#  GET /jobs/{job_id}/download
# ─────────────────────────────────────────────

@router.post("/{job_id}/action", summary="Perform an interactive action on a paused job")
async def perform_job_action(
    job_id: str,
    body: JobActionRequest,
    request: Request,
    x_job_token: Annotated[str | None, Header(description="Job secret token (X-Job-Token)")] = None,
    settings: Settings = Depends(get_settings),
):
    job = await _get_job_or_404(request, job_id)
    _verify_token(job, x_job_token)
    store = _get_store(request)

    if body.action.upper() == "ABORT":
        # Cleanup tiles as requested but keep the PDFs for potential retry
        out_dir = Path(job["output_dir"])
        shutil.rmtree(out_dir, ignore_errors=True)
        await store.update(job_id, {
            "status": JobStatus.FAILED,
            "message": "Job aborted by user.",
            "progress": 0.0
        })
        return {"message": "Job aborted and workspace cleaned."}

    if body.action.upper() == "PROCEED":
        # Move to next logical status
        current_status = job["status"]
        next_status = None
        
        if current_status == JobStatus.AWAITING_DPI_CONFIRM:
            next_status = JobStatus.PROCESSING
        elif current_status == JobStatus.AWAITING_REVIEW:
            next_status = JobStatus.REPORTING
            # Apply callout overrides if any
            if body.overrides:
                await store.update(job_id, {"all_callouts_visible": body.overrides})
        
        if not next_status:
            raise HTTPException(status_code=400, detail=f"Cannot PROCEED from current status: {current_status}")

        await store.update(job_id, {"status": next_status, "message": "Resuming pipeline..."})
        
        # Trigger pipeline again based on pipeline_type
        p_type = job.get("pipeline_type", "coax")
        
        try:
            if p_type == "fiber_after":
                run_fiber_after_task.apply_async(args=[job_id], kwargs={"job_data": _serialize_for_celery(await store.get(job_id))})
            else:
                run_pipeline_task.apply_async(args=[job_id], kwargs={"job_data": _serialize_for_celery(await store.get(job_id))})
        except Exception:
            # Local fallback (mirroring submit logic)
            import asyncio
            from app.workers.pipeline import run_pipeline_sync
            from app.services.fiber_after import run_fiber_after_pipeline
            loop = asyncio.get_running_loop()
            pipeline_pool = request.app.state.pipeline_pool
            
            async def _resume_async():
                _stub = {job_id: await store.get(job_id)}
                if p_type == "fiber_after":
                    await loop.run_in_executor(pipeline_pool, run_fiber_after_pipeline, job_id, _stub, settings)
                else:
                    await loop.run_in_executor(pipeline_pool, run_pipeline_sync, job_id, _stub, settings, request.app.state.detector)
                await store.set(job_id, _stub[job_id])
            
            asyncio.ensure_future(_resume_async())

        return {"message": f"Job resuming with status {next_status}"}

    raise HTTPException(status_code=400, detail=f"Unknown action: {body.action}")


@router.get("/{job_id}/download", summary="Download the annotated output PDF")
async def download_result(
    job_id: str,
    request: Request,
    x_job_token: Annotated[str | None, Header(description="Job secret token (X-Job-Token)")] = None,
    settings: Settings = Depends(get_settings),
):
    job = await _get_job_or_404(request, job_id)
    _verify_token(job, x_job_token)

    if job["status"] != JobStatus.COMPLETED or not job["report_path"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Report not yet available.",
        )

    report_path = settings.BASE_DIR / job["report_path"]
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report file not found on disk.")

    return FileResponse(
        path=str(report_path),
        filename=f"telecom_report_{job_id[:8]}.pdf",
        media_type="application/pdf",
    )


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _write_job_sidecar(job_dir: Path, job_record: dict) -> None:
    """
    Write a job_record.json next to the uploaded PDFs.
    This acts as a filesystem fallback so Celery workers can always find job
    data even when Redis has an async write race condition or is unreachable.
    """
    def _default(obj):
        try:
            from app.models.schemas import JobStatus
            if isinstance(obj, JobStatus):
                return obj.value
        except Exception:
            pass
        return str(obj)
    try:
        sidecar_path = Path(job_dir) / "job_record.json"
        sidecar_path.write_text(_json.dumps(job_record, default=_default), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Could not write job sidecar file: {e}")


def _serialize_for_celery(job_record: dict) -> dict:
    """
    Convert a job_record dict to a fully JSON-serialisable form
    so it can travel with the Celery message. This eliminates the
    need for the worker to look up the job from Redis or any store.
    """
    def _default(obj):
        from app.models.schemas import JobStatus
        if isinstance(obj, JobStatus):
            return obj.value
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)
    return _json.loads(_json.dumps(job_record, default=_default))


async def _get_job_or_404(request: Request, job_id: str) -> dict:
    store = _get_store(request)
    job = await store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


def _verify_token(job: dict, provided: str | None) -> None:
    """S-8: Constant-time token comparison to prevent timing attacks."""
    expected = job.get("token", "")
    if not secrets.compare_digest(provided or "", expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing X-Job-Token header.",
        )


def _validate_pdf(upload: UploadFile) -> None:
    """Two-layer validation: file extension + PDF magic bytes (%PDF-)."""
    if not upload.filename or not upload.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"File '{upload.filename}' does not have a .pdf extension.",
        )
    header = upload.file.read(5)
    upload.file.seek(0)
    if header != b"%PDF-":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"File '{upload.filename}' is not a valid PDF (magic byte check failed).",
        )

"""
Pydantic schemas for all API request/response contracts.
These are the single source of truth for data shapes across the API.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
#  Enumerations
# ─────────────────────────────────────────────

class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    ALIGNING = "ALIGNING"
    TILING = "TILING"
    AWAITING_DPI_CONFIRM = "AWAITING_DPI_CONFIRM"
    PROCESSING = "PROCESSING"
    AWAITING_REVIEW = "AWAITING_REVIEW"
    MATCHING = "MATCHING"
    REPORTING = "REPORTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ─────────────────────────────────────────────
#  Shared / Nested Models
# ─────────────────────────────────────────────

class DetectedObject(BaseModel):
    bbox: list[int] = Field(..., description="[x1, y1, x2, y2] in tile-local pixels")
    cls: str = Field(..., description="Class name, e.g. 'tap', '2way_splitter'")
    conf: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    text: str = Field(default="", description="OCR-extracted text for this object")
    model: str = Field(default="", description="Name of the YOLO model that made this detection")


class Callout(BaseModel):
    gx: Optional[float] = Field(None, description="Global X coordinate")
    gy: Optional[float] = Field(None, description="Global Y coordinate")
    loc: Optional[tuple[int, int]] = Field(None, description="(x, y) position in tile-local pixels")
    text: str = Field(..., description="Short callout label")
    desc: str = Field(default="", description="Human-readable description")
    model: str = Field(default="", description="Model that triggered the callout")
    type: str = Field(default="NORMAL", description="FLAGGED or NORMAL")


# ─────────────────────────────────────────────
#  Job Lifecycle
# ─────────────────────────────────────────────

class JobCreatedResponse(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the submitted analysis job")
    job_token: str = Field(..., description="Secret token — include as X-Job-Token header in all subsequent requests")
    status: JobStatus = JobStatus.QUEUED
    message: str = "Job submitted successfully. Poll /jobs/{job_id}/status for updates."

class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress_pct: Optional[float] = Field(None, ge=0, le=100)
    message: Optional[str] = None
    error: Optional[str] = None
    sample_tiles: Optional[list[int]] = None
    flagged_tiles: Optional[list[int]] = None


class JobResultResponse(BaseModel):
    job_id: str
    status: JobStatus
    callouts: list[Callout] = []
    report_url: Optional[str] = Field(None, description="Pre-signed URL to the output PDF report")
    stats: Optional[dict[str, Any]] = Field(None, description="Summary statistics for the job")


# ─────────────────────────────────────────────
#  Health Check
# ─────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    models_loaded: bool
    job_store_type: str = "unknown"


class JobActionRequest(BaseModel):
    action: str = Field(..., description="Action to perform: 'PROCEED' or 'ABORT'")
    overrides: Optional[list[dict]] = Field(None, description="Optional callout modifications (rename/remove)")

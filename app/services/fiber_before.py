"""
Stateless Service for generating the 'Before' map overview.
Adds Top-Right Survey Info block without invoking ML models.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import fitz

from app.core.config import Settings
from app.models.schemas import JobStatus
from app.services.alignment import pdf_to_image
from app.services.reporting import _draw_legend_stack

logger = logging.getLogger(__name__)


def run_fiber_before_pipeline(
    job_id: str,
    job_store: dict,
    settings: Settings,
) -> None:
    """
    Stateless processing for the Before map. Only converts the PDF and stamps
    the Survey Image & Title Box block using the predefined robust styling.
    """
    job_start = time.perf_counter()

    def _update(status: JobStatus, pct: float, msg: str) -> None:
        job_store[job_id].update({"status": status, "progress": pct, "message": msg})
        logger.info(f"[{job_id}] [{pct:3.0f}%] {msg}")

    def _record(stage: str, t0: float) -> float:
        elapsed = (time.perf_counter() - t0) * 1000
        job_store[job_id]["stage_times"][stage] = round(elapsed, 1)
        return time.perf_counter()

    try:
        job = job_store[job_id]
        pdf_path = Path(job["pdf_path"])
        output_dir = Path(job["output_dir"])
        dpi = job.get("dpi", settings.PDF_DPI)
        survey_image_path = job.get("survey_image_path")
        title_box_data = job.get("title_box", {})

        output_dir.mkdir(parents=True, exist_ok=True)
        job_store[job_id]["stage_times"] = {}
        t0 = time.perf_counter()

        _update(JobStatus.PROCESSING, 10, "Extracting PDF raster bounds...")
        img = pdf_to_image(pdf_path, dpi=dpi)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t0 = _record("CONVERSION", t0)

        _update(JobStatus.REPORTING, 50, "Stamping native Survey and Title Box overlays...")
        report_path = output_dir / "report.pdf"
        
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        
        # Draw Survey Block directly onto the raw map
        _draw_legend_stack(
            page=page,
            img_gray=img_gray,
            callouts=[],
            survey_image_path=survey_image_path,
            title_box_data=title_box_data,
            dpi=dpi,
            include_legend=False,
            title_font_size=34,
        )
        
        doc.save(str(report_path), deflate=True, garbage=4, clean=True, linear=False)
        doc.close()
        t0 = _record("REPORTING", t0)

        total_ms = (time.perf_counter() - job_start) * 1000
        job_store[job_id]["stage_times"]["total_ms"] = round(total_ms, 1)

        job_store[job_id].update({
            "status": JobStatus.COMPLETED,
            "progress": 100.0,
            "message": "Fiber Overview Before pipeline completed.",
            "report_path": str(report_path.relative_to(settings.BASE_DIR)),
        })
        logger.info(f"[{job_id}] ✅ Pipeline complete in {total_ms:.0f} ms.")

    except Exception as exc:
        logger.exception(f"[{job_id}] ❌ Pipeline failed: {exc}")
        job_store[job_id].update({
            "status": JobStatus.FAILED,
            "message": "Fiber Overview Before pipeline encountered an error.",
            "error": str(exc),
        })

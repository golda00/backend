"""
Stateless Service for generating the 'Coax Before' map overview.
Ports the overlay_tool.py logic into the API worker architecture.
Finds the whitest map corner automatically, stamps Survey Image + Title Box.
No callout or ML logic involved.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import fitz
import numpy as np

from app.core.config import Settings
from app.models.schemas import JobStatus

logger = logging.getLogger(__name__)


def _get_best_corner(
    page: fitz.Page,
    ss_width_pts: float,
    full_overlay_h: float,
    margin: float = 30.0,
) -> tuple[float, float, bool]:
    """
    Analyzes the first page raster to find the corner with the highest
    average brightness (whitest area). Returns (x, y, needs_extension).
    """
    pix = page.get_pixmap(dpi=150)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

    if pix.n == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    h_img, w_img = gray.shape
    pw, ph = page.rect.width, page.rect.height

    corners = {
        "TL": (margin, margin),
        "TR": (pw - ss_width_pts - margin, margin),
        "BL": (margin, ph - full_overlay_h - margin),
        "BR": (pw - ss_width_pts - margin, ph - full_overlay_h - margin),
    }

    THRESHOLD = 252  # very strict white check
    brightness_vals: dict[str, float] = {}
    clear_corners: list[str] = []
    scale = 150.0 / 72.0

    for name, (cx, cy) in corners.items():
        x1 = max(0, int(cx * scale))
        y1 = max(0, int(cy * scale))
        x2 = min(w_img, int((cx + ss_width_pts) * scale))
        y2 = min(h_img, int((cy + full_overlay_h) * scale))
        roi = gray[y1:y2, x1:x2]
        if roi.size > 0:
            avg_b = float(np.mean(roi))
            min_b = float(np.min(roi))
            brightness_vals[name] = avg_b
            if min_b >= THRESHOLD:
                clear_corners.append(name)

    logger.debug(f"Corner brightness: {brightness_vals}")
    logger.debug(f"Clear corners (min >= {THRESHOLD}): {clear_corners}")

    if not clear_corners:
        logger.info("No clear corners found — triggering map extension mode.")
        return (margin, -full_overlay_h - margin, True)

    best = max(clear_corners, key=lambda n: brightness_vals[n])
    cx, cy = corners[best]
    return (cx, cy, False)


def _overlay_on_pdf(
    pdf_path: Path,
    screenshot_path: Path | None,
    prism_id: str,
    node_name: str,
    instance: str,
    map_type: str,
    output_path: Path,
) -> None:
    """
    Core overlay function. Mirrors overlay_tool.py business logic.
    """
    doc = fitz.open(str(pdf_path))
    page_orig = doc[0]
    total_pages = doc.page_count

    # ── Image dimensions ──────────────────────────────────────────
    ss_w_pts, ss_h_pts = 0.0, 0.0
    if screenshot_path and screenshot_path.exists():
        temp_doc = fitz.open(str(screenshot_path))
        ir = temp_doc[0].rect
        ss_w_pts = 700.0
        ss_h_pts = (ir.height / ir.width) * ss_w_pts
        temp_doc.close()

    # ── Text box ─────────────────────────────────────────────────
    m_type = map_type.upper() if map_type else "BEFORE"
    print_name = m_type
    if "AFTER" in m_type:
        print_name = "AFTER"
    elif "BEFORE" in m_type:
        print_name = "BEFORE"
    elif "SCHEMATIC" in m_type:
        print_name = "SCHEMATIC REPORT"
    elif "OVERVIEW" in m_type:
        print_name = "OVERVIEW"

    text_lines_base = []
    if prism_id:
        text_lines_base.append(prism_id)
    if node_name:
        text_lines_base.append(node_name)
    if print_name:
        text_lines_base.append(print_name)
    if instance:
        text_lines_base.append(instance)

    font_size = 24
    line_height = font_size + 8
    padding_w, padding_h = 30, 25

    test_lines = text_lines_base.copy()
    test_lines.append(f"PG {total_pages} OF {total_pages}")

    max_line_w = max(
        (fitz.get_text_length(line, fontname="helv", fontsize=font_size) for line in test_lines),
        default=100.0,
    )
    text_box_w = max_line_w + padding_w
    text_box_h = line_height * (len(text_lines_base) + 1) + padding_h

    gap = 10.0
    full_overlay_h = ss_h_pts + text_box_h + gap if ss_h_pts > 0 else text_box_h
    effective_w = max(ss_w_pts, text_box_w)

    for i in range(total_pages):
        page_orig = doc[i]
        start_x, start_y, needs_ext = _get_best_corner(
            page_orig, effective_w, full_overlay_h
        )

        if needs_ext:
            logger.info(f"Extending map page {i+1} upward to create overlay space.")
            ext_h = full_overlay_h + 100
            old_rect = page_orig.rect
            new_height = old_rect.height + ext_h

            new_doc = fitz.open()
            new_page = new_doc.new_page(width=old_rect.width, height=new_height)
            new_page.show_pdf_page(
                fitz.Rect(0, ext_h, old_rect.width, new_height), doc, i
            )
            doc.delete_page(i)
            doc.insert_pdf(new_doc, from_page=0, to_page=0, start_at=i)
            new_doc.close()

            page = doc[i]
            start_x, start_y = 30.0, 50.0
            is_bottom = False
        else:
            page = page_orig
            is_bottom = start_y > (page.rect.height / 2)

        if is_bottom:
            text_rect = fitz.Rect(start_x, start_y, start_x + text_box_w, start_y + text_box_h)
            ss_rect = fitz.Rect(
                start_x, start_y + text_box_h + gap,
                start_x + ss_w_pts, start_y + text_box_h + gap + ss_h_pts
            )
        else:
            ss_rect = fitz.Rect(start_x, start_y, start_x + ss_w_pts, start_y + ss_h_pts)
            text_rect = fitz.Rect(
                start_x, start_y + ss_h_pts + gap,
                start_x + text_box_w, start_y + ss_h_pts + gap + text_box_h
            )

        if screenshot_path and screenshot_path.exists() and ss_h_pts > 0:
            page.insert_image(ss_rect, filename=str(screenshot_path))

        current_lines = []
        if prism_id:
            current_lines.append(prism_id)
        if node_name:
            current_lines.append(node_name)
        if print_name:
            current_lines.append(print_name)
        current_lines.append(f"PG {i + 1} OF {total_pages}")
        if instance:
            current_lines.append(instance)

        text_content = "\n".join(current_lines)

        page.draw_rect(text_rect, color=(1, 0, 0), fill=(1, 1, 0), width=1.5)
        page.insert_textbox(
            text_rect,
            text_content,
            fontsize=font_size,
            fontname="helv",
            color=(0, 0, 0),
            align=0,
        )

    doc.save(str(output_path), deflate=True, garbage=4, clean=True, linear=False)
    doc.close()


def run_coax_before_pipeline(
    job_id: str,
    job_store: dict,
    settings: Settings,
) -> None:
    """
    Stateless processing for the Coax Before map.
    Stamps Survey Image + Title Box using smart whitest-corner placement.
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
        survey_image_path = job.get("survey_image_path")
        title_box = job.get("title_box", {})
        prism_id = title_box.get("prism_id", "")
        node_name = title_box.get("node_name", "")
        instance = title_box.get("instance", "")
        map_type = title_box.get("map_type", "BEFORE PRINT")

        output_dir.mkdir(parents=True, exist_ok=True)
        job_store[job_id]["stage_times"] = {}
        t0 = time.perf_counter()

        _update(JobStatus.PROCESSING, 20, "Analysing map corners...")
        report_path = output_dir / "report.pdf"

        _overlay_on_pdf(
            pdf_path=pdf_path,
            screenshot_path=Path(survey_image_path) if survey_image_path else None,
            prism_id=prism_id,
            node_name=node_name,
            instance=instance,
            map_type=map_type,
            output_path=report_path,
        )
        t0 = _record("REPORTING", t0)

        _update(JobStatus.PROCESSING, 90, "Saving annotated PDF...")
        total_ms = (time.perf_counter() - job_start) * 1000
        job_store[job_id]["stage_times"]["total_ms"] = round(total_ms, 1)

        job_store[job_id].update({
            "status": JobStatus.COMPLETED,
            "progress": 100.0,
            "message": "Coax Before pipeline completed.",
            "report_path": str(report_path.relative_to(settings.BASE_DIR)),
        })
        logger.info(f"[{job_id}] ✅ Coax Before pipeline complete in {total_ms:.0f} ms.")

    except Exception as exc:
        logger.exception(f"[{job_id}] ❌ Coax Before pipeline failed: {exc}")
        job_store[job_id].update({
            "status": JobStatus.FAILED,
            "message": "Coax Before pipeline encountered an error.",
            "error": str(exc),
        })

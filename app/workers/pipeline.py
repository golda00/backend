"""
Pipeline Worker
---------------
Orchestrates the end-to-end processing pipeline for a single analysis job.

Batch 1 Optimizations applied here:
  OPT-1  Detector singleton — models passed in from app.state; NOT re-loaded per job.
  OPT-2  SIFT single-pass   — align_and_pad_maps() now returns W so _compute_W_inv()
           is eliminated entirely (was running SIFT twice on the same images).
  OPT-3  Per-stage timing   — every pipeline stage records wall-clock ms so the
           /jobs/{id}/result response includes a perf breakdown dashboard.

Stage flow:
  1. ALIGNING   — PDF → image, SIFT feature match, universal canvas warp
  2. TILING     — Slice canvas into 640×640 tiles
  3. PROCESSING — YOLO detect + OCR for every tile pair
  4. MATCHING   — 4-pass object matcher
  5. REPORTING  — Annotated callouts → vector PDF overlay
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

from app.core.config import Settings, BASE_DIR
from app.models.schemas import JobStatus
from app.services.alignment import (
    pdf_to_image,
    align_and_pad_maps,
    iter_tiles,         # OPT-4: streaming generator
    save_tiles,         # kept for saving named tile images to disk
)
from app.services.matching import match_objects
from app.services.rules import RuleEngine
from app.services.reporting import generate_vector_report, generate_final_report
from app.services.fiber_overview import FiberOverviewProcessor

logger = logging.getLogger(__name__)


def run_pipeline_sync(
    job_id: str,
    job_store: dict,
    settings: Settings,
    detector=None,
) -> None:
    job_start = time.perf_counter()

    def _update(status: JobStatus, pct: float, msg: str) -> None:
        job_store[job_id].update({"status": status, "progress": pct, "message": msg})
        logger.info(f"[{job_id}] [{pct:3.0f}%] {msg}")

    def _record(stage: str, t0: float) -> float:
        elapsed = (time.perf_counter() - t0) * 1000
        if "stage_times" not in job_store[job_id]:
            job_store[job_id]["stage_times"] = {}
        job_store[job_id]["stage_times"][stage] = round(elapsed, 1)
        logger.info(f"[{job_id}] ⏱  {stage}: {elapsed:.0f} ms")
        return time.perf_counter()

    try:
        job = job_store[job_id]
        status = job.get("status")
        output_dir = Path(job["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ─── PHASE 1: ALIGN & TILE ───────────────────────────────────────────
        if status == JobStatus.QUEUED:
            before_path = Path(job["before_path"])
            after_path  = Path(job["after_path"])
            dpi = job.get("dpi", settings.PDF_DPI)
            
            t0 = time.perf_counter()
            _update(JobStatus.ALIGNING, 5.0, f"Preparing maps at {dpi} DPI...")
            img_before = pdf_to_image(before_path, dpi=dpi)
            img_after  = pdf_to_image(after_path,  dpi=dpi)

            final_before, final_after, W = align_and_pad_maps(img_before, img_after)
            W_inv = np.linalg.inv(W) if W is not None else np.eye(3, dtype=np.float32)

            cv2.imwrite(str(output_dir / "aligned_before.png"), final_before)
            cv2.imwrite(str(output_dir / "aligned_after.png"),  final_after)
            np.save(str(output_dir / "W_inv.npy"), W_inv)
            t0 = _record("alignment_ms", t0)

            _update(JobStatus.TILING, 10.0, "Generating tiles for verification...")
            before_tile_dir = output_dir / "tiles" / "before"
            after_tile_dir  = output_dir / "tiles" / "after"
            before_tile_dir.mkdir(parents=True, exist_ok=True)
            after_tile_dir.mkdir(parents=True, exist_ok=True)

            # Save tiles and pick samples
            sample_indices = []
            for t in iter_tiles(final_before, settings.TILE_SIZE, settings.TILE_OVERLAP):
                cv2.imwrite(str(before_tile_dir / f"before_{t['index']}.png"), t["tile"])
                if t["index"] % 10 == 0: # sample every 10th tile
                    sample_indices.append(t["index"])
            
            for t in iter_tiles(final_after, settings.TILE_SIZE, settings.TILE_OVERLAP):
                cv2.imwrite(str(after_tile_dir / f"after_{t['index']}.png"), t["tile"])
            
            _record("tiling_ms", t0)
            
            # PAUSE for DPI verification
            job_store[job_id].update({
                "status": JobStatus.AWAITING_DPI_CONFIRM,
                "progress": 15.0,
                "message": "Please verify sample tiles for DPI zoom level.",
                "sample_tiles": sample_indices[:3], # first 3 samples
            })
            return

        # ─── PHASE 2: DETECTION & RULES ──────────────────────────────────────
        if status == JobStatus.PROCESSING:
            t0 = time.perf_counter()
            _update(JobStatus.PROCESSING, 20.0, "Running AI detection on maps...")
            
            # Reload aligned images and matrix
            final_before = cv2.imread(str(output_dir / "aligned_before.png"))
            final_after  = cv2.imread(str(output_dir / "aligned_after.png"))
            W_inv = np.load(str(output_dir / "W_inv.npy"))
            
            if detector is None:
                from app.services.vision import TelecomDetector
                detector = TelecomDetector(
                    main_model_path=settings.MAIN_MODEL_PATH,
                    ps_model_path=settings.PS_MODEL_PATH,
                    node_model_path=settings.NODE_MODEL_PATH,
                    internal_model_path=settings.INTERNAL_MODEL_PATH,
                    use_gpu=settings.USE_GPU,
                    dpi=job.get("dpi", settings.PDF_DPI),
                )

            rule_engine = RuleEngine()
            all_callouts_flat = []
            all_callout_records = []
            flagged_tiles = [] # index of tiles with G, POWER BLOCK, or WARNING

            # Iterate tiles again (reloading from disk for RAM safety)
            before_tile_dir = output_dir / "tiles" / "before"
            after_tile_dir  = output_dir / "tiles" / "after"
            
            # Geometry for progress
            h_c, w_c = final_after.shape[:2]
            step = int(settings.TILE_SIZE * (1 - settings.TILE_OVERLAP))
            total_tiles = sum(1 for _ in range(0, h_c, step) for _ in range(0, w_c, step))
            
            tile_offsets = {}
            for idx, t_a in enumerate(iter_tiles(final_after, settings.TILE_SIZE, settings.TILE_OVERLAP)):
                t_idx = t_a["index"]
                img_a = t_a["tile"]
                tile_offsets[t_idx] = (t_a["x"], t_a["y"])
                
                img_b_path = before_tile_dir / f"before_{t_idx}.png"
                if not img_b_path.exists(): continue
                img_b = cv2.imread(str(img_b_path))

                objs_b = detector.detect_objects(img_b, conf_threshold=0.01)
                objs_a = detector.detect_objects(img_a, conf_threshold=0.01)
                objs_b = detector.run_ocr_on_objects(img_b, objs_b)
                objs_a = detector.run_ocr_on_objects(img_a, objs_a)

                matches, removed, added = match_objects(objs_b, objs_a)
                callouts = rule_engine.generate_callouts(
                    matches, removed, added,
                    before_node_type=job.get("before_node_type"),
                    before_node_names=job.get("before_node_names"),
                    after_node_type=job.get("after_node_type"),
                    after_node_names=job.get("after_node_names"),
                )

                has_review_worthy = False
                for c in callouts:
                    c_text = c["text"].upper()
                    # Trigger review for G, Power Block, or any Warning
                    if "G" == c_text or "POWER BLOCK" in c_text or "WARNING" in c_text or "OVER 80%" in c_text:
                        has_review_worthy = True
                    
                    all_callout_records.append({"tile_idx": t_idx, "lx": c["loc"][0], "ly": c["loc"][1], "text": c["text"]})
                    all_callouts_flat.append(c)
                
                if has_review_worthy:
                    flagged_tiles.append(t_idx)

                _update(JobStatus.PROCESSING, 20.0 + (idx/total_tiles)*60.0, f"Processing tile {idx+1}/{total_tiles}")

            t0 = _record("inference_ms", t0)
            
            # Save state for reporting phase
            job_store[job_id].update({
                "all_callouts": all_callouts_flat,
                "all_callout_records": all_callout_records,
                "tile_offsets": tile_offsets,
                "W_inv": W_inv.tolist(),
            })

            # PAUSE for content review if flagged tiles exist
            if flagged_tiles:
                job_store[job_id].update({
                    "status": JobStatus.AWAITING_REVIEW,
                    "progress": 82.0,
                    "message": "Found callouts requiring manual verification.",
                    "flagged_tiles": flagged_tiles[:10], # Limit to first 10 for review speed
                })
                return
            else:
                # If nothing flagged, go straight to reporting
                job_store[job_id]["status"] = JobStatus.REPORTING

        # ─── PHASE 3: REPORTING ──────────────────────────────────────────────
        if status == JobStatus.REPORTING:
            t0 = time.perf_counter()
            _update(JobStatus.REPORTING, 85.0, "Generating final report...")
            
            after_path = Path(job["after_path"])
            report_path = output_dir / "report.pdf"
            
            # Load state
            callouts = job.get("all_callouts_visible") or job.get("all_callout_records")
            tile_offsets = {int(k): v for k, v in job.get("tile_offsets", {}).items()}
            W_inv = np.array(job.get("W_inv"), dtype=np.float32)

            generate_vector_report(
                after_pdf_path=after_path,
                callout_records=callouts,
                tile_offsets=tile_offsets,
                W_inv=W_inv,
                output_path=report_path,
                dpi=job.get("dpi", settings.PDF_DPI),
                survey_image_path=job.get("survey_image"),
                title_box_data=job.get("title_box"),
            )
            _record("reporting_ms", t0)

            total_ms = (time.perf_counter() - job_start) * 1000
            job_store[job_id].update({
                "status": JobStatus.COMPLETED,
                "progress": 100.0,
                "message": "Analysis complete.",
                "report_path": str(report_path.relative_to(settings.BASE_DIR)),
            })
            logger.info(f"[{job_id}] ✅ Pipeline complete.")

    except Exception as exc:
        logger.exception(f"[{job_id}] ❌ Pipeline failed: {exc}")
        job_store[job_id].update({"status": JobStatus.FAILED, "error": str(exc)})


def run_fiber_overview_pipeline(
    job_id: str,
    job_store: dict,
    settings: Settings,
    processor: FiberOverviewProcessor = None,
) -> None:
    """
    Runs the fiber overview pipeline synchronously.
    Logic: PDF -> Image -> Node Detect -> Cable Trace -> Port Detect -> Report.
    """
    job_start = time.perf_counter()

    def _update(status: JobStatus, pct: float, msg: str) -> None:
        job_store[job_id].update({"status": status, "progress": pct, "message": msg})
        logger.info(f"[{job_id}] [{pct:3.0f}%] {msg}")

    def _record(stage: str, t0: float) -> float:
        elapsed = (time.perf_counter() - t0) * 1000
        job_store[job_id]["stage_times"][stage] = round(elapsed, 1)
        logger.info(f"[{job_id}] ⏱  {stage}: {elapsed:.0f} ms")
        return time.perf_counter()

    try:
        job = job_store[job_id]
        pdf_path = Path(job["pdf_path"])
        output_dir = Path(job["output_dir"])
        dpi = job.get("dpi", settings.PDF_DPI)
        
        # Business Logic Parameters
        is_connected = job.get("is_connected", True)
        hub_name = job.get("hub_name", "")
        port_name = job.get("port_name", "")
        splice_can_name = job.get("splice_can_name", "")
        node_name_input = job.get("title_box", {}).get("node_name", "") or "NODE"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        job_store[job_id]["stage_times"] = {}

        if processor is None:
            logger.warning(f"[{job_id}] Fiber processor not available; loading now.")
            processor = FiberOverviewProcessor(model_path=settings.FIBER_NODE_MODEL_PATH)

        t0 = time.perf_counter()

        # 1. CONVERT PDF -> IMAGE
        _update(JobStatus.PROCESSING, 10, "Converting PDF to image...")
        img = pdf_to_image(pdf_path, dpi=dpi)
        t0 = _record("CONVERSION", t0)

        # 2. NODE DETECTION
        _update(JobStatus.PROCESSING, 30, "Detecting fiber node...")
        bbox, center, conf = processor.detect_node(img)
        if bbox is None:
            raise ValueError("No fiber node detected in the overview map.")
        t0 = _record("DETECTION", t0)

        # 3. CABLE TRACING
        _update(JobStatus.PROCESSING, 50, "Tracing fiber cable...")
        skeleton = processor.extract_cable_skeleton(img, bbox)
        if skeleton is None:
            raise ValueError("Could not extract fiber cable connected to node.")
        
        port_pos = processor.find_port_position(skeleton, bbox)
        if port_pos is None:
            raise ValueError("Could not determine port position on cable.")
        t0 = _record("TRACING", t0)

        # 4. REPORT GENERATION
        _update(JobStatus.PROCESSING, 80, "Generating final report...")
        scale = dpi / 72.0
        pdf_node_pos = (center[0] / scale, center[1] / scale)
        pdf_port_pos = (port_pos[0] / scale, port_pos[1] / scale)

        # Construct Port Callout Text based on Business Logic
        if is_connected:
            port_text = f"HUB : {hub_name}\nPORT/PANEL : {port_name}"
        else:
            port_text = (
                f"TRACE STOPS AT RAW CAN ({splice_can_name}) ; "
                "EXISTING SPLICING UNAVAILABLE , A CAN AUDIT REQUIRED FOR VERIFICATION"
            )

        callouts = [
            {"x": pdf_node_pos[0], "y": pdf_node_pos[1], "text": "NODE"},
            {"x": pdf_port_pos[0], "y": pdf_port_pos[1], "text": port_text},
        ]
        
        report_filename = f"report_{job_id}.pdf"
        report_path = output_dir / report_filename
        
        survey_image_path = job.get("survey_image_path")
        title_box = job.get("title_box")
        
        generate_final_report(
            pdf_path=pdf_path,
            callouts=callouts,
            output_path=report_path,
            dpi=dpi,
            survey_image_path=survey_image_path,
            title_box_data=title_box
        )
        t0 = _record("REPORTING", t0)

        # COMPLETE
        total_ms = (time.perf_counter() - job_start) * 1000
        job_store[job_id].update({
            "status": JobStatus.COMPLETED,
            "progress": 100,
            "message": f"Success! Fiber overview processed in {total_ms/1000:.1f}s.",
            "report_path": str(report_path.relative_to(settings.BASE_DIR)),
            "callouts": callouts
        })

    except Exception as e:
        logger.exception(f"[{job_id}] Pipeline failed: {e}")
        job_store[job_id].update({
            "status": JobStatus.FAILED,
            "message": f"Error: {str(e)}",
            "error": str(e)
        })

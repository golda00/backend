"""
Pipeline Orchestrator — Restored Coordinate & Callout Flow
-----------------------------------------------------------
Key fixes:
  1. Coax pipeline:  encodes callouts with GLOBAL image coords (gx/gy) so
     reporting.py does a clean img→PDF conversion without tile offset math.
  2. Fiber Overview: builds node/splice callout records and passes them to
     generate_final_report so annotations actually appear on the PDF.
"""
import logging, time, cv2, numpy as np, fitz
from pathlib import Path
from app.models.schemas import JobStatus
from app.services.alignment import align_and_pad_maps, iter_tiles
from app.services.matching import match_objects
from app.services.rules import RuleEngine
from app.services.reporting import generate_vector_report, generate_final_report
from app.services.fiber_overview import FiberOverviewProcessor

logger = logging.getLogger(__name__)


def _pdf_to_bgr(p, dpi=300):
    """Render first page of PDF to BGR numpy array, safely handling RGBA."""
    doc  = fitz.open(str(p))
    page = doc[0]
    # Force RGB (no alpha) to guarantee 3-channel output
    pix  = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
    samples = pix.samples
    # Defensive: handle both 3-channel (RGB) and 4-channel (RGBA) pixmaps
    n_channels = pix.n
    if n_channels == 4:
        img = np.frombuffer(samples, dtype=np.uint8).reshape(pix.h, pix.w, 4)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = np.frombuffer(samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()
    # Blank guard: if >95% of pixels are black the render likely failed
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if (np.sum(gray < 10) / gray.size) > 0.95:
        logger.warning("_pdf_to_bgr: suspiciously dark image — returning white fallback")
        return np.full((pix.h, pix.w, 3), 255, dtype=np.uint8)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# 1. COAX PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline_sync(job_id, job_store, settings, detector=None, **kwargs):
    detector = detector or kwargs.get("detector") or kwargs.get("processor")
    if detector is None:
        from app.workers.tasks import _get_detector
        detector = _get_detector(settings)

    def _update(s, p, m):
        if job_id not in job_store:
            job_store[job_id] = {}
        job_store[job_id].update({"status": s, "progress": p, "message": m})
        logger.info(f"[{job_id}] [{p:.0f}%] {m}")

    try:
        job = job_store[job_id]
        out = Path(job["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        dpi = job.get("dpi", 300)

        # ── Phase 1: Alignment ────────────────────────────────────────────────
        if job.get("status") == JobStatus.QUEUED:
            _update(JobStatus.ALIGNING, 5.0, "Aligning Coax Maps...")
            fb = _pdf_to_bgr(Path(job["before_path"]), dpi=dpi)
            fa = _pdf_to_bgr(Path(job["after_path"]),  dpi=dpi)
            fb, fa, W = align_and_pad_maps(fb, fa)
            cv2.imwrite(str(out / "aligned_after.png"),  fa)
            cv2.imwrite(str(out / "aligned_before.png"), fb)
            W_inv = np.linalg.inv(W) if W is not None else np.eye(3)
            np.save(str(out / "W_inv.npy"), W_inv)

            # ── Save sample tiles for DPI confirmation preview ────────────────
            # The frontend shows tiles/{before|after}/before_N.png & after_N.png
            tile_size = 640
            sample_indices = []
            
            td_before = out / "tiles" / "before"
            td_after  = out / "tiles" / "after"
            td_before.mkdir(parents=True, exist_ok=True)
            td_after.mkdir(parents=True, exist_ok=True)

            # Save ALL tiles for comprehensive review
            for t in iter_tiles(fa, tile_size, 0.1):
                s_num = t["index"]
                tx, ty = t["x"], t["y"]
                
                # Extract tiles from both maps at the same coordinates
                before_tile = fb[ty:ty+tile_size, tx:tx+tile_size]
                after_tile  = t["tile"] # fa[ty:ty+tile_size, tx:tx+tile_size]
                
                cv2.imwrite(str(td_before / f"before_{s_num}.png"), before_tile)
                cv2.imwrite(str(td_after  / f"after_{s_num}.png"),  after_tile)
                sample_indices.append(s_num)
                
                if s_num % 10 == 0:
                    logger.info(f"[{job_id}] Saved sample tile pair {s_num}")

            if not sample_indices:
                # Fallback: save top-left tile unconditionally
                td_before = out / "tiles" / "before"
                td_after  = out / "tiles" / "after"
                td_before.mkdir(parents=True, exist_ok=True)
                td_after.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(td_before / "before_1.png"), fb[:tile_size, :tile_size])
                cv2.imwrite(str(td_after  / "after_1.png"),  fa[:tile_size, :tile_size])
                logger.info(f"[{job_id}] Saved fallback sample tile pair (1)")
                sample_indices = [1]

            job_store[job_id].update({
                "status":       JobStatus.AWAITING_DPI_CONFIRM,
                "progress":     15.0,
                "sample_tiles": sample_indices,
            })
            return

        # ── Phase 2: Detection + Reporting ───────────────────────────────────
        if job.get("status") == JobStatus.PROCESSING:
            _update(JobStatus.PROCESSING, 20.0, "AI analysis running...")
            fa   = cv2.imread(str(out / "aligned_after.png"))
            fb   = cv2.imread(str(out / "aligned_before.png"))
            W_inv = np.load(str(out / "W_inv.npy"))
            re   = RuleEngine()
            callout_records: list[dict] = []
            tile_offsets:    dict       = {}

            tile_count = 0
            all_tiles = list(iter_tiles(fa, 640, 0.1))
            total_tiles = len(all_tiles)

            for t in all_tiles:
                t_idx = t["index"]
                tx, ty = t["x"], t["y"]
                tile_offsets[t_idx] = (tx, ty)

                tile_count += 1
                if tile_count % 5 == 0 or tile_count == total_tiles:
                    _update(JobStatus.PROCESSING, 20.0 + (tile_count / total_tiles * 50.0), 
                           f"Analysing map... tile {tile_count}/{total_tiles}")

                b_tile = fb[ty: ty + 640, tx: tx + 640]
                a_tile = t["tile"]
                m, r, a = match_objects(
                    detector.detect_objects(b_tile),
                    detector.detect_objects(a_tile),
                )
                for c in re.generate_callouts(m, r, a,
                                                before_node_type=job.get("before_node_type"),
                                                before_node_names=job.get("before_node_names"),
                                                after_node_type=job.get("after_node_type"),
                                                after_node_names=job.get("after_node_names")):
                    # Convert to GLOBAL image coords
                    loc_x, loc_y = c["loc"]
                    gx = tx + loc_x
                    gy = ty + loc_y
                    
                    text_upper = c["text"].upper()
                    is_flagged = "WARNING" in text_upper or text_upper in ["G", "POWERBLOCK", "ADD POWER BLOCK", "REMOVE POWER BLOCK"]
                    
                    callout_records.append({
                        "tile_idx": t_idx,
                        "gx": gx,
                        "gy": gy,
                        "lx": loc_x,  # kept for compat
                        "ly": loc_y,
                        "text": c["text"],
                        "type": "FLAGGED" if is_flagged else "NORMAL"
                    })

            # Save state and halt for human review
            # Extract unique tile indices that have flagged callouts
            flagged_indices = list({c["tile_idx"] for c in callout_records if c.get("type") == "FLAGGED"})

            # Save state and halt for human review
            job_store[job_id].update({
                "status": JobStatus.AWAITING_REVIEW,
                "progress": 80.0,
                "flagged_tiles": flagged_indices,
                "all_callouts": callout_records
            })
            return

        if job.get("status") == JobStatus.REPORTING:
            _update(JobStatus.REPORTING, 85.0, "Generating vector report...")
            fa   = cv2.imread(str(out / "aligned_after.png"))
            W_inv = np.load(str(out / "W_inv.npy"))
            
            # Apply user overrides
            callout_records = job.get("all_callouts", [])
            overrides = job.get("all_callouts_visible", [])
            
            # overrides is a list of {tileIdx, action, newText}
            for override in overrides:
                target_idx = override.get("tileIdx")
                action = override.get("action")
                new_text = override.get("newText")
                
                for c in callout_records:
                    if c["tile_idx"] == target_idx:
                        if action == "REMOVE":
                            c["removed"] = True
                        elif action == "RENAME" and new_text:
                            c["text"] = new_text

            final_callouts = [c for c in callout_records if not c.get("removed")]

            tile_offsets = {}
            for t in iter_tiles(fa, 640, 0.1):
                tile_offsets[t["index"]] = (t["x"], t["y"])

            generate_vector_report(
                after_pdf_path=Path(job["after_path"]),
                callout_records=final_callouts,
                tile_offsets=tile_offsets,
                W_inv=W_inv,
                output_path=out / "report.pdf",
                dpi=dpi,
                survey_image_path=job.get("survey_image_path"),
                title_box_data={
                    "prism_id":   job.get("title_box", {}).get("prism_id", ""),
                    "node_name":  job.get("title_box", {}).get("node_name", ""),
                    "instance":   job.get("title_box", {}).get("instance", ""),
                    "map_type":   job.get("title_box", {}).get("map_type", "AFTER"),
                    "page_count": 1,
                },
                title_font_size=24,
            )
            logger.info(f"[{job_id}] Final report saved to: {out / 'report.pdf'}")
            job_store[job_id].update({
                "status":      JobStatus.COMPLETED,
                "progress":    100.0,
                "callouts":    final_callouts,
                "report_path": str((out / "report.pdf").relative_to(settings.BASE_DIR)),
            })

    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        _update(JobStatus.FAILED, 0, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 2. FIBER OVERVIEW PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_fiber_overview_pipeline(job_id, job_store, settings, processor=None, **kwargs):
    """
    Runs node + cable detection on the Fiber Overview map,
    builds callout records from node / splice-can results,
    and renders them via generate_final_report.
    """
    processor = processor or kwargs.get("processor")
    if processor is None:
        from app.workers.tasks import _get_overview_processor
        processor = _get_overview_processor(settings)

    def _update(s, p, m):
        job_store[job_id].update({"status": s, "progress": p, "message": m})
        logger.info(f"[{job_id}] [{p:.0f}%] {m}")

    try:
        job = job_store[job_id]
        out = Path(job["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        dpi = job.get("dpi", 300)
        title_box = job.get("title_box", {})

        _update(JobStatus.PROCESSING, 20.0, "Rendering fiber overview map...")
        logger.info(f"[{job_id}] Rendering fiber overview map...")

        # ── Render PDF to image ───────────────────────────────────────────────
        pdf_path = Path(job["pdf_path"])
        doc  = fitz.open(str(pdf_path))
        page = doc[0]
        pix  = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
        img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        doc.close()

        # ── Node detection ────────────────────────────────────────────────────
        _update(JobStatus.PROCESSING, 40.0, "Detecting fiber node...")
        logger.info(f"[{job_id}] Detecting fiber node...")
        bbox, center, conf = processor.detect_node(img_bgr)

        callout_records: list[dict] = []
        scale = 72.0 / dpi  # image px → PDF pts

        if bbox and center:
            nx, ny = center
            # NODE callout
            callout_records.append({
                "gx": float(nx), "gy": float(ny),
                "text": "NODE",
            })

            # ── Port/splice-can tracing ───────────────────────────────────────
            _update(JobStatus.PROCESSING, 60.0, "Tracing cable skeleton...")
            logger.info(f"[{job_id}] Tracing cable skeleton...")
            skeleton = processor.extract_cable_skeleton(img_bgr, bbox)
            if skeleton is not None:
                port_pos = processor.find_port_position(skeleton, bbox)
                if port_pos:
                    px, py = port_pos
                    
                    is_connected    = job.get("is_connected", True)
                    hub_name        = job.get("hub_name", "")
                    port_name       = job.get("port_name", "")
                    splice_can_name = job.get("splice_can_name", "")
                    
                    if is_connected:
                        port_text = f"HUB : {hub_name}\nPORT/PANEL : {port_name}"
                    else:
                        port_text = (
                            f"TRACE STOPS AT RAW CAN ({splice_can_name}) ; "
                            "EXISTING SPLICING UNAVAILABLE , A CAN AUDIT REQUIRED FOR VERIFICATION"
                        )
                    
                    callout_records.append({
                        "gx": float(px), "gy": float(py),
                        "text": port_text,
                    })

        _update(JobStatus.REPORTING, 80.0, "Rendering PDF report...")

        report_path = out / "report.pdf"
        generate_final_report(
            pdf_path=pdf_path,
            callouts=callout_records,
            output_path=report_path,
            dpi=dpi,
            survey_image_path=job.get("survey_image_path"),
            title_box_data={
                "prism_id":   title_box.get("prism_id", ""),
                "node_name":  title_box.get("node_name", ""),
                "instance":   title_box.get("instance", ""),
                "map_type":   "FIBER OVERVIEW",
                "page_count": 1,
            },
            title_font_size=14,
        )
        logger.info(f"[{job_id}] Fiber Overview report saved to: {report_path}")

        job_store[job_id].update({
            "status":      JobStatus.COMPLETED,
            "progress":    100.0,
            "message":     "Fiber Overview report complete.",
            "callouts":    callout_records,
            "report_path": str(report_path.relative_to(settings.BASE_DIR)),
        })

    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        if job_id in job_store:
            job_store[job_id].update({"status": JobStatus.FAILED, "message": str(e)})

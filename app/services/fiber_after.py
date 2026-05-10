import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import cv2
import fitz
import numpy as np
from PIL import Image
from ultralytics import YOLO

from app.core.config import Settings
from app.models.schemas import JobStatus
from app.services.reporting import generate_vector_report

logger = logging.getLogger(__name__)

class FiberAfterEngine:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def extract_text(self, pdf_path: str, zoom: float) -> List[Dict[str, Any]]:
        """Extracts vector text from PDF and maps to image space."""
        text_elements = []
        try:
            doc = fitz.open(pdf_path)
            page = doc[0]
            dict_data = page.get_text("dict")
            for block in dict_data.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_elements.append({
                                "text": span["text"].strip(),
                                "bbox": [v * zoom for v in span["bbox"]], # To Image Space
                                "color": span.get("color"),
                                "source": "fitz"
                            })
            doc.close()
        except Exception as e:
            logger.warning(f"Fitz text extraction failed: {e}")
        return text_elements

    def get_region_color(self, img_np: np.ndarray, bbox: List[float]) -> List[float]:
        """Returns average RGB of a region."""
        x0, y0, x1, y1 = map(int, bbox)
        region = img_np[y0:y1, x0:x1]
        if region.size == 0: return [0, 0, 0]
        return np.mean(region, axis=(0, 1)).tolist()

    def is_yellow(self, color: List[float]) -> bool:
        """Heuristic for 'Yellow' tags on telecom maps."""
        r, g, b = color
        return r > 165 and g > 155 and b < 120 and r > b and g > b

    def run_detection(self, 
                      img_np: np.ndarray, 
                      tile_size: int = 1024, 
                      overlap: int = 128, 
                      conf: float = 0.5) -> List[Dict[str, Any]]:
        """Sliding window YOLOv8 detection."""
        h, w = img_np.shape[:2]
        detections = []
        
        y = 0
        while y < h:
            x = 0
            while x < w:
                tile_h = min(tile_size, h - y)
                tile_w = min(tile_size, w - x)
                tile = img_np[y:y+tile_h, x:x+tile_w]
                
                if tile.shape[0] < 32 or tile.shape[1] < 32:
                    x += (tile_size - overlap)
                    continue
                
                results = self.model(tile, conf=conf, verbose=False)
                for r in results:
                    boxes = r.boxes if r.boxes is not None else r.obb
                    if boxes is not None:
                        for box in boxes:
                            if hasattr(box, 'xyxy'):
                                b = box.xyxy[0].cpu().numpy()
                            else:
                                b = box.xywhr[0].cpu().numpy()[:4]
                            
                            detections.append({
                                "class": self.model.names[int(box.cls)],
                                "bbox": [float(b[0] + x), float(b[1] + y), float(b[2] + x), float(b[3] + y)],
                                "conf": float(box.conf)
                            })
                
                if x + tile_w >= w: break
                x += (tile_size - overlap)
            if y + tile_h >= h: break
            y += (tile_size - overlap)
            
        return detections

    def apply_rules(self, 
                    detections: List[Dict[str, Any]], 
                    text_elements: List[Dict[str, Any]], 
                    img_np: np.ndarray, 
                    zoom: float) -> List[Dict[str, Any]]:
        """Applies business rules: Nodes and closest Splice Can to FE1."""
        final_results = []
        
        def get_dist(bbox1, bbox2):
            c1 = [(bbox1[0] + bbox1[2])/2, (bbox1[1] + bbox1[3])/2]
            c2 = [(bbox2[0] + bbox2[2])/2, (bbox2[1] + bbox2[3])/2]
            return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

        # 1. Detect FE1 Tags
        fe1_tags = []
        for t in text_elements:
            if "FE1" in t["text"].upper():
                color = self.get_region_color(img_np, t["bbox"])
                if self.is_yellow(color):
                    fe1_tags.append(t)

        nodes = [d for d in detections if d["class"].lower() == "node"]
        splice_cans = [d for d in detections if "splice" in d["class"].lower()]

        # Rule: Nodes -> SPLICE #1 and HUB callouts
        for node in nodes:
            node["callouts"] = {}
            # Splice #1
            s1_cands = [t for t in text_elements if "SPLICE" in t["text"].upper() and ("#1" in t["text"] or "SPLICE1" in t["text"].upper())]
            if s1_cands:
                nearest = min(s1_cands, key=lambda t: get_dist(node["bbox"], t["bbox"]))
                if get_dist(node["bbox"], nearest["bbox"]) < (600 * zoom):
                    node["callouts"]["splice1"] = nearest["text"]

            # Hub/Port
            h_cands = [t for t in text_elements if any(x in t["text"].upper() for x in ["HUB", "PANEL", "PORT"])]
            if h_cands:
                nearest = min(h_cands, key=lambda t: get_dist(node["bbox"], t["bbox"]))
                if get_dist(node["bbox"], nearest["bbox"]) < (600 * zoom):
                    node["callouts"]["hub"] = nearest["text"]
            
            final_results.append(node)

        # Rule: Nearest Splice Can to FE1 tag -> SPLICE #2 and MUX
        if splice_cans and fe1_tags:
            for fe1 in fe1_tags:
                nearest_can = min(splice_cans, key=lambda c: get_dist(fe1["bbox"], c["bbox"]))
                if get_dist(fe1["bbox"], nearest_can["bbox"]) < (600 * zoom):
                    if "callouts" not in nearest_can: nearest_can["callouts"] = {}
                    
                    # Splice #2
                    s2_cands = [t for t in text_elements if "SPLICE" in t["text"].upper() and ("#2" in t["text"] or "SPLICE2" in t["text"].upper())]
                    if s2_cands:
                        n_s2 = min(s2_cands, key=lambda t: get_dist(nearest_can["bbox"], t["bbox"]))
                        if get_dist(nearest_can["bbox"], n_s2["bbox"]) < (600 * zoom):
                            nearest_can["callouts"]["splice2"] = n_s2["text"]
                    
                    # MUX
                    m_cands = [t for t in text_elements if "MUX" in t["text"].upper()]
                    if m_cands:
                        n_m = min(m_cands, key=lambda t: get_dist(nearest_can["bbox"], t["bbox"]))
                        if get_dist(nearest_can["bbox"], n_m["bbox"]) < (600 * zoom):
                            nearest_can["callouts"]["mux"] = n_m["text"]
                    
                    if nearest_can not in final_results:
                        final_results.append(nearest_can)

        return final_results

def run_fiber_after_pipeline(job_id: str, store: Any, settings: Settings):
    """
    Two-stage execution entry point for Fiber After (Interactive).
    Phase 1: ALIGNING/PROCESSING -> AWAITING_REVIEW
    Phase 2: REPORTING -> COMPLETED
    """
    try:
        if isinstance(store, dict):
             job = store[job_id]
             loop = None
        else:
             import asyncio
             loop = asyncio.new_event_loop()
             asyncio.set_event_loop(loop)
             raw = loop.run_until_complete(store.get(job_id))
             job = raw

        status = job.get("status")
        output_dir = Path(job["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        dpi = job.get("dpi", 50)
        zoom = dpi / 72.0

        def _sync_update(data: dict):
            if "message" in data:
                logger.info(f"[{job_id}] [{data.get('progress', 0.0):.0f}%] {data['message']}")
            job.update(data)
            # Handle both Store object (Celery) and dict stub (Local threading)
            if hasattr(store, 'set_sync'):
                store.set_sync(job_id, job)
            elif isinstance(store, dict):
                store[job_id] = job
            else:
                loop.run_until_complete(store.update(job_id, data))

        # ─── PHASE 1: DETECTION ──────────────────────────────────────────────
        if status in [JobStatus.QUEUED, JobStatus.PROCESSING]:
            _sync_update({"status": JobStatus.PROCESSING, "progress": 10.0, "message": "Rendering map and extracting text..."})
            
            pdf_path = job["pdf_path"]
            doc = fitz.open(pdf_path)
            page = doc[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img_after = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_np = np.array(img_after)
            doc.close()

            engine = FiberAfterEngine(str(settings._resolve_model("fiber_best.pt")))
            text_elements = engine.extract_text(pdf_path, zoom)
            
            _sync_update({"progress": 30.0, "message": "Running AI object detection..."})
            detections = engine.run_detection(img_np)
            
            _sync_update({"progress": 60.0, "message": "Applying telecom rules..."})
            results = engine.apply_rules(detections, text_elements, img_np, zoom)

            # Save tiles for verification modal
            tile_dir = output_dir / "tiles" / "after"
            tile_dir.mkdir(parents=True, exist_ok=True)
            
            flagged_tiles = []
            all_callout_records = []
            
            num_results = len(results)
            for i, r in enumerate(results):
                if i % 5 == 0 or i == num_results - 1:
                     _sync_update({"progress": 60.0 + (i / num_results * 20.0), "message": f"Processing detection {i+1}/{num_results}..."})
                
                bbox = r["bbox"]
                cx, cy = (bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2
                
                # Crop a 640x640 tile around the detection
                tile_size = 640
                tx1 = max(0, int(cx - tile_size/2))
                ty1 = max(0, int(cy - tile_size/2))
                tx2 = min(img_np.shape[1], tx1 + tile_size)
                ty2 = min(img_np.shape[0], ty1 + tile_size)
                
                tile = img_np[ty1:ty2, tx1:tx2]
                tile_idx = i
                cv2.imwrite(str(tile_dir / f"after_{tile_idx}.png"), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
                
                cls = r["class"].upper()
                msg = r.get("message", "").upper()
                
                # Metadata for Node labels
                title_box = job.get("title_box", {})
                hub_val = str(title_box.get("hub", "")).upper()
                port_val = str(title_box.get("port_panel", "")).upper()

                # 2 callouts per symbol, encoded as GLOBAL image coords (gx/gy)
                callout_data = []  # list of (text, y_offset)

                callouts = r.get("callouts", {})
                if "SPLICE CAN" in cls or "SPLICE_CAN" in cls:
                    txt2 = callouts.get("splice2", "SPLICE #2")
                    callout_data = [(txt2, -20)]
                    if job.get("include_mux", True):
                        txt_mux = callouts.get("mux", "MUX LOCATION")
                        callout_data.append((txt_mux, 20))
                elif "NODE" in cls:
                    txt1 = callouts.get("splice1", "SPLICE #1")
                    node_detail = f"HUB: {hub_val}\nPORT/PANEL: {port_val}"
                    callout_data = [
                        (txt1, -22),
                        (node_detail, 22),
                    ]
                else:
                    callout_data = [(cls, 0)]

                for txt, y_off in callout_data:
                    all_callout_records.append({
                        "tile_idx": tile_idx,
                        "gx": cx,           # global image x
                        "gy": cy + y_off,   # global image y (with offset)
                        "lx": cx - tx1,     # kept for compat
                        "ly": (cy - ty1) + y_off,
                        "text": txt,
                    })

                flagged_tiles.append(tile_idx)

            # Store detection state
            # Store detection state
            _sync_update({
                "progress": 82.0,
                "message": "Detection finished. Starting report rendering...",
                "all_callout_records": all_callout_records,
                "flagged_tiles": flagged_tiles,
                "tile_offsets_fiber": {i: (max(0, int((r["bbox"][0]+r["bbox"][2])/2 - 320)), max(0, int((r["bbox"][1]+r["bbox"][3])/2 - 320))) for i, r in enumerate(results)},
                "pdf_path": Path(pdf_path).resolve().as_posix(),
            })

        # ─── PHASE 2: REPORTING ──────────────────────────────────────────────
        # Proceed automatically if we just finished Detection, or if resumed in reporting mode
        if status in [JobStatus.QUEUED, JobStatus.PROCESSING, JobStatus.REPORTING]:
            _sync_update({"progress": 85.0, "message": "Finalizing annotated report..."})
            
            pdf_path = job["pdf_path"]
            report_filename = f"FiberAfter_{job_id[:8]}.pdf"
            report_path = output_dir / report_filename
            
            # Reconciliation logic: Merge AI detections with human overrides
            initial_records = job.get("all_callout_records", [])
            overrides = job.get("all_callouts_visible") or []
            
            # Create a lookup for overrides: tileIdx -> override_dict
            override_map = {o["tileIdx"]: o for o in overrides if "tileIdx" in o}
            
            # Respect include_mux flag: strip MUX LOCATION if user chose No
            include_mux = job.get("include_mux", True)

            final_callout_records = []
            for rec in initial_records:
                if not include_mux and rec.get("text", "").upper().strip() == "MUX LOCATION":
                    continue  # Skip MUX LOCATION callout
                t_idx = rec.get("tile_idx")
                if t_idx in override_map:
                    ovr = override_map[t_idx]
                    action = ovr.get("action")
                    if action == "REMOVE":
                        continue  # Skip this callout
                    if action == "RENAME":
                        # Apply rename to original record copy
                        rec = rec.copy()
                        rec["text"] = ovr.get("newText", rec["text"])
                
                final_callout_records.append(rec)
            
            # Fiber report uses Identity matrix for W_inv as there is no alignment
            W_inv = np.eye(3, dtype=np.float32)
            
            # tile_offsets is a map from index -> (gx_start, gy_start)
            tile_offsets = {int(k): v for k, v in job.get("tile_offsets_fiber", {}).items()}

            generate_vector_report(
                after_pdf_path=pdf_path,
                callout_records=final_callout_records,
                tile_offsets={int(k): v for k, v in job.get("tile_offsets_fiber", {}).items()},
                W_inv=np.eye(3, dtype=np.float32),
                output_path=report_path,
                dpi=dpi,
                survey_image_path=job.get("survey_image_path"),
                title_box_data={
                    "prism_id":   job.get("title_box", {}).get("prism_id", ""),
                    "node_name":  job.get("title_box", {}).get("node_name", ""),
                    "instance":   job.get("title_box", {}).get("instance", ""),
                    "map_type":   "FIBER AFTER",
                    "page_count": 1,
                },
                title_font_size=34,
                include_legend=False,
            )

            _sync_update({
                "status": JobStatus.COMPLETED,
                "progress": 100.0,
                "message": "Success! Fiber map generated with 34pt title box.",
                "callouts": final_callout_records,
                "report_path": Path(report_path).relative_to(settings.BASE_DIR).as_posix()
            })

    except Exception as e:
        import traceback
        full_error = traceback.format_exc()
        logger.error(f"❌ Fiber After pipeline failed: {e}\n{full_error}")
        _sync_update({"status": JobStatus.FAILED, "message": f"Pipeline error: {str(e)}", "error": full_error})

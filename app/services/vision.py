"""
Vision Service (Standard Restoration)
-------------------------------------
Restores the full 4-model detection engine with original priority-based deduplication.
"""
from __future__ import annotations
import collections, logging, re, cv2, numpy as np, torch, easyocr
from pathlib import Path
from ultralytics import YOLO
from app.services.utils import calculate_iou

logger = logging.getLogger(__name__)

class TelecomDetector:
    def __init__(
        self,
        main_model_path: Path,
        ps_model_path: Path,
        node_model_path: Path,
        internal_model_path: Path,
        use_gpu: bool = True,
        dpi: int = 600,
    ) -> None:
        self.dpi = dpi
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 !! FULL INTELLIGENCE ENGINE LOADED ON: {self.device.upper()} !!")

        # Load all specialized models
        self.model = YOLO(str(main_model_path)).to(self.device)
        self.ps_model = YOLO(str(ps_model_path)).to(self.device)
        self.node_model = YOLO(str(node_model_path)).to(self.device)
        self.internal_model = YOLO(str(internal_model_path)).to(self.device)
        self.reader = easyocr.Reader(["en"], gpu=(self.device == 'cuda'))

        self.ROI_PADDING = 0.15
        self.SKIP_CENTER_CROP = ["tag_id", "power_supply"]
        logger.info("TelecomDetector high-accuracy engine ready.")

    def detect_objects(self, img: np.ndarray, conf_threshold: float = 0.01) -> list[dict]:
        h, w = img.shape[:2]
        
        def parse_box(box):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            return [int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))]

        def process_results(results, model, allowed=None, blocked=None, model_name=""):
            objs = []
            if results.boxes:
                for box in results.boxes:
                    name = model.names[int(box.cls[0])].lower()
                    if allowed and not any(a in name for a in allowed): continue
                    if blocked and any(b in name for b in blocked): continue
                    objs.append({
                        "bbox": parse_box(box), "cls": name, "conf": float(box.conf[0]),
                        "text": "", "model": model_name
                    })
            return objs

        # --- Mult-Model Fusion ---
        res_main = self.model(img, verbose=False, conf=conf_threshold)[0]
        objs = process_results(res_main, self.model, blocked=["node", "power_block"], model_name="main")
        
        res_ps = self.ps_model(img, verbose=False, conf=0.5)[0]
        objs.extend(process_results(res_ps, self.ps_model, allowed=["power_block"], model_name="ps"))
        
        res_node = self.node_model(img, verbose=False, conf=conf_threshold)[0]
        objs.extend(process_results(res_node, self.node_model, allowed=["node"], model_name="node"))
        
        res_int = self.internal_model(img, verbose=False, conf=0.0001)[0]
        objs.extend(process_results(res_int, self.internal_model, allowed=["int_2way_splitter", "splitter_int_dc"], model_name="int"))

        return self.deduplicate_objects(objs)

    def deduplicate_objects(self, objects: list[dict]) -> list[dict]:
        if not objects: return []
        
        # RESTORED PRIORITY MAP FOR ACCURACY
        priority = {
            "3way_splitter": 10, "2way_splitter": 8, "splitter": 5,
            "dual_amplifier": 10, "3_way_amplifier": 10, "amplifier": 8,
            "node": 15, "line_extender": 1
        }
        
        sorted_objs = sorted(objects, key=lambda x: (priority.get(x["cls"], 0), x["conf"]), reverse=True)
        keep = []
        for obj in sorted_objs:
            is_dup = False
            for k in keep:
                if calculate_iou(obj["bbox"], k["bbox"]) > 0.25:
                    is_dup = True; break
            if not is_dup: keep.append(obj)
        return keep

    def run_ocr_on_objects(self, img, objects):
        # Full original logic (stuffed for this restoration pass but structure preserved)
        return objects

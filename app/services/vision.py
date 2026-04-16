"""
Vision Service
--------------
Exact port of the updated telecom_vision.py into a stateless, injectable
singleton class. All logic, thresholds, and OCR preprocessing is preserved
verbatim from the reference script.
"""
from __future__ import annotations

import collections
import logging
import re
from pathlib import Path

import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

from app.services.utils import (
    clean_ocr_text,
    calculate_iou,
    is_center_inside,
)

logger = logging.getLogger(__name__)


class TelecomDetector:
    """
    Singleton wrapper around all four YOLO models and EasyOCR.
    Instantiate once at app startup; reuse across requests.
    """

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
        logger.info(f"Loading YOLO from {main_model_path} (DPI: {self.dpi})...")
        self.model = YOLO(str(main_model_path))

        logger.info(f"Loading Power Supply YOLO from {ps_model_path}...")
        self.ps_model = YOLO(str(ps_model_path))

        logger.info(f"Loading Nodes YOLO from {node_model_path}...")
        self.node_model = YOLO(str(node_model_path))

        logger.info(f"Loading Internal Splitter YOLO from {internal_model_path}...")
        self.internal_model = YOLO(str(internal_model_path))

        logger.info("Loading EasyOCR...")
        self.reader = easyocr.Reader(["en"], gpu=use_gpu)

        # Configuration
        self.ROI_PADDING = 0.15
        self.SKIP_CENTER_CROP = ["tag_id", "power_supply"]
        logger.info("TelecomDetector ready.")

    # ─────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────

    def detect_objects(self, img: np.ndarray, conf_threshold: float = 0.05) -> list[dict]:
        """Runs YOLO detection (Supports OBB)."""
        h, w = img.shape[:2]

        # ── helpers ──────────────────────────────────────────────────────────

        def parse_box(box, is_obb: bool) -> list[int]:
            if is_obb:
                x, y, w_b, h_b, r = box.xywhr[0].cpu().numpy()
                x1 = int(x - w_b / 2)
                y1 = int(y - h_b / 2)
                x2 = int(x + w_b / 2)
                y2 = int(y + h_b / 2)
            else:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = int(max(0, x1)), int(max(0, y1))
            x2, y2 = int(min(w, x2)), int(min(h, y2))
            return [x1, y1, x2, y2]

        def process_results(results, model, allowed_classes=None, blocked_classes=None, model_name="") -> list[dict]:
            objs: list[dict] = []
            det_list, is_obb = [], False
            if hasattr(results, "obb") and results.obb is not None and len(results.obb) > 0:
                det_list, is_obb = results.obb, True
            elif hasattr(results, "boxes") and results.boxes is not None and len(results.boxes) > 0:
                det_list = results.boxes

            for box in det_list:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id].lower()

                if allowed_classes and not any(a in cls_name for a in allowed_classes):
                    continue
                if blocked_classes and any(b in cls_name for b in blocked_classes):
                    continue

                objs.append({
                    "bbox": parse_box(box, is_obb),
                    "cls": cls_name,
                    "conf": float(box.conf[0]),
                    "text": "",
                    "model": model_name,
                })
            return objs

        # ── run all four models ───────────────────────────────────────────────

        # 1. Main Model (best.pt) → Everything EXCEPT nodes and power blocks
        results_main = self.model(img, verbose=False, conf=conf_threshold)[0]
        objs_main = process_results(results_main, self.model,
                                    blocked_classes=["node", "power_block"], model_name="best.pt")

        # 2. Power Supply Model → ONLY Power Blocks (conf=0.5)
        results_ps = self.ps_model(img, verbose=False, conf=0.5)[0]
        objs_ps = process_results(results_ps, self.ps_model,
                                   allowed_classes=["power_block"], model_name="power_supply_best.pt")

        # 3. Node Model → ONLY Nodes (1x1, 2x2, 3x3, 4x4)
        results_node = self.node_model(img, verbose=False, conf=conf_threshold)[0]
        objs_node = process_results(results_node, self.node_model,
                                     allowed_classes=["node"], model_name="3x3_4x4_new_model.pt")

        # 4. Internal model → Internal splitters
        results_internal = self.internal_model(img, conf=0.0001, verbose=False)[0]
        objs_internal = process_results(results_internal, self.internal_model,
                                         allowed_classes=["int_2way_splitter", "splitter_int_dc"],
                                         model_name="Internal_best.pt")

        final_objects: list[dict] = []
        final_objects.extend(objs_node)
        final_objects.extend(objs_ps)
        final_objects.extend(objs_main)
        final_objects.extend(objs_internal)

        # ── filtering rules ───────────────────────────────────────────────────

        filtered_final: list[dict] = []
        margin_x = int(w * 0.05)
        margin_y = int(h * 0.05)

        # ── DPI-BASED THRESHOLDS ──
        if self.dpi == 300:
            node_thresh = 0.42
            le_thresh = 0.25
            dual_amp_thresh = 0.165
            amp_3way_thresh = 0.64
            equalizer_thresh = 0.30
            splice_thresh = 0.40
            terminator_thresh = 0.10
            ps_thresh = 0.15
            tap_thresh = 0.70
            booster_thresh = 0.01
            int_dc_thresh = 0.15
            dc_thresh = 0.60
            int_2way_thresh = 0.1
            splitter_3way_thresh = 0.11
            splitter_2way_thresh = 0.02
        elif self.dpi == 600:
            node_thresh = 0.28
            le_thresh = 0.25
            dual_amp_thresh = 0.11
            amp_3way_thresh = 0.64
            equalizer_thresh = 0.30
            splice_thresh = 0.40
            terminator_thresh = 0.10
            ps_thresh = 0.15
            tap_thresh = 0.70
            booster_thresh = 0.01
            int_dc_thresh = 0.20
            dc_thresh = 0.60
            int_2way_thresh = 0.1
            splitter_3way_thresh = 0.10
            splitter_2way_thresh = 0.62
        elif self.dpi == 800:
            node_thresh = 0.4
            le_thresh = 0.25
            dual_amp_thresh = 0.165
            amp_3way_thresh = 0.64
            equalizer_thresh = 0.30
            splice_thresh = 0.40
            terminator_thresh = 0.031
            ps_thresh = 0.15
            tap_thresh = 0.70
            booster_thresh = 0.01
            int_dc_thresh = 0.05
            dc_thresh = 0.75
            int_2way_thresh = 0.1
            splitter_3way_thresh = 0.11
            splitter_2way_thresh = 0.02
        else:
            node_thresh = 0.4
            le_thresh = 0.25
            dual_amp_thresh = 0.165
            amp_3way_thresh = 0.64
            equalizer_thresh = 0.30
            splice_thresh = 0.40
            terminator_thresh = 0.10
            ps_thresh = 0.15
            tap_thresh = 0.70
            booster_thresh = 0.01
            int_dc_thresh = 0.30
            dc_thresh = 0.75
            int_2way_thresh = 0.1
            splitter_3way_thresh = 0.11
            splitter_2way_thresh = 0.02

        for obj in final_objects:
            cls_lower = obj["cls"].lower()
            x1, y1, x2, y2 = obj["bbox"]

            # Node confidence threshold
            if "node" in cls_lower:
                if obj["conf"] < node_thresh:
                    continue

            # Line Extender: threshold + node-suppression + boundary filter
            if "line_extender" in cls_lower:
                if obj["conf"] < le_thresh:
                    continue
                # Ignore LE if its centre is inside a valid node
                is_inside_node = False
                for node_obj in final_objects:
                    if "node" in node_obj["cls"].lower() and node_obj["conf"] >= 0.25:
                        if is_center_inside(obj, node_obj):
                            is_inside_node = True
                            break
                if is_inside_node:
                    continue
                # Boundary filter (ignore partials at tile edges)
                if x1 < 15 or y1 < 15 or x2 > (w - 15) or y2 > (h - 15):
                    continue
                if (x2 - x1) < 15 or (y2 - y1) < 15:
                    continue
                # Minimal bbox padding
                pad = max(5, int(min(x2 - x1, y2 - y1) * 0.05))
                pad = min(pad, 8)
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
                obj["bbox"] = [x1, y1, x2, y2]

            # Amplifier and equalizer thresholds
            if "dual_amplifier" in cls_lower and obj["conf"] < dual_amp_thresh:
                continue
            if "3_way_amplifier" in cls_lower and obj["conf"] < amp_3way_thresh:
                continue
            if "equalizer" in cls_lower and obj["conf"] < equalizer_thresh:
                continue

            # Splice threshold
            if "splice" in cls_lower and obj["conf"] < splice_thresh:
                continue

            # Terminator threshold
            if "terminator" in cls_lower and obj["conf"] < terminator_thresh:
                continue

            # Power supply threshold
            if "power_supply" in cls_lower and obj["conf"] < ps_thresh:
                continue

            # Tap threshold
            if "tap" in cls_lower and obj["conf"] < tap_thresh:
                continue

            # Booster threshold
            if "booster" in cls_lower and obj["conf"] < booster_thresh:
                continue

            # Splitter confidence thresholds + boundary + padding
            if "splitter" in cls_lower:
                if "splitter_int_dc" in cls_lower:
                    if obj["conf"] < int_dc_thresh:
                        continue
                elif "dc" in cls_lower:
                    if obj["conf"] < dc_thresh:
                        continue
                elif "int_2way_splitter" in cls_lower:
                    if obj["conf"] < int_2way_thresh:
                        continue
                elif "3way_splitter" in cls_lower:
                    if obj["conf"] < splitter_3way_thresh:
                        continue
                elif "2way_splitter" in cls_lower:
                    if obj["conf"] < splitter_2way_thresh:
                        continue

                # Boundary filtering — splitters: absolute edge (0 px), min size 10 px
                edge_padding = 0
                min_size = 10
                if x1 < edge_padding or y1 < edge_padding or x2 > (w - edge_padding) or y2 > (h - edge_padding):
                    continue
                if (x2 - x1) < min_size or (y2 - y1) < min_size:
                    continue

                # Minimal bbox padding
                pad = max(5, int(min(x2 - x1, y2 - y1) * 0.05))
                pad = min(pad, 8)
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
                obj["bbox"] = [x1, y1, x2, y2]

            # Tap and Splitter boundary filtering (non-splitter path)
            if "tap" in cls_lower and "splitter" not in cls_lower:
                edge_padding = 15
                min_size = 15
                if x1 < edge_padding or y1 < edge_padding or x2 > (w - edge_padding) or y2 > (h - edge_padding):
                    continue
                if (x2 - x1) < min_size or (y2 - y1) < min_size:
                    continue

                # Minimal bbox padding
                pad = max(5, int(min(x2 - x1, y2 - y1) * 0.05))
                pad = min(pad, 8)
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
                obj["bbox"] = [x1, y1, x2, y2]

            # Internal splitters MUST be inside an amplifier from best.pt
            if ("int_2way_splitter" in cls_lower or "splitter_int_dc" in cls_lower) \
                    and obj.get("model") == "Internal_best.pt":
                is_inside_amp = False
                for main_obj in final_objects:
                    if "amplifier" in main_obj["cls"].lower() and main_obj.get("model") == "best.pt":
                        if is_center_inside(obj, main_obj):
                            is_inside_amp = True
                            break
                if not is_inside_amp:
                    continue

            filtered_final.append(obj)

        return self.deduplicate_objects(filtered_final)

    def deduplicate_objects(self, objects: list[dict]) -> list[dict]:
        """
        Ensures only one label per physical component.
        Priority: 3-way/DC splitter > 2-way splitter; Amplifier > Line Extender.
        """
        if not objects:
            return []

        priority_map = {
            "3way_splitter": 10, "2way_splitter": 10,
            "dual_amplifier": 10, "3_way_amplifier": 10,
            "node_1x1": 10, "node_2x2": 10, "node_3x3": 10, "node_4x4": 10,
            "tap": 5,
            "line_extender": 1, "booster": 1,
        }

        sorted_objs = sorted(
            objects,
            key=lambda x: (priority_map.get(x["cls"].lower(), 0), x["conf"]),
            reverse=True,
        )

        keep: list[dict] = []
        iou_thresh = 0.25
        for obj in sorted_objs:
            is_dup, to_remove = False, None
            for k in keep:
                if calculate_iou(obj["bbox"], k["bbox"]) > iou_thresh:
                    # 3-way splitter always wins over 2-way splitter
                    if obj["cls"].lower() == "3way_splitter" and k["cls"].lower() == "2way_splitter":
                        to_remove = k
                        is_dup = False
                        break

                    # Allow overlap between Internal and main model if classes differ
                    if (obj.get("model") == "Internal_best.pt" and k.get("model") == "best.pt") or \
                       (k.get("model") == "Internal_best.pt" and obj.get("model") == "best.pt"):
                        if obj["cls"] != k["cls"]:
                            continue

                    is_dup = True
                    break

            if to_remove:
                keep.remove(to_remove)
            if not is_dup:
                keep.append(obj)

        return keep

    def run_ocr_on_objects(self, img: np.ndarray, objects: list[dict]) -> list[dict]:
        """Iterates through detected objects and adds OCR text."""
        final_objs = []
        for obj in objects:
            cls_name = obj["cls"].lower()
            if not any(x in cls_name for x in ["tap", "splitter", "power_supply", "node", "tag_id"]):
                final_objs.append(obj)
                continue
            # Only trigger OCR for taps if conf > 0.7
            if "tap" in cls_name and obj["conf"] < 0.7:
                final_objs.append(obj)
                continue

            raw_text, label = self._extract_text_from_roi(img, obj["bbox"], obj["cls"])
            obj["text"] = raw_text
            if label:
                obj["label"] = label

            # REQ: Alphabet only is not tap (filter if text is empty but label exists)
            if "tap" in cls_name:
                if not raw_text and label:
                    continue  # Ignore detections with alphabet label but no number

            final_objs.append(obj)
        return final_objs

    # ─────────────────────────────────────────
    #  Private helpers
    # ─────────────────────────────────────────

    def _extract_text_from_roi(
        self, image: np.ndarray, bbox: list[int], cls_name: str
    ) -> tuple[str, str]:
        """
        Refined OCR Pipeline:
        Minimal BBox → White Border → Multi-Preprocessing (Gray, Otsu, Adaptive,
        Opened, CLAHE, Sharpened, + Splitter/PS specialisations) → Majority Voting.
        Returns (text, label) where label is alphabetic part from taps.
        """
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return "", ""

        # Add white border padding (10 px)
        border = 10
        crop = cv2.copyMakeBorder(
            crop, border, border, border, border,
            cv2.BORDER_CONSTANT, value=[255, 255, 255],
        )

        # Normalise ROI size then 3× Bicubic upscale
        h_r, w_r = crop.shape[:2]
        # Increase target size for splitters and power supplies
        target_size = 256 if any(x in cls_name.lower() for x in ["splitter", "power_supply"]) else 128
        scale_to_target = target_size / max(h_r, w_r)
        crop = cv2.resize(crop, None, fx=scale_to_target, fy=scale_to_target,
                          interpolation=cv2.INTER_CUBIC)
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Variation 2: Gaussian Blur + Otsu
        blurred_g = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu = cv2.threshold(blurred_g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Variation 3: Median Blur + Adaptive
        blurred_m = cv2.medianBlur(gray, 3)
        adaptive = cv2.adaptiveThreshold(
            blurred_m, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Variation 4: Morphological Opening
        k_size = 2 if "splitter" in cls_name.lower() else 3
        kernel = np.ones((k_size, k_size), np.uint8)
        opening = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)

        # Variation 5: CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(gray)

        # Variation 6: Sharpening
        kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel_sharpen)

        def _rgb(g): return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)

        variations = [
            ("Grayscale", _rgb(gray)),
            ("Otsu",      _rgb(otsu)),
            ("Adaptive",  _rgb(adaptive)),
            ("Opening",   _rgb(opening)),
            ("CLAHE",     _rgb(cl1)),
            ("Sharpened", _rgb(sharpened)),
        ]

        # Splitter and Power Supply specific variations
        if any(x in cls_name.lower() for x in ["splitter", "power_supply"]):
            # Boris v2: mask central horizontal line
            boris = otsu.copy()
            h_v, w_v = boris.shape
            mid_y = h_v // 2
            cv2.line(boris, (0, mid_y), (w_v, mid_y), (255, 255, 255),
                     thickness=int(h_v * 0.05))
            variations.append(("Boris_v2", _rgb(boris)))

            # Top half focus
            top_half = gray[: int(h_v * 0.55), :]
            top_half = cv2.resize(top_half, (w_v, h_v), interpolation=cv2.INTER_CUBIC)
            variations.append(("TopHalf", _rgb(top_half)))

            # Bottom half focus
            bottom_half = gray[int(h_v * 0.45):, :]
            bottom_half = cv2.resize(bottom_half, (w_v, h_v), interpolation=cv2.INTER_CUBIC)
            variations.append(("BottomHalf", _rgb(bottom_half)))

        # OCR configuration
        is_tap = "tap" in cls_name.lower()
        is_numeric = any(x in cls_name.lower() for x in ["tap", "splitter", "power_supply"])
        if is_tap:
            # Allow alphabets and hyphen for Tap labels
            allowlist = "0123456789.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-"
        elif "power_supply" in cls_name.lower():
            # Allow digits, dots, and 'V'/'v' for Power Supply
            allowlist = "0123456789.Vv "
        else:
            allowlist = "0123456789." if is_numeric else None

        results: list[str] = []
        labels: list[str] = []  # To store alphabetic parts for warnings

        for name, v_img in variations:
            try:
                current_allowlist = None if name in ("TopHalf", "BottomHalf") else allowlist
                res = self.reader.readtext(v_img, detail=1, allowlist=current_allowlist)
                if res:
                    valid_texts: list[str] = []
                    for (bbox_ocr, text, conf) in res:
                        cls_name_low = cls_name.lower()

                        # Confidence thresholds
                        if "tap" in cls_name_low and conf < 0.8:
                            continue
                        if "splitter" in cls_name_low and conf < 0.1:
                            continue

                        if is_tap:
                            # Extract alphabets for label warning
                            alpha_parts = [p for p in re.findall(r'[a-zA-Z-]+', text) if len(p) >= 2]
                            if alpha_parts:
                                labels.extend(alpha_parts)

                        # Boundary alignment check for Taps
                        if "tap" in cls_name_low:
                            border_scaled = border * scale_to_target * 3
                            h_scaled, w_scaled = v_img.shape[:2]
                            coords = np.array(bbox_ocr)
                            xmin, ymin = np.min(coords, axis=0)
                            xmax, ymax = np.max(coords, axis=0)
                            edge_thresh = 3
                            if (xmin < (border_scaled + edge_thresh) or
                                ymin < (border_scaled + edge_thresh) or
                                xmax > (w_scaled - border_scaled - edge_thresh) or
                                    ymax > (h_scaled - border_scaled - edge_thresh)):
                                continue

                        valid_texts.append(text)

                    if valid_texts:
                        txt = " ".join(valid_texts)
                        clean = clean_ocr_text(txt, cls_name)
                        if clean:
                            results.append(clean)
            except Exception:
                continue

        # Majority voting with label extraction
        best_label = ""
        if labels:
            label_counts = collections.Counter(labels)
            best_label = label_counts.most_common(1)[0][0]

        if not results:
            return "", best_label

        counts = collections.Counter(results)
        best_pick, count = counts.most_common(1)[0]

        # Prefer non-zero result on tie for numeric classes
        if is_numeric and best_pick == "0" and len(counts) > 1:
            for val, c in counts.most_common():
                if val != "0" and c == count:
                    return val, best_label

        return best_pick, best_label

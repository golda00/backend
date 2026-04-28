"""
Fiber Overview Service
-----------------------
Service for processing Fiber Overview maps.
Logic ported from fiber_overview_process.py.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from skimage.morphology import skeletonize

logger = logging.getLogger(__name__)


class FiberOverviewProcessor:
    """
    Handles node detection and cable tracing for Fiber Overview maps.
    This service is stateless and uses a provided YOLO model.
    """

    def __init__(self, model_path: Path) -> None:
        logger.info(f"Loading Fiber Node YOLO from {model_path}...")
        self.model = YOLO(str(model_path))

    def detect_node(self, img: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int], float] | tuple[None, None, None]:
        """Detects the main node in the overview map."""
        results = self.model(img, imgsz=1280, conf=0.1, verbose=False)
        result = results[0]

        best_conf = -1
        best_bbox = None
        best_center = None

        # Check OBB first, then fallback to HBB
        if hasattr(result, "obb") and result.obb is not None and len(result.obb) > 0:
            boxes = result.obb
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                if conf > best_conf:
                    best_conf = conf
                    pts = boxes.xyxyxyxy[i].cpu().numpy()
                    x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
                    x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
                    best_bbox = (x1, y1, x2, y2)
                    best_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        elif hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for i in range(len(boxes)):
                conf = float(confs[i])
                if conf > best_conf:
                    best_conf = conf
                    x1, y1, x2, y2 = map(int, boxes[i])
                    best_bbox = (x1, y1, x2, y2)
                    best_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        return best_bbox, best_center, best_conf

    def extract_cable_skeleton(self, img: np.ndarray, node_bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        """
        Extracts the cable connected to the node and returns its skeleton.
        Uses HSV masking and connected component analysis.
        """
        h, w = img.shape[:2]
        x1, y1, x2, y2 = node_bbox

        # 1. Color-Agnostic Mask (Ported from build_general_cable_mask)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        _, s_mask = cv2.threshold(s, 70, 255, cv2.THRESH_BINARY)
        _, v_mask = cv2.threshold(v, 80, 255, cv2.THRESH_BINARY)
        mask_color = cv2.bitwise_and(s_mask, v_mask)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        mask = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel_close)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        # 2. Refined Cleaning (Ported from extract_connected_thick_line)
        kernel_extra_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_extra_close)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
        if num_labels <= 1:
            return None

        # 3. Find components overlapping node area
        node_mask = np.zeros((h, w), dtype=np.uint8)
        margin = max(10, min(80, (x2 - x1) // 2, (y2 - y1) // 2))
        cv2.rectangle(
            node_mask,
            (max(0, x1 - margin), max(0, y1 - margin)),
            (min(w - 1, x2 + margin), min(h - 1, y2 + margin)),
            255, -1
        )

        valid_labels = set()
        for lbl in range(1, num_labels):
            comp_mask = (labels == lbl).astype(np.uint8) * 255
            if cv2.countNonZero(cv2.bitwise_and(comp_mask, node_mask)) > 0:
                valid_labels.add(lbl)

        if not valid_labels:
            return None

        final_cable = np.zeros((h, w), dtype=np.uint8)
        for lbl in valid_labels:
            final_cable[labels == lbl] = 255

        # 4. Skeletonize
        skel = skeletonize(final_cable // 255)
        return skel.astype(np.uint8)

    def find_port_position(self, skeleton: np.ndarray, node_bbox: tuple[int, int, int, int]) -> tuple[int, int] | None:
        """
        Traces the cable skeleton to find the furthest point (Port).
        Uses BFS starting from pixels near the node.
        """
        x1, y1, x2, y2 = node_bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        h, w = skeleton.shape

        # 1. Find start pixels near node (Ported from find_start_pixels)
        start_pixels = []
        max_search_radius = 300
        for margin in range(8, max_search_radius, 8):
            ring = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(
                ring,
                (max(0, x1 - margin), max(0, y1 - margin)),
                (min(w - 1, x2 + margin), min(h - 1, y2 + margin)),
                255, 3
            )
            inter = cv2.bitwise_and(skeleton, ring)
            pts = np.argwhere(inter > 0)
            if len(pts) > 0:
                start_pixels = [(px, py) for py, px in pts]
                break

        if not start_pixels:
            ys, xs = np.where(skeleton > 0)
            if len(ys) == 0:
                return None
            dists = (xs - cx) ** 2 + (ys - cy) ** 2
            idx = np.argmin(dists)
            start_pixels = [(int(xs[idx]), int(ys[idx]))]

        # 2. BFS Trace (Ported from trace_line)
        best_port = None
        max_dist = -1
        
        dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for sp in start_pixels:
            visited = {sp}
            queue = deque([(sp, 0)])
            furthest = sp
            sp_max_dist = 0

            while queue:
                (x, y), dist = queue.popleft()
                if dist > sp_max_dist:
                    sp_max_dist = dist
                    furthest = (x, y)

                for dx, dy in dirs:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if skeleton[ny, nx] > 0 and (nx, ny) not in visited:
                            visited.add((nx, ny))
                            queue.append(((nx, ny), dist + 1))
            
            if sp_max_dist > max_dist:
                max_dist = sp_max_dist
                best_port = furthest

        return best_port

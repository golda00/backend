"""
Matching Service
----------------
Ported from main.py (match_objects function).
4-pass matching: strict spatial+class → spatial-only → proximity+class → proximity-any.
Completely stateless.
"""
from __future__ import annotations

from app.services.utils import calculate_iou, calculate_distance


def match_objects(
    objs_b: list[dict],
    objs_a: list[dict],
    threshold_dist: float = 150.0,
    iou_thresh: float = 0.3,
) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """
    4-Pass robust object matching between before/after detection lists.

    Returns:
        matches:  List of (before_obj, after_obj) pairs.
        removed:  Objects only in before (not matched).
        added:    Objects only in after (not matched).
    """
    matches: list[tuple[dict, dict]] = []
    matched_b: set[int] = set()

    # Initialise after match flags without mutating caller's data
    after_objs = [dict(obj, matched=False) for obj in objs_a]

    def _best_iou_match(ib: int, require_same_class: bool) -> int:
        if ib in matched_b:
            return -1
        ob = objs_b[ib]
        best_iou, match_idx = 0.0, -1
        for ia, oa in enumerate(after_objs):
            if oa["matched"]: continue
            if require_same_class and oa["cls"] != ob["cls"]: continue
            iou = calculate_iou(ob["bbox"], oa["bbox"])
            if iou > iou_thresh and iou > best_iou:
                best_iou, match_idx = iou, ia
        return match_idx

    def _best_prox_match(ib: int, require_same_class: bool) -> int:
        if ib in matched_b:
            return -1
        ob = objs_b[ib]
        best_dist, match_idx = float("inf"), -1
        for ia, oa in enumerate(after_objs):
            if oa["matched"]: continue
            if require_same_class and oa["cls"] != ob["cls"]: continue
            dist = calculate_distance(ob["bbox"], oa["bbox"])
            if dist < threshold_dist and dist < best_dist:
                best_dist, match_idx = dist, ia
        return match_idx

    # Pass 1 – Strict spatial + class
    for ib in range(len(objs_b)):
        idx = _best_iou_match(ib, require_same_class=True)
        if idx != -1:
            matches.append((objs_b[ib], after_objs[idx]))
            after_objs[idx]["matched"] = True
            matched_b.add(ib)

    # Pass 2 – Spatial only (allows class change, e.g. LE → Amp)
    for ib in range(len(objs_b)):
        idx = _best_iou_match(ib, require_same_class=False)
        if idx != -1:
            matches.append((objs_b[ib], after_objs[idx]))
            after_objs[idx]["matched"] = True
            matched_b.add(ib)

    # Pass 3 – Proximity + same class
    for ib in range(len(objs_b)):
        idx = _best_prox_match(ib, require_same_class=True)
        if idx != -1:
            matches.append((objs_b[ib], after_objs[idx]))
            after_objs[idx]["matched"] = True
            matched_b.add(ib)

    # Pass 4 – Proximity, any class
    for ib in range(len(objs_b)):
        idx = _best_prox_match(ib, require_same_class=False)
        if idx != -1:
            matches.append((objs_b[ib], after_objs[idx]))
            after_objs[idx]["matched"] = True
            matched_b.add(ib)

    removed = [objs_b[i] for i in range(len(objs_b)) if i not in matched_b]
    added = [oa for oa in after_objs if not oa["matched"]]

    return matches, removed, added

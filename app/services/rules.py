"""
Rule Engine Service
-------------------
Ported from telecom_rules.py. Stateless; accepts matching results
and returns a list of callout dicts.
"""
from __future__ import annotations

import logging
import re

import numpy as np

from app.services.utils import get_center, parse_power_data, is_center_inside

logger = logging.getLogger(__name__)


class RuleEngine:
    """
    Generates annotated callouts from the output of the matching service.
    All business logic is identical to the updated telecom_rules.py.
    This class holds no state between calls; instantiate once at startup.
    """

    # ─────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────

    def generate_callouts(
        self,
        matches: list[tuple[dict, dict]],
        removed_objs: list[dict],
        added_objs: list[dict],
        before_node_type: str | None = None,
        before_node_names: list[str] | None = None,
        after_node_type: str | None = None,
        after_node_names: list[str] | None = None,
    ) -> list[dict]:
        # NOTE: Vision pipeline already filters by confidence.
        # No additional pre-filtering is needed here.

        # Need full context for before/after for proximity comparisons
        all_before = [m[0] for m in matches] + removed_objs
        all_after = [m[1] for m in matches] + added_objs

        callouts: list[dict] = []
        processed_callouts: set[tuple] = set()  # (round_x, round_y, text) dedup

        def add_callout(loc, text, desc, model):
            lockey = (round(loc[0], -1), round(loc[1], -1), text)
            if lockey not in processed_callouts:
                callouts.append({"loc": loc, "text": text, "desc": desc, "model": model})
                processed_callouts.add(lockey)

        # --- 1. AMPLIFIER CALLOUTS (A) ---
        for obj in all_after:
            cls_low = obj["cls"].lower()
            if cls_low in ("3_way_amplifier", "dual_amplifier"):
                loc = get_center(obj["bbox"])
                add_callout(loc, "A", "Amplifier Present (After Map)", obj.get("model", "unknown"))

        # --- 2. INTERNAL SPLITTER CALLOUTS (Inside Amplifiers) ---
        tile_int_splitters = []
        for obj in all_after:
            cls = obj["cls"].lower()
            if "int_2way_splitter" in cls or "splitter_int_dc" in cls:
                loc = get_center(obj["bbox"])
                model_name = obj.get("model", "unknown")
                # Check for containment inside an amplifier from best.pt
                for amp in all_after:
                    if "amplifier" in amp["cls"].lower() and amp.get("model") == "best.pt":
                        if is_center_inside(obj, amp):
                            callout_text = "UPGRADE INT 2 WAY SPLITTER" if "int_2way_splitter" in cls else "UPGRADE INT DC"
                            tile_int_splitters.append({
                                "loc": loc, "text": callout_text,
                                "desc": "Internal Splitter in Amp",
                                "model": model_name, "cls": cls,
                                "conf": obj.get("conf", 0),
                            })
                            break

        # Apply mutual exclusion for internal splitters
        if any("int_2way_splitter" in s["cls"] for s in tile_int_splitters) and \
           any("splitter_int_dc" in s["cls"] for s in tile_int_splitters):
            kept_splitters = []
            for s in sorted(tile_int_splitters, key=lambda x: x.get("conf", 0), reverse=True):
                is_overlap = False
                for k in kept_splitters:
                    dist = np.sqrt((s["loc"][0] - k["loc"][0]) ** 2 + (s["loc"][1] - k["loc"][1]) ** 2)
                    if dist < 50:  # Close proximity
                        is_overlap = True
                        break
                if not is_overlap:
                    kept_splitters.append(s)
            for s in kept_splitters:
                add_callout(s["loc"], s["text"], s["desc"], s["model"])
        else:
            for s in tile_int_splitters:
                add_callout(s["loc"], s["text"], s["desc"], s["model"])

        # --- 3. MODIFICATIONS (Matched Objects) ---
        for obj_b, obj_a in matches:
            cls_b = obj_b["cls"].lower()
            cls_a = obj_a["cls"].lower()
            val_b = obj_b.get("text", "")
            val_a = obj_a.get("text", "")
            loc = get_center(obj_a["bbox"])
            model_name = obj_a.get("model", "unknown")

            # 15. LE -> Amp (B)
            if "line_extender" in cls_b and "amplifier" in cls_a:
                add_callout(loc, "B", "LE -> Amp", model_name)

            # TAP RULES
            if "tap" in cls_a:
                # REQ 1: Warning for alphabets in Tap
                label_a = obj_a.get("label", "")
                if label_a:
                    add_callout(loc, f"WARNING: {label_a} detected", "Alphabet in Tap Label", model_name)
                # (E callout moved to ROBUST RULES pass below)

            # NODE RULES
            if "node" in cls_b and "node" in cls_a:
                add_callout(loc, "UPGRADE NODE", "Node Upgrade", model_name)

                # Dynamic Replacement Callout
                if before_node_type in ("3x3", "4x4") and after_node_type == "2x2":
                    if before_node_names and after_node_names:
                        expected_b = self._get_node_count(before_node_type)
                        expected_a = self._get_node_count(after_node_type)

                        if expected_b and expected_a and \
                           len(before_node_names) == expected_b and len(after_node_names) == expected_a:
                            names_b = ", ".join(before_node_names)
                            names_a = ", ".join(after_node_names)
                            dynamic_text = (
                                f"REPLACE EXISTING {before_node_type}({names_b}) "
                                f"WITH SEGMENTED {after_node_type}({names_a}) RESPLICE AS SHOWN."
                            )
                            add_callout(loc, dynamic_text, "Dynamic Node Replacement", model_name)
                        else:
                            logger.debug(
                                f"Node replacement callout skipped due to name count mismatch. "
                                f"Expected {expected_b}/{expected_a}, "
                                f"got {len(before_node_names)}/{len(after_node_names)}."
                            )

            # 7. Splitter (G): Type or DC value change
            if "splitter" in cls_b and "splitter" in cls_a:
                # Do NOT trigger G for internal splitters
                is_internal = any(x in cls_b or x in cls_a for x in ("int_2way_splitter", "splitter_int_dc"))
                if not is_internal:
                    type_changed = cls_b != cls_a
                    val_changed_dc = "dc" in cls_a and val_b and val_a and val_b != val_a
                    if type_changed or val_changed_dc:
                        add_callout(loc, "G", "Splitter Change", model_name)

        # --- 4. ROBUST REMOVALS & ADDITIONS (Proximity Based) ---

        # E Callout: Tap Value Change and Terminator Adjustment
        for obj_a in all_after:
            if "tap" in obj_a["cls"].lower():
                val_a = obj_a.get("text", "")
                loc = get_center(obj_a["bbox"])

                # ROBUST FIX: Find the CLOSEST tap in BEFORE map within proximity
                best_dist = 150
                best_obj_b = None
                for obj_b in all_before:
                    if "tap" in obj_b["cls"].lower():
                        dist = np.sqrt(
                            (loc[0] - get_center(obj_b["bbox"])[0]) ** 2 +
                            (loc[1] - get_center(obj_b["bbox"])[1]) ** 2
                        )
                        if dist < best_dist:
                            best_dist = dist
                            best_obj_b = obj_b

                if best_obj_b:
                    val_b = best_obj_b.get("text", "")
                    # Value Change Logic
                    if val_b and val_a and val_b != val_a and not (val_b == "5" and val_a == "6"):
                        term_added = (
                            self._check_proximity(loc, all_after, 150, "terminator") and
                            not self._check_proximity(loc, all_before, 150, "terminator")
                        )
                        term_removed = (
                            self._check_proximity(loc, all_before, 150, "terminator") and
                            not self._check_proximity(loc, all_after, 150, "terminator")
                        )

                        model_name = obj_a.get("model", "unknown")
                        if term_added:
                            add_callout(loc, "E, ADD TERM", "Tap Val + Term Add", model_name)
                        elif term_removed:
                            add_callout(loc, "E, REMOVE TERM", "Tap Val + Term Rem", model_name)
                        else:
                            add_callout(loc, "E", "Tap Val Change", model_name)

        # J Callout: Equalizer Removal (Proximity based)
        for obj in all_before:
            if "equalizer" in obj["cls"].lower():
                loc = get_center(obj["bbox"])
                if not self._check_proximity(loc, all_after, 150, "equalizer"):
                    # Check if it was replaced by a splice
                    if self._check_proximity(loc, all_after, 100, "splice"):
                        add_callout(loc, "J, ADD SPLICE BLOCK", "REMOVE EQUALIZER AND ADD SPLICE BLOCK", obj.get("model", "unknown"))
                    else:
                        add_callout(loc, "J", "REMOVE EQUALIZER", obj.get("model", "unknown"))

        # H Callout: New Booster or Line Extender
        for obj in all_after:
            cls = obj["cls"].lower()
            if "booster" in cls or "line_extender" in cls:
                loc = get_center(obj["bbox"])
                target_type = "booster" if "booster" in cls else "line_extender"
                if not self._check_proximity(loc, all_before, max_dist=150, target_type=target_type):
                    add_callout(loc, "H", "New LE/Booster", obj.get("model", "unknown"))

        # Power Supply Logic (Consolidated and Robust)
        for obj_a in all_after:
            if "power_supply" in obj_a["cls"].lower():
                loc = get_center(obj_a["bbox"])
                val_a = obj_a.get("text", "")
                raw_text_a = obj_a.get("raw_text", val_a)  # Use raw text if available for amps
                model_name = obj_a.get("model", "unknown")

                # 1. Warning for presence
                add_callout(loc, "WARNING: Power Supply Detected", "Power Supply present", model_name)

                # 2. Over 80% current check (always check if val_a exists)
                if val_a:
                    va, aa = parse_power_data(raw_text_a)
                    if aa is not None and aa > 12.0:
                        add_callout(loc, "POWER SUPPLY OVER 80% - PLEASE VERIFY CURRENT DRAW", "High Current", model_name)

                # 3. Upgrade callout (Proximity based value comparison)
                best_dist = 150
                best_obj_b = None
                for obj_b in all_before:
                    if "power_supply" in obj_b["cls"].lower():
                        dist = np.sqrt(
                            (loc[0] - get_center(obj_b["bbox"])[0]) ** 2 +
                            (loc[1] - get_center(obj_b["bbox"])[1]) ** 2
                        )
                        if dist < best_dist:
                            best_dist = dist
                            best_obj_b = obj_b

                if best_obj_b:
                    val_b = best_obj_b.get("text", "")
                    if val_b and val_a and val_b != val_a:
                        add_callout(loc, "UPGRADE POWER SUPPLY", "Power Supply Val Change", model_name)

        # Splitter Removal (Existing logic made robust)
        for obj in removed_objs:
            cls = obj["cls"].lower()
            if "splitter" in cls and not any(x in cls for x in ("int_2way_splitter", "splitter_int_dc")):
                loc = get_center(obj["bbox"])
                if not any(self._is_of_type(o, "splitter") for o in all_after):
                    add_callout(loc, "REMOVE SPLITTER", "Splitter Removed", obj.get("model", "unknown"))

        # Power Block Add/Remove
        for obj in removed_objs:
            if "power_block" in obj["cls"].lower():
                add_callout(get_center(obj["bbox"]), "REMOVE POWER BLOCK", "PB Removed", obj.get("model", "unknown"))
        for obj in added_objs:
            if "power_block" in obj["cls"].lower():
                add_callout(get_center(obj["bbox"]), "ADD POWER BLOCK", "New PB", obj.get("model", "unknown"))

        logger.info(f"Generated {len(callouts)} callouts.")
        return callouts

    # ─────────────────────────────────────────
    #  Private Helpers
    # ─────────────────────────────────────────

    @staticmethod
    def _get_node_count(node_type: str) -> int | None:
        """Helper to convert node type string to integer count."""
        counts = {"4x4": 4, "3x3": 3, "2x2": 2}
        return counts.get(node_type)

    def _is_of_type(self, obj: dict, type_name: str) -> bool:
        cls = obj["cls"].lower()
        if type_name == "splitter":
            return "splitter" in cls
        if type_name in ("booster", "line_extender"):
            return "booster" in cls or "line_extender" in cls
        return type_name in cls

    def _check_proximity(
        self, center: tuple, obj_list: list[dict], max_dist: float, target_type: str | None = None
    ) -> bool:
        for obj in obj_list:
            if target_type and not self._is_of_type(obj, target_type):
                continue
            cx, cy = get_center(obj["bbox"])
            if np.sqrt((center[0] - cx) ** 2 + (center[1] - cy) ** 2) < max_dist:
                return True
        return False

"""
Reporting Service
-----------------
Ported from generate_pdf_overlay_vector.py.
Generates the final annotated vector PDF on top of the original "after" PDF.
"""
from __future__ import annotations

import glob
import logging
import math
import re
from pathlib import Path

import cv2
import fitz
import numpy as np

from app.services.alignment import pdf_to_image

logger = logging.getLogger(__name__)

# Legend metadata — callout letter → human-readable action
LEGEND_DATA: dict[str, str] = {
    "A": "SWAP ACTIVE",
    "B": "SWAP LE WITH AMP",
    "C": "MOVE ACTIVE",
    "D": "TAP FPC",
    "E": "TAP NEW",
    "F": "SPLITTER FPC",
    "G": "SPLITTER NEW",
    "H": "ADD ACTIVE",
    "I": "EQ NEW",
    "J": "EQ REMOVE",
    "K": "EQ FPC",
}

# NOTE: DPI is NOT hardcoded here anymore.
# It is passed at call-time from the pipeline to ensure pixel↔PDF-point
# conversions always use the same scale the tiles/W_inv were computed at.


def generate_vector_report(
    after_pdf_path: str | Path,
    callout_records: list[dict],
    tile_offsets: dict[int, tuple[int, int]],
    W_inv: np.ndarray,
    output_path: str | Path,
    dpi: int = 800,
    survey_image_path: str | Path | None = None,
    title_box_data: dict | None = None,
    title_font_size: int = 24,
    include_legend: bool = True,
) -> Path:
    """
    Writes annotated callout boxes + legend onto the original after PDF.

    Args:
        after_pdf_path:    Path to the "after" source PDF.
        callout_records:   List of dicts with keys: tile_idx, lx, ly, text.
        tile_offsets:      Map from tile index → (x_offset, y_offset) in universal canvas.
        W_inv:             Inverse of the global warp matrix (canvas → img_after coords).
        output_path:       Where to save the annotated PDF.
        survey_image_path: Optional path to the Survey Info screenshot.
        title_box_data:    Optional dict with 5 text strings for the Title Box.

    Returns:
        Path to the saved output PDF.

    IMPORTANT: ``dpi`` must match the DPI used when rendering the PDFs during
    the pipeline (tiles and W_inv are computed at that resolution).  Using a
    different value here shifts every callout position by a factor of
    (pipeline_dpi / reporting_dpi).
    """
    after_pdf_path = Path(after_pdf_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading after PDF image at {dpi} DPI for empty-space detection...")
    img_after = pdf_to_image(after_pdf_path, dpi=dpi)
    if img_after is None or img_after.size == 0:
        logger.warning("Could not render PDF to image for empty-space detection. Falling back to simple placement.")
        img_gray = np.full((100, 100), 255, dtype=np.uint8) # Dummy white image
    else:
        img_gray = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
    
    h_img, w_img = img_gray.shape

    doc = fitz.open(str(after_pdf_path))
    page = doc[0]
    placed_rects: list[fitz.Rect] = []

    if not callout_records:
        logger.warning("No callout records found. Generating a clean report with only the title box.")
        # Legend + Survey Image + Title Box still need to be drawn
        _draw_legend_stack(page, img_gray, [], survey_image_path, title_box_data, dpi=dpi, include_legend=include_legend, title_font_size=title_font_size)
    else:
        # Convert callout_records to global PDF coordinates
        unique_callouts: list[dict] = []
        for rec in callout_records:
            tile_idx, lx, ly, text = rec["tile_idx"], rec["lx"], rec["ly"], rec["text"]
            if tile_idx not in tile_offsets:
                continue
            ox, oy = tile_offsets[tile_idx]
            gx, gy = ox + lx, oy + ly

            pt_canvas = np.array([gx, gy, 1.0])
            pt_img = W_inv @ pt_canvas
            img_x, img_y = pt_img[0] / pt_img[2], pt_img[1] / pt_img[2]

            pdf_x = img_x * (72.0 / dpi)
            pdf_y = img_y * (72.0 / dpi)

            # Dedup: skip if identical text within 50 PDF pts
            is_dup = any(
                c["text"] == text and math.sqrt((pdf_x - c["x"]) ** 2 + (pdf_y - c["y"]) ** 2) < 50.0
                for c in unique_callouts
            )
            if not is_dup:
                unique_callouts.append({"x": pdf_x, "y": pdf_y, "text": text, "img_x": img_x, "img_y": img_y})

        logger.info(f"Rendering {len(unique_callouts)} unique callouts to PDF.")

        def _is_empty(cx_img: float, cy_img: float, bw: float, bh: float) -> bool:
            x1, y1 = int(cx_img - bw / 2), int(cy_img - bh / 2)
            x2, y2 = int(cx_img + bw / 2), int(cy_img + bh / 2)
            if x1 < 0 or y1 < 0 or x2 > w_img or y2 > h_img:
                return False
            roi = img_gray[y1:y2, x1:x2]
            return roi.size > 0 and (np.sum(roi < 240) / roi.size) < 0.05

        for c in unique_callouts:
            font_size = 12
            lines = c["text"].split("\n")
            max_line_len = max(len(l) for l in lines)
            
            box_w = int(max_line_len * 7.2) + 12
            box_h = 32 if len(lines) > 1 else 18
            
            bw_img = box_w * (dpi / 72.0)
            bh_img = box_h * (dpi / 72.0)

            cx_img, cy_img = c["img_x"], c["img_y"]
            ex_img, ey_img = cx_img + 120, cy_img - 120
            found = False

            for r_img in np.arange(60 * (dpi / 72.0), 400 * (dpi / 72.0), 40 * (dpi / 72.0)):
                for ang in np.linspace(0, 2 * math.pi, 24, endpoint=False):
                    px = cx_img + r_img * math.cos(ang)
                    py = cy_img + r_img * math.sin(ang)
                    if _is_empty(px, py, bw_img, bh_img):
                        test_rect = fitz.Rect(
                            px * (72 / dpi) - box_w / 2 - 5, py * (72 / dpi) - box_h / 2 - 5,
                            px * (72 / dpi) + box_w / 2 + 5, py * (72 / dpi) + box_h / 2 + 5,
                        )
                        if not any(test_rect.intersects(pr) for pr in placed_rects):
                            ex_img, ey_img = px, py
                            found = True
                            break
                if found:
                    break

            ex_pdf, ey_pdf = ex_img * (72 / dpi), ey_img * (72 / dpi)
            cx_pdf, cy_pdf = c["x"], c["y"]

            text_rect = fitz.Rect(
                ex_pdf - box_w / 2, ey_pdf - box_h / 2,
                ex_pdf + box_w / 2, ey_pdf + box_h / 2,
            )
            placed_rects.append(text_rect)

            angle = math.atan2(cy_pdf - ey_pdf, cx_pdf - ex_pdf)
            offset_pts = 40.0  # Wide gap to ensure symbol clearance
            tip_x = cx_pdf - offset_pts * math.cos(angle)
            tip_y = cy_pdf - offset_pts * math.sin(angle)
            knee_x = (ex_pdf + cx_pdf) / 2.0
            knee_y = (ey_pdf + cy_pdf) / 2.0
            attach_x = ex_pdf + (box_w / 2.0) * math.cos(angle)
            attach_y = ey_pdf + (box_h / 2.0) * math.sin(angle)

            try:
                annot = page.add_freetext_annot(
                    text_rect, c["text"], fontsize=font_size, fontname="helv",
                    text_color=(0, 0, 0), fill_color=(1, 1, 0),
                    callout=[fitz.Point(tip_x, tip_y), fitz.Point(knee_x, knee_y), fitz.Point(attach_x, attach_y)],
                    align=1,
                )
            except TypeError:
                annot = page.add_freetext_annot(
                    text_rect, c["text"], fontsize=font_size, fontname="helv",
                    text_color=(0, 0, 0), fill_color=(1, 1, 0),
                    callout=[fitz.Point(tip_x, tip_y), fitz.Point(attach_x, attach_y)],
                )

            try:
                annot.set_border(width=1.5)
            except Exception as e:
                full_error = traceback.format_exc()
                logger.error(f"❌ Fiber After pipeline failed: {e}\n{full_error}")
            annot.update()
            _patch_annot_color(doc, annot)

        # Legend + Survey Image + Title Box
        _draw_legend_stack(page, img_gray, unique_callouts, survey_image_path, title_box_data, dpi=dpi, include_legend=include_legend, title_font_size=title_font_size)

    # ── Save ────────────────────────────────────────────────────────────────
    # Ensure the directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    doc.save(
        str(output_path),
        deflate=True,
        garbage=4,
        clean=True,
        linear=False,
    )
    doc.close()
    logger.info(f"✅ Vector report saved to {output_path.as_posix()}")
    return output_path


# ─────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────

def _patch_annot_color(doc: fitz.Document, annot: fitz.Annot) -> None:
    """Patches annotation appearance stream for consistent red/yellow styling."""
    try:
        ap_type, ap_val = doc.xref_get_key(annot.xref, "AP")
        m = re.search(r"/N\s+(\d+)\s+\d+\s+R", ap_val)
        if m:
            n_xref = int(m.group(1))
            raw = doc.xref_stream(n_xref)
            doc.update_stream(n_xref, raw.replace(b"0 0 0 RG", b"1 0 0 RG"))

        doc.xref_set_key(annot.xref, "C",  "[1 1 0]")
        doc.xref_set_key(annot.xref, "IC", "[1 1 0]")
        doc.xref_set_key(annot.xref, "DA", "(1 0 0 RG 0 0 0 rg /Helv 9 Tf)")
        doc.xref_set_key(
            annot.xref, "DS",
            "(font: Helv 9pt; color: #000000; background-color: #FFFF00; border: 1.5pt solid #FF0000;)"
        )
    except Exception:
        pass  # Non-critical; visual only


def generate_final_report(
    pdf_path: str | Path,
    callouts: list[dict],
    output_path: str | Path,
    dpi: int = 300,
    survey_image_path: str | Path | None = None,
    title_box_data: dict | None = None,
    title_font_size: int = 14,
) -> Path:
    """
    Simpler version of generate_vector_report for single maps.
    Used by Fiber Overview pipeline.
    """
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = pdf_to_image(pdf_path, dpi=dpi)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = img_gray.shape

    doc = fitz.open(pdf_path)
    page = doc.load_page(0)

    processed_callouts = []
    for c in callouts:
        c_text = c["text"]
        pdf_x = c["x"]
        pdf_y = c["y"]
        
        is_dup = any(
            pc["text"] == c_text and math.sqrt((pdf_x - pc["x"]) ** 2 + (pdf_y - pc["y"]) ** 2) < 50.0
            for pc in processed_callouts
        )
        if not is_dup:
            processed_callouts.append({
                "text": c_text,
                "x": pdf_x,
                "y": pdf_y,
                "img_x": pdf_x * (dpi / 72.0),
                "img_y": pdf_y * (dpi / 72.0),
            })

    placed_rects = []
    
    legend_rect = _draw_legend_stack(page, img_gray, [], survey_image_path, title_box_data, dpi=dpi, include_legend=False, title_font_size=title_font_size)
    if legend_rect:
        placed_rects.append(legend_rect)


    def _is_empty(cx_img, cy_img, bw, bh):
        x1, y1 = int(cx_img - bw / 2), int(cy_img - bh / 2)
        x2, y2 = int(cx_img + bw / 2), int(cy_img + bh / 2)
        if x1 < 0 or y1 < 0 or x2 > w_img or y2 > h_img:
            return False
        roi = img_gray[y1:y2, x1:x2]
        return roi.size > 0 and (np.sum(roi < 240) / roi.size) < 0.1

    for c in processed_callouts:
        box_w = max(22, int(len(c["text"]) * 7.0) + 12)
        box_h = 24
        bw_img = box_w * (dpi / 72.0)
        bh_img = box_h * (dpi / 72.0)

        cx_img, cy_img = c["img_x"], c["img_y"]
        found = False
        for r_img in np.arange(80 * (dpi / 72.0), 500 * (dpi / 72.0), 50 * (dpi / 72.0)):
            for ang in np.linspace(0, 2 * math.pi, 24, endpoint=False):
                px, py = cx_img + r_img * math.cos(ang), cy_img + r_img * math.sin(ang)
                if _is_empty(px, py, bw_img, bh_img):
                    test_rect = fitz.Rect(
                        px * (72 / dpi) - box_w / 2 - 5, py * (72 / dpi) - box_h / 2 - 5,
                        px * (72 / dpi) + box_w / 2 + 5, py * (72 / dpi) + box_h / 2 + 5,
                    )
                    if not any(test_rect.intersects(pr) for pr in placed_rects):
                        ex_pdf, ey_pdf = px * (72 / dpi), py * (72 / dpi)
                        found = True; break
            if found: break
        
        if not found:
            ex_pdf, ey_pdf = c["x"] + 100, c["y"] - 100

        text_rect = fitz.Rect(ex_pdf - box_w/2, ey_pdf - box_h/2, ex_pdf + box_w/2, ey_pdf + box_h/2)
        placed_rects.append(text_rect)

        angle = math.atan2(c["y"] - ey_pdf, c["x"] - ex_pdf)
        offset_pts = 8.0
        tip_x, tip_y = c["x"] - offset_pts * math.cos(angle), c["y"] - offset_pts * math.sin(angle)
        knee_x, knee_y = (ex_pdf + c["x"]) / 2.0, (ey_pdf + c["y"]) / 2.0
        attach_x, attach_y = ex_pdf + (box_w / 2.0) * math.cos(angle), ey_pdf + (box_h / 2.0) * math.sin(angle)

        annot = page.add_freetext_annot(
            text_rect, c["text"], fontsize=10, fontname="helv",
            text_color=(0,0,0), fill_color=(1,1,0),
            callout=[fitz.Point(tip_x, tip_y), fitz.Point(knee_x, knee_y), fitz.Point(attach_x, attach_y)],
            align=1
        )
        try:
            annot.set_border(width=1.5)
        except Exception:
            pass
        annot.update()
        _patch_annot_color(doc, annot)

    doc.save(str(output_path), deflate=True, garbage=4, clean=True, linear=False)
    doc.close()
    return output_path


def _draw_legend_stack(
    page: fitz.Page,
    img_gray: np.ndarray,
    callouts: list[dict],
    survey_image_path: str | Path | None,
    title_box_data: dict | None,
    dpi: int = 800,
    include_legend: bool = True,
    title_font_size: int = 14,
) -> fitz.Rect | None:
    """Draws info images + legend table in the top-left whitespace."""
    h_img, w_img = img_gray.shape
    line_h = 24  # Reduced from 30
    col_w = [45, 170, 45]  # Reduced from [60, 230, 60] -> total 260 width
    leg_w = sum(col_w)
    leg_h = line_h * (len(LEGEND_DATA) + 1) if include_legend else 0

    if include_legend:
        counts = {k: 0 for k in LEGEND_DATA}
        for c in callouts:
            if c["text"]:
                first = c["text"][0].upper()
                if first in counts:
                    counts[first] += 1

    stack_items: list[dict] = []
    total_w, total_h = leg_w if include_legend else 0, 0

    # Pre-calculate title box size to inform the image scaling
    tb_w, tb_h = 0, 0
    title_lines = []
    tb_fs = title_font_size
    tb_lh = tb_fs + 6
    if title_box_data and any(title_box_data.values()):
        pid = title_box_data.get("prism_id", "")
        node = title_box_data.get("node_name", "")
        inst = title_box_data.get("instance", "")
        map_type = title_box_data.get("map_type", "")
        pg_text = f"PG 1 OF {page.parent.page_count}"
        
        title_lines = [
            (pid,      (0, 0, 0)),
            (node,     (0, 0, 0)),
            (inst,     (0, 0, 0)),
            (map_type, (0, 0, 0)),
            (pg_text,  (0, 0, 0)),
        ]
        
        # Calculate dynamic width
        tb_w = leg_w if include_legend else 0  # Ensure it is at least as wide as the legend for a clean stack
        for text, _ in title_lines:
            if text:
                w = fitz.get_text_length(text, fontname="helv", fontsize=tb_fs)
                tb_w = max(tb_w, w + 20)  # 10pt padding left and right
                
        # Calculate dynamic height
        non_empty_lines = sum(1 for t, _ in title_lines if t)
        if non_empty_lines > 0:
            tb_h = (non_empty_lines * tb_lh) + 20  # 10pt padding top/bottom
            total_w = max(total_w, tb_w)

    if survey_image_path:
        img = cv2.imread(str(survey_image_path))
        if img is not None:
            ih, iw = img.shape[:2]
            tw = 750.0 if include_legend else 300.0  # Make the survey image big only for coax
            th = (ih / iw) * tw
            stack_items.append({"type": "image", "path": str(survey_image_path), "w": tw, "h": th})
            total_w = max(total_w, tw)
            total_h += th + 15

    if title_box_data and any(title_box_data.values()):
        if tb_h > 0:
            stack_items.append({"type": "title_box", "lines": title_lines, "w": tb_w, "h": tb_h})
            total_h += tb_h + 15

    if include_legend:
        stack_items.append({"type": "legend", "w": leg_w, "h": leg_h})
        total_h += leg_h

    # ──────────────────────────────────────────────────────────────────────
    # Legend placement strategy
    # ──────────────────────────────────────────────────────────────────────
    MARGIN_PTS = 15.0
    page_r = page.rect

    def _is_rect_clear(x1_img: int, y1_img: int, x2_img: int, y2_img: int) -> bool:
        """Check empty-space in the raster img_gray at an image-pixel rect."""
        x1, y1 = max(0, x1_img), max(0, y1_img)
        x2, y2 = min(w_img, x2_img), min(h_img, y2_img)
        if x2 <= x1 or y2 <= y1:
            return False
        roi = img_gray[y1:y2, x1:x2]
        # Use a very strict threshold (0.1%) for the Legend Stack.
        # Since its bounding box is massive (e.g. 750x600 pts = Millions of pixels),
        # even 1% dark pixels can mean it's overlapping thick map lines.
        return roi.size > 0 and (np.sum(roi < 240) / roi.size) < 0.001

    # Target top-right corner coordinates (PDF points)
    start_x_pts = page_r.width - MARGIN_PTS - total_w
    start_y_pts = page_r.y0 + MARGIN_PTS  # Use page_r.y0 to handle existing cropboxes safely

    # Verify if this top-right region in the ORIGINAL map raster is clear
    x1_img = int(start_x_pts * (dpi / 72.0))
    y1_img = int((start_y_pts - page_r.y0) * (dpi / 72.0))
    x2_img = int((start_x_pts + total_w) * (dpi / 72.0))
    y2_img = int(((start_y_pts - page_r.y0) + total_h) * (dpi / 72.0))

    if not _is_rect_clear(x1_img, y1_img, x2_img, y2_img):
        if include_legend:
            # ── Map Overlaps! Dynamically extend the map page upwards ──
            extension_pts = total_h + (MARGIN_PTS * 2)
            
            # To avoid PyMuPDF validation bugs and perfectly extend the page upwards,
            # we parse the raw PDF dictionary array for MediaBox.
            # Format is usually [LLx LLy URx URy]
            doc = page.parent
            _, val = doc.xref_get_key(page.xref, "MediaBox")
            box_coords = [float(v) for v in val.strip("[]").split()]
            
            # Increase the top boundary (highest Y value in PDF coordinate space)
            top_idx = 3 if box_coords[3] > box_coords[1] else 1
            box_coords[top_idx] += extension_pts
            
            box_str = f"[{box_coords[0]} {box_coords[1]} {box_coords[2]} {box_coords[3]}]"
            
            # Update raw PDF properties
            doc.xref_set_key(page.xref, "MediaBox", box_str)
            doc.xref_set_key(page.xref, "CropBox", box_str)
            
            # Tell PyMuPDF to reload page geometry so its (0,0) recalibrates to 
            # the new extended top, making drawing in the new area visible!
            page = doc.reload_page(page)
            
            # The new top starts at relative Y=0 in PyMuPDF's drawing space
            start_y_pts = MARGIN_PTS

    curr_x = start_x_pts
    curr_y = start_y_pts

    for item in stack_items:
        # Pushes elements that are narrower than total_w to the absolute right
        draw_x = curr_x + (total_w - item["w"])
        
        if item["type"] == "image":
            r = fitz.Rect(draw_x, curr_y, draw_x + item["w"], curr_y + item["h"])
            page.insert_image(r, filename=item["path"])
            curr_y += item["h"] + 10
            
        elif item["type"] == "title_box":
            bg = fitz.Rect(draw_x, curr_y, draw_x + item["w"], curr_y + item["h"])
            page.draw_rect(bg, color=(1, 0, 0), fill=(1, 1, 0), width=1.5)

            py = curr_y + 10
            for text, color in item["lines"]:
                if text:
                    # Right-align the text itself
                    text_w = fitz.get_text_length(text, fontname="helv", fontsize=tb_fs)
                    text_x = draw_x + item["w"] - 10 - text_w
                    page.insert_text((text_x, py + tb_fs), text, fontsize=tb_fs, fontname="helv", color=color)
                    py += tb_lh
            curr_y += item["h"] + 10
            
        elif item["type"] == "legend":
            bg = fitz.Rect(draw_x, curr_y, draw_x + leg_w, curr_y + leg_h)
            page.draw_rect(bg, color=(0,0,0), fill=(1,1,1), width=1.5)

            for i, hdr in enumerate(["Code", "Action", "Count"]):
                xc = draw_x + sum(col_w[:i])
                cr = fitz.Rect(xc, curr_y, xc + col_w[i], curr_y + line_h)
                page.draw_rect(cr, color=(0,0,0), fill=(0.8,0.9,1.0), width=1.0)
                tw_pts = fitz.get_text_length(hdr, fontname="hebo", fontsize=11)
                page.insert_text((xc + (col_w[i]-tw_pts)/2, curr_y+(line_h+8.8)/2), hdr, fontsize=11, fontname="hebo")

            for idx, code in enumerate(sorted(LEGEND_DATA)):
                ry = curr_y + (idx+1)*line_h
                for i, val in enumerate([code, LEGEND_DATA[code], str(counts[code])]):
                    xc = draw_x + sum(col_w[:i])
                    cr = fitz.Rect(xc, ry, xc + col_w[i], ry + line_h)
                    page.draw_rect(cr, color=(0,0,0), width=1.0)
                    tw_pts = fitz.get_text_length(val, fontname="helv", fontsize=10)
                    page.insert_text((xc+(col_w[i]-tw_pts)/2, ry+(line_h+8)/2), val, fontsize=10, fontname="helv")

    if stack_items:
        return fitz.Rect(start_x_pts, start_y_pts, start_x_pts + total_w, start_y_pts + total_h)
    return None

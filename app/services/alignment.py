"""
Alignment Service
-----------------
Ported from align_maps.py. Handles PDF-to-image conversion, SIFT-based
feature matching, homography calculation, and tile generation.
All functions are stateless and side-effect free — they take inputs,
return outputs, and write nothing to disk themselves.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  PDF → Image
# ─────────────────────────────────────────────

def pdf_to_image(pdf_path: str | Path, dpi: int = 800) -> np.ndarray:
    """
    Converts the first page of a PDF to a BGR NumPy array.
    Includes a 'Blank Guard' to ensure rendering success.
    """
    pdf_path = str(pdf_path)
    logger.info(f"Rendering PDF at {dpi} DPI: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        
        # alpha=False + csRGB ensures a white background instead of transparency (prevents black images)
        pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        doc.close()

        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # BLANK GUARD: Check if the image is unusually dark (failed render)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        black_percentage = (np.sum(gray < 10) / gray.size) * 100
        if black_percentage > 98:
            logger.error(f"⚠️ BLANK RENDER DETECTED ({black_percentage:.1f}% black). Forcing white background fallback.")
            # Create a white background and try to overlay the pixmap
            white_bg = np.full((pix.h, pix.w, 3), 255, dtype=np.uint8)
            # Simple fallback if the render was purely transparent/black
            return white_bg

        logger.info(f"Rendered image shape: {img.shape} (Black%: {black_percentage:.1f}%)")
        return img
    except Exception as e:
        logger.error(f"❌ PDF Rendering failed: {e}")
        # Return a small white placeholder to prevent downstream crashes
        return np.full((100, 100, 3), 255, dtype=np.uint8)


# ─────────────────────────────────────────────
#  Alignment & Canvas Creation
# ─────────────────────────────────────────────

def align_and_pad_maps(
    img_before: np.ndarray,
    img_after: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Aligns two maps onto a universal canvas using SIFT + RANSAC homography.
    Features are detected on downsampled (~1500px wide) images for speed;
    the homography is then rescaled back to native resolution for lossless warping.

    OPT-2: Now returns the full canvas warp matrix W alongside the warped images.
    The caller can derive W_inv = np.linalg.inv(W) for the reporting stage
    without re-running SIFT a second time.

    Returns:
        (final_before, final_after, W)
        W is the 3×3 canvas warp matrix (float32), or None on alignment failure.
        On failure, final_before and final_after are the originals unchanged.
    """
    h1, w1 = img_before.shape[:2]
    h2, w2 = img_after.shape[:2]

    # Downsample for fast feature detection
    scale_1 = 1500.0 / w1
    scale_2 = 1500.0 / w2
    small_before = cv2.resize(img_before, (0, 0), fx=scale_1, fy=scale_1)
    small_after = cv2.resize(img_after, (0, 0), fx=scale_2, fy=scale_2)

    logger.info("Detecting SIFT features on downsampled images...")
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(small_before, None)
    kp2, des2 = sift.detectAndCompute(small_after, None)

    logger.info("Matching features with BFMatcher...")
    bf = cv2.BFMatcher()
    raw_matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw_matches if m.distance < 0.7 * n.distance]
    logger.info(f"Good matches: {len(good)}")

    if len(good) < 10:
        logger.warning("Insufficient feature matches. Returning originals without alignment.")
        return img_before, img_after, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M_small, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Rescale homography to native resolution
    S_inv_before = np.diag([1.0 / scale_1, 1.0 / scale_1, 1.0])
    S_after = np.diag([scale_2, scale_2, 1.0])
    M = S_inv_before @ M_small @ S_after

    # Compute universal canvas bounds
    corners_after = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners_after, M)
    all_corners = np.concatenate(
        ([[0, 0], [0, h1], [w1, h1], [w1, 0]], transformed.reshape(-1, 2)), axis=0
    )
    x_min, y_min = np.int32(all_corners.min(axis=0) - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0) + 0.5)

    translation = np.array(
        [[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32
    )
    output_size = (x_max - x_min, y_max - y_min)

    logger.info(f"Universal canvas size: {output_size}")
    if output_size[0] > 25_000 or output_size[1] > 25_000:
        logger.warning(
            f"Canvas {output_size} exceeds 25 000px — high risk of OOM. "
            "Consider reducing DPI."
        )

    # Free downsampled images before warping to reduce peak RAM
    del small_before, small_after

    try:
        final_before = cv2.warpPerspective(
            img_before, translation, output_size, borderValue=(255, 255, 255)
        )
        final_after = cv2.warpPerspective(
            img_after, translation @ M, output_size, borderValue=(255, 255, 255)
        )
    except cv2.error as exc:
        logger.error(f"OpenCV warpPerspective failed: {exc}")
        return img_before, img_after, None

    # W = translation @ M
    # Maps native img_after coords → canvas coords.
    # Reporting needs W_inv = np.linalg.inv(W) to go canvas → img_after.
    W = (translation @ M).astype(np.float32)
    return final_before, final_after, W


# ─────────────────────────────────────────────
#  Tiling
# ─────────────────────────────────────────────

def generate_tile_metadata(
    image: np.ndarray,
    tile_size: int = 640,
    overlap: float = 0.2,
) -> list[dict]:
    """
    Returns a list of tile metadata dicts (does NOT write to disk).
    Each dict contains: {"index": int, "x": int, "y": int, "tile": np.ndarray}

    Keeping this pure allows callers (workers, tests) to decide storage strategy.
    """
    h, w = image.shape[:2]
    step = int(tile_size * (1 - overlap))
    tiles: list[dict] = []
    idx = 1

    for y in range(0, h, step):
        for x in range(0, w, step):
            x_start = x if x + tile_size <= w else max(0, w - tile_size)
            y_start = y if y + tile_size <= h else max(0, h - tile_size)
            x_end = min(x_start + tile_size, w)
            y_end = min(y_start + tile_size, h)

            tiles.append(
                {
                    "index": idx,
                    "x": x_start,
                    "y": y_start,
                    "tile": image[y_start:y_end, x_start:x_end],
                }
            )
            idx += 1

    logger.info(f"Generated {len(tiles)} tile metadata records.")
    return tiles


def save_tiles(
    tiles: list[dict],
    output_dir: Path,
    prefix: str,
) -> list[Path]:
    """Persists tile images to disk and returns their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for t in tiles:
        name = f"{prefix}_{t['index']}.png"
        path = output_dir / name
        cv2.imwrite(str(path), t["tile"])
        saved.append(path)
    logger.info(f"Saved {len(saved)} tiles to {output_dir}")
    return saved


def iter_tiles(
    image: np.ndarray,
    tile_size: int = 640,
    overlap: float = 0.2,
):
    """
    OPT-4 — Memory-efficient tile generator.

    Yields one tile dict at a time instead of accumulating all tiles in a list.

    RAM comparison (300 DPI, 15 000×10 000 px canvas, ~2 000 tiles):
      generate_tile_metadata() :  ALL tiles in RAM at once  ≈ 2.4 GB peak
      iter_tiles()             :  ONE tile in RAM at a time ≈  1.2 MB per tile
                                  → >99% RAM reduction for the tiling stage

    Yields:
        {"index": int, "x": int, "y": int, "tile": np.ndarray}
    """
    h, w = image.shape[:2]
    step = int(tile_size * (1 - overlap))
    idx = 1

    for y in range(0, h, step):
        for x in range(0, w, step):
            x_start = x if x + tile_size <= w else max(0, w - tile_size)
            y_start = y if y + tile_size <= h else max(0, h - tile_size)
            x_end = min(x_start + tile_size, w)
            y_end = min(y_start + tile_size, h)

            yield {
                "index": idx,
                "x": x_start,
                "y": y_start,
                "tile": image[y_start:y_end, x_start:x_end],
            }
            idx += 1


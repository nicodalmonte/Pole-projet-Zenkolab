"""Fundus image domain standardization transform.

Pipeline (PIL → PIL), each step individually togglable:
  1. FOV crop        — detect circular field-of-view, square-crop around it
  2. Illumination    — fast illumination normalization via resize proxy
                       (downscale → upscale → subtract), O(n), no Gaussian blur
  3. CLAHE           — adaptive contrast equalization on the green channel
  4. Circular mask   — set pixels outside the FOV disc to 0
  5. Z-score         — per-image normalization within FOV to fixed (μ, σ)
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Step 1 — FOV detection & crop
# ---------------------------------------------------------------------------

def detect_fov(gray: np.ndarray) -> tuple[int, int, int]:
    h, w = gray.shape
    if (gray < 15).mean() < 0.05:
        r = min(h, w) // 2
        return w // 2, h // 2, r
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return w // 2, h // 2, min(h, w) // 2
    (cx, cy), radius = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    if radius < min(h, w) * 0.20:
        return w // 2, h // 2, min(h, w) // 2
    return int(cx), int(cy), int(radius)


def fov_crop(img: np.ndarray, cx: int, cy: int, r: int,
             padding: float = 0.05) -> np.ndarray:
    h, w = img.shape[:2]
    side = int(r * 2 * (1 + padding))
    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(w, x0 + side)
    y1 = min(h, y0 + side)
    return img[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Step 2 — Fast illumination normalization (resize proxy)
#
# Equivalent to Ben Graham but without Gaussian blur:
#   blur_proxy = resize(resize(img, 16×16), original_size)
# This is O(n) and ~10× faster than GaussianBlur on large images.
# ---------------------------------------------------------------------------

def illumination_norm(img: np.ndarray, proxy_size: int = 16) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img.astype(np.float32), (proxy_size, proxy_size),
                       interpolation=cv2.INTER_AREA)
    blur_proxy = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    result = 4.0 * img.astype(np.float32) - 4.0 * blur_proxy + 128.0
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Step 3 — CLAHE on green channel
# ---------------------------------------------------------------------------

def clahe_green(img_bgr: np.ndarray, clip_limit: float = 2.0,
                tile_size: int = 8) -> np.ndarray:
    out = img_bgr.copy()
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                              tileGridSize=(tile_size, tile_size))
    out[:, :, 1] = clahe.apply(out[:, :, 1])
    return out


# ---------------------------------------------------------------------------
# Step 4 — Circular mask
# ---------------------------------------------------------------------------

def circular_mask(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    r = min(h, w) // 2
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), r, 255, -1)
    out = img.copy()
    out[mask == 0] = 0
    return out


# ---------------------------------------------------------------------------
# Step 5 — Per-image z-score within FOV
# ---------------------------------------------------------------------------

def zscore_fov(img: np.ndarray,
               target_mean: float = 128.0,
               target_std: float = 40.0) -> np.ndarray:
    h, w = img.shape[:2]
    r = min(h, w) // 2
    ys, xs = np.ogrid[:h, :w]
    mask = (xs - w // 2) ** 2 + (ys - h // 2) ** 2 <= r ** 2
    out = img.astype(np.float32)
    for c in range(3):
        ch = out[:, :, c]
        vals = ch[mask]
        mu, std = vals.mean(), vals.std()
        if std < 1e-3:
            std = 1.0
        out[:, :, c] = np.clip((ch - mu) / std * target_std + target_mean, 0, 255)
    return out.astype(np.uint8)


# ---------------------------------------------------------------------------
# Public transform
# ---------------------------------------------------------------------------

class FundusStandardize:
    """Domain-standardization transform for fundus photographs (PIL → PIL).

    Each step can be toggled individually for ablation studies.

    Args:
        target_size:   output size (square)
        do_fov_crop:   detect FOV and crop
        do_illumination: fast illumination normalization (resize proxy)
        do_clahe:      CLAHE on green channel
        do_mask:       black out pixels outside the FOV disc
        do_zscore:     per-image z-score normalization within FOV
    """

    def __init__(
        self,
        target_size: int = 512,
        do_fov_crop: bool = True,
        do_illumination: bool = True,
        do_clahe: bool = True,
        do_mask: bool = True,
        do_zscore: bool = True,
    ) -> None:
        self.target_size = target_size
        self.do_fov_crop = do_fov_crop
        self.do_illumination = do_illumination
        self.do_clahe = do_clahe
        self.do_mask = do_mask
        self.do_zscore = do_zscore

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        img = np.array(pil_img.convert("RGB"))
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if self.do_fov_crop:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            cx, cy, r = detect_fov(gray)
            bgr = fov_crop(bgr, cx, cy, r)

        if self.do_illumination:
            bgr = illumination_norm(bgr)

        if self.do_clahe:
            bgr = clahe_green(bgr)

        if self.do_mask:
            bgr = circular_mask(bgr)

        bgr = cv2.resize(bgr, (self.target_size, self.target_size),
                         interpolation=cv2.INTER_AREA)

        if self.do_zscore:
            bgr = zscore_fov(bgr)

        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def __repr__(self) -> str:
        steps = [k.replace("do_", "") for k, v in self.__dict__.items()
                 if k.startswith("do_") and v]
        return f"FundusStandardize(size={self.target_size}, steps={steps})"

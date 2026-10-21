"""Pixel-level filters shared by redaction, adjustments, and plugins."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def pixelate_image(image: Image.Image, block: int) -> Image.Image:
    """Mosaic: average each ``block`` cell, then nearest-neighbor it back up."""
    block = max(1, int(block))
    if block == 1:
        return image.copy()
    width, height = image.size
    pad_w = (block - width % block) % block
    pad_h = (block - height % block) % block
    if pad_w or pad_h:
        padded = Image.new(image.mode, (width + pad_w, height + pad_h))
        padded.paste(image, (0, 0))
    else:
        padded = image
    small_w = max(1, padded.width // block)
    small_h = max(1, padded.height // block)
    small = padded.resize((small_w, small_h), Image.Resampling.BOX)
    enlarged = small.resize((small_w * block, small_h * block), Image.Resampling.NEAREST)
    if enlarged.size == image.size:
        return enlarged
    return enlarged.crop((0, 0, width, height))


def gaussian_image(image: Image.Image, radius: float) -> Image.Image:
    radius = float(radius)
    if radius <= 0:
        return image.copy()
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def straight_to_premul_image(canvas: np.ndarray) -> Image.Image:
    """Float straight RGBA (0..1) → uint8 premultiplied RGBA.

    Blurring straight alpha independently darkens edges because transparent
    pixels often store black. Premultiplying first keeps those edges clean,
    which matters for a privacy blur that should not invent dark halos.
    """
    premul = np.array(canvas, dtype=np.float32, copy=True)
    premul[..., :3] *= premul[..., 3:4]
    uint8 = np.clip(np.rint(premul * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(uint8, "RGBA")


def premul_image_to_straight(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGBA")).astype(np.float32) / 255.0
    rgb = arr[..., :3]
    alpha = arr[..., 3:4]
    straight = np.where(alpha > 1e-6, rgb / np.maximum(alpha, 1e-6), 0.0)
    return np.concatenate(
        [np.clip(straight, 0, 1), np.clip(alpha, 0, 1)],
        axis=-1,
    ).astype(np.float32)


def filter_straight_canvas(canvas: np.ndarray, kind: str, amount: float) -> np.ndarray:
    """Blur or pixelate a float straight-RGBA canvas, preserving coverage."""
    premul = straight_to_premul_image(canvas)
    if kind == "blur":
        filtered = gaussian_image(premul, max(0.1, float(amount)))
    else:
        filtered = pixelate_image(premul, max(1, int(round(amount))))
    return premul_image_to_straight(filtered)

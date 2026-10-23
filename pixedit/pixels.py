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


def mask_effect(
    canvas: np.ndarray,
    kind: str,
    amount: float,
    color: tuple[int, int, int, int] = (0, 0, 0, 255),
    seed: int = 1,
) -> np.ndarray:
    """The image mixed through a mask. ``canvas`` is float straight RGBA, 0..1.

    Blur and pixelate stay on the premultiplied path so a transparent edge
    does not grow a dark halo. Solid replaces RGB and forces the covered
    area opaque. Noise, desaturate, invert, dim, and sharpen edit straight
    RGB and copy the original alpha back.

    Unknown kinds fall through to pixelate, which is what older projects
    used as the default effect.
    """
    if kind not in {"solid", "noise", "desaturate", "invert", "dim", "sharpen"}:
        return filter_straight_canvas(canvas, kind, amount)
    if kind == "solid":
        effect = np.empty_like(canvas, dtype=np.float32)
        effect[..., 0] = float(color[0]) / 255.0
        effect[..., 1] = float(color[1]) / 255.0
        effect[..., 2] = float(color[2]) / 255.0
        effect[..., 3] = 1.0
        return effect
    effect = np.array(canvas, dtype=np.float32, copy=True)
    rgb = effect[..., :3]
    strength = min(1.0, max(0.0, float(amount) / 100.0))
    if kind == "noise":
        # The seed is stored on the layer, so a redraw samples the same grain.
        noise = np.random.default_rng(int(seed) & 0xFFFFFFFF).random(rgb.shape).astype(np.float32)
        effect[..., :3] = rgb * (1.0 - strength) + noise * strength
        return effect
    if kind == "desaturate":
        luma = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
        effect[..., :3] = rgb * (1.0 - strength) + luma[..., None] * strength
        return effect
    if kind == "invert":
        effect[..., :3] = rgb * (1.0 - strength) + (1.0 - rgb) * strength
        return effect
    if kind == "dim":
        effect[..., :3] = rgb * (1.0 - strength)
        return effect
    return _sharpen_straight(canvas, amount)


def _sharpen_straight(canvas: np.ndarray, amount: float) -> np.ndarray:
    """Unsharp mask on straight color. Alpha is the canvas's own coverage."""
    uint8 = np.clip(np.rint(np.array(canvas) * 255.0), 0, 255).astype(np.uint8)
    image = Image.fromarray(np.ascontiguousarray(uint8), "RGBA")
    percent = max(1, int(round(float(amount))) * 2)
    sharpened = image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=percent, threshold=0))
    out = np.asarray(sharpened.convert("RGBA")).astype(np.float32) / 255.0
    out[..., 3] = canvas[..., 3]
    return out

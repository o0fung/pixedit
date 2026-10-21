"""Color adjustments and destructive filters.

Each function returns a new image. Callers intersect with the selection and
wrap the result in ``MutateLayerCommand``.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from pixedit.pixels import gaussian_image, pixelate_image


def _rgba(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGBA"))


def _from_rgb(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8)), "RGBA")


def apply_masked(before: Image.Image, after: Image.Image, selection: Image.Image | None) -> Image.Image:
    """Keep ``before`` outside the selection and ``after`` inside it."""
    if selection is None:
        return after
    original = _rgba(before).astype(np.float32)
    edited = _rgba(after).astype(np.float32)
    mask = np.asarray(selection.convert("L")).astype(np.float32) / 255.0
    if mask.shape != original.shape[:2]:
        return after
    mix = mask[..., None]
    out = edited * mix + original * (1.0 - mix)
    return _from_rgb(out)


def brightness_contrast(image: Image.Image, brightness: float, contrast: float) -> Image.Image:
    """``brightness`` and ``contrast`` are each -100..100."""
    arr = _rgba(image).astype(np.float32)
    factor = (100.0 + float(contrast)) / 100.0
    shift = float(brightness) / 100.0 * 128.0
    rgb = (arr[..., :3] - 128.0) * factor + 128.0 + shift
    arr[..., :3] = np.clip(rgb, 0, 255)
    return _from_rgb(arr)


def levels(
    image: Image.Image,
    in_black: float,
    in_white: float,
    gamma: float,
    out_black: float = 0.0,
    out_white: float = 255.0,
) -> Image.Image:
    """A compact levels adjustment: input window, gamma, output window."""
    arr = _rgba(image).astype(np.float32)
    span = max(1.0, float(in_white) - float(in_black))
    rgb = np.clip((arr[..., :3] - float(in_black)) / span, 0.0, 1.0)
    rgb = np.power(rgb, 1.0 / max(0.05, float(gamma)))
    rgb = rgb * (float(out_white) - float(out_black)) + float(out_black)
    arr[..., :3] = np.clip(rgb, 0, 255)
    return _from_rgb(arr)


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc
    safe = np.where(delta == 0, 1.0, delta)
    rc = (maxc - r) / safe
    gc = (maxc - g) / safe
    bc = (maxc - b) / safe
    hue = np.zeros_like(maxc)
    hue = np.where((maxc == r) & (delta > 0), (bc - gc) % 6.0, hue)
    hue = np.where((maxc == g) & (delta > 0), 2.0 + rc - bc, hue)
    hue = np.where((maxc == b) & (delta > 0), 4.0 + gc - rc, hue)
    hue = (hue / 6.0) % 1.0
    sat = np.where(maxc > 1e-8, delta / np.maximum(maxc, 1e-8), 0.0)
    return hue, sat, maxc


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i_mod = i % 6
    r = np.choose(i_mod, [v, q, p, p, t, v])
    g = np.choose(i_mod, [t, v, v, q, p, p])
    b = np.choose(i_mod, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def hue_saturation(
    image: Image.Image,
    hue_degrees: float = 0.0,
    saturation: float = 0.0,
    lightness: float = 0.0,
) -> Image.Image:
    """Shift hue (degrees), saturation (-100..100), and value (-100..100)."""
    arr = _rgba(image).astype(np.float32)
    rgb = arr[..., :3] / 255.0
    hue, sat, val = _rgb_to_hsv(rgb)
    hue = (hue + float(hue_degrees) / 360.0) % 1.0
    sat = np.clip(sat * (1.0 + float(saturation) / 100.0), 0.0, 1.0)
    val = np.clip(val + float(lightness) / 100.0, 0.0, 1.0)
    arr[..., :3] = _hsv_to_rgb(hue, sat, val) * 255.0
    return _from_rgb(arr)


def desaturate(image: Image.Image) -> Image.Image:
    arr = _rgba(image).astype(np.float32)
    luma = arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722
    arr[..., 0] = luma
    arr[..., 1] = luma
    arr[..., 2] = luma
    return _from_rgb(arr)


def invert(image: Image.Image) -> Image.Image:
    arr = _rgba(image)
    arr[..., :3] = 255 - arr[..., :3]
    return Image.fromarray(arr, "RGBA")


def blur(image: Image.Image, radius: float) -> Image.Image:
    return gaussian_image(image.convert("RGBA"), radius)


def sharpen(image: Image.Image) -> Image.Image:
    return image.convert("RGBA").filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=0))


def pixelate(image: Image.Image, block: int) -> Image.Image:
    return pixelate_image(image.convert("RGBA"), block)

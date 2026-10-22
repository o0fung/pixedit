"""Brush dabs, shapes, bucket fill, and magic-wand masks.

These functions are pure: they return new images and do not touch the document.
Tools snapshot the layer, call in here, and commit one undo command per gesture.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw


def _falloff(dist: np.ndarray, radius: float, hardness: float) -> np.ndarray:
    """1 inside the brush, falling to 0 at ``radius``.

    ``hardness`` is 0..1. 1 is a hard disk. Lower values keep a solid core of
    radius ``hardness * radius`` and ease out to the edge.
    """
    hard = min(1.0, max(0.0, float(hardness)))
    if hard >= 0.999:
        return (dist <= radius).astype(np.float32)
    inner = radius * hard
    return np.clip((radius - dist) / max(1e-4, radius - inner), 0.0, 1.0).astype(np.float32)


def _stamp_bounds(
    width: int, height: int, cx: float, cy: float, radius: float
) -> tuple[int, int, int, int] | None:
    rad = int(math.ceil(radius))
    x0 = max(0, int(math.floor(cx)) - rad)
    y0 = max(0, int(math.floor(cy)) - rad)
    x1 = min(width, int(math.floor(cx)) + rad + 1)
    y1 = min(height, int(math.floor(cy)) + rad + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1, y1


def dab(
    array: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    hardness: float,
    color: tuple[int, int, int, int],
    opacity: float,
    erase: bool,
    selection: np.ndarray | None,
) -> None:
    """Paint one brush stamp into an HxWx4 uint8 array (in place)."""
    radius = max(0.5, float(radius))
    height, width = array.shape[:2]
    bounds = _stamp_bounds(width, height, cx, cy, radius)
    if bounds is None:
        return
    x0, y0, x1, y1 = bounds
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    alpha = _falloff(dist, radius, hardness) * float(opacity)
    if selection is not None:
        alpha = alpha * (selection[y0:y1, x0:x1].astype(np.float32) / 255.0)
    region = array[y0:y1, x0:x1].astype(np.float32)
    if erase:
        region[..., 3] *= 1.0 - alpha
    else:
        src_rgb = np.array(color[:3], dtype=np.float32)
        src_a = alpha * (color[3] / 255.0)
        dst_a = region[..., 3] / 255.0
        out_a = src_a + dst_a * (1.0 - src_a)
        numer = src_rgb * src_a[..., None] + region[..., :3] * dst_a[..., None] * (1.0 - src_a[..., None])
        out_rgb = np.where(out_a[..., None] > 1e-6, numer / np.maximum(out_a[..., None], 1e-6), 0.0)
        region[..., :3] = out_rgb
        region[..., 3] = out_a * 255.0
    array[y0:y1, x0:x1] = np.clip(np.rint(region), 0, 255).astype(np.uint8)


def dab_mask(
    array: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    hardness: float,
    opacity: float,
    erase: bool,
    selection: np.ndarray | None,
) -> None:
    """Paint a redaction mask. ``array`` is HxW uint8, 255 = fully redacted."""
    radius = max(0.5, float(radius))
    height, width = array.shape[:2]
    bounds = _stamp_bounds(width, height, cx, cy, radius)
    if bounds is None:
        return
    x0, y0, x1, y1 = bounds
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    strength = _falloff(dist, radius, hardness) * float(opacity)
    if selection is not None:
        strength = strength * (selection[y0:y1, x0:x1].astype(np.float32) / 255.0)
    region = array[y0:y1, x0:x1].astype(np.float32)
    if erase:
        region *= 1.0 - strength
    else:
        region = region + (255.0 - region) * strength
    array[y0:y1, x0:x1] = np.clip(np.rint(region), 0, 255).astype(np.uint8)


def interpolate(
    start: tuple[float, float], end: tuple[float, float], spacing: float
) -> list[tuple[float, float]]:
    x0, y0 = start
    x1, y1 = end
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist < 1e-6:
        return []
    step = max(1.0, float(spacing))
    count = max(1, int(math.ceil(dist / step)))
    return [
        (x0 + (x1 - x0) * (i / count), y0 + (y1 - y0) * (i / count))
        for i in range(1, count + 1)
    ]


def paint_points(
    image: Image.Image,
    points: list[tuple[float, float]],
    radius: float,
    hardness: float,
    color: tuple[int, int, int, int],
    opacity: float,
    erase: bool = False,
    selection: Image.Image | None = None,
) -> Image.Image:
    array = np.array(image.convert("RGBA"))
    sel = None if selection is None else np.asarray(selection)
    for x, y in points:
        dab(array, x, y, radius, hardness, color, opacity, erase, sel)
    return Image.fromarray(array, "RGBA")


def paint_mask_points(
    mask: Image.Image,
    points: list[tuple[float, float]],
    radius: float,
    hardness: float,
    opacity: float,
    erase: bool = False,
    selection: Image.Image | None = None,
) -> Image.Image:
    array = np.array(mask.convert("L"))
    sel = None if selection is None else np.asarray(selection)
    for x, y in points:
        dab_mask(array, x, y, radius, hardness, opacity, erase, sel)
    return Image.fromarray(array, "L")


def mask_with_selection(overlay: Image.Image, selection: Image.Image | None) -> Image.Image:
    if selection is None:
        return overlay
    arr = np.array(overlay.convert("RGBA"))
    sel = np.asarray(selection.convert("L")).astype(np.float32) / 255.0
    if sel.shape != arr.shape[:2]:
        return overlay
    arr[..., 3] = np.clip(np.rint(arr[..., 3].astype(np.float32) * sel), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def draw_shape(
    image: Image.Image,
    kind: str,
    rect: tuple[float, float, float, float],
    color: tuple[int, int, int, int],
    stroke: int,
    filled: bool,
    selection: Image.Image | None = None,
    line: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> Image.Image:
    """Draw a rectangle, ellipse, or line onto a copy of ``image``."""
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    stroke = max(1, int(stroke))
    if kind == "line" and line is not None:
        draw.line([line[0], line[1]], fill=color, width=stroke)
    else:
        # Same Pillow constraint as the selection masks: the second corner
        # must sit down and to the right of the first.
        box = [
            min(rect[0], rect[2]),
            min(rect[1], rect[3]),
            max(rect[0], rect[2]),
            max(rect[1], rect[3]),
        ]
        if kind == "ellipse":
            if filled:
                draw.ellipse(box, fill=color, outline=color, width=stroke)
            else:
                draw.ellipse(box, outline=color, width=stroke)
        else:
            if filled:
                draw.rectangle(box, fill=color, outline=color, width=stroke)
            else:
                draw.rectangle(box, outline=color, width=stroke)
    overlay = mask_with_selection(overlay, selection)
    return Image.alpha_composite(image.convert("RGBA"), overlay)


def connected_mask(image: Image.Image, x: int, y: int, tolerance: int) -> Image.Image:
    """L mask of pixels flood-filled from ``(x, y)``.

    Tolerance is the maximum per-channel absolute difference from the seed,
    which is how Pillow's ``floodfill(..., thresh=)`` decides membership.
    A sentinel color marks visited pixels so a high tolerance cannot hide
    pixels that already happened to equal the fill color.
    """
    rgba = image.convert("RGBA")
    width, height = rgba.size
    if not (0 <= x < width and 0 <= y < height):
        return Image.new("L", (width, height), 0)
    arr = np.array(rgba)
    seed = arr[y, x].copy()
    sentinel = np.array([1, 2, 3, 4], dtype=np.uint8)
    if np.array_equal(seed, sentinel):
        sentinel = np.array([8, 9, 10, 11], dtype=np.uint8)
    alt = sentinel.copy()
    alt[0] = np.uint8(int(sentinel[0]) + 1)
    # Move any pre-existing sentinel pixels out of the way so "equals sentinel"
    # after the fill means "this pixel was visited", not "it was already that color".
    already = np.all(arr == sentinel, axis=-1)
    arr[already] = alt
    # Pillow images backed by a numpy buffer are read-only. floodfill writes
    # through the pixel accessor and silently no-ops on a read-only image, so
    # copy onto an owned buffer before filling.
    work = Image.fromarray(arr, "RGBA").copy()
    ImageDraw.floodfill(work, (int(x), int(y)), tuple(int(v) for v in sentinel), thresh=int(tolerance))
    after = np.array(work)
    visited = np.all(after == sentinel, axis=-1)
    return Image.fromarray(visited.astype(np.uint8) * 255, "L")


def fill_where(
    image: Image.Image,
    mask: Image.Image,
    color: tuple[int, int, int, int],
) -> Image.Image:
    arr = np.array(image.convert("RGBA"))
    selected = np.asarray(mask.convert("L")) > 0
    if selected.shape != arr.shape[:2]:
        return image.copy()
    arr[selected] = color
    return Image.fromarray(arr, "RGBA")


def sample_color(image: Image.Image, x: float, y: float) -> tuple[int, int, int, int] | None:
    px, py = int(math.floor(x)), int(math.floor(y))
    if not (0 <= px < image.width and 0 <= py < image.height):
        return None
    pixel = image.convert("RGBA").getpixel((px, py))
    if not isinstance(pixel, tuple):
        return None
    if len(pixel) == 4:
        return int(pixel[0]), int(pixel[1]), int(pixel[2]), int(pixel[3])
    return int(pixel[0]), int(pixel[1]), int(pixel[2]), 255

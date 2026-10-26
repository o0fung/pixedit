"""Brush dabs, shapes, bucket fill, and magic-wand masks.

These functions are pure: they return new images and do not touch the document.
Tools snapshot the layer, call in here, and commit one undo command per gesture.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from pixedit.geom import place_image


def brush_coat(hardness: float) -> float:
    """How solid one pass is at the brightest point of the brush.

    Hardness 0.25 and above is a normal solid brush. Below that the coat
    thins toward 0.10, so hardness 0 needs many passes to look solid.
    """
    hard = min(1.0, max(0.0, float(hardness)))
    if hard >= 0.25:
        return 1.0
    return 0.10 + 0.90 * (hard / 0.25) ** 3


def _falloff(dist: np.ndarray, radius: float, hardness: float) -> np.ndarray:
    """Brush strength at ``dist``, from the core down to 0 at ``radius``.

    ``hardness`` is 0..1. 1 is a hard disk. Lower values keep a solid core of
    radius ``hardness * radius`` and ease out to the edge. Below 0.25 the
    whole shape is scaled by ``brush_coat``, so one dab is a light coat
    instead of a solid center.
    """
    hard = min(1.0, max(0.0, float(hardness)))
    if hard >= 0.999:
        shape = (dist <= radius).astype(np.float32)
    else:
        inner = radius * hard
        shape = np.clip((radius - dist) / max(1e-4, radius - inner), 0.0, 1.0).astype(np.float32)
    coat = brush_coat(hard)
    if coat < 1.0:
        shape *= np.float32(coat)
    return shape


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
        # Straight alpha: coverage drops, the color stays until coverage is
        # gone. Zero RGB at that point so a cleared pixel cannot show the old
        # paint if a viewer treats leftover color as opaque.
        keep = 1.0 - alpha
        region[..., 3] *= keep
        region[region[..., 3] <= 0.5, :3] = 0
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


# The options swatch is fixed. A brush larger than the box fills it; the
# slider still shows the real diameter.
BRUSH_PREVIEW_WIDTH = 176
BRUSH_PREVIEW_HEIGHT = 96
BRUSH_PREVIEW_MARGIN = 8


def brush_preview_image(
    diameter: int,
    hardness: float,
    opacity: float,
    color: tuple[int, int, int, int],
    *,
    erase: bool,
) -> Image.Image:
    """One stamp, using the same edge math as ``dab``.

    ``diameter`` is the brush size in image pixels. ``hardness`` and
    ``opacity`` are 0..1. A small brush stays small. A brush taller than the
    swatch is scaled down so the whole circle stays visible.

    The brush is the foreground color on a checkerboard, so a low opacity
    still reads. The eraser is a hole in a solid chip.
    """
    width, height = BRUSH_PREVIEW_WIDTH, BRUSH_PREVIEW_HEIGHT
    shown = min(height - BRUSH_PREVIEW_MARGIN * 2, max(1, int(diameter)))
    radius = max(0.5, shown / 2.0)
    if erase:
        array = np.empty((height, width, 4), np.uint8)
        array[:] = (168, 176, 184, 255)
    else:
        array = _checker(width, height)
    dab(
        array,
        (width - 1) / 2.0,
        (height - 1) / 2.0,
        radius,
        hardness,
        color,
        opacity,
        erase,
        None,
    )
    return Image.fromarray(array, "RGBA")


def _checker(width: int, height: int, cell: int = 8) -> np.ndarray:
    yy, xx = np.indices((height, width))
    dark = ((xx // cell) + (yy // cell)) % 2 == 0
    image = np.empty((height, width, 4), np.uint8)
    image[dark] = (46, 52, 60, 255)
    image[~dark] = (68, 76, 86, 255)
    return image


def stroke_spacing(radius: float, hardness: float) -> float:
    """Distance between brush stamps, in image pixels.

    A fast drag often arrives as one long segment. Stamps have to overlap
    inside the opaque core — a soft brush's core is smaller than its radius —
    or the stroke shows up as separate circles. The step is 15% of that core.
    For a large hard brush the flat spot between disks is a fraction of a
    pixel, so the step is allowed to grow past one pixel. A small or soft
    brush keeps the sub-pixel floor so the 1px grid cannot open a hole.
    """
    radius = max(0.5, float(radius))
    hard = min(1.0, max(0.0, float(hardness)))
    core = max(0.75, radius * max(hard, 0.25))
    return max(0.35, core * 0.15)


def interpolate(
    start: tuple[float, float], end: tuple[float, float], spacing: float
) -> list[tuple[float, float]]:
    x0, y0 = start
    x1, y1 = end
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist < 1e-6:
        return []
    # Honor sub-pixel spacing. A 1px floor left small brushes with gaps
    # whenever the pointer jumped more than one pixel.
    step = max(0.05, float(spacing))
    count = max(1, int(math.ceil(dist / step)))
    return [
        (x0 + (x1 - x0) * (i / count), y0 + (y1 - y0) * (i / count))
        for i in range(1, count + 1)
    ]


def reflect_point(origin: tuple[float, float], other: tuple[float, float]) -> tuple[float, float]:
    """``other`` mirrored through ``origin``. Used as a missing spline neighbor."""
    return (2.0 * origin[0] - other[0], 2.0 * origin[1] - other[1])


def catmull_segment(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    spacing: float,
) -> list[tuple[float, float]]:
    """Stamp centers along a centripetal Catmull-Rom from ``p1`` to ``p2``.

    ``p1`` is omitted — the previous segment already stamped it — and ``p2``
    is kept. Controls that already lie on the line use ``interpolate``, so a
    straight stroke stays on that line instead of drifting off it.
    """
    if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) < 1e-6:
        return []
    if _nearly_straight(p0, p1, p2, p3):
        return interpolate(p1, p2, spacing)
    samples = _centripetal_samples(p0, p1, p2, p3, spacing)
    return _resample_polyline(samples, spacing)


def _nearly_straight(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
) -> bool:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return True

    def side(point: tuple[float, float]) -> float:
        return abs((point[0] - p1[0]) * dy - (point[1] - p1[1]) * dx) / length

    return side(p0) <= 0.05 and side(p3) <= 0.05


def _knot(time: float, start: tuple[float, float], end: tuple[float, float]) -> float:
    # A zero-length knot makes the Barry-Goldman weights divide by zero.
    # Nudge it so a repeated sample still evaluates on the point.
    dist = max(math.hypot(end[0] - start[0], end[1] - start[1]), 1e-6)
    return time + dist ** 0.5


def _lerp(start: tuple[float, float], end: tuple[float, float], amount: float) -> tuple[float, float]:
    return (
        start[0] + (end[0] - start[0]) * amount,
        start[1] + (end[1] - start[1]) * amount,
    )


def _catmull_at(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    t0: float,
    t1: float,
    t2: float,
    t3: float,
    time: float,
) -> tuple[float, float]:
    def mix(start, end, start_t: float, end_t: float):
        span = end_t - start_t
        if abs(span) < 1e-8:
            return start
        return _lerp(start, end, (time - start_t) / span)

    a1 = mix(p0, p1, t0, t1)
    a2 = mix(p1, p2, t1, t2)
    a3 = mix(p2, p3, t2, t3)
    b1 = mix(a1, a2, t0, t2)
    b2 = mix(a2, a3, t1, t3)
    return mix(b1, b2, t1, t2)


def _centripetal_samples(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    spacing: float,
) -> list[tuple[float, float]]:
    t0 = 0.0
    t1 = _knot(t0, p0, p1)
    t2 = _knot(t1, p1, p2)
    t3 = _knot(t2, p2, p3)
    chord = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    steps = max(4, int(math.ceil(chord / max(0.05, spacing))) * 2)
    span = t2 - t1
    return [
        _catmull_at(p0, p1, p2, p3, t0, t1, t2, t3, t1 + span * (index / steps))
        for index in range(steps + 1)
    ]


def _resample_polyline(
    points: list[tuple[float, float]], spacing: float
) -> list[tuple[float, float]]:
    """Drop ``points[0]`` and walk the rest at ``spacing``, keeping the end."""
    if len(points) < 2:
        return []
    step = max(0.05, float(spacing))
    out: list[tuple[float, float]] = []
    traveled = 0.0
    next_at = step
    for index in range(1, len(points)):
        start = points[index - 1]
        end = points[index]
        hop = math.hypot(end[0] - start[0], end[1] - start[1])
        if hop < 1e-9:
            continue
        while next_at <= traveled + hop:
            out.append(_lerp(start, end, (next_at - traveled) / hop))
            next_at += step
        traveled += hop
    end = points[-1]
    if not out:
        return [end]
    if math.hypot(out[-1][0] - end[0], out[-1][1] - end[1]) > step * 0.25:
        out.append(end)
    else:
        out[-1] = end
    return out


class BrushFootprint:
    """Cached falloff for one radius, hardness, and opacity.

    The fractional pixel of the center is rounded to 1/8. A stroke then reuses
    a handful of kernels instead of building a distance field for every stamp.
    The snap is at most 1/16 of a pixel.
    """

    _STEPS = 8

    def __init__(self, radius: float, hardness: float, opacity: float) -> None:
        self.radius = max(0.5, float(radius))
        self.hardness = min(1.0, max(0.0, float(hardness)))
        self.opacity = min(1.0, max(0.0, float(opacity)))
        self._rad = int(math.ceil(self.radius))
        self._kernels: dict[tuple[int, int], np.ndarray] = {}

    def place(self, cx: float, cy: float) -> tuple[int, int, np.ndarray]:
        """Kernel whose top-left pixel is ``(x0, y0)`` in the same space as ``(cx, cy)``."""
        step = self._STEPS
        qx = round(cx * step) / step
        qy = round(cy * step) / step
        frac_x = qx - math.floor(qx)
        frac_y = qy - math.floor(qy)
        key = (int(round(frac_x * step)) % step, int(round(frac_y * step)) % step)
        kernel = self._kernels.get(key)
        if kernel is None:
            size = self._rad * 2 + 1
            yy, xx = np.ogrid[0:size, 0:size]
            dist = np.sqrt((xx - (self._rad + frac_x)) ** 2 + (yy - (self._rad + frac_y)) ** 2)
            kernel = _falloff(dist, self.radius, self.hardness) * float(self.opacity)
            self._kernels[key] = kernel
        x0 = int(math.floor(qx)) - self._rad
        y0 = int(math.floor(qy)) - self._rad
        return x0, y0, kernel


def _window(
    array_shape: tuple[int, ...],
    origin: tuple[int, int],
    cx: float,
    cy: float,
    footprint: BrushFootprint,
    selection: np.ndarray | None,
) -> tuple[int, int, int, int, np.ndarray, tuple[int, int, int, int]] | None:
    """Clip one footprint to ``array`` and return local bounds, alpha, canvas bounds."""
    x0, y0, kernel = footprint.place(cx, cy)
    height, width = array_shape[:2]
    local_x = x0 - origin[0]
    local_y = y0 - origin[1]
    kh, kw = kernel.shape
    ix0 = max(0, local_x)
    iy0 = max(0, local_y)
    ix1 = min(width, local_x + kw)
    iy1 = min(height, local_y + kh)
    if ix0 >= ix1 or iy0 >= iy1:
        return None
    alpha = kernel[iy0 - local_y : iy0 - local_y + (iy1 - iy0), ix0 - local_x : ix0 - local_x + (ix1 - ix0)]
    if selection is not None:
        alpha = alpha * (selection[iy0:iy1, ix0:ix1].astype(np.float32) / 255.0)
    canvas_box = (ix0 + origin[0], iy0 + origin[1], ix1 + origin[0], iy1 + origin[1])
    return iy0, iy1, ix0, ix1, alpha, canvas_box


def _stroke_gain(radius: float, hardness: float) -> float:
    """Fraction of a dab each moving stamp keeps.

    A coat of 1 paints the dab as-is. Below that, stamps sit ``spacing``
    apart across a falloff about ``radius`` wide, so each stamp keeps
    ``spacing / radius`` of the dab and one crossing adds up to one coat.
    """
    if brush_coat(hardness) >= 1.0:
        return 1.0
    radius = max(0.5, float(radius))
    spacing = stroke_spacing(radius, hardness)
    return min(1.0, spacing / radius)


def _with_stroke_gain(alpha: np.ndarray, gain: float) -> np.ndarray:
    """Thin one stamp so a moving stroke deposits a single coat.

    One crossing of the brush equals one coat, not one coat per stamp.
    ``gain`` is spacing/radius, so the stamps along a straight pass multiply
    back to the dab. The next crossing source-overs again and the paint
    keeps building toward solid. A gain of 1 leaves the dab unchanged.
    """
    if gain >= 1.0:
        return alpha
    covered = np.clip(alpha, 0.0, 1.0)
    return (1.0 - np.power(1.0 - covered, gain)).astype(np.float32)


def _union_box(
    left: tuple[int, int, int, int] | None, right: tuple[int, int, int, int] | None
) -> tuple[int, int, int, int] | None:
    if left is None:
        return right
    if right is None:
        return left
    return (min(left[0], right[0]), min(left[1], right[1]), max(left[2], right[2]), max(left[3], right[3]))


def _composite_coverage(
    array: np.ndarray,
    x0: int,
    y0: int,
    coverage: np.ndarray,
    color: tuple[int, int, int, int],
) -> None:
    """Source-over ``color`` by ``coverage`` (0..1 brush alpha) into ``array``."""
    height, width = coverage.shape
    region = array[y0 : y0 + height, x0 : x0 + width].astype(np.float32)
    src_rgb = np.array(color[:3], dtype=np.float32)
    src_a = coverage * (color[3] / 255.0)
    dst_a = region[..., 3] / 255.0
    out_a = src_a + dst_a * (1.0 - src_a)
    numer = src_rgb * src_a[..., None] + region[..., :3] * dst_a[..., None] * (1.0 - src_a[..., None])
    out_rgb = np.where(out_a[..., None] > 1e-6, numer / np.maximum(out_a[..., None], 1e-6), 0.0)
    region[..., :3] = out_rgb
    region[..., 3] = out_a * 255.0
    array[y0 : y0 + height, x0 : x0 + width] = np.clip(np.rint(region), 0, 255).astype(np.uint8)


def _apply_color_alpha(
    array: np.ndarray,
    alpha: np.ndarray,
    x0: int,
    y0: int,
    color: tuple[int, int, int, int],
    erase: bool,
) -> None:
    height, width = alpha.shape
    region = array[y0 : y0 + height, x0 : x0 + width].astype(np.float32)
    if erase:
        keep = 1.0 - alpha
        region[..., 3] *= keep
        region[region[..., 3] <= 0.5, :3] = 0
        array[y0 : y0 + height, x0 : x0 + width] = np.clip(np.rint(region), 0, 255).astype(np.uint8)
        return
    _composite_coverage(array, x0, y0, alpha, color)


def paint_footprints(
    array: np.ndarray,
    origin: tuple[int, int],
    points: list[tuple[float, float]],
    footprint: BrushFootprint,
    color: tuple[int, int, int, int],
    erase: bool,
    selection: np.ndarray | None,
    *,
    gained: bool = False,
) -> tuple[int, int, int, int] | None:
    """Stamp canvas-space ``points`` into ``array``.

    A fully opaque brush combines the new stamps' coverage and blends that
    once. A translucent brush and the eraser blend every stamp, so coverage
    still builds up (or lifts) one dab at a time. ``gained`` thins a moving
    stroke so one crossing equals one soft coat. Returns the canvas rectangle
    that changed.
    """
    if not points:
        return None
    windows = []
    for cx, cy in points:
        window = _window(array.shape, origin, cx, cy, footprint, selection)
        if window is not None:
            windows.append(window)
    if not windows:
        return None
    gain = _stroke_gain(footprint.radius, footprint.hardness) if gained else 1.0
    opaque = (not erase) and footprint.opacity >= 0.999 and int(color[3]) >= 255
    if opaque and len(windows) > 1:
        return _splat_opaque(array, windows, color, gain)
    box = None
    for iy0, iy1, ix0, ix1, alpha, canvas_box in windows:
        _apply_color_alpha(array, _with_stroke_gain(alpha, gain), ix0, iy0, color, erase)
        box = _union_box(box, canvas_box)
    return box


def _splat_opaque(
    array: np.ndarray,
    windows: list[tuple[int, int, int, int, np.ndarray, tuple[int, int, int, int]]],
    color: tuple[int, int, int, int],
    gain: float,
) -> tuple[int, int, int, int]:
    """One source-over of repeated opaque stamps. Same color, so the product of
    the leftover coverage matches blending each stamp in turn.
    """
    ix0 = min(window[2] for window in windows)
    iy0 = min(window[0] for window in windows)
    ix1 = max(window[3] for window in windows)
    iy1 = max(window[1] for window in windows)
    keep = np.ones((iy1 - iy0, ix1 - ix0), dtype=np.float32)
    box = None
    for wy0, wy1, wx0, wx1, alpha, canvas_box in windows:
        stamp = _with_stroke_gain(alpha, gain)
        keep[wy0 - iy0 : wy1 - iy0, wx0 - ix0 : wx1 - ix0] *= 1.0 - stamp
        box = _union_box(box, canvas_box)
    _composite_coverage(array, ix0, iy0, 1.0 - keep, color)
    assert box is not None
    return box


def paint_mask_footprints(
    array: np.ndarray,
    origin: tuple[int, int],
    points: list[tuple[float, float]],
    footprint: BrushFootprint,
    erase: bool,
    selection: np.ndarray | None,
    *,
    gained: bool = False,
) -> tuple[int, int, int, int] | None:
    """Stamp a redaction mask. ``array`` is HxW uint8, 255 = fully redacted.

    ``gained`` uses the same per-stamp thinning as a moving brush stroke.
    """
    gain = _stroke_gain(footprint.radius, footprint.hardness) if gained else 1.0
    box = None
    for cx, cy in points:
        window = _window(array.shape, origin, cx, cy, footprint, selection)
        if window is None:
            continue
        iy0, iy1, ix0, ix1, alpha, canvas_box = window
        alpha = _with_stroke_gain(alpha, gain)
        region = array[iy0:iy1, ix0:ix1].astype(np.float32)
        if erase:
            region *= 1.0 - alpha
        else:
            region = region + (255.0 - region) * alpha
        array[iy0:iy1, ix0:ix1] = np.clip(np.rint(region), 0, 255).astype(np.uint8)
        box = _union_box(box, canvas_box)
    return box


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
    """Paint ``points`` as one stroke. Overlapping soft stamps add up to one coat."""
    array = np.array(image.convert("RGBA"))
    if points:
        sel = None if selection is None else np.asarray(selection)
        paint_footprints(
            array,
            (0, 0),
            list(points),
            BrushFootprint(radius, hardness, opacity),
            color,
            erase,
            sel,
            gained=True,
        )
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
    """Paint ``points`` into a mask as one stroke, with the same soft-coat rule."""
    array = np.array(mask.convert("L"))
    if points:
        sel = None if selection is None else np.asarray(selection)
        paint_mask_footprints(
            array,
            (0, 0),
            list(points),
            BrushFootprint(radius, hardness, opacity),
            erase,
            sel,
            gained=True,
        )
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
    """L mask of pixels 4-connected to ``(x, y)`` and within tolerance of it.

    Tolerance is the maximum per-channel absolute RGBA difference from the
    seed (0 = exact match). Pillow's ``floodfill(thresh=)`` cannot be used
    directly: its threshold is the sum of channel deltas, and it returns
    without filling when the seed is already within that distance of the
    replacement color. A transparent pixel is only 10 away from a dark
    sentinel, so the default tolerance of 32 used to fill nothing.

    Membership is decided here. Pillow only walks the binary mask, with a
    fill value (128) that matches neither candidate (255) nor blocked (0).
    """
    rgba = image.convert("RGBA")
    width, height = rgba.size
    if width < 1 or height < 1 or not (0 <= x < width and 0 <= y < height):
        return Image.new("L", (max(width, 1), max(height, 1)), 0)
    arr = np.asarray(rgba)
    seed = arr[int(y), int(x)].astype(np.int16)
    # Max, not sum: tolerance 32 means each channel may differ by 32, which
    # is the paint-bucket convention. Alpha is included so transparent and
    # opaque pixels stay distinct unless the tolerance is wide enough.
    delta = np.max(np.abs(arr.astype(np.int16) - seed), axis=-1)
    similar = delta <= max(0, int(tolerance))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[similar] = 255
    # fromarray() shares a buffer floodfill will not write. copy() owns it.
    work = Image.fromarray(mask, "L").copy()
    ImageDraw.floodfill(work, (int(x), int(y)), 128, thresh=0)
    filled = np.asarray(work) == 128
    return Image.fromarray(filled.astype(np.uint8) * 255, "L")


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


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _coverage(arr: np.ndarray, seed: np.ndarray, tolerance: int) -> np.ndarray:
    """How far each pixel is from the seed, 0 (same) to 1 (fully different).

    A transparent seed stores paint coverage in alpha. An opaque seed has the
    old color baked into the RGB, so the largest channel change is the coverage.
    """
    if int(seed[3]) <= tolerance:
        return arr[..., 3].astype(np.float32) / 255.0
    delta = np.max(np.abs(arr[..., :3].astype(np.int16) - seed[:3].astype(np.int16)), axis=-1)
    return delta.astype(np.float32) / 255.0


def _max_neighbor(values: np.ndarray) -> np.ndarray:
    """Highest 8-connected neighbor, ignoring the pixel itself."""
    padded = np.pad(values, 1, mode="constant", constant_values=0)
    stacked = np.stack(
        [
            padded[0:-2, 0:-2],
            padded[0:-2, 1:-1],
            padded[0:-2, 2:],
            padded[1:-1, 0:-2],
            padded[1:-1, 2:],
            padded[2:, 0:-2],
            padded[2:, 1:-1],
            padded[2:, 2:],
        ],
        axis=0,
    )
    return np.max(stacked, axis=0)


def _src_over(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Straight-alpha source over destination. Both are Nx4 float, 0..255."""
    src_a = np.clip(src[:, 3:4] / 255.0, 0.0, 1.0)
    dst_a = np.clip(dst[:, 3:4] / 255.0, 0.0, 1.0)
    out_a = src_a + dst_a * (1.0 - src_a)
    out_rgb = src[:, :3] * src_a + dst[:, :3] * dst_a * (1.0 - src_a)
    out_rgb = np.where(out_a > 1e-6, out_rgb / np.maximum(out_a, 1e-6), 0.0)
    out = np.concatenate([out_rgb, out_a * 255.0], axis=1)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _inner_stroke_fringe(
    arr: np.ndarray,
    filled: np.ndarray,
    seed: np.ndarray,
    tolerance: int,
) -> np.ndarray:
    """Pixels of the anti-aliased inner edge of a stroke around ``filled``.

    The bucket stops on a partial dab, so a brush outline keeps a rim of the
    old background between the fill and the solid paint. Those pixels sit on
    a short gradient from the seed color up to the stroke. Walk out at most
    two pixels and take that gradient.

    Do not cross onto seed-colored pixels the flood did not reach — that is
    the outside of the shape. Do not take a flat patch of a different color,
    and do not take the solid body of the stroke. A hard outline has no
    partial pixels, so it is left unchanged.
    """
    tolerance = max(0, int(tolerance))
    seed_vec = np.asarray(seed, dtype=np.int16)
    delta = np.max(np.abs(arr.astype(np.int16) - seed_vec), axis=-1)
    seed_like = delta <= tolerance
    coverage = _coverage(arr, seed_vec, tolerance)
    # Solid paint is the stroke body (or another opaque object). The rim is
    # the partial band below that.
    solid = coverage >= 0.90
    exterior = seed_like & ~filled
    touches_outside = _dilate8(exterior)
    neighbor_cover = _max_neighbor(coverage)
    # A real dab edge gets more covered toward the stroke. A flat region of
    # another color does not, so the fill does not nibble its border.
    rising = neighbor_cover >= coverage + (8.0 / 255.0)

    fringe = np.zeros(filled.shape, dtype=bool)
    region = filled.copy()
    for _step in range(2):
        border = _dilate8(region) & ~region
        take = border & ~seed_like & ~solid & ~touches_outside & rising
        take &= (coverage > 0.03) & (coverage < 0.90)
        if not np.any(take):
            break
        fringe |= take
        region |= take
    return fringe


def _neighbor_range(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lowest and highest 8-connected neighbor, ignoring the pixel itself.

    Edge padding repeats the border. A pad of 0 would look like the seed
    (coverage 0) and pull the canvas edge into the rim.
    """
    padded = np.pad(values, 1, mode="edge")
    stacked = np.stack(
        [
            padded[0:-2, 0:-2],
            padded[0:-2, 1:-1],
            padded[0:-2, 2:],
            padded[1:-1, 0:-2],
            padded[1:-1, 2:],
            padded[2:, 0:-2],
            padded[2:, 1:-1],
            padded[2:, 2:],
        ],
        axis=0,
    )
    return np.min(stacked, axis=0), np.max(stacked, axis=0)


def _outer_blend_fringe(
    arr: np.ndarray,
    filled: np.ndarray,
    seed: np.ndarray,
    tolerance: int,
) -> np.ndarray:
    """Anti-aliased pixels just outside a wand region.

    The flood stops at tolerance, so a soft edge — a blend of the clicked
    color and whatever is outside — stays unselected. Cut then leaves that
    outline on the source. Walk out at most two pixels and take the blend.

    A pixel counts only when its color sits between a closer neighbor and a
    farther one, by more than 8 levels. A hard edge has no such pixel. A flat
    patch of another color does not either, and a slow gradient changes too
    little per pixel to pass. Alpha is part of the distance, so a fade on a
    transparent layer is a blend, and the empty pixels past that fade are not.

    Flow:
    1. Distance from the seed on all four channels (0 same, 1 fully different).
    2. Mark pixels strictly between a nearer and a farther neighbor.
    3. Grow the region by one pixel, twice, taking only those blend pixels
       the flood itself did not already accept.
    """
    tolerance = max(0, int(tolerance))
    seed_vec = np.asarray(seed, dtype=np.int16)
    delta = np.max(np.abs(arr.astype(np.int16) - seed_vec), axis=-1)
    coverage = delta.astype(np.float32) / 255.0
    seed_like = delta <= tolerance
    # 8 levels matches the fill's rising test. A sky or a shadow steps less
    # than that, so the rim stops at a real edge.
    margin = 8.0 / 255.0
    nearer, farther = _neighbor_range(coverage)
    between = (nearer + margin < coverage) & (coverage < farther - margin)

    fringe = np.zeros(filled.shape, dtype=bool)
    region = filled.copy()
    for _step in range(2):
        border = _dilate8(region) & ~region
        take = border & ~seed_like & between
        if not np.any(take):
            break
        fringe |= take
        region |= take
    return fringe


def wand_mask(image: Image.Image, x: int, y: int, tolerance: int) -> Image.Image:
    """Magic-wand mask: the connected flood plus its anti-aliased rim.

    ``connected_mask`` stays the flood alone. The fill tool uses that flood
    and composites a stroke's inner rim on its own; folding the rim into the
    flood would paint over that edge.
    """
    base = connected_mask(image, int(x), int(y), tolerance)
    filled = np.asarray(base) > 0
    if not np.any(filled):
        return base
    rgba = image.convert("RGBA")
    width, height = rgba.size
    if not (0 <= int(y) < height and 0 <= int(x) < width):
        return base
    arr = np.asarray(rgba)
    fringe = _outer_blend_fringe(arr, filled, arr[int(y), int(x)], tolerance)
    if not np.any(fringe):
        return base
    out = np.array(base)
    out[fringe] = 255
    return Image.fromarray(out, "L")


def _recolor_fringe(
    arr: np.ndarray,
    fringe: np.ndarray,
    seed: np.ndarray,
    color: tuple[int, int, int, int],
    tolerance: int,
) -> None:
    """Put the fill under the stroke coverage on ``fringe`` pixels.

    Partial dabs on a transparent layer already store the brush color and an
    alpha. Partial dabs on an opaque background have the old color mixed in;
    recover the brush color from that mix, then composite it over the fill.
    Either way the old background stops showing through the edge.
    """
    if not np.any(fringe):
        return
    original = arr[fringe].astype(np.float32)
    fill = np.array(color, dtype=np.float32)
    dst = np.broadcast_to(fill, original.shape).copy()
    if int(seed[3]) <= max(0, int(tolerance)):
        arr[fringe] = _src_over(original, dst)
        return
    seed_rgb = seed[:3].astype(np.float32)
    delta = np.max(np.abs(original[:, :3] - seed_rgb), axis=1)
    cover = np.clip(delta / 255.0, 0.0, 1.0)
    brush = (original[:, :3] - (1.0 - cover)[:, None] * seed_rgb) / np.maximum(cover, 1e-4)[:, None]
    brush = np.clip(brush, 0.0, 255.0)
    src = np.concatenate([brush, cover[:, None] * 255.0], axis=1)
    arr[fringe] = _src_over(src, dst)


def fill_region(
    image: Image.Image,
    x: int,
    y: int,
    tolerance: int,
    color: tuple[int, int, int, int],
    selection: Image.Image | None = None,
) -> Image.Image:
    """Flood-fill from ``(x, y)`` and close the gap against a soft stroke.

    ``selection``, when given, is a layer-local mask. Pixels outside it are
    not recolored, including the anti-aliased rim.
    """
    rgba = image.convert("RGBA")
    mask = connected_mask(rgba, int(x), int(y), tolerance)
    arr = np.array(rgba)
    filled = np.asarray(mask.convert("L")) > 0
    if selection is not None:
        sel = np.asarray(selection.convert("L"))
        if sel.shape == filled.shape:
            filled &= sel > 0
    if not np.any(filled) or not (0 <= int(y) < arr.shape[0] and 0 <= int(x) < arr.shape[1]):
        return Image.fromarray(arr, "RGBA")
    seed = arr[int(y), int(x)].copy()
    fringe = _inner_stroke_fringe(arr, filled, seed, tolerance)
    if selection is not None:
        sel = np.asarray(selection.convert("L"))
        if sel.shape == fringe.shape:
            fringe &= sel > 0
    _recolor_fringe(arr, fringe, seed, color, tolerance)
    arr[filled] = color
    return Image.fromarray(arr, "RGBA")


def fill_document(
    image: Image.Image,
    origin_x: int,
    origin_y: int,
    doc_width: int,
    doc_height: int,
    x: int,
    y: int,
    tolerance: int,
    color: tuple[int, int, int, int],
    selection: Image.Image | None = None,
) -> tuple[Image.Image, int, int]:
    """Flood-fill from a document pixel and return the layer image and origin.

    The bitmap may be smaller than the canvas. Pixels it does not cover are
    transparent, so a click in that margin is a real seed. The flood stops at
    a pixel outside tolerance, at the selection, and at the document edge.
    Pixels that already sat outside the document are kept and are not filled.

    Flow:
    1. Paste the layer's overlap onto a transparent document-sized plate.
    2. Flood-fill that plate. A document selection lines up with the plate.
    3. A layer that hangs off the document keeps that overhang. The filled
       plate replaces only the document window. A layer inside the document
       becomes the plate, and the caller crops the transparent margin.
    """
    width = int(doc_width)
    height = int(doc_height)
    ox, oy = int(origin_x), int(origin_y)
    plate = place_image(image.convert("RGBA"), ox, oy, width, height)
    # A selection stored at another size is placed at the document origin,
    # the same way a click test reads it. The plate is the document, so the
    # mask lines up with the flood.
    if selection is not None and selection.size != (width, height):
        selection = place_image(selection.convert("L"), 0, 0, width, height, 0)
    filled = fill_region(plate, int(x), int(y), tolerance, color, selection)

    # The plate clipped the overhang away. Copy the old bitmap first so those
    # pixels stay, then replace the document window with the filled plate.
    if ox < 0 or oy < 0 or ox + image.width > width or oy + image.height > height:
        nx0, ny0 = min(ox, 0), min(oy, 0)
        nx1 = max(ox + image.width, width)
        ny1 = max(oy + image.height, height)
        out = np.zeros((ny1 - ny0, nx1 - nx0, 4), dtype=np.uint8)
        src = np.asarray(image.convert("RGBA"))
        out[oy - ny0 : oy - ny0 + src.shape[0], ox - nx0 : ox - nx0 + src.shape[1]] = src
        out[0 - ny0 : height - ny0, 0 - nx0 : width - nx0] = np.asarray(filled.convert("RGBA"))
        return Image.fromarray(out, "RGBA"), nx0, ny0
    return filled, 0, 0


def wand_document(
    image: Image.Image,
    origin_x: int,
    origin_y: int,
    doc_width: int,
    doc_height: int,
    x: int,
    y: int,
    tolerance: int,
) -> Image.Image:
    """Magic-wand mask from a document pixel, including empty space outside the bitmap.

    The bitmap may be smaller than the canvas. Pixels it does not cover are
    transparent, the same plate ``fill_document`` floods, so a click in that
    margin is a real seed. The walk stops at a pixel outside tolerance and at
    the document edge. Pixels that hang off the document are clipped with the
    plate and are not selected.

    The current selection does not clip this mask. Fill uses a selection as a
    paint limit; the wand is what creates the selection.

    Flow:
    1. Paste the layer's overlap onto a transparent document-sized plate.
    2. Run the bitmap wand (connected flood plus anti-aliased rim) on that plate.
    """
    width = int(doc_width)
    height = int(doc_height)
    plate = place_image(image.convert("RGBA"), int(origin_x), int(origin_y), width, height)
    return wand_mask(plate, int(x), int(y), tolerance)


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

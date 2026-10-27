"""Small copies used while a drag is in progress.

The stored pixels stay full resolution. A live grade, resample, or gradient
runs on a screen-sized stand-in when the layer is large, and the release
computes the real bitmap once. A transform drag flattens the selected layers
into one of those stand-ins so the canvas can move it without rebuilding the
document. A picture that already fits on screen is left alone, so a small
edit stays exact the whole way through.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image

# Longest edge of a stand-in. Above this, a drag does not get sharper.
PREVIEW_MAX_EDGE = 1024


def preview_scale(width: int, height: int, zoom: float = 1.0) -> float:
    """Scale from document pixels to the live stand-in. ``1`` means full size.

    The stand-in follows the on-screen size, with a floor so a tiny zoom is
    still readable, and it never exceeds ``PREVIEW_MAX_EDGE`` or the source.
    """
    longest = max(int(width), int(height), 1)
    screen = longest * max(float(zoom), 0.05)
    target = min(float(PREVIEW_MAX_EDGE), max(screen, 64.0))
    if target >= longest:
        return 1.0
    return target / float(longest)


def downscale(image: Image.Image, scale: float) -> Image.Image:
    """Box-filter ``image`` by ``scale``. ``scale`` near 1 returns ``image``."""
    if scale >= 0.999:
        return image
    width = max(1, int(round(image.width * scale)))
    height = max(1, int(round(image.height * scale)))
    if (width, height) == image.size:
        return image
    return image.resize((width, height), Image.Resampling.BOX)


def resize_for_preview(
    image: Image.Image,
    size: tuple[int, int],
    zoom: float,
    *,
    final: bool,
) -> Image.Image:
    """Resize to ``size``. Lanczos when committing or when the picture is small.

    A large drag resamples a reduced copy with bilinear, then stretches that
    result to ``size`` so layer bounds stay on the real pixel grid.
    """
    target = (max(1, int(size[0])), max(1, int(size[1])))
    full = final or (
        preview_scale(image.width, image.height, zoom) >= 1.0
        and preview_scale(target[0], target[1], zoom) >= 1.0
    )
    if image.size == target and full:
        return image.copy()
    if full:
        return image.resize(target, Image.Resampling.LANCZOS)
    scale = min(
        preview_scale(image.width, image.height, zoom),
        preview_scale(target[0], target[1], zoom),
    )
    if scale >= 0.999:
        return image.resize(target, Image.Resampling.BILINEAR)
    small = downscale(image, scale)
    mid_size = (
        max(1, int(round(target[0] * scale))),
        max(1, int(round(target[1] * scale))),
    )
    mid = small.resize(mid_size, Image.Resampling.BILINEAR)
    if mid.size == target:
        return mid
    return mid.resize(target, Image.Resampling.BILINEAR)


def rotated_size(width: int, height: int, angle_ccw: float) -> tuple[int, int]:
    """Output size of Pillow's ``rotate(..., expand=True)`` for this angle.

    The corner math matches Pillow, including the 90-degree fast path, so a
    stand-in can be stretched to the size the full-resolution rotate will use.
    """
    angle = float(angle_ccw) % 360.0
    if angle == 0.0 or angle == 180.0:
        return int(width), int(height)
    if angle in (90.0, 270.0):
        return int(height), int(width)
    center = (width / 2.0, height / 2.0)
    theta = -math.radians(angle)
    matrix = [
        round(math.cos(theta), 15),
        round(math.sin(theta), 15),
        0.0,
        round(-math.sin(theta), 15),
        round(math.cos(theta), 15),
        0.0,
    ]

    def transform(x: float, y: float) -> tuple[float, float]:
        a, b, c, d, e, f = matrix
        return a * x + b * y + c, d * x + e * y + f

    matrix[2], matrix[5] = transform(-center[0], -center[1])
    matrix[2] += center[0]
    matrix[5] += center[1]
    xs: list[float] = []
    ys: list[float] = []
    for x, y in ((0, 0), (width, 0), (width, height), (0, height)):
        transformed_x, transformed_y = transform(x, y)
        xs.append(transformed_x)
        ys.append(transformed_y)
    out_w = math.ceil(max(xs)) - math.floor(min(xs))
    out_h = math.ceil(max(ys)) - math.floor(min(ys))
    return int(out_w), int(out_h)


def rotate_for_preview(
    image: Image.Image,
    angle_ccw: float,
    zoom: float,
    *,
    final: bool,
) -> Image.Image:
    """Rotate like Pillow. A large drag rotates a reduced copy, then stretches it."""
    if final or preview_scale(image.width, image.height, zoom) >= 1.0:
        return image.rotate(angle_ccw, resample=Image.Resampling.BICUBIC, expand=True)
    full = rotated_size(image.width, image.height, angle_ccw)
    small = downscale(image, preview_scale(image.width, image.height, zoom))
    rotated = small.rotate(angle_ccw, resample=Image.Resampling.BICUBIC, expand=True)
    if rotated.size == full:
        return rotated
    return rotated.resize(full, Image.Resampling.BILINEAR)


def set_preview_source(layer, image: Image.Image | None) -> None:
    """Show ``image`` in place of ``layer.image`` until the drag commits.

    The stand-in covers the same rectangle as the layer bitmap. Composite
    scales it. It is not part of the layer dataclass, so save and undo ignore it.
    """
    if image is None:
        if hasattr(layer, "preview_source"):
            delattr(layer, "preview_source")
        return
    layer.preview_source = image


@dataclass
class DragProxy:
    """Low-resolution flatten of the layers a transform drag is moving.

    ``image`` covers ``box`` (x, y, width, height) in document pixels at
    mouse-down. ``live_box`` is where that picture is drawn for a move or a
    scale. Rotation leaves ``live_box`` on ``box`` and sets ``angle`` in
    clockwise degrees. ``layer_ids`` stay out of the plates drawn under and
    over the stand-in, so the sharp composite is not rebuilt on each move.
    """

    image: Image.Image
    box: tuple[float, float, float, float]
    layer_ids: tuple[str, ...]
    live_box: tuple[float, float, float, float]
    angle: float = 0.0


def _drag_keeps_full_resample(layers: list, stack: list) -> bool:
    """True when a flattened stand-in would not match the release.

    A mask is resized with the picture, and blur or noise read the real
    pixels. Baking either into the preview would change what the drag shows.
    """
    from pixedit.model import masks_for

    for layer in layers:
        if layer.kind == "redaction":
            return True
        if any(mask.visible and mask.opacity > 0 for mask in masks_for(stack, layer)):
            return True
    return False


def _stand_slot(start: float, length: float, origin: float, span: float, pixels: int) -> tuple[int, int]:
    """Pixel origin and size of one document interval inside the stand-in."""
    if span <= 0 or pixels < 1 or length <= 0:
        return 0, 0
    begin = (start - origin) / span * pixels
    end = (start + length - origin) / span * pixels
    i0 = int(round(begin))
    i1 = int(round(end))
    if i1 < i0:
        i0, i1 = i1, i0
    if i1 <= i0:
        i1 = i0 + 1
    return i0, i1 - i0


def _blend_stand_in(canvas, image: Image.Image, x: int, y: int, opacity: float, mode: str) -> None:
    """Blend ``image`` into the small float plate. Coordinates may fall outside."""
    from pixedit.composite import blend_arrays, pil_to_float

    src = pil_to_float(image)
    height, width = canvas.shape[:2]
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(width, int(x) + src.shape[1])
    y1 = min(height, int(y) + src.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    cropped = src[y0 - int(y) : y0 - int(y) + (y1 - y0), x0 - int(x) : x0 - int(x) + (x1 - x0)]
    canvas[y0:y1, x0:x1] = blend_arrays(canvas[y0:y1, x0:x1], cropped, opacity, mode)


def flatten_for_drag(
    layers: list,
    stack: list,
    box: tuple[float, float, float, float],
    zoom: float,
) -> DragProxy | None:
    """Flatten ``layers`` once for a transform drag, or None to keep resampling.

    The stand-in is the union of the selected layers, gaps included, at the
    on-screen size capped by ``PREVIEW_MAX_EDGE``. None means the pictures
    already fit that cap, or a mask is in the drag: those keep the per-move
    resample so the preview and the release stay the same.
    """
    if not layers or box[2] < 1 or box[3] < 1:
        return None
    if _drag_keeps_full_resample(layers, stack):
        return None
    pixel_w = max(1, int(round(box[2])))
    pixel_h = max(1, int(round(box[3])))
    scale = preview_scale(pixel_w, pixel_h, zoom)
    if scale >= 1.0:
        return None
    small_w = max(1, int(round(box[2] * scale)))
    small_h = max(1, int(round(box[3] * scale)))
    import numpy as np

    from pixedit.composite import float_to_pil

    plate = np.zeros((small_h, small_w, 4), dtype=np.float32)
    for layer in layers:
        if not layer.visible or layer.opacity <= 0 or layer.kind == "redaction":
            continue
        image = layer.image
        if image.width < 1 or image.height < 1:
            continue
        slot_x, slot_w = _stand_slot(layer.x, image.width, box[0], box[2], small_w)
        slot_y, slot_h = _stand_slot(layer.y, image.height, box[1], box[3], small_h)
        if slot_w < 1 or slot_h < 1:
            continue
        resized = image.convert("RGBA").resize((slot_w, slot_h), Image.Resampling.BOX)
        _blend_stand_in(plate, resized, slot_x, slot_y, layer.opacity, layer.blend_mode)
    return DragProxy(
        image=float_to_pil(plate),
        box=box,
        layer_ids=tuple(layer.id for layer in layers),
        live_box=box,
    )


def layer_sample(layer) -> tuple[Image.Image, float, float]:
    """Bitmap to read, and the scale from layer pixels into that bitmap."""
    image = layer.image
    preview = getattr(layer, "preview_source", None)
    if preview is None or image.width < 1 or image.height < 1:
        return image, 1.0, 1.0
    if preview.size == image.size:
        return preview, 1.0, 1.0
    return preview, preview.width / float(image.width), preview.height / float(image.height)

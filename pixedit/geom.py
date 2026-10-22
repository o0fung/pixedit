"""Placement, crop, trim, and layer/canvas transforms.

Rectangles are half-open on the right and bottom: ``(left, top, right, bottom)``.
Canvas rotation and flip are defined as "the flattened composite transforms
the same way", and the tests check that directly.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from PIL import Image

from pixedit.model import Document, Layer


def alpha_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    """Tight bounds of pixels that are not empty. ``None`` when nothing is there."""
    if image.mode == "L":
        mask = np.asarray(image) > 0
    else:
        mask = np.asarray(image.convert("RGBA"))[..., 3] > 0
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def place_image(
    image: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
    fill: int | tuple[int, ...] = 0,
) -> Image.Image:
    """Paste ``image`` onto a new canvas, clipping anything outside it."""
    if image.mode == "L":
        canvas = Image.new("L", (width, height), int(fill) if not isinstance(fill, tuple) else 0)
    else:
        color = fill if isinstance(fill, tuple) else (0, 0, 0, 0)
        canvas = Image.new("RGBA", (width, height), color)
        image = image.convert("RGBA")
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + image.width)
    y1 = min(height, y + image.height)
    if x1 <= x0 or y1 <= y0:
        return canvas
    src = image.crop((x0 - x, y0 - y, x1 - x, y1 - y))
    canvas.paste(src, (x0, y0))
    return canvas


def crop_placed(
    image: Image.Image,
    x: int,
    y: int,
    left: int,
    top: int,
    width: int,
    height: int,
) -> tuple[Image.Image, int, int]:
    """Crop a placed image to a window and return the image plus its new origin.

    The new origin is relative to the crop window. A layer that misses the
    window becomes a 1×1 empty image at (0, 0) so the layer itself survives.
    """
    ix0 = max(x, left)
    iy0 = max(y, top)
    ix1 = min(x + image.width, left + width)
    iy1 = min(y + image.height, top + height)
    if ix1 <= ix0 or iy1 <= iy0:
        empty_mode = "L" if image.mode == "L" else "RGBA"
        empty_fill = 0 if empty_mode == "L" else (0, 0, 0, 0)
        return Image.new(empty_mode, (1, 1), empty_fill), 0, 0
    cropped = image.crop((ix0 - x, iy0 - y, ix1 - x, iy1 - y))
    return cropped, ix0 - left, iy0 - top


def scaled_top_left(x: int, y: int, old_w: int, old_h: int, new_w: int, new_h: int) -> tuple[int, int]:
    """Top-left that keeps the layer center fixed while its bitmap changes size."""
    cx = x + old_w / 2.0
    cy = y + old_h / 2.0
    return int(round(cx - new_w / 2.0)), int(round(cy - new_h / 2.0))


def size_from_anchor(
    anchor: tuple[float, float],
    mouse: tuple[float, float],
    *,
    lock_aspect: bool,
    aspect: float,
    min_size: float = 1.0,
) -> tuple[float, float, float, float]:
    """Build a rectangle from a fixed corner and the pointer.

    Returns ``(left, top, width, height)``. When ``lock_aspect`` is set, the
    box keeps ``aspect`` (width / height) and still contains the anchor.
    """
    ax, ay = anchor
    mx, my = mouse
    width = max(min_size, abs(mx - ax))
    height = max(min_size, abs(my - ay))
    if lock_aspect and aspect > 0:
        if width / height > aspect:
            width = height * aspect
        else:
            height = width / aspect
        width = max(min_size, width)
        height = max(min_size, height)
    left = ax if mx >= ax else ax - width
    top = ay if my >= ay else ay - height
    return left, top, width, height


def _transformed_layers(
    document: Document,
    transform: Callable[[Image.Image], Image.Image],
    new_width: int,
    new_height: int,
) -> tuple[list[Layer], Image.Image | None]:
    """Apply one full-canvas image transform to every layer and the selection.

    Each raster is materialized onto the old canvas, transformed, then trimmed
    back to its opaque bounds so layer positions stay meaningful. Redaction
    masks stay canvas-sized because their coordinates are the document.
    Text is rasterized: a rotated bitmap no longer matches the font metadata.
    """
    layers: list[Layer] = []
    for layer in document.layers:
        cloned = layer.clone()
        if layer.kind == "redaction":
            mask = layer.mask or Image.new("L", (1, 1), 0)
            full = place_image(mask, layer.x, layer.y, document.width, document.height, 0)
            cloned.mask = transform(full)
            cloned.x = 0
            cloned.y = 0
            cloned.image = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        else:
            full = place_image(layer.image, layer.x, layer.y, document.width, document.height)
            rotated = transform(full.convert("RGBA"))
            box = alpha_bbox(rotated)
            if box is None:
                cloned.image = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
                cloned.x = 0
                cloned.y = 0
            else:
                cloned.image = rotated.crop(box)
                cloned.x = box[0]
                cloned.y = box[1]
            cloned.kind = "raster"
            cloned.text = None
        layers.append(cloned)
    selection = None
    if document.selection is not None:
        full_sel = place_image(document.selection, 0, 0, document.width, document.height, 0)
        selection = transform(full_sel)
        if selection.size != (new_width, new_height):
            selection = selection.resize((new_width, new_height), Image.Resampling.NEAREST)
    return layers, selection


def canvas_rotate_cw(document: Document) -> tuple[list[Layer], int, int, Image.Image | None]:
    new_w, new_h = document.height, document.width
    layers, selection = _transformed_layers(
        document,
        lambda image: image.transpose(Image.Transpose.ROTATE_270),
        new_w,
        new_h,
    )
    return layers, new_w, new_h, selection


def canvas_rotate_ccw(document: Document) -> tuple[list[Layer], int, int, Image.Image | None]:
    new_w, new_h = document.height, document.width
    layers, selection = _transformed_layers(
        document,
        lambda image: image.transpose(Image.Transpose.ROTATE_90),
        new_w,
        new_h,
    )
    return layers, new_w, new_h, selection


def canvas_flip(document: Document, horizontal: bool) -> tuple[list[Layer], int, int, Image.Image | None]:
    method = Image.Transpose.FLIP_LEFT_RIGHT if horizontal else Image.Transpose.FLIP_TOP_BOTTOM
    layers, selection = _transformed_layers(
        document,
        lambda image: image.transpose(method),
        document.width,
        document.height,
    )
    return layers, document.width, document.height, selection


def crop_document(
    document: Document, left: int, top: int, width: int, height: int
) -> tuple[list[Layer], Image.Image | None]:
    """Return new layers and selection for a crop window inside the canvas."""
    left = int(left)
    top = int(top)
    width = max(1, int(width))
    height = max(1, int(height))
    layers: list[Layer] = []
    for layer in document.layers:
        cloned = layer.clone()
        if layer.kind == "redaction":
            mask = layer.mask or Image.new("L", (document.width, document.height), 0)
            placed = place_image(mask, layer.x, layer.y, document.width, document.height, 0)
            cloned.mask = placed.crop((left, top, left + width, top + height))
            cloned.x = 0
            cloned.y = 0
        else:
            image, x, y = crop_placed(layer.image, layer.x, layer.y, left, top, width, height)
            cloned.image = image
            cloned.x = x
            cloned.y = y
        layers.append(cloned)
    selection = None
    if document.selection is not None:
        full = place_image(document.selection, 0, 0, document.width, document.height, 0)
        selection = full.crop((left, top, left + width, top + height))
    return layers, selection


def trim_rect(document: Document) -> tuple[int, int, int, int] | None:
    """Canvas crop that removes transparent edges of every visible layer."""
    min_x, min_y = document.width, document.height
    max_x, max_y = 0, 0
    found = False
    for layer in document.layers:
        if not layer.visible:
            continue
        image = layer.mask if layer.kind == "redaction" and layer.mask is not None else layer.image
        box = alpha_bbox(image)
        if box is None:
            continue
        left = max(0, layer.x + box[0])
        top = max(0, layer.y + box[1])
        right = min(document.width, layer.x + box[2])
        bottom = min(document.height, layer.y + box[3])
        if right <= left or bottom <= top:
            continue
        found = True
        min_x = min(min_x, left)
        min_y = min(min_y, top)
        max_x = max(max_x, right)
        max_y = max(max_y, bottom)
    if not found:
        return None
    return min_x, min_y, max_x - min_x, max_y - min_y


def scale_visual(state: dict[str, Any], new_w: int, new_h: int) -> dict[str, Any]:
    """Scale a visual snapshot around its center.

    Text stays editable: the font size tracks the width ratio and the glyphs
    are rendered again. Redaction masks use nearest-neighbor so the region
    stays binary. Other layers resample the bitmap and become plain rasters.
    """
    new_w = max(1, int(new_w))
    new_h = max(1, int(new_h))
    out = visual_state_like(state)
    if state["kind"] == "redaction" and state["mask"] is not None:
        mask = state["mask"]
        out["mask"] = mask.resize((new_w, new_h), Image.Resampling.NEAREST)
        out["x"], out["y"] = scaled_top_left(state["x"], state["y"], mask.width, mask.height, new_w, new_h)
        return out
    if state["kind"] == "text" and state["text"] is not None:
        from pixedit.textutil import render_text

        info = out["text"]
        old_w = max(1, state["image"].width)
        info.size = max(1, int(round(info.size * (new_w / old_w))))
        image = render_text(info)
        out["image"] = image
        out["kind"] = "text"
        out["x"], out["y"] = scaled_top_left(
            state["x"], state["y"], state["image"].width, state["image"].height, image.width, image.height
        )
        return out
    image = state["image"].resize((new_w, new_h), Image.Resampling.LANCZOS)
    out["image"] = image
    out["x"], out["y"] = scaled_top_left(
        state["x"], state["y"], state["image"].width, state["image"].height, new_w, new_h
    )
    out["kind"] = "raster"
    out["text"] = None
    return out


def rotate_visual(state: dict[str, Any], degrees_clockwise: float) -> dict[str, Any]:
    """Rotate a visual snapshot around its center. Positive degrees are clockwise.

    Text stores rotation in its metadata and is rendered again, so it stays
    editable. Pillow's own rotate is counter-clockwise, hence the negated angle
    on raster bitmaps and masks.
    """
    out = visual_state_like(state)
    if state["kind"] == "text" and state["text"] is not None:
        from pixedit.textutil import render_text

        info = out["text"]
        info.rotation = (float(info.rotation) + float(degrees_clockwise)) % 360.0
        image = render_text(info)
        out["image"] = image
        out["kind"] = "text"
        out["x"], out["y"] = scaled_top_left(
            state["x"], state["y"], state["image"].width, state["image"].height, image.width, image.height
        )
        return out
    angle = -float(degrees_clockwise)
    if state["kind"] == "redaction" and state["mask"] is not None:
        mask = state["mask"]
        rotated = mask.rotate(angle, resample=Image.Resampling.NEAREST, expand=True)
        out["mask"] = rotated
        out["x"], out["y"] = scaled_top_left(
            state["x"], state["y"], mask.width, mask.height, rotated.width, rotated.height
        )
        return out
    image = state["image"]
    rotated = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    out["image"] = rotated
    out["x"], out["y"] = scaled_top_left(
        state["x"], state["y"], image.width, image.height, rotated.width, rotated.height
    )
    out["kind"] = "raster"
    out["text"] = None
    return out


def flip_visual(state: dict[str, Any], horizontal: bool) -> dict[str, Any]:
    out = visual_state_like(state)
    method = Image.Transpose.FLIP_LEFT_RIGHT if horizontal else Image.Transpose.FLIP_TOP_BOTTOM
    if state["kind"] == "redaction" and state["mask"] is not None:
        out["mask"] = state["mask"].transpose(method)
        return out
    out["image"] = state["image"].transpose(method)
    out["kind"] = "raster"
    out["text"] = None
    return out


def visual_state_like(state: dict[str, Any]) -> dict[str, Any]:
    """Copy a visual snapshot so transforms never alias the undo buffer."""
    import copy

    return {
        "image": state["image"].copy(),
        "x": state["x"],
        "y": state["y"],
        "kind": state["kind"],
        "text": copy.deepcopy(state["text"]),
        "mask": None if state["mask"] is None else state["mask"].copy(),
        "redaction": copy.deepcopy(state["redaction"]),
    }


def expand_to_canvas(layer: Layer, width: int, height: int) -> None:
    """Make a raster/text layer cover the whole document so tools can paint anywhere.

    Text metadata is dropped because the bitmap is no longer a pure text render.
    The caller snapshots ``visual_state`` first so undo can bring the text back.
    """
    if (
        layer.kind == "raster"
        and layer.x == 0
        and layer.y == 0
        and layer.image.size == (width, height)
    ):
        return
    layer.image = place_image(layer.image.convert("RGBA"), layer.x, layer.y, width, height)
    layer.x = 0
    layer.y = 0
    layer.kind = "raster"
    layer.text = None


def hit_opaque(document: Document, x: float, y: float) -> int | None:
    """Topmost visible layer whose pixel at (x, y) is not empty."""
    px, py = int(math.floor(x)), int(math.floor(y))
    for index in range(len(document.layers) - 1, -1, -1):
        layer = document.layers[index]
        if not layer.visible:
            continue
        local_x = px - layer.x
        local_y = py - layer.y
        image = layer.bitmap()
        if not (0 <= local_x < image.width and 0 <= local_y < image.height):
            continue
        if image.mode == "L":
            if image.getpixel((local_x, local_y)) > 8:
                return index
        else:
            pixel = image.convert("RGBA").getpixel((local_x, local_y))
            if isinstance(pixel, tuple) and pixel[3] > 8:
                return index
    return None


def selection_for_layer(document: Document, layer: Layer) -> Image.Image | None:
    """Crop the canvas selection down to a layer-local L mask, or None."""
    if document.selection is None:
        return None
    local = Image.new("L", layer.image.size, 0)
    sx0 = max(0, layer.x)
    sy0 = max(0, layer.y)
    sx1 = min(document.width, layer.x + layer.image.width)
    sy1 = min(document.height, layer.y + layer.image.height)
    if sx1 <= sx0 or sy1 <= sy0:
        return local
    full = document.selection
    if full.size != (document.width, document.height):
        full = place_image(full, 0, 0, document.width, document.height, 0)
    crop = full.crop((sx0, sy0, sx1, sy1))
    local.paste(crop, (sx0 - layer.x, sy0 - layer.y))
    return local


def combine_selection(
    existing: Image.Image | None,
    new_mask: Image.Image,
    mode: str,
) -> Image.Image:
    """Combine a new mask with the current selection.

    ``replace`` (or no existing selection) keeps ``new_mask``. ``add`` is a
    union, ``subtract`` punches ``new_mask`` out of the existing selection.
    """
    if mode == "replace" or existing is None:
        return new_mask.copy()
    current = np.asarray(existing).astype(np.uint8)
    incoming = np.asarray(new_mask).astype(np.uint8)
    if incoming.shape != current.shape:
        fitted = Image.new("L", (current.shape[1], current.shape[0]), 0)
        fitted.paste(new_mask, (0, 0))
        incoming = np.asarray(fitted).astype(np.uint8)
    if mode == "add":
        out = np.maximum(current, incoming)
    elif mode == "subtract":
        out = current.copy()
        out[incoming > 0] = 0
    else:
        out = incoming
    return Image.fromarray(out, "L")


def _ordered_box(x0: float, y0: float, x1: float, y1: float) -> list[float]:
    """Corners in Pillow's required order.

    ``ImageDraw`` rejects a box when the second corner is above or left of
    the first. A drag that goes up or left arrives that way.
    """
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def mask_rect(width: int, height: int, x0: float, y0: float, x1: float, y1: float) -> Image.Image:
    from PIL import ImageDraw

    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    draw.rectangle(_ordered_box(x0, y0, x1, y1), fill=255)
    return image


def mask_ellipse(width: int, height: int, x0: float, y0: float, x1: float, y1: float) -> Image.Image:
    from PIL import ImageDraw

    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    draw.ellipse(_ordered_box(x0, y0, x1, y1), fill=255)
    return image


def mask_freehand(
    width: int,
    height: int,
    points: list[tuple[float, float]],
    radius: float = 1.0,
) -> Image.Image:
    """Filled lasso plus a stroke so a short drag still leaves a mark."""
    from PIL import ImageDraw

    image = Image.new("L", (width, height), 0)
    if not points:
        return image
    draw = ImageDraw.Draw(image)
    radius_px = max(1, int(round(radius)))
    if len(points) == 1:
        x, y = points[0]
        draw.ellipse([x - radius_px, y - radius_px, x + radius_px, y + radius_px], fill=255)
        return image
    if len(points) >= 3:
        draw.polygon(points, fill=255)
    draw.line(points + [points[0]], fill=255, width=radius_px * 2)
    return image


def normalized_rect(x0: float, y0: float, x1: float, y1: float) -> tuple[int, int, int, int]:
    left = int(math.floor(min(x0, x1)))
    top = int(math.floor(min(y0, y1)))
    right = int(math.ceil(max(x0, x1)))
    bottom = int(math.ceil(max(y0, y1)))
    return left, top, max(left + 1, right), max(top + 1, bottom)


def clamp_crop(document: Document, x0: float, y0: float, x1: float, y1: float) -> tuple[int, int, int, int]:
    left, top, right, bottom = normalized_rect(x0, y0, x1, y1)
    left = min(max(0, left), document.width - 1)
    top = min(max(0, top), document.height - 1)
    right = min(max(left + 1, right), document.width)
    bottom = min(max(top + 1, bottom), document.height)
    return left, top, right - left, bottom - top

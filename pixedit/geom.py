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


def covers_document(x: int, y: int, width: int, height: int, doc_width: int, doc_height: int) -> bool:
    """True when a placed bitmap already hides every document pixel.

    Backgrounds, imports, and flattened plates use this so a later paint stroke
    does not crop them down to the new mark.
    """
    return int(x) <= 0 and int(y) <= 0 and int(x) + int(width) >= int(doc_width) and int(y) + int(height) >= int(doc_height)


def image_has_opaque(image: Image.Image) -> bool:
    if image.width < 1 or image.height < 1:
        return False
    if image.mode == "L":
        return bool(int(np.asarray(image).max()) > 0)
    return bool(int(np.asarray(image.convert("RGBA"))[..., 3].max()) > 0)


def clip_box(
    box: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int] | None:
    """Clip a half-open box to the document. ``None`` when nothing remains."""
    x0 = max(0, int(box[0]))
    y0 = max(0, int(box[1]))
    x1 = min(int(width), int(box[2]))
    y1 = min(int(height), int(box[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def paint_bounds(
    origin: tuple[int, int],
    size: tuple[int, int],
    dab: tuple[int, int, int, int],
    doc_width: int,
    doc_height: int,
    *,
    slack: int,
    empty: bool,
) -> tuple[int, int, int, int] | None:
    """Canvas box a brush buffer needs in order to show ``dab``.

    ``dab`` and the returned box are half-open. The dab is padded by ``slack``
    so a stroke does not reallocate on every move, then clipped to the
    document. ``None`` means the dab misses the document entirely.

    An empty bitmap adopts that clipped box instead of staying anchored on a
    1×1 placeholder (which would union all the way back to the origin). A
    bitmap that already has pixels keeps those pixels, including any that sit
    outside the document because the layer was moved.
    """
    dab_hit = clip_box(dab, doc_width, doc_height)
    if dab_hit is None:
        return None
    padded = clip_box(
        (dab[0] - slack, dab[1] - slack, dab[2] + slack, dab[3] + slack),
        doc_width,
        doc_height,
    )
    target = padded if padded is not None else dab_hit
    if empty:
        return target
    ox, oy = int(origin[0]), int(origin[1])
    width, height = int(size[0]), int(size[1])
    return (
        min(ox, target[0]),
        min(oy, target[1]),
        max(ox + width, target[2]),
        max(oy + height, target[3]),
    )


def grow_layer_to_cover(
    layer: Layer, box: tuple[int, int, int, int], *, replace_if_empty: bool = True
) -> None:
    """Expand a raster bitmap so its placement includes ``box``.

    ``box`` is a half-open canvas rectangle, already clipped to the document.
    New pixels are transparent. An empty layer is moved onto ``box`` rather
    than unioned with its old origin.
    """
    x0, y0, x1, y1 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    if x1 <= x0 or y1 <= y0:
        return
    if replace_if_empty and not image_has_opaque(layer.image):
        layer.image = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
        layer.x = x0
        layer.y = y0
        return
    lx0, ly0 = int(layer.x), int(layer.y)
    lx1, ly1 = lx0 + layer.image.width, ly0 + layer.image.height
    nx0, ny0 = min(lx0, x0), min(ly0, y0)
    nx1, ny1 = max(lx1, x1), max(ly1, y1)
    if (nx0, ny0, nx1, ny1) == (lx0, ly0, lx1, ly1):
        return
    canvas = Image.new("RGBA", (nx1 - nx0, ny1 - ny0), (0, 0, 0, 0))
    canvas.paste(layer.image.convert("RGBA"), (lx0 - nx0, ly0 - ny0))
    layer.image = canvas
    layer.x = nx0
    layer.y = ny0


def tighten_raster_layer(layer: Layer) -> None:
    """Crop transparent margins and shift ``(x, y)`` onto the opaque pixels.

    A fully clear layer becomes a 1×1 transparent image and keeps its origin.
    Redaction masks are not cropped; they stay in document coordinates.
    """
    if layer.kind == "redaction":
        return
    box = alpha_bbox(layer.image)
    if box is None:
        layer.image = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        layer.kind = "raster"
        layer.text = None
        return
    x0, y0, x1, y1 = box
    if (x0, y0, x1, y1) == (0, 0, layer.image.width, layer.image.height):
        return
    layer.image = layer.image.crop((x0, y0, x1, y1))
    layer.x += x0
    layer.y += y0
    layer.kind = "raster"
    layer.text = None


def layer_hit_box(layer: Layer) -> tuple[int, int, int, int] | None:
    """Canvas rectangle a click uses to select ``layer``.

    Half-open ``(left, top, right, bottom)``. ``None`` when there is nothing
    to select.

    Text uses the rendered bitmap. That rectangle is the text box: the padding
    and the gaps between glyphs are inside it, and rotation is already baked
    into the axis-aligned image. Raster and redaction use the tight rectangle
    around non-empty pixels so a full-canvas plate or a document-sized mask
    does not take clicks that miss its content. A hole inside that tight
    rectangle still counts.
    """
    if layer.kind == "text":
        x, y, width, height = layer.bounds()
        if width < 1 or height < 1:
            return None
        return x, y, x + width, y + height
    box = alpha_bbox(layer.bitmap())
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return layer.x + x0, layer.y + y0, layer.x + x1, layer.y + y1


def hit_layer(document: Document, x: float, y: float) -> int | None:
    """Index of the topmost visible layer whose selection box contains ``(x, y)``.

    The pointer is floored to a pixel. A higher layer wins even when that
    pixel is empty, as long as the point sits inside its box. Hidden layers
    are skipped.
    """
    px, py = int(math.floor(x)), int(math.floor(y))
    for index in range(len(document.layers) - 1, -1, -1):
        layer = document.layers[index]
        if not layer.visible:
            continue
        box = layer_hit_box(layer)
        if box is None:
            continue
        left, top, right, bottom = box
        if left <= px < right and top <= py < bottom:
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

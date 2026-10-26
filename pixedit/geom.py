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

from pixedit.model import Document, Layer, new_id
from pixedit.proxy import resize_for_preview, rotate_for_preview


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


def crop_or_pad(
    image: Image.Image,
    left: int,
    top: int,
    width: int,
    height: int,
    fill: int | tuple[int, ...] = 0,
) -> Image.Image:
    """Copy a window that may extend outside ``image``. Outside pixels are ``fill``."""
    width = max(1, int(width))
    height = max(1, int(height))
    left = int(left)
    top = int(top)
    if image.mode == "L":
        canvas = Image.new("L", (width, height), int(fill) if not isinstance(fill, tuple) else 0)
    else:
        color = fill if isinstance(fill, tuple) else (0, 0, 0, 0)
        canvas = Image.new("RGBA", (width, height), color)
        image = image.convert("RGBA")
    src_left = max(0, left)
    src_top = max(0, top)
    src_right = min(image.width, left + width)
    src_bottom = min(image.height, top + height)
    if src_right <= src_left or src_bottom <= src_top:
        return canvas
    piece = image.crop((src_left, src_top, src_right, src_bottom))
    canvas.paste(piece, (src_left - left, src_top - top))
    return canvas


def crop_document(
    document: Document, left: int, top: int, width: int, height: int
) -> tuple[list[Layer], Image.Image | None]:
    """Return new layers and selection for a crop window.

    The window may start outside the canvas. A negative origin grows that
    edge, and a smaller window trims. A layer that misses the window becomes
    a 1×1 empty image. Redaction masks and the selection stay canvas-sized,
    with empty pixels where the window grew.
    """
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
            cloned.mask = crop_or_pad(placed, left, top, width, height, 0)
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
        selection = crop_or_pad(full, left, top, width, height, 0)
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


# Pixel edges the size dialogs accept. Percent and inches clamp to this too.
MAX_EDGE = 16000

# Anchor fractions: 0 keeps the left or top edge, 1 keeps the right or bottom.
CANVAS_ANCHORS: dict[str, tuple[float, float]] = {
    "nw": (0.0, 0.0),
    "n": (0.5, 0.0),
    "ne": (1.0, 0.0),
    "w": (0.0, 0.5),
    "c": (0.5, 0.5),
    "e": (1.0, 0.5),
    "sw": (0.0, 1.0),
    "s": (0.5, 1.0),
    "se": (1.0, 1.0),
}


def _clamp_edge(value: float) -> int:
    return max(1, min(MAX_EDGE, int(round(value))))


def canvas_window(
    old_w: int, old_h: int, new_w: int, new_h: int, anchor: str = "c"
) -> tuple[int, int, int, int]:
    """Crop window in old-canvas coordinates for a new canvas size.

    The anchor is the point that stays fixed. Its shift onto the new canvas
    is ``(new - old) * anchor``, with 0 at the left or top and 1 at the right
    or bottom. The window origin is the negation of that shift, so a larger
    canvas starts at a negative coordinate and a smaller one starts inset.
    """
    ax, ay = CANVAS_ANCHORS.get(anchor, CANVAS_ANCHORS["c"])
    new_w = _clamp_edge(new_w)
    new_h = _clamp_edge(new_h)
    shift_x = int(round((new_w - int(old_w)) * ax))
    shift_y = int(round((new_h - int(old_h)) * ay))
    return -shift_x, -shift_y, new_w, new_h


def _bottom_is_canvas_plate(document: Document) -> bool:
    """True when the back layer is exactly the current canvas and has no text."""
    if not document.layers:
        return False
    layer = document.layers[0]
    return (
        layer.kind == "raster"
        and layer.text is None
        and layer.x == 0
        and layer.y == 0
        and layer.image.width == document.width
        and layer.image.height == document.height
    )


def refit_canvas(
    document: Document,
    new_w: int,
    new_h: int,
    anchor: str = "c",
    fill: tuple[int, ...] | None = None,
) -> tuple[list[Layer], Image.Image | None]:
    """Layers and selection for a new canvas size. Pixels are not resampled.

    ``fill`` colors the new margin. A back layer that already fills the old
    canvas grows in place. Any other stack gets a new bottom plate, so the
    pictures already there are not painted over. ``None`` leaves the margin empty.
    """
    left, top, width, height = canvas_window(document.width, document.height, new_w, new_h, anchor)
    plate = document.layers[0].image if fill is not None and _bottom_is_canvas_plate(document) else None
    layers, selection = crop_document(document, left, top, width, height)
    if fill is None:
        return layers, selection
    if plate is not None and layers:
        layers[0].image = crop_or_pad(plate, left, top, width, height, fill)
        layers[0].x = 0
        layers[0].y = 0
        layers[0].kind = "raster"
        layers[0].text = None
        return layers, selection
    color = fill if len(fill) == 4 else (*tuple(fill)[:3], 255)
    layers.insert(
        0,
        Layer(
            id=new_id(),
            name="Canvas",
            kind="raster",
            image=Image.new("RGBA", (width, height), color),
            x=0,
            y=0,
        ),
    )
    return layers, selection


def scales_match(old_w: int, old_h: int, new_w: int, new_h: int) -> bool:
    """True when both axes are one scale, aside from rounding to a pixel.

    A font size can follow only one scale. An uneven resize rasterizes text
    so the bitmap matches the requested width and height.
    """
    old_w = max(1, int(old_w))
    old_h = max(1, int(old_h))
    return abs(int(new_w) * old_h - int(new_h) * old_w) <= max(old_w, old_h) / 2


def resample_filter(name: str) -> Image.Resampling:
    filters = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    return filters.get(str(name), Image.Resampling.LANCZOS)


def _scaled_box(layer: Layer, document: Document, new_w: int, new_h: int, sx: float, sy: float) -> tuple[int, int, int, int]:
    """New bitmap size and origin. A full-canvas plate maps onto the new canvas exactly."""
    if layer.x == 0 and layer.y == 0 and layer.image.size == (document.width, document.height):
        return new_w, new_h, 0, 0
    return (
        max(1, int(round(layer.image.width * sx))),
        max(1, int(round(layer.image.height * sy))),
        int(round(layer.x * sx)),
        int(round(layer.y * sy)),
    )


def resample_document(
    document: Document, new_w: int, new_h: int, method: str = "lanczos"
) -> tuple[list[Layer], Image.Image | None]:
    """Scale every layer and the selection onto a new pixel size.

    Positions scale with the document, so the stack still lines up. Masks and
    the selection stay hard-edged. A matching scale re-renders text. An uneven
    scale rasterizes it, because one font size cannot encode two scales.
    """
    new_w = _clamp_edge(new_w)
    new_h = _clamp_edge(new_h)
    old_w = max(1, int(document.width))
    old_h = max(1, int(document.height))
    sx = new_w / old_w
    sy = new_h / old_h
    uniform = scales_match(old_w, old_h, new_w, new_h)
    sample = resample_filter(method)
    layers: list[Layer] = []
    for layer in document.layers:
        cloned = layer.clone()
        if layer.kind == "redaction":
            # Masks are document-sized. Materialize, then scale, so an offset
            # mask still covers the same part of the picture.
            mask = layer.mask or Image.new("L", (old_w, old_h), 0)
            full = place_image(mask, layer.x, layer.y, old_w, old_h, 0)
            cloned.mask = full.resize((new_w, new_h), Image.Resampling.NEAREST)
            cloned.image = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
            cloned.x = 0
            cloned.y = 0
            layers.append(cloned)
            continue
        if layer.kind == "text" and layer.text is not None and uniform:
            from pixedit.textutil import render_text

            info = cloned.text
            assert info is not None
            info.size = max(1, int(round(info.size * sx)))
            cloned.image = render_text(info)
            cloned.kind = "text"
            cloned.x = int(round(layer.x * sx))
            cloned.y = int(round(layer.y * sy))
            layers.append(cloned)
            continue
        width, height, x, y = _scaled_box(layer, document, new_w, new_h, sx, sy)
        cloned.image = layer.image.resize((width, height), sample)
        cloned.x = x
        cloned.y = y
        cloned.kind = "raster"
        cloned.text = None
        if layer.mask is not None:
            cloned.mask = layer.mask.resize(
                (max(1, int(round(layer.mask.width * sx))), max(1, int(round(layer.mask.height * sy)))),
                Image.Resampling.NEAREST,
            )
        layers.append(cloned)
    selection = None
    if document.selection is not None:
        full = place_image(document.selection, 0, 0, old_w, old_h, 0)
        selection = full.resize((new_w, new_h), Image.Resampling.NEAREST)
    return layers, selection


def pixels_from_percent(original: int, percent: float) -> int:
    return _clamp_edge(max(1, int(original)) * float(percent) / 100.0)


def percent_from_pixels(original: int, pixels: int) -> float:
    return 100.0 * int(pixels) / max(1, int(original))


def inches_from_pixels(pixels: int, dpi: float) -> float:
    return int(pixels) / max(1.0, float(dpi))


def pixels_from_inches(inches: float, dpi: float) -> int:
    return _clamp_edge(float(inches) * max(1.0, float(dpi)))


def pixels_keeping_inches(pixels: int, old_dpi: float, new_dpi: float) -> int:
    """Pixel count that keeps the printed size when the resolution changes."""
    return pixels_from_inches(inches_from_pixels(pixels, old_dpi), new_dpi)


def linked_percent(
    original_w: int,
    original_h: int,
    width_percent: float,
    height_percent: float,
    *,
    axis: str,
    keep: bool,
) -> tuple[int, int, float, float]:
    """Pixels and percents after a percent edit.

    With ``keep``, both axes take the edited percent. The percent stays at
    the typed value. Pixels are that percent of the original size, rounded.
    """
    if keep:
        chosen = width_percent if axis == "width" else height_percent
        width_percent = height_percent = float(chosen)
    return (
        pixels_from_percent(original_w, width_percent),
        pixels_from_percent(original_h, height_percent),
        float(width_percent),
        float(height_percent),
    )


def linked_pixels(
    original_w: int,
    original_h: int,
    width_px: int,
    height_px: int,
    *,
    axis: str,
    keep: bool,
) -> tuple[int, int, float, float]:
    """Pixels and percents after a pixel edit. ``keep`` uses the original aspect."""
    width_px = _clamp_edge(width_px)
    height_px = _clamp_edge(height_px)
    if keep:
        aspect = max(1, int(original_w)) / max(1, int(original_h))
        if axis == "width":
            height_px = _clamp_edge(width_px / aspect)
        else:
            width_px = _clamp_edge(height_px * aspect)
    return (
        width_px,
        height_px,
        percent_from_pixels(original_w, width_px),
        percent_from_pixels(original_h, height_px),
    )


def linked_inches(
    original_w: int,
    original_h: int,
    width_px: int,
    height_px: int,
    width_in: float,
    height_in: float,
    dpi: float,
    *,
    axis: str,
    keep: bool,
    resample: bool,
) -> tuple[int, int, float, float, float]:
    """Pixels, inches, and dpi after an inch edit.

    One document has one resolution. With resample off, the pixels stay and
    the edited side sets the dpi; the other side's inches follow that dpi.
    With resample on, inches become pixels at the current dpi. Keep
    proportions then sets the other side from the original aspect.
    """
    dpi = max(1.0, float(dpi))
    if not resample:
        edited = width_in if axis == "width" else height_in
        pixels = width_px if axis == "width" else height_px
        dpi = max(1.0, int(pixels) / max(0.001, float(edited)))
        return (
            int(width_px),
            int(height_px),
            inches_from_pixels(width_px, dpi),
            inches_from_pixels(height_px, dpi),
            dpi,
        )
    if axis == "width":
        width_px = pixels_from_inches(width_in, dpi)
        if keep:
            height_px = _clamp_edge(width_px * max(1, int(original_h)) / max(1, int(original_w)))
    else:
        height_px = pixels_from_inches(height_in, dpi)
        if keep:
            width_px = _clamp_edge(height_px * max(1, int(original_w)) / max(1, int(original_h)))
    return (
        int(width_px),
        int(height_px),
        inches_from_pixels(width_px, dpi),
        inches_from_pixels(height_px, dpi),
        dpi,
    )


def linked_dpi(
    width_px: int,
    height_px: int,
    old_dpi: float,
    new_dpi: float,
    *,
    resample: bool,
) -> tuple[int, int, float, float, float]:
    """Pixels and inches after a resolution edit.

    Resample on keeps the printed size, so the pixel count follows the dpi.
    Resample off keeps the pixels, so the printed size follows the dpi.
    """
    new_dpi = max(1.0, float(new_dpi))
    if resample:
        width_px = pixels_keeping_inches(width_px, old_dpi, new_dpi)
        height_px = pixels_keeping_inches(height_px, old_dpi, new_dpi)
    return (
        int(width_px),
        int(height_px),
        inches_from_pixels(width_px, new_dpi),
        inches_from_pixels(height_px, new_dpi),
        new_dpi,
    )


def scale_visual(
    state: dict[str, Any],
    new_w: int,
    new_h: int,
    *,
    zoom: float = 1.0,
    final: bool = True,
) -> dict[str, Any]:
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
    image = resize_for_preview(state["image"], (new_w, new_h), zoom, final=final)
    out["image"] = image
    out["x"], out["y"] = scaled_top_left(
        state["x"], state["y"], state["image"].width, state["image"].height, new_w, new_h
    )
    out["kind"] = "raster"
    out["text"] = None
    return out


def rotate_visual(
    state: dict[str, Any],
    degrees_clockwise: float,
    *,
    zoom: float = 1.0,
    final: bool = True,
) -> dict[str, Any]:
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
    rotated = rotate_for_preview(image, angle, zoom, final=final)
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


def _visual_size(state: dict[str, Any]) -> tuple[int, int]:
    """Pixel size of the bitmap a snapshot occupies. Masks use the mask."""
    if state["kind"] == "redaction" and state["mask"] is not None:
        return state["mask"].width, state["mask"].height
    return state["image"].width, state["image"].height


def union_bounds(states: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    """``(x, y, width, height)`` around every snapshot's bitmap."""
    if not states:
        return 0.0, 0.0, 1.0, 1.0
    left = min(float(state["x"]) for state in states)
    top = min(float(state["y"]) for state in states)
    right = max(float(state["x"]) + _visual_size(state)[0] for state in states)
    bottom = max(float(state["y"]) + _visual_size(state)[1] for state in states)
    return left, top, max(1.0, right - left), max(1.0, bottom - top)


def layer_union_bounds(layers: list[Layer]) -> tuple[float, float, float, float]:
    """``(x, y, width, height)`` around every layer's bitmap. No pixel copy."""
    if not layers:
        return 0.0, 0.0, 1.0, 1.0
    boxes = [layer.bounds() for layer in layers]
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    bottom = max(box[1] + box[3] for box in boxes)
    return float(left), float(top), max(1.0, float(right - left)), max(1.0, float(bottom - top))


def _corner_of(x: float, y: float, width: float, height: float, name: str) -> tuple[float, float]:
    if name == "ne":
        return x + width, y
    if name == "sw":
        return x, y + height
    if name == "se":
        return x + width, y + height
    return x, y


def _place_from_corner(name: str, corner: tuple[float, float], width: float, height: float) -> tuple[int, int]:
    """Top-left that puts ``name`` of a ``width``×``height`` bitmap on ``corner``."""
    cx, cy = corner
    if name == "ne":
        return int(round(cx - width)), int(round(cy))
    if name == "sw":
        return int(round(cx)), int(round(cy - height))
    if name == "se":
        return int(round(cx - width)), int(round(cy - height))
    return int(round(cx)), int(round(cy))


def scale_visual_in_box(
    state: dict[str, Any],
    anchor: tuple[float, float],
    anchor_name: str,
    scale_x: float,
    scale_y: float,
    *,
    zoom: float = 1.0,
    final: bool = True,
) -> dict[str, Any]:
    """Scale one snapshot as part of a shared box.

    ``anchor`` is the corner of that box that stays put (``nw``, ``ne``,
    ``sw``, or ``se``). Each layer's matching corner is scaled away from it
    by ``scale_x`` and ``scale_y``, then that layer's own bitmap is resampled.
    Layers stay separate, so blend modes and masks survive. A flattened copy
    could not be split back.

    Text stays editable: the font size follows the width factor and the new
    glyphs hang from the scaled corner, because a re-rendered line is not the
    exact pixel size. Masks stay nearest-neighbor. One layer whose own corner
    is ``anchor`` matches a corner drag of that layer alone.
    """
    scale_x = float(scale_x)
    scale_y = float(scale_y)
    old_w, old_h = _visual_size(state)
    corner = _corner_of(float(state["x"]), float(state["y"]), old_w, old_h, anchor_name)
    ax, ay = anchor
    mapped = (ax + (corner[0] - ax) * scale_x, ay + (corner[1] - ay) * scale_y)
    target_w = max(1, int(round(old_w * scale_x)))
    target_h = max(1, int(round(old_h * scale_y)))
    out = visual_state_like(state)
    if state["kind"] == "redaction" and state["mask"] is not None:
        out["mask"] = state["mask"].resize((target_w, target_h), Image.Resampling.NEAREST)
        out["x"], out["y"] = _place_from_corner(anchor_name, mapped, target_w, target_h)
        return out
    if state["kind"] == "text" and state["text"] is not None:
        from pixedit.textutil import render_text

        info = out["text"]
        info.size = max(1, int(round(info.size * scale_x)))
        image = render_text(info)
        out["image"] = image
        out["kind"] = "text"
        out["x"], out["y"] = _place_from_corner(anchor_name, mapped, image.width, image.height)
        return out
    image = resize_for_preview(state["image"], (target_w, target_h), zoom, final=final)
    out["image"] = image
    out["kind"] = "raster"
    out["text"] = None
    out["x"], out["y"] = _place_from_corner(anchor_name, mapped, target_w, target_h)
    return out


def _orbit_point(px: float, py: float, cx: float, cy: float, degrees_clockwise: float) -> tuple[float, float]:
    """Rotate a canvas point clockwise around ``(cx, cy)``.

    Screen Y grows downward. A positive angle sends a point on the right
    to below the center, which is the same direction as ``rotate_visual``.
    """
    theta = math.radians(degrees_clockwise)
    cosine = math.cos(theta)
    sine = math.sin(theta)
    dx = px - cx
    dy = py - cy
    return cx + dx * cosine - dy * sine, cy + dx * sine + dy * cosine


def rotate_visual_around(
    state: dict[str, Any],
    degrees_clockwise: float,
    center: tuple[float, float],
    *,
    zoom: float = 1.0,
    final: bool = True,
) -> dict[str, Any]:
    """Turn a snapshot around ``center`` and turn its pixels by the same angle.

    ``rotate_visual`` spins the bitmap about its own center. The center is
    then orbited around ``center`` so several layers move as one box. A
    single layer whose center is ``center`` stays on ``rotate_visual``.
    """
    width, height = _visual_size(state)
    old_cx = float(state["x"]) + width / 2.0
    old_cy = float(state["y"]) + height / 2.0
    turned = rotate_visual(state, degrees_clockwise, zoom=zoom, final=final)
    new_w, new_h = _visual_size(turned)
    nx, ny = _orbit_point(old_cx, old_cy, center[0], center[1], degrees_clockwise)
    turned["x"] = int(round(nx - new_w / 2.0))
    turned["y"] = int(round(ny - new_h / 2.0))
    return turned


def flip_visual_across(
    state: dict[str, Any],
    horizontal: bool,
    center: tuple[float, float],
) -> dict[str, Any]:
    """Mirror a snapshot's pixels and its position across ``center``.

    ``flip_visual`` mirrors the bitmap and leaves ``x, y`` alone, which is a
    flip inside one layer. Mirroring the center across the shared box also
    swaps a row of layers from side to side. One layer flipped across its
    own center does not move.
    """
    width, height = _visual_size(state)
    old_cx = float(state["x"]) + width / 2.0
    old_cy = float(state["y"]) + height / 2.0
    flipped = flip_visual(state, horizontal)
    new_w, new_h = _visual_size(flipped)
    if horizontal:
        new_cx = 2.0 * center[0] - old_cx
        new_cy = old_cy
    else:
        new_cx = old_cx
        new_cy = 2.0 * center[1] - old_cy
    flipped["x"] = int(round(new_cx - new_w / 2.0))
    flipped["y"] = int(round(new_cy - new_h / 2.0))
    return flipped


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
        "target_id": state.get("target_id"),
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


def union_boxes(
    first: tuple[int, int, int, int] | None,
    second: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    """The half-open box covering both, or whichever one is present."""
    if first is None:
        return second
    if second is None:
        return first
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def placed_box(layer: Layer, width: int, height: int) -> tuple[int, int, int, int] | None:
    """The layer bitmap clipped to the document."""
    return clip_box(
        (layer.x, layer.y, layer.x + layer.image.width, layer.y + layer.image.height),
        width,
        height,
    )


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
    union. ``subtract`` fades the existing coverage by ``new_mask``: 255
    clears a pixel, and a value between leaves that fraction. A hard mask is
    only 0 or 255, so it still punches a hole.
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
        # Hard coverage is 0 or 255 and still clears the pixel. A feathered
        # edge is between those, so it thins the selection instead of cutting
        # every touched pixel down to nothing.
        keep = (255 - incoming).astype(np.float32) / 255.0
        out = np.clip(np.rint(current.astype(np.float32) * keep), 0, 255).astype(np.uint8)
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


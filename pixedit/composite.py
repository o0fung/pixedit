"""Flatten a document into one RGBA image.

Pipeline for each visible layer, bottom to top:

1. A mask layer does not draw on its own turn. It belongs to one picture or
   text layer. When that layer is drawn, the effect (pixelate, blur, solid,
   noise, and the rest) runs on a buffer that contains only that layer, then
   the buffer is blended with the layer's opacity and blend mode. Layers
   below are not sampled, and layers above the group paint on top.
2. Raster and text layers without a mask are blended with a CSS-style
   "simple alpha compositing" formula (source-over, with the blend mode
   applied only where both coverages overlap). That keeps Multiply, Screen,
   and Overlay from turning into black when the backdrop is transparent.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from pixedit.geom import alpha_bbox, covers_document
from pixedit.model import BLEND_MODES, Document, Layer, LayerGroup, masks_for
from pixedit.pixels import mask_effect

# geom.place_image is imported lazily inside mask placement to keep import
# order boring: composite -> model, and geom -> model, neither imports the other
# at module level for placement helpers defined here.


def _lum(rgb: np.ndarray) -> np.ndarray:
    """CSS relative luminance weights used by Color and Luminosity."""
    return rgb[..., 0] * 0.3 + rgb[..., 1] * 0.59 + rgb[..., 2] * 0.11


def _set_lum(color: np.ndarray, lum: np.ndarray) -> np.ndarray:
    """Give ``color`` the luminosity ``lum``, then clip it back into 0..1.

    This is the CSS compositing SetLum. The dark clip runs first, then the
    light clip, each from the luminosity of the shifted color.
    """
    shifted = color + (lum - _lum(color))[..., None]
    luma = _lum(shifted)
    low = np.min(shifted, axis=-1)
    dark = low < 0
    span = np.maximum(luma - low, 1e-6)
    scaled = luma[..., None] + (shifted - luma[..., None]) * (luma / span)[..., None]
    shifted = np.where(dark[..., None], scaled, shifted)
    luma = _lum(shifted)
    high = np.max(shifted, axis=-1)
    light = high > 1
    span = np.maximum(high - luma, 1e-6)
    scaled = luma[..., None] + (shifted - luma[..., None]) * ((1.0 - luma) / span)[..., None]
    shifted = np.where(light[..., None], scaled, shifted)
    return np.clip(shifted, 0.0, 1.0)


def _blend_color(mode: str, dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    if mode == "multiply":
        return src * dst
    if mode == "screen":
        return 1.0 - (1.0 - src) * (1.0 - dst)
    if mode == "overlay":
        # Overlay is multiply on the dark half of the backdrop and screen on the light half.
        return np.where(
            dst < 0.5,
            2.0 * src * dst,
            1.0 - 2.0 * (1.0 - src) * (1.0 - dst),
        )
    if mode == "darken":
        return np.minimum(src, dst)
    if mode == "lighten":
        return np.maximum(src, dst)
    if mode == "color":
        # Source hue and saturation, backdrop luminosity.
        return _set_lum(src, _lum(dst))
    if mode == "luminosity":
        # Backdrop hue and saturation, source luminosity.
        return _set_lum(dst, _lum(src))
    return src


def blend_arrays(
    dst: np.ndarray,
    src: np.ndarray,
    opacity: float = 1.0,
    mode: str = "normal",
) -> np.ndarray:
    """Composite ``src`` over ``dst``. Both are float HxWx4 straight alpha, 0..1."""
    if mode not in BLEND_MODES:
        mode = "normal"
    src_a = np.clip(src[..., 3:4] * float(opacity), 0.0, 1.0)
    dst_a = np.clip(dst[..., 3:4], 0.0, 1.0)
    src_rgb = src[..., :3]
    dst_rgb = dst[..., :3]
    blended = _blend_color(mode, dst_rgb, src_rgb)
    out_a = src_a + dst_a * (1.0 - src_a)
    # Where the backdrop is clear, fall back to the source color so blend
    # modes do not mix against an imaginary black canvas.
    numer = src_a * (src_rgb * (1.0 - dst_a) + blended * dst_a) + dst_rgb * dst_a * (1.0 - src_a)
    out_rgb = np.where(out_a > 1e-6, numer / np.maximum(out_a, 1e-6), 0.0)
    return np.concatenate(
        [np.clip(out_rgb, 0.0, 1.0), np.clip(out_a, 0.0, 1.0)],
        axis=-1,
    ).astype(np.float32)


def pil_to_float(image: Image.Image) -> np.ndarray:
    rgba = image.convert("RGBA")
    return np.asarray(rgba).astype(np.float32) / 255.0


def float_to_pil(array: np.ndarray) -> Image.Image:
    uint8 = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(uint8), "RGBA")


def _intersection(
    ax: int, ay: int, aw: int, ah: int, bw: int, bh: int
) -> tuple[int, int, int, int, int, int] | None:
    """Overlap of a placed image (ax, ay, aw, ah) with a canvas of size bw×bh.

    Returns canvas x, y, width, height and the source origin inside the image.
    """
    x0 = max(0, ax)
    y0 = max(0, ay)
    x1 = min(bw, ax + aw)
    y1 = min(bh, ay + ah)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0, x0 - ax, y0 - ay


def blit_layer(canvas: np.ndarray, layer: Layer, origin_x: int = 0, origin_y: int = 0) -> np.ndarray:
    image = layer.image
    overlap = _intersection(
        layer.x - origin_x,
        layer.y - origin_y,
        image.width,
        image.height,
        canvas.shape[1],
        canvas.shape[0],
    )
    if overlap is None or layer.opacity <= 0:
        return canvas
    x0, y0, width, height, sx, sy = overlap
    cropped = image.crop((sx, sy, sx + width, sy + height))
    src = pil_to_float(cropped)
    dst = canvas[y0 : y0 + height, x0 : x0 + width]
    canvas[y0 : y0 + height, x0 : x0 + width] = blend_arrays(dst, src, layer.opacity, layer.blend_mode)
    return canvas


def mask_to_canvas(layer: Layer, width: int, height: int) -> np.ndarray:
    """Place a redaction mask into a float HxW array covering the document."""
    full = np.zeros((height, width), dtype=np.float32)
    if layer.mask is None:
        return full
    overlap = _intersection(layer.x, layer.y, layer.mask.width, layer.mask.height, width, height)
    if overlap is None:
        return full
    x0, y0, mw, mh, sx, sy = overlap
    cropped = layer.mask.crop((sx, sy, sx + mw, sy + mh))
    full[y0 : y0 + mh, x0 : x0 + mw] = np.asarray(cropped).astype(np.float32) / 255.0
    return full


def apply_redaction(canvas: np.ndarray, layer: Layer) -> np.ndarray:
    """Mix the mask effect through ``layer``'s mask.

    ``canvas`` is the picture the mask belongs to, not the stack underneath.
    Pixels outside the mask stay as they were passed in. The mask's opacity
    fades the effect.
    """
    height, width = canvas.shape[:2]
    mask = mask_to_canvas(layer, width, height) * float(layer.opacity)
    if float(mask.max()) <= 0.0:
        return canvas
    info = layer.redaction
    kind = info.kind if info is not None else "pixelate"
    amount = info.amount if info is not None else 12.0
    color = info.color if info is not None else (0, 0, 0, 255)
    seed = info.seed if info is not None else 1
    effect = mask_effect(canvas, kind, amount, color, seed)
    mix = mask[..., None]
    return (effect * mix + canvas * (1.0 - mix)).astype(np.float32)


def shows_redaction(document: Document) -> bool:
    """A visible mask layer. Blur and noise read outside a partial repaint."""
    return any(layer.visible and layer.opacity > 0 and layer.kind == "redaction" for layer in document.layers)


def _plain_target(layer: Layer) -> Layer:
    """A stand-in that blits ``layer`` at full opacity and Normal blend.

    The real opacity and blend are applied after the mask, so Multiply is
    not mixed against an empty buffer and a later fade covers the effect
    and the untouched pixels together.
    """
    return Layer(
        id=layer.id,
        name=layer.name,
        kind="raster",
        visible=True,
        opacity=1.0,
        blend_mode="normal",
        image=layer.image,
        x=layer.x,
        y=layer.y,
    )


def target_with_masks(document: Document, target: Layer, masks: list[Layer]) -> np.ndarray:
    """Document-sized float RGBA of ``target`` with ``masks`` applied in order.

    Hidden masks and masks at zero opacity are skipped. The buffer starts
    empty, so a blur samples this layer's pixels and transparency, not the
    pictures underneath.
    """
    solo = np.zeros((document.height, document.width, 4), dtype=np.float32)
    blit_layer(solo, _plain_target(target))
    for mask in masks:
        if not mask.visible or mask.opacity <= 0:
            continue
        solo = apply_redaction(solo, mask)
    return solo


def baked_placement(
    document: Document, target: Layer, masks: list[Layer]
) -> tuple[Image.Image, int, int]:
    """Pixels, x, and y after baking ``masks`` into ``target``.

    A layer that already covers the document keeps that full plate, so a
    background does not shrink to the painted region. Any smaller layer is
    cropped to the opaque result, which includes blur or a solid fill that
    spreads past the old bitmap.
    """
    flat = float_to_pil(target_with_masks(document, target, masks))
    if covers_document(
        target.x,
        target.y,
        target.image.width,
        target.image.height,
        document.width,
        document.height,
    ):
        return flat, 0, 0
    box = alpha_bbox(flat)
    if box is None:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0)), int(target.x), int(target.y)
    x0, y0, x1, y1 = box
    return flat.crop((x0, y0, x1, y1)), x0, y0


def _view_size(span: float, scale: float) -> int:
    """Screen pixels covering ``span`` document pixels, never above one per source pixel."""
    if span <= 0 or scale <= 0:
        return 1
    return max(1, min(int(math.ceil(span)), int(round(span * scale))))


def _blit_scaled(
    canvas: np.ndarray,
    layer: Layer,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> None:
    """Blend ``layer`` into a view that maps ``(x0, y0, x1, y1)`` onto ``canvas``.

    The source is resampled with a box filter, so a zoomed-out view averages the
    document pixels that land in each screen pixel instead of flattening the
    whole canvas first.
    """
    image = layer.image
    out_h, out_w = canvas.shape[:2]
    span_x = x1 - x0
    span_y = y1 - y0
    if image.width < 1 or image.height < 1 or span_x <= 0 or span_y <= 0:
        return
    ox0 = max(x0, float(layer.x))
    oy0 = max(y0, float(layer.y))
    ox1 = min(x1, float(layer.x + image.width))
    oy1 = min(y1, float(layer.y + image.height))
    if ox1 <= ox0 or oy1 <= oy0:
        return

    def to_px_x(doc_x: float) -> float:
        return (doc_x - x0) / span_x * out_w

    def to_px_y(doc_y: float) -> float:
        return (doc_y - y0) / span_y * out_h

    ix0 = max(0, math.floor(to_px_x(ox0)))
    iy0 = max(0, math.floor(to_px_y(oy0)))
    ix1 = min(out_w, math.ceil(to_px_x(ox1)))
    iy1 = min(out_h, math.ceil(to_px_y(oy1)))
    if ix1 <= ix0 or iy1 <= iy0:
        return

    def to_src_x(px: float) -> float:
        return x0 + px / out_w * span_x - layer.x

    def to_src_y(py: float) -> float:
        return y0 + py / out_h * span_y - layer.y

    sx0 = min(max(0.0, to_src_x(ix0)), float(image.width))
    sy0 = min(max(0.0, to_src_y(iy0)), float(image.height))
    sx1 = min(max(0.0, to_src_x(ix1)), float(image.width))
    sy1 = min(max(0.0, to_src_y(iy1)), float(image.height))
    if sx1 - sx0 < 1e-4 or sy1 - sy0 < 1e-4:
        return
    resized = image.resize(
        (ix1 - ix0, iy1 - iy0),
        Image.Resampling.BOX,
        box=(sx0, sy0, sx1, sy1),
    )
    src = pil_to_float(resized)
    dst = canvas[iy0:iy1, ix0:ix1]
    canvas[iy0:iy1, ix0:ix1] = blend_arrays(dst, src, layer.opacity, layer.blend_mode)


def composite_view(
    document: Document,
    box: tuple[float, float, float, float],
    scale: float,
) -> Image.Image:
    """Flatten ``box`` at ``scale`` screen pixels per document pixel.

    ``box`` is ``(x0, y0, x1, y1)`` in document pixels. The result is only as
    large as that rectangle on screen, and never larger than the rectangle in
    document pixels. Redaction layers are skipped: blur and noise read outside
    a downscaled box, so the caller uses ``composite`` when a mask is visible.
    Export and the eyedropper stay on ``composite``.
    """
    x0, y0, x1, y1 = (float(value) for value in box)
    span_x = x1 - x0
    span_y = y1 - y0
    if span_x <= 0 or span_y <= 0:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    out_w = _view_size(span_x, scale)
    out_h = _view_size(span_y, scale)
    canvas = np.zeros((out_h, out_w, 4), dtype=np.float32)
    for layer in document.layers:
        if not layer.visible or layer.opacity <= 0 or layer.kind == "redaction":
            continue
        _blit_scaled(canvas, layer, x0, y0, x1, y1)
    return float_to_pil(canvas)


def composite_region(document: Document, box: tuple[int, int, int, int]) -> Image.Image:
    """Flatten ``box`` so it matches ``composite`` cropped to the same pixels.

    ``box`` is ``(x0, y0, x1, y1)`` already clipped to the document. Redaction
    layers are skipped: blur and noise read pixels outside the box, so the
    caller flattens the whole document when a mask is visible.
    """
    x0, y0, x1, y1 = box
    canvas = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.float32)
    for layer in document.layers:
        if not layer.visible or layer.opacity <= 0 or layer.kind == "redaction":
            continue
        blit_layer(canvas, layer, x0, y0)
    return float_to_pil(canvas)


def composite(document: Document) -> Image.Image:
    """Flatten ``document`` to one RGBA image the size of the canvas.

    Masks are skipped on their own turn and applied when their picture is
    drawn. A hidden picture hides its masks. A hidden mask is left out of
    that picture's buffer, and the other masks of the same picture still run.
    """
    canvas = np.zeros((document.height, document.width, 4), dtype=np.float32)
    layers = document.layers
    for layer in layers:
        if layer.kind == "redaction":
            continue
        if not layer.visible or layer.opacity <= 0:
            continue
        owned = [mask for mask in masks_for(layers, layer) if mask.visible and mask.opacity > 0]
        if not owned:
            canvas = blit_layer(canvas, layer)
            continue
        solo = target_with_masks(document, layer, owned)
        canvas = blend_arrays(canvas, solo, layer.opacity, layer.blend_mode)
    return float_to_pil(canvas)


def composite_over(document: Document, background: tuple[int, int, int, int]) -> Image.Image:
    """Flatten onto a solid background. Used by JPEG export."""
    flat = composite(document)
    base = Image.new("RGBA", flat.size, background)
    return Image.alpha_composite(base, flat)


def composite_group(group: LayerGroup) -> Image.Image:
    """Flatten one page without changing which page the document is showing."""
    document = Document(
        width=group.width,
        height=group.height,
        groups=[group],
        active_group_id=group.id,
    )
    return composite(document)


def composite_group_over(group: LayerGroup, background: tuple[int, int, int, int]) -> Image.Image:
    """Flatten one page onto a solid background."""
    flat = composite_group(group)
    base = Image.new("RGBA", flat.size, background)
    return Image.alpha_composite(base, flat)


def merge_pair(lower: Layer, upper: Layer) -> Layer:
    """Bake ``upper`` into ``lower`` and return a new Normal raster layer.

    The temporary canvas covers the union of both bitmaps, including pixels
    that sit outside the document, so a merge does not clip overflow.
    Redaction is baked by running the normal compositor on just these two
    layers: the mask still samples only ``lower``, which is the whole stack
    below the redaction *within this pair*.
    """
    from pixedit.model import Layer as LayerCls
    from pixedit.model import new_id

    def extent(layer: Layer) -> tuple[int, int, int, int]:
        image = layer.bitmap()
        return layer.x, layer.y, layer.x + image.width, layer.y + image.height

    boxes = [extent(lower), extent(upper)]
    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)
    width = max(1, max_x - min_x)
    height = max(1, max_y - min_y)
    a = lower.clone()
    b = upper.clone()
    a.x -= min_x
    a.y -= min_y
    b.x -= min_x
    b.y -= min_y
    temp = Document(width=width, height=height, layers=[a, b])
    flat = composite(temp)
    alpha = np.asarray(flat)[..., 3]
    ys, xs = np.nonzero(alpha > 0)
    if len(xs) == 0:
        image = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        x = y = 0
    else:
        left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        image = flat.crop((left, top, right, bottom))
        x = min_x + left
        y = min_y + top
    return LayerCls(
        id=new_id(),
        name=lower.name,
        kind="raster",
        visible=lower.visible,
        opacity=1.0,
        blend_mode="normal",
        image=image,
        x=x,
        y=y,
    )

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
from pixedit.proxy import layer_sample

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


def _read_sample(
    layer: Layer,
    sx0: float,
    sy0: float,
    sx1: float,
    sy1: float,
    dest_w: int,
    dest_h: int,
    resample: Image.Resampling,
) -> Image.Image:
    """Pixels of ``layer`` covering layer-space ``(sx0, sy0, sx1, sy1)``.

    A live stand-in is a reduced copy of the same rectangle. It is scaled up
    to the destination here, so the drag never has to store a full-size preview.
    """
    sample, scale_x, scale_y = layer_sample(layer)
    box = (sx0 * scale_x, sy0 * scale_y, sx1 * scale_x, sy1 * scale_y)
    if dest_w < 1 or dest_h < 1:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    exact = (
        scale_x == 1.0
        and scale_y == 1.0
        and dest_w == int(round(sx1 - sx0))
        and dest_h == int(round(sy1 - sy0))
        and float(sx0).is_integer()
        and float(sy0).is_integer()
    )
    if exact:
        return sample.crop((int(sx0), int(sy0), int(sx0) + dest_w, int(sy0) + dest_h))
    sw, sh = sample.size
    x0 = min(max(0.0, box[0]), float(sw))
    y0 = min(max(0.0, box[1]), float(sh))
    x1 = min(max(x0, box[2]), float(sw))
    y1 = min(max(y0, box[3]), float(sh))
    if x1 - x0 < 1e-4 or y1 - y0 < 1e-4:
        return Image.new("RGBA", (dest_w, dest_h), (0, 0, 0, 0))
    return sample.resize((dest_w, dest_h), resample, box=(x0, y0, x1, y1))


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
    cropped = _read_sample(layer, sx, sy, sx + width, sy + height, width, height, Image.Resampling.BOX)
    src = pil_to_float(cropped)
    dst = canvas[y0 : y0 + height, x0 : x0 + width]
    canvas[y0 : y0 + height, x0 : x0 + width] = blend_arrays(dst, src, layer.opacity, layer.blend_mode)
    return canvas


def mask_to_canvas(layer: Layer, width: int, height: int, origin_x: int = 0, origin_y: int = 0) -> np.ndarray:
    """Place a redaction mask into a float HxW array.

    ``(origin_x, origin_y)`` is the document pixel at the buffer's top-left,
    so a partial flatten can hold a crop instead of the whole page.
    """
    full = np.zeros((height, width), dtype=np.float32)
    if layer.mask is None:
        return full
    overlap = _intersection(
        layer.x - origin_x,
        layer.y - origin_y,
        layer.mask.width,
        layer.mask.height,
        width,
        height,
    )
    if overlap is None:
        return full
    x0, y0, mw, mh, sx, sy = overlap
    cropped = layer.mask.crop((sx, sy, sx + mw, sy + mh))
    full[y0 : y0 + mh, x0 : x0 + mw] = np.asarray(cropped).astype(np.float32) / 255.0
    return full


def apply_redaction(
    canvas: np.ndarray, layer: Layer, origin_x: int = 0, origin_y: int = 0
) -> np.ndarray:
    """Mix the mask effect through ``layer``'s mask.

    ``canvas`` is the picture the mask belongs to, not the stack underneath.
    Pixels outside the mask stay as they were passed in. The mask's opacity
    fades the effect.
    """
    height, width = canvas.shape[:2]
    mask = mask_to_canvas(layer, width, height, origin_x, origin_y) * float(layer.opacity)
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
    """A visible mask layer. The zoomed-out view cache does not downsample one."""
    return any(layer.visible and layer.opacity > 0 and layer.kind == "redaction" for layer in document.layers)


def redaction_padding(document: Document) -> int | None:
    """Extra pixels a partial flatten needs, or None when a crop cannot match.

    Noise is one random field the size of the plate, so a crop would draw
    different grain. Blur, pixelate, and sharpen read a neighborhood. Solid,
    dim, invert, and desaturate are per pixel and need no padding.
    """
    pad = 0
    for layer in document.layers:
        if layer.kind != "redaction" or not layer.visible or layer.opacity <= 0:
            continue
        info = layer.redaction
        kind = info.kind if info is not None else "pixelate"
        amount = float(info.amount if info is not None else 12.0)
        if kind == "noise":
            return None
        if kind == "blur":
            pad = max(pad, int(math.ceil(amount * 2.0)) + 2)
        elif kind == "pixelate":
            pad = max(pad, max(1, int(round(amount))))
        elif kind == "sharpen":
            pad = max(pad, 4)
    return pad


def _pixelate_block(document: Document) -> int:
    block = 1
    for layer in document.layers:
        if layer.kind != "redaction" or not layer.visible or layer.opacity <= 0:
            continue
        info = layer.redaction
        kind = info.kind if info is not None else "pixelate"
        if kind != "pixelate":
            continue
        amount = float(info.amount if info is not None else 12.0)
        block = max(block, max(1, int(round(amount))))
    return block


# Reused full-page float plates. A later flatten clears the same array instead
# of allocating another. Results are copied out before the next call.
_FLOATS: dict[str, np.ndarray | None] = {"canvas": None, "solo": None}


def _borrow_float(name: str, height: int, width: int) -> np.ndarray:
    buf = _FLOATS.get(name)
    if buf is None or buf.shape != (height, width, 4):
        buf = np.zeros((height, width, 4), dtype=np.float32)
        _FLOATS[name] = buf
    else:
        buf.fill(0)
    return buf


def _opaque_pixels(layer: Layer) -> np.ndarray | None:
    """RGBA of an opaque Normal layer, or None when it has to be blended."""
    if layer.opacity < 1.0 or layer.blend_mode != "normal":
        return None
    if getattr(layer, "preview_source", None) is not None:
        return None
    image = layer.image if layer.image.mode == "RGBA" else layer.image.convert("RGBA")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 4 or array.shape[0] < 1 or array.shape[1] < 1:
        return None
    if int(array[..., 3].min()) != 255:
        return None
    return array


def _composite_opaque(document: Document) -> Image.Image | None:
    """Copy opaque Normal layers straight into a byte plate.

    None when any visible pixel needs a blend, a mask, or a live stand-in.
    Those stay on the float path so partial opacity does not change.
    """
    prepared: list[tuple[Layer, np.ndarray]] = []
    for layer in document.layers:
        if layer.kind == "redaction":
            if layer.visible and layer.opacity > 0:
                return None
            continue
        if not layer.visible or layer.opacity <= 0:
            continue
        if any(mask.visible and mask.opacity > 0 for mask in masks_for(document.layers, layer)):
            return None
        pixels = _opaque_pixels(layer)
        if pixels is None:
            return None
        prepared.append((layer, pixels))
    canvas = np.zeros((document.height, document.width, 4), dtype=np.uint8)
    for layer, pixels in prepared:
        overlap = _intersection(
            layer.x,
            layer.y,
            int(pixels.shape[1]),
            int(pixels.shape[0]),
            document.width,
            document.height,
        )
        if overlap is None:
            continue
        x0, y0, width, height, sx, sy = overlap
        canvas[y0 : y0 + height, x0 : x0 + width] = pixels[sy : sy + height, sx : sx + width]
    return Image.fromarray(canvas, "RGBA")


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
    solo = _borrow_float("solo", document.height, document.width)
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
    resized = _read_sample(
        layer, sx0, sy0, sx1, sy1, ix1 - ix0, iy1 - iy0, Image.Resampling.BOX
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

    ``box`` is ``(x0, y0, x1, y1)`` already clipped to the document. A mask
    that reads a neighborhood is given that padding, then the padding is
    cropped off. Noise cannot be cropped, so that case flattens the page and
    returns the box.
    """
    if redaction_padding(document) is None:
        x0, y0, x1, y1 = box
        return composite(document).crop((x0, y0, x1, y1))
    x0, y0, x1, y1 = box
    pad = int(redaction_padding(document) or 0)
    block = _pixelate_block(document)
    ex0 = max(0, x0 - pad)
    ey0 = max(0, y0 - pad)
    if block > 1:
        ex0 -= ex0 % block
        ey0 -= ey0 % block
    ex1 = min(document.width, x1 + pad)
    ey1 = min(document.height, y1 + pad)
    if block > 1:
        ex1 = min(document.width, int(math.ceil(ex1 / block)) * block)
        ey1 = min(document.height, int(math.ceil(ey1 / block)) * block)
    if ex1 <= ex0 or ey1 <= ey0:
        return Image.new("RGBA", (max(1, x1 - x0), max(1, y1 - y0)), (0, 0, 0, 0))
    canvas = np.zeros((ey1 - ey0, ex1 - ex0, 4), dtype=np.float32)
    layers = document.layers
    for layer in layers:
        if layer.kind == "redaction" or not layer.visible or layer.opacity <= 0:
            continue
        owned = [mask for mask in masks_for(layers, layer) if mask.visible and mask.opacity > 0]
        if not owned:
            blit_layer(canvas, layer, ex0, ey0)
            continue
        solo = np.zeros_like(canvas)
        blit_layer(solo, _plain_target(layer), ex0, ey0)
        for mask in owned:
            solo = apply_redaction(solo, mask, ex0, ey0)
        canvas = blend_arrays(canvas, solo, layer.opacity, layer.blend_mode)
    flat = float_to_pil(canvas)
    return flat.crop((x0 - ex0, y0 - ey0, x1 - ex0, y1 - ey0))


def composite(document: Document) -> Image.Image:
    """Flatten ``document`` to one RGBA image the size of the canvas.

    Masks are skipped on their own turn and applied when their picture is
    drawn. A hidden picture hides its masks. A hidden mask is left out of
    that picture's buffer, and the other masks of the same picture still run.
    Opaque Normal layers are copied as bytes. Anything blended uses a reused
    float plate so a second flatten does not allocate another one.
    """
    fast = _composite_opaque(document)
    if fast is not None:
        return fast
    canvas = _borrow_float("canvas", document.height, document.width)
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

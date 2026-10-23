"""Flatten a document into one RGBA image.

Pipeline for each visible layer, bottom to top:

1. Redaction layers do not draw their own pixels. They take the canvas so
   far, run the mask effect (pixelate, blur, solid, noise, and the rest),
   and mix that back through the mask. Source layers underneath stay untouched.
2. Raster and text layers are blended with a CSS-style "simple alpha
   compositing" formula (source-over, with the blend mode applied only where
   both coverages overlap). That keeps Multiply/Screen/Overlay from turning
   into black when the backdrop is transparent.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from pixedit.model import BLEND_MODES, Document, Layer
from pixedit.pixels import mask_effect

# geom.place_image is imported lazily inside mask placement to keep import
# order boring: composite -> model, and geom -> model, neither imports the other
# at module level for placement helpers defined here.


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


def blit_layer(canvas: np.ndarray, layer: Layer) -> np.ndarray:
    image = layer.image
    overlap = _intersection(layer.x, layer.y, image.width, image.height, canvas.shape[1], canvas.shape[0])
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

    The effect is computed from the stack *below* this layer, so pixels in
    earlier layers are not rewritten. Later layers still composite normally
    and can sit above the mask (annotations, for example).
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


def composite(document: Document) -> Image.Image:
    """Flatten ``document`` to one RGBA image the size of the canvas."""
    canvas = np.zeros((document.height, document.width, 4), dtype=np.float32)
    for layer in document.layers:
        if not layer.visible or layer.opacity <= 0:
            continue
        if layer.kind == "redaction":
            canvas = apply_redaction(canvas, layer)
        else:
            canvas = blit_layer(canvas, layer)
    return float_to_pil(canvas)


def composite_over(document: Document, background: tuple[int, int, int, int]) -> Image.Image:
    """Flatten onto a solid background. Used by JPEG export."""
    flat = composite(document)
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

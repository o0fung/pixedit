"""Small layer previews for the layer list.

The list asks for a square picture of a raster or mask layer. Text layers do
not come through here: the row draws the text-tool icon and the start of the
string instead, because a sentence shrunk into this square is not readable.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from pixedit.model import Layer

# Logical size of a row icon. The panel multiplies by the screen scale.
THUMB_SIZE = 32
# The head, including the ellipsis, stays about this many characters.
TEXT_HEAD_LIMIT = 22

# Same two squares as the canvas checker. Even cells are the lighter one.
CHECKER_LIGHT = (0x2A, 0x30, 0x38, 255)
CHECKER_DARK = (0x1B, 0x20, 0x26, 255)
# Accent used by the sliders and the text-tool icon.
MASK_TINT = (0x3E, 0xC6, 0xFF)
# Light rim around ink that disappears on the dark checker. Not a recolor of the ink.
KEYLINE = (0xD5, 0xDE, 0xE6, 255)
# About 20% opaque. A fainter dab stays as it was painted.
_INK_ALPHA = 51
_CONTRAST_MIN = 3.0


def text_head(text: str, limit: int = TEXT_HEAD_LIMIT) -> str:
    """The front of a text layer's string, short enough for the row.

    Line breaks and repeated spaces collapse to one space. A string longer
    than ``limit`` keeps the front and ends with an ellipsis. The result is
    at most ``limit`` characters, and a space that would sit against the
    ellipsis is dropped.
    """
    flat = " ".join(text.split())
    if not flat or limit < 1:
        return ""
    if len(flat) <= limit:
        return flat
    if limit == 1:
        return "…"
    return flat[: limit - 1].rstrip() + "…"


def checker_image(size: int = THUMB_SIZE) -> Image.Image:
    """Opaque checker square. Transparent layer pixels sit on top of this."""
    return Image.fromarray(_checker(max(1, int(size))), "RGBA")


def layer_thumb(layer: Layer, size: int = THUMB_SIZE) -> Image.Image:
    """Square preview of a raster or mask layer.

    Text layers are not previewed here. The layer list draws the text icon.

    Flow:
    1. Start from a checker so transparent and white stay distinct.
    2. No ink, or an empty mask: return that checker.
    3. Crop to opaque pixels, or to the covered part of a mask.
    4. Fit the crop in the square. Growing uses nearest-neighbor so a sprite
       stays blocky. Shrinking takes every Nth pixel so a large photo is not
       fully resampled on every list refresh.
    5. A mask is tinted cyan from its coverage. A raster keeps its own alpha.
       Opacity and blend are left out; the row already shows the percent.
    6. Dark raster ink that sinks into the checker gets a one-pixel light rim.
       The ink color itself stays. Masks skip this; the cyan accent already shows.
    """
    size = max(1, int(size))
    canvas = _checker(size)
    if layer.kind == "redaction":
        return _mask_thumb(layer, canvas)
    return _raster_thumb(layer, canvas)


def _checker(size: int) -> np.ndarray:
    cell = max(1, size // 8)
    yy, xx = np.indices((size, size))
    light = ((xx // cell) + (yy // cell)) % 2 == 0
    image = np.empty((size, size, 4), np.uint8)
    image[light] = CHECKER_LIGHT
    image[~light] = CHECKER_DARK
    return image


def _raster_thumb(layer: Layer, canvas: np.ndarray) -> Image.Image:
    image = layer.image.convert("RGBA")
    box = image.getchannel("A").getbbox()
    if box is None:
        return Image.fromarray(canvas, "RGBA")
    fitted = _fit(image.crop(box), canvas.shape[0])
    return _composite(canvas, fitted)


def _mask_thumb(layer: Layer, canvas: np.ndarray) -> Image.Image:
    if layer.mask is None:
        return Image.fromarray(canvas, "RGBA")
    mask = layer.mask.convert("L")
    box = mask.getbbox()
    if box is None:
        return Image.fromarray(canvas, "RGBA")
    fitted = _fit(mask.crop(box), canvas.shape[0])
    return _tint(canvas, fitted)


def _fit(image: Image.Image, size: int) -> Image.Image:
    """Scale ``image`` so its long side equals ``size``, keeping the aspect."""
    width, height = image.size
    longest = max(width, height, 1)
    scale = size / longest
    target_w = min(size, max(1, int(round(width * scale))))
    target_h = min(size, max(1, int(round(height * scale))))
    if scale >= 1.0:
        if image.size == (target_w, target_h):
            return image
        return image.resize((target_w, target_h), Image.Resampling.NEAREST)
    # step is how many source pixels one thumbnail pixel stands for.
    step = max(1, int(math.floor(1.0 / scale)))
    sampled = np.ascontiguousarray(np.asarray(image)[::step, ::step])
    reduced = Image.fromarray(sampled)
    if reduced.size == (target_w, target_h):
        return reduced
    return reduced.resize((target_w, target_h), Image.Resampling.NEAREST)


def _place(canvas: np.ndarray, fitted: Image.Image) -> tuple[np.ndarray, int, int]:
    sprite = np.asarray(fitted)
    height, width = sprite.shape[:2]
    top = (canvas.shape[0] - height) // 2
    left = (canvas.shape[1] - width) // 2
    return sprite, top, left


def _composite(canvas: np.ndarray, fitted: Image.Image) -> Image.Image:
    """Draw a straight-alpha sprite over the opaque checker."""
    out = canvas.copy()
    sprite, top, left = _place(out, fitted.convert("RGBA"))
    src = sprite.astype(np.float32)
    alpha = src[..., 3:4] / 255.0
    patch = out[top : top + src.shape[0], left : left + src.shape[1], :3].astype(np.float32)
    mixed = src[..., :3] * alpha + patch * (1.0 - alpha)
    out[top : top + src.shape[0], left : left + src.shape[1], :3] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
    _paint_keyline(out, sprite, top, left)
    return Image.fromarray(out, "RGBA")


def _tint(canvas: np.ndarray, fitted: Image.Image) -> Image.Image:
    """Use mask coverage as the amount of accent color over the checker."""
    out = canvas.copy()
    coverage, top, left = _place(out, fitted.convert("L"))
    alpha = coverage.astype(np.float32) / 255.0
    patch = out[top : top + alpha.shape[0], left : left + alpha.shape[1], :3].astype(np.float32)
    tint = np.array(MASK_TINT, np.float32)
    mixed = tint * alpha[..., None] + patch * (1.0 - alpha[..., None])
    out[top : top + alpha.shape[0], left : left + alpha.shape[1], :3] = np.clip(np.rint(mixed), 0, 255).astype(
        np.uint8
    )
    return Image.fromarray(out, "RGBA")


def _srgb_luminance(rgb: tuple[int, int, int]) -> float:
    """Relative luminance of one sRGB color, in 0..1."""

    def channel(value: int) -> float:
        s = value / 255.0
        if s <= 0.04045:
            return s / 12.92
        return ((s + 0.055) / 1.055) ** 2.4

    red, green, blue = (channel(c) for c in rgb)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


_DARK_LUM = _srgb_luminance(CHECKER_DARK[:3])


def _luminance(rgb: np.ndarray) -> np.ndarray:
    s = rgb.astype(np.float32) * (1.0 / 255.0)
    linear = np.where(s <= 0.04045, s * (1.0 / 12.92), ((s + 0.055) * (1.0 / 1.055)) ** 2.4)
    return 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]


def _below_contrast(rgb: np.ndarray, background: float) -> np.ndarray:
    """True where ``rgb`` is under ``_CONTRAST_MIN`` against ``background`` luminance.

    Contrast is ``(lighter + 0.05) / (darker + 0.05)`` on relative luminance.
    """
    lum = _luminance(rgb)
    lighter = np.maximum(lum, background)
    darker = np.minimum(lum, background)
    return (lighter + 0.05) / (darker + 0.05) < _CONTRAST_MIN


def _dilate(mask: np.ndarray) -> np.ndarray:
    """One-pixel 8-neighborhood. The pad stops the rim wrapping to the other edge."""
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    height, width = mask.shape
    near = np.zeros_like(mask)
    for dy in range(3):
        for dx in range(3):
            near |= padded[dy : dy + height, dx : dx + width]
    return near


def _paint_keyline(canvas: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    """Light rim around raster ink that disappears on the dark checker.

    The stroke color stays. Flow:
    1. Keep pixels that are about 20% opaque and under 3:1 contrast against
       the darker checker square.
    2. Paint the rim on the checker just outside that ink, so a black stroke
       stays black and the edge is what shows.
    3. If that ink is the icon border, paint the border pixel itself. A solid
       dark layer would otherwise match the panel behind the icon.
    """
    alpha = sprite[..., 3]
    ink = alpha >= _INK_ALPHA
    if not np.any(ink):
        return
    fails_sprite = ink & _below_contrast(sprite[..., :3], _DARK_LUM)
    if not np.any(fails_sprite):
        return
    height, width = canvas.shape[:2]
    sprite_h, sprite_w = fails_sprite.shape
    fails = np.zeros((height, width), dtype=bool)
    covered = np.zeros((height, width), dtype=bool)
    fails[top : top + sprite_h, left : left + sprite_w] = fails_sprite
    covered[top : top + sprite_h, left : left + sprite_w] = ink
    tone = np.array(KEYLINE[:3], np.uint8)
    # Outside the ink only. A white pixel beside a black one keeps its color.
    canvas[_dilate(fails) & ~covered, :3] = tone
    edge = np.zeros((height, width), dtype=bool)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True
    canvas[fails & edge, :3] = tone

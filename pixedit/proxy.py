"""Small copies used while a drag is in progress.

The stored pixels stay full resolution. A live grade, resample, or gradient
runs on a screen-sized stand-in when the layer is large, and the release
computes the real bitmap once. A picture that already fits on screen is left
alone, so a small edit stays exact the whole way through.
"""

from __future__ import annotations

import math

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


def layer_sample(layer) -> tuple[Image.Image, float, float]:
    """Bitmap to read, and the scale from layer pixels into that bitmap."""
    image = layer.image
    preview = getattr(layer, "preview_source", None)
    if preview is None or image.width < 1 or image.height < 1:
        return image, 1.0, 1.0
    if preview.size == image.size:
        return preview, 1.0, 1.0
    return preview, preview.width / float(image.width), preview.height / float(image.height)

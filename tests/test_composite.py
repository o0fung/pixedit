"""Blend modes and flattening."""

from __future__ import annotations

import numpy as np
from PIL import Image

from pixedit.composite import blend_arrays, composite
from pixedit.factory import raster_layer
from pixedit.model import Document


def _solid(color, size=2):
    return raster_layer("L", Image.new("RGBA", (size, size), color))


def test_normal_half_alpha_over_white():
    dst = np.zeros((1, 1, 4), np.float32)
    dst[..., :] = (1, 1, 1, 1)
    src = np.zeros((1, 1, 4), np.float32)
    src[..., :] = (0, 0, 0, 0.5)
    out = blend_arrays(dst, src, 1.0, "normal")
    assert out[0, 0, 3] == 1
    assert np.allclose(out[0, 0, :3], 0.5, atol=1e-5)


def test_multiply_and_screen_opaque():
    dst = np.ones((1, 1, 4), np.float32)
    dst[..., :3] = (1, 0, 0)
    src = np.ones((1, 1, 4), np.float32)
    src[..., :3] = (0, 0, 1)
    multiply = blend_arrays(dst, src, 1, "multiply")
    screen = blend_arrays(dst, src, 1, "screen")
    assert np.allclose(multiply[0, 0, :3], 0)
    assert np.allclose(screen[0, 0, :3], (1, 0, 1))


def test_overlay_known_sample():
    dst = np.ones((1, 1, 4), np.float32)
    dst[..., :3] = (0.25, 0.75, 0.5)
    src = np.ones((1, 1, 4), np.float32)
    src[..., :3] = (1, 0, 0.5)
    out = blend_arrays(dst, src, 1, "overlay")
    # Dark backdrop channel uses multiply; light channel uses screen.
    assert np.allclose(out[0, 0, :3], (0.5, 0.5, 0.5), atol=1e-5)


def test_multiply_over_transparent_keeps_source():
    dst = np.zeros((1, 1, 4), np.float32)
    src = np.zeros((1, 1, 4), np.float32)
    src[..., :] = (0.2, 0.4, 0.6, 1)
    out = blend_arrays(dst, src, 1, "multiply")
    assert np.allclose(out[0, 0], (0.2, 0.4, 0.6, 1), atol=1e-5)


def test_composite_respects_visibility_opacity_and_order():
    bottom = _solid((255, 0, 0, 255))
    top = _solid((0, 0, 255, 255))
    top.opacity = 0.5
    hidden = _solid((0, 255, 0, 255))
    hidden.visible = False
    doc = Document(2, 2, [bottom, hidden, top])
    pixel = composite(doc).getpixel((0, 0))
    # 50% opaque blue over opaque red is an even mix. The hidden green layer stays out.
    assert pixel[0] == pixel[2] == 128
    assert pixel[1] == 0

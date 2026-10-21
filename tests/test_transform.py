"""Crop, trim, scale, rotate, and canvas transforms."""

from __future__ import annotations

import numpy as np
from PIL import Image

from pixedit.composite import composite
from pixedit.factory import raster_layer
from pixedit.geom import (
    canvas_flip,
    canvas_rotate_cw,
    crop_document,
    scaled_top_left,
    size_from_anchor,
    trim_rect,
)
from pixedit.model import Document, new_document, visual_state
from pixedit.geom import rotate_visual, scale_visual
from pixedit.session import Session


def test_scaled_top_left_keeps_center():
    assert scaled_top_left(0, 0, 10, 10, 4, 6) == (3, 2)


def test_size_from_anchor_locks_aspect():
    left, top, width, height = size_from_anchor((0, 0), (10, 2), lock_aspect=True, aspect=2)
    assert width / height == 2
    assert left == 0 and top == 0


def test_crop_shifts_layer_content():
    image = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    image.putpixel((2, 3), (10, 20, 30, 255))
    doc = Document(10, 10, [raster_layer("A", image)])
    layers, _selection = crop_document(doc, 2, 3, 4, 4)
    assert layers[0].x == 0 and layers[0].y == 0
    assert layers[0].image.getpixel((0, 0)) == (10, 20, 30, 255)
    assert layers[0].image.size == (4, 4)


def test_trim_removes_empty_edges():
    session = Session(new_document(20, 18, "transparent"))
    layer = session.document.layers[0]
    layer.image.putpixel((5, 6), (255, 0, 0, 255))
    layer.image.putpixel((7, 9), (0, 255, 0, 255))
    rect = trim_rect(session.document)
    assert rect == (5, 6, 3, 4)
    session.trim()
    assert (session.document.width, session.document.height) == (3, 4)
    assert session.document.layers[0].image.getpixel((0, 0))[0] == 255
    assert session.document.layers[0].image.getpixel((2, 3))[1] == 255


def test_scale_and_rotate_keep_center():
    image = Image.new("RGBA", (8, 4), (255, 0, 0, 255))
    layer = raster_layer("A", image, x=2, y=6)
    state = visual_state(layer)
    scaled = scale_visual(state, 16, 8)
    old_cx = state["x"] + 8 / 2
    new_cx = scaled["x"] + scaled["image"].width / 2
    assert abs(old_cx - new_cx) <= 0.6
    rotated = rotate_visual(state, 90)
    rot_cx = rotated["x"] + rotated["image"].width / 2
    rot_cy = rotated["y"] + rotated["image"].height / 2
    assert abs(rot_cx - (state["x"] + 4)) <= 1
    assert abs(rot_cy - (state["y"] + 2)) <= 1


def test_canvas_rotate_matches_composite():
    image = Image.new("RGBA", (5, 3), (0, 0, 0, 0))
    image.putpixel((0, 0), (255, 0, 0, 255))
    image.putpixel((4, 2), (0, 0, 255, 255))
    doc = Document(5, 3, [raster_layer("A", image)])
    expected = composite(doc).transpose(Image.Transpose.ROTATE_270)
    layers, width, height, _sel = canvas_rotate_cw(doc)
    rotated = Document(width, height, layers)
    assert (width, height) == (3, 5)
    assert np.array_equal(np.array(composite(rotated)), np.array(expected))


def test_canvas_flip_matches_composite():
    image = Image.new("RGBA", (4, 3), (0, 0, 0, 255))
    image.putpixel((0, 1), (255, 255, 0, 255))
    doc = Document(4, 3, [raster_layer("A", image)])
    expected = composite(doc).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    layers, width, height, _sel = canvas_flip(doc, True)
    flipped = Document(width, height, layers)
    assert np.array_equal(np.array(composite(flipped)), np.array(expected))

"""Crop, trim, scale, rotate, and canvas transforms."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from pixedit.composite import composite
from pixedit.factory import raster_layer, text_layer
from pixedit.geom import (
    canvas_flip,
    canvas_rotate_cw,
    canvas_window,
    crop_document,
    flip_visual,
    flip_visual_across,
    linked_dpi,
    linked_inches,
    linked_percent,
    linked_pixels,
    refit_canvas,
    resample_document,
    rotate_visual,
    rotate_visual_around,
    scale_visual,
    scale_visual_in_box,
    scaled_top_left,
    scales_match,
    size_from_anchor,
    trim_rect,
)
from pixedit.model import Document, TextInfo, new_document, visual_state
from pixedit.session import Session
from pixedit.tools.base import PointerEvent
from pixedit.tools.builtins import CropTool, TransformTool


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


def test_proportion_edits_link_percent_and_pixels():
    assert linked_percent(8, 4, 50, 100, axis="width", keep=True) == (4, 2, 50.0, 50.0)
    assert linked_percent(8, 4, 50, 25, axis="height", keep=False) == (4, 1, 50.0, 25.0)
    assert linked_pixels(8, 4, 4, 4, axis="width", keep=True)[:2] == (4, 2)
    width, height, width_in, height_in, dpi = linked_dpi(100, 50, 72, 144, resample=True)
    assert (width, height) == (200, 100)
    assert abs(width_in - 200 / 144) < 1e-6
    width, height, width_in, height_in, dpi = linked_dpi(100, 50, 72, 144, resample=False)
    assert (width, height, dpi) == (100, 50, 144)
    assert abs(width_in - 100 / 144) < 1e-6
    width, height, _width_in, height_in, dpi = linked_inches(
        100, 50, 100, 50, 2, 50 / 72, 72, axis="width", keep=True, resample=False
    )
    assert (width, height) == (100, 50)
    assert abs(dpi - 50) < 1e-6
    assert abs(height_in - 1) < 1e-6


def test_canvas_anchor_expands_and_trims():
    assert canvas_window(10, 10, 14, 12, "nw") == (0, 0, 14, 12)
    assert canvas_window(10, 10, 14, 12, "se") == (-4, -2, 14, 12)
    assert canvas_window(10, 10, 6, 6, "c") == (2, 2, 6, 6)

    image = Image.new("RGBA", (10, 10), (255, 255, 255, 255))
    image.putpixel((1, 1), (255, 0, 0, 255))
    doc = Document(10, 10, [raster_layer("Background", image)])
    layers, _selection = refit_canvas(doc, 14, 10, "ne", (255, 255, 255, 255))
    assert len(layers) == 1
    assert layers[0].image.size == (14, 10)
    assert layers[0].x == 0 and layers[0].y == 0
    assert layers[0].image.getpixel((5, 1))[:3] == (255, 0, 0)
    assert layers[0].image.getpixel((0, 0))[:3] == (255, 255, 255)

    session = Session(new_document(10, 10, "white"))
    session.document.layers[0].image.putpixel((2, 2), (0, 255, 0, 255))
    session.document.layers[0].image.putpixel((7, 7), (0, 0, 255, 255))
    session.resize_canvas(6, 6, "c")
    assert (session.document.width, session.document.height) == (6, 6)
    assert session.document.layers[0].image.getpixel((0, 0))[1] == 255
    assert session.document.layers[0].image.getpixel((5, 5))[2] == 255
    session.undo()
    assert (session.document.width, session.document.height) == (10, 10)

    small = raster_layer("Mark", Image.new("RGBA", (4, 4), (255, 0, 0, 255)))
    session = Session(Document(10, 10, [small]))
    session.resize_canvas(12, 10, "c", (0, 0, 255, 255))
    assert session.document.layers[0].name == "Canvas"
    assert session.document.layers[0].image.getpixel((0, 0))[:3] == (0, 0, 255)
    assert session.document.layers[1].image.getpixel((0, 0))[0] == 255


def test_crop_drag_can_expand_and_lock_proportion():
    session = Session(new_document(10, 10, "white"))
    session.document.layers[0].image.putpixel((1, 1), (255, 0, 0, 255))
    tool = CropTool()
    tool.pointer_down(session, PointerEvent(-2, -2))
    tool.pointer_up(session, PointerEvent(12, 12))
    assert (session.document.width, session.document.height) == (14, 14)
    assert session.document.layers[0].x == 2 and session.document.layers[0].y == 2
    assert session.document.layers[0].image.getpixel((1, 1))[0] == 255

    wide = Session(new_document(20, 10, "white"))
    tool = CropTool()
    wide.options["crop_ratio"] = "1:1"
    tool.pointer_down(wide, PointerEvent(0, 0))
    tool.pointer_move(wide, PointerEvent(10, 4))
    assert tool._overlay_rect is not None
    assert tool._overlay_rect[2] == tool._overlay_rect[3]

    tool.pointer_cancel(wide)
    wide.options["crop_ratio"] = "free"
    tool.pointer_down(wide, PointerEvent(0, 0))
    tool.pointer_move(wide, PointerEvent(10, 10, shift=True))
    assert tool._overlay_rect is not None
    assert tool._overlay_rect[2] / tool._overlay_rect[3] == 2

    tool.pointer_cancel(wide)
    wide.options["crop_ratio"] = "custom"
    wide.options["crop_ratio_w"] = 2
    wide.options["crop_ratio_h"] = 1
    tool.pointer_down(wide, PointerEvent(0, 0))
    tool.pointer_move(wide, PointerEvent(8, 8))
    assert tool._overlay_rect is not None
    assert tool._overlay_rect[2] / tool._overlay_rect[3] == 2


def test_resample_document_scales_pixels_or_only_dpi():
    assert scales_match(40, 20, 20, 10)
    assert scales_match(3, 2, 2, 1)
    assert not scales_match(40, 20, 30, 20)

    session = Session(new_document(20, 10, "white"))
    # Nearest samples the center of the destination pixel, so a single source
    # pixel at (0, 0) can miss. A 2×2 block covers that sample.
    for x in range(2):
        for y in range(2):
            session.document.layers[0].image.putpixel((x, y), (9, 8, 7, 255))
    session.document.selection = Image.new("L", (20, 10), 0)
    for x in range(2):
        for y in range(2):
            session.document.selection.putpixel((x, y), 255)
    session.resize_image(10, 5, 72, True, "nearest")
    assert (session.document.width, session.document.height) == (10, 5)
    assert session.document.layers[0].image.size == (10, 5)
    assert session.document.layers[0].image.getpixel((0, 0))[:3] == (9, 8, 7)
    assert session.document.selection is not None
    assert session.document.selection.size == (10, 5)
    assert session.document.selection.getpixel((0, 0)) == 255
    session.undo()
    assert (session.document.width, session.document.height) == (20, 10)
    assert session.document.dpi == 72

    session.resize_image(20, 10, 144, resample=False)
    assert session.document.dpi == 144
    assert session.document.layers[0].image.size == (20, 10)
    session.undo()
    assert session.document.dpi == 72

    info = TextInfo("Hi", None, "Default", 40, (0, 0, 0, 255))
    doc = new_document(40, 20, "white")
    doc.layers.append(text_layer("Title", info, 2, 3))
    uniform = Session(doc)
    uniform.resize_image(20, 10, 72, True, "lanczos")
    text = uniform.document.layers[-1]
    assert text.kind == "text" and text.text is not None
    assert text.text.size == 20
    assert (text.x, text.y) == (1, 2)

    uneven = Session(new_document(40, 20, "white"))
    uneven.document.layers.append(text_layer("Title", TextInfo("Hi", None, "Default", 40, (0, 0, 0, 255)), 2, 3))
    uneven.resize_image(30, 20, 72, True, "lanczos")
    assert uneven.document.layers[-1].kind == "raster"
    assert uneven.document.layers[-1].text is None

    mask = Image.new("L", (20, 10), 0)
    for x in range(2):
        for y in range(2):
            mask.putpixel((x, y), 255)
    from pixedit.factory import redaction_layer

    covered = Session(new_document(20, 10, "white"))
    covered.document.layers.append(redaction_layer("Mask", mask, "pixelate", 8))
    layers, _selection = resample_document(covered.document, 10, 5, "nearest")
    assert layers[-1].kind == "redaction"
    assert layers[-1].mask is not None
    assert layers[-1].mask.size == (10, 5)
    assert layers[-1].mask.getpixel((0, 0)) == 255


def test_canvas_flip_matches_composite():
    image = Image.new("RGBA", (4, 3), (0, 0, 0, 255))
    image.putpixel((0, 1), (255, 255, 0, 255))
    doc = Document(4, 3, [raster_layer("A", image)])
    expected = composite(doc).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    layers, width, height, _sel = canvas_flip(doc, True)
    flipped = Document(width, height, layers)
    assert np.array_equal(np.array(composite(flipped)), np.array(expected))


def _center(layer) -> tuple[float, float]:
    image = layer.bitmap()
    return layer.x + image.width / 2.0, layer.y + image.height / 2.0


def test_shared_transform_of_one_layer_matches_the_old_one():
    image = Image.new("RGBA", (4, 3), (0, 0, 0, 255))
    image.putpixel((0, 1), (255, 0, 0, 255))
    state = visual_state(raster_layer("A", image, 2, 5))
    center = (2 + 4 / 2.0, 5 + 3 / 2.0)
    flipped = flip_visual_across(state, True, center)
    direct = flip_visual(state, True)
    assert (flipped["x"], flipped["y"]) == (direct["x"], direct["y"]) == (2, 5)
    assert flipped["image"].getpixel((3, 1)) == (255, 0, 0, 255)

    turned = rotate_visual_around(state, 90, center)
    direct_turn = rotate_visual(state, 90)
    assert (turned["x"], turned["y"]) == (direct_turn["x"], direct_turn["y"])
    assert turned["image"].size == direct_turn["image"].size

    # The fixed corner is the layer's own bottom-right, so it stays put.
    scaled = scale_visual_in_box(state, (6, 8), "se", 2, 2)
    assert scaled["image"].size == (8, 6)
    assert (scaled["x"], scaled["y"]) == (-2, 2)


def test_single_layer_corner_scale_stays_on_the_anchor():
    image = Image.new("RGBA", (8, 4), (255, 0, 0, 255))
    layer = raster_layer("A", image, 2, 6)
    session = Session(Document(40, 40, [layer]))
    tool = TransformTool()
    tool.pointer_down(session, PointerEvent(10, 10))
    tool.pointer_up(session, PointerEvent(18, 14))
    assert layer.image.size == (16, 8)
    assert (layer.x, layer.y) == (2, 6)
    assert session.history.undo_stack[-1].name == "Transform"


def test_rotate_drag_keeps_the_frame_size():
    image = Image.new("RGBA", (40, 10), (255, 0, 0, 255))
    layer = raster_layer("A", image, 10, 20)
    session = Session(Document(120, 80, [layer]))
    tool = TransformTool()
    # The handle is 28px above the top edge at zoom 1. The box center is (30, 25).
    tool.pointer_down(session, PointerEvent(30, -8))
    # Same radius, to the right of the center: 90° clockwise.
    tool.pointer_move(session, PointerEvent(63, 25))
    overlay = tool.overlays(session)[0]
    corners = overlay.points
    edges = [
        math.hypot(corners[i][0] - corners[(i + 1) % 4][0], corners[i][1] - corners[(i + 1) % 4][1])
        for i in range(4)
    ]
    assert abs(edges[0] - 40) < 0.01
    assert abs(edges[1] - 10) < 0.01
    assert abs(edges[2] - 40) < 0.01
    assert abs(edges[3] - 10) < 0.01
    # The straight bitmap has swapped sides. The frame did not follow it.
    assert layer.image.size == (10, 40)
    assert abs(layer.x + layer.image.width / 2 - 30) <= 1
    assert abs(layer.y + layer.image.height / 2 - 25) <= 1
    rot = next(handle for handle in overlay.handles if handle[0] == "rot")
    assert abs(rot[1] - 63) < 0.01 and abs(rot[2] - 25) < 0.01

    tool.pointer_up(session, PointerEvent(63, 25))
    released = tool.overlays(session)[0]
    assert released.points == []
    assert released.rect is not None
    assert abs(released.rect[2] - 10) <= 1 and abs(released.rect[3] - 40) <= 1


def test_transform_hover_cursor_follows_the_handle():
    image = Image.new("RGBA", (40, 30), (255, 0, 0, 255))
    layer = raster_layer("A", image, 10, 20)
    session = Session(Document(120, 80, [layer]))
    tool = TransformTool()
    # Corners of the straight box, and the rotate handle 28px above the top.
    assert tool.hover_cursor(session, PointerEvent(10, 20)) == "size-fdiag"
    assert tool.hover_cursor(session, PointerEvent(50, 50)) == "size-fdiag"
    assert tool.hover_cursor(session, PointerEvent(50, 20)) == "size-bdiag"
    assert tool.hover_cursor(session, PointerEvent(10, 50)) == "size-bdiag"
    assert tool.hover_cursor(session, PointerEvent(30, -8)) == "rotate"
    assert tool.hover_cursor(session, PointerEvent(30, 35)) == "size"
    assert TransformTool().hover_cursor(Session(Document(40, 40, [])), PointerEvent(10, 20)) == "size"

    tool.pointer_down(session, PointerEvent(30, -8))
    tool.pointer_move(session, PointerEvent(80, 70))
    assert tool.hover_cursor(session, PointerEvent(80, 70)) == "rotate"
    tool.pointer_cancel(session)
    assert tool.hover_cursor(session, PointerEvent(30, 35)) == "size"


def test_group_move_is_one_undo_step():
    session = Session(new_document(80, 80, "white"))
    left = raster_layer("Left", Image.new("RGBA", (20, 20), (255, 0, 0, 255)), 10, 40)
    right = raster_layer("Right", Image.new("RGBA", (20, 20), (0, 255, 0, 255)), 50, 40)
    session.add_layer(left)
    session.add_layer(right)
    session.select_layers([left.id, right.id], active_id=left.id)
    steps = len(session.history.undo_stack)
    tool = TransformTool()
    tool.pointer_down(session, PointerEvent(20, 50))
    tool.pointer_up(session, PointerEvent(28, 54))
    assert (left.x, left.y) == (18, 44)
    assert (right.x, right.y) == (58, 44)
    assert len(session.history.undo_stack) == steps + 1
    assert session.history.undo_stack[-1].name == "Move layers"
    session.undo()
    assert (left.x, left.y) == (10, 40)
    assert (right.x, right.y) == (50, 40)
    assert set(session.selected_ids()) == {left.id, right.id}


def test_group_scale_keeps_the_anchored_corner_layer():
    left = raster_layer("Left", Image.new("RGBA", (10, 10), (255, 0, 0, 255)))
    right = raster_layer("Right", Image.new("RGBA", (10, 10), (0, 0, 255, 255)), 10, 0)
    session = Session(Document(80, 80, [left, right]))
    session.select_layers([left.id, right.id], active_id=left.id)
    tool = TransformTool()
    # The union's bottom-right is fixed to the opposite corner, the top-left.
    tool.pointer_down(session, PointerEvent(20, 10))
    tool.pointer_up(session, PointerEvent(40, 20))
    assert (left.x, left.y) == (0, 0)
    assert left.image.size == (20, 20)
    assert (right.x, right.y) == (20, 0)
    assert right.image.size == (20, 20)
    session.undo()
    assert (left.x, left.y) == (0, 0)
    assert left.image.size == (10, 10)
    assert (right.x, right.y) == (10, 0)


def test_group_rotate_orbits_layers_around_the_shared_center():
    left_image = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
    left_image.putpixel((0, 2), (0, 255, 0, 255))
    left = raster_layer("Left", left_image)
    right = raster_layer("Right", Image.new("RGBA", (10, 10), (0, 0, 255, 255)), 10, 0)
    session = Session(Document(40, 40, [left, right]))
    session.select_layers([left.id, right.id], active_id=left.id)
    session.rotate_active_cw()
    # 90° clockwise around the pair's center (10, 5): the left layer goes
    # above that center and the right layer goes below it.
    left_c = _center(left)
    right_c = _center(right)
    assert abs(left_c[0] - 10) <= 0.6 and abs(left_c[1] - 0) <= 0.6
    assert abs(right_c[0] - 10) <= 0.6 and abs(right_c[1] - 10) <= 0.6
    assert left.kind == "raster" and right.kind == "raster"
    session.undo()
    assert (left.x, left.y) == (0, 0)
    assert (right.x, right.y) == (10, 0)
    assert left.image.getpixel((0, 2))[:3] == (0, 255, 0)


def test_group_flip_swaps_sides_and_keeps_text_on_a_turn():
    left_image = Image.new("RGBA", (10, 10), (0, 0, 255, 255))
    left_image.putpixel((0, 2), (255, 0, 0, 255))
    left = raster_layer("Left", left_image)
    right = raster_layer("Right", Image.new("RGBA", (10, 10), (0, 255, 0, 255)), 10, 0)
    session = Session(Document(40, 40, [left, right]))
    session.select_layers([left.id, right.id], active_id=right.id)
    session.flip_active_horizontal()
    assert (left.x, left.y) == (10, 0)
    assert (right.x, right.y) == (0, 0)
    assert left.image.getpixel((9, 2))[:3] == (255, 0, 0)
    assert session.history.undo_stack[-1].name == "Flip horizontal"
    session.undo()
    assert (left.x, left.y) == (0, 0)
    assert left.image.getpixel((0, 2))[:3] == (255, 0, 0)

    title = text_layer("Title", TextInfo("Hi", None, "Default", 24, (0, 0, 0, 255)), 4, 4)
    session.add_layer(title)
    session.select_layers([left.id, title.id], active_id=title.id)
    session.rotate_active_cw()
    assert title.kind == "text"
    assert title.text is not None and title.text.text == "Hi"
    assert left.kind == "raster"


def test_clicking_changes_the_layer_selection():
    session = Session(new_document(80, 80, "white"))
    background = session.document.layers[0]
    left = raster_layer("Left", Image.new("RGBA", (20, 20), (255, 0, 0, 255)), 10, 40)
    right = raster_layer("Right", Image.new("RGBA", (20, 20), (0, 255, 0, 255)), 50, 40)
    session.add_layer(left)
    session.add_layer(right)
    session.select_layers([left.id], active_id=left.id)
    tool = TransformTool()

    # A plain drag that starts on another layer moves the selection only.
    tool.pointer_down(session, PointerEvent(60, 50))
    tool.pointer_up(session, PointerEvent(68, 54))
    assert session.selected_ids() == [left.id]
    assert (left.x, left.y) == (18, 44)
    assert (right.x, right.y) == (50, 40)
    assert (background.x, background.y) == (0, 0)
    session.undo()
    assert (left.x, left.y) == (10, 40)

    # A drag that starts outside the selection does the same.
    tool.pointer_down(session, PointerEvent(-4, -6))
    tool.pointer_up(session, PointerEvent(2, 0))
    assert session.selected_ids() == [left.id]
    assert (left.x, left.y) == (16, 46)
    assert (right.x, right.y) == (50, 40)
    assert (background.x, background.y) == (0, 0)
    session.undo()

    # A click with no movement does not retarget and does not commit.
    tool.pointer_down(session, PointerEvent(5, 5))
    tool.pointer_up(session, PointerEvent(5, 5))
    assert session.selected_ids() == [left.id]
    assert (left.x, left.y) == (10, 40)
    assert (background.x, background.y) == (0, 0)

    # A plain click on a member of a multi-selection keeps the set.
    session.select_layers([left.id, right.id], active_id=right.id)
    tool.pointer_down(session, PointerEvent(20, 50))
    tool.pointer_up(session, PointerEvent(20, 50))
    assert set(session.selected_ids()) == {left.id, right.id}
    assert session.active_layer().id == right.id

    # Shift adds. Command/Ctrl removes, and the last layer stays put.
    session.layer_index = session.document.index_of(left.id)
    tool.pointer_down(session, PointerEvent(60, 50, shift=True))
    tool.pointer_up(session, PointerEvent(60, 50, shift=True))
    assert set(session.selected_ids()) == {left.id, right.id}
    assert session.active_layer().id == right.id
    tool.pointer_down(session, PointerEvent(60, 50, ctrl=True))
    tool.pointer_up(session, PointerEvent(60, 50, ctrl=True))
    assert session.selected_ids() == [left.id]


def test_scale_dialog_stays_on_the_active_layer():
    session = Session(new_document(40, 40, "white"))
    left = raster_layer("Left", Image.new("RGBA", (10, 10), (255, 0, 0, 255)), 4, 4)
    right = raster_layer("Right", Image.new("RGBA", (10, 10), (0, 255, 0, 255)), 20, 4)
    session.add_layer(left)
    session.add_layer(right)
    session.select_layers([left.id, right.id], active_id=left.id)
    session.transform_active("scale", width=20, height=20)
    assert left.image.size == (20, 20)
    assert right.image.size == (10, 10)
    assert set(session.selected_ids()) == {left.id, right.id}

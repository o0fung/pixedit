"""Brush, eraser, fill, selection, and eyedropper behavior without Qt."""

from __future__ import annotations

from PIL import Image, ImageDraw

from pixedit.model import new_document
from pixedit.paint import connected_mask, dab, paint_points
from pixedit.session import Session
from pixedit.tools.base import PointerEvent
from pixedit.tools.builtins import BrushTool, EraserTool, EyedropperTool, FillTool, SelectTool, ShapeTool
import numpy as np


def test_hard_brush_is_a_disk_and_respects_selection():
    image = Image.new("RGBA", (30, 30), (0, 0, 0, 0))
    selection = Image.new("L", (30, 30), 0)
    selection.paste(255, (0, 0, 15, 30))
    painted = paint_points(
        image,
        [(10, 15), (20, 15)],
        radius=3,
        hardness=1,
        color=(255, 0, 0, 255),
        opacity=1,
        selection=selection,
    )
    assert painted.getpixel((10, 15))[0] == 255
    assert painted.getpixel((20, 15))[3] == 0
    # Outside the radius the left dab did not paint.
    assert painted.getpixel((10, 15 + 5))[3] == 0


def test_eraser_reduces_alpha():
    arr = np.full((20, 20, 4), 255, np.uint8)
    dab(arr, 10, 10, radius=4, hardness=1, color=(0, 0, 0, 255), opacity=1, erase=True, selection=None)
    assert arr[10, 10, 3] == 0
    assert arr[0, 0, 3] == 255


def test_brush_tool_undo():
    session = Session(new_document(40, 40, "white"))
    session.foreground = (255, 0, 0, 255)
    session.options["brush_size"] = 6
    session.options["brush_hardness"] = 100
    session.options["brush_opacity"] = 100
    tool = BrushTool()
    tool.pointer_down(session, PointerEvent(12, 12))
    tool.pointer_move(session, PointerEvent(20, 16))
    tool.pointer_up(session, PointerEvent(22, 18))
    layer = session.active_layer()
    assert layer.image.getpixel((12, 12))[0] > 200
    session.undo()
    assert layer.image.getpixel((12, 12))[:3] == (255, 255, 255)


def test_eraser_tool_and_fill_and_shape():
    session = Session(new_document(24, 24, "white"))
    session.foreground = (0, 0, 255, 255)
    tool = EraserTool()
    tool.pointer_down(session, PointerEvent(4, 4))
    tool.pointer_up(session, PointerEvent(4, 4))
    assert session.active_layer().image.getpixel((4, 4))[3] < 255

    session.foreground = (0, 180, 0, 255)
    session.options["tolerance"] = 0
    FillTool().pointer_down(session, PointerEvent(20, 20))
    assert session.active_layer().image.getpixel((20, 20))[:3] == (0, 180, 0)

    session.foreground = (255, 0, 0, 255)
    session.options["shape_kind"] = "rect"
    session.options["shape_filled"] = True
    session.options["stroke_width"] = 1
    shape = ShapeTool()
    shape.pointer_down(session, PointerEvent(8, 8))
    shape.pointer_up(session, PointerEvent(14, 14))
    assert session.active_layer().image.getpixel((11, 11))[0] > 200


def test_magic_wand_is_contiguous():
    image = Image.new("RGBA", (20, 20), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((2, 2, 6, 6), fill=(255, 0, 0, 255))
    draw.rectangle((12, 12, 16, 16), fill=(255, 0, 0, 255))
    mask = connected_mask(image, 4, 4, tolerance=0)
    assert mask.getpixel((4, 4)) == 255
    assert mask.getpixel((14, 14)) == 0
    assert mask.getpixel((0, 0)) == 0

    session = Session(new_document(20, 20, "white"))
    session.document.layers[0].image = image
    session.options["select_mode"] = "wand"
    session.options["tolerance"] = 0
    SelectTool().pointer_down(session, PointerEvent(4, 4))
    selection = session.document.selection
    assert selection is not None
    assert selection.getpixel((4, 4)) == 255
    assert selection.getpixel((14, 14)) == 0
    session.undo()
    assert session.document.selection is None


def test_rect_selection_shift_adds():
    session = Session(new_document(30, 30, "white"))
    session.options["select_mode"] = "rect"
    tool = SelectTool()
    tool.pointer_down(session, PointerEvent(1, 1))
    tool.pointer_up(session, PointerEvent(8, 8))
    tool.pointer_down(session, PointerEvent(12, 12, shift=True))
    tool.pointer_up(session, PointerEvent(18, 18, shift=True))
    selection = session.document.selection
    assert selection.getpixel((3, 3)) == 255
    assert selection.getpixel((15, 15)) == 255
    assert selection.getpixel((10, 10)) == 0


def test_inverted_drag_still_fills_rect_and_ellipse():
    """A drag up and to the left used to abort: Pillow requires x1 >= x0."""
    session = Session(new_document(30, 30, "white"))
    session.options["select_mode"] = "rect"
    tool = SelectTool()
    tool.pointer_down(session, PointerEvent(18, 18))
    tool.pointer_up(session, PointerEvent(2, 2))
    selection = session.document.selection
    assert selection.getpixel((10, 10)) == 255
    assert selection.getpixel((0, 0)) == 0

    session.options["select_mode"] = "ellipse"
    tool.pointer_down(session, PointerEvent(18, 18))
    tool.pointer_up(session, PointerEvent(2, 2))
    selection = session.document.selection
    assert selection.getpixel((10, 10)) == 255
    assert selection.getpixel((0, 0)) == 0

    session.foreground = (255, 0, 0, 255)
    session.options["shape_kind"] = "rect"
    session.options["shape_filled"] = True
    session.options["stroke_width"] = 1
    shape = ShapeTool()
    shape.pointer_down(session, PointerEvent(18, 18))
    shape.pointer_up(session, PointerEvent(2, 2))
    pixel = session.active_layer().image.getpixel((10, 10))
    assert pixel[:3] == (255, 0, 0)
    assert session.active_layer().image.getpixel((0, 0))[:3] == (255, 255, 255)


def test_eyedropper_reads_composite():
    session = Session(new_document(8, 8, "black"))
    tool = EyedropperTool()
    tool.pointer_down(session, PointerEvent(1, 1))
    assert session.foreground[:3] == (0, 0, 0)

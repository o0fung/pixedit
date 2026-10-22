"""Brush, eraser, fill, selection, and eyedropper behavior without Qt."""

from __future__ import annotations

from PIL import Image, ImageDraw

from pixedit.factory import raster_layer
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


def test_connected_mask_uses_per_channel_tolerance_and_fills_transparency():
    """Default tolerance used to no-op on transparent pixels.

    Pillow's flood fill sums channel deltas and also refuses to start when
    the seed is within that distance of the replacement color. A transparent
    pixel is only 10 away from a dark sentinel, so tolerance 32 filled nothing.
    """
    clear = Image.new("RGBA", (6, 6), (0, 0, 0, 0))
    filled = connected_mask(clear, 2, 2, tolerance=32)
    assert filled.getpixel((2, 2)) == 255
    assert filled.getpixel((5, 5)) == 255

    white = Image.new("RGBA", (8, 8), (255, 255, 255, 255))
    # 20 per channel: max is 20 (inside 32) but the channel sum is 60.
    white.putpixel((3, 2), (235, 235, 235, 255))
    white.putpixel((4, 2), (200, 255, 255, 255))  # one channel 55 away from the seed
    mask = connected_mask(white, 1, 1, tolerance=32)
    assert mask.getpixel((1, 1)) == 255
    assert mask.getpixel((3, 2)) == 255
    assert mask.getpixel((4, 2)) == 0
    # A second red block is not connected, even when the color matches.
    white.putpixel((7, 7), (255, 255, 255, 255))
    blocked = Image.new("RGBA", (8, 8), (0, 0, 0, 255))
    blocked.paste(white.crop((0, 0, 5, 4)), (0, 0))
    blocked.putpixel((7, 7), (255, 255, 255, 255))
    separate = connected_mask(blocked, 1, 1, tolerance=0)
    assert separate.getpixel((1, 1)) == 255
    assert separate.getpixel((7, 7)) == 0


def test_fill_tool_click_path_tolerance_selection_and_messages():
    session = Session(new_document(16, 16, "transparent"))
    session.foreground = (20, 30, 40, 255)
    session.options["tolerance"] = 32
    FillTool().pointer_down(session, PointerEvent(4, 5))
    layer = session.active_layer()
    assert layer.image.getpixel((4, 5)) == (20, 30, 40, 255)
    assert layer.image.getpixel((15, 15))[3] == 255
    assert layer.image.size == (16, 16)

    # A small sprite stays small. Fill recolors the seed's component only.
    sprite = raster_layer("Sprite", Image.new("RGBA", (6, 6), (255, 0, 0, 255)), 8, 8)
    sprite.image.putpixel((2, 2), (0, 0, 255, 255))
    session.add_layer(sprite)
    session.layer_index = session.document.index_of(sprite.id)
    session.foreground = (0, 255, 0, 255)
    session.options["tolerance"] = 0
    FillTool().pointer_down(session, PointerEvent(10, 10))
    edited = session.active_layer()
    assert edited.image.size == (6, 6)
    assert (edited.x, edited.y) == (8, 8)
    assert edited.image.getpixel((2, 2))[:3] == (0, 255, 0)
    assert edited.image.getpixel((0, 0))[:3] == (255, 0, 0)

    messages: list[str] = []
    session.set_message_handler(messages.append)
    before = edited.image.getpixel((0, 0))
    steps = len(session.history.undo_stack)
    FillTool().pointer_down(session, PointerEvent(1, 1))
    assert any("outside the active layer" in message for message in messages)
    assert session.active_layer().image.getpixel((0, 0)) == before
    assert len(session.history.undo_stack) == steps

    mask = Image.new("L", (16, 16), 0)
    mask.paste(255, (0, 0, 8, 16))
    session.set_selection(mask)
    messages.clear()
    # The sprite sits at x=8, so a click on its left edge is canvas x=8, outside
    # the selection (0..7). Use a click on the white background inside the layer
    # but the background is not the active layer. Click the sprite at (9, 9),
    # which is outside the selection, and confirm nothing is recolored.
    session.foreground = (255, 255, 0, 255)
    FillTool().pointer_down(session, PointerEvent(9, 9))
    assert any("outside the selection" in message for message in messages)
    assert session.active_layer().image.getpixel((1, 1))[:3] == (255, 0, 0)

    messages.clear()
    session.layer_index = 0
    session.document.layers[0].image = Image.new("RGBA", (16, 16), (255, 255, 255, 255))
    session.foreground = (0, 0, 255, 255)
    session.options["tolerance"] = 0
    FillTool().pointer_down(session, PointerEvent(2, 2))
    base = session.document.layers[0]
    assert base.image.getpixel((2, 2))[:3] == (0, 0, 255)
    assert base.image.getpixel((12, 12))[:3] == (255, 255, 255)
    assert base.image.size == (16, 16)


def test_brush_and_shape_on_a_new_layer_stay_content_sized():
    session = Session(new_document(80, 60, "white"))
    session.add_blank_layer()
    assert session.active_layer().image.size == (1, 1)
    session.foreground = (255, 0, 0, 255)
    session.options["brush_size"] = 6
    session.options["brush_hardness"] = 100
    session.options["brush_opacity"] = 100
    brush = BrushTool()
    brush.pointer_down(session, PointerEvent(40, 30))
    brush.pointer_up(session, PointerEvent(40, 30))
    layer = session.active_layer()
    assert layer.image.width < 24
    assert layer.image.height < 24
    assert layer.image.width > 2
    center = (int(round(40 - layer.x)), int(round(30 - layer.y)))
    assert layer.image.getpixel(center)[0] > 200
    assert session.document.layers[0].image.size == (80, 60)
    session.undo()
    assert session.active_layer().image.size == (1, 1)
    assert session.active_layer().image.getpixel((0, 0))[3] == 0

    # Painting past an existing sprite keeps the old pixels and does not jump
    # to the full canvas.
    sprite = raster_layer("Sprite", Image.new("RGBA", (4, 4), (0, 255, 0, 255)), 5, 5)
    session.add_layer(sprite)
    session.layer_index = session.document.index_of(sprite.id)
    session.options["brush_size"] = 4
    brush.pointer_down(session, PointerEvent(40, 20))
    brush.pointer_up(session, PointerEvent(40, 20))
    grown = session.active_layer()
    assert grown.image.size != (80, 60)
    assert grown.image.width < 80
    green = grown.image.getpixel((5 - grown.x, 5 - grown.y))
    assert green[1] > 200 and green[3] == 255
    red = grown.image.getpixel((int(round(40 - grown.x)), int(round(20 - grown.y))))
    assert red[0] > 200
    session.undo()
    assert session.active_layer().image.size == (4, 4)
    assert (session.active_layer().x, session.active_layer().y) == (5, 5)

    session.layer_index = 0
    brush.pointer_down(session, PointerEvent(12, 12))
    brush.pointer_up(session, PointerEvent(12, 12))
    assert session.document.layers[0].image.size == (80, 60)
    assert session.document.layers[0].x == 0

    session.add_blank_layer()
    session.foreground = (0, 0, 255, 255)
    session.options["shape_kind"] = "rect"
    session.options["shape_filled"] = True
    session.options["stroke_width"] = 2
    shape = ShapeTool()
    shape.pointer_down(session, PointerEvent(20, 15))
    shape.pointer_up(session, PointerEvent(36, 28))
    shaped = session.active_layer()
    assert shaped.image.width < 40
    assert shaped.image.height < 40
    pixel = shaped.image.getpixel((28 - shaped.x, 21 - shaped.y))
    assert pixel[2] > 200
    assert session.document.layers[0].image.size == (80, 60)


def test_clear_selection_contents_and_undo():
    session = Session(new_document(20, 20, "white"))
    messages: list[str] = []
    session.set_message_handler(messages.append)
    session.clear_selection_contents()
    assert any("No selection" in message for message in messages)

    session.options["select_mode"] = "rect"
    messages.clear()
    tool = SelectTool()
    tool.pointer_down(session, PointerEvent(4, 4))
    tool.pointer_up(session, PointerEvent(10, 12))
    assert any("Selection clips" in message for message in messages)
    session.clear_selection_contents()
    layer = session.active_layer()
    assert layer.image.size == (20, 20)
    assert layer.image.getpixel((6, 6))[3] == 0
    assert layer.image.getpixel((2, 2))[3] == 255
    session.undo()
    assert layer.image.getpixel((6, 6))[3] == 255

    sprite = raster_layer("Sprite", Image.new("RGBA", (8, 8), (255, 0, 0, 255)), 2, 2)
    session.add_layer(sprite)
    session.layer_index = session.document.index_of(sprite.id)
    session.clear_selection_contents()
    cleared = session.active_layer()

    def _alpha_at(layer, x: int, y: int) -> int:
        local_x, local_y = x - layer.x, y - layer.y
        if not (0 <= local_x < layer.image.width and 0 <= local_y < layer.image.height):
            return 0
        return int(layer.image.getpixel((local_x, local_y))[3])

    # Canvas (6, 6) is inside both the sprite and the selection. (2, 2) is the
    # sprite's corner and sits outside the selection, so it stays opaque.
    assert _alpha_at(cleared, 6, 6) == 0
    assert _alpha_at(cleared, 2, 2) == 255
    session.undo()
    assert session.active_layer().image.size == (8, 8)
    assert session.active_layer().image.getpixel((4, 4))[3] == 255

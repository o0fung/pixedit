"""Brush, eraser, fill, selection, and color sampling without Qt."""

from __future__ import annotations

from PIL import Image, ImageDraw

from pixedit.factory import raster_layer, redaction_layer
from pixedit.model import new_document
from pixedit.paint import connected_mask, dab, fill_region, interpolate, paint_points, stroke_spacing
from pixedit.session import Session
from pixedit.tools.base import PointerEvent
from pixedit.tools.builtins import BrushTool, ColorTool, EraserTool, FillTool, SelectTool, ShapeTool
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


def test_color_tool_reads_composite():
    session = Session(new_document(8, 8, "black"))
    tool = ColorTool()
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
    messages: list[str] = []
    session.set_message_handler(messages.append)
    steps = len(session.history.undo_stack)
    FillTool().pointer_down(session, PointerEvent(4, 5))
    layer = session.active_layer()
    # Empty pixels run to the layer edge, so the canvas stays transparent.
    assert layer.image.getpixel((4, 5))[3] == 0
    assert layer.image.getpixel((15, 15))[3] == 0
    assert layer.image.size == (16, 16)
    assert any("empty pixels" in message for message in messages)
    assert len(session.history.undo_stack) == steps

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


def _outline_on_transparent_layer() -> Session:
    """A 32×32 closed rectangle on a larger white canvas, placed at (10, 10)."""
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((8, 8, 23, 23), outline=(0, 0, 0, 255), width=2)
    session = Session(new_document(64, 64, "white"))
    outline = raster_layer("Outline", image, 10, 10)
    session.add_layer(outline)
    session.layer_index = session.document.index_of(outline.id)
    session.foreground = (0, 180, 40, 255)
    session.options["tolerance"] = 0
    return session


def test_fill_paints_an_enclosed_hole_and_skips_open_empty_pixels():
    session = _outline_on_transparent_layer()
    FillTool().pointer_down(session, PointerEvent(26, 26))
    edited = session.active_layer()
    # The hole is painted. Transparent margins are cropped; the layer does not
    # grow out to the canvas.
    assert edited.image.size != (64, 64)
    assert edited.image.width < 64 and edited.image.height < 64
    filled = edited.image.getpixel((26 - edited.x, 26 - edited.y))
    assert filled[:3] == (0, 180, 40)
    assert session.document.layers[0].image.getpixel((0, 0))[:3] == (255, 255, 255)

    session = _outline_on_transparent_layer()
    messages: list[str] = []
    session.set_message_handler(messages.append)
    steps = len(session.history.undo_stack)
    FillTool().pointer_down(session, PointerEvent(11, 11))
    assert any("empty pixels" in message for message in messages)
    layer = session.active_layer()
    assert layer.image.size == (32, 32)
    assert (layer.x, layer.y) == (10, 10)
    assert layer.image.getpixel((1, 1))[3] == 0
    assert layer.image.getpixel((0, 0))[3] == 0
    assert len(session.history.undo_stack) == steps
    assert session.document.layers[0].image.getpixel((0, 0))[:3] == (255, 255, 255)


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


def _pixel(layer, x: int, y: int):
    local_x, local_y = x - layer.x, y - layer.y
    if not (0 <= local_x < layer.image.width and 0 <= local_y < layer.image.height):
        return (0, 0, 0, 0)
    return layer.image.getpixel((local_x, local_y))


def test_fast_stroke_stays_continuous_for_brush_and_eraser():
    assert stroke_spacing(12, 0.85) <= 1.5
    assert len(interpolate((0, 0), (30, 0), 0.4)) >= 60

    session = Session(new_document(80, 80, "transparent"))
    session.foreground = (255, 0, 0, 255)
    session.options["brush_size"] = 5
    session.options["brush_hardness"] = 100
    session.options["brush_opacity"] = 100
    brush = BrushTool()
    brush.pointer_down(session, PointerEvent(5, 5))
    brush.pointer_up(session, PointerEvent(70, 60))
    layer = session.active_layer()
    for step in range(41):
        x = 5 + (70 - 5) * step / 40
        y = 5 + (60 - 5) * step / 40
        assert _pixel(layer, int(round(x)), int(round(y)))[3] > 200

    session.foreground = (0, 180, 0, 255)
    session.options["brush_size"] = 8
    brush.pointer_down(session, PointerEvent(10, 40))
    brush.pointer_up(session, PointerEvent(10, 40))
    assert _pixel(session.active_layer(), 10, 40)[1] > 150
    eraser = EraserTool()
    eraser.pointer_down(session, PointerEvent(4, 40))
    eraser.pointer_up(session, PointerEvent(70, 40))
    cleared = session.active_layer()
    for x in range(8, 66, 4):
        assert _pixel(cleared, x, 40)[:4] == (0, 0, 0, 0)


def test_fill_meets_a_soft_brush_outline_without_a_rim_or_a_leak():
    def _closed_circle(background: str, hardness: int) -> np.ndarray:
        session = Session(new_document(90, 90, background))
        session.foreground = (20, 20, 20, 255)
        session.options["brush_size"] = 18
        session.options["brush_hardness"] = hardness
        session.options["brush_opacity"] = 100
        tool = BrushTool()
        cx, cy, radius = 45.0, 45.0, 22.0
        tool.pointer_down(session, PointerEvent(cx + radius, cy))
        for index in range(1, 81):
            angle = 2 * np.pi * index / 80
            tool.pointer_move(session, PointerEvent(cx + radius * np.cos(angle), cy + radius * np.sin(angle)))
        tool.pointer_up(session, PointerEvent(cx + radius, cy))
        session.foreground = (220, 40, 50, 255)
        session.options["tolerance"] = 32
        FillTool().pointer_down(session, PointerEvent(45, 45))
        return np.array(session.active_layer().image)

    def _counts(arr: np.ndarray) -> tuple[int, int, int]:
        red, green, blue, alpha = (arr[..., i] for i in range(4))
        fill = (np.abs(red.astype(int) - 220) < 12) & (np.abs(green.astype(int) - 40) < 12) & (alpha > 180)
        dark = (red < 40) & (green < 40) & (blue < 40) & (alpha > 200)
        adjacent = np.zeros(fill.shape, dtype=bool)
        adjacent[1:] |= fill[:-1]
        adjacent[:-1] |= fill[1:]
        adjacent[:, 1:] |= fill[:, :-1]
        adjacent[:, :-1] |= fill[:, 1:]
        # A leftover rim is a dim or gray pixel sitting on the fill's border.
        spread = arr[..., :3].max(axis=-1).astype(int) - arr[..., :3].min(axis=-1).astype(int)
        gray = (spread < 18) & (alpha > 180) & (arr[..., :3].max(axis=-1) > 50) & (arr[..., :3].max(axis=-1) < 210)
        halo = adjacent & ~fill & ((alpha < 200) | gray)
        return int(fill.sum()), int(dark.sum()), int(halo.sum())

    fill_count, dark_count, halo = _counts(_closed_circle("white", 85))
    assert fill_count > 200
    assert dark_count > 50
    assert halo == 0
    fill_count, dark_count, halo = _counts(_closed_circle("transparent", 85))
    assert fill_count > 200
    assert dark_count > 50
    assert halo == 0
    # A hard outline has no partial edge. The stroke body stays dark.
    _fill_count, dark_count, halo = _counts(_closed_circle("white", 100))
    assert dark_count > 200
    assert halo == 0

    flat = Image.new("RGBA", (40, 40), (255, 255, 255, 255))
    for x in range(20, 40):
        for y in range(40):
            flat.putpixel((x, y), (200, 200, 200, 255))
    painted = np.array(fill_region(flat, 5, 5, 32, (220, 40, 50, 255)))
    assert tuple(painted[5, 5]) == (220, 40, 50, 255)
    assert tuple(painted[0, 0]) == (220, 40, 50, 255)
    assert tuple(painted[5, 25]) == (200, 200, 200, 255)
    assert tuple(painted[0, 39]) == (200, 200, 200, 255)


def test_eraser_clears_color_and_respects_a_selection():
    session = Session(new_document(30, 30, "white"))
    session.foreground = (220, 30, 40, 255)
    session.options["brush_size"] = 8
    session.options["brush_hardness"] = 100
    session.options["brush_opacity"] = 100
    brush = BrushTool()
    brush.pointer_down(session, PointerEvent(15, 15))
    brush.pointer_up(session, PointerEvent(15, 15))
    assert session.active_layer().image.getpixel((15, 15))[:3] == (220, 30, 40)
    eraser = EraserTool()
    eraser.pointer_down(session, PointerEvent(15, 15))
    eraser.pointer_up(session, PointerEvent(15, 15))
    assert session.active_layer().image.getpixel((15, 15)) == (0, 0, 0, 0)
    assert session.active_layer().image.getpixel((1, 1)) == (255, 255, 255, 255)

    session.undo()
    mask = Image.new("L", (30, 30), 0)
    mask.paste(255, (0, 0, 15, 30))
    session.set_selection(mask)
    eraser.pointer_down(session, PointerEvent(22, 15))
    eraser.pointer_up(session, PointerEvent(22, 15))
    assert session.active_layer().image.getpixel((22, 15))[0] > 200
    eraser.pointer_down(session, PointerEvent(8, 15))
    eraser.pointer_up(session, PointerEvent(8, 15))
    assert session.active_layer().image.getpixel((8, 15)) == (0, 0, 0, 0)


def test_shape_preview_matches_the_committed_shape_including_alpha():
    session = Session(new_document(40, 40, "transparent"))
    session.foreground = (255, 0, 0, 128)
    session.options["shape_kind"] = "rect"
    session.options["shape_filled"] = True
    session.options["stroke_width"] = 2
    shape = ShapeTool()
    shape.pointer_down(session, PointerEvent(8, 8))
    shape.pointer_move(session, PointerEvent(30, 28))
    preview = session.active_layer().image.getpixel((19, 18))
    assert abs(preview[3] - 128) <= 8
    assert preview[0] > 200
    steps = len(session.history.undo_stack)
    shape.pointer_up(session, PointerEvent(30, 28))
    assert session.active_layer().image.getpixel((19, 18)) == preview
    assert len(session.history.undo_stack) == steps + 1
    session.undo()
    assert session.active_layer().image.getpixel((19, 18))[3] == 0

    session.options["shape_filled"] = False
    session.options["stroke_width"] = 2
    session.foreground = (0, 0, 255, 255)
    shape.pointer_down(session, PointerEvent(6, 6))
    shape.pointer_move(session, PointerEvent(34, 34))
    assert session.active_layer().image.getpixel((20, 20))[3] == 0
    # Pillow's stroke sits on the rectangle edge. Width 2 lands just outside the drag box.
    assert session.active_layer().image.getpixel((3, 20))[2] > 200
    assert shape.pointer_cancel(session) is True
    assert session.active_layer().image.getpixel((3, 20))[3] == 0
    assert len(session.history.undo_stack) == steps


def test_cut_and_copy_selection_to_a_new_layer():
    session = Session(new_document(30, 30, "white"))
    messages: list[str] = []
    session.set_message_handler(messages.append)
    session.copy_selection_to_layer()
    assert any("Draw a selection" in message for message in messages)
    assert len(session.document.layers) == 1

    image = session.active_layer().image.copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle((6, 6, 14, 14), fill=(255, 0, 0, 255))
    session.active_layer().image = image
    mask = Image.new("L", (30, 30), 0)
    mask.paste(255, (6, 6, 12, 12))
    session.set_selection(mask)
    session.copy_selection_to_layer()
    assert len(session.document.layers) == 2
    copied = session.active_layer()
    assert copied.name.endswith("copy")
    assert _pixel(copied, 8, 8)[:3] == (255, 0, 0)
    assert _pixel(session.document.layers[0], 8, 8)[:3] == (255, 0, 0)
    assert session.document.selection is not None
    session.undo()
    assert len(session.document.layers) == 1

    session.cut_selection_to_layer()
    assert len(session.document.layers) == 2
    source = session.document.layers[0]
    moved = session.document.layers[1]
    assert _pixel(source, 8, 8) == (0, 0, 0, 0)
    assert _pixel(moved, 8, 8)[:3] == (255, 0, 0)
    assert _pixel(source, 13, 13)[:3] == (255, 0, 0)
    session.undo()
    assert len(session.document.layers) == 1
    assert session.document.layers[0].image.getpixel((8, 8))[:3] == (255, 0, 0)

    redacted = redaction_layer("Hide", Image.new("L", (30, 30), 0), "blur", 8)
    session.add_layer(redacted)
    session.layer_index = session.document.index_of(redacted.id)
    messages.clear()
    before = len(session.history.undo_stack)
    session.cut_selection_to_layer()
    assert any("mask" in message for message in messages)
    assert len(session.history.undo_stack) == before


def test_place_layers_keeps_the_document_and_undoes():
    session = Session(new_document(40, 30, "white"))
    green = raster_layer("Green", Image.new("RGBA", (10, 8), (0, 255, 0, 255)))
    padded = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    padded.paste(Image.new("RGBA", (4, 4), (0, 0, 255, 255)), (8, 8))
    blue = raster_layer("Blue", padded)
    blue.visible = False
    session.place_layers([green, blue])
    assert session.document.width == 40
    assert session.document.height == 30
    assert len(session.document.layers) == 3
    placed = session.document.layers[1]
    assert placed.image.size == (10, 8)
    assert (placed.x, placed.y) == (15, 11)
    assert placed.visible
    hidden = session.document.layers[2]
    assert hidden.image.size == (4, 4)
    assert not hidden.visible
    assert session.layer_index == 1
    session.undo()
    assert len(session.document.layers) == 1
    assert session.document.layers[0].image.size == (40, 30)

    messages: list[str] = []
    session.set_message_handler(messages.append)
    session.place_layers([redaction_layer("Hide", Image.new("L", (8, 8), 255), "pixelate", 8)])
    assert any("no image" in message for message in messages)
    assert len(session.document.layers) == 1


def test_crop_to_selection_uses_the_opaque_bounds():
    session = Session(new_document(20, 20, "white"))
    messages: list[str] = []
    session.set_message_handler(messages.append)
    session.crop_to_selection()
    assert any("Draw a selection" in message for message in messages)
    assert session.document.width == 20

    empty = Image.new("L", (20, 20), 0)
    session.set_selection(empty)
    messages.clear()
    session.crop_to_selection()
    assert any("empty" in message for message in messages)
    assert session.document.width == 20

    mask = Image.new("L", (20, 20), 0)
    mask.paste(255, (2, 3, 10, 12))
    session.set_selection(mask)
    session.crop_to_selection()
    assert session.document.width == 8
    assert session.document.height == 9


def _red_width(image: Image.Image) -> int:
    import numpy as np

    arr = np.array(image)
    cols = np.where((arr[..., 0] > 200) & (arr[..., 3] > 200))[0]
    if len(cols) == 0:
        return 0
    return int(cols.max() - cols.min())


def test_brush_preview_shows_size_hardness_opacity_and_eraser_hole():
    from pixedit.paint import brush_preview_image

    color = (255, 0, 0, 255)
    small = brush_preview_image(8, 1.0, 1.0, color, erase=False)
    large = brush_preview_image(80, 1.0, 1.0, color, erase=False)
    huge = brush_preview_image(400, 1.0, 1.0, color, erase=False)
    assert _red_width(large) > _red_width(small) + 20
    assert abs(_red_width(huge) - _red_width(large)) <= 2

    hard = np.array(brush_preview_image(48, 1.0, 1.0, color, erase=False))
    soft = np.array(brush_preview_image(48, 0.0, 1.0, color, erase=False))
    faint = np.array(brush_preview_image(48, 1.0, 0.35, color, erase=False))
    assert hard[48, 88, 0] > 240
    assert hard[48, 108, 0] > 200
    assert soft[48, 108, 0] < 120
    assert faint[48, 88, 0] < hard[48, 88, 0] - 40

    hole = np.array(brush_preview_image(40, 1.0, 1.0, color, erase=True))
    assert hole[2, 2, 3] == 255
    assert hole[48, 88, 3] == 0
    soft_hole = np.array(brush_preview_image(48, 0.0, 1.0, color, erase=True))
    assert 0 < int(soft_hole[48, 108, 3]) < 255

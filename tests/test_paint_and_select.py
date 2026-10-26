"""Brush, eraser, fill, selection, and color sampling without Qt."""

from __future__ import annotations

from PIL import Image, ImageDraw

from pixedit.factory import raster_layer, redaction_layer
from pixedit.model import new_document
from pixedit.paint import connected_mask, dab, fill_region, interpolate, paint_points, stroke_spacing, wand_mask
from pixedit.session import Session
from pixedit.tools.base import PointerEvent
from pixedit.tools.brush import BrushTool
from pixedit.tools.color import ColorTool
from pixedit.tools.eraser import EraserTool
from pixedit.tools.fill import FillTool
from pixedit.tools.select import SelectTool
from pixedit.tools.shape import ShapeTool
import numpy as np


def _stroke_alpha(hardness: int, passes: int) -> int:
    """Alpha at the middle of a horizontal stroke, after ``passes`` separate gestures.

    The layer is cropped to the paint, so the sample is in canvas coordinates.
    """
    session = Session(new_document(96, 64, "transparent"))
    session.foreground = (255, 0, 0, 255)
    session.options["brush_size"] = 24
    session.options["brush_hardness"] = hardness
    session.options["brush_opacity"] = 100
    tool = BrushTool()
    for _ in range(passes):
        tool.pointer_down(session, PointerEvent(12, 32))
        tool.pointer_up(session, PointerEvent(84, 32))
    return _pixel(session.active_layer(), 48, 32)[3]


def test_soft_brush_builds_up_instead_of_filling_solid():
    """Hardness 0 lays down a light coat. Many passes reach solid; 85 and 100 do it in one.

    One coat is about 25/255. Twenty-five passes are already near solid, and
    the last few points of 255 take a couple more because each pass only
    fills what is still missing.
    """
    once = _stroke_alpha(0, 1)
    assert 15 <= once <= 50, once
    assert _stroke_alpha(0, 25) > 220
    assert _stroke_alpha(0, 28) > 240
    assert _stroke_alpha(100, 1) > 240
    assert _stroke_alpha(85, 1) > 240


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


def _soft_circle(size: int = 80, radius: float = 22.0) -> Image.Image:
    """Red disk on white with a one-pixel blend at the edge.

    The center is a pixel, so the blend lands on whole pixels instead of a
    sub-pixel sliver that is already the background.
    """
    arr = np.zeros((size, size, 4), np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    center = float(size // 2)
    dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    cover = np.clip(radius - dist, 0, 1)
    red = np.array([220, 30, 40], np.float32)
    white = np.array([255, 255, 255], np.float32)
    arr[..., :3] = np.rint(red * cover[..., None] + white * (1 - cover[..., None])).astype(np.uint8)
    arr[..., 3] = 255
    return Image.fromarray(arr, "RGBA")


def _colored_opaque(arr: np.ndarray, background: tuple[int, int, int, int]) -> np.ndarray:
    delta = np.max(np.abs(arr.astype(np.int16) - np.array(background, dtype=np.int16)), axis=-1)
    return (delta > 0) & (arr[..., 3] > 0)


def _wand_and_cut(image: Image.Image, x: int, y: int, background: str) -> Session:
    session = Session(new_document(image.width, image.height, background))
    session.document.layers[0].image = image
    session.options["select_mode"] = "wand"
    session.options["tolerance"] = 32
    SelectTool().pointer_down(session, PointerEvent(x, y))
    session.cut_selection_to_layer()
    return session


def test_wand_cut_takes_the_soft_rim():
    """A blended edge used to stay on the source after cut to layer."""
    circle = _soft_circle()
    flood = np.asarray(connected_mask(circle, 40, 40, 32)) > 0
    wand = np.asarray(wand_mask(circle, 40, 40, 32)) > 0
    shape = _colored_opaque(np.asarray(circle), (255, 255, 255, 255))
    assert int((shape & ~flood).sum()) > 0
    assert int((shape & ~wand).sum()) == 0

    session = _wand_and_cut(circle, 40, 40, "white")
    source = np.asarray(session.document.layers[0].image)
    assert int(_colored_opaque(source, (255, 255, 255, 255)).sum()) == 0
    moved = np.asarray(session.document.layers[1].image)
    seed = np.array([220, 30, 40, 255], dtype=np.int16)
    blend = (moved[..., 3] > 0) & (np.max(np.abs(moved.astype(np.int16) - seed), axis=-1) > 32)
    assert int(blend.sum()) > 0

    dabbed = np.full((80, 80, 4), 255, np.uint8)
    dab(dabbed, 40, 40, radius=15, hardness=0.85, color=(220, 30, 40, 255), opacity=1, erase=False, selection=None)
    session = _wand_and_cut(Image.fromarray(dabbed, "RGBA"), 40, 40, "white")
    source = np.asarray(session.document.layers[0].image)
    assert int(_colored_opaque(source, (255, 255, 255, 255)).sum()) == 0
    moved = np.asarray(session.document.layers[1].image)
    blend = (moved[..., 3] > 0) & (np.max(np.abs(moved.astype(np.int16) - seed), axis=-1) > 32)
    assert int(blend.sum()) > 0

    clear = np.zeros((80, 80, 4), np.uint8)
    dab(clear, 40, 40, radius=15, hardness=0.85, color=(220, 30, 40, 255), opacity=1, erase=False, selection=None)
    session = _wand_and_cut(Image.fromarray(clear, "RGBA"), 40, 40, "transparent")
    source = np.asarray(session.document.layers[0].image)
    assert int((source[..., 3] > 0).sum()) == 0
    moved = np.asarray(session.document.layers[1].image)
    faded = (moved[..., 3] > 0) & (moved[..., 3] < 250)
    assert int(faded.sum()) > 0


def test_wand_rim_on_a_wide_image():
    """A seed to the right of the image height still takes the soft rim.

    Pillow's size is (width, height). Reading that backwards used to reject
    this click and return the flood alone, so the blend stayed unselected.
    """
    width, height = 120, 40
    arr = np.full((height, width, 4), 255, np.uint8)
    yy, xx = np.mgrid[0:height, 0:width]
    # Integer center, same as the square disk, so the blend sits on whole pixels.
    cx, cy, radius = 80, 20, 12.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    cover = np.clip(radius - dist, 0, 1)
    red = np.array([220, 30, 40], np.float32)
    white = np.array([255, 255, 255], np.float32)
    arr[..., :3] = np.rint(red * cover[..., None] + white * (1 - cover[..., None])).astype(np.uint8)
    image = Image.fromarray(arr, "RGBA")
    assert cx > height
    flood = np.asarray(connected_mask(image, cx, cy, 32)) > 0
    wand = np.asarray(wand_mask(image, cx, cy, 32)) > 0
    shape = _colored_opaque(arr, (255, 255, 255, 255))
    assert int((shape & ~flood).sum()) > 0
    assert int((shape & ~wand).sum()) == 0

    session = Session(new_document(width, height, "white"))
    session.document.layers[0].image = image
    session.options["select_mode"] = "wand"
    session.options["tolerance"] = 32
    SelectTool().pointer_down(session, PointerEvent(cx, cy))
    assert np.array_equal(np.asarray(session.document.selection) > 0, wand)


def test_wand_hard_edge_matches_the_flood():
    image = Image.new("RGBA", (80, 80), (255, 255, 255, 255))
    ImageDraw.Draw(image).ellipse((15, 15, 65, 65), fill=(220, 30, 40, 255))
    flood = np.asarray(connected_mask(image, 40, 40, 32))
    assert np.array_equal(np.asarray(wand_mask(image, 40, 40, 32)), flood)

    session = Session(new_document(80, 80, "white"))
    session.document.layers[0].image = image
    session.options["select_mode"] = "wand"
    session.options["tolerance"] = 32
    SelectTool().pointer_down(session, PointerEvent(40, 40))
    assert np.array_equal(np.asarray(session.document.selection), flood)
    session.cut_selection_to_layer()
    source = np.asarray(session.document.layers[0].image)
    assert int(_colored_opaque(source, (255, 255, 255, 255)).sum()) == 0


def test_wand_does_not_take_a_flat_neighbor_or_a_slow_gradient():
    flat = np.full((40, 40, 4), (200, 200, 200, 255), np.uint8)
    flat[:, :20] = (220, 30, 40, 255)
    image = Image.fromarray(flat, "RGBA")
    mask = wand_mask(image, 5, 5, 32)
    assert mask.getpixel((19, 5)) == 255
    assert mask.getpixel((20, 5)) == 0
    assert mask.getpixel((30, 5)) == 0
    other = wand_mask(image, 30, 5, 32)
    assert other.getpixel((20, 5)) == 255
    assert other.getpixel((19, 5)) == 0

    gradient = np.zeros((80, 80, 4), np.uint8)
    for x in range(80):
        gradient[:, x, :3] = int(x * 255 / 79)
        gradient[:, x, 3] = 255
    ramp = Image.fromarray(gradient, "RGBA")
    assert np.array_equal(
        np.asarray(wand_mask(ramp, 10, 40, 32)),
        np.asarray(connected_mask(ramp, 10, 40, 32)),
    )


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


def test_freehand_select_keeps_sampled_bends():
    """Points from the cursor poll fill a jog that the straight chord misses."""
    from pixedit.geom import mask_freehand

    session = Session(new_document(80, 80, "white"))
    tool = SelectTool()
    for mode in ("rect", "ellipse", "wand"):
        session.options["select_mode"] = mode
        assert not tool.wants_stroke_sampling(session)

    session.options["select_mode"] = "rect"
    tool.pointer_down(session, PointerEvent(1, 1))
    tool.pointer_up(session, PointerEvent(8, 8))
    assert not tool.stroke_live()
    selection = session.document.selection
    assert selection.getpixel((4, 4)) == 255

    # Right triangle. (48, 38) sits in the corner, off the hypotenuse.
    start = (10.0, 40.0)
    bend = (50.0, 40.0)
    end = (50.0, 10.0)
    inside = (48, 38)
    session.options["select_mode"] = "freehand"
    assert tool.wants_stroke_sampling(session)
    notes: list[str] = []
    session.add_listener(notes.append)
    tool.pointer_down(session, PointerEvent(*start))
    assert tool.stroke_live()
    tool.pointer_move(session, PointerEvent(*bend))
    # A sample 0.02px from the bend is the same point. The poll drops it.
    # The new point only redraws the rubber band, so the photo is not rebuilt.
    tool.extend_points(session, [(bend[0] + 0.02, bend[1]), end])
    assert tool._points == [start, bend, end]
    # pointer_move and the batch each redraw the line. Neither rebuilds the photo.
    assert notes == ["overlay", "overlay"]
    tool.commit_stroke(session, PointerEvent(*end, shift=True))
    assert not tool.stroke_live()
    selection = session.document.selection
    assert selection.getpixel(inside) == 255
    assert selection.getpixel((4, 4)) == 255
    chord = mask_freehand(session.document.width, session.document.height, [start, end], radius=1)
    assert chord.getpixel(inside) == 0

    # A later rectangle drag still replaces the selection through pointer_up.
    session.options["select_mode"] = "rect"
    assert not tool.wants_stroke_sampling(session)
    tool.pointer_down(session, PointerEvent(2, 2))
    assert not tool.stroke_live()
    tool.pointer_move(session, PointerEvent(12, 14))
    tool.pointer_up(session, PointerEvent(12, 14))
    selection = session.document.selection
    assert selection.getpixel((6, 6)) == 255
    assert selection.getpixel(inside) == 0


def test_rect_and_ellipse_drags_redraw_the_outline_only():
    """A selection drag is chrome. The photo is flattened when the selection is stored."""
    session = Session(new_document(30, 30, "white"))
    notes: list[str] = []
    session.add_listener(notes.append)
    tool = SelectTool()
    session.options["select_mode"] = "rect"
    tool.pointer_down(session, PointerEvent(2, 2))
    tool.pointer_move(session, PointerEvent(14, 12))
    tool.pointer_cancel(session)
    assert notes == ["overlay", "overlay"]
    assert session.document.selection is None

    session.options["select_mode"] = "ellipse"
    tool.pointer_down(session, PointerEvent(2, 2))
    tool.pointer_move(session, PointerEvent(16, 14))
    assert notes[-1] == "overlay"
    assert "preview" not in notes
    tool.pointer_up(session, PointerEvent(16, 14))
    selection = session.document.selection
    assert selection is not None
    assert selection.getpixel((8, 8)) == 255
    assert "change" in notes


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
    # The empty canvas is one connected transparent region, so the click paints it.
    assert layer.image.getpixel((4, 5)) == (20, 30, 40, 255)
    assert layer.image.getpixel((15, 15)) == (20, 30, 40, 255)
    assert layer.image.size == (16, 16)
    assert (layer.x, layer.y) == (0, 0)
    assert messages == []
    assert len(session.history.undo_stack) == steps + 1

    # A click on one color recolors that component only. The layer stays tight.
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

    # A click in the empty margin fills that region and leaves the sprite colors.
    session.foreground = (0, 0, 255, 255)
    FillTool().pointer_down(session, PointerEvent(1, 1))
    edited = session.active_layer()
    assert edited.image.size == (16, 16)
    assert (edited.x, edited.y) == (0, 0)
    assert edited.image.getpixel((1, 1))[:3] == (0, 0, 255)
    assert edited.image.getpixel((8, 8))[:3] == (255, 0, 0)
    assert edited.image.getpixel((10, 10))[:3] == (0, 255, 0)
    assert session.document.layers[0].image.getpixel((1, 1))[:3] == (20, 30, 40)

    messages.clear()
    steps = len(session.history.undo_stack)
    before = edited.image.getpixel((1, 1))
    FillTool().pointer_down(session, PointerEvent(-1, 1))
    assert any("outside the canvas" in message for message in messages)
    assert session.active_layer().image.getpixel((1, 1)) == before
    assert len(session.history.undo_stack) == steps

    mask = Image.new("L", (16, 16), 0)
    mask.paste(255, (0, 0, 8, 16))
    session.set_selection(mask)
    messages.clear()
    # Canvas (9, 9) is the sprite's red pixel and sits outside the selection.
    session.foreground = (255, 255, 0, 255)
    steps = len(session.history.undo_stack)
    FillTool().pointer_down(session, PointerEvent(9, 9))
    assert any("outside the selection" in message for message in messages)
    assert session.active_layer().image.getpixel((9, 9))[:3] == (255, 0, 0)
    assert len(session.history.undo_stack) == steps

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


def test_fill_on_an_empty_layer_covers_the_canvas():
    session = Session(new_document(20, 12, "white"))
    session.add_blank_layer()
    assert session.active_layer().image.size == (1, 1)
    session.foreground = (9, 8, 7, 255)
    session.options["tolerance"] = 0
    FillTool().pointer_down(session, PointerEvent(15, 6))
    layer = session.active_layer()
    assert layer.image.size == (20, 12)
    assert (layer.x, layer.y) == (0, 0)
    assert layer.image.getpixel((15, 6)) == (9, 8, 7, 255)
    assert layer.image.getpixel((0, 0)) == (9, 8, 7, 255)
    assert session.document.layers[0].image.getpixel((0, 0))[:3] == (255, 255, 255)


def test_fill_outside_a_small_layer_stays_inside_the_selection():
    session = Session(new_document(16, 16, "transparent"))
    sprite = raster_layer("Sprite", Image.new("RGBA", (6, 6), (255, 0, 0, 255)), 8, 8)
    session.add_layer(sprite)
    session.layer_index = session.document.index_of(sprite.id)
    mask = Image.new("L", (16, 16), 0)
    mask.paste(255, (0, 0, 8, 16))
    session.set_selection(mask)
    session.foreground = (0, 0, 255, 255)
    session.options["tolerance"] = 0
    FillTool().pointer_down(session, PointerEvent(1, 1))
    edited = session.active_layer()
    assert edited.image.getpixel((1 - edited.x, 1 - edited.y))[:3] == (0, 0, 255)
    assert edited.image.getpixel((8 - edited.x, 8 - edited.y))[:3] == (255, 0, 0)
    # Canvas (12, 1) is outside the selection and outside the sprite.
    assert edited.image.getpixel((12 - edited.x, 1 - edited.y))[3] == 0


def test_fill_keeps_pixels_that_hang_off_the_canvas():
    session = Session(new_document(16, 16, "white"))
    layer = raster_layer("Overhang", Image.new("RGBA", (8, 8), (255, 0, 0, 255)), -4, -4)
    session.add_layer(layer)
    session.layer_index = session.document.index_of(layer.id)
    session.foreground = (0, 0, 255, 255)
    session.options["tolerance"] = 0
    FillTool().pointer_down(session, PointerEvent(10, 10))
    edited = session.active_layer()
    # Local (0, 0) is canvas (-4, -4), outside the document, and stays red.
    assert (edited.x, edited.y) == (-4, -4)
    assert edited.image.getpixel((0, 0))[:3] == (255, 0, 0)
    # Local (4, 4) is canvas (0, 0), the red that sits on the canvas.
    assert edited.image.getpixel((4, 4))[:3] == (255, 0, 0)
    # Local (14, 14) is canvas (10, 10), empty space that takes the fill.
    assert edited.image.getpixel((14, 14))[:3] == (0, 0, 255)
    assert session.document.layers[0].image.getpixel((10, 10))[:3] == (255, 255, 255)


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


def test_fill_paints_an_enclosed_hole_and_the_open_margin():
    session = _outline_on_transparent_layer()
    FillTool().pointer_down(session, PointerEvent(26, 26))
    edited = session.active_layer()
    # The hole is painted. The open margin stays empty, so the layer stays
    # smaller than the canvas.
    assert edited.image.size != (64, 64)
    assert edited.image.width < 64 and edited.image.height < 64
    filled = edited.image.getpixel((26 - edited.x, 26 - edited.y))
    assert filled[:3] == (0, 180, 40)
    assert session.document.layers[0].image.getpixel((0, 0))[:3] == (255, 255, 255)

    session = _outline_on_transparent_layer()
    FillTool().pointer_down(session, PointerEvent(11, 11))
    layer = session.active_layer()
    # The open margin reaches the canvas edge, so the layer grows to the canvas.
    # The outline stops the flood, and the hole inside it stays empty.
    assert layer.image.size == (64, 64)
    assert (layer.x, layer.y) == (0, 0)
    assert layer.image.getpixel((11, 11))[:3] == (0, 180, 40)
    assert layer.image.getpixel((0, 0))[:3] == (0, 180, 40)
    assert layer.image.getpixel((26, 26))[3] == 0
    assert layer.image.getpixel((18, 18))[:3] == (0, 0, 0)
    assert session.document.layers[0].image.getpixel((0, 0))[:3] == (255, 255, 255)


def test_wand_selects_empty_space_outside_a_small_layer():
    session = Session(new_document(16, 16, "transparent"))
    sprite = raster_layer("Sprite", Image.new("RGBA", (6, 6), (255, 0, 0, 255)), 8, 8)
    sprite.image.putpixel((2, 2), (0, 0, 255, 255))
    session.add_layer(sprite)
    session.layer_index = session.document.index_of(sprite.id)
    session.options["select_mode"] = "wand"
    session.options["tolerance"] = 0

    SelectTool().pointer_down(session, PointerEvent(1, 1))
    selection = session.document.selection
    assert selection is not None
    assert selection.size == (16, 16)
    # The empty margin is one connected region. The sprite colors are not.
    assert selection.getpixel((1, 1)) == 255
    assert selection.getpixel((15, 15)) == 255
    assert selection.getpixel((8, 8)) == 0
    assert selection.getpixel((10, 10)) == 0

    SelectTool().pointer_down(session, PointerEvent(8, 8))
    selection = session.document.selection
    assert selection.getpixel((8, 8)) == 255
    assert selection.getpixel((13, 13)) == 255
    # The blue pixel is a different color, and the far margin is outside the sprite.
    assert selection.getpixel((10, 10)) == 0
    assert selection.getpixel((1, 1)) == 0


def test_wand_enclosed_hole_and_open_margin():
    session = _outline_on_transparent_layer()
    session.options["select_mode"] = "wand"
    SelectTool().pointer_down(session, PointerEvent(26, 26))
    selection = session.document.selection
    assert selection is not None
    assert selection.getpixel((26, 26)) == 255
    # The outline stops the walk, so the open margin and the canvas edge stay out.
    assert selection.getpixel((11, 11)) == 0
    assert selection.getpixel((0, 0)) == 0
    assert selection.getpixel((18, 18)) == 0

    session = _outline_on_transparent_layer()
    session.options["select_mode"] = "wand"
    SelectTool().pointer_down(session, PointerEvent(11, 11))
    selection = session.document.selection
    assert selection.size == (64, 64)
    assert selection.getpixel((11, 11)) == 255
    assert selection.getpixel((0, 0)) == 255
    assert selection.getpixel((26, 26)) == 0
    assert selection.getpixel((18, 18)) == 0


def test_wand_click_outside_the_canvas_leaves_the_selection():
    session = Session(new_document(16, 16, "white"))
    session.options["select_mode"] = "wand"
    mask = Image.new("L", (16, 16), 0)
    mask.putpixel((2, 2), 255)
    session.set_selection(mask)
    before = np.asarray(session.document.selection).copy()
    steps = len(session.history.undo_stack)
    messages: list[str] = []
    session.set_message_handler(messages.append)
    SelectTool().pointer_down(session, PointerEvent(-1, 1))
    assert any("outside the canvas" in message for message in messages)
    assert np.array_equal(np.asarray(session.document.selection), before)
    assert len(session.history.undo_stack) == steps


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
    # radius 12, hardness 0.85: the solid core is 10.2 and the step is 15% of it.
    assert abs(stroke_spacing(12, 0.85) - (12 * 0.85 * 0.15)) < 1e-9
    # A large hard brush may step more than a pixel. The step stays inside the core.
    assert stroke_spacing(80, 1.0) > 1.5
    assert stroke_spacing(80, 1.0) < 80
    # A small or soft brush keeps the sub-pixel floor.
    assert stroke_spacing(1.0, 0.2) == 0.35
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


def test_sparse_circle_bows_past_the_straight_chords():
    """Eight samples of a circle. The spline covers the arc; the chords do not."""
    cx, cy, radius = 50.0, 50.0, 36.0
    points = []
    for index in range(8):
        angle = 2 * np.pi * index / 8
        points.append((cx + radius * np.cos(angle), cy + radius * np.sin(angle)))
    points.append(points[0])
    arc = 2 * np.pi * 0.5 / 8
    pixel = (int(round(cx + radius * np.cos(arc))), int(round(cy + radius * np.sin(arc))))

    session = Session(new_document(100, 100, "transparent"))
    session.foreground = (255, 0, 0, 255)
    session.options["brush_size"] = 3
    session.options["brush_hardness"] = 100
    session.options["brush_opacity"] = 100
    tool = BrushTool()
    tool.pointer_down(session, PointerEvent(*points[0]))
    for point in points[1:-1]:
        tool.pointer_move(session, PointerEvent(*point))
    tool.pointer_up(session, PointerEvent(*points[-1]))
    curve = np.array(session.active_layer().image)
    assert curve[pixel[1], pixel[0], 3] > 200

    spacing = stroke_spacing(1.5, 1.0)
    centers = [points[0]]
    for start, end in zip(points, points[1:]):
        centers.extend(interpolate(start, end, spacing))
    chords = np.array(
        paint_points(
            Image.new("RGBA", (100, 100), (0, 0, 0, 0)),
            centers,
            1.5,
            1.0,
            (255, 0, 0, 255),
            1.0,
        )
    )
    assert chords[pixel[1], pixel[0], 3] == 0


def test_tip_redraw_does_not_darken_a_translucent_stroke():
    """Redrawing the provisional tip must match painting the same points once."""
    points = [(8.0, 20.0), (28.0, 20.0), (48.0, 21.0), (68.0, 20.0), (90.0, 20.0)]

    def _stroke(incremental: bool) -> np.ndarray:
        session = Session(new_document(110, 40, "transparent"))
        session.foreground = (0, 0, 0, 255)
        session.options["brush_size"] = 6
        session.options["brush_hardness"] = 100
        session.options["brush_opacity"] = 50
        tool = BrushTool()
        tool.pointer_down(session, PointerEvent(*points[0]))
        if incremental:
            for point in points[1:-1]:
                tool.pointer_move(session, PointerEvent(*point))
            tool.pointer_up(session, PointerEvent(*points[-1]))
        else:
            tool.extend_points(session, points[1:])
            tool.commit_stroke(session)
        return np.array(session.active_layer().image)

    incremental = _stroke(True)
    batched = _stroke(False)
    assert incremental.shape == batched.shape
    assert np.max(np.abs(incremental.astype(int) - batched.astype(int))) <= 1
    # The outer pixels of a 50% stamp are the only coverage there. Redrawing
    # the tip would push those up toward opaque.
    assert np.any((incremental[..., 3] > 40) & (incremental[..., 3] < 200))


def test_region_composite_matches_a_cropped_document():
    from pixedit.composite import composite, composite_region

    document = new_document(40, 30, "white")
    spot = raster_layer("Spot", Image.new("RGBA", (12, 10), (255, 0, 0, 180)), 8, 6)
    spot.opacity = 0.6
    spot.blend_mode = "multiply"
    document.layers.append(spot)
    full = np.array(composite(document))
    box = (4, 2, 24, 20)
    region = np.array(composite_region(document, box))
    assert np.array_equal(region, full[box[1] : box[3], box[0] : box[2]])


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
    # Hardness 0 is a light coat at the center, not a solid disk with a soft edge.
    assert soft[48, 88, 0] < 140
    assert hard[48, 88, 0] - soft[48, 88, 0] > 100
    assert soft[48, 108, 0] < 120
    assert faint[48, 88, 0] < hard[48, 88, 0] - 40

    hole = np.array(brush_preview_image(40, 1.0, 1.0, color, erase=True))
    assert hole[2, 2, 3] == 255
    assert hole[48, 88, 3] == 0
    soft_hole = np.array(brush_preview_image(48, 0.0, 1.0, color, erase=True))
    assert 0 < int(soft_hole[48, 108, 3]) < 255

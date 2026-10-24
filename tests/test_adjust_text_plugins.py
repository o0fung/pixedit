"""Adjustments, text rendering, and the shipped plugin."""

from __future__ import annotations

import numpy as np
from PIL import Image

from pixedit.adjust import brightness_contrast, desaturate, hue_saturation, invert, levels
from pixedit.factory import raster_layer, redaction_layer, text_layer
from pixedit.geom import hit_layer
from pixedit.model import Layer, TextInfo, new_document
from pixedit.plugins.examples.basic import MarkerTool, apply_vignette, vignette_image
from pixedit.plugins.loader import load_plugins, reset_plugins
from pixedit.session import Session
from pixedit.textutil import render_text
from pixedit.tools.base import PointerEvent
from pixedit.tools.builtins import TextTool, TransformTool


def test_adjustments_are_predictable():
    image = Image.new("RGBA", (2, 2), (40, 40, 40, 255))
    brighter = np.array(brightness_contrast(image, 50, 0))
    assert brighter[0, 0, 0] > 40
    identity = brightness_contrast(image, 0, 0)
    assert identity.getpixel((0, 0)) == (40, 40, 40, 255)
    leveled = levels(image, 0, 255, 1)
    assert leveled.getpixel((0, 0)) == (40, 40, 40, 255)
    assert invert(image).getpixel((0, 0)) == (215, 215, 215, 255)
    color = Image.new("RGBA", (2, 2), (200, 10, 10, 255))
    gray = desaturate(color).getpixel((0, 0))
    assert abs(gray[0] - gray[1]) <= 1 and abs(gray[1] - gray[2]) <= 1
    shifted = hue_saturation(color, hue_degrees=120)
    pixel = shifted.getpixel((0, 0))
    assert pixel[1] > pixel[0]
    assert pixel[1] > pixel[2]


def test_text_render_and_edit():
    info = TextInfo("Hi", None, "Default", 28, (255, 0, 0, 255), 0)
    image = render_text(info)
    assert image.getextrema()[3][1] > 0
    info.rotation = 35
    rotated = render_text(info)
    assert rotated.size != image.size

    session = Session(new_document(80, 80, "white"))
    session.set_text_prompter(lambda _initial: TextInfo("Note", None, "Default", 20, (0, 0, 0, 255), 15))

    TextTool().pointer_down(session, PointerEvent(8, 10))
    layer = session.active_layer()
    assert layer.kind == "text"
    assert layer.text is not None and layer.text.text == "Note"
    assert layer.text.rotation == 15
    assert layer.image.getextrema()[3][1] > 0
    session.undo()
    assert session.active_layer().kind == "raster"


def _empty_pixel(layer: Layer) -> tuple[int, int]:
    """A canvas pixel inside the bitmap that has no paint. Text padding qualifies."""
    alpha = np.asarray(layer.image.convert("RGBA"))[..., 3]
    ys, xs = np.nonzero(alpha == 0)
    assert len(xs) > 0
    return int(layer.x + xs[0]), int(layer.y + ys[0])


def _accept(text: str):
    return lambda initial: TextInfo(
        text,
        None if initial is None else initial.font_path,
        "Default" if initial is None else initial.font_family,
        20 if initial is None else initial.size,
        (0, 0, 0, 255) if initial is None else initial.color,
        0.0 if initial is None else initial.rotation,
        "left" if initial is None else initial.align,
        4 if initial is None else initial.line_spacing,
    )


def test_text_tool_edits_from_a_click_in_the_box():
    session = Session(new_document(200, 200, "white"))
    layer = text_layer("Greeting", TextInfo("Hello", None, "Default", 28, (0, 0, 0, 255)), 30, 30)
    session.add_layer(layer)
    session.layer_index = 0
    px, py = _empty_pixel(layer)
    session.set_text_prompter(_accept("Edited"))
    TextTool().pointer_down(session, PointerEvent(px, py))
    assert len(session.document.layers) == 2
    edited = session.active_layer()
    assert edited.id == layer.id
    assert edited.kind == "text"
    assert edited.text is not None and edited.text.text == "Edited"


def test_text_tool_adds_when_the_click_misses_every_text_box():
    session = Session(new_document(200, 200, "white"))
    layer = text_layer("Greeting", TextInfo("Hello", None, "Default", 20, (0, 0, 0, 255)), 8, 8)
    session.add_layer(layer)
    session.layer_index = 0
    x, y, width, height = layer.bounds()
    outside = (x + width + 4, y + height + 4)
    assert not (x <= outside[0] < x + width and y <= outside[1] < y + height)
    session.set_text_prompter(_accept("New"))
    TextTool().pointer_down(session, PointerEvent(*outside))
    assert len(session.document.layers) == 3
    added = session.active_layer()
    assert added.kind == "text"
    assert added.text is not None and added.text.text == "New"
    assert layer.text is not None and layer.text.text == "Hello"


def test_text_tool_overlapping_boxes_pick_the_upper_layer():
    session = Session(new_document(200, 200, "white"))
    lower = text_layer("Lower", TextInfo("Low", None, "Default", 28, (0, 0, 0, 255)), 20, 20)
    upper = text_layer("Upper", TextInfo("Upper", None, "Default", 28, (255, 0, 0, 255)), 20, 20)
    session.add_layer(lower)
    session.add_layer(upper)
    session.layer_index = 0
    px, py = _empty_pixel(upper)
    lx, ly, lw, lh = lower.bounds()
    assert lx <= px < lx + lw and ly <= py < ly + lh
    seen: list[str] = []

    def prompter(initial):
        assert initial is not None
        seen.append(initial.text)
        return _accept(initial.text + "!")(initial)

    session.set_text_prompter(prompter)
    TextTool().pointer_down(session, PointerEvent(px, py))
    assert seen == ["Upper"]
    assert upper.text is not None and upper.text.text == "Upper!"
    assert lower.text is not None and lower.text.text == "Low"
    assert len(session.document.layers) == 3


def test_move_selects_text_box_and_background():
    session = Session(new_document(200, 200, "white"))
    background = session.document.layers[0]
    layer = text_layer("Greeting", TextInfo("Hello", None, "Default", 20, (0, 0, 0, 255)), 30, 30)
    session.add_layer(layer)
    px, py = _empty_pixel(layer)
    text_index = session.document.index_of(layer.id)
    assert hit_layer(session.document, px, py) == text_index
    x, y, width, height = layer.bounds()
    outside = (x + width + 30, y + height + 30)
    assert hit_layer(session.document, *outside) == 0

    # A drag that starts on the text, or outside it, keeps the background
    # selected and leaves the text where it is.
    session.layer_index = 0
    tool = TransformTool()
    tool.pointer_down(session, PointerEvent(px, py))
    tool.pointer_up(session, PointerEvent(px + 8, py + 6))
    assert session.active_layer() is background
    assert (layer.x, layer.y) == (x, y)
    assert (background.x, background.y) == (8, 6)
    session.undo()
    tool.pointer_down(session, PointerEvent(*outside))
    tool.pointer_up(session, PointerEvent(outside[0] + 4, outside[1] + 3))
    assert session.active_layer() is background
    assert (layer.x, layer.y) == (x, y)
    assert (background.x, background.y) == (4, 3)


def test_move_selects_a_raster_hole_and_ignores_redaction_margins():
    session = Session(new_document(200, 200, "white"))
    background = session.document.layers[0]
    image = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    image.putpixel((0, 0), (255, 0, 0, 255))
    image.putpixel((9, 9), (255, 0, 0, 255))
    sprite = raster_layer("Hole", image, 40, 40)
    session.add_layer(sprite)
    assert hit_layer(session.document, 45, 45) == session.document.index_of(sprite.id)

    mask = Image.new("L", (200, 200), 0)
    mask.putpixel((10, 10), 255)
    mask.putpixel((14, 12), 255)
    redaction = redaction_layer("Hide", mask, "pixelate", 8)
    session.add_layer(redaction)
    assert hit_layer(session.document, 80, 80) == 0
    assert hit_layer(session.document, 12, 11) == session.document.index_of(redaction.id)

    # Those hits do not retarget Transform. The background stays selected
    # and is the only layer that moves.
    session.layer_index = 0
    tool = TransformTool()
    tool.pointer_down(session, PointerEvent(45, 45))
    tool.pointer_up(session, PointerEvent(53, 49))
    assert session.active_layer() is background
    assert (sprite.x, sprite.y) == (40, 40)
    assert (redaction.x, redaction.y) == (0, 0)
    assert (background.x, background.y) == (8, 4)
    session.undo()
    tool.pointer_down(session, PointerEvent(12, 11))
    tool.pointer_up(session, PointerEvent(16, 15))
    assert session.active_layer() is background
    assert (sprite.x, sprite.y) == (40, 40)
    assert (redaction.x, redaction.y) == (0, 0)
    assert (background.x, background.y) == (4, 4)


def test_example_plugin_registers_and_darkens_corners():
    reset_plugins()
    api = load_plugins()
    assert api.has("example.vignette")
    assert api.has("example.marker")
    image = Image.new("RGBA", (40, 40), (220, 220, 220, 255))
    out = np.array(vignette_image(image, 0.8))
    assert out[20, 20, 0] > out[0, 0, 0]

    session = Session(new_document(40, 40, "white"))
    apply_vignette(session)
    assert session.history.can_undo()
    center = session.active_layer().image.getpixel((20, 20))[0]
    corner = session.active_layer().image.getpixel((0, 0))[0]
    assert center > corner

    session.undo()
    session.foreground = (255, 0, 0, 255)
    session.options["marker_size"] = 8
    messages: list[str] = []
    session.set_message_handler(messages.append)
    MarkerTool().pointer_down(session, PointerEvent(20, 20))
    assert session.active_layer().image.getpixel((20, 20))[0] > 200
    assert any("Stamped marker" in message for message in messages)

    from pixedit.factory import raster_layer

    sprite = raster_layer("Dot", Image.new("RGBA", (3, 3), (0, 255, 0, 255)), 4, 4)
    session.add_layer(sprite)
    session.layer_index = session.document.index_of(sprite.id)
    MarkerTool().pointer_down(session, PointerEvent(30, 22))
    marked = session.active_layer()
    assert marked.image.size != (40, 40)
    assert marked.image.getpixel((4 - marked.x, 4 - marked.y))[1] > 200
    assert marked.image.getpixel((int(round(30 - marked.x)), int(round(22 - marked.y))))[0] > 200


def test_new_text_uses_panel_style_and_a_click_keeps_the_layer_style():
    session = Session(new_document(80, 80, "white"))
    session.options["text_size"] = 72
    session.options["text_rotation"] = 15
    session.options["text_align"] = "right"
    session.options["text_line_spacing"] = 9
    session.foreground = (1, 2, 3, 200)
    seen = []
    session.set_text_prompter(lambda initial: seen.append(initial) or None)
    TextTool().pointer_down(session, PointerEvent(70, 70))
    assert seen[0].text == "Text"
    assert seen[0].size == 72
    assert seen[0].rotation == 15
    assert seen[0].align == "right"
    assert seen[0].line_spacing == 9
    assert seen[0].color == (1, 2, 3, 200)
    assert session.active_layer().kind == "raster"

    info = TextInfo("Hi", None, "Default", 28, (255, 0, 0, 255), 0)
    session.add_text(info, 4, 4)
    session.options["text_size"] = 99
    session.foreground = (0, 255, 0, 255)
    layer = session.active_layer()
    assert layer.text is not None
    x, y, _width, _height = layer.bounds()

    def edit(initial):
        seen.append(initial)
        return TextInfo(
            initial.text + "!",
            initial.font_path,
            initial.font_family,
            initial.size,
            initial.color,
            initial.rotation,
            initial.align,
            initial.line_spacing,
        )

    session.set_text_prompter(edit)
    TextTool().pointer_down(session, PointerEvent(x + 1, y + 1))
    edited = session.active_layer()
    assert edited.text is not None
    assert edited.text.text == "Hi!"
    assert edited.text.size == 28
    assert edited.text.color == (255, 0, 0, 255)


def test_text_style_preview_commits_once_and_keeps_the_center():
    session = Session(new_document(80, 80, "white"))
    session.add_text(TextInfo("Hi", None, "Default", 28, (255, 0, 0, 255), 0), 6, 8)
    layer = session.active_layer()
    center = (layer.x + layer.image.width / 2, layer.y + layer.image.height / 2)
    session.options["text_size"] = 60
    session.foreground = (0, 0, 255, 255)
    steps = len(session.history.undo_stack)
    session.apply_text_style(transient=True)
    assert session.active_layer().text.size == 60
    assert session.active_layer().text.color[2] == 255
    assert len(session.history.undo_stack) == steps
    session.apply_text_style(transient=False)
    assert len(session.history.undo_stack) == steps + 1
    moved = session.active_layer()
    assert abs((moved.x + moved.image.width / 2) - center[0]) <= 1
    assert abs((moved.y + moved.image.height / 2) - center[1]) <= 1
    session.undo()
    assert session.active_layer().text.size == 28
    assert session.active_layer().text.color[0] == 255
    session.adopt_text_style(session.active_layer().text)
    steps = len(session.history.undo_stack)
    session.apply_text_style(transient=False)
    assert len(session.history.undo_stack) == steps


def test_highlight_and_shade_move_opposite_ends():
    from pixedit.adjust import color_grade, highlights_shadows

    image = Image.new("RGBA", (2, 1), (0, 0, 0, 0))
    image.putpixel((0, 0), (20, 20, 20, 200))
    image.putpixel((1, 0), (230, 230, 230, 200))
    assert color_grade(image, 0, 0, 0, 0, 0, 0).getpixel((0, 0)) == (20, 20, 20, 200)

    shaded = highlights_shadows(image, 0, 80)
    assert shaded.getpixel((0, 0))[0] - 20 > shaded.getpixel((1, 0))[0] - 230
    assert shaded.getpixel((0, 0))[3] == 200

    lifted = highlights_shadows(image, 80, 0)
    assert lifted.getpixel((1, 0))[0] - 230 > lifted.getpixel((0, 0))[0] - 20
    assert lifted.getpixel((1, 0))[3] == 200


def test_color_preview_commits_inside_the_selection_and_cancels():
    from pixedit.tools.builtins import ColorTool

    session = Session()
    session.new_canvas(8, 4, "white")
    layer = session.active_layer()
    layer.image = Image.new("RGBA", (8, 4), (40, 40, 40, 255))
    selection = Image.new("L", (8, 4), 0)
    selection.paste(255, (0, 0, 4, 4))
    session.document.selection = selection
    session.options["color_brightness"] = 100
    session.preview_color_adjust()
    assert session.color_adjust_active()
    assert layer.image.getpixel((1, 1))[0] > 40
    assert layer.image.getpixel((6, 1))[0] == 40

    session.commit_color_adjust()
    assert session.history.undo_stack[-1].name == "Color"
    assert session.option("color_brightness") == 0
    assert not session.color_adjust_active()
    graded = layer.image.getpixel((1, 1))[0]
    assert graded > 40
    session.undo()
    assert layer.image.getpixel((1, 1))[0] == 40

    session.options["color_brightness"] = 80
    session.preview_color_adjust()
    assert ColorTool().pointer_cancel(session) is True
    assert layer.image.getpixel((1, 1))[0] == 40
    assert session.option("color_brightness") == 0

    other = raster_layer("Other", Image.new("RGBA", (4, 4), (0, 0, 0, 255)))
    session.add_layer(other)
    session.layer_index = 0
    session.options["color_brightness"] = 60
    session.preview_color_adjust()
    session.layer_index = session.document.layers.index(other)
    assert session.document.layers[0].image.getpixel((1, 1))[0] == 40
    assert session.option("color_brightness") == 0
    assert not session.color_adjust_active()


def test_color_preview_restores_text_until_apply():
    session = Session(new_document(80, 80, "white"))
    layer = text_layer("Note", TextInfo("Hi", None, "Default", 24, (20, 20, 20, 255)), 8, 8)
    session.add_layer(layer)
    session.options["color_brightness"] = 40
    session.preview_color_adjust()
    assert session.active_layer().kind == "raster"
    assert session.active_layer().text is None
    session.cancel_color_adjust()
    assert session.active_layer().kind == "text"
    assert session.active_layer().text is not None
    assert session.active_layer().text.text == "Hi"

    session.options["color_brightness"] = 40
    session.preview_color_adjust()
    session.commit_color_adjust()
    assert session.active_layer().kind == "raster"
    assert session.active_layer().text is None
    session.undo()
    assert session.active_layer().kind == "text"
    assert session.active_layer().text.text == "Hi"


def _ink_bands(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """Opaque rows grouped into lines: (y0, y1, x0, x1)."""
    arr = np.array(image.convert("RGBA"))
    rows = np.where(arr[..., 3].max(axis=1) > 20)[0]
    assert len(rows) > 0
    bands: list[tuple[int, int]] = []
    start = previous = int(rows[0])
    for y in rows[1:]:
        y = int(y)
        if y > previous + 2:
            bands.append((start, previous))
            start = y
        previous = y
    bands.append((start, previous))
    spans = []
    for y0, y1 in bands:
        cols = np.where(arr[y0 : y1 + 1, :, 3].max(axis=0) > 20)[0]
        spans.append((y0, y1, int(cols[0]), int(cols[-1])))
    return spans


def test_group_fonts_keeps_one_family_and_prefers_regular():
    from pixedit.textutil import group_fonts

    grouped = group_fonts(
        (
            ("Arial", "Bold", "/a-bold.ttf"),
            ("Arial", "Italic", "/a-italic.ttf"),
            ("Arial", "Regular", "/a.ttf"),
            ("Arial", "Bold", "/a-bold-dup.ttf"),
            ("Times", "Italic", "/t.ttf"),
        )
    )
    assert [name for name, _styles in grouped] == ["Arial", "Times"]
    assert grouped[0][1][0] == ("Regular", "/a.ttf")
    assert ("Bold", "/a-bold.ttf") in grouped[0][1]
    assert all(path != "/a-bold-dup.ttf" for _style, path in grouped[0][1])
    assert grouped[1][1] == (("Italic", "/t.ttf"),)


def test_text_align_spacing_and_justify():
    black = (0, 0, 0, 255)
    single_left = render_text(TextInfo("Hello", None, "Default", 28, black, 0, "left", 4))
    single_justify = render_text(TextInfo("Hello", None, "Default", 28, black, 0, "justify", 4))
    assert single_left.tobytes() == single_justify.tobytes()

    tight = render_text(TextInfo("A\nB", None, "Default", 32, black, 0, "left", 0))
    loose = render_text(TextInfo("A\nB", None, "Default", 32, black, 0, "left", 40))
    assert loose.height > tight.height + 20

    words = "Hi there\nHello world"
    left = render_text(TextInfo(words, None, "Default", 36, black, 0, "left", 8))
    center = render_text(TextInfo(words, None, "Default", 36, black, 0, "center", 8))
    justify = render_text(TextInfo(words, None, "Default", 36, black, 0, "justify", 8))
    left_bands = _ink_bands(left)
    center_bands = _ink_bands(center)
    just_bands = _ink_bands(justify)
    assert len(left_bands) >= 2 and len(center_bands) >= 2 and len(just_bands) >= 2
    left_short = left_bands[0][3] - left_bands[0][2]
    left_long = left_bands[1][3] - left_bands[1][2]
    assert left_short < left_long - 4
    left_mid = (left_bands[0][2] + left_bands[0][3]) / 2
    center_mid = (center_bands[0][2] + center_bands[0][3]) / 2
    long_mid = (center_bands[1][2] + center_bands[1][3]) / 2
    assert abs(center_mid - long_mid) < abs(left_mid - (left_bands[1][2] + left_bands[1][3]) / 2)
    just_short = just_bands[0][3] - just_bands[0][2]
    just_long = just_bands[1][3] - just_bands[1][2]
    assert abs(just_short - just_long) <= 4


def test_align_change_is_one_undo_step():
    session = Session(new_document(160, 120, "white"))
    session.add_text(TextInfo("Hi there\nHello", None, "Default", 28, (0, 0, 0, 255), 0), 8, 8)
    session.options["text_align"] = "justify"
    steps = len(session.history.undo_stack)
    session.apply_text_style(transient=False)
    assert session.active_layer().text.align == "justify"
    assert len(session.history.undo_stack) == steps + 1
    session.options["text_line_spacing"] = 24
    session.apply_text_style(transient=True)
    assert session.active_layer().text.line_spacing == 24
    assert len(session.history.undo_stack) == steps + 1
    session.apply_text_style(transient=False)
    assert len(session.history.undo_stack) == steps + 2
    session.undo()
    assert session.active_layer().text.line_spacing == 4
    assert session.active_layer().text.align == "justify"

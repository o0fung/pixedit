"""Adjustments, text rendering, and the shipped plugin."""

from __future__ import annotations

import numpy as np
from PIL import Image

from pixedit.adjust import brightness_contrast, desaturate, hue_saturation, invert, levels
from pixedit.model import TextInfo, new_document
from pixedit.plugins.examples.basic import MarkerTool, apply_vignette, vignette_image
from pixedit.plugins.loader import load_plugins, reset_plugins
from pixedit.session import Session
from pixedit.textutil import render_text
from pixedit.tools.base import PointerEvent


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
    from pixedit.tools.builtins import TextTool

    TextTool().pointer_down(session, PointerEvent(8, 10))
    layer = session.active_layer()
    assert layer.kind == "text"
    assert layer.text is not None and layer.text.text == "Note"
    assert layer.text.rotation == 15
    assert layer.image.getextrema()[3][1] > 0
    session.undo()
    assert session.active_layer().kind == "raster"


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

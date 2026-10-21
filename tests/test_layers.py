"""Layer stack commands: order, opacity, duplicate, delete, merge, undo."""

from __future__ import annotations

from PIL import Image

from pixedit.composite import composite
from pixedit.factory import raster_layer
from pixedit.model import new_document
from pixedit.session import Session


def test_stack_ops_and_undo():
    session = Session(new_document(8, 8, "white"))
    assert len(session.document.layers) == 1
    session.add_blank_layer()
    assert len(session.document.layers) == 2
    session.document.layers[1].name = "Paint"
    session.rename_active("Sketch")
    assert session.active_layer().name == "Sketch"
    session.set_opacity(session.active_layer(), 0.4)
    assert session.active_layer().opacity == 0.4
    session.set_blend(session.active_layer(), "screen")
    assert session.active_layer().blend_mode == "screen"
    session.set_visibility(session.active_layer(), False)
    assert session.active_layer().visible is False
    session.duplicate_active()
    assert len(session.document.layers) == 3
    assert session.active_layer().name.endswith("copy")
    session.move_active(-1)
    session.delete_active()
    assert len(session.document.layers) == 2
    session.undo()
    assert len(session.document.layers) == 3
    session.redo()
    assert len(session.document.layers) == 2


def test_merge_down_bakes_upper_layer():
    session = Session(new_document(4, 4, "white"))
    red = raster_layer("Red", Image.new("RGBA", (4, 4), (255, 0, 0, 255)))
    session.add_layer(red)
    session.layer_index = 1
    session.merge_down_active()
    assert len(session.document.layers) == 1
    pixel = session.document.layers[0].image.getpixel((1, 1))
    assert pixel[0] > 250
    assert pixel[1] < 5
    assert pixel[3] == 255
    session.undo()
    assert len(session.document.layers) == 2
    assert composite(session.document).getpixel((0, 0))[0] > 250

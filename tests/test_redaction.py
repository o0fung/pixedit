"""Privacy masks change the composite and leave source pixels alone."""

from __future__ import annotations

import numpy as np
from PIL import Image

from pixedit.composite import composite
from pixedit.factory import raster_layer, redaction_layer
from pixedit.model import Document
from pixedit.session import Session
from pixedit.tools.base import PointerEvent
from pixedit.tools.builtins import RedactTool


def _half_image(size=32):
    arr = np.zeros((size, size, 4), np.uint8)
    arr[:, : size // 2] = (255, 0, 0, 255)
    arr[:, size // 2 :] = (0, 0, 255, 255)
    return Image.fromarray(arr, "RGBA")


def test_pixelate_mask_does_not_destroy_source():
    size = 16
    source = Image.new("RGBA", (size, size), (0, 0, 0, 255))
    source.putpixel((1, 1), (255, 255, 255, 255))
    before = np.array(source)
    mask = Image.new("L", (size, size), 0)
    # Cover the whole frame so the mosaic cell that holds the white pixel is included.
    mask.paste(255, (0, 0, size, size))
    doc = Document(
        size,
        size,
        [
            raster_layer("Photo", source),
            redaction_layer("Privacy", mask, "pixelate", 8),
        ],
    )
    flat = composite(doc)
    assert not np.array_equal(np.array(flat), before)
    assert flat.getpixel((1, 1))[0] < 40
    assert np.array_equal(np.array(doc.layers[0].image), before)


def test_partial_mask_leaves_unmasked_pixels():
    size = 16
    source = Image.new("RGBA", (size, size), (0, 0, 0, 255))
    source.putpixel((1, 1), (255, 255, 255, 255))
    mask = Image.new("L", (size, size), 0)
    mask.paste(255, (8, 0, size, size))
    doc = Document(
        size,
        size,
        [raster_layer("Photo", source), redaction_layer("Privacy", mask, "pixelate", 8)],
    )
    flat = composite(doc)
    assert flat.getpixel((1, 1)) == (255, 255, 255, 255)
    assert doc.layers[0].image.getpixel((1, 1)) == (255, 255, 255, 255)


def test_blur_mask_softens_an_edge_only_inside_the_mask():
    image = _half_image(48)
    before = np.array(image)
    mask = Image.new("L", (48, 48), 0)
    mask.paste(255, (16, 0, 32, 48))
    doc = Document(
        48,
        48,
        [raster_layer("Photo", image), redaction_layer("Privacy", mask, "blur", 6)],
    )
    flat = np.array(composite(doc))
    # The boundary sits at x=24, inside the mask, so blur mixes red and blue.
    assert flat[24, 24, 0] < 250
    assert flat[24, 24, 2] > 5
    # Far from the mask the original red column is unchanged.
    assert flat[8, 4, 0] > 250
    assert np.array_equal(np.array(doc.layers[0].image), before)


def test_redact_tool_is_undoable_and_nondestructive():
    session = Session()
    session.new_canvas(32, 32, "white")
    base = session.document.layers[0]
    base.image = _half_image(32)
    before = np.array(base.image)
    session.options["redact_shape"] = "rect"
    session.options["redact_kind"] = "blur"
    session.options["redact_amount"] = 4
    tool = RedactTool()
    tool.pointer_down(session, PointerEvent(8, 4))
    tool.pointer_move(session, PointerEvent(24, 20))
    tool.pointer_up(session, PointerEvent(24, 28))
    assert any(layer.kind == "redaction" for layer in session.document.layers)
    assert np.array_equal(np.array(session.document.layers[0].image), before)
    mask = next(layer for layer in session.document.layers if layer.kind == "redaction").mask
    assert mask.getpixel((16, 16)) == 255
    assert mask.getpixel((1, 1)) == 0
    session.undo()
    assert all(layer.kind != "redaction" for layer in session.document.layers)
    assert np.array_equal(np.array(session.document.layers[0].image), before)

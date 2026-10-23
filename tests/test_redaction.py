"""Privacy masks change the composite and leave source pixels alone."""

from __future__ import annotations

import numpy as np
from PIL import Image

from pixedit.composite import composite
from pixedit.factory import raster_layer, redaction_layer
from pixedit.model import Document
from pixedit.session import Session
from pixedit.tools.base import PointerEvent
from pixedit.tools.builtins import MaskTool


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
    session.options["mask_shape"] = "rect"
    session.options["mask_kind"] = "blur"
    session.options["mask_amount"] = 4
    tool = MaskTool()
    tool.pointer_down(session, PointerEvent(8, 4))
    tool.pointer_move(session, PointerEvent(24, 20))
    tool.pointer_up(session, PointerEvent(24, 28))
    assert any(layer.kind == "redaction" for layer in session.document.layers)
    assert np.array_equal(np.array(session.document.layers[0].image), before)
    mask = next(layer for layer in session.document.layers if layer.kind == "redaction").mask
    assert mask.size == (32, 32)
    assert mask.getpixel((16, 16)) == 255
    assert mask.getpixel((1, 1)) == 0
    session.undo()
    assert all(layer.kind != "redaction" for layer in session.document.layers)
    assert np.array_equal(np.array(session.document.layers[0].image), before)


def _split_mask_doc(kind: str, amount: float, *, color=(0, 0, 0, 255), seed=7, fill=(200, 30, 30, 255)):
    image = Image.new("RGBA", (8, 8), fill)
    before = np.array(image)
    mask = Image.new("L", (8, 8), 0)
    mask.paste(255, (4, 0, 8, 8))
    doc = Document(
        8,
        8,
        [
            raster_layer("Photo", image),
            redaction_layer("Mask", mask, kind, amount, color=color, seed=seed),
        ],
    )
    flat = np.array(composite(doc))
    assert np.array_equal(np.array(doc.layers[0].image), before)
    assert np.array_equal(flat[:, :4], before[:, :4])
    return flat, before


def test_mask_effects_stay_inside_the_mask_and_keep_the_source():
    from pixedit.model import clamp_mask_amount

    assert clamp_mask_amount("blur", 100) == 40
    assert clamp_mask_amount("pixelate", 1) == 2
    assert clamp_mask_amount("solid", 80) == 80

    solid, _before = _split_mask_doc("solid", 12, color=(9, 8, 7, 255))
    assert tuple(solid[2, 6]) == (9, 8, 7, 255)

    clear = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    full = Image.new("L", (4, 4), 255)
    covered = Document(
        4,
        4,
        [raster_layer("Empty", clear), redaction_layer("Bar", full, "solid", 0, color=(9, 8, 7, 128), seed=3)],
    )
    assert composite(covered).getpixel((1, 1)) == (9, 8, 7, 255)
    assert clear.getpixel((1, 1))[3] == 0

    noise_a, noise_before = _split_mask_doc("noise", 100, seed=11)
    noise_b = np.array(composite(Document(
        8,
        8,
        [
            raster_layer("Photo", Image.fromarray(noise_before, "RGBA")),
            redaction_layer("Mask", _right_mask(), "noise", 100, seed=11),
        ],
    )))
    assert np.array_equal(noise_a, noise_b)
    assert not np.array_equal(noise_a[:, 4:], noise_before[:, 4:])

    gray, _red = _split_mask_doc("desaturate", 100)
    assert abs(int(gray[2, 6, 0]) - int(gray[2, 6, 1])) <= 1
    assert abs(int(gray[2, 6, 1]) - int(gray[2, 6, 2])) <= 1

    flipped, _red = _split_mask_doc("invert", 100)
    assert flipped[2, 6, 1] > flipped[2, 6, 0]

    dimmed, _red = _split_mask_doc("dim", 100)
    assert tuple(dimmed[2, 6, :3]) == (0, 0, 0)
    assert dimmed[2, 6, 3] == 255

    sharp, sharp_before = _split_mask_doc("sharpen", 100, fill=(120, 120, 120, 255))
    # A flat field does not move. Rebuild with an edge through the mask.
    edge = np.full((8, 8, 4), (80, 80, 80, 255), np.uint8)
    edge[:, 4:] = (180, 180, 180, 255)
    edge_before = edge.copy()
    edge_mask = Image.new("L", (8, 8), 0)
    edge_mask.paste(255, (2, 0, 6, 8))
    edge_doc = Document(
        8,
        8,
        [
            raster_layer("Edge", Image.fromarray(edge, "RGBA")),
            redaction_layer("Mask", edge_mask, "sharpen", 100, seed=5),
        ],
    )
    edge_flat = np.array(composite(edge_doc))
    assert np.array_equal(edge_flat[:, :2], edge_before[:, :2])
    assert not np.array_equal(edge_flat[:, 2:6], edge_before[:, 2:6])
    assert sharp[0, 0, 0] == sharp_before[0, 0, 0]


def _right_mask() -> Image.Image:
    mask = Image.new("L", (8, 8), 0)
    mask.paste(255, (4, 0, 8, 8))
    return mask


def test_switching_mask_effect_clamps_the_amount_and_keeps_the_seed():
    session = Session()
    session.new_canvas(8, 8, "white")
    layer = redaction_layer("Mask", Image.new("L", (8, 8), 255), "noise", 100, seed=21)
    session.add_layer(layer)
    session.layer_index = session.document.index_of(layer.id)
    session.set_redaction_settings("blur", 100, transient=False)
    info = session.active_layer().redaction
    assert info is not None
    assert info.kind == "blur"
    assert info.amount == 40
    assert info.seed == 21
    session.undo()
    assert session.active_layer().redaction.kind == "noise"
    assert session.active_layer().redaction.amount == 100

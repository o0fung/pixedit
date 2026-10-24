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
    mask_layer = next(layer for layer in session.document.layers if layer.kind == "redaction")
    assert mask_layer.target_id == base.id
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


def test_mask_filters_its_picture_and_leaves_the_layer_below():
    bottom = _half_image(48)
    top = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
    for y in range(20, 28):
        for x in range(20, 28):
            top.putpixel((x, y), (0, 180, 0, 255))
    mask = Image.new("L", (48, 48), 0)
    mask.paste(255, (16, 0, 32, 48))
    photo = raster_layer("Photo", top)
    band = redaction_layer("Privacy", mask, "blur", 6, target_id=photo.id)
    doc = Document(48, 48, [raster_layer("Back", bottom), photo, band])
    flat = np.array(composite(doc))
    # Far from the green square the red/blue edge is the lower layer, unblurred.
    assert flat[2, 18, 0] > 250 and flat[2, 18, 2] < 5
    assert flat[2, 30, 2] > 250 and flat[2, 30, 0] < 5
    assert np.array_equal(np.array(doc.layers[0].image), np.array(bottom))
    assert np.array_equal(np.array(doc.layers[1].image), np.array(top))


def test_hidden_picture_hides_its_mask_and_a_hidden_mask_is_skipped():
    photo = raster_layer("Photo", _half_image(16))
    mask = redaction_layer("Mask", Image.new("L", (16, 16), 255), "solid", 0, color=(0, 255, 0, 255), target_id=photo.id)
    back = raster_layer("Back", Image.new("RGBA", (16, 16), (255, 0, 0, 255)))
    doc = Document(16, 16, [back, photo, mask])
    assert composite(doc).getpixel((14, 1)) == (0, 255, 0, 255)
    photo.visible = False
    assert composite(doc).getpixel((14, 1)) == (255, 0, 0, 255)
    photo.visible = True
    mask.visible = False
    assert composite(doc).getpixel((14, 1))[:3] == (0, 0, 255)


def test_apply_mask_bakes_only_the_covered_pixels_and_undoes():
    session = Session()
    session.new_canvas(8, 8, "white")
    photo = raster_layer("Photo", _half_image(8))
    session.add_layer(photo)
    mask = Image.new("L", (8, 8), 0)
    mask.paste(255, (4, 0, 8, 8))
    band = redaction_layer("Mask", mask, "solid", 0, color=(0, 255, 0, 255), target_id=photo.id)
    session.add_layer(band)
    session.layer_index = session.document.index_of(band.id)
    session.apply_active_mask()
    assert [layer.name for layer in session.document.layers] == ["Background", "Photo"]
    baked = session.document.layers[1]
    assert baked.id == photo.id
    assert baked.image.size == (8, 8)
    assert baked.image.getpixel((1, 1)) == (255, 0, 0, 255)
    assert baked.image.getpixel((6, 1)) == (0, 255, 0, 255)
    assert session.document.layers[0].image.getpixel((6, 1)) == (255, 255, 255, 255)
    session.undo()
    assert [layer.name for layer in session.document.layers] == ["Background", "Photo", "Mask"]
    assert session.document.layers[1].image.getpixel((6, 1)) == (0, 0, 255, 255)


def test_apply_stops_at_the_selected_mask_and_leaves_later_masks():
    photo = raster_layer("Photo", _half_image(8))
    left = Image.new("L", (8, 8), 0)
    left.paste(255, (0, 0, 4, 8))
    right = Image.new("L", (8, 8), 0)
    right.paste(255, (4, 0, 8, 8))
    lower = redaction_layer("Lower", left, "solid", 0, color=(0, 255, 0, 255), target_id=photo.id)
    upper = redaction_layer("Upper", right, "solid", 0, color=(255, 255, 255, 255), target_id=photo.id)
    session = Session(Document(8, 8, [photo, lower, upper]))
    session.layer_index = 1
    session.apply_active_mask()
    assert [layer.name for layer in session.document.layers] == ["Photo", "Upper"]
    baked = session.document.layers[0]
    assert baked.image.getpixel((1, 1)) == (0, 255, 0, 255)
    assert baked.image.getpixel((6, 1)) == (0, 0, 255, 255)
    assert composite(session.document).getpixel((6, 1)) == (255, 255, 255, 255)
    session.undo()
    assert [layer.name for layer in session.document.layers] == ["Photo", "Lower", "Upper"]
    assert session.document.layers[0].image.getpixel((1, 1)) == (255, 0, 0, 255)


def test_apply_on_text_becomes_a_raster_and_a_small_layer_grows_to_the_mask():
    from pixedit.factory import text_layer
    from pixedit.model import TextInfo

    info = TextInfo("Hi", None, "Default", 18, (0, 0, 0, 255))
    title = text_layer("Title", info, 2, 2)
    mask = redaction_layer(
        "Mask", Image.new("L", (32, 32), 255), "solid", 0, color=(9, 8, 7, 255), target_id=title.id
    )
    session = Session(Document(32, 32, [title, mask]))
    session.layer_index = 1
    session.apply_active_mask()
    baked = session.document.layers[0]
    assert baked.kind == "raster"
    assert baked.text is None
    assert baked.id == title.id
    assert len(session.document.layers) == 1

    session = Session()
    session.new_canvas(16, 16, "white")
    small = raster_layer("Dot", Image.new("RGBA", (2, 2), (255, 0, 0, 255)), 6, 6)
    session.add_layer(small)
    region = Image.new("L", (16, 16), 0)
    region.paste(255, (4, 4, 10, 10))
    band = redaction_layer("Mask", region, "solid", 0, color=(0, 0, 255, 255), target_id=small.id)
    session.add_layer(band)
    session.layer_index = session.document.index_of(band.id)
    session.apply_active_mask()
    grown = next(layer for layer in session.document.layers if layer.id == small.id)
    assert grown.image.size == (6, 6)
    assert (grown.x, grown.y) == (4, 4)
    assert grown.image.getpixel((0, 0)) == (0, 0, 255, 255)
    assert session.document.layers[0].image.getpixel((0, 0)) == (255, 255, 255, 255)


def test_delete_mask_leaves_the_picture_and_delete_picture_removes_its_masks():
    session = Session()
    session.new_canvas(8, 8, "white")
    photo = raster_layer("Photo", _half_image(8))
    session.add_layer(photo)
    before = np.array(photo.image)
    band = redaction_layer("Mask", Image.new("L", (8, 8), 255), "blur", 4, target_id=photo.id)
    session.add_layer(band)
    session.layer_index = session.document.index_of(band.id)
    session.delete_active()
    assert [layer.name for layer in session.document.layers] == ["Background", "Photo"]
    assert np.array_equal(np.array(session.document.layers[1].image), before)
    session.undo()
    session.layer_index = session.document.index_of(photo.id)
    session.delete_active()
    assert [layer.name for layer in session.document.layers] == ["Background"]
    session.undo()
    assert [layer.name for layer in session.document.layers] == ["Background", "Photo", "Mask"]


def test_mask_block_moves_together_and_duplicate_keeps_the_masks():
    session = Session()
    session.new_canvas(8, 8, "white")
    session.document.layers[0].name = "Back"
    photo = raster_layer("Photo", _half_image(8))
    session.add_layer(photo)
    lower = redaction_layer("A", Image.new("L", (8, 8), 255), "blur", 2, target_id=photo.id)
    upper = redaction_layer("B", Image.new("L", (8, 8), 255), "dim", 10, target_id=photo.id)
    session.add_layer(lower)
    session.add_layer(upper)
    label = raster_layer("Label", Image.new("RGBA", (8, 8), (0, 255, 0, 255)))
    session.add_layer(label)
    session.layer_index = session.document.index_of(lower.id)
    session.move_active(1)
    assert [layer.name for layer in session.document.layers] == ["Back", "Photo", "B", "A", "Label"]
    # A is now the front mask. Moving it again slides the whole block past Label.
    session.layer_index = session.document.index_of(lower.id)
    session.move_active(1)
    assert [layer.name for layer in session.document.layers] == ["Back", "Label", "Photo", "B", "A"]
    session.undo()
    session.undo()
    assert [layer.name for layer in session.document.layers] == ["Back", "Photo", "A", "B", "Label"]

    session.layer_index = session.document.index_of(photo.id)
    session.duplicate_active()
    assert [layer.name for layer in session.document.layers] == ["Back", "Photo", "A", "B", "Photo copy", "Label"]
    assert session.document.layers[1].id == photo.id
    assert session.document.layers[2].target_id == photo.id
    session.undo()
    session.layer_index = session.document.index_of(lower.id)
    session.duplicate_active()
    assert [layer.name for layer in session.document.layers] == ["Back", "Photo", "A", "A copy", "B", "Label"]
    assert session.active_layer().target_id == photo.id


def test_merge_down_on_a_mask_bakes_into_the_picture_only():
    session = Session()
    session.new_canvas(8, 8, "white")
    photo = raster_layer("Photo", _half_image(8))
    session.add_layer(photo)
    band = redaction_layer("Mask", Image.new("L", (8, 8), 255), "solid", 0, color=(0, 255, 0, 255), target_id=photo.id)
    session.add_layer(band)
    session.layer_index = session.document.index_of(band.id)
    session.merge_down_active()
    assert [layer.name for layer in session.document.layers] == ["Background", "Photo"]
    assert session.document.layers[1].image.getpixel((1, 1)) == (0, 255, 0, 255)
    assert session.document.layers[0].image.getpixel((1, 1)) == (255, 255, 255, 255)
    session.undo()
    assert [layer.name for layer in session.document.layers] == ["Background", "Photo", "Mask"]
    session.layer_index = session.document.index_of(photo.id)
    session.merge_down_active()
    assert len(session.document.layers) == 1
    assert session.document.layers[0].image.getpixel((1, 1)) == (0, 255, 0, 255)
    session.undo()
    assert [layer.name for layer in session.document.layers] == ["Background", "Photo", "Mask"]
    assert session.document.layers[1].image.getpixel((6, 1)) == (0, 0, 255, 255)


def test_new_mask_targets_the_active_layer_and_stacks_above_older_ones():
    session = Session()
    session.new_canvas(32, 32, "white")
    base = session.document.layers[0]
    older = redaction_layer("Old", Image.new("L", (32, 32), 0), "blur", 4, target_id=base.id)
    session.add_layer(older)
    session.layer_index = 0
    session.options["mask_shape"] = "rect"
    session.options["mask_kind"] = "pixelate"
    tool = MaskTool()
    tool.pointer_down(session, PointerEvent(4, 4))
    tool.pointer_up(session, PointerEvent(12, 12))
    assert [layer.name for layer in session.document.layers] == ["Background", "Old", "Mask"]
    assert session.document.layers[2].target_id == base.id
    assert session.document.layers[1].target_id == base.id


def test_project_roundtrip_keeps_an_explicit_mask_target(tmp_path):
    from pixedit.io import load_project, save_project

    photo = raster_layer("Photo", Image.new("RGBA", (8, 8), (255, 0, 0, 255)))
    other = raster_layer("Other", Image.new("RGBA", (8, 8), (0, 0, 255, 255)))
    band = redaction_layer("Mask", Image.new("L", (8, 8), 255), "pixelate", 8, target_id=photo.id)
    doc = Document(8, 8, [photo, other, band])
    path = tmp_path / "masks.pix"
    save_project(doc, str(path))
    loaded = load_project(str(path))
    assert loaded.layers[2].target_id == loaded.layers[0].id
    assert loaded.layers[2].target_id != loaded.layers[1].id

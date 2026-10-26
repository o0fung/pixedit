"""Proxies, dirty rectangles, masked patches, and the open-size choice."""

from __future__ import annotations

import numpy as np
from PIL import Image
from PyQt6.QtWidgets import QApplication

from pixedit.adjust import color_grade
from pixedit.composite import composite, composite_region, redaction_padding
from pixedit.factory import raster_layer, redaction_layer
from pixedit.io import limit_longest_edge
from pixedit.model import Document, new_document
from pixedit.proxy import preview_scale, rotated_size
from pixedit.session import Session
from pixedit.tools.base import PointerEvent
from pixedit.tools.brush import BrushTool
from pixedit.tools.gradient import GradientTool
from pixedit.tools.shape import ShapeTool
from pixedit.tools.transform import TransformTool
from pixedit.ui.canvas import CanvasView
from pixedit.ui.dialogs import PdfImportDialog, RasterSizeDialog
from pixedit.ui.layers_panel import LayerPanel


def test_rotated_size_matches_pillow_and_large_pictures_scale_down():
    image = Image.new("RGBA", (40, 17), (255, 0, 0, 255))
    for angle in (0, 15, 90, 180, 270, -20):
        assert rotated_size(40, 17, angle) == image.rotate(angle, expand=True).size
    assert preview_scale(40, 17, 1) == 1
    assert preview_scale(4000, 3000, 1) < 1
    assert preview_scale(4000, 3000, 0.2) == 0.2


def test_limit_longest_edge_leaves_small_pictures_and_shrinks_large_ones():
    small = Image.new("RGBA", (30, 10), (1, 2, 3, 255))
    assert limit_longest_edge(small, None) is small
    assert limit_longest_edge(small, 100).size == (30, 10)
    limited = limit_longest_edge(Image.new("RGBA", (400, 200), (9, 9, 9, 255)), 100)
    assert max(limited.size) == 100
    assert limited.size == (100, 50)


def test_pdf_import_defaults_to_screen_dpi_and_raster_dialog_offers_a_cap():
    app = QApplication.instance() or QApplication([])
    assert PdfImportDialog().scale() == 1.0
    dialog = RasterSizeDialog(None, 5000, 3000)
    assert dialog.max_edge() is None
    dialog.choice.setCurrentIndex(1)
    assert dialog.max_edge() == 4096
    app.processEvents()


def test_color_drag_keeps_a_stand_in_until_commit_matches_the_full_grade():
    session = Session(new_document(400, 120, "white"))
    session.view_zoom = 0.05
    layer = session.active_layer()
    source = Image.new("RGBA", (400, 120), (40, 80, 120, 255))
    source.putpixel((10, 10), (200, 30, 30, 255))
    layer.image = source
    original = source.copy()
    session.options["color_brightness"] = 50
    session.preview_color_adjust()
    assert layer.image.getpixel((10, 10)) == original.getpixel((10, 10))
    assert layer.preview_source.width < layer.image.width
    session.commit_color_adjust()
    assert not hasattr(layer, "preview_source")
    expected = color_grade(original, brightness=50)
    assert layer.image.getpixel((10, 10)) == expected.getpixel((10, 10))
    assert layer.image.getpixel((0, 0)) == expected.getpixel((0, 0))


def test_gradient_stand_in_commits_the_same_pixels_as_a_full_draw():
    def drag(zoom: float):
        session = Session(new_document(180, 100, "white"))
        session.view_zoom = zoom
        session.foreground = (255, 0, 0, 255)
        session.options["gradient_color"] = (0, 0, 255, 255)
        session.options["gradient_kind"] = "linear"
        tool = GradientTool()
        tool.pointer_down(session, PointerEvent(8, 50))
        tool.pointer_move(session, PointerEvent(160, 50))
        stand_in = getattr(tool._layer, "preview_source", None)
        tool.pointer_up(session, PointerEvent(160, 50))
        return np.array(session.active_layer().image), None if stand_in is None else stand_in.size

    full, full_stand_in = drag(1.0)
    proxy, proxy_stand_in = drag(0.05)
    assert full_stand_in is None
    assert proxy_stand_in is not None and proxy_stand_in[0] < 180
    assert np.array_equal(full, proxy)


def test_transform_release_matches_a_full_resample():
    def drag(zoom: float):
        image = Image.new("RGBA", (300, 80), (20, 40, 200, 255))
        for x in range(0, 300, 2):
            image.putpixel((x, 20), (255, 220, 0, 255))
        layer = raster_layer("Photo", image, 10, 10)
        session = Session(Document(700, 300, [layer]))
        tool = TransformTool()
        event = lambda x, y: PointerEvent(x, y, zoom=zoom)
        tool.pointer_down(session, event(310, 90))
        tool.pointer_move(session, event(460, 150))
        tool.pointer_up(session, event(460, 150))
        return np.array(layer.image), (layer.x, layer.y)

    assert drag(1.0)[1] == drag(0.05)[1]
    assert np.array_equal(drag(1.0)[0], drag(0.05)[0])


def test_shape_drag_reports_the_shape_box():
    session = Session(new_document(80, 60, "transparent"))
    session.foreground = (0, 255, 0, 255)
    session.options["shape_kind"] = "rect"
    session.options["shape_filled"] = True
    session.options["stroke_width"] = 2
    seen: list[tuple[int, int, int, int] | None] = []

    def listen(kind: str) -> None:
        if kind == "preview":
            seen.append(session.preview_bounds)

    session.add_listener(listen)
    tool = ShapeTool()
    tool.pointer_down(session, PointerEvent(10, 8))
    tool.pointer_move(session, PointerEvent(40, 30))
    assert seen[-1] is not None
    x0, y0, x1, y1 = seen[-1]
    assert x1 - x0 < session.document.width
    assert y1 - y0 < session.document.height
    assert session.active_layer().image.getpixel((20, 18))[1] > 200


def test_opacity_drag_limits_the_preview_to_the_layer():
    app = QApplication.instance() or QApplication([])
    session = Session(new_document(30, 20, "white"))
    spot = raster_layer("Spot", Image.new("RGBA", (8, 6), (255, 0, 0, 255)), 4, 3)
    session.add_layer(spot)
    panel = LayerPanel(session, lambda: None)
    seen: list[tuple[int, int, int, int] | None] = []

    def listen(kind: str) -> None:
        if kind == "preview":
            seen.append(session.preview_bounds)

    session.add_listener(listen)
    panel._opacity_before = spot.opacity
    panel._opacity_moved(40)
    assert seen[-1] == (4, 3, 12, 9)
    app.processEvents()


def test_masked_region_matches_the_full_flatten():
    image = Image.new("RGBA", (48, 36), (0, 0, 0, 255))
    for y in range(36):
        for x in range(48):
            image.putpixel((x, y), ((x * 5) % 256, (y * 7) % 256, 80, 255))
    mask = Image.new("L", (48, 36), 0)
    mask.paste(255, (8, 6, 30, 24))
    for kind, amount in (("blur", 4), ("pixelate", 8), ("solid", 12), ("sharpen", 40)):
        photo = raster_layer("Photo", image.copy())
        doc = Document(
            48,
            36,
            [photo, redaction_layer("Mask", mask.copy(), kind, amount, target_id=photo.id)],
        )
        assert redaction_padding(doc) is not None
        box = (10, 8, 28, 22)
        assert np.array_equal(
            np.array(composite_region(doc, box)),
            np.array(composite(doc))[box[1] : box[3], box[0] : box[2]],
        )
    noisy = raster_layer("Photo", image.copy())
    noise_doc = Document(
        48,
        36,
        [noisy, redaction_layer("Noise", mask.copy(), "noise", 80, seed=3, target_id=noisy.id)],
    )
    assert redaction_padding(noise_doc) is None


def test_brush_patch_with_a_blur_mask_skips_a_second_full_flatten(monkeypatch):
    app = QApplication.instance() or QApplication([])
    session = Session(new_document(64, 48, "white"))
    photo = session.active_layer()
    mask = Image.new("L", (64, 48), 0)
    mask.paste(255, (4, 4, 40, 36))
    session.document.layers.append(redaction_layer("Blur", mask, "blur", 3, target_id=photo.id))
    canvas = CanvasView(session, lambda: None, lambda *_args: None)
    canvas.resize(400, 300)
    canvas.zoom = 1.0
    calls = {"n": 0}
    import pixedit.ui.canvas as canvas_mod

    real = canvas_mod.composite

    def spy(document):
        calls["n"] += 1
        return real(document)

    monkeypatch.setattr(canvas_mod, "composite", spy)

    def on_note(kind: str) -> None:
        if kind == "preview":
            canvas.invalidate(session.preview_bounds)
        elif kind == "change":
            canvas.invalidate()

    session.add_listener(on_note)
    canvas.show()
    app.processEvents()
    baseline = calls["n"]
    assert baseline >= 1
    session.foreground = (0, 0, 255, 255)
    session.options["brush_size"] = 6
    session.options["brush_hardness"] = 100
    session.options["brush_opacity"] = 100
    tool = BrushTool()
    tool.pointer_down(session, PointerEvent(20, 18))
    tool.pointer_move(session, PointerEvent(36, 22))
    assert calls["n"] == baseline
    tool.pointer_cancel(session)
    canvas.close()
    app.processEvents()

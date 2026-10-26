"""Zoomed-out display stays off the full-document flatten."""

from __future__ import annotations

import numpy as np
from PIL import Image
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication

from pixedit.composite import composite, composite_view
from pixedit.factory import raster_layer, redaction_layer
from pixedit.model import Document, new_document
from pixedit.proxy import PREVIEW_MAX_EDGE
from pixedit.session import Session
from pixedit.tools.base import PointerEvent
from pixedit.tools.brush import BrushTool
from pixedit.tools.hand import HandTool
from pixedit.ui.canvas import CanvasView, pil_to_pixmap


def test_composite_view_matches_a_box_downsample_and_full_scale_blend():
    base = Image.new("RGBA", (8, 4), (255, 0, 0, 255))
    spot = raster_layer("Spot", Image.new("RGBA", (4, 4), (0, 0, 255, 255)), 2, 0)
    spot.blend_mode = "darken"
    document = Document(8, 4, [raster_layer("Base", base), spot])

    view = np.array(composite_view(document, (0, 0, 8, 4), 0.5))
    expected = np.array(composite(document).resize((4, 2), Image.Resampling.BOX))
    assert view.shape == expected.shape
    assert np.array_equal(view, expected)

    cropped = np.array(composite_view(document, (2, 0, 6, 4), 0.5))
    cropped_expected = np.array(composite(document).crop((2, 0, 6, 4)).resize((2, 2), Image.Resampling.BOX))
    assert np.array_equal(cropped, cropped_expected)

    full = np.array(composite_view(document, (0, 0, 8, 4), 1))
    assert np.array_equal(full, np.array(composite(document)))


def test_zoomed_out_paint_skips_full_composite(monkeypatch):
    app = QApplication.instance() or QApplication([])
    session = Session(new_document(400, 300, "white"))
    session.document.layers[0].image.putpixel((8, 8), (255, 0, 0, 255))
    canvas = CanvasView(session, lambda: None, lambda *_args: None)
    canvas.resize(220, 180)
    canvas.zoom = 0.25
    canvas.offset_x = 8
    canvas.offset_y = 8
    calls = {"n": 0}
    import pixedit.ui.canvas as canvas_mod

    real = canvas_mod.composite

    def spy(document):
        calls["n"] += 1
        return real(document)

    monkeypatch.setattr(canvas_mod, "composite", spy)
    canvas.show()
    app.processEvents()
    assert calls["n"] == 0
    assert canvas._pixmap_mode == "view"
    assert canvas._pixmap is not None
    assert canvas._pixmap.width() < session.document.width
    assert canvas._pixmap.height() < session.document.height

    canvas.offset_x = -40
    canvas.repaint()
    assert calls["n"] == 0
    assert canvas._pixmap_mode == "view"

    canvas.zoom = 1.0
    canvas.repaint()
    assert calls["n"] == 1
    assert canvas._pixmap_mode == "full"
    assert canvas._pixmap.width() == session.document.width

    mask = Image.new("L", (session.document.width, session.document.height), 0)
    mask.paste(255, (0, 0, 12, 12))
    session.document.layers.append(
        redaction_layer("Mask", mask, "pixelate", 4, target_id=session.document.layers[0].id)
    )
    canvas.zoom = 0.25
    canvas.invalidate()
    app.processEvents()
    assert calls["n"] == 2
    assert canvas._pixmap_mode == "full"
    canvas.close()
    app.processEvents()


def test_letterbox_of_a_large_page_skips_full_composite(monkeypatch):
    app = QApplication.instance() or QApplication([])
    session = Session(new_document(800, 600, "white"))
    canvas = CanvasView(session, lambda: None, lambda *_args: None)
    canvas.resize(320, 240)
    canvas.letterbox = True
    calls = {"n": 0}
    import pixedit.ui.canvas as canvas_mod

    real = canvas_mod.composite

    def spy(document):
        calls["n"] += 1
        return real(document)

    monkeypatch.setattr(canvas_mod, "composite", spy)
    canvas.show()
    app.processEvents()
    assert canvas.zoom < 1
    assert calls["n"] == 0
    assert canvas._pixmap_mode == "view"
    canvas.close()
    app.processEvents()


def test_brush_patch_matches_full_flatten(monkeypatch):
    app = QApplication.instance() or QApplication([])
    session = Session(new_document(80, 60, "white"))
    canvas = CanvasView(session, lambda: None, lambda *_args: None)
    canvas.resize(400, 320)
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
    assert canvas._pixmap_mode == "full"
    baseline = calls["n"]
    assert baseline >= 1

    session.foreground = (0, 0, 255, 255)
    session.options["brush_size"] = 8
    session.options["brush_hardness"] = 100
    session.options["brush_opacity"] = 100
    tool = BrushTool()
    tool.pointer_down(session, PointerEvent(24, 24))
    tool.pointer_move(session, PointerEvent(46, 28))
    assert calls["n"] == baseline
    flat = pil_to_pixmap(composite(session.document))
    assert _rgba(canvas._pixmap).shape == _rgba(flat).shape
    assert np.array_equal(_rgba(canvas._pixmap), _rgba(flat))
    tool.pointer_cancel(session)
    canvas.close()
    app.processEvents()


def test_pan_of_a_large_zoomed_out_page_reuses_one_low_res_plate(monkeypatch):
    """Moves slide one stand-in. Release rebuilds the screen-sized view once."""
    app = QApplication.instance() or QApplication([])
    session = Session(new_document(1600, 1200, "white"))
    canvas = _hand_canvas(session)
    canvas.resize(480, 360)
    canvas.zoom = 0.2
    canvas.offset_x = 30
    canvas.offset_y = 30
    canvas.show()
    app.processEvents()
    assert canvas._pixmap_mode == "view"
    calls = _spy_flatten(monkeypatch)

    _pointer(canvas, "press", 40, 40)
    assert calls["composite"] == 0
    assert calls["view"] == 1
    assert canvas._pan_proxy is not None
    assert max(canvas._pan_proxy.width(), canvas._pan_proxy.height()) <= PREVIEW_MAX_EDGE
    _pointer(canvas, "move", 70, 55)
    canvas.repaint()
    _pointer(canvas, "move", 100, 70)
    canvas.repaint()
    assert calls["composite"] == 0
    assert calls["view"] == 1
    assert canvas._panning

    _pointer(canvas, "release", 100, 70)
    canvas.repaint()
    app.processEvents()
    assert calls["composite"] == 0
    assert calls["view"] == 2
    assert canvas._pan_proxy is None
    assert canvas._pixmap_mode == "view"
    assert canvas._pixmap is not None
    assert max(canvas._pixmap.width(), canvas._pixmap.height()) < PREVIEW_MAX_EDGE
    canvas.close()
    app.processEvents()


def test_pan_at_full_zoom_keeps_the_sharp_plate(monkeypatch):
    """A full-page cache is scaled for the drag and restored without flattening again."""
    app = QApplication.instance() or QApplication([])
    session = Session(new_document(1600, 1200, "white"))
    canvas = _hand_canvas(session)
    canvas.resize(480, 360)
    canvas.zoom = 1.0
    canvas.show()
    app.processEvents()
    assert canvas._pixmap_mode == "full"
    sharp = canvas._pixmap
    calls = _spy_flatten(monkeypatch)

    _pointer(canvas, "press", 40, 40)
    _pointer(canvas, "move", 90, 70)
    canvas.repaint()
    _pointer(canvas, "move", 140, 100)
    canvas.repaint()
    assert calls["composite"] == 0
    assert calls["view"] == 0
    assert canvas._pixmap is sharp
    assert canvas._pan_proxy is not None
    assert max(canvas._pan_proxy.width(), canvas._pan_proxy.height()) <= PREVIEW_MAX_EDGE

    _pointer(canvas, "release", 140, 100)
    canvas.repaint()
    app.processEvents()
    assert calls["composite"] == 0
    assert calls["view"] == 0
    assert canvas._pixmap is sharp
    assert canvas._pixmap_mode == "full"
    assert canvas._pan_proxy is None
    canvas.close()
    app.processEvents()


def test_pan_of_a_small_page_does_not_use_a_stand_in():
    """A page that already fits the preview cap stays on its sharp cache."""
    app = QApplication.instance() or QApplication([])
    session = Session(new_document(PREVIEW_MAX_EDGE, 600, "white"))
    canvas = _hand_canvas(session)
    canvas.resize(400, 300)
    canvas.zoom = 0.5
    canvas.offset_x = 16
    canvas.offset_y = 16
    canvas.show()
    app.processEvents()
    assert canvas._pixmap_mode == "view"
    mode = canvas._pixmap_mode

    _pointer(canvas, "press", 30, 30)
    _pointer(canvas, "move", 80, 60)
    canvas.repaint()
    assert canvas._pan_proxy is None
    assert canvas._panning
    assert canvas._pixmap_mode == mode

    _pointer(canvas, "release", 80, 60)
    canvas.repaint()
    app.processEvents()
    assert canvas._pan_proxy is None
    assert canvas._pixmap_mode == mode
    canvas.close()
    app.processEvents()


def _hand_canvas(session) -> CanvasView:
    return CanvasView(session, lambda: HandTool(), lambda *_args: None)


def _spy_flatten(monkeypatch):
    import pixedit.ui.canvas as canvas_mod

    calls = {"composite": 0, "view": 0}
    real_composite = canvas_mod.composite
    real_view = canvas_mod.composite_view

    def spy_composite(document):
        calls["composite"] += 1
        return real_composite(document)

    def spy_view(document, box, scale):
        calls["view"] += 1
        return real_view(document, box, scale)

    monkeypatch.setattr(canvas_mod, "composite", spy_composite)
    monkeypatch.setattr(canvas_mod, "composite_view", spy_view)
    return calls


def _pointer(widget, kind: str, x: float, y: float) -> None:
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtWidgets import QApplication

    pos = QPointF(x, y)
    if kind == "press":
        event = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    elif kind == "move":
        event = QMouseEvent(
            QEvent.Type.MouseMove,
            pos,
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    else:
        event = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
    QApplication.sendEvent(widget, event)


def _rgba(pixmap) -> np.ndarray:
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    width = image.width()
    height = image.height()
    raw = image.bits()
    raw.setsize(image.sizeInBytes())
    rows = np.frombuffer(raw, dtype=np.uint8).reshape((height, image.bytesPerLine()))
    return np.ascontiguousarray(rows[:, : width * 4]).reshape((height, width, 4))

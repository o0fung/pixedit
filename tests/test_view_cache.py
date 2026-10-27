"""Zoomed-out display stays off the full-document flatten."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from PyQt6.QtCore import QRectF, Qt
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
from pixedit.tools.zoom import ZoomTool
from pixedit.ui.canvas import (
    PAN_CROP_MAX_EDGE,
    CanvasView,
    checker_loop_rect,
    pil_to_pixmap,
    plate_visible_source,
    ruler_tick_span,
)


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
    assert canvas._pan_proxy_box is None
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
    """The drag shows a crop of the sharp plate and does not flatten again."""
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
    box = canvas._pan_proxy_box
    assert box is not None
    visible = canvas._visible_document_box()
    assert visible is not None
    assert box[0] <= visible[0] and box[1] <= visible[1]
    assert box[2] >= visible[2] and box[3] >= visible[3]
    assert canvas._pan_proxy.width() == int(box[2] - box[0])
    assert canvas._pan_proxy.height() == int(box[3] - box[1])
    assert max(box[2] - box[0], box[3] - box[1]) <= PAN_CROP_MAX_EDGE

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


def test_zoom_drag_of_a_large_page_stays_on_one_low_res_plate(monkeypatch):
    """Scrubbing zoom scales the stand-in. Release rebuilds the sharp view once."""
    app = QApplication.instance() or QApplication([])
    session = Session(new_document(1600, 1200, "white"))
    canvas = _zoom_canvas(session)
    canvas.resize(480, 360)
    canvas.zoom = 0.2
    canvas.offset_x = 30
    canvas.offset_y = 30
    canvas.show()
    app.processEvents()
    assert canvas._pixmap_mode == "view"
    calls = _spy_flatten(monkeypatch)

    _pointer(canvas, "press", 80, 240)
    assert calls["view"] == 0
    assert canvas._pan_proxy is None
    _pointer(canvas, "move", 80, 80)
    canvas.repaint()
    assert calls["composite"] == 0
    assert calls["view"] == 1
    assert canvas._zoom_scrub
    assert canvas._pan_proxy is not None
    assert canvas._pan_proxy_box is None
    assert max(canvas._pan_proxy.width(), canvas._pan_proxy.height()) <= PREVIEW_MAX_EDGE
    assert canvas.zoom == pytest.approx(0.4)
    _pointer(canvas, "move", 80, 40)
    canvas.repaint()
    assert calls["view"] == 1
    assert canvas.zoom > 0.4

    _pointer(canvas, "release", 80, 40)
    canvas.repaint()
    app.processEvents()
    assert calls["composite"] == 0
    assert calls["view"] == 2
    assert canvas._pan_proxy is None
    assert not canvas._zooming
    assert canvas._pixmap_mode == "view"
    assert canvas.zoom > 0.4
    canvas.close()
    app.processEvents()


def test_zoom_drag_at_full_zoom_keeps_the_sharp_plate(monkeypatch):
    """Scrubbing above 100% scales the full plate and does not flatten again."""
    app = QApplication.instance() or QApplication([])
    session = Session(new_document(1600, 1200, "white"))
    canvas = _zoom_canvas(session)
    canvas.resize(480, 360)
    canvas.zoom = 1.0
    canvas.show()
    app.processEvents()
    assert canvas._pixmap_mode == "full"
    sharp = canvas._pixmap
    calls = _spy_flatten(monkeypatch)

    _pointer(canvas, "press", 80, 260)
    _pointer(canvas, "move", 80, 100)
    canvas.repaint()
    assert calls["composite"] == 0
    assert calls["view"] == 0
    assert canvas.zoom == pytest.approx(2.0)
    assert canvas._pixmap is sharp
    assert canvas._pan_proxy is not None
    assert canvas._pan_proxy_box is None
    assert max(canvas._pan_proxy.width(), canvas._pan_proxy.height()) <= PREVIEW_MAX_EDGE

    _pointer(canvas, "release", 80, 100)
    canvas.repaint()
    app.processEvents()
    assert calls["composite"] == 0
    assert calls["view"] == 0
    assert canvas._pixmap is sharp
    assert canvas._pixmap_mode == "full"
    assert canvas._pan_proxy is None
    assert canvas.zoom == pytest.approx(2.0)
    canvas.close()
    app.processEvents()


def test_zoom_drag_below_full_zoom_rebuilds_the_view(monkeypatch):
    """Dropping under 100% leaves the full plate and sharpens the screen view once."""
    app = QApplication.instance() or QApplication([])
    session = Session(new_document(1600, 1200, "white"))
    canvas = _zoom_canvas(session)
    canvas.resize(480, 360)
    canvas.zoom = 1.0
    canvas.show()
    app.processEvents()
    calls = _spy_flatten(monkeypatch)

    _pointer(canvas, "press", 80, 80)
    _pointer(canvas, "move", 80, 240)
    canvas.repaint()
    assert calls["composite"] == 0
    assert calls["view"] == 0
    assert canvas.zoom == pytest.approx(0.5)
    assert canvas._pixmap_mode == "full"

    _pointer(canvas, "release", 80, 240)
    canvas.repaint()
    app.processEvents()
    assert calls["composite"] == 0
    assert calls["view"] == 1
    assert canvas._pixmap_mode == "view"
    assert canvas._pan_proxy is None
    assert canvas.zoom == pytest.approx(0.5)
    canvas.close()
    app.processEvents()


def test_zoom_click_steps_without_a_stand_in(monkeypatch):
    """A click still steps the zoom. The stand-in is only for a drag."""
    app = QApplication.instance() or QApplication([])
    session = Session(new_document(1600, 1200, "white"))
    canvas = _zoom_canvas(session)
    canvas.resize(480, 360)
    canvas.zoom = 0.2
    canvas.offset_x = 30
    canvas.offset_y = 30
    canvas.show()
    app.processEvents()
    calls = _spy_flatten(monkeypatch)

    _pointer(canvas, "press", 80, 80)
    _pointer(canvas, "release", 80, 80)
    canvas.repaint()
    app.processEvents()
    assert canvas.zoom == pytest.approx(0.25)
    assert canvas._pan_proxy is None
    assert calls["composite"] == 0
    assert calls["view"] == 1

    _pointer(canvas, "press", 80, 80, alt=True)
    _pointer(canvas, "release", 80, 80)
    canvas.repaint()
    app.processEvents()
    assert canvas.zoom == pytest.approx(0.2)
    assert calls["view"] == 2
    canvas.close()
    app.processEvents()


def test_zoom_drag_of_a_small_page_does_not_use_a_stand_in():
    """A page at the preview cap scrubs on its sharp cache."""
    app = QApplication.instance() or QApplication([])
    session = Session(new_document(PREVIEW_MAX_EDGE, 600, "white"))
    canvas = _zoom_canvas(session)
    canvas.resize(400, 300)
    canvas.zoom = 0.5
    canvas.offset_x = 16
    canvas.offset_y = 16
    canvas.show()
    app.processEvents()
    assert canvas._pixmap_mode == "view"

    _pointer(canvas, "press", 40, 180)
    _pointer(canvas, "move", 40, 100)
    canvas.repaint()
    assert canvas._pan_proxy is None
    assert canvas._zoom_scrub
    assert canvas.zoom > 0.5
    assert canvas._pixmap_mode == "view"

    _pointer(canvas, "release", 40, 100)
    canvas.repaint()
    app.processEvents()
    assert canvas._pan_proxy is None
    assert canvas._pixmap_mode == "view"
    canvas.close()
    app.processEvents()


def test_close_zoom_pan_crops_the_reachable_page(monkeypatch):
    """A close zoom preloads the rectangle one window of travel can reach.

    The plate is a crop of the sharp page. Moves inside the window reuse it.
    A move that leaves the window builds one new crop, and the next small
    move keeps that crop.
    """
    app = QApplication.instance() or QApplication([])
    session = Session(new_document(2400, 1800, "white"))
    image = session.document.layers[0].image
    image.putpixel((1200, 900), (255, 0, 0, 255))
    image.putpixel((1, 1), (0, 255, 0, 255))
    canvas = _hand_canvas(session)
    canvas.resize(480, 360)
    canvas.zoom = 8
    _center_on(canvas, 1200, 900)
    canvas.show()
    app.processEvents()
    assert canvas.zoom == 8
    assert canvas._pixmap_mode == "full"
    calls = _spy_flatten(monkeypatch)

    _pointer(canvas, "press", 1, 1)
    assert calls["composite"] == 0
    assert calls["view"] == 0
    box = canvas._pan_proxy_box
    assert box is not None
    x0, y0, x1, y1 = box
    assert x1 - x0 < session.document.width / 4
    assert y1 - y0 < session.document.height / 4
    assert max(x1 - x0, y1 - y0) <= PAN_CROP_MAX_EDGE
    visible = canvas._visible_document_box()
    assert visible is not None
    assert x0 <= visible[0] and y0 <= visible[1] and x1 >= visible[2] and y1 >= visible[3]
    assert x0 > 2
    assert x0 <= 1200 < x1 and y0 <= 900 < y1
    crop = _rgba(canvas._pan_proxy)
    assert crop.shape[1] == int(x1 - x0)
    assert crop.shape[0] == int(y1 - y0)
    assert tuple(int(v) for v in crop[int(900 - y0), int(1200 - x0)]) == (255, 0, 0, 255)
    assert not np.any(np.all(crop == (0, 255, 0, 255), axis=-1))
    plate = canvas._pan_proxy

    # Corner to corner is the farthest drag that stays inside the window.
    _pointer(canvas, "move", canvas.width() - 1, canvas.height() - 1)
    canvas.repaint()
    assert canvas._pan_proxy is plate
    assert canvas._pan_proxy_box == box
    assert calls["view"] == 0
    assert calls["composite"] == 0

    _pointer(canvas, "move", -4000, 200)
    canvas.repaint()
    assert calls["view"] == 0
    assert calls["composite"] == 0
    moved = canvas._pan_proxy_box
    assert moved is not None and moved != box
    visible = canvas._visible_document_box()
    assert visible is not None
    assert moved[0] <= visible[0] and moved[1] <= visible[1]
    assert moved[2] >= visible[2] and moved[3] >= visible[3]
    rebuilt = canvas._pan_proxy

    _pointer(canvas, "move", -3980, 210)
    canvas.repaint()
    assert canvas._pan_proxy is rebuilt
    assert canvas._pan_proxy_box == moved
    assert calls["view"] == 0

    _pointer(canvas, "release", -3980, 210)
    canvas.repaint()
    app.processEvents()
    assert canvas._pan_proxy is None
    assert calls["composite"] == 0
    assert calls["view"] == 0
    assert canvas._pixmap_mode == "full"
    canvas.close()
    app.processEvents()


def test_close_zoom_pan_without_a_full_plate_flattens_the_crop_once(monkeypatch):
    """With no sharp page cached, press flattens the crop once, not the page."""
    app = QApplication.instance() or QApplication([])
    session = Session(new_document(2400, 1800, "white"))
    session.document.layers[0].image.putpixel((1200, 900), (0, 0, 255, 255))
    canvas = _hand_canvas(session)
    canvas.resize(480, 360)
    canvas.zoom = 8
    _center_on(canvas, 1200, 900)
    canvas.show()
    app.processEvents()
    canvas._pixmap = None
    canvas._dirty = True
    calls = _spy_flatten(monkeypatch)

    _pointer(canvas, "press", 200, 180)
    assert calls["composite"] == 0
    assert calls["view"] == 1
    box = canvas._pan_proxy_box
    assert box is not None
    x0, y0, x1, y1 = box
    assert x0 <= 1200 < x1 and y0 <= 900 < y1
    crop = _rgba(canvas._pan_proxy)
    assert tuple(int(v) for v in crop[int(900 - y0), int(1200 - x0)]) == (0, 0, 255, 255)

    _pointer(canvas, "move", 240, 190)
    canvas.repaint()
    assert calls["view"] == 1
    assert canvas._pan_proxy_box == box

    _pointer(canvas, "move", -4000, 190)
    canvas.repaint()
    assert calls["composite"] == 0
    assert calls["view"] == 2
    again = canvas._pan_proxy_box
    assert again is not None and again != box

    _pointer(canvas, "move", -3970, 200)
    canvas.repaint()
    assert calls["view"] == 2
    assert canvas._pan_proxy_box == again
    canvas.close()
    app.processEvents()


def test_checker_loop_stays_inside_the_widget_when_zoomed_in():
    """A close zoom walks the widget, not the page rectangle times zoom."""
    page = QRectF(-64000.0, -48000.0, 128000.0, 96000.0)
    widget = QRectF(22.0, 22.0, 458.0, 338.0)
    span = checker_loop_rect(page, widget, 12)
    assert span is not None
    left, top, right, bottom = span
    assert right - left <= widget.width() + 24
    assert bottom - top <= widget.height() + 24
    assert right - left < page.width() / 10
    assert bottom - top < page.height() / 10
    assert left <= widget.left()
    assert top <= widget.top()
    assert right >= widget.right()
    assert bottom >= widget.bottom()
    origin_x = int(page.left())
    origin_y = int(page.top())
    assert (left - origin_x) % 12 == 0
    assert (top - origin_y) % 12 == 0
    assert (right - origin_x) % 12 == 0
    assert (bottom - origin_y) % 12 == 0
    assert checker_loop_rect(QRectF(0, 0, 10, 10), QRectF(100, 100, 10, 10)) is None


def test_plate_visible_source_is_the_window_not_the_page():
    """A huge page rectangle contributes only the window's share of the plate."""
    dest = QRectF(-80000.0, -60000.0, 160000.0, 120000.0)
    window = QRectF(22.0, 22.0, 458.0, 338.0)
    pair = plate_visible_source(dest, window, 1024, 768)
    assert pair is not None
    visible, source = pair
    assert visible.left() == pytest.approx(window.left())
    assert visible.width() == pytest.approx(window.width())
    assert source.width() == pytest.approx(1024 * window.width() / dest.width())
    assert source.height() == pytest.approx(768 * window.height() / dest.height())
    assert source.width() < 1024 * 0.02
    assert plate_visible_source(QRectF(0, 0, 10, 10), QRectF(100, 100, 10, 10), 8, 8) is None


def test_ruler_ticks_stay_near_the_view():
    """A view in the middle of a large page does not walk ticks from zero."""
    start, end = ruler_tick_span(5000, 5300, 10)
    assert start == 5000
    assert end == 5300
    assert end - start < 1000
    start, end = ruler_tick_span(-15, 40, 10)
    assert start == -20
    assert end == 40


def test_zoom_drag_to_a_close_zoom_does_not_flatten_again(monkeypatch):
    """Moves after the stand-in exists only scale it, including a jump past 100%."""
    app = QApplication.instance() or QApplication([])
    session = Session(new_document(1600, 1200, "white"))
    canvas = _zoom_canvas(session)
    canvas.resize(480, 360)
    canvas.zoom = 0.2
    canvas.offset_x = 30
    canvas.offset_y = 30
    canvas.show()
    app.processEvents()
    calls = _spy_flatten(monkeypatch)

    _pointer(canvas, "press", 80, 400)
    _pointer(canvas, "move", 80, 80)
    canvas.repaint()
    assert calls["composite"] == 0
    assert calls["view"] == 1
    assert canvas._pan_proxy is not None
    _pointer(canvas, "move", 80, -400)
    canvas.repaint()
    assert calls["composite"] == 0
    assert calls["view"] == 1
    assert canvas.zoom > 1
    assert canvas._pan_proxy_box is None
    canvas.close()
    app.processEvents()


def test_space_hand_pans_instead_of_zooming():
    """Space is the hand. Left and right drags pan, and the zoom stays put."""
    app = QApplication.instance() or QApplication([])
    session = Session(new_document(400, 300, "white"))
    canvas = _zoom_canvas(session)
    canvas.resize(480, 360)
    canvas.zoom = 1.0
    canvas.show()
    app.processEvents()

    _pointer(canvas, "press", 80, 200)
    _pointer(canvas, "move", 80, 40)
    assert canvas.zoom == pytest.approx(2.0)
    _pointer(canvas, "release", 80, 40)
    assert not canvas._zooming

    held = canvas.zoom
    offset = canvas.offset_x
    _space(canvas, True)
    _pointer(canvas, "press", 40, 40)
    _pointer(canvas, "move", 120, 70)
    assert canvas.zoom == held
    assert canvas._panning
    assert not canvas._zooming
    assert canvas.offset_x != offset
    _pointer(canvas, "release", 120, 70)
    assert not canvas._panning

    offset = canvas.offset_y
    _pointer(canvas, "press", 40, 40, button=Qt.MouseButton.RightButton)
    _pointer(canvas, "move", 70, 110, button=Qt.MouseButton.RightButton)
    assert canvas.zoom == held
    assert canvas._panning
    assert not canvas._zooming
    assert canvas.offset_y != offset
    _pointer(canvas, "release", 70, 110, button=Qt.MouseButton.RightButton)
    assert not canvas._panning
    assert canvas.zoom == held
    canvas.close()
    app.processEvents()


def test_space_during_a_zoom_drag_stops_the_zoom():
    """Pressing Space mid-drag keeps the zoom reached and pans the rest of the hold."""
    app = QApplication.instance() or QApplication([])
    session = Session(new_document(400, 300, "white"))
    canvas = _zoom_canvas(session)
    canvas.resize(480, 360)
    canvas.zoom = 1.0
    canvas.show()
    app.processEvents()

    _pointer(canvas, "press", 80, 200)
    _pointer(canvas, "move", 80, 40)
    assert canvas.zoom == pytest.approx(2.0)
    reached = canvas.zoom
    _space(canvas, True)
    assert not canvas._zooming
    assert canvas.zoom == reached
    _pointer(canvas, "move", 80, 20)
    parked = canvas.offset_x
    _pointer(canvas, "move", 160, 20)
    assert canvas.zoom == reached
    assert canvas._panning
    assert canvas.offset_x == pytest.approx(parked + 80)
    _space(canvas, False)
    assert canvas._panning
    parked = canvas.offset_x
    _pointer(canvas, "move", 200, 20)
    assert canvas._panning
    assert canvas.offset_x == pytest.approx(parked + 40)
    assert canvas.zoom == reached
    assert not canvas._zooming
    _pointer(canvas, "release", 200, 20)
    assert not canvas._panning
    canvas.close()
    app.processEvents()


def test_brush_stroke_keeps_every_mouse_move(monkeypatch):
    """A burst of moves is stored. One tick paints the bend a cursor poll would skip."""
    import pixedit.ui.canvas as canvas_mod

    app = QApplication.instance() or QApplication([])
    session = Session(new_document(160, 120, "white"))
    tool = BrushTool()
    canvas = CanvasView(session, lambda: tool, lambda *_args: None)
    canvas.resize(480, 360)
    canvas.show()
    app.processEvents()
    canvas.zoom = 1.0
    canvas.offset_x = 24.0
    canvas.offset_y = 24.0

    def widget_at(x: float, y: float) -> tuple[float, float]:
        ruler = canvas.ruler()
        return ruler + canvas.offset_x + x * canvas.zoom, ruler + canvas.offset_y + y * canvas.zoom

    # Right angle. The corner is not on the straight run from the start to the end.
    start = (20.0, 40.0)
    mid = (45.0, 40.0)
    bend = (70.0, 40.0)
    end = (70.0, 90.0)

    _pointer(canvas, "press", *widget_at(*start))
    canvas._stroke_timer.stop()
    assert tool._raw == [start]
    assert canvas._stroke_points == []

    _pointer(canvas, "move", *widget_at(*mid))
    _pointer(canvas, "move", *widget_at(*bend))
    _pointer(canvas, "move", *widget_at(*end))
    assert canvas._stroke_points == [mid, bend, end]
    assert tool._raw == [start]

    monkeypatch.setattr(canvas_mod.QApplication, "mouseButtons", lambda: Qt.MouseButton.LeftButton)
    canvas._on_stroke_sample()
    assert tool._raw == [start, mid, bend, end]
    assert canvas._stroke_points == []
    assert tool.stroke_live()

    canvas._stroke_timer.stop()
    tool.pointer_cancel(session)
    canvas.close()
    app.processEvents()


def _center_on(canvas: CanvasView, doc_x: float, doc_y: float) -> None:
    """Put a document point at the widget center."""
    ruler = canvas.ruler()
    canvas.offset_x = canvas.width() / 2 - ruler - doc_x * canvas.zoom
    canvas.offset_y = canvas.height() / 2 - ruler - doc_y * canvas.zoom


def _hand_canvas(session) -> CanvasView:
    return CanvasView(session, lambda: HandTool(), lambda *_args: None)


def _zoom_canvas(session) -> CanvasView:
    return CanvasView(session, lambda: ZoomTool(), lambda *_args: None)


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


def _pointer(widget, kind: str, x: float, y: float, *, alt: bool = False, button=None) -> None:
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtWidgets import QApplication

    if button is None:
        button = Qt.MouseButton.LeftButton
    pos = QPointF(x, y)
    mods = Qt.KeyboardModifier.AltModifier if alt else Qt.KeyboardModifier.NoModifier
    if kind == "press":
        event = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            pos,
            button,
            button,
            mods,
        )
    elif kind == "move":
        event = QMouseEvent(
            QEvent.Type.MouseMove,
            pos,
            Qt.MouseButton.NoButton,
            button,
            mods,
        )
    else:
        event = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            pos,
            button,
            Qt.MouseButton.NoButton,
            mods,
        )
    QApplication.sendEvent(widget, event)


def _space(widget, down: bool) -> None:
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QKeyEvent

    kind = QEvent.Type.KeyPress if down else QEvent.Type.KeyRelease
    QApplication.sendEvent(widget, QKeyEvent(kind, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier))


def _rgba(pixmap) -> np.ndarray:
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    width = image.width()
    height = image.height()
    raw = image.bits()
    raw.setsize(image.sizeInBytes())
    rows = np.frombuffer(raw, dtype=np.uint8).reshape((height, image.bytesPerLine()))
    return np.ascontiguousarray(rows[:, : width * 4]).reshape((height, width, 4))

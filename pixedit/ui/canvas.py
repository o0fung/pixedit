"""Zoomable canvas. Document pixels stay in the session; this widget only views them."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image
from PyQt6.QtCore import QPoint, Qt, QRectF, QTimer
from PyQt6.QtGui import QColor, QCursor, QImage, QPainter, QPen, QPixmap, QPolygon
from PyQt6.QtWidgets import QWidget

from pixedit.composite import composite
from pixedit.tools.base import PointerEvent

_CURSORS = {
    "cross": Qt.CursorShape.CrossCursor,
    "hand": Qt.CursorShape.OpenHandCursor,
    "size": Qt.CursorShape.SizeAllCursor,
    "ibeam": Qt.CursorShape.IBeamCursor,
    "arrow": Qt.CursorShape.ArrowCursor,
}

# A brush larger than this, or zoomed far in, would make a pixmap the window
# system will not use as a cursor. The ring is clamped to this many logical pixels.
_MAX_BRUSH_CURSOR = 256


def brush_circle_cursor(screen_diameter: float, *, device_pixel_ratio: float = 1.0) -> QCursor:
    """Circle cursor whose diameter is the brush size in screen pixels.

    The white ring is the brush diameter. A black ring one pixel outside it
    keeps the outline visible on a light canvas. The hotspot is the center.
    """
    logical = int(round(max(1.0, float(screen_diameter))))
    logical = min(_MAX_BRUSH_CURSOR, logical)
    dpr = max(1.0, float(device_pixel_ratio))
    physical = max(1, int(round(logical * dpr)))
    # Pad for the contrast ring and a one-pixel margin so the pen is not clipped.
    pad = int(math.ceil(2 * dpr)) + 1
    size = physical + pad * 2
    if size % 2 == 0:
        size += 1
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, physical >= 12)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    center = size / 2.0

    def ring(diameter: float, color: QColor) -> None:
        pen = QPen(color)
        pen.setWidthF(max(1.0, dpr))
        painter.setPen(pen)
        left = center - diameter / 2.0
        painter.drawEllipse(QRectF(left, left, diameter, diameter))

    ring(physical + max(1.0, dpr), QColor(0, 0, 0, 230))
    ring(float(physical), QColor(255, 255, 255, 240))
    # A tiny ring is hard to aim with. A short crosshair marks the center
    # without changing the circle's diameter.
    if physical < int(8 * dpr):
        arm = max(3.0, 4.0 * dpr)
        pen = QPen(QColor(255, 255, 255, 230))
        pen.setWidthF(max(1.0, dpr))
        painter.setPen(pen)
        painter.drawLine(int(center - arm), int(center), int(center + arm), int(center))
        painter.drawLine(int(center), int(center - arm), int(center), int(center + arm))
    painter.end()
    pixmap.setDevicePixelRatio(dpr)
    hot = int(round(center / dpr))
    return QCursor(pixmap, hot, hot)


def pil_to_pixmap(image: Image.Image) -> QPixmap:
    rgba = image.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimage = QImage(data, rgba.width, rgba.height, rgba.width * 4, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimage.copy())


class CanvasView(QWidget):
    def __init__(self, session, tool_getter, on_status) -> None:
        super().__init__()
        self.session = session
        self.tool_getter = tool_getter
        self.on_status = on_status
        self.zoom = 1.0
        self.offset_x = 24.0
        self.offset_y = 24.0
        self._pixmap: QPixmap | None = None
        self._dirty = True
        self._space = False
        self._panning = False
        self._pan_last = None
        self._selection_id = None
        self._selection_pixmap: QPixmap | None = None
        self._edge_id = None
        self._edge_xs: np.ndarray | None = None
        self._edge_ys: np.ndarray | None = None
        self._ant_phase = 0
        # March the outline without rebuilding the composite. The edge is a
        # list of pixels, not a second full-canvas buffer, so a large photo
        # does not allocate another RGBA frame on every tick.
        self._ant_timer = QTimer(self)
        self._ant_timer.setInterval(120)
        self._ant_timer.timeout.connect(self._advance_ants)
        self._cursor_name = "arrow"
        self._brush_cursor_key: tuple[int, int] | None = None
        self._brush_cursor: QCursor | None = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(320, 240)

    def invalidate(self) -> None:
        self._dirty = True
        self._selection_id = None
        self.update()

    def ruler(self) -> int:
        return 22 if self.session.option("rulers", True) else 0

    def set_tool_cursor(self, name: str) -> None:
        self._cursor_name = name
        self._apply_cursor()

    def refresh_tool_cursor(self) -> None:
        """Rebuild the brush circle after a size or zoom change."""
        self._brush_cursor_key = None
        self._apply_cursor()

    def _apply_cursor(self) -> None:
        # Pan grabs the cursor for the gesture. Space shows an open hand
        # until the key is released, then the tool cursor comes back.
        if self._panning:
            return
        if self._space:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        tool = self.tool_getter()
        if tool is not None and tool.id in {"brush", "eraser"}:
            self.setCursor(self._circle_cursor())
            return
        name = self._cursor_name
        if tool is not None and name in {"", "brush"}:
            name = tool.cursor
        self.setCursor(_CURSORS.get(name, Qt.CursorShape.ArrowCursor))

    def _circle_cursor(self) -> QCursor:
        diameter = float(self.session.option("brush_size", 24)) * float(self.zoom)
        dpr = float(self.devicePixelRatioF() or 1.0)
        key = (int(round(diameter * 10)), int(round(dpr * 100)))
        if self._brush_cursor is None or self._brush_cursor_key != key:
            self._brush_cursor = brush_circle_cursor(diameter, device_pixel_ratio=dpr)
            self._brush_cursor_key = key
        return self._brush_cursor

    def widget_to_canvas(self, wx: float, wy: float) -> tuple[float, float]:
        ruler = self.ruler()
        return (wx - ruler - self.offset_x) / self.zoom, (wy - ruler - self.offset_y) / self.zoom

    def zoom_at(self, wx: float, wy: float, factor: float) -> None:
        before_x, before_y = self.widget_to_canvas(wx, wy)
        self.zoom = min(32.0, max(0.02, self.zoom * factor))
        ruler = self.ruler()
        self.offset_x = (wx - ruler) - before_x * self.zoom
        self.offset_y = (wy - ruler) - before_y * self.zoom
        self.session.view_zoom = self.zoom
        self.update()
        self._report_status(wx, wy)
        self.refresh_tool_cursor()

    def zoom_100(self) -> None:
        center = self.widget_to_canvas(self.width() / 2, self.height() / 2)
        self.zoom = 1.0
        ruler = self.ruler()
        self.offset_x = (self.width() / 2 - ruler) - center[0]
        self.offset_y = (self.height() / 2 - ruler) - center[1]
        self.session.view_zoom = self.zoom
        self.update()
        self.refresh_tool_cursor()

    def fit(self) -> None:
        doc = self.session.document
        ruler = self.ruler()
        avail_w = max(1.0, self.width() - ruler - 36)
        avail_h = max(1.0, self.height() - ruler - 36)
        self.zoom = min(32.0, max(0.02, min(avail_w / max(1, doc.width), avail_h / max(1, doc.height))))
        self.offset_x = (self.width() - ruler - doc.width * self.zoom) / 2
        self.offset_y = (self.height() - ruler - doc.height * self.zoom) / 2
        self.session.view_zoom = self.zoom
        self.update()
        self.refresh_tool_cursor()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101318"))
        self._ensure_pixmap()
        ruler = self.ruler()
        doc = self.session.document
        target = QRectF(
            ruler + self.offset_x,
            ruler + self.offset_y,
            doc.width * self.zoom,
            doc.height * self.zoom,
        )
        painter.setClipRect(ruler, ruler, max(0, self.width() - ruler), max(0, self.height() - ruler))
        self._paint_checker(painter, target)
        if self._pixmap is not None and not self._pixmap.isNull():
            painter.drawPixmap(target, self._pixmap, QRectF(0, 0, self._pixmap.width(), self._pixmap.height()))
        self._paint_selection(painter, target)
        self._paint_ants(painter)
        self._paint_overlays(painter)
        painter.setClipping(False)
        pen = QPen(QColor("#3ec6ff"))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawRect(target.adjusted(0, 0, -1, -1))
        if ruler:
            self._paint_rulers(painter, ruler)

    def _ensure_pixmap(self) -> None:
        if not self._dirty and self._pixmap is not None:
            return
        image = composite(self.session.document)
        self._pixmap = pil_to_pixmap(image)
        self._dirty = False
        selection = self.session.document.selection
        if selection is None:
            self._selection_pixmap = None
            self._selection_id = None
            self._sync_ants(None)
            return
        # Rebuild only when the selection object changes. Commands replace it,
        # so identity is a reliable cache key.
        if id(selection) != self._selection_id or self._selection_pixmap is None:
            self._selection_pixmap = pil_to_pixmap(_selection_tint(selection))
            self._selection_id = id(selection)
        self._sync_ants(selection)

    def _paint_checker(self, painter: QPainter, target: QRectF) -> None:
        painter.fillRect(target, QColor("#1b2026"))
        size = 12
        c2 = QColor("#2a3038")
        left = int(math.floor(target.left()))
        top = int(math.floor(target.top()))
        right = int(math.ceil(target.right()))
        bottom = int(math.ceil(target.bottom()))
        origin_x = int(target.left())
        origin_y = int(target.top())
        y = top
        row = 0
        while y < bottom:
            x = left
            col = 0
            while x < right:
                if ((x - origin_x) // size + (y - origin_y) // size) % 2 == 0:
                    painter.fillRect(x, y, size, size, c2)
                x += size
                col += 1
            y += size
            row += 1

    def _paint_selection(self, painter: QPainter, target: QRectF) -> None:
        if self._selection_pixmap is None:
            return
        painter.drawPixmap(
            target,
            self._selection_pixmap,
            QRectF(0, 0, self._selection_pixmap.width(), self._selection_pixmap.height()),
        )

    def _advance_ants(self) -> None:
        if self.session.document.selection is None or self._edge_xs is None:
            self._ant_timer.stop()
            return
        self._ant_phase = (self._ant_phase + 1) % 8
        self.update()

    def _sync_ants(self, selection: Image.Image | None) -> None:
        if selection is None:
            self._edge_xs = None
            self._edge_ys = None
            self._edge_id = None
            self._ant_timer.stop()
            return
        if id(selection) == self._edge_id and self._edge_xs is not None:
            if not self._ant_timer.isActive():
                self._ant_timer.start()
            return
        self._edge_id = id(selection)
        mask = np.asarray(selection.convert("L")) > 0
        # One-pixel boundary: a pixel is an edge when any 4-neighbor is outside.
        eroded = mask.copy()
        if mask.shape[0] > 1:
            eroded[1:, :] &= mask[:-1, :]
            eroded[:-1, :] &= mask[1:, :]
        if mask.shape[1] > 1:
            eroded[:, 1:] &= mask[:, :-1]
            eroded[:, :-1] &= mask[:, 1:]
        ys, xs = np.nonzero(mask & ~eroded)
        # A noisy wand outline can be huge. Keep enough points to read the
        # shape; the tint underneath still shows the full region.
        if len(xs) > 12000:
            step = int(math.ceil(len(xs) / 12000))
            xs = xs[::step]
            ys = ys[::step]
        self._edge_xs = xs
        self._edge_ys = ys
        if len(xs) == 0:
            self._ant_timer.stop()
            return
        if not self._ant_timer.isActive():
            self._ant_timer.start()

    def _paint_ants(self, painter: QPainter) -> None:
        if self._edge_xs is None or self._edge_ys is None or len(self._edge_xs) == 0:
            return
        light = ((self._edge_xs + self._edge_ys + self._ant_phase) & 7) < 4
        ruler = self.ruler()
        painter.save()
        painter.translate(ruler + self.offset_x, ruler + self.offset_y)
        painter.scale(self.zoom, self.zoom)
        # At least one screen pixel, and one image pixel when zoomed in.
        pen_w = max(1.0, 1.0 / max(self.zoom, 0.02))
        self._draw_ant_points(painter, self._edge_xs[light], self._edge_ys[light], QColor("#f4fbff"), pen_w)
        self._draw_ant_points(painter, self._edge_xs[~light], self._edge_ys[~light], QColor("#12161c"), pen_w)
        painter.restore()

    def _draw_ant_points(self, painter: QPainter, xs: np.ndarray, ys: np.ndarray, color: QColor, pen_w: float) -> None:
        if len(xs) == 0:
            return
        pen = QPen(color)
        pen.setWidthF(pen_w)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        points = [QPoint(int(x), int(y)) for x, y in zip(xs.tolist(), ys.tolist())]
        painter.drawPoints(QPolygon(points))

    def _paint_overlays(self, painter: QPainter) -> None:
        tool = self.tool_getter()
        if tool is None:
            return
        self.session.view_zoom = self.zoom
        overlays = tool.overlays(self.session)
        if not overlays:
            return
        ruler = self.ruler()
        painter.save()
        painter.translate(ruler + self.offset_x, ruler + self.offset_y)
        painter.scale(self.zoom, self.zoom)
        pen = QPen(QColor("#d8f6ff"))
        pen.setWidthF(1.25 / self.zoom)
        pen.setCosmetic(False)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for item in overlays:
            if item.kind in {"rect", "ellipse"} and item.rect is not None:
                rect = QRectF(*item.rect)
                if item.kind == "ellipse":
                    painter.drawEllipse(rect)
                else:
                    painter.drawRect(rect)
            elif item.kind == "polyline" and len(item.points) >= 2:
                from PyQt6.QtGui import QPolygonF
                from PyQt6.QtCore import QPointF

                painter.drawPolyline(QPolygonF([QPointF(x, y) for x, y in item.points]))
            elif item.kind == "handles" and item.rect is not None:
                painter.drawRect(QRectF(*item.rect))
                size = 4.0 / self.zoom
                for name, hx, hy in item.handles:
                    color = QColor("#ffb020") if name == "rot" else QColor("#ffffff")
                    painter.fillRect(QRectF(hx - size, hy - size, size * 2, size * 2), color)
                # Stem from the top edge to the rotate handle.
                if item.rect is not None:
                    x, y, w, h = item.rect
                    rot = next((pt for pt in item.handles if pt[0] == "rot"), None)
                    if rot is not None:
                        painter.drawLine(int(x + w / 2), int(y), int(rot[1]), int(rot[2]))
        painter.restore()

    def _paint_rulers(self, painter: QPainter, ruler: int) -> None:
        painter.fillRect(0, 0, self.width(), ruler, QColor("#1a1f26"))
        painter.fillRect(0, 0, ruler, self.height(), QColor("#1a1f26"))
        painter.fillRect(0, 0, ruler, ruler, QColor("#14181d"))
        pen = QPen(QColor("#8aa0b2"))
        painter.setPen(pen)
        step = _tick_step(self.zoom)
        doc = self.session.document
        # Horizontal ticks along the top, vertical ticks down the side.
        # Visible range is whatever canvas pixels currently fall inside the view.
        view_left, view_top = self.widget_to_canvas(ruler, ruler)
        view_right, view_bottom = self.widget_to_canvas(self.width(), self.height())
        start = int(math.floor(min(view_left, 0) / step) * step)
        end = int(math.ceil(max(view_right, doc.width) / step) * step)
        y_start = int(math.floor(min(view_top, 0) / step) * step)
        y_end = int(math.ceil(max(view_bottom, doc.height) / step) * step)
        x = start
        while x <= end:
            sx = ruler + self.offset_x + x * self.zoom
            if ruler <= sx <= self.width():
                painter.drawLine(int(sx), ruler - 8, int(sx), ruler)
                painter.drawText(int(sx) + 2, ruler - 8, str(int(x)))
            x += step
        y = y_start
        while y <= y_end:
            sy = ruler + self.offset_y + y * self.zoom
            if ruler <= sy <= self.height():
                painter.drawLine(ruler - 8, int(sy), ruler, int(sy))
                painter.drawText(2, int(sy) - 2, str(int(y)))
            y += step

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        self.zoom_at(event.position().x(), event.position().y(), 1.1 if delta > 0 else 1 / 1.1)

    def mousePressEvent(self, event) -> None:
        self.setFocus()
        pos = event.position()
        if event.button() == Qt.MouseButton.MiddleButton or self._space or self._is_hand():
            self._panning = True
            self._pan_last = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        tool = self.tool_getter()
        if tool is not None and tool.id == "zoom" and event.button() == Qt.MouseButton.LeftButton:
            alt = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
            self.zoom_at(pos.x(), pos.y(), 1 / 1.25 if alt else 1.25)
            return
        if tool is not None and event.button() == Qt.MouseButton.LeftButton:
            tool.pointer_down(self.session, self._pointer(event))

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        self._report_status(pos.x(), pos.y())
        if self._panning and self._pan_last is not None:
            self.offset_x += pos.x() - self._pan_last.x()
            self.offset_y += pos.y() - self._pan_last.y()
            self._pan_last = pos
            self.update()
            return
        if event.buttons() & Qt.MouseButton.LeftButton:
            tool = self.tool_getter()
            if tool is not None and tool.id not in {"hand", "zoom"}:
                tool.pointer_move(self.session, self._pointer(event))

    def mouseReleaseEvent(self, event) -> None:
        if self._panning and event.button() in {Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton}:
            self._panning = False
            self._pan_last = None
            self.set_tool_cursor("hand" if self._space or self._is_hand() else self._current_cursor())
            return
        tool = self.tool_getter()
        if tool is not None and event.button() == Qt.MouseButton.LeftButton and tool.id not in {"hand", "zoom"}:
            tool.pointer_up(self.session, self._pointer(event))

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            self._space = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        if event.key() == Qt.Key.Key_Escape:
            # A half-drawn shape or stroke is cancelled first. Escape with
            # nothing in progress clears the selection, which is the step
            # people look for after Select.
            tool = self.tool_getter()
            busy = bool(tool.pointer_cancel(self.session)) if tool is not None else False
            if not busy and self.session.document.selection is not None:
                self.session.deselect()
            event.accept()
            return
        if _is_clear_shortcut(event):
            self.session.clear_selection_contents()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            self._space = False
            self.set_tool_cursor(self._current_cursor())
            return
        super().keyReleaseEvent(event)

    def _pointer(self, event) -> PointerEvent:
        x, y = self.widget_to_canvas(event.position().x(), event.position().y())
        mods = event.modifiers()
        return PointerEvent(
            x=x,
            y=y,
            shift=bool(mods & Qt.KeyboardModifier.ShiftModifier),
            alt=bool(mods & Qt.KeyboardModifier.AltModifier),
            ctrl=bool(mods & Qt.KeyboardModifier.ControlModifier),
            zoom=self.zoom,
        )

    def _report_status(self, wx: float, wy: float) -> None:
        x, y = self.widget_to_canvas(wx, wy)
        self.on_status(x, y, self.zoom)

    def _is_hand(self) -> bool:
        tool = self.tool_getter()
        return tool is not None and tool.id == "hand"

    def _current_cursor(self) -> str:
        tool = self.tool_getter()
        return tool.cursor if tool is not None else "arrow"


def _tick_step(zoom: float) -> float:
    raw = 80.0 / max(zoom, 0.02)
    power = 10 ** math.floor(math.log10(max(raw, 1e-6)))
    for mult in (1, 2, 5, 10):
        if power * mult >= raw:
            return power * mult
    return power * 10


def _is_clear_shortcut(event) -> bool:
    """Delete/Backspace clears a selection, but not while a modifier is held."""
    if event.isAutoRepeat():
        return False
    mods = event.modifiers()
    blocked = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier
    if mods & blocked:
        return False
    return event.key() in {Qt.Key.Key_Delete, Qt.Key.Key_Backspace}


def _selection_tint(selection: Image.Image) -> Image.Image:
    mask = np.asarray(selection.convert("L"))
    height, width = mask.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    on = mask > 0
    rgba[on] = (70, 170, 255, 70)
    # One-pixel edge so the boundary stays readable over a busy photo.
    eroded = on.copy()
    eroded[1:, :] &= on[:-1, :]
    eroded[:-1, :] &= on[1:, :]
    eroded[:, 1:] &= on[:, :-1]
    eroded[:, :-1] &= on[:, 1:]
    edge = on & ~eroded
    rgba[edge] = (236, 248, 255, 220)
    return Image.fromarray(rgba, "RGBA")

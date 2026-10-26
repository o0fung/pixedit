"""Zoomable canvas. Document pixels stay in the session; this widget only views them."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image
from PyQt6.QtCore import QPoint, QPointF, QRect, Qt, QRectF, QTimer
from PyQt6.QtGui import QColor, QCursor, QImage, QPainter, QPen, QPixmap, QPolygon, QPolygonF
from PyQt6.QtWidgets import QApplication, QWidget

from pixedit.composite import composite, composite_region, shows_redaction
from pixedit.frames import letterbox_offset, shared_frame_zoom
from pixedit.tools.base import PointerEvent
from pixedit.ui.cursors import tool_pixmap_cursor

# While a sampled tool's button is down, store the cursor this often.
# Two stores are handed over together, so one redraw covers more than one coordinate.
_STROKE_SAMPLE_MS = 8
_STROKE_FLUSH_TICKS = 2

_CURSORS = {
    "cross": Qt.CursorShape.CrossCursor,
    "hand": Qt.CursorShape.OpenHandCursor,
    "size": Qt.CursorShape.SizeAllCursor,
    "size-fdiag": Qt.CursorShape.SizeFDiagCursor,
    "size-bdiag": Qt.CursorShape.SizeBDiagCursor,
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
    def __init__(self, session, tool_getter, on_status, on_drop=None) -> None:
        super().__init__()
        self.session = session
        self.tool_getter = tool_getter
        self.on_status = on_status
        self.on_drop = on_drop
        self.setAcceptDrops(on_drop is not None)
        self.zoom = 1.0
        self.offset_x = 24.0
        self.offset_y = 24.0
        self._pixmap: QPixmap | None = None
        self._dirty = True
        self._view_group_id = session.document.active_group_id
        # Zoom and pan for pages already visited. The first visit of a page fits it.
        self._group_views: dict[str, tuple[float, float, float]] = {}
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
        # Playback of pages shares one zoom and centers each page. The
        # document pixels stay at their own size.
        self.letterbox = False
        # A guide drag started in a ruler. None when the pointer is a tool.
        self._guide: dict | None = None
        # Samples only append a coordinate. A second tick flushes the batch
        # so the stroke is not redrawn on every sample. Single-shot, restarted
        # at the end of the tick, so a slow redraw cannot queue a burst.
        self._stroke_sampling = False
        self._stroke_tool_id = ""
        self._stroke_points: list[tuple[float, float]] = []
        self._stroke_ticks = 0
        self._stroke_timer = QTimer(self)
        self._stroke_timer.setSingleShot(True)
        self._stroke_timer.timeout.connect(self._on_stroke_sample)
        # Alt swaps the zoom glass between plus and minus. Key release passes
        # the new state in, because the keyboard query can still report the
        # key that is being released.
        self._alt = False
        # Last handle the pointer was on. Cleared when the tool changes, so a
        # leftover rotate mark is not shown for the next tool.
        self._hover_name: str | None = None
        self._hover_tool_id = ""
        self._brush_cursor_key: tuple[int, int] | None = None
        self._brush_cursor: QCursor | None = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(320, 240)

    def invalidate(self, bounds: tuple[int, int, int, int] | None = None) -> None:
        if self._note_group_view():
            bounds = None
        if bounds is not None and self._patch_preview(bounds):
            self._update_region(bounds)
            return
        self._dirty = True
        self._selection_id = None
        self.update()

    def _note_group_view(self) -> bool:
        """Fit a page the first time it is shown. Later visits keep its zoom and pan.

        The image rectangle is the active group's pixel size. Leaving a page
        stores the current zoom, and coming back puts that rectangle where
        it was. A page that has not been shown yet is fitted in the view.
        """
        group_id = self.session.document.active_group_id
        if group_id == self._view_group_id:
            return False
        # Letterbox zoom is not the user's view. Don't store it, and don't
        # fit each page on its own, or a small page would grow to fill the window.
        if self._view_group_id and not self.letterbox:
            self._group_views[self._view_group_id] = (self.zoom, self.offset_x, self.offset_y)
        self._view_group_id = group_id
        if self.letterbox:
            self._dirty = True
            self._selection_id = None
            return True
        self._dirty = True
        self._selection_id = None
        saved = self._group_views.get(group_id)
        if saved is None:
            self.fit()
        else:
            self.zoom, self.offset_x, self.offset_y = saved
            self.session.view_zoom = self.zoom
            self.update()
            self.refresh_tool_cursor()
        return True

    def ruler(self) -> int:
        return 22 if self.session.option("rulers", True) else 0

    def refresh_tool_cursor(self) -> None:
        """Rebuild the pointer after a size, zoom, or tool-option change."""
        self._brush_cursor_key = None
        self._apply_cursor()

    def set_alt_held(self, held: bool) -> None:
        """Alt changes the zoom pointer even when the mouse is still."""
        if held == self._alt:
            return
        self._apply_cursor(alt=held)

    def _apply_cursor(self, *, alt: bool | None = None) -> None:
        """Pick the pointer for what the next click will do.

        A pan keeps the closed hand. Space forces an open hand until it is
        released. Brush and eraser keep the circle that matches the diameter.
        Every other tool is asked for its current name, so a shape or selection
        option is not stuck on the pointer from when the tool was chosen. A
        tool that tracks the pointer, such as Transform, uses the last handle
        the mouse crossed. Zoom shows a minus while Alt is held. A drawn name
        becomes a pixmap; any other name is a system pointer.
        """
        if alt is None:
            alt = bool(QApplication.queryKeyboardModifiers() & Qt.KeyboardModifier.AltModifier)
        self._alt = alt
        if self._panning:
            return
        if self._space:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        tool = self.tool_getter()
        tool_id = tool.id if tool is not None else ""
        if tool_id != self._hover_tool_id:
            self._hover_name = None
            self._hover_tool_id = tool_id
        if tool is not None and tool.id in {"brush", "eraser"}:
            self.setCursor(self._circle_cursor())
            return
        name = tool.cursor_name(self.session) if tool is not None else "arrow"
        if self._hover_name:
            name = self._hover_name
        elif tool is not None and tool.id == "zoom" and self._alt:
            name = "zoom-out"
        pixmap_cursor = tool_pixmap_cursor(name, device_pixel_ratio=float(self.devicePixelRatioF() or 1.0))
        if pixmap_cursor is not None:
            self.setCursor(pixmap_cursor)
            return
        self.setCursor(_CURSORS.get(name, Qt.CursorShape.ArrowCursor))

    def _sync_hover_cursor(self, event) -> None:
        """Follow a handle as the pointer moves. Pan, Space, and a sampled stroke win."""
        if self._panning or self._space or self._stroke_sampling:
            return
        tool = self.tool_getter()
        if tool is None or not hasattr(tool, "hover_cursor"):
            if self._hover_name is not None:
                self._hover_name = None
                self._apply_cursor()
            return
        name = tool.hover_cursor(self.session, self._pointer(event))
        if name == self._hover_name and self._hover_tool_id == tool.id:
            return
        self._hover_name = name
        self._hover_tool_id = tool.id
        self._apply_cursor()

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

    def _apply_letterbox(self) -> None:
        """Fit the largest page, then center the page on screen at that scale."""
        groups = self.session.document.groups
        ruler = self.ruler()
        zoom = shared_frame_zoom(
            self.width(),
            self.height(),
            [(group.width, group.height) for group in groups],
            ruler=ruler,
        )
        doc = self.session.document
        offset_x, offset_y = letterbox_offset(self.width(), self.height(), doc.width, doc.height, zoom, ruler=ruler)
        self.zoom = zoom
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.session.view_zoom = self.zoom

    def paintEvent(self, event) -> None:
        self._note_group_view()
        if self.letterbox:
            self._apply_letterbox()
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
        self._paint_guides(painter, ruler)
        if ruler:
            self._paint_rulers(painter, ruler)

    def _paint_guides(self, painter: QPainter, ruler: int) -> None:
        group = self.session.document.group
        vertical = list(group.guides_v)
        horizontal = list(group.guides_h)
        if self._guide is not None:
            # The line under the pointer replaces the guide being moved.
            if self._guide["axis"] == "v":
                if self._guide["index"] is not None and 0 <= self._guide["index"] < len(vertical):
                    vertical.pop(self._guide["index"])
                vertical.append(int(round(self._guide["value"])))
            else:
                if self._guide["index"] is not None and 0 <= self._guide["index"] < len(horizontal):
                    horizontal.pop(self._guide["index"])
                horizontal.append(int(round(self._guide["value"])))
        if not vertical and not horizontal:
            return
        pen = QPen(QColor("#5ec8ff"))
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        for x in vertical:
            sx = ruler + self.offset_x + x * self.zoom
            painter.drawLine(int(sx), ruler, int(sx), self.height())
        for y in horizontal:
            sy = ruler + self.offset_y + y * self.zoom
            painter.drawLine(ruler, int(sy), self.width(), int(sy))

    def _begin_guide(self, pos) -> bool:
        """Start a guide drag from a ruler. The canvas tools do not see that press."""
        ruler = self.ruler()
        if ruler <= 0 or self._space:
            return False
        x, y = float(pos.x()), float(pos.y())
        if y < ruler and x >= ruler:
            axis = "h"
        elif x < ruler and y >= ruler:
            axis = "v"
        else:
            return False
        cx, cy = self.widget_to_canvas(x, y)
        value = cy if axis == "h" else cx
        guides = self.session.document.group.guides_h if axis == "h" else self.session.document.group.guides_v
        index = None
        threshold = 6.0 / max(self.zoom, 0.02)
        for guide_index, guide in enumerate(guides):
            if abs(float(guide) - value) <= threshold:
                index = guide_index
                value = float(guide)
                break
        self._guide = {"axis": axis, "index": index, "value": value}
        self.update()
        return True

    def _move_guide(self, pos) -> None:
        if self._guide is None:
            return
        cx, cy = self.widget_to_canvas(float(pos.x()), float(pos.y()))
        self._guide["value"] = cy if self._guide["axis"] == "h" else cx
        self.update()

    def _finish_guide(self, pos) -> None:
        drag = self._guide
        self._guide = None
        if drag is None:
            return
        cx, cy = self.widget_to_canvas(float(pos.x()), float(pos.y()))
        value = cy if drag["axis"] == "h" else cx
        doc = self.session.document
        ruler = self.ruler()
        on_page = float(pos.x()) >= ruler and float(pos.y()) >= ruler
        if drag["axis"] == "h":
            on_page = on_page and 0 <= value <= doc.height
        else:
            on_page = on_page and 0 <= value <= doc.width
        vertical = list(doc.group.guides_v)
        horizontal = list(doc.group.guides_h)
        target = horizontal if drag["axis"] == "h" else vertical
        if drag["index"] is not None and 0 <= drag["index"] < len(target):
            target.pop(drag["index"])
        if on_page:
            target.append(int(round(value)))
        self.session.set_guides(vertical, horizontal)
        if on_page:
            axis_name = "Horizontal" if drag["axis"] == "h" else "Vertical"
            self.session.show_message(
                f"{axis_name} guide at {int(round(value))}. Drag it back to the ruler to remove it."
            )
        elif drag["index"] is not None:
            self.session.show_message("Removed the guide.")
        self.update()

    def dragEnterEvent(self, event) -> None:
        if self.on_drop is not None and event.mimeData() is not None and event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event) -> None:
        if self.on_drop is None:
            event.ignore()
            return
        self.on_drop(event)

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
        # Cells are 12px, larger than the leftover space on the right and bottom.
        # Clip to the canvas or those squares paint into the margin.
        painter.save()
        painter.setClipRect(target, Qt.ClipOperation.IntersectClip)
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
        painter.restore()

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
                painter.drawPolyline(QPolygonF([QPointF(x, y) for x, y in item.points]))
            elif item.kind == "handles" and (item.rect is not None or len(item.points) >= 4):
                # A turn passes the four corners. A move or scale keeps the straight rectangle.
                if len(item.points) >= 4:
                    painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in item.points]))
                elif item.rect is not None:
                    painter.drawRect(QRectF(*item.rect))
                size = 4.0 / self.zoom
                for name, hx, hy in item.handles:
                    color = QColor("#ffb020") if name == "rot" else QColor("#ffffff")
                    painter.fillRect(QRectF(hx - size, hy - size, size * 2, size * 2), color)
                # Stem from the middle of the top edge (nw–ne) to the rotate handle.
                nw = next((pt for pt in item.handles if pt[0] == "nw"), None)
                ne = next((pt for pt in item.handles if pt[0] == "ne"), None)
                rot = next((pt for pt in item.handles if pt[0] == "rot"), None)
                if nw is not None and ne is not None and rot is not None:
                    painter.drawLine(
                        int((nw[1] + ne[1]) / 2),
                        int((nw[2] + ne[2]) / 2),
                        int(rot[1]),
                        int(rot[2]),
                    )
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
        if event.button() == Qt.MouseButton.LeftButton and self._begin_guide(pos):
            return
        tool = self.tool_getter()
        if tool is not None and tool.id == "zoom" and event.button() == Qt.MouseButton.LeftButton:
            alt = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
            self.zoom_at(pos.x(), pos.y(), 1 / 1.25 if alt else 1.25)
            return
        if tool is not None and event.button() == Qt.MouseButton.LeftButton:
            tool.pointer_down(self.session, self._pointer(event))
            if not self._space and tool.wants_stroke_sampling(self.session):
                self._start_stroke_sampling(tool.id)
            self._sync_hover_cursor(event)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        self._report_status(pos.x(), pos.y())
        # The key may have gone down while another widget had the keyboard.
        self.set_alt_held(bool(event.modifiers() & Qt.KeyboardModifier.AltModifier))
        if self._panning and self._pan_last is not None:
            self.offset_x += pos.x() - self._pan_last.x()
            self.offset_y += pos.y() - self._pan_last.y()
            self._pan_last = pos
            self.update()
            return
        if self._guide is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self._move_guide(pos)
            return
        self._sync_hover_cursor(event)
        # The stroke timer owns coordinates for tools that poll, such as the
        # brush and freehand Select. A mouse-move during the drag only
        # refreshes the status readout above.
        if self._stroke_sampling:
            return
        if event.buttons() & Qt.MouseButton.LeftButton:
            tool = self.tool_getter()
            if tool is not None and tool.id not in {"hand", "zoom"}:
                tool.pointer_move(self.session, self._pointer(event))

    def mouseReleaseEvent(self, event) -> None:
        if self._guide is not None and event.button() == Qt.MouseButton.LeftButton:
            self._finish_guide(event.position())
            return
        if self._panning and event.button() in {Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton}:
            self._panning = False
            self._pan_last = None
            tool = self.tool_getter()
            if (
                tool is not None
                and hasattr(tool, "hover_cursor")
                and not self._space
                and not self._stroke_sampling
            ):
                self._hover_name = tool.hover_cursor(self.session, self._pointer(event))
                self._hover_tool_id = tool.id
            self._apply_cursor()
            return
        if self._stroke_sampling and event.button() == Qt.MouseButton.LeftButton:
            self._finish_stroke_sampling(event)
            return
        tool = self.tool_getter()
        if tool is not None and event.button() == Qt.MouseButton.LeftButton and tool.id not in {"hand", "zoom"}:
            tool.pointer_up(self.session, self._pointer(event))
        self._sync_hover_cursor(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Alt and not event.isAutoRepeat():
            self.set_alt_held(True)
        if event.key() == Qt.Key.Key_Space:
            self._space = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        if event.key() == Qt.Key.Key_Escape:
            if self._guide is not None:
                self._guide = None
                self.update()
                event.accept()
                return
            # A half-drawn shape or stroke is cancelled first. Escape with
            # nothing in progress clears the selection, which is the step
            # people look for after Select. Drop stored samples without
            # flushing them; the tool restores whatever the drag changed.
            self._cancel_stroke_sampling()
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
        if event.key() == Qt.Key.Key_Alt and not event.isAutoRepeat():
            # False is passed through. A keyboard query during this release can
            # still report Alt, which would leave the zoom glass on minus.
            self.set_alt_held(False)
        if event.key() == Qt.Key.Key_Space:
            self._space = False
            self._apply_cursor()
            return
        super().keyReleaseEvent(event)

    def _start_stroke_sampling(self, tool_id: str) -> None:
        self._stroke_sampling = True
        self._stroke_tool_id = tool_id
        self._stroke_points = []
        self._stroke_ticks = 0
        self._stroke_timer.start(_STROKE_SAMPLE_MS)

    def _cancel_stroke_sampling(self) -> None:
        self._stroke_sampling = False
        self._stroke_timer.stop()
        self._stroke_points = []
        self._stroke_ticks = 0

    def _on_stroke_sample(self) -> None:
        if not self._stroke_sampling:
            return
        tool = self.tool_getter()
        button_down = bool(QApplication.mouseButtons() & Qt.MouseButton.LeftButton)
        live = (
            tool is not None
            and tool.id == self._stroke_tool_id
            and bool(getattr(tool, "stroke_live", lambda: False)())
        )
        # Escape or a tool change already restored the layer. A release that
        # the canvas missed still has to flush the stored coordinates.
        if not live:
            self._cancel_stroke_sampling()
            return
        if not button_down:
            self._finish_stroke_sampling(None)
            return
        self._remember_cursor(None)
        self._stroke_ticks += 1
        if self._stroke_ticks >= _STROKE_FLUSH_TICKS:
            self._flush_stroke_points()
            self._stroke_ticks = 0
        if self._stroke_sampling:
            self._stroke_timer.start(_STROKE_SAMPLE_MS)

    def _finish_stroke_sampling(self, event) -> None:
        """Store the last coordinate, hand over the batch, and commit.

        ``event`` is the mouse release. None means the button was already up
        when the timer fired, so Shift and Alt come from the keyboard.
        """
        tool = self.tool_getter()
        self._stroke_sampling = False
        self._stroke_timer.stop()
        if tool is None or tool.id != self._stroke_tool_id:
            self._stroke_points = []
            return
        self._remember_cursor(event)
        self._flush_stroke_points()
        if hasattr(tool, "commit_stroke"):
            tool.commit_stroke(self.session, self._pointer_now(event))

    def _remember_cursor(self, event) -> None:
        if event is None:
            local = self.mapFromGlobal(QCursor.pos())
            x, y = self.widget_to_canvas(float(local.x()), float(local.y()))
        else:
            x, y = self.widget_to_canvas(event.position().x(), event.position().y())
        if self._stroke_points:
            last_x, last_y = self._stroke_points[-1]
            if math.hypot(x - last_x, y - last_y) < 0.05:
                return
        self._stroke_points.append((x, y))

    def _flush_stroke_points(self) -> None:
        if not self._stroke_points:
            return
        tool = self.tool_getter()
        points = self._stroke_points
        self._stroke_points = []
        if tool is None or not hasattr(tool, "extend_points"):
            return
        tool.extend_points(self.session, points)

    def _patch_preview(self, bounds: tuple[int, int, int, int]) -> bool:
        """Paint one rectangle into the current pixmap. False means do a full rebuild."""
        doc = self.session.document
        if self._dirty or self._pixmap is None or self._pixmap.isNull():
            return False
        if self._pixmap.width() != doc.width or self._pixmap.height() != doc.height:
            return False
        if shows_redaction(doc):
            return False
        box = _clip_document_box(doc.width, doc.height, bounds)
        if box is None:
            return True
        region = composite_region(doc, box)
        painter = QPainter(self._pixmap)
        painter.drawPixmap(box[0], box[1], pil_to_pixmap(region))
        painter.end()
        return True

    def _update_region(self, bounds: tuple[int, int, int, int]) -> None:
        ruler = self.ruler()
        x0, y0, x1, y1 = bounds
        left = ruler + self.offset_x + x0 * self.zoom
        top = ruler + self.offset_y + y0 * self.zoom
        right = ruler + self.offset_x + x1 * self.zoom
        bottom = ruler + self.offset_y + y1 * self.zoom
        self.update(
            QRect(
                int(math.floor(min(left, right))) - 2,
                int(math.floor(min(top, bottom))) - 2,
                int(math.ceil(abs(right - left))) + 4,
                int(math.ceil(abs(bottom - top))) + 4,
            )
        )

    def _pointer_now(self, event) -> PointerEvent:
        """Release pointer, or the cursor and keys if the release event was missed."""
        if event is None:
            local = self.mapFromGlobal(QCursor.pos())
            x, y = self.widget_to_canvas(float(local.x()), float(local.y()))
            return self._pointer_at(x, y, QApplication.keyboardModifiers())
        return self._pointer(event)

    def _pointer(self, event) -> PointerEvent:
        x, y = self.widget_to_canvas(event.position().x(), event.position().y())
        return self._pointer_at(x, y, event.modifiers())

    def _pointer_at(self, x: float, y: float, mods) -> PointerEvent:
        return PointerEvent(
            x=x,
            y=y,
            shift=bool(mods & Qt.KeyboardModifier.ShiftModifier),
            alt=bool(mods & Qt.KeyboardModifier.AltModifier),
            # Command on macOS is Meta. The Transform tool treats either key as
            # "add or remove the layer under the pointer."
            ctrl=bool(mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)),
            zoom=self.zoom,
        )

    def _report_status(self, wx: float, wy: float) -> None:
        x, y = self.widget_to_canvas(wx, wy)
        self.on_status(x, y, self.zoom)

    def _is_hand(self) -> bool:
        tool = self.tool_getter()
        return tool is not None and tool.id == "hand"


def _clip_document_box(
    width: int, height: int, box: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    x0 = max(0, int(math.floor(box[0])))
    y0 = max(0, int(math.floor(box[1])))
    x1 = min(width, int(math.ceil(box[2])))
    y1 = min(height, int(math.ceil(box[3])))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


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

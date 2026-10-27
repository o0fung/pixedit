"""Zoomable canvas. Document pixels stay in the session; this widget only views them."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image
from PyQt6.QtCore import QPoint, QPointF, QRect, Qt, QRectF, QTimer
from PyQt6.QtGui import QColor, QCursor, QImage, QPainter, QPen, QPixmap, QPolygon, QPolygonF
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from pixedit.composite import composite, composite_region, composite_view, redaction_padding, shows_redaction
from pixedit.frames import letterbox_offset, shared_frame_zoom
from pixedit.model import Document, mask_target
from pixedit.proxy import PREVIEW_MAX_EDGE
from pixedit.tools.base import PointerEvent
from pixedit.ui.cursors import tool_pixmap_cursor

# How often a sampled stroke paints the moves stored since the last tick.
# Mouse moves keep every coordinate. One redraw covers the whole batch.
_STROKE_SAMPLE_MS = 8
# A shorter move is still a click. Past this, the zoom tool scrubs instead.
_ZOOM_DRAG_THRESHOLD = 6.0
# Vertical pixels that double the zoom, or halve it when dragging down.
_ZOOM_DRAG_SPAN = 160.0
# Longest edge of a close-zoom pan crop, in document pixels. Past this, one
# window of travel still sweeps most of a large page, so the drag keeps the
# low-resolution whole-page plate.
PAN_CROP_MAX_EDGE = 4096
# Extra screen pixels past one window of travel. A drag that ends on the
# window edge stays inside the crop and does not rebuild it.
_PAN_CROP_SLACK = 8
# Checker cell, in screen pixels. The loop is aligned to this grid.
_CHECKER = 12

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
    # fromImage copies the pixels, so the extra QImage copy is not needed.
    # ``data`` stays alive until that copy returns.
    return QPixmap.fromImage(qimage)


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
        # "full" is a document-sized pixmap. "view" is the visible rectangle at
        # screen resolution, used while zoomed out so a large photo is not
        # flattened at full size just to fit the window.
        self._pixmap_mode = "full"
        self._view_box: tuple[float, float, float, float] | None = None
        self._view_key: tuple | None = None
        # Plates for a transform drag. Built once, then the stand-in moves on top.
        self._drag_plates_key = None
        self._drag_below: QPixmap | None = None
        self._drag_above: QPixmap | None = None
        self._drag_sprite: QPixmap | None = None
        self._drag_selection: QPixmap | None = None
        self._drag_plate_mode = "full"
        self._drag_view_box: tuple[float, float, float, float] | None = None
        self._view_group_id = session.document.active_group_id
        # Zoom and pan for pages already visited. The first visit of a page fits it.
        self._group_views: dict[str, tuple[float, float, float]] = {}
        self._space = False
        self._panning = False
        self._pan_last = None
        # Plate shown while a large page is panned or scrub-zoomed.
        # None means this drag is sliding the sharp cache.
        # _pan_proxy_box is the document rectangle of a close-zoom crop.
        # None means the plate is the whole page, shrunk to PREVIEW_MAX_EDGE.
        self._pan_proxy: QPixmap | None = None
        self._pan_selection_proxy: QPixmap | None = None
        self._pan_proxy_box: tuple[float, float, float, float] | None = None
        # Zoom-tool button. Scrub starts after the pointer moves, and the
        # sharp picture is built when the button comes up.
        self._zooming = False
        self._zoom_scrub = False
        self._zoom_anchor: tuple[float, float] | None = None
        self._zoom_origin = 1.0
        self._zoom_alt = False
        # Space interrupted a zoom drag and the button is still down. The next
        # move pans. Other tools keep their own drag.
        self._space_pan_pending = False
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
        # Mouse moves append a coordinate. This timer paints the batch so the
        # stroke is not redrawn on every sample. Single-shot and precise,
        # restarted after the paint, so a slow dab cannot queue a burst and
        # the interval is not coarsened.
        self._stroke_sampling = False
        self._stroke_tool_id = ""
        self._stroke_points: list[tuple[float, float]] = []
        self._stroke_timer = QTimer(self)
        self._stroke_timer.setTimerType(Qt.TimerType.PreciseTimer)
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
        self._welcome = QWidget(self)
        self._welcome.setObjectName("welcome")
        # The hint sits on the canvas. Clicks and drops pass through to the view.
        self._welcome.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        welcome_layout = QVBoxLayout(self._welcome)
        welcome_layout.setContentsMargins(24, 0, 24, 0)
        welcome_layout.setSpacing(14)
        self.welcome_title = QLabel("PixEdit")
        self.welcome_title.setObjectName("welcomeTitle")
        self.welcome_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.welcome_hint = QLabel(
            "New… starts a blank canvas\nOpen… edits a picture or a project\nOr drop a file here"
        )
        self.welcome_hint.setObjectName("welcomeHint")
        self.welcome_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_layout.addWidget(self.welcome_title)
        welcome_layout.addWidget(self.welcome_hint)
        self.sync_welcome()

    def invalidate(self, bounds: tuple[int, int, int, int] | None = None) -> None:
        self.sync_welcome()
        if not self.session.document.groups:
            self._dirty = True
            self._clear_pan_proxy()
            self.update()
            return
        if self._note_group_view():
            bounds = None
        # A change under an active pan or zoom scrub stays on the low-res plate.
        # Rebuilding the sharp page here would stall the drag the plate is meant to avoid.
        if (self._panning or self._zoom_scrub) and self._refresh_pan_proxy():
            return
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
        if not self.session.document.groups:
            return 0
        return 22 if self.session.option("rulers", True) else 0

    def _viewport_rect(self) -> QRectF:
        """Widget area under the rulers, where the page is drawn."""
        ruler = float(self.ruler())
        return QRectF(
            ruler,
            ruler,
            max(0.0, float(self.width()) - ruler),
            max(0.0, float(self.height()) - ruler),
        )

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
        if not self.session.document.groups:
            return
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

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place_welcome()

    def sync_welcome(self) -> None:
        """Show the name and the start hint only while no page is open."""
        empty = not self.session.document.groups
        self._welcome.setVisible(empty)
        if empty:
            self._place_welcome()

    def _place_welcome(self) -> None:
        self._welcome.adjustSize()
        size = self._welcome.sizeHint()
        width = max(size.width(), 280)
        height = size.height()
        self._welcome.resize(width, height)
        self._welcome.move(max(0, (self.width() - width) // 2), max(0, (self.height() - height) // 2))

    def _has_page(self) -> bool:
        return bool(self.session.document.groups)

    def paintEvent(self, event) -> None:
        # No page yet. The welcome labels are child widgets on this fill.
        if not self.session.document.groups:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor("#101318"))
            return
        self._note_group_view()
        if self.letterbox:
            self._apply_letterbox()
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101318"))
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
        # Transform drag and pan/scrub each keep their own stand-in. The sharp
        # cache is rebuilt only when neither preview is on screen.
        proxy = self._transform_proxy()
        if proxy is not None:
            # The photo stays in the cached pixmap. This paint draws the plates
            # around it and the low-resolution flatten, and does not resample.
            self._paint_transform_proxy(painter, target, proxy)
        elif self._using_pan_proxy():
            self._paint_pan_proxy(painter, target)
        else:
            self._clear_drag_plates()
            self._ensure_pixmap()
            dest = self._pixmap_dest(target)
            if self._pixmap is not None and not self._pixmap.isNull():
                painter.drawPixmap(dest, self._pixmap, QRectF(0, 0, self._pixmap.width(), self._pixmap.height()))
            self._paint_selection(painter, dest)
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

    def _ensure_pixmap(self, *, chrome: bool = True) -> None:
        # The pan plate is already the picture on screen. Rebuilding the sharp
        # cache here would follow the offset and stall every move.
        if self._using_pan_proxy():
            return
        if self._pixmap_is_current():
            return
        doc = self.session.document
        box = self._visible_document_box() if self._use_view_cache() else None
        if box is not None:
            self._pixmap = pil_to_pixmap(composite_view(doc, box, self._view_scale()))
            self._pixmap_mode = "view"
            self._view_box = box
            self._view_key = self._current_view_key()
        else:
            self._pixmap = pil_to_pixmap(composite(doc))
            self._pixmap_mode = "full"
            self._view_box = None
            self._view_key = None
        self._dirty = False
        if chrome:
            self._sync_selection_chrome()

    def _pixmap_is_current(self) -> bool:
        if self._dirty or self._pixmap is None or self._pixmap.isNull():
            return False
        if self._use_view_cache():
            return self._pixmap_mode == "view" and self._view_key == self._current_view_key()
        return self._pixmap_mode == "full"

    def _use_view_cache(self) -> bool:
        """True when a screen-sized picture is enough and no mask needs the full plate.

        At 100% and above the cache stays document-sized, so a brush patch lines
        up with document pixels. A visible mask also stays on the full flatten:
        blur and noise read pixels outside a downscaled rectangle.
        """
        doc = self.session.document
        if self.zoom >= 1.0 or shows_redaction(doc):
            return False
        box = self._visible_document_box()
        if box is None:
            return False
        x0, y0, x1, y1 = box
        screen_w = (x1 - x0) * self._view_scale()
        screen_h = (y1 - y0) * self._view_scale()
        # Either edge being smaller than the document is enough. A wide window
        # can show the full width at 1:1 and still only a strip of the height.
        smaller = screen_w + 0.5 < doc.width or screen_h + 0.5 < doc.height
        return smaller and screen_w >= 1 and screen_h >= 1

    def _view_scale(self) -> float:
        """Device pixels per document pixel, capped so the view never upscales."""
        dpr = float(self.devicePixelRatioF() or 1.0)
        return min(1.0, self.zoom * max(1.0, dpr))

    def _visible_document_box(self) -> tuple[float, float, float, float] | None:
        """Document rectangle currently inside the widget, or None when it misses."""
        doc = self.session.document
        if self.zoom <= 0 or doc.width < 1 or doc.height < 1:
            return None
        ruler = self.ruler()
        x0 = max(0.0, -self.offset_x / self.zoom)
        y0 = max(0.0, -self.offset_y / self.zoom)
        x1 = min(float(doc.width), (self.width() - ruler - self.offset_x) / self.zoom)
        y1 = min(float(doc.height), (self.height() - ruler - self.offset_y) / self.zoom)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def _current_view_key(self) -> tuple:
        doc = self.session.document
        return (
            self.zoom,
            self.offset_x,
            self.offset_y,
            self.width(),
            self.height(),
            self.ruler(),
            float(self.devicePixelRatioF() or 1.0),
            doc.width,
            doc.height,
            self._visible_document_box(),
        )

    def _pixmap_dest(self, full_target: QRectF) -> QRectF:
        """Widget rectangle the cached pixmap covers.

        A full-resolution cache is the whole document. A zoomed-out cache is
        only the part that is on screen, so it must not be stretched across
        the off-screen part of the document.
        """
        if self._pixmap_mode != "view" or self._view_box is None:
            return full_target
        x0, y0, x1, y1 = self._view_box
        ruler = self.ruler()
        return QRectF(
            ruler + self.offset_x + x0 * self.zoom,
            ruler + self.offset_y + y0 * self.zoom,
            (x1 - x0) * self.zoom,
            (y1 - y0) * self.zoom,
        )

    def _transform_proxy(self):
        tool = self.tool_getter()
        if tool is None:
            return None
        drag_proxy = getattr(tool, "drag_proxy", None)
        if drag_proxy is None:
            return None
        return drag_proxy()

    def _clear_drag_plates(self) -> None:
        if self._drag_plates_key is None and self._drag_sprite is None:
            return
        self._drag_plates_key = None
        self._drag_below = None
        self._drag_above = None
        self._drag_sprite = None
        self._drag_selection = None
        self._drag_plate_mode = "full"
        self._drag_view_box = None

    def _ensure_drag_plates(self, proxy) -> None:
        """Flatten what sits under and over the selection, once per gesture.

        The stand-in is drawn between those plates. A later move only changes
        where that pixmap goes, so the document composite stays cached.
        """
        use_view = self._use_view_cache()
        view_box = self._visible_document_box() if use_view else None
        if use_view and view_box is None:
            use_view = False
        doc = self.session.document
        selection = doc.selection
        if use_view:
            view_key: tuple = ("view", self._current_view_key())
        else:
            view_key = ("full", doc.width, doc.height)
        key = (proxy.layer_ids, id(proxy.image), id(selection) if selection is not None else None, view_key)
        if key == self._drag_plates_key:
            return
        self._drag_plate_mode = "view" if use_view else "full"
        self._drag_view_box = view_box if use_view else None
        below, above = _split_around_selection(doc.layers, set(proxy.layer_ids))
        self._drag_below = self._plate_pixmap(below, use_view, view_box)
        self._drag_above = self._plate_pixmap(above, use_view, view_box)
        self._drag_sprite = pil_to_pixmap(proxy.image)
        self._drag_selection = self._selection_plate(selection, use_view, view_box)
        self._drag_plates_key = key

    def _plate_pixmap(self, layers: list, use_view: bool, view_box) -> QPixmap | None:
        doc = self.session.document
        members = _plate_members(doc.layers, layers)
        if not any(layer.visible and layer.opacity > 0 and layer.kind != "redaction" for layer in members):
            return None
        temp = Document(doc.width, doc.height, members)
        if use_view and view_box is not None:
            image = composite_view(temp, view_box, self._view_scale())
        else:
            image = composite(temp)
        return pil_to_pixmap(image)

    def _selection_plate(self, selection, use_view: bool, view_box) -> QPixmap | None:
        if selection is None:
            return None
        if use_view and view_box is not None:
            x0, y0, x1, y1 = view_box
            scale = self._view_scale()
            image = _view_selection_tint(
                selection,
                view_box,
                _span_pixels(x1 - x0, scale),
                _span_pixels(y1 - y0, scale),
            )
        else:
            image = _selection_tint(selection)
        return pil_to_pixmap(image)

    def _plate_dest(self, full_target: QRectF) -> QRectF:
        if self._drag_plate_mode != "view" or self._drag_view_box is None:
            return full_target
        x0, y0, x1, y1 = self._drag_view_box
        ruler = self.ruler()
        return QRectF(
            ruler + self.offset_x + x0 * self.zoom,
            ruler + self.offset_y + y0 * self.zoom,
            (x1 - x0) * self.zoom,
            (y1 - y0) * self.zoom,
        )

    def _paint_plate(self, painter: QPainter, target: QRectF, pixmap: QPixmap | None) -> None:
        if pixmap is None or pixmap.isNull():
            return
        painter.drawPixmap(
            self._plate_dest(target),
            pixmap,
            QRectF(0, 0, pixmap.width(), pixmap.height()),
        )

    def _paint_stand_in(self, painter: QPainter, proxy) -> None:
        pixmap = self._drag_sprite
        if pixmap is None or pixmap.isNull():
            return
        ruler = self.ruler()
        painter.save()
        painter.translate(ruler + self.offset_x, ruler + self.offset_y)
        painter.scale(self.zoom, self.zoom)
        # Qt rotates clockwise, which is the same direction as the tool angle.
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        rect = proxy.live_box
        if proxy.angle:
            x, y, width, height = proxy.box
            cx = x + width / 2.0
            cy = y + height / 2.0
            painter.translate(cx, cy)
            painter.rotate(proxy.angle)
            painter.translate(-cx, -cy)
            rect = proxy.box
        painter.drawPixmap(
            QRectF(rect[0], rect[1], rect[2], rect[3]),
            pixmap,
            QRectF(0, 0, pixmap.width(), pixmap.height()),
        )
        painter.restore()

    def _paint_transform_proxy(self, painter: QPainter, target: QRectF, proxy) -> None:
        self._ensure_drag_plates(proxy)
        # Keep the flatten inside the page. Handles are drawn later and may
        # sit above the top edge, so this clip does not cover them.
        painter.save()
        painter.setClipRect(target, Qt.ClipOperation.IntersectClip)
        self._paint_plate(painter, target, self._drag_below)
        self._paint_stand_in(painter, proxy)
        self._paint_plate(painter, target, self._drag_above)
        self._paint_plate(painter, target, self._drag_selection)
        painter.restore()
        self._sync_ants(self.session.document.selection)

    def _sync_selection_chrome(self) -> None:
        selection = self.session.document.selection
        if selection is None:
            self._selection_pixmap = None
            self._selection_id = None
            self._sync_ants(None)
            return
        # The tint follows the pixmap: full document, or the visible rectangle
        # only. Identity plus the view key catches a pan without a new mask.
        cache_id = (id(selection), self._pixmap_mode, self._view_key)
        if cache_id != self._selection_id or self._selection_pixmap is None:
            if self._pixmap_mode == "view" and self._view_box is not None and self._pixmap is not None:
                tint = _view_selection_tint(
                    selection,
                    self._view_box,
                    self._pixmap.width(),
                    self._pixmap.height(),
                )
            else:
                tint = _selection_tint(selection)
            self._selection_pixmap = pil_to_pixmap(tint)
            self._selection_id = cache_id
        self._sync_ants(selection)

    def _paint_checker(self, painter: QPainter, target: QRectF) -> None:
        # Cells are 12px, larger than the leftover space on the right and bottom.
        # The page rectangle grows with zoom. Walk only the cells that meet the
        # widget, still on the page-corner grid, so a close zoom does not loop
        # over the off-screen part of the page.
        painter.save()
        painter.setClipRect(target, Qt.ClipOperation.IntersectClip)
        clip = self._viewport_rect()
        span = checker_loop_rect(target, clip, _CHECKER)
        if span is None:
            painter.restore()
            return
        left, top, right, bottom = span
        painter.fillRect(target.intersected(clip), QColor("#1b2026"))
        size = _CHECKER
        c2 = QColor("#2a3038")
        # Same phase as a full-page loop: truncate the corner, so the pattern
        # scrolls with the photo instead of sticking to the widget.
        origin_x = int(target.left())
        origin_y = int(target.top())
        y = top
        while y < bottom:
            x = left
            while x < right:
                if ((x - origin_x) // size + (y - origin_y) // size) % 2 == 0:
                    painter.fillRect(x, y, size, size, c2)
                x += size
            y += size
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
        # The outline is the 50% contour. A hard selection is 0 or 255, so
        # this is the same edge as before. A feather fades through that line.
        mask = _selection_core(np.asarray(selection.convert("L")))
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
        if step <= 0:
            return
        # Ticks are only the document range that can fall inside the ruler.
        # A close zoom used to walk the whole page and skip the off-screen ones.
        view_left, view_top = self.widget_to_canvas(ruler, ruler)
        view_right, view_bottom = self.widget_to_canvas(self.width(), self.height())
        start, end = ruler_tick_span(view_left, view_right, step)
        y_start, y_end = ruler_tick_span(view_top, view_bottom, step)
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
        if not self._has_page():
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        self.zoom_at(event.position().x(), event.position().y(), 1.1 if delta > 0 else 1 / 1.1)

    def _wants_pan_proxy(self) -> bool:
        """A page this large is worth a stand-in while it is dragged."""
        doc = self.session.document
        if doc.width < 1 or doc.height < 1:
            return False
        return max(doc.width, doc.height) > PREVIEW_MAX_EDGE

    def _pan_proxy_size(self, width: int, height: int) -> tuple[int, int]:
        longest = max(int(width), int(height), 1)
        scale = PREVIEW_MAX_EDGE / float(longest)
        return max(1, int(round(width * scale))), max(1, int(round(height * scale)))

    def _full_plate_current(self) -> bool:
        """The sharp cache is the whole page and matches the document."""
        doc = self.session.document
        pixmap = self._pixmap
        return (
            not self._dirty
            and pixmap is not None
            and not pixmap.isNull()
            and self._pixmap_mode == "full"
            and pixmap.width() == doc.width
            and pixmap.height() == doc.height
        )

    def _using_pan_proxy(self) -> bool:
        """A pan or zoom scrub is on screen with its stand-in."""
        proxy = self._pan_proxy
        active = self._panning or self._zoom_scrub
        return active and proxy is not None and not proxy.isNull()

    def _clear_pan_proxy(self) -> None:
        self._pan_proxy = None
        self._pan_selection_proxy = None
        self._pan_proxy_box = None

    def _begin_pan_proxy(self, *, crop: bool = False) -> None:
        """Build the drag plate once. Small pages keep the sharp cache.

        ``crop`` is for a pan. A close zoom loads the reachable document
        rectangle at full resolution. Zoom scrub leaves this false: the zoom
        changes during that drag, so the rectangle is not known at press.
        """
        if not self._wants_pan_proxy():
            return
        self._install_pan_proxy(crop=crop)

    def _install_pan_proxy(self, *, crop: bool = False) -> None:
        """Replace the pan plate.

        A close zoom crops the document rectangle one in-window drag can
        reach, at one plate pixel per document pixel. The full-page cache is
        copied when it is current. Otherwise that rectangle is flattened on
        its own. A visible mask uses ``composite_region`` because a plain
        view flatten skips blur and noise.

        Zoom scrub, and a pan whose reachable rectangle is the whole page or
        longer than ``PAN_CROP_MAX_EDGE``, keep the low-resolution whole-page
        plate. A full-page cache is scaled for that plate. A visible mask
        with no full plate yet is flattened once at full size, then scaled,
        because blur and noise cannot be cropped from a downsample of the
        whole page.
        """
        self._clear_pan_proxy()
        if not self._wants_pan_proxy():
            return
        if crop and self._install_pan_crop():
            return
        doc = self.session.document
        width, height = self._pan_proxy_size(doc.width, doc.height)
        plate = self._pixmap if self._full_plate_current() else None
        if plate is None and shows_redaction(doc):
            # Skip the full-page selection tint. The drag draws a scaled one.
            self._ensure_pixmap(chrome=False)
            plate = self._pixmap if self._full_plate_current() else None
        if plate is not None and not plate.isNull():
            self._pan_proxy = plate.scaled(
                width,
                height,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        else:
            longest = max(doc.width, doc.height, 1)
            image = composite_view(
                doc,
                (0.0, 0.0, float(doc.width), float(doc.height)),
                PREVIEW_MAX_EDGE / float(longest),
            )
            self._pan_proxy = pil_to_pixmap(image)
        self._install_pan_selection()

    def _reachable_pan_box(self) -> tuple[float, float, float, float] | None:
        """Document rectangle a drag that stays in the window can reveal.

        Start from the rectangle on screen. Add one widget of pointer travel
        on each side: the button is down, so the pointer can cross the window
        once, and that is as far as the view can move. A few extra screen
        pixels keep a drag that ends on the window edge inside the crop.
        The result is clamped to the page.
        """
        visible = self._visible_document_box()
        if visible is None or self.zoom <= 0:
            return None
        doc = self.session.document
        pad_x = (float(self.width()) + _PAN_CROP_SLACK) / self.zoom
        pad_y = (float(self.height()) + _PAN_CROP_SLACK) / self.zoom
        x0, y0, x1, y1 = visible
        return (
            max(0.0, x0 - pad_x),
            max(0.0, y0 - pad_y),
            min(float(doc.width), x1 + pad_x),
            min(float(doc.height), y1 + pad_y),
        )

    def _pan_crop_pixels(self) -> tuple[int, int, int, int] | None:
        """Integer pixels of the reachable pan rectangle, or None for the whole page.

        The plate is one document pixel per output pixel, so its long edge is
        the crop's long edge. Floor and ceil keep every document pixel the
        float rectangle touches. A crop of the entire page would be another
        full-size picture; the low-resolution stand-in already covers that.
        A crop longer than ``PAN_CROP_MAX_EDGE`` is the same kind of drag:
        one window of travel still sweeps most of a large photo.
        """
        reached = self._reachable_pan_box()
        if reached is None:
            return None
        doc = self.session.document
        x0, y0, x1, y1 = reached
        ix0 = max(0, int(math.floor(x0)))
        iy0 = max(0, int(math.floor(y0)))
        ix1 = min(doc.width, int(math.ceil(x1)))
        iy1 = min(doc.height, int(math.ceil(y1)))
        if ix1 <= ix0 or iy1 <= iy0:
            return None
        if ix0 == 0 and iy0 == 0 and ix1 == doc.width and iy1 == doc.height:
            return None
        if max(ix1 - ix0, iy1 - iy0) > PAN_CROP_MAX_EDGE:
            return None
        return ix0, iy0, ix1, iy1

    def _install_pan_crop(self) -> bool:
        """Load a 1:1 crop of the reachable page. False keeps the whole-page plate."""
        pixels = self._pan_crop_pixels()
        if pixels is None:
            return False
        ix0, iy0, ix1, iy1 = pixels
        doc = self.session.document
        box = (float(ix0), float(iy0), float(ix1), float(iy1))
        plate = self._pixmap if self._full_plate_current() else None
        if plate is not None and not plate.isNull():
            pixmap = plate.copy(ix0, iy0, ix1 - ix0, iy1 - iy0)
        elif shows_redaction(doc):
            pixmap = pil_to_pixmap(composite_region(doc, pixels))
        else:
            pixmap = pil_to_pixmap(composite_view(doc, box, 1.0))
        if pixmap.isNull() or pixmap.width() < 1 or pixmap.height() < 1:
            return False
        self._pan_proxy = pixmap
        self._pan_proxy_box = box
        self._install_pan_selection()
        return True

    def _extend_pan_crop(self) -> None:
        """Rebuild the crop when the visible rectangle leaves it.

        The plate from press covers one window of travel, so an ordinary
        drag does not get here. The pointer can leave the widget while the
        button stays down. That move loads one new crop around the new view.
        A later move inside the new crop does not load again.
        """
        box = self._pan_proxy_box
        if box is None:
            return
        visible = self._visible_document_box()
        if visible is None or _contains_box(box, visible):
            return
        self._install_pan_proxy(crop=True)

    def _install_pan_selection(self) -> None:
        """Tint the selection at the stand-in's size, not the page's."""
        selection = self.session.document.selection
        proxy = self._pan_proxy
        if selection is None or proxy is None or proxy.isNull():
            self._pan_selection_proxy = None
            self._sync_ants(selection)
            return
        # A crop tints only the preloaded rectangle. The whole-page plate
        # still tints the whole selection, scaled down with the page.
        box = self._pan_proxy_box
        if box is None:
            box = (0.0, 0.0, float(selection.width), float(selection.height))
        tint = _view_selection_tint(
            selection,
            box,
            proxy.width(),
            proxy.height(),
        )
        self._pan_selection_proxy = pil_to_pixmap(tint)
        self._sync_ants(selection)

    def _refresh_pan_proxy(self) -> bool:
        """Rebuild the stand-in after the document changes. False if this drag has none."""
        if self._pan_proxy is None and not self._wants_pan_proxy():
            return False
        # Install clears the box. Remember which plate this drag is using.
        crop = self._pan_proxy_box is not None
        self._dirty = True
        self._selection_id = None
        self._install_pan_proxy(crop=crop)
        if self._pan_proxy is None:
            return False
        self.update()
        return True

    def _end_pan_proxy(self) -> None:
        """Drop the stand-in and show the sharp picture again.

        A view cache's key includes the offset and the zoom, so the cached
        rectangle is the one from before the drag and has to be rebuilt. A
        full-page cache does not, and release only repaints it unless the
        document changed or the zoom now wants the screen-sized cache.
        """
        had = self._pan_proxy is not None
        self._clear_pan_proxy()
        if not had:
            return
        if self._pixmap_mode == "view" or self._dirty or self._use_view_cache():
            self._dirty = True
            self._selection_id = None
        elif self._selection_id is None:
            self._sync_selection_chrome()
        self.update()

    def _pan_proxy_dest(self, full_target: QRectF) -> QRectF:
        """Widget rectangle the pan plate covers.

        A whole-page stand-in is the page. A close-zoom crop is only the
        reachable document rectangle, so the draw does not target page times zoom.
        """
        box = self._pan_proxy_box
        if box is None:
            return full_target
        x0, y0, x1, y1 = box
        ruler = self.ruler()
        return QRectF(
            ruler + self.offset_x + x0 * self.zoom,
            ruler + self.offset_y + y0 * self.zoom,
            (x1 - x0) * self.zoom,
            (y1 - y0) * self.zoom,
        )

    def _paint_pan_proxy(self, painter: QPainter, target: QRectF) -> None:
        proxy = self._pan_proxy
        if proxy is None or proxy.isNull():
            return
        # The plate may cover the whole page. Draw only the part inside the
        # window, mapped back into the plate, so a close zoom does not ask Qt
        # to scale into a rectangle the size of the page times the zoom.
        dest = self._pan_proxy_dest(target)
        pair = plate_visible_source(dest, self._viewport_rect(), proxy.width(), proxy.height())
        if pair is None:
            return
        visible, source = pair
        # A 1:1 crop is the original pixels. Nearest-neighbor matches the
        # resting view. The whole-page stand-in is coarser than the screen,
        # so smooth that stretch. Rulers are drawn after this.
        painter.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform,
            self._pan_proxy_box is None,
        )
        painter.drawPixmap(visible, proxy, source)
        selection = self._pan_selection_proxy
        if selection is not None and not selection.isNull():
            selected = plate_visible_source(
                dest,
                self._viewport_rect(),
                selection.width(),
                selection.height(),
            )
            if selected is not None:
                painter.drawPixmap(selected[0], selection, selected[1])
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

    def _begin_zoom(self, pos, *, alt: bool) -> None:
        """Remember where a zoom-tool press started. The step waits for release."""
        # Space is the temporary hand. That hold does not start a zoom drag.
        if self._space:
            return
        self._zooming = True
        self._zoom_scrub = False
        self._zoom_anchor = (float(pos.x()), float(pos.y()))
        self._zoom_origin = self.zoom
        self._zoom_alt = alt

    def _stop_zoom_for_hand(self) -> None:
        """Space ends a zoom drag at the level it has already reached.

        The drag is not a click, so the zoom does not step, and it is not
        Escape, so the level is not put back. The stand-in is dropped. The
        caller pans when a button is still down.
        """
        if not self._zooming:
            return
        self._zooming = False
        self._zoom_scrub = False
        self._zoom_anchor = None
        self._space_pan_pending = True
        self._end_pan_proxy()

    def _begin_space_pan(self, buttons, pos=None) -> None:
        """Pan for the rest of a button hold that Space took over.

        Releasing Space does not clear this pan and does not hand the drag
        back to the zoom tool. The button coming up ends it, including the
        right button, so the next press can zoom again.
        """
        if self._panning or not self._has_page():
            return
        if not _pan_button_held(buttons):
            return
        if pos is None:
            local = self.mapFromGlobal(QCursor.pos())
            pos = QPointF(local)
        self._panning = True
        self._pan_last = pos
        self._space_pan_pending = False
        self._begin_pan_proxy(crop=True)
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _move_zoom(self, pos) -> None:
        """Scrub zoom from the press point. The low-res plate covers the drag.

        The plate is built on the first real move, before the zoom changes, so
        later moves only scale that picture. Release sharpens whatever zoom
        the drag reached.
        """
        anchor = self._zoom_anchor
        if not self._zooming or anchor is None:
            return
        ax, ay = anchor
        dy = ay - float(pos.y())
        if not self._zoom_scrub:
            if abs(dy) < _ZOOM_DRAG_THRESHOLD:
                return
            # Whole-page stand-in first. zoom_at would otherwise resample.
            self._zoom_scrub = True
            self._begin_pan_proxy()
        factor = 2.0 ** (dy / _ZOOM_DRAG_SPAN)
        target = min(32.0, max(0.02, self._zoom_origin * factor))
        if abs(target - self.zoom) < 1e-4:
            self.update()
            return
        self.zoom_at(ax, ay, target / self.zoom)

    def _finish_zoom(self, event) -> None:
        """A click steps once. A drag keeps the scrubbed zoom, then sharpens."""
        anchor = self._zoom_anchor
        scrubbed = self._zoom_scrub
        alt = self._zoom_alt or bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
        self._zooming = False
        self._zoom_scrub = False
        self._zoom_anchor = None
        if not scrubbed and anchor is not None:
            self.zoom_at(anchor[0], anchor[1], (1 / 1.25) if alt else 1.25)
        self._end_pan_proxy()

    def _cancel_zoom(self) -> None:
        """Escape puts the zoom back and drops the stand-in."""
        anchor = self._zoom_anchor
        origin = self._zoom_origin
        self._zooming = False
        self._zoom_scrub = False
        self._zoom_anchor = None
        if anchor is not None and abs(self.zoom - origin) > 1e-6:
            self.zoom_at(anchor[0], anchor[1], origin / self.zoom)
        self._end_pan_proxy()

    # Panning a large page builds one plate and slides it until the button
    # comes up. Close zoom crops the rectangle one window of travel can reach,
    # at full resolution. A wider drag, and a zoom-tool scrub, use one
    # whole-page plate of PREVIEW_MAX_EDGE: scrub changes the zoom, so the
    # document rectangle is not known at press. A full-page cache is copied
    # or scaled rather than flattened again, and release keeps it while the
    # zoom is still at 100% or more. A view cache is dropped on release
    # because the offset and zoom are part of its key. A document change
    # during the drag rebuilds the plate.
    def mousePressEvent(self, event) -> None:
        self.setFocus()
        if not self._has_page():
            return
        pos = event.position()
        if event.button() == Qt.MouseButton.MiddleButton or self._space or self._is_hand():
            self._panning = True
            self._pan_last = pos
            self._begin_pan_proxy(crop=True)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton and self._begin_guide(pos):
            return
        tool = self.tool_getter()
        if tool is not None and tool.id == "zoom" and event.button() == Qt.MouseButton.LeftButton:
            alt = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
            self._begin_zoom(pos, alt=alt)
            return
        if tool is not None and event.button() == Qt.MouseButton.LeftButton:
            tool.pointer_down(self.session, self._pointer(event))
            if not self._space and tool.wants_stroke_sampling(self.session):
                self._start_stroke_sampling(tool.id)
            self._sync_hover_cursor(event)

    def mouseMoveEvent(self, event) -> None:
        if not self._has_page():
            return
        pos = event.position()
        self._report_status(pos.x(), pos.y())
        # The key may have gone down while another widget had the keyboard.
        self.set_alt_held(bool(event.modifiers() & Qt.KeyboardModifier.AltModifier))
        # Space took over a zoom drag. A button that is still down pans.
        # A brush or other tool drag is left alone until the next press.
        if self._space and self._space_pan_pending and not self._panning:
            self._begin_space_pan(event.buttons(), pos)
        if self._panning and self._pan_last is not None:
            # Slide the plate. An in-window drag stays inside the crop loaded
            # at press. Leaving the window can walk off that crop; rebuild then.
            self.offset_x += pos.x() - self._pan_last.x()
            self.offset_y += pos.y() - self._pan_last.y()
            self._pan_last = pos
            self._extend_pan_crop()
            self.update()
            return
        if self._zooming and not self._space and event.buttons() & Qt.MouseButton.LeftButton:
            self._move_zoom(pos)
            return
        if self._guide is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self._move_guide(pos)
            return
        self._sync_hover_cursor(event)
        # A sampled stroke (brush, eraser, freehand Select) stores this move
        # and returns. The timer paints the batch. pointer_move would stamp
        # now, and the timer would stamp the same points again.
        if self._stroke_sampling:
            if event.buttons() & Qt.MouseButton.LeftButton:
                self._remember_cursor(event)
            return
        if event.buttons() & Qt.MouseButton.LeftButton:
            tool = self.tool_getter()
            if tool is not None and tool.id not in {"hand", "zoom"}:
                tool.pointer_move(self.session, self._pointer(event))

    def mouseReleaseEvent(self, event) -> None:
        if not self._has_page():
            return
        # The hold Space might take over ends with the button. A later move
        # must not start a pan from a zoom drag that already finished.
        self._space_pan_pending = False
        if self._guide is not None and event.button() == Qt.MouseButton.LeftButton:
            self._finish_guide(event.position())
            return
        if self._panning and event.button() in {
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
        }:
            self._panning = False
            self._pan_last = None
            self._end_pan_proxy()
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
        if self._zooming and event.button() == Qt.MouseButton.LeftButton:
            self._finish_zoom(event)
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
            if event.isAutoRepeat():
                return
            # Space is only the temporary hand. A zoom drag in progress keeps
            # the level it has reached and pans for the rest of the button hold.
            self._space = True
            self._stop_zoom_for_hand()
            self._begin_space_pan(QApplication.mouseButtons())
            if not self._panning:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        if event.key() == Qt.Key.Key_Escape:
            if self._guide is not None:
                self._guide = None
                self.update()
                event.accept()
                return
            if self._zooming:
                self._cancel_zoom()
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
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            # The drag keeps panning until the button comes up. Letting Space
            # go first must not turn that same drag into a zoom.
            self._space = False
            self._space_pan_pending = False
            self._apply_cursor()
            return
        super().keyReleaseEvent(event)

    def _start_stroke_sampling(self, tool_id: str) -> None:
        self._stroke_sampling = True
        self._stroke_tool_id = tool_id
        self._stroke_points = []
        self._stroke_timer.start(_STROKE_SAMPLE_MS)

    def _cancel_stroke_sampling(self) -> None:
        self._stroke_sampling = False
        self._stroke_timer.stop()
        self._stroke_points = []

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
        # Escape or a tool change already restored the layer. A release the
        # canvas missed still flushes the stored moves and commits. While the
        # button is down this tick only paints moves the pointer already stored.
        if not live:
            self._cancel_stroke_sampling()
            return
        if not button_down:
            self._finish_stroke_sampling(None)
            return
        self._flush_stroke_points()
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
        # A zoomed-out cache is screen-sized, so a document-pixel patch would
        # land in the wrong place. The caller rebuilds that small cache instead.
        if self._pixmap_mode != "full":
            return False
        if self._pixmap.width() != doc.width or self._pixmap.height() != doc.height:
            return False
        # Noise grain is one field the size of the page. Other masks can be
        # patched: composite_region pads the sample and crops it back.
        if redaction_padding(doc) is None:
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


def plate_visible_source(
    dest: QRectF,
    clip: QRectF,
    plate_width: float,
    plate_height: float,
) -> tuple[QRectF, QRectF] | None:
    """Widget rect and plate rect for the part of ``dest`` inside ``clip``.

    ``dest`` is where the whole plate would be drawn. A zoom drag makes that
    the page times the zoom, far larger than the window. The pair is only the
    overlap, so the draw stays the size of the window.
    """
    if dest.width() <= 0 or dest.height() <= 0 or plate_width <= 0 or plate_height <= 0:
        return None
    visible = dest.intersected(clip)
    if visible.isEmpty():
        return None
    source = QRectF(
        plate_width * (visible.left() - dest.left()) / dest.width(),
        plate_height * (visible.top() - dest.top()) / dest.height(),
        plate_width * visible.width() / dest.width(),
        plate_height * visible.height() / dest.height(),
    )
    if source.width() <= 0 or source.height() <= 0:
        return None
    return visible, source


def ruler_tick_span(view_min: float, view_max: float, step: float) -> tuple[float, float]:
    """First and last tick that can fall inside the view, snapped to ``step``.

    The span is the document range on screen, not the whole page. Ticks are
    aligned to the same grid a full walk would use, so the marks that show
    stay put.
    """
    if step <= 0:
        return view_min, view_max
    if view_max < view_min:
        view_min, view_max = view_max, view_min
    start = math.floor(view_min / step) * step
    end = math.ceil(view_max / step) * step
    return start, end


def _pan_button_held(buttons) -> bool:
    """True when the left, middle, or right button is in ``buttons``."""
    held = Qt.MouseButton.LeftButton | Qt.MouseButton.MiddleButton | Qt.MouseButton.RightButton
    return bool(buttons & held)


def checker_loop_rect(
    target: QRectF,
    clip: QRectF,
    cell: int = _CHECKER,
) -> tuple[int, int, int, int] | None:
    """Pixel span of the checker cells that meet ``clip``.

    Cells are ``cell`` pixels on the grid that starts at the page corner
    (``target`` floored), the same grid a full-page loop would walk. A close
    zoom makes ``target`` enormous. Walking every cell of it is what stalled
    a pan, so the span is only the cells that touch ``clip``. The phase still
    follows the page, and the pattern scrolls with the photo.
    """
    if cell < 1:
        return None
    inter = target.intersected(clip)
    if inter.isEmpty():
        return None
    grid_x = int(math.floor(target.left()))
    grid_y = int(math.floor(target.top()))
    left = grid_x + ((int(math.floor(inter.left())) - grid_x) // cell) * cell
    top = grid_y + ((int(math.floor(inter.top())) - grid_y) // cell) * cell
    right = grid_x + _grid_ceil(int(math.ceil(inter.right())) - grid_x, cell)
    bottom = grid_y + _grid_ceil(int(math.ceil(inter.bottom())) - grid_y, cell)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _grid_ceil(delta: int, cell: int) -> int:
    """Smallest multiple of ``cell`` that is >= ``delta``."""
    return ((delta + cell - 1) // cell) * cell


def _contains_box(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
) -> bool:
    """True when ``inner`` sits inside ``outer``, aside from float noise."""
    x0, y0, x1, y1 = outer
    ix0, iy0, ix1, iy1 = inner
    return ix0 >= x0 - 1e-3 and iy0 >= y0 - 1e-3 and ix1 <= x1 + 1e-3 and iy1 <= y1 + 1e-3


def _span_pixels(span: float, scale: float) -> int:
    """Screen pixels covering ``span`` document pixels, never above one per source pixel.

    Matches ``composite._view_size`` so a view-sized selection tint lines up
    with the plates.
    """
    if span <= 0 or scale <= 0:
        return 1
    return max(1, min(int(math.ceil(span)), int(round(span * scale))))


def _split_around_selection(layers: list, selected_ids: set[str]) -> tuple[list, list]:
    """Pictures under the selection, and pictures over it.

    A layer sitting between two selected layers stays on the lower plate, so
    it does not move with the drag and paints under the whole selection until
    the handle is released.
    """
    indexes = [index for index, layer in enumerate(layers) if layer.id in selected_ids]
    if not indexes:
        return [], []
    hi = max(indexes)
    below = [layer for layer in layers[: hi + 1] if layer.id not in selected_ids]
    above = [layer for layer in layers[hi + 1 :] if layer.id not in selected_ids]
    return below, above


def _plate_members(all_layers: list, plate_layers: list) -> list:
    """``plate_layers`` plus the masks that filter them, in stack order.

    A mask that lives on the other side of the selection still belongs to its
    picture. Leaving it on the opposite plate would attach it to the wrong layer.
    """
    content_ids = {layer.id for layer in plate_layers if layer.kind != "redaction"}
    wanted = set(content_ids)
    for layer in all_layers:
        if layer.kind != "redaction":
            continue
        owner = mask_target(all_layers, layer)
        if owner is not None and owner.id in content_ids:
            wanted.add(layer.id)
    return [layer for layer in all_layers if layer.id in wanted]


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


def _selection_core(mask: np.ndarray) -> np.ndarray:
    """True where coverage is at least half. The marching outline follows this edge."""
    return mask >= 128


def _view_selection_tint(
    selection: Image.Image,
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> Image.Image:
    """Tint the visible part of a selection at the view cache's size."""
    x0, y0, x1, y1 = box
    if width < 1 or height < 1:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    sx0 = min(max(0.0, x0), float(selection.width))
    sy0 = min(max(0.0, y0), float(selection.height))
    sx1 = min(max(sx0, x1), float(selection.width))
    sy1 = min(max(sy0, y1), float(selection.height))
    if sx1 <= sx0 or sy1 <= sy0:
        return Image.new("RGBA", (width, height), (0, 0, 0, 0))
    mask = selection.convert("L").resize(
        (width, height),
        Image.Resampling.BOX,
        box=(sx0, sy0, sx1, sy1),
    )
    return _selection_tint(mask)


def _selection_tint(selection: Image.Image) -> Image.Image:
    mask = np.asarray(selection.convert("L"))
    height, width = mask.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    # The tint follows coverage, so a feather reads as a fade. A hard
    # selection is 0 or 255, which keeps the same 70-alpha blue as before.
    visible = mask > 0
    rgba[visible, 0] = 70
    rgba[visible, 1] = 170
    rgba[visible, 2] = 255
    rgba[visible, 3] = (mask[visible].astype(np.uint16) * 70 // 255).astype(np.uint8)
    # One-pixel edge so the 50% boundary stays readable over a busy photo.
    on = _selection_core(mask)
    eroded = on.copy()
    eroded[1:, :] &= on[:-1, :]
    eroded[:-1, :] &= on[1:, :]
    eroded[:, 1:] &= on[:, :-1]
    eroded[:, :-1] &= on[:, 1:]
    edge = on & ~eroded
    rgba[edge] = (236, 248, 255, 220)
    return Image.fromarray(rgba, "RGBA")

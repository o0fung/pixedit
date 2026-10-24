"""Canvas pointers drawn as small high-contrast glyphs.

Toolbar icons are light gray for the dark chrome, and their center is not the
click. These are white with a black edge so they stay visible on a photo, and
the hotspot is the part that acts: the dropper tip, the bucket drip, or the
corner a drag starts from.
"""

from __future__ import annotations

import math

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QCursor, QPainter, QPainterPath, QPainterPathStroker, QPen, QPixmap

# Logical pixels. The pixmap is scaled by the screen's device pixel ratio.
_LOGICAL = 32

_INK = QColor(255, 255, 255, 245)
_EDGE = QColor(0, 0, 0, 230)

# name -> cached cursor, keyed again by the screen scale.
_CACHE: dict[tuple[str, int], QCursor] = {}


def tool_pixmap_cursor(name: str, *, device_pixel_ratio: float = 1.0) -> QCursor | None:
    """A drawn pointer, or None when ``name`` is a system cursor.

    The same name and screen scale reuse one pixmap. Unknown names return
    None so the canvas can fall back to a system pointer.
    """
    painter = _PAINTERS.get(name)
    if painter is None:
        return None
    dpr = max(1.0, float(device_pixel_ratio))
    key = (name, int(round(dpr * 100)))
    cached = _CACHE.get(key)
    if cached is None:
        cached = painter(dpr)
        _CACHE[key] = cached
    return cached


def _canvas(dpr: float) -> tuple[QPixmap, QPainter]:
    side = max(1, int(round(_LOGICAL * dpr)))
    pixmap = QPixmap(side, side)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.scale(dpr, dpr)
    return pixmap, painter


def _finish(pixmap: QPixmap, painter: QPainter, dpr: float, hot_x: float, hot_y: float) -> QCursor:
    painter.end()
    pixmap.setDevicePixelRatio(dpr)
    # Hotspot is in logical pixels. Qt applies the pixmap's device pixel ratio.
    return QCursor(pixmap, int(round(hot_x)), int(round(hot_y)))


def _pen(color: QColor, width: float, *, dashed: bool = False) -> QPen:
    pen = QPen(color)
    pen.setWidthF(width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    if dashed:
        # Dash length is in units of the pen width, so both passes share one width.
        pen.setDashPattern([2.1, 1.55])
    return pen


def _stroke(painter: QPainter, path: QPainterPath, *, dashed: bool = False) -> None:
    """White stroke with a black edge. Offsets keep dash gaps transparent."""
    painter.setBrush(Qt.BrushStyle.NoBrush)
    width = 1.7
    painter.setPen(_pen(_EDGE, width, dashed=dashed))
    for dx, dy in ((0.85, 0.0), (-0.85, 0.0), (0.0, 0.85), (0.0, -0.85)):
        painter.save()
        painter.translate(dx, dy)
        painter.drawPath(path)
        painter.restore()
    painter.setPen(_pen(_INK, width, dashed=dashed))
    painter.drawPath(path)


def _fill(painter: QPainter, path: QPainterPath) -> None:
    """Filled glyph with a black rim so the silhouette reads on any canvas."""
    stroker = QPainterPathStroker()
    stroker.setWidth(2.4)
    stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
    stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(_EDGE)
    painter.drawPath(stroker.createStroke(path).united(path))
    painter.setBrush(_INK)
    painter.drawPath(path)


def _aim(painter: QPainter, x: float, y: float) -> None:
    """Solid dot on the hotspot. The click has to land on ink, not a gap."""
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(_EDGE)
    painter.drawEllipse(QRectF(x - 2.05, y - 2.05, 4.1, 4.1))
    painter.setBrush(_INK)
    painter.drawEllipse(QRectF(x - 1.15, y - 1.15, 2.3, 2.3))


def _paint_eyedrop(dpr: float) -> QCursor:
    # Tip is the lower-left point. The bulb sits up and to the right.
    hot_x, hot_y = 6.0, 26.0
    pixmap, painter = _canvas(dpr)
    tip = QPainterPath()
    tip.moveTo(hot_x, hot_y)
    tip.lineTo(12.4, 19.2)
    tip.lineTo(9.6, 17.6)
    tip.closeSubpath()
    tube = QPainterPath()
    tube.moveTo(10.2, 19.6)
    tube.lineTo(19.6, 10.2)
    tube.lineTo(16.6, 7.2)
    tube.lineTo(7.2, 16.6)
    tube.closeSubpath()
    bulb = QPainterPath()
    bulb.addEllipse(QRectF(15.0, 2.2, 11.4, 11.4))
    _fill(painter, tip.united(tube).united(bulb))
    _aim(painter, hot_x, hot_y)
    return _finish(pixmap, painter, dpr, hot_x, hot_y)


def _paint_bucket(dpr: float) -> QCursor:
    # The drip is the pixel that gets filled.
    hot_x, hot_y = 7.0, 25.5
    pixmap, painter = _canvas(dpr)
    body = QPainterPath()
    body.moveTo(14.0, 11.0)
    body.lineTo(27.2, 11.0)
    body.lineTo(24.4, 21.2)
    body.quadTo(20.4, 24.0, 16.6, 21.2)
    body.closeSubpath()
    spout = QPainterPath()
    spout.moveTo(14.0, 12.4)
    spout.lineTo(14.0, 17.2)
    spout.lineTo(hot_x + 1.2, hot_y - 3.4)
    spout.closeSubpath()
    drip = QPainterPath()
    drip.addEllipse(QRectF(hot_x - 2.5, hot_y - 2.7, 5.2, 5.6))
    _fill(painter, body.united(spout).united(drip))
    handle = QPainterPath()
    handle.moveTo(16.2, 11.0)
    handle.cubicTo(16.2, 4.6, 25.4, 4.6, 25.4, 11.0)
    _stroke(painter, handle)
    _aim(painter, hot_x, hot_y)
    return _finish(pixmap, painter, dpr, hot_x, hot_y)


def _paint_shape_rect(dpr: float) -> QCursor:
    # Drag starts at the top-left corner.
    hot_x, hot_y = 5.0, 5.0
    pixmap, painter = _canvas(dpr)
    path = QPainterPath()
    path.addRect(QRectF(hot_x, hot_y, 17.0, 13.0))
    _stroke(painter, path)
    _aim(painter, hot_x, hot_y)
    return _finish(pixmap, painter, dpr, hot_x, hot_y)


def _paint_shape_ellipse(dpr: float) -> QCursor:
    # The click is the upper-left of the ellipse, where a corner drag starts.
    rect = QRectF(5.0, 5.0, 18.0, 14.0)
    angle = math.radians(225.0)
    hot_x = rect.center().x() + rect.width() / 2.0 * math.cos(angle)
    hot_y = rect.center().y() + rect.height() / 2.0 * math.sin(angle)
    pixmap, painter = _canvas(dpr)
    path = QPainterPath()
    path.addEllipse(rect)
    _stroke(painter, path)
    _aim(painter, hot_x, hot_y)
    return _finish(pixmap, painter, dpr, hot_x, hot_y)


def _paint_shape_line(dpr: float) -> QCursor:
    hot_x, hot_y = 6.0, 25.0
    pixmap, painter = _canvas(dpr)
    path = QPainterPath()
    path.moveTo(hot_x, hot_y)
    path.lineTo(26.0, 6.0)
    _stroke(painter, path)
    _aim(painter, hot_x, hot_y)
    return _finish(pixmap, painter, dpr, hot_x, hot_y)


def _paint_select_rect(dpr: float) -> QCursor:
    # Dashed, so a selection does not look like the shape tool's solid rectangle.
    hot_x, hot_y = 5.0, 5.0
    pixmap, painter = _canvas(dpr)
    path = QPainterPath()
    path.addRect(QRectF(hot_x, hot_y, 17.0, 13.0))
    _stroke(painter, path, dashed=True)
    _aim(painter, hot_x, hot_y)
    return _finish(pixmap, painter, dpr, hot_x, hot_y)


def _paint_select_ellipse(dpr: float) -> QCursor:
    rect = QRectF(5.0, 5.0, 18.0, 14.0)
    angle = math.radians(225.0)
    hot_x = rect.center().x() + rect.width() / 2.0 * math.cos(angle)
    hot_y = rect.center().y() + rect.height() / 2.0 * math.sin(angle)
    pixmap, painter = _canvas(dpr)
    path = QPainterPath()
    path.addEllipse(rect)
    _stroke(painter, path, dashed=True)
    _aim(painter, hot_x, hot_y)
    return _finish(pixmap, painter, dpr, hot_x, hot_y)


def _paint_lasso(dpr: float) -> QCursor:
    # The cord ends at the click. The loop shows a freehand region.
    hot_x, hot_y = 6.5, 26.0
    pixmap, painter = _canvas(dpr)
    cord = QPainterPath()
    cord.moveTo(hot_x, hot_y)
    cord.quadTo(9.0, 18.0, 14.5, 15.5)
    loop = QPainterPath()
    loop.addEllipse(QRectF(11.0, 3.5, 14.5, 12.5))
    _stroke(painter, cord)
    _stroke(painter, loop)
    _aim(painter, hot_x, hot_y)
    return _finish(pixmap, painter, dpr, hot_x, hot_y)


def _star(cx: float, cy: float, outer: float, inner: float, rotation: float) -> QPainterPath:
    path = QPainterPath()
    for index in range(10):
        radius = outer if index % 2 == 0 else inner
        angle = rotation + index * math.pi / 5.0
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        if index == 0:
            path.moveTo(x, y)
        else:
            path.lineTo(x, y)
    path.closeSubpath()
    return path


def _paint_wand(dpr: float) -> QCursor:
    # One star point is the click. The handle runs away from that point.
    center_x, center_y = 13.0, 12.0
    outer = 7.2
    tip = math.radians(225.0)
    hot_x = center_x + outer * math.cos(tip)
    hot_y = center_y + outer * math.sin(tip)
    pixmap, painter = _canvas(dpr)
    _fill(painter, _star(center_x, center_y, outer, 3.0, tip))
    handle = QPainterPath()
    handle.moveTo(
        center_x + outer * 0.55 * math.cos(tip + math.pi),
        center_y + outer * 0.55 * math.sin(tip + math.pi),
    )
    handle.lineTo(28.0, 27.0)
    _stroke(painter, handle)
    _aim(painter, hot_x, hot_y)
    return _finish(pixmap, painter, dpr, hot_x, hot_y)


def _paint_zoom(dpr: float, *, minus: bool) -> QCursor:
    # The click is the center of the lens, which is where the view zooms.
    hot_x, hot_y = 11.0, 11.0
    pixmap, painter = _canvas(dpr)
    lens = QPainterPath()
    lens.addEllipse(QRectF(3.0, 3.0, 16.0, 16.0))
    handle = QPainterPath()
    handle.moveTo(16.6, 16.6)
    handle.lineTo(27.5, 27.5)
    mark = QPainterPath()
    mark.moveTo(6.6, hot_y)
    mark.lineTo(15.4, hot_y)
    if not minus:
        mark.moveTo(hot_x, 6.6)
        mark.lineTo(hot_x, 15.4)
    _stroke(painter, lens)
    _stroke(painter, handle)
    _stroke(painter, mark)
    _aim(painter, hot_x, hot_y)
    return _finish(pixmap, painter, dpr, hot_x, hot_y)


def _paint_zoom_in(dpr: float) -> QCursor:
    return _paint_zoom(dpr, minus=False)


def _paint_zoom_out(dpr: float) -> QCursor:
    return _paint_zoom(dpr, minus=True)


def _paint_rotate(dpr: float) -> QCursor:
    # The center sits on the rotate handle above the box.
    hot_x, hot_y = 16.0, 16.0
    pixmap, painter = _canvas(dpr)
    arc = QPainterPath()
    rect = QRectF(6.5, 6.5, 19.0, 19.0)
    arc.arcMoveTo(rect, 40)
    arc.arcTo(rect, 40, 270)
    _stroke(painter, arc)
    # Qt angles start at 3 o'clock and grow counter-clockwise, with 90° up.
    end = math.radians(40.0 + 270.0)
    tip_x = rect.center().x() + rect.width() / 2.0 * math.cos(end)
    tip_y = rect.center().y() - rect.height() / 2.0 * math.sin(end)
    tangent_x = -math.sin(end)
    tangent_y = -math.cos(end)
    side_x = -tangent_y
    side_y = tangent_x
    head = QPainterPath()
    head.moveTo(tip_x, tip_y)
    head.lineTo(tip_x - tangent_x * 5.0 + side_x * 3.1, tip_y - tangent_y * 5.0 + side_y * 3.1)
    head.lineTo(tip_x - tangent_x * 5.0 - side_x * 3.1, tip_y - tangent_y * 5.0 - side_y * 3.1)
    head.closeSubpath()
    _fill(painter, head)
    _aim(painter, hot_x, hot_y)
    return _finish(pixmap, painter, dpr, hot_x, hot_y)


def _paint_crop(dpr: float) -> QCursor:
    # Two corner brackets. The click is the inner corner of the first one.
    hot_x, hot_y = 6.0, 6.0
    pixmap, painter = _canvas(dpr)
    near = QPainterPath()
    near.moveTo(hot_x, 17.0)
    near.lineTo(hot_x, hot_y)
    near.lineTo(17.0, hot_y)
    far = QPainterPath()
    far.moveTo(15.0, 26.0)
    far.lineTo(26.0, 26.0)
    far.lineTo(26.0, 15.0)
    _stroke(painter, near)
    _stroke(painter, far)
    _aim(painter, hot_x, hot_y)
    return _finish(pixmap, painter, dpr, hot_x, hot_y)


_PAINTERS = {
    "eyedrop": _paint_eyedrop,
    "bucket": _paint_bucket,
    "shape-rect": _paint_shape_rect,
    "shape-ellipse": _paint_shape_ellipse,
    "shape-line": _paint_shape_line,
    "select-rect": _paint_select_rect,
    "select-ellipse": _paint_select_ellipse,
    "lasso": _paint_lasso,
    "wand": _paint_wand,
    "zoom-in": _paint_zoom_in,
    "zoom-out": _paint_zoom_out,
    "crop": _paint_crop,
    "rotate": _paint_rotate,
}

GLYPH_CURSORS = tuple(_PAINTERS)

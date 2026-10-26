"""Shape tool. Drag a rectangle, ellipse, or line."""

from __future__ import annotations

import math

from PIL import Image

from pixedit.geom import (
    clip_box,
    grow_layer_to_cover,
    selection_for_layer,
)
from pixedit.guides import snap_point
from pixedit.model import Layer
from pixedit.paint import draw_shape
from pixedit.tools.base import (
    PointerEvent,
    Tool,
)


class ShapeTool(Tool):
    id = "shape"
    label = "Shape"
    shortcut = "U"
    cursor = "shape-rect"
    hint = "Drag a rectangle, ellipse, or line. The canvas shows the shape, fill, and stroke while you drag."

    def __init__(self) -> None:
        self._dragging = False
        self._start = (0.0, 0.0)
        self._end = (0.0, 0.0)
        self._layer: Layer | None = None
        self._before: dict | None = None
        self._created = False
        self._snap_image: Image.Image | None = None
        self._snap_xy = (0, 0)

    def option_schema(self):
        return [
            {
                "key": "shape_kind",
                "label": "Shape",
                "type": "choice",
                "icons": True,
                "choices": [
                    ("rect", "Rectangle", "geom-rect"),
                    ("ellipse", "Ellipse", "geom-ellipse"),
                    ("line", "Line", "geom-line"),
                ],
                "default": "rect",
            },
            {"key": "stroke_width", "label": "Stroke", "type": "int", "min": 1, "max": 64, "default": 4},
            {"key": "shape_filled", "label": "Fill shape?", "type": "bool", "default": False},
        ]

    def cursor_name(self, session) -> str:
        kind = str(session.option("shape_kind", "rect"))
        if kind == "ellipse":
            return "shape-ellipse"
        if kind == "line":
            return "shape-line"
        return "shape-rect"

    def pointer_down(self, session, event: PointerEvent) -> None:
        self._cancel_edit(session)
        self._dragging = True
        self._start = self._snap(session, event.x, event.y)
        self._end = self._start

    def pointer_move(self, session, event: PointerEvent) -> None:
        if not self._dragging:
            return
        self._end = self._snap(session, event.x, event.y)
        if math.hypot(self._end[0] - self._start[0], self._end[1] - self._start[1]) < 1:
            return
        self._show_preview(session)

    def pointer_up(self, session, event: PointerEvent) -> None:
        if not self._dragging:
            return
        self._end = self._snap(session, event.x, event.y)
        self._dragging = False
        if math.hypot(self._end[0] - self._start[0], self._end[1] - self._start[1]) < 1:
            self._cancel_edit(session)
            session.preview()
            return
        box = self._canvas_box(session)
        if box is None:
            self._cancel_edit(session)
            session.show_message("The shape is outside the canvas.")
            session.preview()
            return
        # The preview pass paints with the same draw used at commit time, so
        # release does not change fill, stroke, color, or alpha.
        self._show_preview(session)
        layer, before, created = self._take_edit()
        if layer is None:
            return
        session.finish_raster_edit(layer, before, created, "Shape")

    def pointer_cancel(self, session) -> bool:
        active = self._dragging or self._layer is not None
        self._dragging = False
        self._cancel_edit(session)
        return active

    def _ensure_edit(self, session) -> None:
        if self._layer is not None:
            return
        layer, before, created = session.begin_raster_edit(paint_redaction=False)
        self._layer = layer
        self._before = before
        self._created = created
        # Snapshot after text is rasterized. Each move restores this and
        # redraws, so dragging does not stack copies of the shape.
        self._snap_image = layer.image.copy()
        self._snap_xy = (layer.x, layer.y)

    def _restore_snap(self) -> None:
        if self._layer is None or self._snap_image is None:
            return
        self._layer.image = self._snap_image.copy()
        self._layer.x, self._layer.y = self._snap_xy

    def _paint(self, session, box: tuple[int, int, int, int]) -> None:
        layer = self._layer
        if layer is None:
            return
        # Grow to the stroke's canvas box first so a small layer can hold it.
        # finish_raster_edit then crops back to the opaque pixels.
        grow_layer_to_cover(layer, box)
        kind = str(session.option("shape_kind", "rect"))
        selection = selection_for_layer(session.document, layer)
        ox, oy = float(layer.x), float(layer.y)
        layer.image = draw_shape(
            layer.image,
            kind,
            (self._start[0] - ox, self._start[1] - oy, self._end[0] - ox, self._end[1] - oy),
            session.foreground,
            int(session.option("stroke_width", 4)),
            bool(session.option("shape_filled", False)),
            selection,
            line=((self._start[0] - ox, self._start[1] - oy), (self._end[0] - ox, self._end[1] - oy)),
        )

    def _show_preview(self, session) -> None:
        self._ensure_edit(session)
        self._restore_snap()
        box = self._canvas_box(session)
        if box is None:
            session.preview()
            return
        self._paint(session, box)
        session.preview()

    def _take_edit(self) -> tuple[Layer | None, dict | None, bool]:
        layer, before, created = self._layer, self._before, self._created
        self._layer = None
        self._before = None
        self._created = False
        self._snap_image = None
        return layer, before, created

    def _cancel_edit(self, session) -> None:
        layer, before, created = self._take_edit()
        if layer is None:
            return
        session.cancel_raster_edit(layer, before, created)

    def _snap(self, session, x: float, y: float) -> tuple[float, float]:
        group = session.document.group
        return snap_point(x, y, group.guides_v, group.guides_h, bool(session.option("snap", True)))

    def _canvas_box(self, session) -> tuple[int, int, int, int] | None:
        """Half-open document box covering the shape, including its stroke."""
        stroke = int(session.option("stroke_width", 4))
        pad = max(1, stroke)
        x0 = math.floor(min(self._start[0], self._end[0])) - pad
        y0 = math.floor(min(self._start[1], self._end[1])) - pad
        x1 = math.ceil(max(self._start[0], self._end[0])) + pad + 1
        y1 = math.ceil(max(self._start[1], self._end[1])) + pad + 1
        return clip_box((int(x0), int(y0), int(x1), int(y1)), session.document.width, session.document.height)

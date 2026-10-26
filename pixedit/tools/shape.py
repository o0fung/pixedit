"""Shape tool. Drag a rectangle, ellipse, or line."""

from __future__ import annotations

import math

from PIL import Image

from pixedit.geom import (
    clip_box,
    grow_layer_to_cover,
    selection_for_layer,
    union_boxes,
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
        self._last_box: tuple[int, int, int, int] | None = None

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
        # Own the bitmap once. Later moves restore only the previous shape's
        # rectangle from this snapshot instead of copying the whole layer.
        layer.image = self._snap_image.copy()
        self._last_box = None

    def _restore_box(self, box: tuple[int, int, int, int]) -> None:
        """Put the snapshot's pixels back inside ``box`` and clear the rest of it."""
        layer = self._layer
        snap = self._snap_image
        if layer is None or snap is None:
            return
        x0, y0, x1, y1 = box
        lx0 = max(0, x0 - layer.x)
        ly0 = max(0, y0 - layer.y)
        lx1 = min(layer.image.width, x1 - layer.x)
        ly1 = min(layer.image.height, y1 - layer.y)
        if lx1 <= lx0 or ly1 <= ly0:
            return
        patch = Image.new("RGBA", (lx1 - lx0, ly1 - ly0), (0, 0, 0, 0))
        sx0, sy0 = self._snap_xy
        dx0 = max(x0, sx0, layer.x)
        dy0 = max(y0, sy0, layer.y)
        dx1 = min(x1, sx0 + snap.width, layer.x + layer.image.width)
        dy1 = min(y1, sy0 + snap.height, layer.y + layer.image.height)
        if dx1 > dx0 and dy1 > dy0:
            crop = snap.crop((dx0 - sx0, dy0 - sy0, dx1 - sx0, dy1 - sy0))
            patch.paste(crop, (dx0 - (layer.x + lx0), dy0 - (layer.y + ly0)))
        layer.image.paste(patch, (lx0, ly0))

    def _paint(self, session, box: tuple[int, int, int, int]) -> None:
        layer = self._layer
        if layer is None:
            return
        # Grow to the stroke's canvas box first so a small layer can hold it.
        # finish_raster_edit then crops back to the opaque pixels.
        grow_layer_to_cover(layer, box)
        lx0 = max(0, box[0] - layer.x)
        ly0 = max(0, box[1] - layer.y)
        lx1 = min(layer.image.width, box[2] - layer.x)
        ly1 = min(layer.image.height, box[3] - layer.y)
        if lx1 <= lx0 or ly1 <= ly0:
            return
        crop = layer.image.crop((lx0, ly0, lx1, ly1))
        ox = float(layer.x + lx0)
        oy = float(layer.y + ly0)
        selection = selection_for_layer(session.document, layer)
        local_selection = None if selection is None else selection.crop((lx0, ly0, lx1, ly1))
        kind = str(session.option("shape_kind", "rect"))
        painted = draw_shape(
            crop,
            kind,
            (self._start[0] - ox, self._start[1] - oy, self._end[0] - ox, self._end[1] - oy),
            session.foreground,
            int(session.option("stroke_width", 4)),
            bool(session.option("shape_filled", False)),
            local_selection,
            line=((self._start[0] - ox, self._start[1] - oy), (self._end[0] - ox, self._end[1] - oy)),
        )
        layer.image.paste(painted, (lx0, ly0))

    def _show_preview(self, session) -> None:
        self._ensure_edit(session)
        box = self._canvas_box(session)
        if box is None:
            if self._last_box is not None:
                self._restore_box(self._last_box)
                self._last_box = None
            session.preview()
            return
        if self._last_box is not None:
            self._restore_box(self._last_box)
        self._paint(session, box)
        dirty = union_boxes(self._last_box, box)
        self._last_box = box
        session.preview(dirty)

    def _take_edit(self) -> tuple[Layer | None, dict | None, bool]:
        layer, before, created = self._layer, self._before, self._created
        self._layer = None
        self._before = None
        self._created = False
        self._snap_image = None
        self._last_box = None
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

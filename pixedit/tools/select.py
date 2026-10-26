"""Selection tool. Rectangle, ellipse, freehand, and magic wand."""

from __future__ import annotations

import math

from PIL import Image

from pixedit.geom import (
    combine_selection,
    mask_ellipse,
    mask_freehand,
    mask_rect,
)
from pixedit.paint import wand_document
from pixedit.tools.base import (
    Overlay,
    PointerEvent,
    Tool,
)
from pixedit.tools.region import _region_cursor


class SelectTool(Tool):
    id = "select"
    label = "Select"
    shortcut = "S"
    cursor = "select-rect"
    hint = (
        "Drag a selection. Ctrl+X cuts it to a new layer, Ctrl+C copies it, Delete clears pixels, Esc deselects. "
        "Wand also selects empty space outside the layer, and stops at the canvas edge."
    )

    def __init__(self) -> None:
        self._dragging = False
        # Locked at press time. The poll keeps running until release even if
        # the mode menu changes during the drag.
        self._freehand = False
        self._start = (0.0, 0.0)
        self._points: list[tuple[float, float]] = []

    def option_schema(self):
        return [
            {
                "key": "select_mode",
                "label": "Mode",
                "type": "choice",
                "icons": True,
                "choices": [
                    ("rect", "Rectangular", "select-rect"),
                    ("ellipse", "Elliptical", "select-ellipse"),
                    ("freehand", "Freehand", "select-freehand"),
                    ("wand", "Magic wand", "geom-wand"),
                ],
                "default": "rect",
            },
            {"key": "tolerance", "label": "Wand tolerance", "type": "int", "min": 0, "max": 255, "default": 32},
        ]

    def cursor_name(self, session) -> str:
        return _region_cursor(str(session.option("select_mode", "rect")))

    def wants_stroke_sampling(self, session) -> bool:
        return str(session.option("select_mode", "rect")) == "freehand"

    def stroke_live(self) -> bool:
        return self._dragging and self._freehand

    def panel_actions(self):
        return [
            {"id": "select_all", "label": "Select all", "shortcut": "Ctrl+A", "icon": "select-all"},
            {"id": "deselect", "label": "Deselect", "shortcut": "Ctrl+D", "needs_selection": True, "icon": "deselect"},
            {"id": "invert_selection", "label": "Invert", "shortcut": "Ctrl+Shift+I", "needs_selection": True, "icon": "select-invert"},
            {"id": "cut_selection_to_layer", "label": "Cut to layer", "shortcut": "Ctrl+X", "needs_selection": True, "icon": "cut-to-layer"},
            {"id": "copy_selection_to_layer", "label": "Copy to layer", "shortcut": "Ctrl+C", "needs_selection": True, "icon": "copy-to-layer"},
            {"id": "clear_selection_contents", "label": "Clear pixels", "shortcut": "Delete", "needs_selection": True, "icon": "clear-pixels"},
            {"id": "crop_to_selection", "label": "Crop to selection", "needs_selection": True, "icon": "crop-to-selection"},
        ]

    def pointer_down(self, session, event: PointerEvent) -> None:
        if str(session.option("select_mode", "rect")) == "wand":
            self._wand(session, event)
            return
        self._dragging = True
        self._freehand = str(session.option("select_mode", "rect")) == "freehand"
        self._start = (event.x, event.y)
        self._points = [(event.x, event.y)]

    def pointer_move(self, session, event: PointerEvent) -> None:
        if not self._dragging:
            return
        if self._freehand:
            self.extend_points(session, [(event.x, event.y)])
            return
        self._points.append((event.x, event.y))
        # The rectangle and ellipse are chrome on top of the photo. A full
        # preview would flatten the document on every move.
        session.notify("overlay")

    def extend_points(self, session, points: list[tuple[float, float]]) -> None:
        """Append freehand samples. The canvas polls the cursor and flushes batches here.

        A full preview would rebuild the photo and stall that poll, so a new
        point only asks the view to redraw the rubber band. A point within
        0.05 image pixels of the last one is dropped, the same rule as the brush.
        """
        if not self._dragging or not self._freehand:
            return
        added = False
        for x, y in points:
            point = (float(x), float(y))
            if self._points:
                last_x, last_y = self._points[-1]
                if math.hypot(point[0] - last_x, point[1] - last_y) < 0.05:
                    continue
            self._points.append(point)
            added = True
        if added:
            session.notify("overlay")

    def pointer_up(self, session, event: PointerEvent) -> None:
        if not self._dragging:
            return
        if self._freehand:
            self.extend_points(session, [(event.x, event.y)])
            self.commit_stroke(session, event)
            return
        self._points.append((event.x, event.y))
        self._apply_selection(session, self._mask(session, event), event)

    def commit_stroke(self, session, event: PointerEvent | None = None) -> None:
        """Close the lasso from the stored points.

        The canvas flushes the last samples before this call. The points are
        the polygon vertices. Shift adds to the selection and Alt subtracts.
        """
        if not self._dragging or not self._freehand:
            return
        doc = session.document
        mask = mask_freehand(doc.width, doc.height, self._points, radius=1)
        self._apply_selection(session, mask, event)

    def _apply_selection(self, session, mask: Image.Image, event: PointerEvent | None) -> None:
        self._dragging = False
        self._freehand = False
        self._points = []
        mode = "replace"
        if event is not None:
            if event.shift:
                mode = "add"
            elif event.alt:
                mode = "subtract"
        session.set_selection(combine_selection(session.document.selection, mask, mode))

    def pointer_cancel(self, session) -> bool:
        active = self._dragging
        self._dragging = False
        self._freehand = False
        self._points = []
        if active:
            session.notify("overlay")
        return active

    def _mask(self, session, event: PointerEvent) -> Image.Image:
        doc = session.document
        kind = str(session.option("select_mode", "rect"))
        if kind == "freehand":
            return mask_freehand(doc.width, doc.height, self._points, radius=1)
        x0, y0 = self._start
        if kind == "ellipse":
            return mask_ellipse(doc.width, doc.height, x0, y0, event.x, event.y)
        return mask_rect(doc.width, doc.height, x0, y0, event.x, event.y)

    def _wand(self, session, event: PointerEvent) -> None:
        layer = session.active_layer()
        if layer is None or layer.kind == "redaction":
            session.show_message("Magic wand samples the active raster or text layer.")
            return
        # A click anywhere on the canvas is a seed, including empty space the
        # layer bitmap does not cover. Outside the canvas: explain and do not
        # change the selection. The plate matches Fill, so the walk stops at a
        # different pixel or the canvas edge. Shift and Alt still combine
        # afterwards; an existing selection does not clip the walk.
        doc = session.document
        px = int(math.floor(event.x))
        py = int(math.floor(event.y))
        if not (0 <= px < doc.width and 0 <= py < doc.height):
            session.show_message("Click is outside the canvas.")
            return
        canvas = wand_document(
            layer.image,
            layer.x,
            layer.y,
            doc.width,
            doc.height,
            px,
            py,
            int(session.option("tolerance", 32)),
        )
        mode = "replace"
        if event.shift:
            mode = "add"
        elif event.alt:
            mode = "subtract"
        session.set_selection(combine_selection(doc.selection, canvas, mode), name="Magic wand")

    def overlays(self, session) -> list[Overlay]:
        if not self._dragging:
            return []
        # The gesture keeps the shape it started with. A mode change while
        # the button is down does not turn a lasso into a rectangle.
        if self._freehand:
            return [Overlay(kind="polyline", points=list(self._points))]
        kind = str(session.option("select_mode", "rect"))
        if kind == "freehand":
            return [Overlay(kind="polyline", points=list(self._points))]
        x0, y0 = self._start
        x1, y1 = self._points[-1] if self._points else self._start
        left, top = min(x0, x1), min(y0, y1)
        return [
            Overlay(
                kind="ellipse" if kind == "ellipse" else "rect",
                rect=(left, top, abs(x1 - x0), abs(y1 - y0)),
            )
        ]

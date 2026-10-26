"""Gradient tool. Drag a linear or radial blend and release to commit it."""

from __future__ import annotations

import math

from PIL import Image

from pixedit.geom import grow_layer_to_cover, selection_for_layer
from pixedit.guides import snap_point
from pixedit.model import Layer
from pixedit.paint import draw_gradient
from pixedit.tools.base import PointerEvent, Tool


def gradient_end_color(session) -> tuple[int, int, int, int]:
    """The second swatch. The foreground color is the start of the blend."""
    raw = session.option("gradient_color", (255, 255, 255, 255))
    if not isinstance(raw, (tuple, list)) or len(raw) < 3:
        return (255, 255, 255, 255)
    red = max(0, min(255, int(raw[0])))
    green = max(0, min(255, int(raw[1])))
    blue = max(0, min(255, int(raw[2])))
    alpha = 255 if len(raw) < 4 else max(0, min(255, int(raw[3])))
    return (red, green, blue, alpha)


class GradientTool(Tool):
    id = "gradient"
    label = "Gradient"
    shortcut = ""
    cursor = "cross"
    hint = (
        "Drag a linear or radial gradient from the foreground color to the end color. "
        "Release commits it. The end color is the button under the options."
    )

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
                "key": "gradient_kind",
                "label": "Gradient",
                "type": "choice",
                "choices": [("linear", "Linear"), ("radial", "Radial")],
                "default": "linear",
            },
        ]

    def panel_actions(self):
        return [{"id": "gradient_end_color", "label": "End color…"}]

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
        self._show_preview(session)
        layer, before, created = self._take_edit()
        if layer is None:
            return
        session.finish_raster_edit(layer, before, created, "Gradient")

    def pointer_cancel(self, session) -> bool:
        active = self._dragging or self._layer is not None
        self._dragging = False
        self._cancel_edit(session)
        return active

    def _snap(self, session, x: float, y: float) -> tuple[float, float]:
        group = session.document.group
        return snap_point(x, y, group.guides_v, group.guides_h, bool(session.option("snap", True)))

    def _ensure_edit(self, session) -> None:
        if self._layer is not None:
            return
        layer, before, created = session.begin_raster_edit(paint_redaction=False)
        self._layer = layer
        self._before = before
        self._created = created
        self._snap_image = layer.image.copy()
        self._snap_xy = (layer.x, layer.y)

    def _restore_snap(self) -> None:
        if self._layer is None or self._snap_image is None:
            return
        self._layer.image = self._snap_image.copy()
        self._layer.x, self._layer.y = self._snap_xy

    def _show_preview(self, session) -> None:
        self._ensure_edit(session)
        self._restore_snap()
        layer = self._layer
        if layer is None:
            return
        # The blend covers the page, not just the drag line. A selection
        # still limits it, the same way a shape is clipped.
        grow_layer_to_cover(layer, (0, 0, session.document.width, session.document.height))
        ox, oy = float(layer.x), float(layer.y)
        layer.image = draw_gradient(
            layer.image,
            str(session.option("gradient_kind", "linear")),
            (self._start[0] - ox, self._start[1] - oy),
            (self._end[0] - ox, self._end[1] - oy),
            session.foreground,
            gradient_end_color(session),
            selection_for_layer(session.document, layer),
        )
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

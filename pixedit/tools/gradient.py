"""Gradient tool. Drag a linear or radial blend and release to commit it."""

from __future__ import annotations

import math

from PIL import Image

from pixedit.geom import grow_layer_to_cover, placed_box, selection_for_layer
from pixedit.guides import snap_point
from pixedit.model import Layer
from pixedit.paint import draw_gradient
from pixedit.proxy import downscale, preview_scale, set_preview_source
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
        self._base: Image.Image | None = None

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
        self._show_preview(session, final=True)
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
        layer.image = self._snap_image.copy()
        # The blend covers the page. Grow once so later moves do not allocate
        # another full plate just to throw the previous one away.
        grow_layer_to_cover(layer, (0, 0, session.document.width, session.document.height))
        self._base = layer.image

    def _show_preview(self, session, *, final: bool = False) -> None:
        self._ensure_edit(session)
        layer = self._layer
        base = getattr(self, "_base", None)
        if layer is None or base is None:
            return
        kind = str(session.option("gradient_kind", "linear"))
        selection = selection_for_layer(session.document, layer)
        zoom = float(getattr(session, "view_zoom", 1.0) or 1.0)
        scale = 1.0 if final else preview_scale(base.width, base.height, zoom)
        if scale >= 1.0:
            set_preview_source(layer, None)
            ox, oy = float(layer.x), float(layer.y)
            layer.image = draw_gradient(
                base,
                kind,
                (self._start[0] - ox, self._start[1] - oy),
                (self._end[0] - ox, self._end[1] - oy),
                session.foreground,
                gradient_end_color(session),
                selection,
            )
        else:
            # Keep the full plate underneath. The stand-in is what the canvas
            # scales into that rectangle until release draws the real gradient.
            layer.image = base
            small = downscale(base, scale)
            small_selection = None if selection is None else selection.resize(small.size, Image.Resampling.NEAREST)
            ox, oy = float(layer.x), float(layer.y)
            set_preview_source(
                layer,
                draw_gradient(
                    small,
                    kind,
                    ((self._start[0] - ox) * scale, (self._start[1] - oy) * scale),
                    ((self._end[0] - ox) * scale, (self._end[1] - oy) * scale),
                    session.foreground,
                    gradient_end_color(session),
                    small_selection,
                ),
            )
        session.preview(placed_box(layer, session.document.width, session.document.height))

    def _take_edit(self) -> tuple[Layer | None, dict | None, bool]:
        layer, before, created = self._layer, self._before, self._created
        self._layer = None
        self._before = None
        self._created = False
        self._snap_image = None
        self._base = None
        if layer is not None:
            set_preview_source(layer, None)
        return layer, before, created

    def _cancel_edit(self, session) -> None:
        layer, before, created = self._take_edit()
        if layer is None:
            return
        session.cancel_raster_edit(layer, before, created)

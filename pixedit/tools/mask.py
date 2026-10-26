"""Mask tool. Draws a live effect on the active picture or text layer."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from pixedit.geom import (
    mask_ellipse,
    mask_freehand,
    mask_rect,
    place_image,
)
from pixedit.model import (
    Layer,
    apply_visual,
    visual_state,
)
from pixedit.tools.base import (
    Overlay,
    PointerEvent,
    Tool,
)
from pixedit.tools.region import _region_cursor


class MaskTool(Tool):
    id = "mask"
    label = "Mask"
    shortcut = "R"
    cursor = "select-rect"
    hint = "Draw a mask on the active layer. The effect stays separate until you apply it, merge, or export."

    def __init__(self) -> None:
        self._dragging = False
        # Latched at press. The cursor poll keeps running until release even
        # if the mode control changes during the drag.
        self._freehand = False
        self._start = (0.0, 0.0)
        self._points: list[tuple[float, float]] = []
        self._layer: Layer | None = None
        self._created = False
        self._before: dict | None = None

    def option_schema(self):
        from pixedit.model import MASK_EFFECTS

        return [
            {
                "key": "mask_shape",
                "label": "Shape",
                "type": "choice",
                "icons": True,
                "choices": [
                    ("rect", "Rectangle", "geom-rect"),
                    ("ellipse", "Ellipse", "geom-ellipse"),
                    ("freehand", "Freehand", "geom-freehand"),
                ],
                "default": "rect",
            },
            {
                "key": "mask_kind",
                "label": "Effect",
                "type": "choice",
                "choices": [
                    ("pixelate", "Pixelate"),
                    ("blur", "Blur"),
                    ("solid", "Solid"),
                    ("noise", "Noise"),
                    ("desaturate", "Desaturate"),
                    ("invert", "Invert"),
                    ("dim", "Dim"),
                    ("sharpen", "Sharpen"),
                ],
            "default": "pixelate",
            "amounts": MASK_EFFECTS,
            "amount_key": "mask_amount",
        },
        {
            "key": "mask_amount",
            "label": "Block",
            "type": "int",
            "min": 2,
            "max": 64,
            "default": 12,
        },
    ]

    def cursor_name(self, session) -> str:
        # Same glyphs as Select. The gesture is the region, not the effect.
        return _region_cursor(str(session.option("mask_shape", "rect")))

    def pointer_down(self, session, event: PointerEvent) -> None:
        # The mask is baked on release. Dragging only updates the rubber band,
        # because a live Gaussian blur of the whole canvas would stall the UI.
        self._dragging = True
        self._start = (event.x, event.y)
        self._points = [(event.x, event.y)]
        self._layer = None
        self._created = False
        self._before = None

    def pointer_move(self, session, event: PointerEvent) -> None:
        if not self._dragging:
            return
        self._points.append((event.x, event.y))
        session.preview()

    def pointer_up(self, session, event: PointerEvent) -> None:
        if not self._dragging:
            return
        self._points.append((event.x, event.y))
        self._dragging = False
        kind = str(session.option("mask_shape", "rect"))
        if kind != "freehand" and math.hypot(event.x - self._start[0], event.y - self._start[1]) < 2:
            self._points = []
            session.preview()
            return
        layer, created = session.ensure_redaction_layer()
        self._layer = layer
        self._created = created
        self._before = None if created else visual_state(layer)
        layer.redaction = session.current_mask_info(layer.redaction)
        self._preview(session, event)
        baked = self._layer
        before = self._before
        self._layer = None
        self._points = []
        if baked is None:
            return
        if created:
            try:
                index = session.document.layers.index(baked)
            except ValueError:
                index = len(session.document.layers)
            else:
                session.document.layers.pop(index)
            session.add_layer(baked, index=index, name="Mask")
            return
        if before is None:
            return
        new_state = visual_state(baked)
        apply_visual(baked, before)
        session.apply_layer_image(baked, new_state, "Mask", before=before)

    def pointer_cancel(self, session) -> bool:
        active = self._dragging or self._layer is not None
        self._dragging = False
        self._points = []
        self._layer = None
        if active:
            session.preview()
        return active

    def _preview(self, session, event: PointerEvent) -> None:
        if self._layer is None:
            return
        doc = session.document
        shape = self._shape_mask(session, event)
        if self._before is not None and self._before["mask"] is not None:
            base = place_image(self._before["mask"], self._before["x"], self._before["y"], doc.width, doc.height, 0)
        else:
            base = Image.new("L", (doc.width, doc.height), 0)
        # Union, so a second stroke on the same redaction layer adds to the mask
        # instead of replacing regions the user already drew.
        combined = np.maximum(np.asarray(base), np.asarray(shape))
        self._layer.mask = Image.fromarray(combined, "L")
        self._layer.x = 0
        self._layer.y = 0

    def _shape_mask(self, session, event: PointerEvent) -> Image.Image:
        doc = session.document
        kind = str(session.option("mask_shape", "rect"))
        if kind == "freehand":
            return mask_freehand(doc.width, doc.height, self._points, radius=8)
        x0, y0 = self._start
        x1, y1 = event.x, event.y
        if kind == "ellipse":
            return mask_ellipse(doc.width, doc.height, x0, y0, x1, y1)
        return mask_rect(doc.width, doc.height, x0, y0, x1, y1)

    def overlays(self, session) -> list[Overlay]:
        if not self._dragging or len(self._points) < 1:
            return []
        kind = str(session.option("mask_shape", "rect"))
        if kind == "freehand":
            return [Overlay(kind="polyline", points=list(self._points))]
        x0, y0 = self._start
        x1, y1 = self._points[-1]
        left, top = min(x0, x1), min(y0, y1)
        rect = (left, top, abs(x1 - x0), abs(y1 - y0))
        return [Overlay(kind="ellipse" if kind == "ellipse" else "rect", rect=rect)]

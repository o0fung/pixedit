"""Sizes and crop tool. Drag a window to crop, or past the edge to grow the canvas."""

from __future__ import annotations

import math

from pixedit.geom import size_from_anchor
from pixedit.guides import snap_point
from pixedit.tools.base import (
    Overlay,
    PointerEvent,
    Tool,
)


class CropTool(Tool):
    id = "crop"
    label = "Sizes & Crop"
    shortcut = "C"
    cursor = "crop"
    hint = (
        "Drag a rectangle. Release crops the canvas, and a drag past the edge expands it. "
        "Shift keeps the proportion."
    )

    def __init__(self) -> None:
        self._dragging = False
        self._start = (0.0, 0.0)
        self._end = (0.0, 0.0)
        self._overlay_rect: tuple[int, int, int, int] | None = None

    def option_schema(self):
        return [
            {
                "key": "crop_ratio",
                "label": "Proportion",
                "type": "choice",
                "choices": [
                    ("free", "Free"),
                    ("original", "Original"),
                    ("1:1", "1:1"),
                    ("4:3", "4:3"),
                    ("16:9", "16:9"),
                    ("custom", "Custom"),
                ],
                "default": "free",
            },
            {"key": "crop_ratio_w", "label": "Ratio width", "type": "spin", "min": 1, "max": 10000, "default": 1},
            {"key": "crop_ratio_h", "label": "Ratio height", "type": "spin", "min": 1, "max": 10000, "default": 1},
        ]

    def panel_actions(self):
        return [
            {"id": "canvas_size", "label": "Canvas size…"},
            {"id": "image_size", "label": "Image size…"},
            {"id": "trim", "label": "Trim transparent edges"},
            {"id": "crop_to_selection", "label": "Crop to selection", "needs_selection": True},
        ]

    def _locked_aspect(self, session, event: PointerEvent) -> float | None:
        """Width/height to lock, or None for a free rectangle.

        A chosen proportion always locks. Shift while Free is selected locks
        the canvas proportion, the same way Transform's Shift locks the layer.
        """
        choice = str(session.option("crop_ratio", "free"))
        if choice == "1:1":
            return 1.0
        if choice == "4:3":
            return 4.0 / 3.0
        if choice == "16:9":
            return 16.0 / 9.0
        if choice == "custom":
            width = max(1, int(session.option("crop_ratio_w", 1)))
            height = max(1, int(session.option("crop_ratio_h", 1)))
            return width / height
        if choice == "original" or event.shift:
            document = session.document
            return document.width / max(1, document.height)
        return None

    def _crop_box(self, session, event: PointerEvent) -> tuple[int, int, int, int]:
        aspect = self._locked_aspect(session, event)
        group = session.document.group
        enabled = bool(session.option("snap", True))
        start = snap_point(self._start[0], self._start[1], group.guides_v, group.guides_h, enabled)
        end = snap_point(event.x, event.y, group.guides_v, group.guides_h, enabled)
        left, top, width, height = size_from_anchor(
            start,
            end,
            lock_aspect=aspect is not None,
            aspect=aspect or 1.0,
        )
        return (
            int(round(left)),
            int(round(top)),
            max(1, int(round(width))),
            max(1, int(round(height))),
        )

    def pointer_down(self, session, event: PointerEvent) -> None:
        self._dragging = True
        self._start = (event.x, event.y)
        self._end = (event.x, event.y)
        self._overlay_rect = self._crop_box(session, event)

    def pointer_move(self, session, event: PointerEvent) -> None:
        if not self._dragging:
            return
        self._end = (event.x, event.y)
        self._overlay_rect = self._crop_box(session, event)
        session.preview()

    def pointer_up(self, session, event: PointerEvent) -> None:
        if not self._dragging:
            return
        self._end = (event.x, event.y)
        self._dragging = False
        box = self._crop_box(session, event)
        self._overlay_rect = None
        if math.hypot(self._end[0] - self._start[0], self._end[1] - self._start[1]) < 2:
            session.preview()
            return
        session.crop_to(*box)

    def pointer_cancel(self, session) -> bool:
        active = self._dragging
        self._dragging = False
        self._overlay_rect = None
        if active:
            session.preview()
        return active

    def overlays(self, session) -> list[Overlay]:
        if not self._dragging or self._overlay_rect is None:
            return []
        left, top, width, height = self._overlay_rect
        return [Overlay(kind="rect", rect=(left, top, width, height))]

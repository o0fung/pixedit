"""Image and eyedropper. Click to pick a color. The panel grades the active layer."""

from __future__ import annotations

from pixedit.composite import composite
from pixedit.paint import sample_color
from pixedit.tools.base import (
    PointerEvent,
    Tool,
)


class ColorTool(Tool):
    id = "color"
    label = "Image & Eyedrop"
    shortcut = "I"
    cursor = "eyedrop"
    hint = "Click to pick a color. Sliders adjust the active layer; Apply keeps the change."

    def option_schema(self):
        return [
            {"key": "color_hue", "label": "Hue", "type": "int", "min": -180, "max": 180, "default": 0, "suffix": "°", "live": "color"},
            {"key": "color_saturation", "label": "Saturation", "type": "int", "min": -100, "max": 100, "default": 0, "live": "color"},
            {"key": "color_brightness", "label": "Brightness", "type": "int", "min": -100, "max": 100, "default": 0, "live": "color"},
            {"key": "color_contrast", "label": "Contrast", "type": "int", "min": -100, "max": 100, "default": 0, "live": "color"},
            {"key": "color_highlight", "label": "Highlight", "type": "int", "min": -100, "max": 100, "default": 0, "live": "color"},
            {"key": "color_shade", "label": "Shade", "type": "int", "min": -100, "max": 100, "default": 0, "live": "color"},
        ]

    def panel_actions(self):
        return [
            {"id": "apply_color", "label": "Apply", "needs_color_preview": True},
            {"id": "reset_color", "label": "Reset", "needs_color_preview": True},
        ]

    def pointer_down(self, session, event: PointerEvent) -> None:
        color = sample_color(composite(session.document), event.x, event.y)
        if color is None or color[3] == 0:
            session.show_message("Nothing opaque to sample there.")
            return
        session.foreground = color
        session.notify("color")

    def pointer_cancel(self, session) -> bool:
        return bool(session.cancel_color_adjust())

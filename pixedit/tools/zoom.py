"""Zoom tool. Click to zoom in. Alt-click zooms out."""

from __future__ import annotations

from pixedit.tools.base import Tool


class ZoomTool(Tool):
    id = "zoom"
    label = "Zoom"
    shortcut = "Z"
    cursor = "zoom-in"
    hint = "Click to zoom in. Alt-click zooms out. The wheel zooms at the cursor."

    def panel_actions(self):
        return [
            {"id": "zoom_in", "label": "Zoom in", "shortcut": "Ctrl++", "icon": "zoom"},
            {"id": "zoom_out", "label": "Zoom out", "shortcut": "Ctrl+-", "icon": "zoom-out"},
            {"id": "zoom_fit", "label": "Fit", "shortcut": "Ctrl+0", "icon": "zoom-fit"},
            {"id": "zoom_100", "label": "100%", "shortcut": "Ctrl+1", "icon": "zoom-100"},
        ]

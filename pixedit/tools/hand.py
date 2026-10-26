"""Hand tool. Drag to pan the canvas."""

from __future__ import annotations

from pixedit.tools.base import Tool


class HandTool(Tool):
    id = "hand"
    label = "Hand"
    shortcut = "H"
    cursor = "hand"
    hint = "Drag to pan. You can also hold Space with any tool."

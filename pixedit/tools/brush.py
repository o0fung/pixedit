"""Brush tool. Paints the foreground color on the active layer."""

from __future__ import annotations

from pixedit.tools.paint_tool import _PaintTool


class BrushTool(_PaintTool):
    id = "brush"
    label = "Brush"
    shortcut = "B"
    hint = "Paint on the active layer. The cursor matches the diameter, and the panel shows the stamp. [ and ] change the size."

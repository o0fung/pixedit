"""Eraser tool. Clears pixels, or lifts a mask when a mask layer is active."""

from __future__ import annotations

from pixedit.tools.paint_tool import _PaintTool


class EraserTool(_PaintTool):
    id = "eraser"
    label = "Eraser"
    shortcut = "E"
    erase = True
    hint = "Erase pixels, or lift a mask when a mask layer is active. The panel shows the hole. [ and ] change the size."

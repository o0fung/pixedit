"""Tool interface shared by built-in tools and plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PointerEvent:
    """Canvas-space pointer event. ``zoom`` is screen pixels per image pixel."""

    x: float
    y: float
    shift: bool = False
    alt: bool = False
    ctrl: bool = False
    button: int = 1
    zoom: float = 1.0


@dataclass
class Overlay:
    """Something the canvas draws on top of the composite, in canvas pixels."""

    kind: str  # rect, ellipse, polyline, handles
    rect: tuple[float, float, float, float] | None = None
    points: list[tuple[float, float]] = field(default_factory=list)
    handles: list[tuple[str, float, float]] = field(default_factory=list)


class Tool:
    """Interactive canvas tool.

    Subclasses override the pointer methods they need. ``option_schema`` drives
    the left-hand options panel; keys land in ``session.options``.
    """

    id = "tool"
    label = "Tool"
    shortcut = ""
    cursor = "cross"
    hint = ""

    def button_caption(self) -> str:
        """Toolbar text. A shortcut is shown beside the name, as in ``Brush (B)``."""
        if self.shortcut:
            return f"{self.label} ({self.shortcut})"
        return self.label

    def option_schema(self) -> list[dict[str, Any]]:
        return []

    def panel_actions(self) -> list[dict[str, Any]]:
        """Buttons under the option controls.

        Each entry has ``id`` and ``label``. The window runs the id. Select
        actions name session methods. ``needs_selection`` disables the button
        until a selection exists. ``shortcut`` is only the tooltip.
        """
        return []

    def pointer_down(self, session, event: PointerEvent) -> None:
        return None

    def pointer_move(self, session, event: PointerEvent) -> None:
        return None

    def pointer_up(self, session, event: PointerEvent) -> None:
        return None

    def pointer_cancel(self, session) -> bool:
        """Cancel an in-progress gesture. True means a drag was actually cancelled."""
        return False

    def overlays(self, session) -> list[Overlay]:
        return []


def builtin_tools() -> list[Tool]:
    from pixedit.tools.builtins import (
        BrushTool,
        ColorTool,
        CropTool,
        EraserTool,
        FillTool,
        HandTool,
        MaskTool,
        SelectTool,
        ShapeTool,
        TextTool,
        TransformTool,
        ZoomTool,
    )

    return [
        TransformTool(),
        BrushTool(),
        EraserTool(),
        TextTool(),
        MaskTool(),
        SelectTool(),
        CropTool(),
        FillTool(),
        ColorTool(),
        ShapeTool(),
        HandTool(),
        ZoomTool(),
    ]

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

    def cursor_name(self, _session) -> str:
        """Pointer for this tool. Options can change it; the canvas asks every time."""
        return self.cursor

    def button_caption(self) -> str:
        """Toolbar text. A shortcut is shown beside the name, as in ``Brush (B)``."""
        if self.shortcut:
            return f"{self.label} ({self.shortcut})"
        return self.label

    def option_schema(self) -> list[dict[str, Any]]:
        return []

    def panel_actions(self) -> list[dict[str, Any]]:
        """Buttons under the option controls.

        Each entry has ``id`` and ``label``. ``icon`` names a file in
        ``resources/icons`` and draws an icon-only button; the label stays
        the tooltip and the accessible name. ``new_row`` starts a new row
        even when the next icon would still fit. The window runs the id.
        Select actions name session methods. ``needs_selection`` disables
        the button until a selection exists. ``shortcut`` is only the tooltip.
        """
        return []

    def wants_stroke_sampling(self, _session) -> bool:
        """True when the canvas should store mouse moves and paint them in batches.

        Mouse-move events arrive in bursts. Each move stores a coordinate.
        About every 8 ms the canvas passes that batch to ``extend_points``.
        Release calls ``commit_stroke`` with the pointer, including Shift and Alt.
        """
        return False

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
    # Imported here so tool modules can import Tool from this file without a cycle.
    from pixedit.tools.brush import BrushTool
    from pixedit.tools.color import ColorTool
    from pixedit.tools.crop import CropTool
    from pixedit.tools.eraser import EraserTool
    from pixedit.tools.fill import FillTool
    from pixedit.tools.hand import HandTool
    from pixedit.tools.mask import MaskTool
    from pixedit.tools.select import SelectTool
    from pixedit.tools.shape import ShapeTool
    from pixedit.tools.text import TextTool
    from pixedit.tools.transform import TransformTool
    from pixedit.tools.zoom import ZoomTool

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

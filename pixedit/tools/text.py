"""Text tool. Click a text layer to edit it, or click empty space to add one."""

from __future__ import annotations

from pixedit.geom import top_layer_at
from pixedit.model import Layer
from pixedit.tools.base import (
    PointerEvent,
    Tool,
)


class TextTool(Tool):
    id = "text"
    label = "Text"
    shortcut = "T"
    cursor = "ibeam"
    hint = "Click inside a text box to edit it. Click empty space to add text."

    def option_schema(self):
        return [
            {"key": "text_font", "label": "Font", "type": "font"},
            {"key": "text_style", "label": "Style", "type": "font_style"},
            {
                "key": "text_align",
                "label": "Align",
                "type": "choice",
                "icons": True,
                "choices": [
                    ("left", "Left", "align-left"),
                    ("center", "Center", "align-center"),
                    ("right", "Right", "align-right"),
                    ("justify", "Justify", "align-justify"),
                ],
                "default": "left",
            },
            {"key": "text_size", "label": "Size", "type": "spin", "min": 1, "max": 512, "default": 48},
            {
                "key": "text_line_spacing",
                "label": "Spacing",
                "type": "int",
                "min": 0,
                "max": 80,
                "default": 4,
                "suffix": " px",
            },
            {
                "key": "text_rotation",
                "label": "Rotation",
                "type": "int",
                "min": -180,
                "max": 180,
                "default": 0,
                "suffix": "°",
            },
        ]

    def pointer_down(self, session, event: PointerEvent) -> None:
        target = self._text_target(session, event)
        if target is None or target.text is None:
            initial = session.text_info_from_options("Text")
        else:
            # Load this layer into the panel before the word dialog. The dialog
            # keeps that style, so a click does not restyle it from the previous
            # panel values. Cancel still leaves the layer selected.
            session.layer_index = session.document.index_of(target.id)
            session.adopt_text_style(target.text)
            session.notify("change")
            initial = target.text
        edited = session.prompt_text(initial)
        if edited is None or not edited.text.strip():
            return
        if target is None:
            session.add_text(edited, int(round(event.x)), int(round(event.y)))
        else:
            session.focus_layer(target)
            session.update_text(target, edited)

    def _text_target(self, session, event: PointerEvent) -> Layer | None:
        # The topmost box wins, including a shared header. A picture stacked
        # above a text layer is not edited by a click on the picture.
        layer = top_layer_at(session.document, event.x, event.y)
        if layer is None or layer.kind != "text":
            return None
        return layer

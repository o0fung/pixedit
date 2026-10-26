"""Editable text layers.

Mixed into ``Session``. The panel previews a style change, then one undo step
keeps it. The words stay on the layer. The style comes from the tool options.
"""

from __future__ import annotations

import copy

from pixedit.factory import text_layer
from pixedit.model import Layer, MutateLayerCommand, TextInfo, apply_visual, visual_state
from pixedit.textutil import render_text


def _text_same(left: TextInfo, right: TextInfo) -> bool:
    return (
        left.text == right.text
        and left.font_path == right.font_path
        and left.font_family == right.font_family
        and int(left.size) == int(right.size)
        and tuple(int(channel) for channel in left.color) == tuple(int(channel) for channel in right.color)
        and abs(float(left.rotation) - float(right.rotation)) < 1e-4
        and (left.align or "left") == (right.align or "left")
        and int(left.line_spacing) == int(right.line_spacing)
    )


class TextMixin:
    """Add a text layer and restyle the active one."""

    def add_text(self, info: TextInfo, x: int, y: int) -> None:
        if not info.text.strip():
            return
        layer = text_layer("Text", info, x, y)
        self.add_layer(layer, name="Add text")

    def text_info_from_options(self, text: str) -> TextInfo:
        """Words plus the Text panel's font, align, spacing, rotation, and palette color."""
        red, green, blue, alpha = (int(channel) for channel in self.foreground)
        align = str(self.option("text_align", "left") or "left")
        if align not in ("left", "center", "right", "justify"):
            align = "left"
        return TextInfo(
            text=text,
            font_path=self.option("text_font"),
            font_family=str(self.option("text_font_family") or "Default"),
            size=max(1, int(self.option("text_size", 48))),
            color=(red, green, blue, alpha),
            rotation=float(self.option("text_rotation", 0)),
            align=align,
            line_spacing=max(0, int(self.option("text_line_spacing", 4))),
        )

    def adopt_text_style(self, info: TextInfo) -> None:
        """Copy a text layer's style into the tool options and the palette.

        This does not render. Loading a layer into the panel must not restyle it.
        """
        color = tuple(int(channel) for channel in info.color)
        if len(color) == 3:
            color = (*color, 255)
        self.options["text_font"] = info.font_path
        self.options["text_font_family"] = info.font_family or "Default"
        self.options["text_size"] = max(1, int(info.size))
        self.options["text_align"] = info.align if info.align in ("left", "center", "right", "justify") else "left"
        self.options["text_line_spacing"] = max(0, int(info.line_spacing))
        self.options["text_rotation"] = float(info.rotation)
        self.foreground = color

    def text_edit_in_progress(self) -> bool:
        return self._text_before is not None

    def apply_text_style(self, *, transient: bool) -> None:
        """Restyle the active text layer from the panel and the foreground color.

        A drag (rotation, line spacing, or the palette) previews on every move
        and commits once on release. The preview is not an undo step. Releasing
        on the same style the drag started from leaves history alone.

        Flow:
        1. No active text layer: forget any preview and return.
        2. Build the style from the panel options plus the palette color,
           keeping the layer's words.
        3. While dragging, remember the pre-drag pixels on the first move,
           draw the preview, and stop.
        4. On release, put those pre-drag pixels back, then commit one edit
           if the style actually changed.
        """
        layer = self.active_layer()
        if layer is None or layer.kind != "text" or layer.text is None:
            self._text_before = None
            return
        info = self.text_info_from_options(layer.text.text)
        if transient:
            if self._text_before is None:
                self._text_before = visual_state(layer)
            if _text_same(layer.text, info):
                return
            self._write_text(layer, info)
            self.preview()
            return
        before = self._text_before
        self._text_before = None
        if before is not None:
            apply_visual(layer, before)
        if layer.text is None or _text_same(layer.text, info):
            if before is not None:
                self.preview()
            return
        self.update_text(layer, info)

    def _write_text(self, layer: Layer, info: TextInfo) -> None:
        image = render_text(info)
        # Keep the center stable when the render size changes (rotation, font size).
        cx = layer.x + layer.image.width / 2
        cy = layer.y + layer.image.height / 2
        layer.image = image
        layer.text = copy.deepcopy(info)
        layer.kind = "text"
        layer.x = int(round(cx - image.width / 2))
        layer.y = int(round(cy - image.height / 2))

    def update_text(self, layer: Layer, info: TextInfo) -> None:
        before = visual_state(layer)
        self._write_text(layer, info)
        new_state = visual_state(layer)
        apply_visual(layer, before)
        self.commit(MutateLayerCommand(layer, new_state, name="Edit text"))

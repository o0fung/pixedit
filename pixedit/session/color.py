"""Live color grade and one-shot filters.

Mixed into ``Session``. Slider drags preview from a snapshot and are not undo
steps. Apply keeps the grade as one step and returns the sliders to zero.
"""

from __future__ import annotations

from typing import Callable

from PIL import Image

from pixedit.adjust import apply_masked, color_grade
from pixedit.geom import selection_for_layer
from pixedit.model import Layer, MutateLayerCommand, apply_visual, visual_state

_COLOR_OPTION_KEYS = (
    "color_hue",
    "color_saturation",
    "color_brightness",
    "color_contrast",
    "color_highlight",
    "color_shade",
)


class ColorMixin:
    """Preview and commit a color grade, or run a menu filter."""

    def color_adjust_values(self) -> dict[str, float]:
        return {key: float(self.option(key, 0)) for key in _COLOR_OPTION_KEYS}

    def color_adjust_active(self) -> bool:
        return self._color_before is not None

    def _color_is_neutral(self, values: dict[str, float] | None = None) -> bool:
        data = self.color_adjust_values() if values is None else values
        return all(abs(value) < 1e-6 for value in data.values())

    def _zero_color_options(self) -> None:
        for key in _COLOR_OPTION_KEYS:
            self.options[key] = 0

    def _restore_color_snapshot(self) -> bool:
        before = self._color_before
        layer = self._color_layer
        self._color_before = None
        self._color_layer = None
        if before is None or layer is None:
            return False
        apply_visual(layer, before)
        return True

    def _discard_color_preview(self) -> None:
        """Put the snapshot back and zero the sliders. The caller notifies."""
        self._restore_color_snapshot()
        self._zero_color_options()

    def cancel_color_adjust(self) -> bool:
        """Drop an unapplied grade. True when a preview was actually showing.

        Escape uses that flag so a preview is cancelled first and a selection
        is cleared only when nothing was previewing. Switching tools and
        selecting another layer use the same restore.
        """
        if self._ending_color:
            return False
        self._ending_color = True
        try:
            neutral = self._color_is_neutral()
            restored = self._restore_color_snapshot()
            self._zero_color_options()
            if restored or not neutral:
                self.notify("change")
            return restored
        finally:
            self._ending_color = False

    def preview_color_adjust(self) -> None:
        """Redraw the active layer from the sliders without writing history.

        Flow:
        1. A mask layer is refused. The sliders return to zero.
        2. All sliders at zero put the snapshot back, if a drag had started.
        3. The first non-zero value snapshots the layer, including text.
        4. Later values recompute from that snapshot so the sliders combine
           into one grade. The selection, when there is one, limits the grade.
        """
        layer = self.active_layer()
        if layer is None or layer.kind == "redaction":
            self.cancel_color_adjust()
            self.show_message("Choose a raster or text layer for this filter.")
            return
        if self._color_layer is not None and layer is not self._color_layer:
            self.cancel_color_adjust()
            return
        values = self.color_adjust_values()
        if self._color_is_neutral(values):
            if self._color_before is not None and self._color_layer is layer:
                apply_visual(layer, self._color_before)
                self._color_before = None
                self._color_layer = None
                self.preview()
            return
        if self._color_before is None:
            self._color_before = visual_state(layer)
            self._color_layer = layer
        self._paint_color_preview(layer)
        self.preview()

    def _paint_color_preview(self, layer: Layer) -> None:
        before = self._color_before
        if before is None:
            return
        values = self.color_adjust_values()
        edited = color_grade(
            before["image"],
            hue=values["color_hue"],
            saturation=values["color_saturation"],
            brightness=values["color_brightness"],
            contrast=values["color_contrast"],
            highlight=values["color_highlight"],
            shade=values["color_shade"],
        )
        selection = selection_for_layer(self.document, layer)
        layer.image = apply_masked(before["image"], edited, selection)
        # Text is shown as pixels for the preview. Cancel restores ``before``,
        # which still has the text metadata. Apply keeps the pixels.
        layer.kind = "raster"
        layer.text = None

    def commit_color_adjust(self) -> None:
        """Keep the preview as one undo step and return the sliders to zero."""
        layer = self._color_layer
        before = self._color_before
        if layer is None or before is None or self._color_is_neutral():
            self.cancel_color_adjust()
            return
        self._paint_color_preview(layer)
        new_state = visual_state(layer)
        new_state["kind"] = "raster"
        new_state["text"] = None
        apply_visual(layer, before)
        self._color_before = None
        self._color_layer = None
        self._zero_color_options()
        self.commit(MutateLayerCommand(layer, new_state, name="Color"))

    def run_filter(self, name: str, function: Callable[[Image.Image], Image.Image]) -> None:
        """Apply a pixel function to the active raster or text layer, honoring the selection."""
        if not self.require_page():
            return
        self._discard_color_preview()
        layer = self.active_layer()
        if layer is None or layer.kind == "redaction":
            self.show_message("Choose a raster or text layer for this filter.")
            return
        before = visual_state(layer)
        edited = function(layer.image)
        selection = selection_for_layer(self.document, layer)
        edited = apply_masked(layer.image, edited, selection)
        layer.image = edited
        layer.kind = "raster"
        layer.text = None
        new_state = visual_state(layer)
        apply_visual(layer, before)
        self.commit(MutateLayerCommand(layer, new_state, name=name))

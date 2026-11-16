"""Live color grade and one-shot filters.

Mixed into ``Session``. Slider drags preview from a snapshot and are not undo
steps. Apply keeps the grade as one step and returns the sliders to zero.
"""

from __future__ import annotations

from typing import Callable

from PIL import Image

from pixedit.adjust import apply_masked, color_grade
from pixedit.geom import placed_box, selection_for_layer
from pixedit.model import Layer, MutateLayerCommand, apply_visual, visual_state
from pixedit.proxy import downscale, preview_scale, set_preview_source

def _union_box(left, right):
    """The rectangle covering both boxes. Either may be missing."""
    if left is None:
        return right
    if right is None:
        return left
    return (min(left[0], right[0]), min(left[1], right[1]), max(left[2], right[2]), max(left[3], right[3]))


_COLOR_OPTION_KEYS = (
    "color_colorize",
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
        return bool(self._color_grades) or self._color_before is not None

    def _color_grade_targets(self) -> list[Layer]:
        """Pictures and text layers the sliders should grade.

        One selected layer is that layer, unless it is a mask. Two or more
        skip masks and grade the rest. An empty result means the status line
        should ask for a picture or text layer.
        """
        selected = self.selected_layers()
        if len(selected) > 1:
            return [layer for layer in selected if layer.kind in ("raster", "text")]
        layer = self.active_layer()
        if layer is None or layer.kind == "redaction":
            return []
        return [layer]

    def _color_is_neutral(self, values: dict[str, float] | None = None) -> bool:
        data = self.color_adjust_values() if values is None else values
        return all(abs(value) < 1e-6 for value in data.values())

    def _zero_color_options(self) -> None:
        for key in _COLOR_OPTION_KEYS:
            self.options[key] = 0

    def _restore_color_snapshot(self) -> bool:
        grades = list(self._color_grades)
        self._color_grades = []
        self._color_before = None
        self._color_layer = None
        restored = False
        for layer, before in grades:
            set_preview_source(layer, None)
            if before is None:
                continue
            apply_visual(layer, before)
            restored = True
        return restored

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
        """Redraw the selected layers from the sliders without writing history.

        Flow:
        1. A mask, or a selection of only masks, is refused. The sliders return
           to zero.
        2. A different set of layers than the one already previewing cancels
           that preview. Moving the active layer inside the same set does not.
        3. All sliders at zero, with Colorize off, put every snapshot back.
        4. The first non-zero value, or Colorize, snapshots each picture and text layer.
        5. Later values recompute from those snapshots so the sliders combine
           into one grade. A marching-ants selection limits each layer.
        """
        targets = self._color_grade_targets()
        if not targets:
            self.cancel_color_adjust()
            self.show_message("Choose a raster or text layer for this filter.")
            return
        stored = [layer for layer, _before in self._color_grades]
        if stored and [layer.id for layer in stored] != [layer.id for layer in targets]:
            self.cancel_color_adjust()
            return
        values = self.color_adjust_values()
        if self._color_is_neutral(values):
            if self._color_grades:
                self._restore_color_snapshot()
                self.preview()
            return
        if not self._color_grades:
            self._color_grades = [(layer, visual_state(layer)) for layer in targets]
            self._color_layer = targets[0]
            self._color_before = self._color_grades[0][1]
        bounds = None
        for layer, before in self._color_grades:
            self._paint_color_preview(layer, before, final=False)
            bounds = _union_box(bounds, placed_box(layer, self.document.width, self.document.height))
        self.preview(bounds)

    def _paint_color_preview(self, layer: Layer, before: dict | None, *, final: bool) -> None:
        if before is None:
            return
        values = self.color_adjust_values()
        source = before["image"]
        selection = selection_for_layer(self.document, layer)
        zoom = float(getattr(self, "view_zoom", 1.0) or 1.0)
        scale = 1.0 if final else preview_scale(source.width, source.height, zoom)

        def grade(image: Image.Image) -> Image.Image:
            return color_grade(
                image,
                hue=values["color_hue"],
                saturation=values["color_saturation"],
                brightness=values["color_brightness"],
                contrast=values["color_contrast"],
                highlight=values["color_highlight"],
                shade=values["color_shade"],
                colorize=bool(values["color_colorize"]),
            )

        if scale >= 1.0:
            set_preview_source(layer, None)
            layer.image = apply_masked(source, grade(source), selection)
        else:
            small = downscale(source, scale)
            small_selection = None if selection is None else selection.resize(small.size, Image.Resampling.NEAREST)
            layer.image = source
            set_preview_source(layer, apply_masked(small, grade(small), small_selection))
        # Text is shown as pixels for the preview. Cancel restores ``before``,
        # which still has the text metadata. Apply keeps the pixels.
        layer.kind = "raster"
        layer.text = None

    def commit_color_adjust(self) -> None:
        """Keep the preview as one undo step and return the sliders to zero."""
        grades = list(self._color_grades)
        if not grades or self._color_is_neutral():
            self.cancel_color_adjust()
            return
        edits = []
        for layer, before in grades:
            self._paint_color_preview(layer, before, final=True)
            new_state = visual_state(layer)
            new_state["kind"] = "raster"
            new_state["text"] = None
            edits.append((layer, new_state, before))
        self._color_grades = []
        self._color_before = None
        self._color_layer = None
        self._zero_color_options()
        self.commit_layer_visuals(edits, "Color")

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

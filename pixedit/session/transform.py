"""Crop, canvas size, image size, and flips and rotations.

Mixed into ``Session``. Canvas commands replace the page. Layer commands
change the selected layers and leave the canvas size alone.
"""

from __future__ import annotations

from typing import Any

from pixedit.geom import (
    MAX_EDGE,
    alpha_bbox,
    canvas_flip,
    canvas_rotate_ccw,
    canvas_rotate_cw,
    canvas_window,
    crop_document,
    flip_visual,
    flip_visual_across,
    refit_canvas,
    resample_document,
    rotate_visual,
    rotate_visual_around,
    scale_visual,
    trim_rect,
    union_bounds,
)
from pixedit.guides import flip_guides, rotate_guides_ccw, rotate_guides_cw, scale_guides, shift_guides
from pixedit.model import (
    CompoundCommand,
    Layer,
    MutateLayerCommand,
    ReplaceDocumentContent,
    SetGuidesCommand,
    SetPositionCommand,
    apply_visual,
    visual_state,
)


class TransformMixin:
    """Resize the canvas or transform the selected layers."""

    def crop_to(self, left: int, top: int, width: int, height: int) -> None:
        """Crop or expand the canvas to a window. A negative origin grows that edge."""
        left = int(left)
        top = int(top)
        if width < 1 or height < 1:
            return
        width = max(1, min(MAX_EDGE, int(width)))
        height = max(1, min(MAX_EDGE, int(height)))
        if left == 0 and top == 0 and width == self.document.width and height == self.document.height:
            return
        layers, selection = crop_document(self.document, left, top, width, height)
        guides = shift_guides(self.document.group.guides_v, self.document.group.guides_h, left, top)
        self.commit(ReplaceDocumentContent(self.document, layers, width, height, selection, name="Crop", guides=guides))

    def resize_canvas(
        self,
        width: int,
        height: int,
        anchor: str = "c",
        fill: tuple[int, ...] | None = None,
    ) -> None:
        """Change the frame without resampling. ``fill`` colors a new margin."""
        width = max(1, min(MAX_EDGE, int(width)))
        height = max(1, min(MAX_EDGE, int(height)))
        if width == self.document.width and height == self.document.height:
            self.show_message("Canvas is already that size.")
            return
        layers, selection = refit_canvas(self.document, width, height, anchor, fill)
        origin_x, origin_y, _, _ = canvas_window(self.document.width, self.document.height, width, height, anchor)
        guides = shift_guides(self.document.group.guides_v, self.document.group.guides_h, origin_x, origin_y)
        self.commit(
            ReplaceDocumentContent(
                self.document, layers, width, height, selection, name="Canvas size", guides=guides
            )
        )

    def resize_image(
        self,
        width: int,
        height: int,
        dpi: float,
        resample: bool = True,
        interpolation: str = "lanczos",
    ) -> None:
        """Resample the picture, or change only its resolution.

        Resample on rebuilds every layer at the new pixel size and keeps the
        dpi the caller passes. Resample off leaves the pixels alone and stores
        the dpi, which changes the printed size and nothing else.
        """
        width = max(1, min(MAX_EDGE, int(width)))
        height = max(1, min(MAX_EDGE, int(height)))
        dpi = float(dpi)
        if dpi <= 0:
            dpi = 72.0
        document = self.document
        same_pixels = width == document.width and height == document.height
        same_dpi = abs(dpi - float(document.dpi)) < 1e-6
        if same_pixels and same_dpi:
            return
        if not resample or same_pixels:
            layers = [layer.clone() for layer in document.layers]
            selection = None if document.selection is None else document.selection.copy()
            self.commit(
                ReplaceDocumentContent(
                    document,
                    layers,
                    document.width,
                    document.height,
                    selection,
                    name="Image size",
                    dpi=dpi,
                )
            )
            return
        layers, selection = resample_document(document, width, height, interpolation)
        guides = scale_guides(
            document.group.guides_v,
            document.group.guides_h,
            document.width,
            document.height,
            width,
            height,
        )
        self.commit(
            ReplaceDocumentContent(
                document, layers, width, height, selection, name="Image size", dpi=dpi, guides=guides
            )
        )

    def crop_to_selection(self) -> None:
        """Crop the canvas to the selection's opaque bounds.

        The Image menu and the Select panel share this path. An empty
        selection explains itself and leaves the canvas alone.
        """
        selection = self.document.selection
        if selection is None:
            self.show_message("Draw a selection first.")
            return
        box = alpha_bbox(selection)
        if box is None:
            self.show_message("The selection is empty.")
            return
        left, top, right, bottom = box
        self.crop_to(left, top, right - left, bottom - top)

    def trim(self) -> None:
        rect = trim_rect(self.document)
        if rect is None:
            self.show_message("Nothing opaque to trim to.")
            return
        left, top, width, height = rect
        if left == 0 and top == 0 and width == self.document.width and height == self.document.height:
            self.show_message("Canvas is already tight.")
            return
        self.crop_to(left, top, width, height)

    def transform_canvas(self, kind: str) -> None:
        if kind == "rotate-cw":
            layers, width, height, selection = canvas_rotate_cw(self.document)
            name = "Rotate canvas"
        elif kind == "rotate-ccw":
            layers, width, height, selection = canvas_rotate_ccw(self.document)
            name = "Rotate canvas"
        elif kind == "flip-h":
            layers, width, height, selection = canvas_flip(self.document, True)
            name = "Flip canvas"
        elif kind == "flip-v":
            layers, width, height, selection = canvas_flip(self.document, False)
            name = "Flip canvas"
        else:
            return
        group = self.document.group
        if kind == "rotate-cw":
            guides = rotate_guides_cw(group.guides_v, group.guides_h, self.document.height)
        elif kind == "rotate-ccw":
            guides = rotate_guides_ccw(group.guides_v, group.guides_h, self.document.width)
        elif kind == "flip-h":
            guides = flip_guides(group.guides_v, group.guides_h, self.document.width, self.document.height, horizontal_flip=True)
        else:
            guides = flip_guides(group.guides_v, group.guides_h, self.document.width, self.document.height, horizontal_flip=False)
        self.commit(ReplaceDocumentContent(self.document, layers, width, height, selection, name=name, guides=guides))

    def set_guides(self, vertical: tuple[int, ...] | list[int], horizontal: tuple[int, ...] | list[int]) -> None:
        """Replace the guides on the page on screen. One undo step."""
        new_v = tuple(sorted({int(round(value)) for value in vertical}))
        new_h = tuple(sorted({int(round(value)) for value in horizontal}))
        group = self.document.group
        if new_v == tuple(sorted(group.guides_v)) and new_h == tuple(sorted(group.guides_h)):
            return
        self.commit(SetGuidesCommand(self.document, new_v, new_h))

    def flip_active_horizontal(self) -> None:
        self.transform_selected("flip-h")

    def flip_active_vertical(self) -> None:
        self.transform_selected("flip-v")

    def rotate_active_cw(self) -> None:
        self.transform_selected("rotate-cw")

    def rotate_active_ccw(self) -> None:
        self.transform_selected("rotate-ccw")

    def transform_selected(self, kind: str) -> None:
        """Flip or quarter-turn every selected layer around one shared center.

        The layers stay separate. One selected layer matches a flip or 90°
        turn of that layer alone, including the undo name. The Scale and
        Rotate dialogs stay on the active layer via ``transform_active``.
        """
        self._discard_color_preview()
        layers = self.selected_layers()
        if not layers:
            return
        states = [visual_state(layer) for layer in layers]
        box = union_bounds(states)
        center = (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)
        if kind == "flip-h":
            new_states = [flip_visual_across(state, True, center) for state in states]
            name = "Flip horizontal"
        elif kind == "flip-v":
            new_states = [flip_visual_across(state, False, center) for state in states]
            name = "Flip vertical"
        elif kind == "rotate-cw":
            new_states = [rotate_visual_around(state, 90.0, center) for state in states]
            name = "Rotate"
        elif kind == "rotate-ccw":
            new_states = [rotate_visual_around(state, -90.0, center) for state in states]
            name = "Rotate"
        else:
            return
        self.commit_layer_visuals(list(zip(layers, new_states, states)), name)

    def transform_active(self, kind: str, **kwargs: Any) -> None:
        self._discard_color_preview()
        layer = self.active_layer()
        if layer is None:
            return
        state = visual_state(layer)
        if kind == "scale":
            new_state = scale_visual(state, int(kwargs["width"]), int(kwargs["height"]))
            name = "Scale"
        elif kind == "rotate":
            new_state = rotate_visual(state, float(kwargs["degrees"]))
            name = "Rotate"
        elif kind == "flip-h":
            new_state = flip_visual(state, True)
            name = "Flip horizontal"
        elif kind == "flip-v":
            new_state = flip_visual(state, False)
            name = "Flip vertical"
        else:
            return
        self.commit(MutateLayerCommand(layer, new_state, name=name))

    def move_layer_to(self, layer: Layer, x: int, y: int, old_x: int, old_y: int) -> None:
        if layer.x == old_x and layer.y == old_y and (x != old_x or y != old_y):
            pass
        if x == old_x and y == old_y:
            return
        self.commit(SetPositionCommand(layer, x, y, old_x=old_x, old_y=old_y))

    def apply_layer_image(self, layer: Layer, new_state: dict[str, Any], name: str, before: dict[str, Any] | None = None) -> None:
        if before is not None:
            apply_visual(layer, before)
        self.commit(MutateLayerCommand(layer, new_state, old_state=before, name=name))

    def commit_layer_moves(self, moves: list[tuple[Layer, int, int, int, int]]) -> None:
        """Move several layers as one undo step.

        Each tuple is ``(layer, x, y, old_x, old_y)``. One layer keeps the
        existing "Move layer" name. Several layers are one "Move layers" step.
        """
        commands = []
        for layer, x, y, old_x, old_y in moves:
            if x == old_x and y == old_y:
                continue
            commands.append(SetPositionCommand(layer, x, y, old_x=old_x, old_y=old_y))
        self._commit_group(commands, "Move layers")

    def commit_layer_visuals(self, edits: list[tuple[Layer, dict[str, Any], dict[str, Any]]], name: str) -> None:
        """Replace each layer's pixels from a live preview, as one undo step.

        Each tuple is ``(layer, new_state, before)``. The layer is put back to
        ``before`` first, matching ``apply_layer_image``, so undo restores the
        gesture start rather than the preview.
        """
        commands = []
        for layer, new_state, before in edits:
            apply_visual(layer, before)
            commands.append(MutateLayerCommand(layer, new_state, old_state=before, name=name))
        self._commit_group(commands, name)

    def _commit_group(self, commands: list[Any], name: str) -> None:
        if not commands:
            return
        if len(commands) == 1:
            self.commit(commands[0])
            return
        self.commit(CompoundCommand(commands, name))

"""Editing session shared by the window, tools, and plugins.

The session owns the document, the undo history, the foreground color, and
tool options. It does not import Qt. The window listens for ``change`` and
``preview`` notifications and redraws.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Callable

import numpy as np
from PIL import Image

from pixedit.composite import baked_placement, composite, merge_pair
from pixedit.factory import blank_layer, empty_raster_layer, raster_layer, redaction_layer, text_layer
from pixedit.geom import (
    alpha_bbox,
    canvas_flip,
    canvas_rotate_ccw,
    canvas_rotate_cw,
    covers_document,
    MAX_EDGE,
    crop_document,
    flip_visual,
    flip_visual_across,
    refit_canvas,
    resample_document,
    rotate_visual,
    rotate_visual_around,
    scale_visual,
    union_bounds,
    selection_for_layer,
    tighten_raster_layer,
    trim_rect,
)
from pixedit.model import (
    CompoundCommand,
    DeleteGroupCommand,
    DeleteLayersCommand,
    Document,
    DuplicateLayerCommand,
    History,
    InsertGroupCommand,
    InsertLayerCommand,
    Layer,
    LayerGroup,
    MoveLayersToGroupCommand,
    MutateLayerCommand,
    RedactionInfo,
    RenameGroupCommand,
    ReorderGroupsCommand,
    ReorderStackCommand,
    ReplaceDocumentContent,
    SetLayerAttrCommand,
    SetPositionCommand,
    SetRedactionSettings,
    SetSelectionCommand,
    TextInfo,
    apply_visual,
    clamp_mask_amount,
    contiguous_mask_span,
    layers_leaving,
    mask_target,
    masks_for,
    move_stack,
    new_document,
    new_id,
    slot_above_block,
    visual_state,
)
from pixedit.textutil import default_font_path, render_text

Listener = Callable[[str], None]

_COLOR_OPTION_KEYS = (
    "color_hue",
    "color_saturation",
    "color_brightness",
    "color_contrast",
    "color_highlight",
    "color_shade",
)


def default_options() -> dict[str, Any]:
    return {
        "brush_size": 24,
        "brush_hardness": 85,
        "brush_opacity": 100,
        "mask_shape": "rect",
        "mask_kind": "pixelate",
        "mask_amount": 12,
        "color_hue": 0,
        "color_saturation": 0,
        "color_brightness": 0,
        "color_contrast": 0,
        "color_highlight": 0,
        "color_shade": 0,
        "select_mode": "rect",
        "tolerance": 32,
        "shape_kind": "rect",
        "stroke_width": 4,
        "shape_filled": False,
        "marker_size": 12,
        "rulers": True,
        "text_font": default_font_path(),
        "text_font_family": "DejaVu Sans",
        "text_size": 48,
        "text_align": "left",
        "text_line_spacing": 4,
        "text_rotation": 0,
    }


class Session:
    def __init__(self, document: Document | None = None) -> None:
        self.document = document or new_document(1280, 800, "white")
        self.history = History()
        # Set before any layer_index assignment. The setter drops an in-progress
        # color preview when the active layer changes, and it reads this flag.
        self._color_before: dict[str, Any] | None = None
        self._color_layer: Layer | None = None
        self._ending_color = False
        self._layer_index = len(self.document.layers) - 1
        # Ids in stack order, back to front. Assigning layer_index replaces
        # this with that one layer. select_layers is the multi-select path.
        self._selected_ids: list[str] = []
        # Each page remembers the layer that was active on it. Switching
        # back restores that choice instead of jumping to the top layer.
        self._group_ui: dict[str, tuple[int, list[str]]] = {}
        if 0 <= self._layer_index < len(self.document.layers):
            self._selected_ids = [self.document.layers[self._layer_index].id]
        self.foreground: tuple[int, int, int, int] = (20, 24, 28, 255)
        self.options = default_options()
        self.path: str | None = None
        # Filename of an opened image or PDF, such as "photo.png". This is not
        # a save path. Save still writes a .pix project and must not overwrite
        # the source file. A new canvas clears it.
        self.source_name: str | None = None
        self.dirty = False
        self.view_zoom = 1.0
        self._listeners: list[Listener] = []
        self._message_handler: Callable[[str], None] | None = None
        self._text_prompter: Callable[[TextInfo | None], TextInfo | None] | None = None
        self._redact_before: RedactionInfo | None = None
        self._text_before: dict | None = None
        # Set for the duration of a preview notify. The canvas reads it to
        # repaint one rectangle instead of the whole document.
        self.preview_bounds: tuple[int, int, int, int] | None = None

    @property
    def layer_index(self) -> int:
        return self._layer_index

    @layer_index.setter
    def layer_index(self, value: int) -> None:
        """Select this layer alone. A color preview is restored when the layer changes."""
        new_index = int(value)
        self._activate(new_index, self._ids_at(new_index))

    def _ids_at(self, index: int) -> list[str]:
        layers = self.document.layers
        if 0 <= index < len(layers):
            return [layers[index].id]
        return []

    def _activate(self, index: int, selected_ids: list[str]) -> None:
        """Store the active layer and the selection, then drop a stale color preview.

        The preview belongs to one layer. Selection is stored first so a
        refresh caused by restoring that preview sees the new set.
        """
        preview = self._color_layer
        self._layer_index = int(index)
        self._selected_ids = list(selected_ids)
        if preview is None:
            return
        current = None
        if self.document.layers and 0 <= self._layer_index < len(self.document.layers):
            current = self.document.layers[self._layer_index]
        if current is not preview:
            self.cancel_color_adjust()

    def selected_ids(self) -> list[str]:
        """Selected layer ids in stack order, back to front."""
        self.clamp_index()
        return list(self._selected_ids)

    def selected_layers(self) -> list[Layer]:
        """Selected layers in stack order, back to front."""
        chosen = set(self.selected_ids())
        return [layer for layer in self.document.layers if layer.id in chosen]

    def select_layers(self, layer_ids: list[str], active_id: str | None = None) -> None:
        """Select these layers. Paint, opacity, and blend stay on ``active_id``.

        Unknown ids are dropped. The stored order is the stack, back to front.
        ``active_id`` is included when it names a real layer. If it does not
        survive, the front-most remaining layer becomes active. An empty
        result leaves the current selection alone.
        """
        layers = self.document.layers
        known = {layer.id for layer in layers}
        wanted = {layer_id for layer_id in layer_ids if layer_id in known}
        if active_id in known:
            wanted.add(active_id)
        ordered = [layer.id for layer in layers if layer.id in wanted]
        if not ordered:
            return
        if active_id not in ordered:
            active_id = ordered[-1]
        self._activate(self.document.index_of(active_id), ordered)

    def add_listener(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def set_message_handler(self, handler: Callable[[str], None] | None) -> None:
        self._message_handler = handler

    def set_text_prompter(self, prompter: Callable[[TextInfo | None], TextInfo | None] | None) -> None:
        self._text_prompter = prompter

    def notify(self, kind: str = "change") -> None:
        for listener in list(self._listeners):
            listener(kind)

    def preview(self, bounds: tuple[int, int, int, int] | None = None) -> None:
        """Redraw after a live edit that is not an undo step yet.

        ``bounds`` is the canvas rectangle that changed, ``(x0, y0, x1, y1)``.
        The view can repaint just that rectangle. ``None`` rebuilds the whole view.
        """
        self.preview_bounds = bounds
        try:
            self.notify("preview")
        finally:
            self.preview_bounds = None

    def show_message(self, text: str) -> None:
        if self._message_handler is not None:
            self._message_handler(text)

    def prompt_text(self, initial: TextInfo | None) -> TextInfo | None:
        if self._text_prompter is None:
            return None
        return self._text_prompter(initial)

    def _stamp_group(self, command: Any) -> None:
        """Remember which page a layer edit belongs to.

        Group add, delete, and reorder commands set ``document_level`` and
        name their pages themselves. They must not be retargeted at whatever
        page happens to be showing when undo runs.
        """
        if getattr(command, "document_level", False):
            return
        if getattr(command, "group_id", None):
            return
        command.group_id = self.document.active_group_id

    def _remember_group_ui(self) -> None:
        group_id = self.document.active_group_id
        self._group_ui[group_id] = (self._layer_index, list(self._selected_ids))

    def _restore_group_ui(self, group_id: str) -> None:
        """Select the layer this page was using, or the front layer on a first visit."""
        layers = self.document.layers
        if not layers:
            self._layer_index = -1
            self._selected_ids = []
            return
        saved = self._group_ui.get(group_id)
        if saved is None:
            self._activate(len(layers) - 1, [layers[-1].id])
            return
        index, selected_ids = saved
        alive = {layer.id for layer in layers}
        kept = [layer_id for layer_id in selected_ids if layer_id in alive]
        if kept:
            if 0 <= index < len(layers) and layers[index].id in kept:
                active_id = layers[index].id
            else:
                active_id = kept[-1]
            self._activate(self.document.index_of(active_id), kept)
            return
        index = max(0, min(index, len(layers) - 1))
        self._activate(index, [layers[index].id])

    def _finish_group_change(self, before: str) -> None:
        current = self.document.active_group_id
        if current != before:
            self._restore_group_ui(current)
        self.clamp_index()

    def commit(self, command: Any) -> None:
        # An unapplied color grade is not part of the next undo step. The
        # notify below reloads the sliders after the options are zeroed.
        self._discard_color_preview()
        self._stamp_group(command)
        before = self.document.active_group_id
        self._remember_group_ui()
        self.history.push(command)
        self.dirty = True
        self._finish_group_change(before)
        self.notify("change")

    def _history_step(self, action: str) -> None:
        """Undo or redo, showing the page that step belongs to.

        A layer command is applied only after that page is active, because
        the command writes ``document.layers`` and that name means the page
        on screen. A group command then sets the active page itself. Either
        way the layer that was selected on the landing page comes back.
        """
        self._discard_color_preview()
        stack = self.history.undo_stack if action == "undo" else self.history.redo_stack
        if not stack:
            return
        command = stack[-1]
        before = self.document.active_group_id
        self._remember_group_ui()
        target = getattr(command, "group_id", None)
        if target and target != before and self.document.group_by_id(target) is not None:
            self.document.active_group_id = target
            self._restore_group_ui(target)
        applied = self.history.undo() if action == "undo" else self.history.redo()
        if not applied:
            return
        self.dirty = True
        self._finish_group_change(before)
        self.notify("change")

    def undo(self) -> None:
        self._history_step("undo")

    def redo(self) -> None:
        self._history_step("redo")

    def mark_clean(self) -> None:
        self.dirty = False
        self.notify("change")

    def clamp_index(self) -> None:
        """Keep the active index in range and drop ids of layers that are gone.

        This does not collapse a multi-selection. Assigning ``layer_index``
        does. The active id is always one of the remaining ids. If it was
        deleted, the front-most survivor becomes active.
        """
        layers = self.document.layers
        if not layers:
            self._layer_index = -1
            self._selected_ids = []
            return
        alive = {layer.id for layer in layers}
        kept = [layer_id for layer_id in self._selected_ids if layer_id in alive]
        active_id = None
        if 0 <= self._layer_index < len(layers):
            active_id = layers[self._layer_index].id
        if kept and active_id in kept and kept == self._selected_ids:
            return
        self._selected_ids = kept
        if kept and active_id in kept:
            return
        if kept:
            self._layer_index = self.document.index_of(kept[-1])
            return
        self._layer_index = max(0, min(self._layer_index, len(layers) - 1))
        self._selected_ids = [layers[self._layer_index].id]

    def active_layer(self) -> Layer | None:
        self.clamp_index()
        if self.layer_index < 0:
            return None
        return self.document.layers[self.layer_index]

    def option(self, key: str, default: Any = None) -> Any:
        return self.options.get(key, default)

    def replace_document(
        self,
        document: Document,
        path: str | None = None,
        source_name: str | None = None,
    ) -> None:
        """Replace the open document and clear the undo history.

        ``path`` is a ``.pix`` project Save will overwrite. ``source_name`` is
        only the filename of an opened image or PDF. Passing neither, as
        ``new_canvas`` does, leaves the document untitled.
        """
        self.document = document
        self.history = History()
        self._group_ui = {}
        self.layer_index = len(document.layers) - 1
        self.path = path
        self.source_name = source_name
        self.dirty = False
        self.notify("change")

    def new_canvas(self, width: int, height: int, background: str, dpi: float = 72.0) -> None:
        self.replace_document(new_document(width, height, background, dpi), None)

    def add_layer(self, layer: Layer, index: int | None = None, name: str = "Add layer") -> None:
        if index is None:
            index = len(self.document.layers) if self.layer_index < 0 else self.layer_index + 1
            # A new picture lands above a mask run, not between a picture and its masks.
            index = slot_above_block(self.document.layers, index)
        self.commit(InsertLayerCommand(self.document, layer, index, name=name))
        self.layer_index = max(0, min(index, len(self.document.layers) - 1))
        self.notify("change")

    def add_blank_layer(self) -> None:
        """Add an empty content-sized layer. The first stroke grows its bitmap."""
        number = len(self.document.layers) + 1
        layer = empty_raster_layer(f"Layer {number}")
        self.add_layer(layer, name="New layer")

    def duplicate_active(self) -> None:
        if self.layer_index < 0:
            return
        command = DuplicateLayerCommand(self.document, self.layer_index)
        self.commit(command)
        if command.copy is None:
            return
        for index, layer in enumerate(self.document.layers):
            if layer is command.copy:
                self.layer_index = index
                return

    def delete_active(self) -> None:
        """Delete the active layer. A picture takes its masks with it.

        Deleting a mask leaves the picture. Undo puts every removed layer
        back in its old slot.
        """
        if self.layer_index < 0 or not self.document.layers:
            return
        layer = self.document.layers[self.layer_index]
        if layer.kind in ("raster", "text"):
            victims = [layer, *masks_for(self.document.layers, layer)]
        else:
            victims = [layer]
        self.commit(DeleteLayersCommand(self.document, victims))
        self.clamp_index()

    def rename_active(self, name: str) -> None:
        layer = self.active_layer()
        if layer is None:
            return
        cleaned = name.strip() or layer.name
        if cleaned == layer.name:
            return
        self.commit(SetLayerAttrCommand(layer, "name", cleaned, name="Rename layer"))

    def move_active(self, delta: int) -> None:
        """``delta`` +1 moves the active layer up the stack (toward the front)."""
        if self.layer_index < 0:
            return
        self.move_layer(self.layer_index, self.layer_index + delta)

    def move_layer(self, old_index: int, new_index: int) -> None:
        """Move one layer from ``old_index`` to ``new_index``.

        Index 0 is the back of the stack (painted first). A higher index is
        in front. A picture and the masks that filter it move as one block,
        unless the drag only reorders those masks. One undo step. The moved
        layer stays selected.
        """
        count = len(self.document.layers)
        if not (0 <= old_index < count and 0 <= new_index < count):
            return
        if old_index == new_index:
            return
        reordered, landed = move_stack(self.document.layers, old_index, new_index)
        if [layer.id for layer in reordered] == [layer.id for layer in self.document.layers]:
            return
        self.commit(ReorderStackCommand(self.document, reordered))
        self.layer_index = landed

    def _next_group_name(self) -> str:
        names = {group.name for group in self.document.groups}
        number = len(self.document.groups) + 1
        while f"Group {number}" in names:
            number += 1
        return f"Group {number}"

    def activate_group(
        self,
        group_id: str,
        layer_ids: list[str] | None = None,
        active_id: str | None = None,
    ) -> None:
        """Show this page. The image area follows; other pages are not composited.

        Passing layer ids selects those layers on the page. With no ids, the
        layer that was active last time this page was shown stays active.
        """
        if self.document.group_by_id(group_id) is None:
            return
        same = group_id == self.document.active_group_id
        if same:
            if layer_ids or active_id:
                self.select_layers(list(layer_ids or []), active_id)
            return
        self._discard_color_preview()
        self._remember_group_ui()
        self.document.active_group_id = group_id
        if layer_ids or active_id:
            self.select_layers(list(layer_ids or []), active_id)
            if not self._selected_ids:
                self._restore_group_ui(group_id)
        else:
            self._restore_group_ui(group_id)
        self.clamp_index()
        self.notify("change")

    def add_group(self) -> None:
        """Add a blank page the same pixel size as the one on screen and show it."""
        current = self.document.group
        image = Image.new("RGBA", (current.width, current.height), (255, 255, 255, 255))
        group = LayerGroup(
            id=new_id(),
            name=self._next_group_name(),
            width=current.width,
            height=current.height,
            layers=[raster_layer("Background", image)],
        )
        index = self.document.group_index(current.id) + 1
        self.commit(InsertGroupCommand(self.document, group, index))

    def delete_active_group(self) -> None:
        """Remove the page on screen. The last page stays."""
        if len(self.document.groups) <= 1:
            self.show_message("The last group stays. A document always has one page.")
            return
        self.commit(DeleteGroupCommand(self.document, self.document.active_group_id))

    def rename_group(self, group_id: str, name: str) -> None:
        group = self.document.group_by_id(group_id)
        if group is None:
            return
        cleaned = name.strip() or group.name
        if cleaned == group.name:
            return
        self.commit(RenameGroupCommand(group, cleaned))

    def reorder_groups(self, ordered_ids: list[str]) -> None:
        """Put the pages in this order. That order is the PDF page order."""
        current = [group.id for group in self.document.groups]
        if ordered_ids == current:
            return
        if set(ordered_ids) != set(current) or len(ordered_ids) != len(current):
            return
        self.commit(ReorderGroupsCommand(self.document, ordered_ids))

    def move_layers_to_group(self, layer_ids: list[str], dest_group_id: str, dest_index: int) -> bool:
        """Move pictures onto another page. A mask follows its picture.

        Returns false when nothing eligible moved, so the layer list can snap
        back. The destination page is shown and the moved picture is selected.
        """
        dest = self.document.group_by_id(dest_group_id)
        if dest is None or not layer_ids:
            return False
        source: LayerGroup | None = None
        for group in self.document.groups:
            if any(layer.id == layer_ids[0] for layer in group.layers):
                source = group
                break
        if source is None or source.id == dest.id:
            return False
        leaving = layers_leaving(source.layers, layer_ids)
        if not leaving or not set(layer_ids) <= {layer.id for layer in leaving}:
            return False
        self.commit(
            MoveLayersToGroupCommand(
                self.document,
                source.id,
                dest.id,
                [layer.id for layer in leaving],
                dest_index,
            )
        )
        anchor = next((layer for layer in leaving if layer.kind != "redaction"), leaving[0])
        try:
            self.layer_index = self.document.index_of(anchor.id)
        except KeyError:
            self.clamp_index()
        self.notify("change")
        return True

    def merge_down_active(self) -> None:
        """Bake the active layer downward, as one undo step.

        A mask bakes into the picture it belongs to and leaves that picture
        in the stack. A picture bakes every mask that filters it, then merges
        that result into the layer below.
        """
        layer = self.active_layer()
        if layer is None:
            self.show_message("Nothing below this layer to merge into.")
            return
        if layer.kind == "redaction":
            self._bake_masks_through(layer, "Merge down")
            return
        if self.document.index_of(layer.id) <= 0:
            self.show_message("Nothing below this layer to merge into.")
            return
        self._merge_content_down(layer)

    def apply_active_mask(self) -> None:
        """Bake the active mask into its picture and remove the masks up through it.

        Masks of the same picture that sit above the active one stay live.
        Text becomes a normal raster layer. One undo step.
        """
        layer = self.active_layer()
        if layer is None or layer.kind != "redaction":
            self.show_message("Choose a mask layer to apply.")
            return
        self._bake_masks_through(layer, "Apply mask")

    def _stack_with_bake(self, target: Layer, masks: list[Layer]) -> list[Layer]:
        """A new stack with ``masks`` baked into a raster copy of ``target``.

        The copy keeps ``target``'s id so masks that were not baked still
        point at it. Text metadata is dropped because the pixels are now
        the rendered result.
        """
        image, x, y = baked_placement(self.document, target, masks)
        baked = target.clone()
        baked.id = target.id
        baked.image = image
        baked.x = x
        baked.y = y
        baked.kind = "raster"
        baked.text = None
        skip = {id(mask) for mask in masks}
        result: list[Layer] = []
        for layer in self.document.layers:
            if layer is target:
                result.append(baked)
            elif id(layer) in skip:
                continue
            else:
                result.append(layer)
        return result

    def _commit_stack(self, layers: list[Layer], name: str, select_id: str) -> None:
        self.commit(
            ReplaceDocumentContent(
                self.document,
                layers,
                self.document.width,
                self.document.height,
                self.document.selection,
                name=name,
            )
        )
        self.layer_index = self.document.index_of(select_id)

    def _bake_masks_through(self, mask: Layer, name: str) -> None:
        """Bake every mask of this picture up through ``mask``, then remove them.

        Masks above ``mask`` stay. They already saw the lower effects, so
        those lower masks have to be baked too. Leaving them would run the
        same effect again on the baked pixels.
        """
        target = mask_target(self.document.layers, mask)
        if target is None:
            self.show_message("This mask has no layer to apply to.")
            return
        mask_index = self.document.index_of(mask.id)
        through = [
            item
            for item in masks_for(self.document.layers, target)
            if self.document.index_of(item.id) <= mask_index
        ]
        if not through:
            return
        self._commit_stack(self._stack_with_bake(target, through), name, target.id)

    def _merge_content_down(self, layer: Layer) -> None:
        """Bake ``layer``'s masks, then merge that picture into the layer below."""
        masks = masks_for(self.document.layers, layer)
        working = self._stack_with_bake(layer, masks) if masks else list(self.document.layers)
        try:
            index = next(i for i, item in enumerate(working) if item.id == layer.id)
        except StopIteration:
            return
        if index <= 0:
            self.show_message("Nothing below this layer to merge into.")
            return
        lower = working[index - 1]
        upper = working[index]
        merged = merge_pair(lower, upper)
        merged.name = lower.name
        # The merged plate is a new object. Undo restores the previous stack,
        # including any masks this step baked in.
        layers = working[: index - 1] + [merged] + working[index + 1 :]
        self._commit_stack(layers, "Merge down", merged.id)

    def flatten(self) -> None:
        image = composite(self.document)
        layer = blank_layer("Flattened", self.document.width, self.document.height)
        layer.image = image
        self.commit(
            ReplaceDocumentContent(
                self.document,
                [layer],
                self.document.width,
                self.document.height,
                None,
                name="Flatten",
            )
        )
        self.layer_index = 0

    def set_visibility(self, layer: Layer, visible: bool) -> None:
        if layer.visible == visible:
            return
        command = SetLayerAttrCommand(layer, "visible", visible, name="Toggle layer")
        for group in self.document.groups:
            if any(item is layer for item in group.layers):
                command.group_id = group.id
                break
        self.commit(command)

    def toggle_selected_visibility(self) -> None:
        """Flip every selected layer's visibility as one undo step.

        Each layer inverts on its own, so a mix of shown and hidden layers
        swaps instead of collapsing to all shown or all hidden. A second
        click restores that mix. Layers outside the selection stay put.
        The row checkbox still changes only the layer that was clicked.
        """
        layers = self.selected_layers()
        if not layers:
            return
        commands = [
            SetLayerAttrCommand(layer, "visible", not layer.visible, name="Toggle visibility")
            for layer in layers
        ]
        if len(commands) == 1:
            self.commit(commands[0])
            return
        self.commit(CompoundCommand(commands, "Toggle visibility"))

    def set_opacity(self, layer: Layer, opacity: float) -> None:
        opacity = min(1.0, max(0.0, float(opacity)))
        if abs(layer.opacity - opacity) < 1e-6:
            return
        self.commit(SetLayerAttrCommand(layer, "opacity", opacity, name="Layer opacity"))

    def set_blend(self, layer: Layer, mode: str) -> None:
        if layer.blend_mode == mode:
            return
        self.commit(SetLayerAttrCommand(layer, "blend_mode", mode, name="Blend mode"))

    def set_selection(self, selection: Image.Image | None, name: str = "Selection") -> None:
        introducing = self.document.selection is None and selection is not None
        self.commit(SetSelectionCommand(self.document, selection, name=name))
        if introducing:
            self.show_message(
                "Selection clips brush, fill, shapes, and filters. "
                "Ctrl+X cuts it to a new layer · Ctrl+C copies it · Delete clears pixels · "
                "Ctrl+D or Esc deselects."
            )

    def clear_selection_contents(self) -> None:
        """Erase pixels of the active raster or text layer inside the selection.

        Text is rasterized in the same undo step. A layer that already covers
        the document keeps its size, so a hole stays a hole in the plate.
        Smaller layers are cropped back to whatever opaque pixels remain.
        Redaction masks are left alone.
        """
        if self.document.selection is None:
            self.show_message("No selection. Draw one with Select (S), then Delete clears its pixels.")
            return
        layer = self.active_layer()
        if layer is None:
            self.show_message("No layer to clear.")
            return
        if layer.kind == "redaction":
            self.show_message("Delete clears pixels on a raster layer. Use the eraser to lift a mask.")
            return
        before = visual_state(layer)
        local = selection_for_layer(self.document, layer)
        if local is None:
            self.show_message("The selection does not overlap the active layer.")
            return
        selected = np.asarray(local) > 0
        if not bool(selected.any()):
            self.show_message("The selection does not overlap the active layer.")
            return
        preserve = covers_document(
            layer.x, layer.y, layer.image.width, layer.image.height, self.document.width, self.document.height
        )
        arr = np.array(layer.image.convert("RGBA"))
        # Zero RGB as well as alpha so a later tight crop and any sampling
        # of the cleared pixels do not resurrect the old color.
        arr[selected, :3] = 0
        arr[selected, 3] = 0
        layer.image = Image.fromarray(arr, "RGBA")
        layer.kind = "raster"
        layer.text = None
        if not preserve:
            tighten_raster_layer(layer)
        new_state = visual_state(layer)
        apply_visual(layer, before)
        self.commit(MutateLayerCommand(layer, new_state, name="Clear selection"))

    def select_all(self) -> None:
        mask = Image.new("L", (self.document.width, self.document.height), 255)
        self.set_selection(mask, name="Select all")

    def deselect(self) -> None:
        if self.document.selection is None:
            self.show_message("No selection. Draw one with Select (S). Esc or Ctrl+D deselects.")
            return
        self.set_selection(None, name="Deselect")
        self.show_message("Deselected.")

    def copy_selection_to_layer(self) -> None:
        """Copy opaque selected pixels onto a new layer above the active one."""
        self._selection_to_new_layer(clear=False)

    def cut_selection_to_layer(self) -> None:
        """Move opaque selected pixels onto a new layer and clear them on the source."""
        self._selection_to_new_layer(clear=True)

    def _selection_to_new_layer(self, *, clear: bool) -> None:
        # One undo step either duplicates the selected pixels or moves them.
        # Cut rasterizes text, the same way Delete does, because the source
        # bitmap changes. Copy leaves a text layer editable.
        if self.document.selection is None:
            self.show_message("Draw a selection first. Select (S), then Ctrl+C copies it to a new layer.")
            return
        layer = self.active_layer()
        if layer is None:
            self.show_message("Choose a layer to copy from.")
            return
        if layer.kind == "redaction":
            self.show_message("A mask is not pixels. Copy the layer it belongs to.")
            return
        local = selection_for_layer(self.document, layer)
        if local is None:
            self.show_message("The selection does not overlap the active layer.")
            return
        mask = np.asarray(local) > 0
        if not bool(mask.any()):
            self.show_message("The selection does not overlap the active layer.")
            return
        source = np.array(layer.image.convert("RGBA"))
        opaque = mask & (source[..., 3] > 0)
        ys, xs = np.nonzero(opaque)
        if len(xs) == 0:
            self.show_message("The selection has no opaque pixels on the active layer.")
            return
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        extracted = source.copy()
        extracted[~mask] = 0
        placed = raster_layer(
            f"{layer.name} copy",
            Image.fromarray(extracted[y0:y1, x0:x1], "RGBA"),
            layer.x + x0,
            layer.y + y0,
        )
        index = len(self.document.layers) if self.layer_index < 0 else self.layer_index + 1
        index = slot_above_block(self.document.layers, index)
        name = "Cut selection" if clear else "Copy selection"
        commands: list[Any] = []
        if clear:
            before = visual_state(layer)
            cleared = source.copy()
            cleared[mask, :3] = 0
            cleared[mask, 3] = 0
            scratch = layer.clone()
            scratch.image = Image.fromarray(cleared, "RGBA")
            scratch.kind = "raster"
            scratch.text = None
            full = covers_document(
                layer.x,
                layer.y,
                layer.image.width,
                layer.image.height,
                self.document.width,
                self.document.height,
            )
            if not full:
                tighten_raster_layer(scratch)
            commands.append(MutateLayerCommand(layer, visual_state(scratch), old_state=before, name=name))
        commands.append(InsertLayerCommand(self.document, placed, index, name=name))
        self.commit(CompoundCommand(commands, name))
        self.layer_index = index
        verb = "Cut" if clear else "Copied"
        self.show_message(f"{verb} the selection to a new layer. Esc or Ctrl+D deselects.")

    def place_layers(self, layers: list[Layer]) -> None:
        """Add image layers on top of the current document, centered.

        The document size and the layers already there stay put. Each placed
        bitmap is cropped to its opaque pixels, then centered. Transparent
        padding in a PNG does not become a full-canvas plate. Undo removes
        every layer this call added.
        """
        copies = [layer.clone() for layer in layers if layer.kind != "redaction"]
        if not copies:
            self.show_message("That file has no image layers to place.")
            return
        index = len(self.document.layers) if self.layer_index < 0 else self.layer_index + 1
        index = slot_above_block(self.document.layers, index)
        commands = []
        for offset, layer in enumerate(copies):
            tighten_raster_layer(layer)
            layer.x = int(round((self.document.width - layer.image.width) / 2))
            layer.y = int(round((self.document.height - layer.image.height) / 2))
            commands.append(InsertLayerCommand(self.document, layer, index + offset, name="Place image"))
        self.commit(CompoundCommand(commands, "Place image"))
        # The first placed frame is the visible one for a multi-frame file.
        self.layer_index = index
        count = len(copies)
        noun = "layer" if count == 1 else "layers"
        self.show_message(f"Placed {count} {noun}. The current document was kept.")

    def place_groups(self, groups: list[LayerGroup]) -> None:
        """Append each page as its own group. The pages already open stay put.

        Used when placing a PDF. The new pages are added in the order given,
        and the first of them is shown. Undo removes every page this call added.
        """
        copies = [group.clone() for group in groups]
        if not copies:
            self.show_message("That file has no pages to place.")
            return
        start = len(self.document.groups)
        commands: list[Any] = [
            InsertGroupCommand(self.document, group, start + offset) for offset, group in enumerate(copies)
        ]
        command = CompoundCommand(commands, "Place pages")
        command.document_level = True
        self.commit(command)
        first = copies[0]
        if self.document.active_group_id != first.id:
            self.activate_group(first.id)
        count = len(copies)
        noun = "page" if count == 1 else "pages"
        self.show_message(f"Placed {count} {noun}. Each one is its own group.")

    def invert_selection(self) -> None:
        if self.document.selection is None:
            self.set_selection(Image.new("L", (self.document.width, self.document.height), 0), name="Invert selection")
            return
        from pixedit.geom import place_image

        full = place_image(self.document.selection, 0, 0, self.document.width, self.document.height, 0)
        inverted = Image.eval(full, lambda value: 255 - value)
        self.set_selection(inverted, name="Invert selection")

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
        self.commit(ReplaceDocumentContent(self.document, layers, width, height, selection, name="Crop"))

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
        self.commit(
            ReplaceDocumentContent(self.document, layers, width, height, selection, name="Canvas size")
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
        self.commit(
            ReplaceDocumentContent(
                document, layers, width, height, selection, name="Image size", dpi=dpi
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
        self.commit(ReplaceDocumentContent(self.document, layers, width, height, selection, name=name))

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

    def begin_raster_edit(self, *, paint_redaction: bool = False) -> tuple[Layer, dict[str, Any] | None, bool]:
        """Prepare a layer for a brush, fill, or shape.

        Returns ``(layer, before_state, created)``. When ``created`` is true the
        layer was inserted immediately for preview and is not yet an undo step;
        ``finish_raster_edit`` turns that into one command. Text layers are
        rasterized after ``before_state`` is captured, so undo restores the text.

        The bitmap is not expanded to the document. Callers grow it around the
        gesture, and ``finish_raster_edit`` crops transparent margins unless
        the layer already covered the canvas.
        """
        self._discard_color_preview()
        layer = self.active_layer()
        if layer is not None and layer.kind == "redaction" and paint_redaction:
            return layer, visual_state(layer), False
        if layer is None or layer.kind == "redaction":
            created = empty_raster_layer(f"Layer {len(self.document.layers) + 1}")
            index = len(self.document.layers) if self.layer_index < 0 else self.layer_index + 1
            index = slot_above_block(self.document.layers, index)
            self.document.layers.insert(index, created)
            self.layer_index = index
            return created, None, True
        before = visual_state(layer)
        if layer.kind == "text":
            layer.kind = "raster"
            layer.text = None
        return layer, before, False

    def _fit_content_unless_full_canvas(self, layer: Layer, before: dict[str, Any] | None) -> None:
        """Crop a layer to its opaque pixels unless it was already a full-canvas plate.

        Redaction masks stay document-sized. A background, import, or flatten
        covers the document in ``before`` and keeps that extent even if the
        gesture only touched a corner.
        """
        if layer.kind == "redaction":
            return
        if before is not None:
            image = before["image"]
            if covers_document(
                int(before["x"]),
                int(before["y"]),
                image.width,
                image.height,
                self.document.width,
                self.document.height,
            ):
                return
        tighten_raster_layer(layer)

    def finish_raster_edit(
        self,
        layer: Layer,
        before: dict[str, Any] | None,
        created: bool,
        name: str,
    ) -> None:
        self._fit_content_unless_full_canvas(layer, before)
        if created:
            try:
                index = self.document.layers.index(layer)
            except ValueError:
                index = len(self.document.layers)
            else:
                self.document.layers.pop(index)
            self.add_layer(layer, index=index, name=name)
            return
        if before is None:
            return
        new_state = visual_state(layer)
        apply_visual(layer, before)
        self.commit(MutateLayerCommand(layer, new_state, name=name))

    def cancel_raster_edit(self, layer: Layer, before: dict[str, Any] | None, created: bool) -> None:
        if created:
            self.document.layers = [item for item in self.document.layers if item is not layer]
            self.clamp_index()
            self.preview()
            return
        if before is not None:
            apply_visual(layer, before)
            self.preview()

    def ensure_redaction_layer(self) -> tuple[Layer, bool]:
        """Return the active mask, or create one for the active picture.

        A second stroke on a mask adds to that mask and keeps its picture.
        A stroke on a picture or text layer creates a mask for that layer,
        placed above any masks it already has. The new layer is inserted for
        preview only. The mask tool removes it and re-adds it through
        ``add_layer`` on release so undo drops the whole region in one step.
        """
        self._discard_color_preview()
        layer = self.active_layer()
        if layer is not None and layer.kind == "redaction":
            return layer, False
        target = layer if layer is not None and layer.kind in ("raster", "text") else None
        if target is None:
            for candidate in reversed(self.document.layers):
                if candidate.kind in ("raster", "text"):
                    target = candidate
                    break
        mask = Image.new("L", (self.document.width, self.document.height), 0)
        info = self.current_mask_info()
        created = redaction_layer(
            "Mask",
            mask,
            info.kind,
            info.amount,
            color=info.color,
            seed=info.seed,
            target_id=None if target is None else target.id,
        )
        if target is None:
            index = len(self.document.layers)
            self.show_message("Add a picture layer before drawing a mask.")
        else:
            span = contiguous_mask_span(self.document.layers, target)
            index = len(self.document.layers) if span is None else span[1] + 1
        self.document.layers.insert(index, created)
        self.layer_index = index
        return created, True

    def current_mask_info(self, previous: RedactionInfo | None = None) -> RedactionInfo:
        """Effect settings from the Mask panel.

        A new region gets a fresh noise seed. Another stroke on the same layer
        keeps that seed, and keeps the stored color unless the effect is Solid,
        which takes the foreground swatch.
        """
        kind = str(self.option("mask_kind", "pixelate"))
        amount = clamp_mask_amount(kind, float(self.option("mask_amount", 12)))
        if kind == "solid":
            color = tuple(int(channel) for channel in self.foreground)
        elif previous is not None:
            color = tuple(int(channel) for channel in previous.color)
        else:
            color = (0, 0, 0, 255)
        seed = int(previous.seed) if previous is not None else random.randrange(1, 2**31)
        return RedactionInfo(kind, amount, color, seed)

    def set_redaction_settings(
        self,
        kind: str,
        amount: float,
        *,
        transient: bool,
        color: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Update the active mask layer while a slider is dragged, then commit."""
        self._discard_color_preview()
        layer = self.active_layer()
        if layer is None or layer.kind != "redaction" or layer.redaction is None:
            self._redact_before = None
            return
        base = self._redact_before or layer.redaction
        chosen = tuple(int(channel) for channel in (color if color is not None else base.color))
        amount = clamp_mask_amount(str(kind), float(amount))
        updated = RedactionInfo(str(kind), amount, chosen, int(base.seed))
        if transient:
            if self._redact_before is None:
                self._redact_before = copy.deepcopy(layer.redaction)
            layer.redaction = updated
            self.preview()
            return
        old = self._redact_before or copy.deepcopy(layer.redaction)
        self._redact_before = None
        layer.redaction = old
        if old.kind == kind and abs(old.amount - amount) < 1e-6 and tuple(int(channel) for channel in old.color) == chosen:
            self.preview()
            return
        self.commit(SetRedactionSettings(layer, str(kind), amount, chosen))

    def recolor_solid_mask(self, *, transient: bool) -> None:
        """While the Mask tool is active, the foreground swatch is Solid's color."""
        layer = self.active_layer()
        if layer is None or layer.kind != "redaction" or layer.redaction is None:
            return
        if layer.redaction.kind != "solid":
            return
        self.set_redaction_settings(
            layer.redaction.kind,
            layer.redaction.amount,
            transient=transient,
            color=tuple(int(channel) for channel in self.foreground),
        )

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
        from pixedit.adjust import apply_masked, color_grade

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
        self._discard_color_preview()
        layer = self.active_layer()
        if layer is None or layer.kind == "redaction":
            self.show_message("Choose a raster or text layer for this filter.")
            return
        before = visual_state(layer)
        from pixedit.adjust import apply_masked
        from pixedit.geom import selection_for_layer

        edited = function(layer.image)
        selection = selection_for_layer(self.document, layer)
        edited = apply_masked(layer.image, edited, selection)
        layer.image = edited
        layer.kind = "raster"
        layer.text = None
        new_state = visual_state(layer)
        apply_visual(layer, before)
        self.commit(MutateLayerCommand(layer, new_state, name=name))


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

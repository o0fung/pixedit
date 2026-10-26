"""Layer stack, pages, merge, and placing files.

Mixed into ``Session``. A page is a group. Masks stay with the picture they
filter. Every change goes through ``commit`` so it is one undo step.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from pixedit.composite import baked_placement, composite, merge_pair
from pixedit.factory import blank_layer, empty_raster_layer, raster_layer
from pixedit.geom import tighten_raster_layer
from pixedit.model import (
    CompoundCommand,
    DeleteGroupCommand,
    DeleteLayersCommand,
    DuplicateLayerCommand,
    InsertGroupCommand,
    InsertLayerCommand,
    Layer,
    LayerGroup,
    MoveLayersToGroupCommand,
    RenameGroupCommand,
    ReorderGroupsCommand,
    ReorderStackCommand,
    ReplaceDocumentContent,
    SetLayerAttrCommand,
    layers_leaving,
    mask_target,
    masks_for,
    move_stack,
    new_id,
    slot_above_block,
)


class LayersMixin:
    """Add, delete, reorder, merge, and move layers between pages."""

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

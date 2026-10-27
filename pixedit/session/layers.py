"""Layer stack, pages, groups, merge, and placing files.

Mixed into ``Session``. A page is one canvas. A group is a tag whose layers
are drawn on every page that lists it. Masks stay with the picture they
filter. Every change goes through ``commit`` so it is one undo step.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image

from pixedit.composite import baked_placement, composite, merge_pair
from pixedit.factory import blank_layer, empty_raster_layer, raster_layer
from pixedit.geom import alpha_bbox, tighten_raster_layer
from pixedit.model import (
    CompoundCommand,
    DeleteGroupCommand,
    DeleteLayersCommand,
    DeleteSharedGroupCommand,
    DuplicateLayerCommand,
    InsertGroupCommand,
    InsertLayerCommand,
    InsertSharedGroupCommand,
    Layer,
    LayerGroup,
    MoveLayersToGroupCommand,
    RenameGroupCommand,
    ReorderGroupsCommand,
    ReorderSharedGroupsCommand,
    ReorderStackCommand,
    ReplaceDocumentContent,
    SetLayerAttrCommand,
    SetPageTagsCommand,
    SharedGroup,
    TransferLayersCommand,
    layers_leaving,
    mask_target,
    masks_for,
    move_stack,
    new_id,
    slot_above_block,
)

# A page that was never renamed. "Page 1" and the older "Group 1" count.
_AUTOMATIC_GROUP_NAME = re.compile(r"^(?:Page|Group) \d+$")


def apply_pix_group_names(groups: list[LayerGroup], path: str) -> None:
    """Name automatic pages from a ``.pix`` file. A typed name stays.

    One automatic page takes the file stem, so holiday.pix shows holiday.
    Several take holiday 1, holiday 2, in page order. A text that is already
    used moves to the next free number.
    """
    stem = Path(path).stem.strip()
    if not stem:
        return
    targets = [group for group in groups if _AUTOMATIC_GROUP_NAME.match(group.name)]
    if not targets:
        return
    taken = {group.name for group in groups if group not in targets}
    # One page is the bare file name. More than one page is numbered from 1
    # so the names stay distinct and follow the page order.
    labels = [stem] if len(targets) == 1 else [f"{stem} {index}" for index in range(1, len(targets) + 1)]
    for group, label in zip(targets, labels):
        group.name = _free_group_name(label, taken)
        taken.add(group.name)


def _free_group_name(label: str, taken: set[str]) -> str:
    """``label``, or the same words with a higher trailing number."""
    if label not in taken:
        return label
    match = re.match(r"^(.*?)(?: (\d+))?$", label)
    base = match.group(1) if match else label
    number = int(match.group(2)) + 1 if match and match.group(2) else 2
    while f"{base} {number}" in taken:
        number += 1
    return f"{base} {number}"


class LayersMixin:
    """Add, delete, reorder, merge, and move layers between pages."""

    def add_layer(self, layer: Layer, index: int | None = None, name: str = "Add layer") -> None:
        if not self.require_page():
            return
        shared_id = self._editing_shared_id()
        stack = self._stack()
        if index is None:
            index = len(stack) if self.layer_index < 0 else self.layer_index + 1
            # A new picture lands above a mask run, not between a picture and its masks.
            index = slot_above_block(stack, index)
        self.commit(InsertLayerCommand(self.document, layer, index, name=name, shared_id=shared_id))
        self.layer_index = max(0, min(index, len(self._stack()) - 1))
        self.notify("change")

    def add_blank_layer(self) -> None:
        """Add an empty content-sized layer. The first stroke grows its bitmap."""
        if not self.require_page():
            return
        number = len(self._stack()) + 1
        layer = empty_raster_layer(f"Layer {number}")
        self.add_layer(layer, name="New layer")

    def duplicate_active(self) -> None:
        if not self.require_page():
            return
        if self.layer_index < 0:
            return
        command = DuplicateLayerCommand(self.document, self.layer_index, shared_id=self._editing_shared_id())
        self.commit(command)
        if command.copy is None:
            return
        for index, layer in enumerate(self._stack()):
            if layer is command.copy:
                self.layer_index = index
                return

    def delete_active(self) -> None:
        """Delete the active layer. A picture takes its masks with it.

        Deleting a mask leaves the picture. Undo puts every removed layer
        back in its old slot.
        """
        if not self.require_page():
            return
        stack = self._stack()
        if self.layer_index < 0 or not stack:
            return
        layer = stack[self.layer_index]
        if layer.kind in ("raster", "text"):
            victims = [layer, *masks_for(stack, layer)]
        else:
            victims = [layer]
        self.commit(DeleteLayersCommand(self.document, victims, shared_id=self._editing_shared_id()))
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
        if not self.require_page():
            return
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
        stack = self._stack()
        count = len(stack)
        if not (0 <= old_index < count and 0 <= new_index < count):
            return
        if old_index == new_index:
            return
        reordered, landed = move_stack(stack, old_index, new_index)
        if [layer.id for layer in reordered] == [layer.id for layer in stack]:
            return
        self.commit(ReorderStackCommand(self.document, reordered, shared_id=self._editing_shared_id()))
        self.layer_index = landed

    def _next_group_name(self) -> str:
        """The next ``Page N``. N starts at the count of pages plus one.

        A number whose name is already used is skipped, so two pages never share it.
        """
        names = {group.name for group in self.document.groups}
        number = len(self.document.groups) + 1
        while f"Page {number}" in names:
            number += 1
        return f"Page {number}"

    def _next_shared_name(self) -> str:
        """The next ``Group N`` that is not already a group's name."""
        names = {group.name for group in self.document.shared_groups}
        number = len(self.document.shared_groups) + 1
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
        self._shared_id = None
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
        if not self.require_page():
            return
        # The new page is what you edit next, not the group that was selected.
        self._shared_id = None
        current = self.document.group
        image = Image.new("RGBA", (current.width, current.height), (255, 255, 255, 255))
        group = LayerGroup(
            id=new_id(),
            name=self._next_group_name(),
            width=current.width,
            height=current.height,
            layers=[raster_layer("Background", image)],
            shared_ids=list(current.shared_ids),
        )
        index = self.document.group_index(current.id) + 1
        self.commit(InsertGroupCommand(self.document, group, index))

    def delete_active_group(self) -> None:
        """Remove the page on screen. The last page stays."""
        if not self.require_page():
            return
        if len(self.document.groups) <= 1:
            self.show_message("The last page stays. A document always has one page.")
            return
        self.commit(DeleteGroupCommand(self.document, self.document.active_group_id))

    def delete_groups(self, group_ids: list[str]) -> None:
        """Remove these pages. The last page stays.

        Ids that are not pages are ignored. The rest are removed in the order
        they appear in the document, so undo puts each page back in its old
        slot. One undo step. Removing every page is refused: a document
        always keeps one.
        """
        if not self.require_page():
            return
        wanted = set(group_ids)
        # Document order, not the order the rows were clicked. A later page
        # is still there when an earlier page's slot is recorded.
        ordered = [group.id for group in self.document.groups if group.id in wanted]
        if not ordered:
            return
        if len(self.document.groups) - len(ordered) < 1:
            self.show_message("The last page stays. A document always has one page.")
            return
        # document_level so undo is not pinned to whatever page is showing.
        # Each inner command names its page and restores the page that was up.
        command = CompoundCommand(
            [DeleteGroupCommand(self.document, group_id) for group_id in ordered],
            "Delete page" if len(ordered) == 1 else "Delete pages",
        )
        command.document_level = True
        self.commit(command)

    def rename_group(self, group_id: str, name: str) -> None:
        group = self.document.group_by_id(group_id)
        if group is None:
            return
        cleaned = name.strip() or group.name
        if cleaned == group.name:
            return
        command = RenameGroupCommand(group, cleaned)
        command.name = "Rename page"
        self.commit(command)

    def reorder_groups(self, ordered_ids: list[str]) -> None:
        """Put the pages in this order. That order is the PDF page order."""
        current = [group.id for group in self.document.groups]
        if ordered_ids == current:
            return
        if set(ordered_ids) != set(current) or len(ordered_ids) != len(current):
            return
        self.commit(ReorderGroupsCommand(self.document, ordered_ids))

    def add_shared_group(self) -> None:
        """Add an empty group and tag the page on screen with it."""
        if not self.require_page():
            return
        group = SharedGroup(id=new_id(), name=self._next_shared_name(), layers=[])
        index = len(self.document.shared_groups)
        self.commit(InsertSharedGroupCommand(self.document, group, index, self.document.active_group_id))
        self.focus_shared(group.id)
        self.notify("change")

    def delete_shared_groups(self, group_ids: list[str]) -> None:
        """Remove these groups and their tags. Pages stay.

        Unknown ids are ignored. One undo step.
        """
        if not self.require_page():
            return
        wanted = set(group_ids)
        ordered = [group.id for group in self.document.shared_groups if group.id in wanted]
        if not ordered:
            return
        command = CompoundCommand(
            [DeleteSharedGroupCommand(self.document, group_id) for group_id in ordered],
            "Delete group" if len(ordered) == 1 else "Delete groups",
        )
        command.document_level = True
        if self._shared_id in wanted:
            self._shared_id = None
        self.commit(command)

    def rename_shared_group(self, group_id: str, name: str) -> None:
        group = self.document.shared_by_id(group_id)
        if group is None:
            return
        cleaned = name.strip() or group.name
        if cleaned == group.name:
            return
        command = RenameGroupCommand(group, cleaned)
        command.name = "Rename group"
        self.commit(command)

    def reorder_shared_groups(self, ordered_ids: list[str]) -> None:
        """Put the groups in this catalog order. Paint order is per page."""
        current = [group.id for group in self.document.shared_groups]
        if ordered_ids == current:
            return
        if set(ordered_ids) != set(current) or len(ordered_ids) != len(current):
            return
        self.commit(ReorderSharedGroupsCommand(self.document, ordered_ids))

    def toggle_page_tags(self, shared_id: str, page_ids: list[str]) -> None:
        """Add this group to each page, or remove it when every page already has it.

        A mixed selection gains the tag. Tag order stays put except for a new
        tag, which is painted on top. One undo step.
        """
        if self.document.shared_by_id(shared_id) is None:
            return
        pages = []
        for page_id in page_ids:
            page = self.document.group_by_id(page_id)
            if page is not None and page not in pages:
                pages.append(page)
        if not pages:
            return
        remove = all(shared_id in page.shared_ids for page in pages)
        updates = []
        for page in pages:
            if remove:
                tags = [tag for tag in page.shared_ids if tag != shared_id]
            elif shared_id in page.shared_ids:
                continue
            else:
                tags = [*page.shared_ids, shared_id]
            updates.append((page.id, tags))
        if not updates:
            return
        self.commit(SetPageTagsCommand(self.document, updates))

    def transfer_layers(
        self,
        layer_ids: list[str],
        source_kind: str,
        source_id: str,
        dest_kind: str,
        dest_id: str,
        dest_index: int,
    ) -> bool:
        """Move pictures between a page and a group. A mask follows its picture."""
        source = self.document.group_by_id(source_id) if source_kind == "page" else self.document.shared_by_id(source_id)
        dest = self.document.group_by_id(dest_id) if dest_kind == "page" else self.document.shared_by_id(dest_id)
        if source is None or dest is None or source is dest or not layer_ids:
            return False
        leaving = layers_leaving(source.layers, layer_ids)
        if not leaving or not set(layer_ids) <= {layer.id for layer in leaving}:
            return False
        self.commit(
            TransferLayersCommand(
                self.document,
                source_kind,
                source_id,
                dest_kind,
                dest_id,
                [layer.id for layer in leaving],
                dest_index,
            )
        )
        if dest_kind == "shared":
            anchor = next((layer for layer in leaving if layer.kind != "redaction"), leaving[0])
            self.focus_shared(dest_id, [anchor.id], anchor.id)
        else:
            self._shared_id = None
            anchor = next((layer for layer in leaving if layer.kind != "redaction"), leaving[0])
            try:
                self.layer_index = self.document.index_of(anchor.id)
            except KeyError:
                self.clamp_index()
        self.notify("change")
        return True

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
        in the stack.         A picture bakes every mask that filters it, then merges
        that result into the layer below.
        """
        if not self.require_page():
            return
        layer = self.active_layer()
        if layer is None:
            self.show_message("Nothing below this layer to merge into.")
            return
        if layer.kind == "redaction":
            self._bake_masks_through(layer, "Merge down")
            return
        stack = self._stack()
        try:
            index = next(i for i, item in enumerate(stack) if item.id == layer.id)
        except StopIteration:
            self.show_message("Nothing below this layer to merge into.")
            return
        if index <= 0:
            self.show_message("Nothing below this layer to merge into.")
            return
        self._merge_content_down(layer)

    def apply_active_mask(self) -> None:
        """Bake the active mask into its picture and remove the masks up through it.

        Masks of the same picture that sit above the active one stay live.
        Text becomes a normal raster layer. One undo step.
        """
        if not self.require_page():
            return
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
        for layer in self._stack():
            if layer is target:
                result.append(baked)
            elif id(layer) in skip:
                continue
            else:
                result.append(layer)
        return result

    def _commit_stack(self, layers: list[Layer], name: str, select_id: str) -> None:
        shared_id = self._editing_shared_id()
        if shared_id:
            command = ReorderStackCommand(self.document, layers, shared_id=shared_id)
            command.name = name
            self.commit(command)
            self.layer_index = self._index_in_stack(select_id)
            return
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
        stack = self._stack()
        target = mask_target(stack, mask)
        if target is None:
            self.show_message("This mask has no layer to apply to.")
            return
        mask_index = next(index for index, item in enumerate(stack) if item.id == mask.id)
        through = [
            item
            for item in masks_for(stack, target)
            if next(index for index, candidate in enumerate(stack) if candidate.id == item.id) <= mask_index
        ]
        if not through:
            return
        self._commit_stack(self._stack_with_bake(target, through), name, target.id)

    def _merge_content_down(self, layer: Layer) -> None:
        """Bake ``layer``'s masks, then merge that picture into the layer below."""
        stack = self._stack()
        masks = masks_for(stack, layer)
        working = self._stack_with_bake(layer, masks) if masks else list(stack)
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
        if not self.require_page():
            return
        image = composite(self.document)
        layer = blank_layer("Flattened", self.document.width, self.document.height)
        layer.image = image
        self._shared_id = None
        self.commit(
            ReplaceDocumentContent(
                self.document,
                [layer],
                self.document.width,
                self.document.height,
                None,
                name="Flatten",
                shared_ids=[],
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
        if not self.require_page():
            return
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
        if not self.require_page():
            return
        # Place lands on the page, even when a shared layer was selected.
        self._shared_id = None
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

    def place_groups(self, groups: list[LayerGroup], shared_groups: list[SharedGroup] | None = None) -> None:
        """Append each page. Shared groups those pages use are copied too.

        Used when placing a PDF or a project. The new pages are added in the
        order given, and the first of them is shown. Undo removes every page
        and group this call added.
        """
        if not self.require_page():
            return
        copies = [group.clone() for group in groups]
        if not copies:
            self.show_message("That file has no pages to place.")
            return
        id_map: dict[str, str] = {}
        brought: list[SharedGroup] = []
        for shared in shared_groups or []:
            if any(shared.id in page.shared_ids for page in copies):
                cloned = shared.clone()
                id_map[shared.id] = cloned.id
                brought.append(cloned)
        for page in copies:
            page.shared_ids = [id_map[tag] for tag in page.shared_ids if tag in id_map]
        start = len(self.document.groups)
        commands: list[Any] = [
            InsertGroupCommand(self.document, group, start + offset) for offset, group in enumerate(copies)
        ]
        shared_start = len(self.document.shared_groups)
        for offset, shared in enumerate(brought):
            commands.append(InsertSharedGroupCommand(self.document, shared, shared_start + offset, None))
        command = CompoundCommand(commands, "Place pages")
        command.document_level = True
        self.commit(command)
        first = copies[0]
        if self.document.active_group_id != first.id:
            self.activate_group(first.id)
        count = len(copies)
        noun = "page" if count == 1 else "pages"
        self.show_message(f"Placed {count} {noun}.")

    def paste_image(self, image: Image.Image) -> None:
        """Add ``image`` as a new layer on the page on screen, centered.

        The page size does not change. A picture larger than the canvas hangs
        off the edges, and Transform can move it. Transparent margins are
        cropped first, so the layer is only as big as its opaque pixels.
        One undo step. With no page open, stop before reading the canvas size.
        """
        if not self.require_page():
            return
        self._shared_id = None
        # Copy so later edits to the caller's image cannot change the layer.
        rgba = image.convert("RGBA").copy()
        if alpha_bbox(rgba) is None:
            self.show_message("The clipboard image has no opaque pixels.")
            return
        layer = raster_layer("Pasted", rgba)
        tighten_raster_layer(layer)
        layer.x = int(round((self.document.width - layer.image.width) / 2))
        layer.y = int(round((self.document.height - layer.image.height) / 2))
        self.add_layer(layer, name="Paste")
        self.show_message("Pasted the clipboard image as a new layer.")

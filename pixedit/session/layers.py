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

from pixedit.composite import baked_placement, composite, merge_layers, merge_pair
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
    contiguous_mask_span,
    duplicate_slot,
    layers_leaving,
    mask_target,
    masks_for,
    move_stack,
    new_id,
    pack_layer_block,
    slot_above_block,
)

# A page that was never renamed. "Page 1" and the older "Group 1" count.
_AUTOMATIC_GROUP_NAME = re.compile(r"^(?:Page|Group) \d+$")


def _victims(stack: list[Layer], chosen: list[Layer]) -> list[Layer]:
    """Layers a cut or delete removes, in stack order.

    A selected picture or text layer takes its masks. A selected mask is only
    itself. A mask already covered by its picture is listed once.
    """
    ids: set[str] = set()
    for layer in chosen:
        ids.add(layer.id)
        if layer.kind in ("raster", "text"):
            for mask in masks_for(stack, layer):
                ids.add(mask.id)
    return [layer for layer in stack if layer.id in ids]


def _mask_picture_missing(placed: list[Layer], stack: list[Layer]) -> bool:
    """True when a pasted mask would not attach to a picture on this page.

    A mask whose picture is in the pasted block points at that copy. Any other
    mask must still find its picture in the stack.
    """
    present = {layer.id for layer in placed}
    present.update(layer.id for layer in stack if layer.kind in ("raster", "text"))
    for layer in placed:
        if layer.kind == "redaction" and layer.target_id not in present:
            return True
    return False


def _clone_layer_block(layers: list[Layer]) -> list[Layer]:
    """Fresh copies of a cut or paste block.

    A mask in the block points at the copied picture. A mask whose picture
    is not in the block keeps the id it had, so it still filters that picture.
    """
    copies = [layer.clone() for layer in layers]
    id_map = {original.id: copy.id for original, copy in zip(layers, copies)}
    for layer in copies:
        if layer.target_id in id_map:
            layer.target_id = id_map[layer.target_id]
    return copies


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

    def copy_active_layer(self) -> bool:
        """Remember the selected layers so a later paste can duplicate them.

        One selected layer copies that layer. Two or more copy every selected
        layer, in stack order. The snapshot is taken now. Painting an original
        afterwards does not change what paste inserts. This is not an undo
        step. False means the status line already says why nothing was copied.
        """
        if not self.require_page():
            return False
        chosen = self.selected_layers()
        if len(chosen) > 1:
            self._copied_layers = _clone_layer_block(chosen)
            self._copied_layer = None
            self._copied_cut = False
            self._copied_multi = True
            self._copied_source_id = chosen[-1].id
            self.show_message(f"Copied {len(chosen)} layers. Ctrl+V pastes them.")
            return True
        layer = self.active_layer()
        if layer is None:
            self.show_message("Choose a layer to copy.")
            return False
        self._copied_layer = layer.clone()
        self._copied_layers = None
        self._copied_cut = False
        self._copied_multi = False
        self._copied_source_id = layer.id
        self.show_message(f'Copied "{layer.name}". Ctrl+V pastes a duplicate.')
        return True

    def cut_active_layer(self) -> bool:
        """Remove the selected layers and remember them for Ctrl+V.

        One selected layer is cut, and a picture takes its masks. Two or more
        cut every selected layer the same way: each picture takes its masks,
        and a mask is not removed twice. The snapshot is taken first, then one
        undo step removes them. A later paste keeps the names. False means the
        status line already says why nothing was cut.
        """
        if not self.require_page():
            return False
        stack = self._stack()
        if self.layer_index < 0 or not stack:
            self.show_message("Choose a layer to cut.")
            return False
        chosen = self.selected_layers()
        if len(chosen) > 1:
            victims = _victims(stack, chosen)
            self._copied_layers = _clone_layer_block(victims)
            self._copied_cut = True
            self._copied_multi = True
            self._copied_layer = None
            self._copied_source_id = chosen[-1].id
            self._delete_victims(victims, name="Cut")
            self.show_message(f"Cut {len(chosen)} layers. Ctrl+V pastes them.")
            return True
        layer = stack[self.layer_index]
        if layer.kind in ("raster", "text"):
            victims = [layer, *masks_for(stack, layer)]
        else:
            victims = [layer]
        self._copied_layers = _clone_layer_block(victims)
        self._copied_cut = True
        self._copied_multi = False
        self._copied_layer = None
        self._copied_source_id = layer.id
        self.delete_active()
        self.show_message(f'Cut "{layer.name}". Ctrl+V pastes it.')
        return True

    def paste_copied_layer(self) -> bool:
        """Insert a duplicate of the layer stored by ``copy_active_layer``.

        Flow:
        1. Stop when no page is open, or when copy has not stored a layer.
        2. Clone that snapshot. A second paste clones it again, so the copies
           do not share pixels.
        3. A picture or text layer is inserted above the active layer. The
           index moves above a mask run so the new layer does not split a
           picture from its masks.
        4. A mask is inserted only when the picture it filters is on this
           page. The original mask still on the page puts the copy directly
           above it, which is the same slot Duplicate uses. Otherwise the
           copy goes at the top of that picture's mask run. A picture that
           lives on another page is refused, so the mask cannot attach to a
           different picture.

        One undo step. False means nothing was inserted.
        """
        if not self.require_page():
            return False
        if self._copied_cut:
            return self._paste_cut_block()
        if self._copied_multi and self._copied_layers:
            return self._paste_multi_copy()
        source = self._copied_layer
        if source is None:
            self.show_message("Copy a layer first. Ctrl+C copies the active layer.")
            return False
        placed = source.clone()
        placed.name = f"{source.name} copy"
        if placed.kind == "redaction":
            index = self._copied_mask_index(placed)
            if index is None:
                self.show_message("That mask's picture is not on this page.")
                return False
        else:
            index = 0 if self.layer_index < 0 else self.layer_index + 1
            index = slot_above_block(self.document.layers, index)
        self.add_layer(placed, index, name="Paste")
        self.show_message(f'Pasted a copy of "{source.name}".')
        return True

    def _paste_cut_block(self) -> bool:
        """Insert the layers stored by ``cut_active_layer``, keeping their names.

        A lone mask goes back onto its picture. A picture and its masks are
        inserted together, above the active layer, as one undo step. Each
        paste clones the snapshot again.
        """
        stored = self._copied_layers
        if not stored:
            self.show_message("Cut a layer first. Ctrl+X cuts the active layer.")
            return False
        placed = _clone_layer_block(stored)
        stack = self._stack()
        if len(placed) == 1 and placed[0].kind == "redaction":
            index = self._mask_slot(placed[0], stack)
            if index is None:
                self.show_message("That mask's picture is not on this page.")
                return False
            self.add_layer(placed[0], index, name="Paste")
            self.show_message(f'Pasted "{placed[0].name}".')
            return True
        index = self._paste_index(stack)
        self._commit_inserts(placed, index)
        if self._copied_multi:
            self._select_block(placed)
            self.show_message(f"Pasted {len(placed)} layers.")
            return True
        anchor = next((layer for layer in placed if layer.kind != "redaction"), placed[0])
        for slot, layer in enumerate(self._stack()):
            if layer is anchor:
                self.layer_index = slot
                break
        self.notify("change")
        self.show_message(f'Pasted "{anchor.name}".')
        return True

    def _paste_multi_copy(self) -> bool:
        """Insert a duplicate of every layer stored by a multi-layer copy.

        Names gain " copy". The block sits just in front of the front-most
        selected layer. One undo step. The pasted layers become the selection.
        """
        stored = self._copied_layers
        if not stored:
            self.show_message("Copy a layer first. Ctrl+C copies the selected layers.")
            return False
        placed = _clone_layer_block(stored)
        for layer in placed:
            layer.name = f"{layer.name} copy"
        stack = self._stack()
        if _mask_picture_missing(placed, stack):
            self.show_message("That mask's picture is not on this page.")
            return False
        self._commit_inserts(placed, self._paste_index(stack))
        self._select_block(placed)
        self.show_message(f"Pasted {len(placed)} layers.")
        return True

    def _paste_index(self, stack: list[Layer]) -> int:
        """Insert slot just in front of the front-most selected layer.

        With nothing selected, the slot is above the active layer. Either way
        it moves above a mask run so the paste does not split a picture from
        its masks.
        """
        selected = self.selected_layers()
        if selected:
            index = stack.index(selected[-1]) + 1
        else:
            index = 0 if self.layer_index < 0 else self.layer_index + 1
        return slot_above_block(stack, index)

    def _commit_inserts(self, placed: list[Layer], index: int) -> None:
        """Insert ``placed`` as one undo step, keeping their order."""
        shared_id = self._editing_shared_id()
        commands = [
            InsertLayerCommand(self.document, layer, index + offset, name="Paste", shared_id=shared_id)
            for offset, layer in enumerate(placed)
        ]
        if len(commands) == 1:
            self.commit(commands[0])
            return
        self.commit(CompoundCommand(commands, "Paste"))

    def _select_block(self, placed: list[Layer]) -> None:
        """Select the pasted layers. The front one is active."""
        if not placed:
            return
        shared_id = self._editing_shared_id()
        ids = [layer.id for layer in placed]
        front = placed[-1].id
        if shared_id:
            self.focus_shared(shared_id, ids, front)
        else:
            self.select_layers(ids, active_id=front)
        self._remember_group_ui()
        self.notify("change")

    def _delete_victims(self, victims: list[Layer], *, name: str) -> None:
        """Remove ``victims`` as one undo step and keep the selection valid."""
        if not victims:
            return
        command = DeleteLayersCommand(self.document, victims, shared_id=self._editing_shared_id())
        command.name = name
        self.commit(command)
        self.clamp_index()
        self._remember_group_ui()

    def _mask_slot(self, copied: Layer, layers: list[Layer]) -> int | None:
        """Where a pasted mask sits on ``layers``, or None when its picture is absent."""
        target = None
        if copied.target_id:
            for layer in layers:
                if layer.id == copied.target_id and layer.kind in ("raster", "text"):
                    target = layer
                    break
        if target is None:
            return None
        span = contiguous_mask_span(layers, target)
        if span is None:
            return None
        return span[1] + 1

    def _copied_mask_index(self, copied: Layer) -> int | None:
        """Insert slot for a copied mask, or None when its picture is absent.

        ``copied.target_id`` names the picture. The id is checked on the page
        on screen only, because each page has its own stack.
        """
        layers = self.document.layers
        target = None
        if copied.target_id:
            for layer in layers:
                if layer.id == copied.target_id and layer.kind in ("raster", "text"):
                    target = layer
                    break
        if target is None:
            return None
        source_id = self._copied_source_id
        for index, layer in enumerate(layers):
            if layer.id == source_id and layer.kind == "redaction":
                return duplicate_slot(layers, index)
        span = contiguous_mask_span(layers, target)
        if span is None:
            return None
        return span[1] + 1

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

    def delete_selected(self) -> None:
        """Delete every selected layer. One layer deletes the active layer.

        Two or more use the same set as Cut: each picture takes its masks, and
        a mask is not removed twice. One undo step. A highlighted page or group
        is not handled here.
        """
        if not self.require_page():
            return
        chosen = self.selected_layers()
        if len(chosen) <= 1:
            self.delete_active()
            return
        self._delete_victims(_victims(self._stack(), chosen), name="Delete layers")

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

    def move_layer_block(
        self,
        requested_ids: list[str],
        ordered_ids: list[str],
        active_id: str | None = None,
    ) -> None:
        """Move these layers as one block inside the current page or tag.

        ``ordered_ids`` is the back-to-front stack with ``requested_ids``
        already gathered into one block. A picture takes its masks. One undo
        step. The requested layers stay selected, and ``active_id`` stays the
        layer paint, opacity, and blend follow.
        """
        if not self.require_page():
            return
        stack = self._stack()
        packed = pack_layer_block(stack, requested_ids, ordered_ids)
        if packed is None:
            return
        if [layer.id for layer in packed] == [layer.id for layer in stack]:
            return
        shared_id = self._editing_shared_id()
        command = ReorderStackCommand(self.document, packed, shared_id=shared_id)
        command.name = "Reorder layers"
        self.commit(command)
        wanted = [layer.id for layer in packed if layer.id in set(requested_ids)]
        if not wanted:
            return
        if shared_id:
            self.focus_shared(shared_id, wanted, active_id)
        else:
            self.select_layers(wanted, active_id)
        self.notify("change")

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
        """The next ``Tag N`` that is not already a tag's name."""
        names = {group.name for group in self.document.shared_groups}
        number = len(self.document.shared_groups) + 1
        while f"Tag {number}" in names:
            number += 1
        return f"Tag {number}"

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
        """Add a blank page the same pixel size as the one on screen and show it.

        The page starts with no tags. Tag it afterward if it should share a header or footer.
        """
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
            "Delete tag" if len(ordered) == 1 else "Delete tags",
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
        command.name = "Rename tag"
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

    def _select_transferred(
        self,
        leaving: list[Layer],
        layer_ids: list[str],
        active_id: str | None,
        dest_kind: str,
        dest_id: str,
    ) -> None:
        """Keep the dragged layers selected on the page or tag they landed on."""
        requested = set(layer_ids)
        chosen = [layer.id for layer in leaving if layer.id in requested]
        if not chosen:
            anchor = next((layer for layer in leaving if layer.kind != "redaction"), leaving[0])
            chosen = [anchor.id]
        active = active_id if active_id in set(chosen) else None
        if dest_kind == "shared":
            self.focus_shared(dest_id, chosen, active)
            return
        self._shared_id = None
        self.select_layers(chosen, active)

    def transfer_layers(
        self,
        layer_ids: list[str],
        source_kind: str,
        source_id: str,
        dest_kind: str,
        dest_id: str,
        dest_index: int,
        active_id: str | None = None,
    ) -> bool:
        """Move pictures between a page and a group. A mask follows its picture.

        The layers named in ``layer_ids`` stay selected. ``active_id`` is the
        one that was dragged, when it is one of them.
        """
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
        self._select_transferred(leaving, layer_ids, active_id, dest_kind, dest_id)
        self.notify("change")
        return True

    def move_layers_to_group(
        self,
        layer_ids: list[str],
        dest_group_id: str,
        dest_index: int,
        active_id: str | None = None,
    ) -> bool:
        """Move pictures onto another page. A mask follows its picture.

        Returns false when nothing eligible moved, so the layer list can snap
        back. The destination page is shown and the dragged layers stay selected.
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
        self._select_transferred(leaving, layer_ids, active_id, "page", dest.id)
        self.notify("change")
        return True

    def merge_down_active(self) -> None:
        """Bake the selection downward, as one undo step.

        One selected layer keeps the old meaning. A mask bakes into the
        picture it belongs to. A picture bakes every mask that filters it,
        then merges into the layer below.

        Two or more selected layers merge with each other, back to front.
        A layer that is not selected stays, even when it sat between them.
        A selected mask joins that merge only when its picture is selected
        too. A selection that is only masks asks for the layers instead.
        """
        if not self.require_page():
            return
        selected = self.selected_layers()
        if len(selected) >= 2:
            self._merge_selected(selected)
            return
        self._merge_down_one()

    def _merge_down_one(self) -> None:
        """Bake the active layer into the layer below, or a mask into its picture."""
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

    def _merge_selected(self, selected: list) -> None:
        """Merge the selected layers into one at the front selected layer's place.

        Pictures and text in the selection are the layers that merge. A mask
        is included when its picture is one of those. Anything else stays in
        the stack. The plate keeps the front selected layer's name and id, so
        a mask that still points at that layer keeps filtering the result.
        """
        stack = self._stack()
        pictures = [layer for layer in selected if layer.kind != "redaction"]
        if not pictures:
            self.show_message("Select the layers to merge.")
            return
        picture_ids = {layer.id for layer in pictures}
        merging_ids = set(picture_ids)
        for layer in selected:
            if layer.kind != "redaction":
                continue
            target = mask_target(stack, layer)
            if target is not None and target.id in picture_ids:
                merging_ids.add(layer.id)
        # One picture plus masks of some other picture is not a merge of
        # those rows. Leave the neighbor below it alone.
        if len(merging_ids) < 2:
            self.show_message("Select the layers to merge.")
            return
        merging = [layer for layer in stack if layer.id in merging_ids]
        front = merging[-1]
        merged = merge_layers(merging)
        merged.name = front.name
        merged.id = front.id
        merged.visible = True
        removed = merging_ids - {front.id}
        layers = []
        for layer in stack:
            if layer.id == front.id:
                layers.append(merged)
                continue
            if layer.id in removed:
                continue
            # A mask of a picture that was merged away follows the plate.
            # Clone it so undo still has the mask that pointed at the old picture.
            if layer.kind == "redaction" and layer.target_id in removed:
                carried = layer.clone()
                carried.id = layer.id
                carried.target_id = merged.id
                layers.append(carried)
                continue
            layers.append(layer)
        self._commit_stack(layers, "Merge layers", merged.id)

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
        command = self._visibility_command(layer, visible)
        command.name = "Toggle layer"
        self.commit(command)

    def set_selected_visibility(self, visible: bool) -> None:
        """Show or hide every selected layer as one undo step.

        Layers already in that state are left out of the step. Layers outside
        the selection stay put. A checkbox on one selected row uses this so
        the whole selection follows that check.
        """
        if not self.require_page():
            return
        commands = [
            self._visibility_command(layer, visible)
            for layer in self.selected_layers()
            if layer.visible != visible
        ]
        if not commands:
            return
        if len(commands) == 1:
            self.commit(commands[0])
            return
        self.commit(CompoundCommand(commands, "Toggle visibility"))

    def _visibility_command(self, layer: Layer, visible: bool) -> SetLayerAttrCommand:
        command = SetLayerAttrCommand(layer, "visible", visible, name="Toggle visibility")
        for group in self.document.groups:
            if any(item is layer for item in group.layers):
                command.group_id = group.id
                break
        return command

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

        Used when placing a PDF, a project, or a multi-frame image. The new
        pages are added in the order given, and the first of them is shown.
        Undo removes every page and shared group this call added.
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

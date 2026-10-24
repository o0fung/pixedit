"""Document model and undoable commands.

The model has no Qt dependency. A document is an ordered list of groups.
Each group is its own canvas: a size, a selection, and an ordered layer list
(index 0 is the bottom). The document's width, layers, and selection are the
group on screen. Tools and plugins mutate the model only through ``Command``
objects so every edit can be undone.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

BLEND_MODES = ("normal", "multiply", "screen", "overlay")
LAYER_KINDS = ("raster", "text", "redaction")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class TextInfo:
    """Re-editable text. The layer image is a render of these fields."""

    text: str
    font_path: str | None
    font_family: str
    size: int
    color: tuple[int, int, int, int]
    rotation: float = 0.0  # degrees clockwise
    align: str = "left"  # left, center, right, justify
    line_spacing: int = 4


@dataclass
class RedactionInfo:
    """How a mask is applied at composite time.

    ``kind`` selects the effect. ``amount`` is the mosaic block, blur radius,
    or strength. ``color`` is the flat color used by Solid. ``seed`` keeps
    Noise from changing every time the canvas is redrawn.
    """

    kind: str = "pixelate"
    amount: float = 12.0
    color: tuple[int, int, int, int] = (0, 0, 0, 255)
    seed: int = 1


# Solid has no strength slider. The other entries are the amount control:
# label, inclusive range, and the value used when that effect is chosen fresh.
MASK_EFFECTS: dict[str, dict[str, int | str] | None] = {
    "pixelate": {"label": "Block", "min": 2, "max": 64, "default": 12},
    "blur": {"label": "Radius", "min": 1, "max": 40, "default": 8},
    "solid": None,
    "noise": {"label": "Strength", "min": 1, "max": 100, "default": 100},
    "desaturate": {"label": "Strength", "min": 0, "max": 100, "default": 100},
    "invert": {"label": "Strength", "min": 0, "max": 100, "default": 100},
    "dim": {"label": "Strength", "min": 0, "max": 100, "default": 60},
    "sharpen": {"label": "Strength", "min": 1, "max": 100, "default": 50},
}


def clamp_mask_amount(kind: str, amount: float) -> float:
    """Fit ``amount`` to the effect's slider. Solid keeps the number it had."""
    spec = MASK_EFFECTS.get(kind)
    if not spec:
        return float(amount)
    low = int(spec["min"])
    high = int(spec["max"])
    return float(min(high, max(low, int(round(amount)))))


@dataclass
class Layer:
    """One stack entry.

    Raster and text layers store straight RGBA pixels in ``image``, placed at
    ``(x, y)`` on the canvas. Paint and shapes grow that bitmap around new
    marks; they do not resize every layer to the document. A layer that
    already covers the canvas (background, import, flatten) keeps that size.
    A redaction layer stores an ``L`` mask (255 = fully covered) in document
    coordinates and does not paint its own color. ``target_id`` is the picture
    or text layer whose pixels the effect filters. Compositing mixes the
    effect through the mask onto that layer only. The layer kind stays
    ``redaction`` so existing projects still open.
    """

    id: str
    name: str
    kind: str = "raster"
    visible: bool = True
    opacity: float = 1.0
    blend_mode: str = "normal"
    image: Image.Image = field(default_factory=lambda: Image.new("RGBA", (1, 1), (0, 0, 0, 0)))
    x: int = 0
    y: int = 0
    text: TextInfo | None = None
    redaction: RedactionInfo | None = None
    mask: Image.Image | None = None
    # Picture or text layer this mask filters. Empty until a mask is bound.
    target_id: str | None = None

    def clone(self) -> Layer:
        return Layer(
            id=new_id(),
            name=self.name,
            kind=self.kind,
            visible=self.visible,
            opacity=self.opacity,
            blend_mode=self.blend_mode,
            image=self.image.copy(),
            x=self.x,
            y=self.y,
            text=copy.deepcopy(self.text),
            redaction=copy.deepcopy(self.redaction),
            mask=None if self.mask is None else self.mask.copy(),
            target_id=self.target_id,
        )

    def bitmap(self) -> Image.Image:
        """The image that occupies space: the mask for redactions, else pixels."""
        if self.kind == "redaction" and self.mask is not None:
            return self.mask
        return self.image

    def bounds(self) -> tuple[int, int, int, int]:
        image = self.bitmap()
        return self.x, self.y, image.width, image.height


def visual_state(layer: Layer) -> dict[str, Any]:
    """Deep snapshot of the fields a pixel edit restores on undo."""
    return {
        "image": layer.image.copy(),
        "x": layer.x,
        "y": layer.y,
        "kind": layer.kind,
        "text": copy.deepcopy(layer.text),
        "mask": None if layer.mask is None else layer.mask.copy(),
        "redaction": copy.deepcopy(layer.redaction),
        "target_id": layer.target_id,
    }


def apply_visual(layer: Layer, state: dict[str, Any]) -> None:
    layer.image = state["image"].copy()
    layer.x = int(state["x"])
    layer.y = int(state["y"])
    layer.kind = state["kind"]
    layer.text = copy.deepcopy(state["text"])
    layer.mask = None if state["mask"] is None else state["mask"].copy()
    layer.redaction = copy.deepcopy(state["redaction"])
    if "target_id" in state:
        layer.target_id = state["target_id"]


def _same_layer(layers: list[Layer], layer: Layer) -> int:
    """Index of this object. Layer equality compares pixels, so identity is used."""
    for index, item in enumerate(layers):
        if item is layer:
            return index
    raise ValueError(layer.id)


def mask_target(layers: list[Layer], mask: Layer) -> Layer | None:
    """The picture or text layer ``mask`` filters.

    A stored id wins when that layer is still in the stack. Otherwise the
    mask attaches to the nearest picture or text layer below it. A mask with
    nothing below it filters nothing.
    """
    if mask.kind != "redaction":
        return None
    if mask.target_id:
        for layer in layers:
            if layer.id == mask.target_id and layer.kind in ("raster", "text"):
                return layer
    try:
        index = _same_layer(layers, mask)
    except ValueError:
        return None
    for below in range(index - 1, -1, -1):
        if layers[below].kind in ("raster", "text"):
            return layers[below]
    return None


def bind_mask_targets(layers: list[Layer]) -> None:
    """Fill in ``target_id`` for masks that do not already name a real layer.

    Opening an older project, or building a stack by hand, leaves the id
    empty. The mask then belongs to the picture or text layer directly
    under it, which is the same layer a new mask would be drawn on.
    """
    content_ids = {layer.id for layer in layers if layer.kind in ("raster", "text")}
    for index, layer in enumerate(layers):
        if layer.kind != "redaction":
            continue
        if layer.target_id and layer.target_id in content_ids:
            continue
        layer.target_id = None
        for below in range(index - 1, -1, -1):
            if layers[below].kind in ("raster", "text"):
                layer.target_id = layers[below].id
                break


def masks_for(layers: list[Layer], target: Layer) -> list[Layer]:
    """Masks that filter ``target``, from the back of the stack toward the front."""
    owned: list[Layer] = []
    for layer in layers:
        if layer.kind != "redaction":
            continue
        owner = mask_target(layers, layer)
        if owner is not None and owner.id == target.id:
            owned.append(layer)
    return owned


def layers_leaving(layers: list[Layer], layer_ids: list[str]) -> list[Layer]:
    """Layers that may move to another group, in stack order.

    A picture takes the masks that filter it. A mask stays behind unless that
    picture is moving too, so the mask cannot land on a page that does not
    contain the picture it filters.
    """
    chosen = set(layer_ids)
    moving_ids: set[str] = set()
    for layer in layers:
        if layer.id not in chosen or layer.kind == "redaction":
            continue
        moving_ids.add(layer.id)
        for mask in masks_for(layers, layer):
            moving_ids.add(mask.id)
    return [layer for layer in layers if layer.id in moving_ids]


def contiguous_mask_span(layers: list[Layer], target: Layer) -> tuple[int, int] | None:
    """Inclusive range of ``target`` plus the masks sitting directly on it.

    Masks further up the stack, with other layers between, are not part of
    this span. Moving and inserting use the span so a new layer cannot land
    between a picture and the masks that belong to it.
    """
    try:
        start = _same_layer(layers, target)
    except ValueError:
        return None
    end = start
    while end + 1 < len(layers):
        above = layers[end + 1]
        if above.kind != "redaction":
            break
        owner = mask_target(layers, above)
        if owner is None or owner.id != target.id:
            break
        end += 1
    return start, end


def mask_block(layers: list[Layer], layer: Layer) -> list[Layer]:
    """``layer``'s picture, then every mask that filters it, in stack order.

    Passing a mask returns that mask's picture plus its sibling masks. A
    picture with no masks, or a mask with no picture, is a block of one.
    """
    if layer.kind == "redaction":
        target = mask_target(layers, layer)
        if target is None:
            return [layer]
        return [target, *masks_for(layers, target)]
    if layer.kind in ("raster", "text"):
        return [layer, *masks_for(layers, layer)]
    return [layer]


def slot_above_block(layers: list[Layer], index: int) -> int:
    """Move an insert index so it does not split a picture from its masks.

    ``index`` is a Python insert position (``0`` .. ``len``). Landing inside
    the run, or between the picture and its first mask, becomes the slot
    just above the run.
    """
    if index <= 0 or index > len(layers):
        return max(0, index)
    below = layers[index - 1]
    if below.kind in ("raster", "text"):
        target = below
    elif below.kind == "redaction":
        target = mask_target(layers, below)
        if target is None:
            return index
    else:
        return index
    span = contiguous_mask_span(layers, target)
    if span is None:
        return index
    start, end = span
    if start < index <= end:
        return end + 1
    return index


def duplicate_slot(layers: list[Layer], index: int) -> int:
    """Where a copy of ``layers[index]`` is inserted.

    A mask copy sits directly above the mask, still on the same picture.
    A picture copy sits above that picture's whole mask run, and the masks
    stay on the original.
    """
    source = layers[index]
    if source.kind == "redaction":
        return index + 1
    span = contiguous_mask_span(layers, source)
    if span is None:
        return index + 1
    return span[1] + 1


def _move_one(layers: list[Layer], old: int, new: int) -> list[Layer]:
    moved = list(layers)
    layer = moved.pop(old)
    moved.insert(new, layer)
    return moved


def move_stack(layers: list[Layer], old: int, new: int) -> tuple[list[Layer], int]:
    """Reorder one drag and return ``(stack, index of the dragged layer)``.

    Flow:
    1. A mask dropped on another mask of the same picture only swaps inside
       that run. The picture stays put.
    2. Dragging a mask past the run, or dragging the picture, slides the
       picture and every mask that filters it as one block.
    3. The block is packed as the picture, then its masks in their old order,
       so a mask cannot end up underneath its picture.

    ``new`` is the index the dragged layer would have after a one-layer move.
    """
    count = len(layers)
    if not (0 <= old < count and 0 <= new < count) or old == new:
        return list(layers), old
    subject = layers[old]
    if subject.kind == "redaction":
        target = mask_target(layers, subject)
        if target is not None:
            span = contiguous_mask_span(layers, target)
            # Both indices sit in the mask run above the picture, so this
            # drag is a reorder of sibling masks and not a move of the block.
            if span is not None:
                start, end = span
                if start < old <= end and start < new <= end:
                    moved = _move_one(layers, old, new)
                    return moved, _same_layer(moved, subject)
    block = mask_block(layers, subject)
    if len(block) == 1:
        moved = _move_one(layers, old, new)
        return moved, _same_layer(moved, subject)
    block_ids = {id(layer) for layer in block}
    rest = [layer for layer in layers if id(layer) not in block_ids]
    offset = 0
    for index, layer in enumerate(block):
        if layer is subject:
            offset = index
            break
    # ``new - offset`` is where the picture would start if the dragged layer
    # landed on ``new``. Clamp so the block stays inside the stack.
    final_start = new - offset
    limit = len(layers) - len(block)
    if final_start < 0:
        final_start = 0
    elif final_start > limit:
        final_start = limit
    moved = rest[:final_start] + list(block) + rest[final_start:]
    return moved, _same_layer(moved, subject)


@dataclass
class LayerGroup:
    """One page. Its layers do not composite with any other group."""

    id: str
    name: str
    width: int
    height: int
    layers: list[Layer] = field(default_factory=list)
    # None means "no selection" (edits hit the whole target). An L image is a
    # real selection: 255 is included, 0 is excluded. The mask is this page's
    # size, not the size of any other group.
    selection: Image.Image | None = None

    def clone(self) -> LayerGroup:
        """A new group with copied layers. Mask targets point at the copies."""
        copies = [layer.clone() for layer in self.layers]
        id_map = {original.id: copy.id for original, copy in zip(self.layers, copies)}
        for layer in copies:
            if layer.target_id in id_map:
                layer.target_id = id_map[layer.target_id]
        return LayerGroup(
            id=new_id(),
            name=self.name,
            width=self.width,
            height=self.height,
            layers=copies,
            selection=None if self.selection is None else self.selection.copy(),
        )


class Document:
    """Pages that share one printed resolution.

    ``width``, ``height``, ``layers``, and ``selection`` read and write the
    active group. Existing edits keep using those names and only touch the
    page on screen. ``dpi`` belongs to the whole file: one PDF import is one
    render scale, so changing it changes every page's printed size.
    """

    def __init__(
        self,
        width: int,
        height: int,
        layers: list[Layer] | None = None,
        selection: Image.Image | None = None,
        dpi: float = 72.0,
        groups: list[LayerGroup] | None = None,
        active_group_id: str | None = None,
    ) -> None:
        stored = float(dpi)
        if stored <= 0:
            stored = 72.0
        self.dpi = stored
        if groups is not None:
            if not groups:
                raise ValueError("A document needs at least one group")
            self.groups = list(groups)
        else:
            self.groups = [
                LayerGroup(
                    id=new_id(),
                    name="Group 1",
                    width=max(1, int(width)),
                    height=max(1, int(height)),
                    layers=list(layers or []),
                    selection=selection,
                )
            ]
        self.active_group_id = active_group_id or self.groups[0].id
        if self.group_by_id(self.active_group_id) is None:
            self.active_group_id = self.groups[0].id

    def group_by_id(self, group_id: str) -> LayerGroup | None:
        for group in self.groups:
            if group.id == group_id:
                return group
        return None

    def group_index(self, group_id: str) -> int:
        for index, group in enumerate(self.groups):
            if group.id == group_id:
                return index
        raise KeyError(group_id)

    @property
    def group(self) -> LayerGroup:
        """The page on screen. A stale id falls back to the first page."""
        if not self.groups:
            raise RuntimeError("A document has no groups")
        found = self.group_by_id(self.active_group_id)
        if found is None:
            self.active_group_id = self.groups[0].id
            return self.groups[0]
        return found

    @property
    def width(self) -> int:
        return self.group.width

    @width.setter
    def width(self, value: int) -> None:
        self.group.width = int(value)

    @property
    def height(self) -> int:
        return self.group.height

    @height.setter
    def height(self, value: int) -> None:
        self.group.height = int(value)

    @property
    def layers(self) -> list[Layer]:
        """The active page's stack. In-place edits mutate that page only."""
        return self.group.layers

    @layers.setter
    def layers(self, value: list[Layer]) -> None:
        self.group.layers = list(value)

    @property
    def selection(self) -> Image.Image | None:
        return self.group.selection

    @selection.setter
    def selection(self, value: Image.Image | None) -> None:
        self.group.selection = value

    def index_of(self, layer_id: str) -> int:
        for index, layer in enumerate(self.layers):
            if layer.id == layer_id:
                return index
        raise KeyError(layer_id)

    def layer_by_id(self, layer_id: str) -> Layer:
        return self.layers[self.index_of(layer_id)]


def new_document(width: int, height: int, background: str = "white", dpi: float = 72.0) -> Document:
    """Create a document with a single base layer.

    ``background`` is ``white``, ``black``, or ``transparent``. ``dpi`` is the
    printed resolution stored with the canvas. A non-positive value becomes 72,
    which is the same default an untouched document uses.
    """
    width = max(1, int(width))
    height = max(1, int(height))
    fills = {
        "white": (255, 255, 255, 255),
        "black": (0, 0, 0, 255),
        "transparent": (0, 0, 0, 0),
    }
    if background not in fills:
        raise ValueError(f"Unknown background {background!r}")
    image = Image.new("RGBA", (width, height), fills[background])
    name = "Background" if background != "transparent" else "Layer 1"
    layer = Layer(id=new_id(), name=name, kind="raster", image=image, x=0, y=0)
    stored = float(dpi)
    if stored <= 0:
        stored = 72.0
    return Document(width=width, height=height, layers=[layer], dpi=stored)


class Command:
    """An undoable mutation. ``History.push`` calls ``execute`` once.

    ``group_id`` is the page this edit belongs to. Undo and redo show that
    page first, so an assignment to ``document.layers`` hits the page that
    was edited even if another page is on screen. Commands that create,
    delete, or reorder groups set ``document_level`` and address groups by id
    instead of borrowing the active page.
    """

    name = "Edit"
    document_level = False
    group_id: str | None = None

    def execute(self) -> None:
        raise NotImplementedError

    def undo(self) -> None:
        raise NotImplementedError


class CompoundCommand(Command):
    """Several document edits that undo and redo as a single step."""

    def __init__(self, commands: list[Command], name: str) -> None:
        self.commands = list(commands)
        self.name = name

    def execute(self) -> None:
        for command in self.commands:
            command.execute()

    def undo(self) -> None:
        for command in reversed(self.commands):
            command.undo()


class History:
    def __init__(self, limit: int = 40) -> None:
        self.limit = limit
        self.undo_stack: list[Command] = []
        self.redo_stack: list[Command] = []

    def push(self, command: Command) -> None:
        command.execute()
        self.undo_stack.append(command)
        if len(self.undo_stack) > self.limit:
            del self.undo_stack[: len(self.undo_stack) - self.limit]
        self.redo_stack.clear()

    def undo(self) -> Command | None:
        if not self.undo_stack:
            return None
        command = self.undo_stack.pop()
        command.undo()
        self.redo_stack.append(command)
        return command

    def redo(self) -> Command | None:
        if not self.redo_stack:
            return None
        command = self.redo_stack.pop()
        command.execute()
        self.undo_stack.append(command)
        return command

    def can_undo(self) -> bool:
        return bool(self.undo_stack)

    def can_redo(self) -> bool:
        return bool(self.redo_stack)


class SetLayerAttrCommand(Command):
    def __init__(self, layer: Layer, attr: str, value: Any, name: str | None = None) -> None:
        self.layer = layer
        self.attr = attr
        self.value = value
        self.name = name or f"Set {attr}"
        self._old: Any = None
        self._captured = False

    def execute(self) -> None:
        if not self._captured:
            self._old = getattr(self.layer, self.attr)
            self._captured = True
        setattr(self.layer, self.attr, self.value)

    def undo(self) -> None:
        setattr(self.layer, self.attr, self._old)


class SetPositionCommand(Command):
    """Move a layer without copying its pixels."""

    name = "Move layer"

    def __init__(
        self,
        layer: Layer,
        new_x: int,
        new_y: int,
        old_x: int | None = None,
        old_y: int | None = None,
    ) -> None:
        self.layer = layer
        self.new_x = int(new_x)
        self.new_y = int(new_y)
        self._old_x = old_x
        self._old_y = old_y
        self._captured = old_x is not None and old_y is not None

    def execute(self) -> None:
        if not self._captured:
            self._old_x = self.layer.x
            self._old_y = self.layer.y
            self._captured = True
        self.layer.x = self.new_x
        self.layer.y = self.new_y

    def undo(self) -> None:
        self.layer.x = int(self._old_x)
        self.layer.y = int(self._old_y)


class MutateLayerCommand(Command):
    """Replace a layer's pixels, placement, text, or redaction mask.

    Pass ``old_state`` when the layer was already mutated for a live preview;
    otherwise the previous pixels are captured on the first ``execute``.
    """

    def __init__(
        self,
        layer: Layer,
        new_state: dict[str, Any],
        old_state: dict[str, Any] | None = None,
        name: str = "Edit",
    ) -> None:
        self.layer = layer
        self.new_state = new_state
        self.old_state = old_state
        self.name = name
        self._captured = old_state is not None

    def execute(self) -> None:
        if not self._captured:
            self.old_state = visual_state(self.layer)
            self._captured = True
        apply_visual(self.layer, self.new_state)

    def undo(self) -> None:
        if self.old_state is None:
            return
        apply_visual(self.layer, self.old_state)


class SetRedactionSettings(Command):
    name = "Mask"

    def __init__(
        self,
        layer: Layer,
        kind: str,
        amount: float,
        color: tuple[int, int, int, int] | None = None,
    ) -> None:
        self.layer = layer
        self.kind = kind
        self.amount = float(amount)
        self.color = None if color is None else tuple(int(channel) for channel in color)
        self._old: RedactionInfo | None = None
        self._captured = False

    def execute(self) -> None:
        if not self._captured:
            self._old = copy.deepcopy(self.layer.redaction)
            self._captured = True
        # Seed stays with the layer. A new color is stored only when the
        # caller passed one (choosing Solid, or moving the swatch).
        old = self._old
        color = self.color if self.color is not None else (old.color if old is not None else (0, 0, 0, 255))
        seed = old.seed if old is not None else 1
        self.layer.redaction = RedactionInfo(self.kind, self.amount, color, seed)

    def undo(self) -> None:
        self.layer.redaction = copy.deepcopy(self._old)


class InsertLayerCommand(Command):
    def __init__(self, document: Document, layer: Layer, index: int, name: str = "Add layer") -> None:
        self.document = document
        self.layer = layer
        self.index = index
        self.name = name

    def execute(self) -> None:
        index = max(0, min(self.index, len(self.document.layers)))
        self.document.layers.insert(index, self.layer)
        self.index = index

    def undo(self) -> None:
        if self.index < len(self.document.layers) and self.document.layers[self.index] is self.layer:
            self.document.layers.pop(self.index)
            return
        self.document.layers = [layer for layer in self.document.layers if layer is not self.layer]


class DeleteLayersCommand(Command):
    """Remove these layers and put them back in their old slots.

    Slots are restored from the lowest index upward. Inserting in that order
    puts each layer back where it was, even when several were removed.
    """

    name = "Delete layer"

    def __init__(self, document: Document, victims: list[Layer]) -> None:
        self.document = document
        self.victims = list(victims)
        self.slots: list[tuple[int, Layer]] | None = None

    def execute(self) -> None:
        if self.slots is None:
            wanted = {id(layer) for layer in self.victims}
            self.slots = [
                (index, layer)
                for index, layer in enumerate(self.document.layers)
                if id(layer) in wanted
            ]
        wanted = {id(layer) for layer in self.victims}
        self.document.layers = [layer for layer in self.document.layers if id(layer) not in wanted]

    def undo(self) -> None:
        if not self.slots:
            return
        for index, layer in self.slots:
            self.document.layers.insert(index, layer)


class DuplicateLayerCommand(Command):
    name = "Duplicate layer"

    def __init__(self, document: Document, index: int) -> None:
        self.document = document
        self.index = index
        self.copy: Layer | None = None
        self.insert_at = index + 1

    def execute(self) -> None:
        if self.copy is None:
            source = self.document.layers[self.index]
            self.copy = source.clone()
            self.copy.name = f"{source.name} copy"
            # A picture copy goes above that picture's masks. A mask copy
            # stays in the same run, on the same picture.
            self.insert_at = duplicate_slot(self.document.layers, self.index)
        self.document.layers.insert(self.insert_at, self.copy)

    def undo(self) -> None:
        if self.copy is None:
            return
        self.document.layers = [layer for layer in self.document.layers if layer is not self.copy]


class ReorderStackCommand(Command):
    """Replace the layer order and restore the previous order on undo.

    A clip-block move changes several indexes at once, so it is not a single
    pop-and-insert. The layer objects themselves are not copied.
    """

    name = "Reorder layer"

    def __init__(self, document: Document, layers: list[Layer]) -> None:
        self.document = document
        self.new_layers = list(layers)
        self.old_layers: list[Layer] | None = None

    def execute(self) -> None:
        if self.old_layers is None:
            self.old_layers = list(self.document.layers)
        self.document.layers = list(self.new_layers)

    def undo(self) -> None:
        if self.old_layers is None:
            return
        self.document.layers = list(self.old_layers)


class SetSelectionCommand(Command):
    def __init__(
        self,
        document: Document,
        selection: Image.Image | None,
        name: str = "Selection",
    ) -> None:
        self.document = document
        self.new = None if selection is None else selection.copy()
        self.name = name
        self.old: Image.Image | None = None
        self._captured = False

    def execute(self) -> None:
        if not self._captured:
            self.old = None if self.document.selection is None else self.document.selection.copy()
            self._captured = True
        self.document.selection = None if self.new is None else self.new.copy()

    def undo(self) -> None:
        self.document.selection = None if self.old is None else self.old.copy()


def _copy_selection(selection: Image.Image | None) -> Image.Image | None:
    return None if selection is None else selection.copy()


class ReplaceDocumentContent(Command):
    """Swap the layer list, canvas size, and selection as one undo step.

    Used by crop, trim, canvas rotate/flip, and flatten. The replacement
    layer objects become the live layers; the previous objects are kept for
    undo and are not mutated by later commands (later edits target the new
    objects only).
    """

    def __init__(
        self,
        document: Document,
        layers: list[Layer],
        width: int,
        height: int,
        selection: Image.Image | None,
        name: str = "Edit",
        dpi: float | None = None,
    ) -> None:
        self.document = document
        self.new_layers = list(layers)
        self.new_width = int(width)
        self.new_height = int(height)
        self.new_selection = _copy_selection(selection)
        self.new_dpi = None if dpi is None else float(dpi)
        self.name = name
        self.old_layers: list[Layer] | None = None
        self.old_width = 0
        self.old_height = 0
        self.old_selection: Image.Image | None = None
        self.old_dpi: float | None = None
        self._captured = False

    def execute(self) -> None:
        if not self._captured:
            self.old_layers = list(self.document.layers)
            self.old_width = self.document.width
            self.old_height = self.document.height
            self.old_selection = _copy_selection(self.document.selection)
            self.old_dpi = float(self.document.dpi)
            self._captured = True
        self.document.layers = list(self.new_layers)
        self.document.width = self.new_width
        self.document.height = self.new_height
        self.document.selection = _copy_selection(self.new_selection)
        # Crop and rotate keep the resolution. Image size passes a new one,
        # and undo puts the previous resolution back with the pixels.
        if self.new_dpi is not None:
            self.document.dpi = self.new_dpi

    def undo(self) -> None:
        if self.old_layers is None:
            return
        self.document.layers = list(self.old_layers)
        self.document.width = self.old_width
        self.document.height = self.old_height
        self.document.selection = _copy_selection(self.old_selection)
        if self.new_dpi is not None and self.old_dpi is not None:
            self.document.dpi = self.old_dpi


class RenameGroupCommand(Command):
    name = "Rename group"

    def __init__(self, group: LayerGroup, name: str) -> None:
        self.group = group
        self.new_name = name
        self.old_name = group.name
        self.group_id = group.id

    def execute(self) -> None:
        self.group.name = self.new_name

    def undo(self) -> None:
        self.group.name = self.old_name


class InsertGroupCommand(Command):
    """Insert a page and show it. Undo returns to the page that was showing."""

    document_level = True
    name = "New group"

    def __init__(self, document: Document, group: LayerGroup, index: int) -> None:
        self.document = document
        self.group = group
        self.index = int(index)
        self._previous_active: str | None = None
        self._captured = False

    def execute(self) -> None:
        if not self._captured:
            self._previous_active = self.document.active_group_id
            self._captured = True
        if self.group not in self.document.groups:
            index = max(0, min(self.index, len(self.document.groups)))
            self.document.groups.insert(index, self.group)
        self.index = self.document.groups.index(self.group)
        self.document.active_group_id = self.group.id

    def undo(self) -> None:
        self.document.groups = [group for group in self.document.groups if group is not self.group]
        if self._previous_active and self.document.group_by_id(self._previous_active) is not None:
            self.document.active_group_id = self._previous_active
        elif self.document.groups:
            neighbor = min(self.index, len(self.document.groups) - 1)
            self.document.active_group_id = self.document.groups[neighbor].id


class DeleteGroupCommand(Command):
    """Remove one page. The last page cannot be deleted; the caller checks."""

    document_level = True
    name = "Delete group"

    def __init__(self, document: Document, group_id: str) -> None:
        self.document = document
        self.group_id = group_id
        self._group: LayerGroup | None = None
        self._index = 0
        self._previous_active: str | None = None
        self._captured = False

    def execute(self) -> None:
        if not self._captured:
            self._index = self.document.group_index(self.group_id)
            self._group = self.document.groups[self._index]
            self._previous_active = self.document.active_group_id
            self._captured = True
        self.document.groups = [group for group in self.document.groups if group.id != self.group_id]
        if not self.document.groups:
            raise RuntimeError("The last group cannot be deleted")
        if self.document.group_by_id(self.document.active_group_id) is None:
            neighbor = min(self._index, len(self.document.groups) - 1)
            self.document.active_group_id = self.document.groups[neighbor].id

    def undo(self) -> None:
        if self._group is None:
            return
        index = max(0, min(self._index, len(self.document.groups)))
        self.document.groups.insert(index, self._group)
        if self._previous_active and self.document.group_by_id(self._previous_active) is not None:
            self.document.active_group_id = self._previous_active
        else:
            self.document.active_group_id = self._group.id


class ReorderGroupsCommand(Command):
    """Page order is the PDF export order. The page on screen stays on screen."""

    document_level = True
    name = "Reorder groups"

    def __init__(self, document: Document, ordered_ids: list[str]) -> None:
        self.document = document
        self.new_ids = list(ordered_ids)
        self.old_ids: list[str] | None = None

    def execute(self) -> None:
        if self.old_ids is None:
            self.old_ids = [group.id for group in self.document.groups]
        self._apply(self.new_ids)

    def undo(self) -> None:
        if self.old_ids is None:
            return
        self._apply(self.old_ids)

    def _apply(self, ordered_ids: list[str]) -> None:
        by_id = {group.id: group for group in self.document.groups}
        if set(ordered_ids) != set(by_id) or len(ordered_ids) != len(by_id):
            return
        self.document.groups = [by_id[group_id] for group_id in ordered_ids]


class MoveLayersToGroupCommand(Command):
    """Move a picture, and the masks that filter it, onto another page.

    The destination page is shown. Undo puts the same layer objects back in
    their old slots and shows the page that was active before the move.
    """

    document_level = True
    name = "Move to group"

    def __init__(
        self,
        document: Document,
        source_id: str,
        dest_id: str,
        layer_ids: list[str],
        dest_index: int,
    ) -> None:
        self.document = document
        self.source_id = source_id
        self.dest_id = dest_id
        self.layer_ids = list(layer_ids)
        self.dest_index = int(dest_index)
        self._moving: list[Layer] = []
        self._source_before: list[Layer] | None = None
        self._dest_before: list[Layer] | None = None
        self._previous_active: str | None = None
        self._captured = False

    def execute(self) -> None:
        source = self.document.group_by_id(self.source_id)
        dest = self.document.group_by_id(self.dest_id)
        if source is None or dest is None or source is dest:
            return
        if not self._captured:
            self._moving = layers_leaving(source.layers, self.layer_ids)
            self._source_before = list(source.layers)
            self._dest_before = list(dest.layers)
            self._previous_active = self.document.active_group_id
            self._captured = True
        if not self._moving:
            return
        # Redo finds the objects back on the source page. Strip them from
        # both pages first so a second execute cannot insert a duplicate.
        moving_ids = {id(layer) for layer in self._moving}
        source.layers = [layer for layer in source.layers if id(layer) not in moving_ids]
        dest.layers = [layer for layer in dest.layers if id(layer) not in moving_ids]
        index = max(0, min(self.dest_index, len(dest.layers)))
        for offset, layer in enumerate(self._moving):
            dest.layers.insert(index + offset, layer)
        self.document.active_group_id = self.dest_id

    def undo(self) -> None:
        if self._source_before is None or self._dest_before is None:
            return
        source = self.document.group_by_id(self.source_id)
        dest = self.document.group_by_id(self.dest_id)
        if source is None or dest is None:
            return
        source.layers = list(self._source_before)
        dest.layers = list(self._dest_before)
        if self._previous_active and self.document.group_by_id(self._previous_active) is not None:
            self.document.active_group_id = self._previous_active

"""Document model and undoable commands.

The model has no Qt dependency. A document is a canvas size plus an ordered
layer list (index 0 is the bottom). Tools and plugins mutate it only through
``Command`` objects so every edit can be undone.
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


@dataclass
class RedactionInfo:
    """How a redaction mask is applied at composite time.

    ``kind`` is ``blur`` (``amount`` is Gaussian radius) or ``pixelate``
    (``amount`` is the mosaic block size in pixels).
    """

    kind: str = "pixelate"
    amount: float = 12.0


@dataclass
class Layer:
    """One stack entry.

    Raster and text layers store straight RGBA pixels in ``image``, placed at
    ``(x, y)`` on the canvas. Paint and shapes grow that bitmap around new
    marks; they do not resize every layer to the document. A layer that
    already covers the canvas (background, import, flatten) keeps that size.
    A redaction layer stores an ``L`` mask (255 = fully redacted) in document
    coordinates and does not paint its own color; compositing blurs or
    pixelates whatever is already underneath the mask.
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
    }


def apply_visual(layer: Layer, state: dict[str, Any]) -> None:
    layer.image = state["image"].copy()
    layer.x = int(state["x"])
    layer.y = int(state["y"])
    layer.kind = state["kind"]
    layer.text = copy.deepcopy(state["text"])
    layer.mask = None if state["mask"] is None else state["mask"].copy()
    layer.redaction = copy.deepcopy(state["redaction"])


@dataclass
class Document:
    width: int
    height: int
    layers: list[Layer] = field(default_factory=list)
    # None means "no selection" (edits hit the whole target). An L image is a
    # real selection: 255 is included, 0 is excluded.
    selection: Image.Image | None = None

    def index_of(self, layer_id: str) -> int:
        for index, layer in enumerate(self.layers):
            if layer.id == layer_id:
                return index
        raise KeyError(layer_id)

    def layer_by_id(self, layer_id: str) -> Layer:
        return self.layers[self.index_of(layer_id)]


def new_document(width: int, height: int, background: str = "white") -> Document:
    """Create a document with a single base layer.

    ``background`` is ``white``, ``black``, or ``transparent``.
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
    return Document(width=width, height=height, layers=[layer])


class Command:
    """An undoable mutation. ``History.push`` calls ``execute`` once."""

    name = "Edit"

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
    name = "Redaction settings"

    def __init__(self, layer: Layer, kind: str, amount: float) -> None:
        self.layer = layer
        self.kind = kind
        self.amount = float(amount)
        self._old: RedactionInfo | None = None
        self._captured = False

    def execute(self) -> None:
        if not self._captured:
            self._old = copy.deepcopy(self.layer.redaction)
            self._captured = True
        self.layer.redaction = RedactionInfo(self.kind, self.amount)

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


class DeleteLayerCommand(Command):
    name = "Delete layer"

    def __init__(self, document: Document, index: int) -> None:
        self.document = document
        self.index = index
        self.removed: Layer | None = None

    def execute(self) -> None:
        if self.removed is None:
            self.removed = self.document.layers[self.index]
        if self.index < len(self.document.layers) and self.document.layers[self.index] is self.removed:
            self.document.layers.pop(self.index)
            return
        self.document.layers = [layer for layer in self.document.layers if layer is not self.removed]

    def undo(self) -> None:
        if self.removed is None:
            return
        self.document.layers.insert(self.index, self.removed)


class DuplicateLayerCommand(Command):
    name = "Duplicate layer"

    def __init__(self, document: Document, index: int) -> None:
        self.document = document
        self.index = index
        self.copy: Layer | None = None

    def execute(self) -> None:
        if self.copy is None:
            source = self.document.layers[self.index]
            self.copy = source.clone()
            self.copy.name = f"{source.name} copy"
        self.document.layers.insert(self.index + 1, self.copy)

    def undo(self) -> None:
        if self.copy is None:
            return
        if (
            self.index + 1 < len(self.document.layers)
            and self.document.layers[self.index + 1] is self.copy
        ):
            self.document.layers.pop(self.index + 1)


class ReorderLayerCommand(Command):
    name = "Reorder layer"

    def __init__(self, document: Document, old_index: int, new_index: int) -> None:
        self.document = document
        self.old_index = old_index
        self.new_index = new_index

    def _move(self, source: int, dest: int) -> None:
        layer = self.document.layers.pop(source)
        self.document.layers.insert(dest, layer)

    def execute(self) -> None:
        self._move(self.old_index, self.new_index)

    def undo(self) -> None:
        self._move(self.new_index, self.old_index)


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
    ) -> None:
        self.document = document
        self.new_layers = list(layers)
        self.new_width = int(width)
        self.new_height = int(height)
        self.new_selection = _copy_selection(selection)
        self.name = name
        self.old_layers: list[Layer] | None = None
        self.old_width = 0
        self.old_height = 0
        self.old_selection: Image.Image | None = None
        self._captured = False

    def execute(self) -> None:
        if not self._captured:
            self.old_layers = list(self.document.layers)
            self.old_width = self.document.width
            self.old_height = self.document.height
            self.old_selection = _copy_selection(self.document.selection)
            self._captured = True
        self.document.layers = list(self.new_layers)
        self.document.width = self.new_width
        self.document.height = self.new_height
        self.document.selection = _copy_selection(self.new_selection)

    def undo(self) -> None:
        if self.old_layers is None:
            return
        self.document.layers = list(self.old_layers)
        self.document.width = self.old_width
        self.document.height = self.old_height
        self.document.selection = _copy_selection(self.old_selection)

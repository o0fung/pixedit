"""Listeners, undo, the active layer, and opening a document.

Mixed into ``Session``. History, notifications, and which layer is selected
live here. Stack edits are in ``layers``. The pixel selection is in ``selection``.
"""

from __future__ import annotations

from typing import Any, Callable

from pixedit.model import Document, History, Layer, TextInfo, new_document
from pixedit.textutil import default_font_path

Listener = Callable[[str], None]


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
        "snap": True,
        "gradient_kind": "linear",
        "gradient_color": (255, 255, 255, 255),
        "frame_delay": 100,
        "text_font": default_font_path(),
        "text_font_family": "DejaVu Sans",
        "text_size": 48,
        "text_align": "left",
        "text_line_spacing": 4,
        "text_rotation": 0,
    }


class CoreMixin:
    """Undo, the active layer, and replacing the open document."""

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

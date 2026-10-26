"""Editing session shared by the window, tools, and plugins.

The session owns the document, the undo history, the foreground color, and
tool options. It does not import Qt. The window listens for ``change`` and
``preview`` notifications and redraws.

``Session`` is one class. The methods are split by job so each file stays
readable. Every method is still called on the session:

- ``core`` — listeners, undo and redo, the active layer, open and new
- ``layers`` — stack edits, pages, merge, and place
- ``selection`` — the pixel selection
- ``transform`` — crop, canvas size, image size, flip and rotate
- ``raster`` — brush lifecycle and masks
- ``text`` — text layers
- ``color`` — the live color grade and menu filters
"""

from __future__ import annotations

from typing import Any, Callable

from pixedit.model import Document, History, Layer, RedactionInfo, TextInfo, new_document
from pixedit.session.color import ColorMixin
from pixedit.session.core import CoreMixin, Listener, default_options
from pixedit.session.layers import LayersMixin
from pixedit.session.raster import RasterMixin
from pixedit.session.selection import SelectionMixin
from pixedit.session.text import TextMixin
from pixedit.session.transform import TransformMixin

__all__ = ["Listener", "Session", "default_options"]


class Session(
    LayersMixin,
    SelectionMixin,
    TransformMixin,
    RasterMixin,
    TextMixin,
    ColorMixin,
    CoreMixin,
):
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

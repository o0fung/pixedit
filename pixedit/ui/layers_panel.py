"""The layer list.

Rows show each page and its layers. Drags, visibility, and opacity go
through the session so they stay one undo step.
"""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QSlider,
    QStyledItemDelegate,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pixedit.model import BLEND_MODES, layers_leaving, mask_target
from pixedit.ui.canvas import pil_to_pixmap
from pixedit.ui.icons import icon_named
from pixedit.ui.thumbs import THUMB_SIZE, checker_image, layer_thumb, text_head
# Same square picture buttons as the actions under the tool options.
from pixedit.ui.tool_options import _ACTION_ICON, _ACTION_SIDE, _icon_button


def stack_move(model_ids: list, visual_ids_front_first: list) -> tuple[int, int] | None:
    """Map a layer-list drop to one ``(old_index, new_index)`` in the document.

    The list paints the front of the stack on the first row, so the visual
    order is the reverse of ``document.layers``. A drag moves exactly one
    layer; the others keep their relative order. ``None`` means the drop
    did not change the stack, or the ids are not a single move.
    """
    desired = list(reversed(visual_ids_front_first))
    if desired == list(model_ids):
        return None
    if len(desired) != len(model_ids) or set(desired) != set(model_ids):
        return None
    for layer_id in model_ids:
        rest_now = [item for item in model_ids if item != layer_id]
        rest_next = [item for item in desired if item != layer_id]
        if rest_now == rest_next:
            return model_ids.index(layer_id), desired.index(layer_id)
    return None


def _text_mark_pixmap(pixels: int) -> QPixmap:
    """The text-tool icon centered on a checker square of ``pixels`` pixels."""
    pixmap = pil_to_pixmap(checker_image(pixels))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    # A margin keeps the mark inside the square, matching the other thumbnails.
    margin = pixels // 6
    if margin * 2 >= pixels:
        margin = 0
    span = max(1, pixels - 2 * margin)
    icon_named("text").paint(painter, margin, margin, span, span)
    painter.end()
    return pixmap


class LayerRowDelegate(QStyledItemDelegate):
    """Rows stay as wide as the list, so a long name or caption ellipsizes.

    The default item width is the full text, which opens a sideways scrollbar
    in this narrow column. Matching the viewport width lets ElideRight cut the
    line that does not fit.
    """

    def sizeHint(self, option, index):
        hint = super().sizeHint(option, index)
        widget = option.widget
        if widget is not None:
            width = widget.viewport().width()
            if width > 0:
                hint.setWidth(width)
        return hint


_ITEM_ID = Qt.ItemDataRole.UserRole
_ITEM_KIND = Qt.ItemDataRole.UserRole + 1


def _drop_index(visual_front_first: list[str], moved: set[str]) -> int:
    """Model index where a block dropped into a page should be inserted.

    The list is front-first, matching the layer tree. Layers in ``moved`` are
    the incoming rows. Destination layers after that block are behind it, so
    their count is the index in the page's back-to-front stack.
    """
    behind = 0
    seen = False
    for layer_id in visual_front_first:
        if layer_id in moved:
            seen = True
            continue
        if seen:
            behind += 1
    return behind


class LayerTree(QTreeWidget):
    """Groups, then the layers inside each group.

    ``startDrag`` does not return until the drop or cancel has finished, so
    ``drag_finished`` is emitted outside Qt's nested drag loop. The panel
    updates the document from that signal, not from inside the drop.
    """

    drag_started = pyqtSignal()
    drag_finished = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setColumnCount(1)
        self.setHeaderHidden(True)
        self.setIndentation(16)

    def startDrag(self, supported_actions) -> None:
        self.drag_started.emit()
        super().startDrag(supported_actions)
        self.drag_finished.emit()

    def dropEvent(self, event) -> None:
        super().dropEvent(event)
        # Dropping on the middle of a row nests the layer under it. A layer
        # has no children; the row belongs beside that layer, just below it.
        for index in range(self.topLevelItemCount()):
            group = self.topLevelItem(index)
            if group is None:
                continue
            child_index = 0
            while child_index < group.childCount():
                child = group.child(child_index)
                nested = [child.takeChild(0) for _ in range(child.childCount())]
                for offset, item in enumerate(nested, start=1):
                    group.insertChild(child_index + offset, item)
                child_index += 1

    def keyPressEvent(self, event) -> None:
        # Space is the temporary hand tool. The base tree would use it to
        # flip this row's checkbox, which is the layer's visibility.
        if event.key() == Qt.Key.Key_Space:
            event.ignore()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        # The matching release has to reach the canvas, or the hand cursor sticks.
        if event.key() == Qt.Key.Key_Space:
            event.ignore()
            return
        super().keyReleaseEvent(event)


class LayerPanel(QWidget):
    def __init__(self, session, on_rename) -> None:
        super().__init__()
        self.session = session
        self.on_rename = on_rename
        self._building = False
        # True from the drop until the document matches the list, so a
        # selection change during the drag does not pick the wrong layer.
        self._reordering = False
        self._opacity_before: float | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        title = QLabel("Layers")
        title.setObjectName("section")
        layout.addWidget(title)
        self.list = LayerTree()
        self.list.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self.list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setItemDelegate(LayerRowDelegate(self.list))
        self.list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.list.setDropIndicatorShown(True)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.setToolTip(
            "Click a group, or a layer in it, to show that page. "
            "Drag a layer to restack it, or onto another group to move it. "
            "Drag a group to change the PDF page order. "
            "Shift-click selects a range inside one group. "
            "Command-click or Ctrl-click adds or removes a layer."
        )
        self.list.itemSelectionChanged.connect(self._on_selection)
        self.list.itemChanged.connect(self._on_item_changed)
        self.list.itemDoubleClicked.connect(self._on_double_click)
        self.list.drag_started.connect(self._on_drag_started)
        # Queued so a rebuild of the rows waits until the drag's mouse
        # handling has returned, not while Qt is still inside it.
        self.list.drag_finished.connect(self._on_drag_finished, Qt.ConnectionType.QueuedConnection)
        layout.addWidget(self.list, 1)

        buttons = QHBoxLayout()
        self._add_button(buttons, "New", self.session.add_blank_layer, icon="layer-new")
        self._add_button(buttons, "Copy", self.session.duplicate_active, icon="layer-copy")
        self._add_button(buttons, "Delete", self.session.delete_active, icon="layer-delete")
        self._add_button(
            buttons,
            "Visibility",
            self.session.toggle_selected_visibility,
            "Toggle visibility of every selected layer. Click again to put them back.",
            icon="layer-visibility",
        )
        buttons.addStretch(1)
        layout.addLayout(buttons)
        buttons2 = QHBoxLayout()
        self._add_button(buttons2, "Up", lambda: self.session.move_active(1), icon="layer-up")
        self._add_button(buttons2, "Down", lambda: self.session.move_active(-1), icon="layer-down")
        self._add_button(
            buttons2,
            "Group",
            self.session.add_group,
            "New group. It has its own canvas and does not mix with the other groups.",
            icon="folder",
        )
        self._add_button(
            buttons2,
            "Merge down",
            self.session.merge_down_active,
            "Bake this layer into the one below it. A mask bakes into the layer it belongs to. "
            "Flatten image, in the Layer menu, collapses the group you are viewing (Ctrl+Shift+E).",
            icon="layer-merge-down",
        )
        buttons2.addStretch(1)
        layout.addLayout(buttons2)

        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.sliderPressed.connect(self._opacity_pressed)
        self.opacity.valueChanged.connect(self._opacity_moved)
        self.opacity.sliderReleased.connect(self._opacity_released)
        self.opacity_label = QLabel("Opacity  100%")
        layout.addWidget(self.opacity_label)
        layout.addWidget(self.opacity)

        self.blend = QComboBox()
        for mode in BLEND_MODES:
            self.blend.addItem(mode.title(), mode)
        self.blend.currentIndexChanged.connect(self._blend_changed)
        layout.addWidget(QLabel("Blend"))
        layout.addWidget(self.blend)

    def _add_button(self, layout: QHBoxLayout, text: str, callback, tip: str = "", icon: str = "") -> None:
        if icon:
            button = _icon_button(text, icon, tip, side=_ACTION_SIDE, icon_px=_ACTION_ICON)
        else:
            button = QPushButton(text)
            button.setAccessibleName(text)
            if tip:
                button.setToolTip(tip)
        # A focused button treats Space as a click. These actions stay on
        # the pointer so Space can still pan.
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.clicked.connect(callback)
        layout.addWidget(button)

    def refresh(self) -> None:
        self._building = True
        # Block the tree, not its model.
        # 1. clear() and addTopLevelItem() would otherwise emit itemSelectionChanged
        #    and itemChanged, which this panel treats as a click or a visibility toggle.
        # 2. The view lays rows out only from the model's reset and insert
        #    signals. Blocking those leaves the default Background row, and
        #    every layer opened later, in the model but not on screen.
        self.list.blockSignals(True)
        self.list.clear()
        document = self.session.document
        active_group_id = document.active_group_id
        active_layer_id = None
        active_layers = document.layers
        if 0 <= self.session.layer_index < len(active_layers):
            active_layer_id = active_layers[self.session.layer_index].id
        chosen = set(self.session.selected_ids())
        current_item = None
        rows: list[tuple[QTreeWidgetItem, str, str]] = []
        # Pages run top to bottom in export order. Inside a page the front
        # layer is the first row, matching the old flat list.
        for group in document.groups:
            header = QTreeWidgetItem([self._group_label(group)])
            header.setData(0, _ITEM_ID, group.id)
            header.setData(0, _ITEM_KIND, "group")
            header.setFlags(
                Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsDragEnabled
                | Qt.ItemFlag.ItemIsDropEnabled
            )
            if group.id == active_group_id:
                font = QFont()
                font.setBold(True)
                header.setFont(0, font)
                header.setBackground(0, QBrush(QColor("#1c3a4a")))
            self.list.addTopLevelItem(header)
            for layer in reversed(group.layers):
                item = QTreeWidgetItem([self._label(layer, group.layers)])
                item.setData(0, _ITEM_ID, layer.id)
                item.setData(0, _ITEM_KIND, "layer")
                item.setFlags(
                    Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsDragEnabled
                    | Qt.ItemFlag.ItemIsDropEnabled
                )
                item.setCheckState(0, Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked)
                item.setIcon(0, self._thumb_icon(layer))
                if layer.kind == "text" and layer.text is not None and layer.text.text.strip():
                    item.setToolTip(0, layer.text.text)
                header.addChild(item)
                rows.append((item, layer.id, group.id))
                if group.id == active_group_id and layer.id == active_layer_id:
                    current_item = item
            header.setExpanded(True)
        # setCurrentItem clears every other row. The selection flags are
        # applied after that, or a multi-selection would collapse to one layer.
        if current_item is not None:
            self.list.setCurrentItem(current_item)
        for item, layer_id, group_id in rows:
            item.setSelected(group_id == active_group_id and layer_id in chosen)
        self.list.blockSignals(False)
        self._sync_controls()
        self._building = False

    def _group_label(self, group) -> str:
        name = group.name.replace("\r", " ").replace("\n", " ").strip()
        return f"{name}    {group.width} × {group.height}"

    def _label(self, layer, layers) -> str:
        """Name on the first line. Opacity, and the text head or the kind, on the second.

        A text layer quotes the opening words instead of the word "text".
        The icon already says that the layer is text. A mask is indented and
        names the picture it filters, such as "on Photo".
        """
        opacity = f"{int(round(layer.opacity * 100))}%"
        if layer.kind == "text" and layer.text is not None:
            head = text_head(layer.text.text)
            detail = f"“{head}”  {opacity}" if head else opacity
        elif layer.kind == "redaction":
            target = mask_target(layers, layer)
            owner = target.name if target is not None else "no layer"
            detail = f"on {owner}  {opacity}"
        else:
            detail = f"{opacity}  {layer.blend_mode}"
        name = layer.name.replace("\r", " ").replace("\n", " ").strip()
        if layer.kind == "redaction":
            name = f"  {name}"
        return f"{name}\n{detail}"

    def _thumb_icon(self, layer) -> QIcon:
        """Pixel preview, or the text-tool mark on the same checker square.

        The pixmap is built at the screen's pixel size and tagged with the
        device ratio so a retina display does not blur the 32px slot.
        """
        dpr = max(1.0, float(self.devicePixelRatioF()))
        pixels = max(1, int(round(THUMB_SIZE * dpr)))
        if layer.kind == "text":
            pixmap = _text_mark_pixmap(pixels)
        else:
            pixmap = pil_to_pixmap(layer_thumb(layer, pixels))
        pixmap.setDevicePixelRatio(dpr)
        return QIcon(pixmap)

    def _on_drag_started(self) -> None:
        self._reordering = True
        # Restack moves one row. A multi-selection would otherwise drag every
        # highlighted row, which the document cannot record as one move.
        current = self.list.currentItem()
        selected = self.list.selectedItems()
        if current is None or len(selected) <= 1:
            return
        self.list.blockSignals(True)
        for item in selected:
            if item is not current:
                item.setSelected(False)
        self.list.blockSignals(False)

    def _on_drag_finished(self) -> None:
        if self._building:
            self._reordering = False
            return
        # The tree is already in the dropped order. The document still has
        # the previous order until the apply below.
        self._reordering = True
        self._apply_pending_drop()

    def _read_visual(self) -> list[tuple[str, list[str]]] | None:
        """(group id, layer ids front-first) or None when the drop nested badly."""
        seen_groups: list[str] = []
        seen_layers: list[str] = []
        visual: list[tuple[str, list[str]]] = []
        for index in range(self.list.topLevelItemCount()):
            item = self.list.topLevelItem(index)
            if item is None or item.data(0, _ITEM_KIND) != "group":
                return None
            group_id = item.data(0, _ITEM_ID)
            if not isinstance(group_id, str) or group_id in seen_groups:
                return None
            seen_groups.append(group_id)
            layer_ids: list[str] = []
            for child_index in range(item.childCount()):
                child = item.child(child_index)
                if child.data(0, _ITEM_KIND) != "layer" or child.childCount():
                    return None
                layer_id = child.data(0, _ITEM_ID)
                if not isinstance(layer_id, str) or layer_id in seen_layers:
                    return None
                seen_layers.append(layer_id)
                layer_ids.append(layer_id)
            visual.append((group_id, layer_ids))
        return visual

    def _apply_pending_drop(self) -> None:
        # One drag does one thing, then the tree is rebuilt from the document.
        # 1. Group headers reordered, layers unmoved: that becomes the page order.
        # 2. One layer moved inside its page: restack, masks staying with their picture.
        # 3. A picture dropped on another page: move it and its masks, then show that page.
        # A mask dropped without its picture, or a group nested inside a group, snaps back.
        try:
            visual = self._read_visual()
            document = self.session.document
            if visual is None:
                return
            model = [(group.id, [layer.id for layer in reversed(group.layers)]) for group in document.groups]
            if visual == model:
                return
            visual_ids = [group_id for group_id, _layer_ids in visual]
            model_ids = [group_id for group_id, _layer_ids in model]
            if set(visual_ids) != set(model_ids) or len(visual_ids) != len(model_ids):
                return
            model_layers = {group_id: layer_ids for group_id, layer_ids in model}
            membership = {
                layer_id: group_id for group_id, layer_ids in model for layer_id in layer_ids
            }
            moved = [
                layer_id
                for group_id, layer_ids in visual
                for layer_id in layer_ids
                if membership.get(layer_id) != group_id
            ]
            if not moved and visual_ids != model_ids:
                if all(layer_ids == model_layers[group_id] for group_id, layer_ids in visual):
                    self.session.reorder_groups(visual_ids)
                return
            if not moved and visual_ids == model_ids:
                changes = [
                    (group_id, layer_ids)
                    for group_id, layer_ids in visual
                    if layer_ids != model_layers[group_id]
                ]
                if len(changes) != 1:
                    return
                group_id, visual_front = changes[0]
                if document.active_group_id != group_id:
                    self.session.activate_group(group_id)
                group = document.group_by_id(group_id)
                if group is None:
                    return
                move = stack_move([layer.id for layer in group.layers], visual_front)
                if move is not None:
                    old_index, new_index = move
                    self.session.move_layer(old_index, new_index)
                return
            if moved and visual_ids == model_ids:
                sources = {membership[layer_id] for layer_id in moved if layer_id in membership}
                dests = {group_id for group_id, layer_ids in visual for layer_id in layer_ids if layer_id in moved}
                if len(sources) != 1 or len(dests) != 1:
                    return
                source_id = next(iter(sources))
                dest_id = next(iter(dests))
                source = document.group_by_id(source_id)
                if source is None:
                    return
                leaving = layers_leaving(source.layers, moved)
                leaving_ids = {layer.id for layer in leaving}
                # A lone mask is not in the leaving set. Reject it so the mask
                # cannot be separated from the picture it filters.
                if not leaving or not set(moved) <= leaving_ids:
                    return
                dest_visual = next(layer_ids for group_id, layer_ids in visual if group_id == dest_id)
                self.session.move_layers_to_group(moved, dest_id, _drop_index(dest_visual, set(moved)))
        finally:
            self._reordering = False
            # Rebuild from the document so a rejected drop snaps back, and
            # labels stay in sync when no window listener refreshes us.
            self.refresh()

    def _on_double_click(self, item: QTreeWidgetItem, _column: int) -> None:
        if item.data(0, _ITEM_KIND) == "group":
            group_id = item.data(0, _ITEM_ID)
            group = self.session.document.group_by_id(group_id) if isinstance(group_id, str) else None
            if group is None:
                return
            name, ok = QInputDialog.getText(self, "Rename group", "Name", text=group.name)
            if ok:
                self.session.rename_group(group.id, name)
            return
        self.on_rename()

    def _on_selection(self) -> None:
        if self._building or self._reordering:
            return
        items = self.list.selectedItems()
        if not items:
            # The active layer always stays selected. Rebuild from the session
            # so a toggle of the last row does not leave the list empty.
            self.refresh()
            return
        current = self.list.currentItem()
        if current is None or not current.isSelected():
            current = items[-1]
        if current.data(0, _ITEM_KIND) == "group":
            group_id = current.data(0, _ITEM_ID)
            if isinstance(group_id, str) and group_id != self.session.document.active_group_id:
                self.session.activate_group(group_id)
            else:
                # The header is not a paint target. Put the highlight back
                # on the layer this page was already using.
                self.refresh()
            return
        parent = current.parent()
        if parent is None:
            return
        group_id = parent.data(0, _ITEM_ID)
        if not isinstance(group_id, str):
            return
        layer_ids = []
        for item in items:
            if item.data(0, _ITEM_KIND) != "layer" or item.parent() is not parent:
                continue
            layer_id = item.data(0, _ITEM_ID)
            if isinstance(layer_id, str):
                layer_ids.append(layer_id)
        active_id = current.data(0, _ITEM_ID)
        if not isinstance(active_id, str):
            active_id = None
        if group_id != self.session.document.active_group_id:
            self.session.activate_group(group_id, layer_ids=layer_ids, active_id=active_id)
            return
        self.session.select_layers(layer_ids, active_id=active_id)
        self._sync_controls()
        self.session.preview()

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._building or column != 0:
            return
        if item.data(0, _ITEM_KIND) != "layer":
            return
        layer_id = item.data(0, _ITEM_ID)
        if not isinstance(layer_id, str):
            return
        layer = None
        for group in self.session.document.groups:
            for candidate in group.layers:
                if candidate.id == layer_id:
                    layer = candidate
                    break
            if layer is not None:
                break
        if layer is None:
            return
        visible = item.checkState(0) == Qt.CheckState.Checked
        if layer.visible != visible:
            self.session.set_visibility(layer, visible)

    def _sync_controls(self) -> None:
        layer = self.session.active_layer()
        self.opacity.blockSignals(True)
        self.blend.blockSignals(True)
        if layer is None:
            self.opacity.setValue(100)
            self.opacity_label.setText("Opacity")
        else:
            value = int(round(layer.opacity * 100))
            self.opacity.setValue(value)
            self.opacity_label.setText(f"Opacity  {value}%")
            index = self.blend.findData(layer.blend_mode)
            if index >= 0:
                self.blend.setCurrentIndex(index)
        self.opacity.blockSignals(False)
        self.blend.blockSignals(False)

    def _opacity_pressed(self) -> None:
        layer = self.session.active_layer()
        self._opacity_before = None if layer is None else layer.opacity

    def _opacity_moved(self, value: int) -> None:
        if self._building:
            return
        layer = self.session.active_layer()
        if layer is None or self._opacity_before is None:
            return
        layer.opacity = value / 100.0
        self.opacity_label.setText(f"Opacity  {value}%")
        self.session.preview()

    def _opacity_released(self) -> None:
        layer = self.session.active_layer()
        if layer is None or self._opacity_before is None:
            return
        final = layer.opacity
        start = self._opacity_before
        self._opacity_before = None
        layer.opacity = start
        self.session.set_opacity(layer, final)

    def _blend_changed(self, index: int) -> None:
        if self._building:
            return
        layer = self.session.active_layer()
        if layer is None:
            return
        mode = self.blend.itemData(index)
        self.session.set_blend(layer, str(mode))

"""Tool options and the layer stack. Both talk to the session, not to pixels directly."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from pixedit.model import BLEND_MODES


class ToolOptions(QWidget):
    """Builds controls from ``tool.option_schema()`` so plugins get a panel for free."""

    def __init__(self, session, on_option=None) -> None:
        super().__init__()
        self.session = session
        self._on_option = on_option
        self._widgets: dict[str, QWidget] = {}
        self._labels: dict[str, QLabel] = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        self.heading = QLabel("Tool")
        self.heading.setObjectName("section")
        self.hint = QLabel("")
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        outer.addWidget(self.heading)
        outer.addWidget(self.hint)
        self.form_host = QWidget()
        self.form = QFormLayout(self.form_host)
        self.form.setContentsMargins(0, 8, 0, 0)
        outer.addWidget(self.form_host)
        outer.addStretch(1)

    def set_tool(self, tool) -> None:
        self.heading.setText(tool.label)
        self.hint.setText(tool.hint or "")
        while self.form.rowCount():
            self.form.removeRow(0)
        self._widgets.clear()
        self._labels.clear()
        for spec in tool.option_schema():
            key = spec["key"]
            self.session.options.setdefault(key, spec.get("default"))
            widget = self._make(spec)
            label = QLabel(spec["label"])
            self.form.addRow(label, widget)
            self._widgets[key] = widget
            self._labels[key] = label

    def sync(self, key: str, value) -> None:
        widget = self._widgets.get(key)
        if isinstance(widget, QSlider):
            widget.blockSignals(True)
            widget.setValue(int(value))
            widget.blockSignals(False)
            label = self._labels.get(key)
            if label is not None:
                base = label.text().split("  ")[0]
                label.setText(f"{base}  {int(value)}")

    def _make(self, spec: dict) -> QWidget:
        key = spec["key"]
        kind = spec["type"]
        if kind == "int":
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(int(spec["min"]))
            slider.setMaximum(int(spec["max"]))
            slider.setValue(int(self.session.option(key, spec.get("default", spec["min"]))))
            label = None

            def on_change(value, key=key, spec=spec, slider=slider):
                self.session.options[key] = int(value)
                row_label = self._labels.get(key)
                if row_label is not None:
                    row_label.setText(f"{spec['label']}  {int(value)}")
                # Only the live drag updates a redaction layer. Building the
                # slider, or releasing it, must not start a half-finished edit.
                if key == "redact_amount" and slider.isSliderDown():
                    self.session.set_redaction_settings(
                        str(self.session.option("redact_kind", "pixelate")),
                        float(value),
                        transient=True,
                    )
                if self._on_option is not None:
                    self._on_option(key, int(value))

            def on_release(key=key):
                if key == "redact_amount":
                    self.session.set_redaction_settings(
                        str(self.session.option("redact_kind", "pixelate")),
                        float(self.session.option(key, 12)),
                        transient=False,
                    )

            slider.valueChanged.connect(on_change)
            slider.sliderReleased.connect(on_release)
            on_change(slider.value())
            return slider
        if kind == "choice":
            combo = QComboBox()
            for choice in spec["choices"]:
                combo.addItem(str(choice))
            current = str(self.session.option(key, spec.get("default")))
            index = combo.findText(current)
            combo.setCurrentIndex(max(0, index))

            def on_choice(text, key=key):
                self.session.options[key] = text
                if key == "redact_kind":
                    self.session.set_redaction_settings(
                        text,
                        float(self.session.option("redact_amount", 12)),
                        transient=False,
                    )

            combo.currentTextChanged.connect(on_choice)
            return combo
        if kind == "bool":
            box = QCheckBox()
            box.setChecked(bool(self.session.option(key, spec.get("default", False))))

            def on_check(checked, key=key):
                self.session.options[key] = bool(checked)

            box.toggled.connect(on_check)
            return box
        return QLabel(str(self.session.option(key, "")))


class LayerPanel(QWidget):
    def __init__(self, session, on_rename) -> None:
        super().__init__()
        self.session = session
        self.on_rename = on_rename
        self._building = False
        self._opacity_before: float | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        title = QLabel("Layers")
        title.setObjectName("section")
        layout.addWidget(title)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._on_row)
        self.list.itemChanged.connect(self._on_item_changed)
        self.list.itemDoubleClicked.connect(lambda _item: self.on_rename())
        layout.addWidget(self.list, 1)

        buttons = QHBoxLayout()
        self._add_button(buttons, "New", self.session.add_blank_layer)
        self._add_button(buttons, "Copy", self.session.duplicate_active)
        self._add_button(buttons, "Delete", self.session.delete_active)
        layout.addLayout(buttons)
        buttons2 = QHBoxLayout()
        self._add_button(buttons2, "Up", lambda: self.session.move_active(1))
        self._add_button(buttons2, "Down", lambda: self.session.move_active(-1))
        self._add_button(buttons2, "Merge", self.session.merge_down_active)
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

    def _add_button(self, layout: QHBoxLayout, text: str, callback) -> None:
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)

    def refresh(self) -> None:
        self._building = True
        self.list.blockSignals(True)
        self.list.clear()
        layers = self.session.document.layers
        # The list shows the front of the stack first, which is the last model entry.
        for layer in reversed(layers):
            item = QListWidgetItem(self._label(layer))
            item.setData(Qt.ItemDataRole.UserRole, layer.id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            item.setCheckState(Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked)
            self.list.addItem(item)
        if self.session.layer_index >= 0:
            row = len(layers) - 1 - self.session.layer_index
            self.list.setCurrentRow(row)
        self.list.blockSignals(False)
        self._sync_controls()
        self._building = False

    def _label(self, layer) -> str:
        kind = {"redaction": "mask", "text": "text"}.get(layer.kind, layer.blend_mode)
        return f"{layer.name}   {int(round(layer.opacity * 100))}%  {kind}"

    def _on_row(self, row: int) -> None:
        if self._building:
            return
        # Clicking off the rows clears the selection. Paint tools then pick a
        # visible layer themselves instead of editing a highlight that is gone.
        if row < 0:
            if self.session.layer_index != -1:
                self.session.layer_index = -1
                self._sync_controls()
            return
        index = len(self.session.document.layers) - 1 - row
        self.session.layer_index = index
        self._sync_controls()
        self.session.preview()

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if self._building:
            return
        layer_id = item.data(Qt.ItemDataRole.UserRole)
        try:
            layer = self.session.document.layer_by_id(layer_id)
        except KeyError:
            return
        visible = item.checkState() == Qt.CheckState.Checked
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

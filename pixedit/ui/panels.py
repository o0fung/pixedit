"""Tool options and the layer stack. Both talk to the session, not to pixels directly."""

from __future__ import annotations

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pixedit.model import BLEND_MODES
from pixedit.paint import brush_preview_image
from pixedit.textutil import group_fonts
from pixedit.ui.canvas import pil_to_pixmap


def wrap_degrees(degrees: float) -> int:
    """Nearest angle in -180..180. 200° and -160° are the same turn."""
    wrapped = (float(degrees) + 180.0) % 360.0 - 180.0
    rounded = int(round(wrapped))
    if rounded > 180:
        rounded -= 360
    elif rounded < -180:
        rounded += 360
    return rounded


class ToolOptions(QWidget):
    """Builds controls from ``tool.option_schema()`` so plugins get a panel for free."""

    def __init__(self, session, on_option=None, on_action=None) -> None:
        super().__init__()
        self.session = session
        self._on_option = on_option
        self._on_action = on_action
        self._widgets: dict[str, QWidget] = {}
        self._labels: dict[str, QLabel] = {}
        self._specs: dict[str, dict] = {}
        self._font_catalog: dict[str, list[tuple[str, str]]] = {}
        self._font_by_path: dict[str, tuple[str, str]] = {}
        self._loading = False
        self.action_buttons: list[QPushButton] = []
        self._action_specs: list[dict] = []
        # Width the command rows were packed for. A later resize reflows only
        # when the visible column width changes, so a rebuild cannot loop.
        self._action_packed_width: int | None = None
        # Parent the timer to this widget so a close cancels a pending reflow.
        self._reflow_timer = QTimer(self)
        self._reflow_timer.setSingleShot(True)
        self._reflow_timer.timeout.connect(self._reflow_later)
        self.brush_preview: QLabel | None = None
        self._brush_erase = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        # The scroll area must not squash the font menu or the command buttons.
        # A shorter viewport scrolls instead of overlapping the rows.
        outer.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.heading = QLabel("Tool")
        self.heading.setObjectName("section")
        self.hint = QLabel("")
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        outer.addWidget(self.heading)
        outer.addWidget(self.hint)
        self.form_host = QWidget()
        self.form = QFormLayout(self.form_host)
        self.form.setContentsMargins(0, 4, 0, 0)
        self.form.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        outer.addWidget(self.form_host)
        self.actions_host = QWidget()
        self.actions_layout = QVBoxLayout(self.actions_host)
        self.actions_layout.setContentsMargins(0, 4, 0, 0)
        self.actions_layout.setSpacing(6)
        self.actions_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        outer.addWidget(self.actions_host)
        outer.addStretch(1)

    def set_tool(self, tool) -> None:
        self.brush_preview = None
        self._brush_erase = False
        self._loading = True
        try:
            self.heading.setText(tool.label)
            self.hint.setText(tool.hint or "")
            while self.form.rowCount():
                self.form.removeRow(0)
            self._widgets.clear()
            self._labels.clear()
            self._specs.clear()
            for spec in tool.option_schema():
                key = spec["key"]
                self._specs[key] = spec
                self.session.options.setdefault(key, spec.get("default"))
                widget = self._make(spec)
                label = QLabel(spec["label"])
                self.form.addRow(label, widget)
                self._widgets[key] = widget
                self._labels[key] = label
                if isinstance(widget, QSlider):
                    self._set_slider_label(key, widget.value())
            if "brush_size" in self._specs:
                self._install_brush_preview(bool(getattr(tool, "erase", False)))
            self._rebuild_actions(getattr(tool, "panel_actions", lambda: [])())
            self._sync_mask_amount()
        finally:
            self._loading = False
        self.refresh_brush_preview()
        self.update_actions()

    def sync(self, key: str, value) -> None:
        widget = self._widgets.get(key)
        if isinstance(widget, QSlider):
            widget.blockSignals(True)
            widget.setValue(int(value))
            widget.blockSignals(False)
            self._set_slider_label(key, int(value))
        elif isinstance(widget, QSpinBox):
            widget.blockSignals(True)
            widget.setValue(int(value))
            widget.blockSignals(False)
        if key in ("brush_size", "brush_hardness", "brush_opacity"):
            self.refresh_brush_preview()

    def reload_from_session(self) -> None:
        """Show the stored options in the widgets without restyling a layer."""
        self._loading = True
        try:
            for key, widget in self._widgets.items():
                spec = self._specs.get(key, {})
                value = self.session.option(key, spec.get("default"))
                if isinstance(widget, QSlider):
                    shown = int(value)
                    if key == "text_rotation":
                        shown = wrap_degrees(float(value or 0))
                    widget.blockSignals(True)
                    widget.setValue(shown)
                    widget.blockSignals(False)
                    self._set_slider_label(key, shown)
                elif isinstance(widget, QSpinBox):
                    widget.blockSignals(True)
                    widget.setValue(int(value))
                    widget.blockSignals(False)
                elif isinstance(widget, QComboBox) and spec.get("type") == "font":
                    self._select_font(widget, value)
                elif isinstance(widget, QComboBox) and spec.get("type") == "font_style":
                    self._reload_style_combo(widget)
                elif isinstance(widget, QComboBox):
                    widget.blockSignals(True)
                    stored = value
                    index = widget.findData(stored)
                    if index < 0:
                        index = widget.findText(str(stored))
                    if index >= 0:
                        widget.setCurrentIndex(index)
                    widget.blockSignals(False)
                elif isinstance(widget, QCheckBox):
                    widget.blockSignals(True)
                    widget.setChecked(bool(value))
                    widget.blockSignals(False)
        finally:
            self._loading = False

    def update_actions(self) -> None:
        has_selection = self.session.document.selection is not None
        preview = bool(getattr(self.session, "color_adjust_active", lambda: False)())
        for button in self.action_buttons:
            enabled = True
            if getattr(button, "needs_selection", False):
                enabled = has_selection
            if getattr(button, "needs_color_preview", False):
                enabled = preview
            button.setEnabled(enabled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Widening can put a long label beside its neighbor. Narrowing has to
        # split that pair before the right-hand button slides under the splitter.
        self._schedule_reflow()

    def _schedule_reflow(self) -> None:
        if self._reflow_timer.isActive() or not self._action_specs:
            return
        if self._action_row_budget() == self._action_packed_width:
            return
        # Rebuilding inside resizeEvent runs while the layout is already
        # activating, and the new rows keep Qt's default 640×480 size, stacked
        # on one spot. Pack on the next turn, after this pass has finished.
        self._reflow_timer.start(0)

    def _reflow_later(self) -> None:
        if not self._action_specs or self._action_row_budget() == self._action_packed_width:
            return
        self._rebuild_actions(self._action_specs)
        # show() above makes the new rows eligible for layout. Activate after
        # that, or they stay at the size they had while hidden.
        self.actions_layout.invalidate()
        layout = self.layout()
        if layout is not None:
            layout.activate()

    def _action_row_budget(self) -> int:
        """Width available for one command row, inside the options margins.

        The scroll area's viewport is the visible column. Before the window is
        shown that viewport has no width yet; 232 is the 248px column minus
        the frame's 8px margins. The widget's own width is the wrong source:
        a previous two-column layout can leave it 358px, and the right-hand
        buttons stay under the splitter.
        """
        margins = self.layout().contentsMargins()
        parent = self.parentWidget()
        if parent is not None and parent.width() > 40:
            width = parent.width()
        else:
            width = 232
        return max(1, width - margins.left() - margins.right())

    def _make_action_button(self, spec: dict) -> QPushButton:
        button = QPushButton(str(spec["label"]))
        # Ignored horizontally so a pair does not force the panel to the sum
        # of the two labels. The row still gives each button its text width
        # through the stretch factor below.
        button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        shortcut = spec.get("shortcut")
        button.setToolTip(f"{spec['label']} ({shortcut})" if shortcut else str(spec["label"]))
        button.needs_selection = bool(spec.get("needs_selection", False))
        button.needs_color_preview = bool(spec.get("needs_color_preview", False))
        action_id = str(spec["id"])
        button.clicked.connect(lambda _checked=False, chosen=action_id: self._emit_action(chosen))
        return button

    def _rebuild_actions(self, actions: list[dict]) -> None:
        self._action_specs = list(actions)
        budget = self._action_row_budget()
        self._action_packed_width = budget
        while self.actions_layout.count():
            item = self.actions_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.action_buttons = []
        # Pair two commands only when both full labels fit in the visible
        # column. A label that does not fit beside the one already on the
        # row starts a new row. One label wider than the column still gets
        # its own row; the row's minimum is that label, so the splitter
        # cannot clip it.
        gap = 6
        pending: list[tuple[QPushButton, int]] = []
        for spec in actions:
            button = self._make_action_button(spec)
            button.setParent(self)
            button.ensurePolished()
            pending.append((button, button.sizeHint().width()))
        rows: list[list[tuple[QPushButton, int]]] = []
        current: list[tuple[QPushButton, int]] = []
        used = 0
        for button, width in pending:
            extra = gap if current else 0
            # Two labels on a row, and only when both full widths fit.
            # A third label always starts the next row.
            if current and (len(current) >= 2 or used + extra + width > budget):
                rows.append(current)
                current = []
                used = 0
                extra = 0
            current.append((button, width))
            used += extra + width
        if current:
            rows.append(current)
        for row_buttons in rows:
            row_host = QWidget()
            row_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            row = QHBoxLayout(row_host)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(gap)
            widest = 0
            tallest = 0
            for button, width in row_buttons:
                widest = max(widest, width)
                tallest = max(tallest, button.sizeHint().height())
                row.addWidget(button, max(1, width))
                self.action_buttons.append(button)
            # Floor is the wider label, not the sum. The sum would lock the
            # options widget at the two-column width and keep the right button
            # under the splitter. The height floor keeps an Ignored button
            # from collapsing the row.
            row_host.setMinimumWidth(widest)
            row_host.setMinimumHeight(tallest)
            self.actions_layout.addWidget(row_host)
            # Rows created after the panel is already on screen stay hidden,
            # and a layout skips hidden widgets, so they would keep Qt's
            # default 640×480 size and draw on top of each other.
            if self.isVisible():
                row_host.show()
        self.actions_host.setVisible(bool(actions))
        self.update_actions()

    def _emit_action(self, action_id: str) -> None:
        if self._on_action is not None:
            self._on_action(action_id)

    def _sync_mask_amount(self) -> None:
        """Match the amount slider to the selected mask effect.

        Solid has no strength, so that row is hidden. Every other effect has
        its own label and range. The stored amount is clamped before the mask
        layer is updated, so a blur radius cannot become an oversized mosaic block.
        """
        kind_spec = self._specs.get("mask_kind")
        if not kind_spec or not kind_spec.get("amounts"):
            return
        kind = str(self.session.option("mask_kind", "pixelate"))
        amount_key = str(kind_spec.get("amount_key", "mask_amount"))
        widget = self._widgets.get(amount_key)
        label = self._labels.get(amount_key)
        amount_spec = self._specs.get(amount_key)
        if widget is None or label is None or amount_spec is None:
            return
        effect = kind_spec["amounts"].get(kind)
        if effect is None:
            self.form.setRowVisible(widget, False)
            return
        self.form.setRowVisible(widget, True)
        low = int(effect["min"])
        high = int(effect["max"])
        amount_spec["label"] = str(effect["label"])
        amount_spec["min"] = low
        amount_spec["max"] = high
        clamped = int(min(high, max(low, round(float(self.session.option(amount_key, effect["default"]))))))
        self.session.options[amount_key] = clamped
        if isinstance(widget, QSlider):
            widget.blockSignals(True)
            widget.setMinimum(low)
            widget.setMaximum(high)
            widget.setValue(clamped)
            widget.blockSignals(False)
        self._set_slider_label(amount_key, clamped)

    def _set_slider_label(self, key: str, value: int) -> None:
        label = self._labels.get(key)
        if label is None:
            return
        spec = self._specs.get(key, {})
        name = str(spec.get("label") or label.text().split("  ")[0])
        suffix = str(spec.get("suffix", ""))
        label.setText(f"{name}  {int(value)}{suffix}")

    def _apply_text_style(self, *, transient: bool) -> None:
        if self._loading:
            return
        self.session.apply_text_style(transient=transient)

    def _make(self, spec: dict) -> QWidget:
        key = spec["key"]
        kind = spec["type"]
        if kind == "font":
            return self._make_font(key)
        if kind == "font_style":
            return self._make_font_style()
        if kind == "spin":
            return self._make_spin(spec)
        if kind == "int":
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(int(spec["min"]))
            slider.setMaximum(int(spec["max"]))
            stored = self.session.option(key, spec.get("default", spec["min"]))
            shown = wrap_degrees(float(stored)) if key == "text_rotation" else int(stored)
            slider.setValue(shown)

            def on_change(value, key=key, spec=spec, slider=slider):
                # Building the rotation slider shows a wrapped angle. Writing
                # that back would change a layer that the user has not touched.
                if key == "text_rotation" and self._loading:
                    self._set_slider_label(key, int(value))
                    return
                self.session.options[key] = int(value)
                self._set_slider_label(key, int(value))
                # Only the live drag updates a redaction layer. Building the
                # slider, or releasing it, must not start a half-finished edit.
                if self._loading:
                    return
                if key == "mask_amount" and slider.isSliderDown():
                    self.session.set_redaction_settings(
                        str(self.session.option("mask_kind", "pixelate")),
                        float(value),
                        transient=True,
                    )
                if key in ("text_rotation", "text_line_spacing"):
                    # Mouse-down previews. A keyboard nudge is not a drag, so
                    # it commits here. Release commits the mouse drag once.
                    self._apply_text_style(transient=slider.isSliderDown())
                if key in ("brush_size", "brush_hardness", "brush_opacity"):
                    self.refresh_brush_preview()
                if spec.get("live") == "color":
                    self.session.preview_color_adjust()
                    self.update_actions()
                if self._on_option is not None:
                    self._on_option(key, int(value))

            def on_release(key=key):
                if key == "mask_amount":
                    self.session.set_redaction_settings(
                        str(self.session.option("mask_kind", "pixelate")),
                        float(self.session.option(key, 12)),
                        transient=False,
                    )
                if key in ("text_rotation", "text_line_spacing"):
                    self._apply_text_style(transient=False)

            slider.valueChanged.connect(on_change)
            slider.sliderReleased.connect(on_release)
            if key == "text_rotation":
                self._set_slider_label(key, slider.value())
            else:
                on_change(slider.value())
            return slider
        if kind == "choice":
            combo = QComboBox()
            for choice in spec["choices"]:
                if isinstance(choice, (tuple, list)):
                    combo.addItem(str(choice[1]), str(choice[0]))
                else:
                    combo.addItem(str(choice), str(choice))
            current = str(self.session.option(key, spec.get("default")))
            index = combo.findData(current)
            combo.setCurrentIndex(max(0, index))

            def on_choice(index, key=key, combo=combo, spec=spec):
                if self._loading or index < 0:
                    return
                value = combo.itemData(index)
                self.session.options[key] = value
                if key == "text_align":
                    self._apply_text_style(transient=False)
                if spec.get("amounts"):
                    # Clamp first, then commit the effect once. The slider's
                    # own handler must not also commit, so its signals stay blocked.
                    self._sync_mask_amount()
                    amount_key = str(spec.get("amount_key", "mask_amount"))
                    color = None
                    if value == "solid":
                        color = tuple(int(channel) for channel in self.session.foreground)
                    self.session.set_redaction_settings(
                        str(value),
                        float(self.session.option(amount_key, 12)),
                        transient=False,
                        color=color,
                    )

            combo.currentIndexChanged.connect(on_choice)
            return combo
        if kind == "bool":
            box = QCheckBox()
            box.setChecked(bool(self.session.option(key, spec.get("default", False))))

            def on_check(checked, key=key):
                if self._loading:
                    return
                self.session.options[key] = bool(checked)

            box.toggled.connect(on_check)
            return box
        return QLabel(str(self.session.option(key, "")))

    def refresh_brush_preview(self) -> None:
        """Redraw the stamp from the current size, hardness, opacity, and color."""
        label = self.brush_preview
        if label is None:
            return
        color = tuple(int(channel) for channel in self.session.foreground)
        image = brush_preview_image(
            int(self.session.option("brush_size", 24)),
            float(self.session.option("brush_hardness", 85)) / 100.0,
            float(self.session.option("brush_opacity", 100)) / 100.0,
            color,
            erase=self._brush_erase,
        )
        label.setPixmap(pil_to_pixmap(image))

    def _install_brush_preview(self, erase: bool) -> None:
        self._brush_erase = erase
        label = QLabel()
        label.setObjectName("brushPreview")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if erase:
            label.setToolTip("The hole this eraser removes. A large size fills the box.")
        else:
            label.setToolTip("One stamp at the current size, hardness, and opacity. A large brush fills this box.")
        self.form.addRow(label)
        self.brush_preview = label

    def _load_font_catalog(self) -> None:
        self._font_catalog = {family: list(styles) for family, styles in group_fonts()}
        self._font_by_path = {}
        for family, styles in self._font_catalog.items():
            for style, path in styles:
                self._font_by_path[path] = (family, style)

    def _ensure_font_path(self, combo: QComboBox, path, family_hint: str) -> None:
        """Keep a project font that is not installed as its own family row."""
        if not path or path in self._font_by_path:
            return
        family = family_hint or str(path).rsplit("/", 1)[-1]
        styles = self._font_catalog.setdefault(family, [])
        style = "Regular"
        if any(name == style for name, _item_path in styles):
            style = str(path).rsplit("/", 1)[-1]
        styles.append((style, path))
        self._font_by_path[path] = (family, style)
        if combo.findData(family) < 0:
            combo.addItem(family, family)

    def _path_for_style(self, family: str | None, style_name: str) -> str | None:
        """Pick a file in ``family``. Keep ``style_name`` when that cut exists.

        Otherwise Regular, then the first cut. Default (no family) has no file.
        """
        if not family:
            return None
        styles = self._font_catalog.get(family, [])
        if not styles:
            return None
        for name, path in styles:
            if name == style_name:
                return path
        for preferred in ("Regular", "Normal", "Roman", "Book"):
            for name, path in styles:
                if name.lower() == preferred.lower():
                    return path
        return styles[0][1]

    def _fill_style_combo(self, combo: QComboBox, family: str | None, path) -> None:
        combo.blockSignals(True)
        combo.clear()
        styles = self._font_catalog.get(family, []) if family else []
        if not styles:
            combo.addItem("Regular", None)
        else:
            for name, style_path in styles:
                combo.addItem(name, style_path)
        index = combo.findData(path)
        combo.setCurrentIndex(max(0, index))
        combo.blockSignals(False)

    def _family_for_path(self, path) -> str | None:
        if not path:
            return None
        found = self._font_by_path.get(path)
        return None if found is None else found[0]

    def _make_font(self, key: str) -> QComboBox:
        self._load_font_catalog()
        combo = QComboBox()
        combo.addItem("Default", None)
        for family in self._font_catalog:
            combo.addItem(family, family)
        self._select_font(combo, self.session.option(key))

        def on_font(index, combo=combo, key=key):
            # Keep the current cut when the new family has it. One commit
            # covers the family and the file the style menu lands on.
            if self._loading or index < 0:
                return
            family = combo.itemData(index)
            style_combo = self._widgets.get("text_style")
            style_name = "Regular"
            if isinstance(style_combo, QComboBox) and style_combo.count():
                style_name = style_combo.currentText()
            path = self._path_for_style(family, style_name)
            self.session.options[key] = path
            self.session.options["text_font_family"] = family or "Default"
            if isinstance(style_combo, QComboBox):
                self._fill_style_combo(style_combo, family, path)
            self._apply_text_style(transient=False)

        combo.currentIndexChanged.connect(on_font)
        return combo

    def _make_font_style(self) -> QComboBox:
        if not self._font_catalog and not self._font_by_path:
            self._load_font_catalog()
        combo = QComboBox()
        path = self.session.option("text_font")
        self._reload_style_for(combo, path)

        def on_style(index, combo=combo):
            if self._loading or index < 0:
                return
            path = combo.itemData(index)
            self.session.options["text_font"] = path
            family = self._family_for_path(path)
            if family:
                self.session.options["text_font_family"] = family
            self._apply_text_style(transient=False)

        combo.currentIndexChanged.connect(on_style)
        return combo

    def _reload_style_for(self, combo: QComboBox, path) -> None:
        family_hint = str(self.session.option("text_font_family") or "")
        font_combo = self._widgets.get("text_font")
        if isinstance(font_combo, QComboBox):
            font_combo.blockSignals(True)
            self._ensure_font_path(font_combo, path, family_hint)
            family = self._family_for_path(path)
            index = font_combo.findData(family)
            if index >= 0:
                font_combo.setCurrentIndex(index)
            font_combo.blockSignals(False)
        family = self._family_for_path(path)
        self._fill_style_combo(combo, family, path if family else None)

    def _reload_style_combo(self, combo: QComboBox) -> None:
        self._reload_style_for(combo, self.session.option("text_font"))

    def _select_font(self, combo: QComboBox, path) -> None:
        combo.blockSignals(True)
        self._ensure_font_path(combo, path, str(self.session.option("text_font_family") or ""))
        family = self._family_for_path(path)
        index = combo.findData(family)
        combo.setCurrentIndex(max(0, index))
        combo.blockSignals(False)
        style = self._widgets.get("text_style")
        if isinstance(style, QComboBox):
            self._fill_style_combo(style, family, path if family else None)

    def _make_spin(self, spec: dict) -> QSpinBox:
        key = spec["key"]
        box = QSpinBox()
        box.setRange(int(spec["min"]), int(spec["max"]))
        box.setKeyboardTracking(False)
        box.setValue(int(self.session.option(key, spec.get("default", spec["min"]))))

        def on_spin(value, key=key):
            if self._loading:
                return
            self.session.options[key] = int(value)
            if key == "text_size":
                self._apply_text_style(transient=False)

        box.valueChanged.connect(on_spin)
        return box


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


class LayerList(QListWidget):
    """The layer rows. Dragging one row restacks it.

    ``startDrag`` does not return until the drop or cancel has finished, so
    ``drag_finished`` is emitted outside Qt's nested drag loop. The panel
    updates the document from that signal, not from inside the drop.
    """

    drag_started = pyqtSignal()
    drag_finished = pyqtSignal()

    def startDrag(self, supported_actions) -> None:
        self.drag_started.emit()
        super().startDrag(supported_actions)
        self.drag_finished.emit()


class LayerPanel(QWidget):
    def __init__(self, session, on_rename) -> None:
        super().__init__()
        self.session = session
        self.on_rename = on_rename
        self._building = False
        # True from the drop until the document matches the list, so a
        # selection change during the drag does not pick the wrong layer.
        self._reordering = False
        self._pending_visual_ids: list | None = None
        self._opacity_before: float | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        title = QLabel("Layers")
        title.setObjectName("section")
        layout.addWidget(title)
        self.list = LayerList()
        self.list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.list.setDropIndicatorShown(True)
        self.list.setToolTip("Drag a row to change stacking order. The top row is in front.")
        self.list.currentRowChanged.connect(self._on_row)
        self.list.itemChanged.connect(self._on_item_changed)
        self.list.itemDoubleClicked.connect(lambda _item: self.on_rename())
        self.list.drag_started.connect(self._on_drag_started)
        # Queued so a rebuild of the rows waits until the drag's mouse
        # handling has returned, not while Qt is still inside it.
        self.list.drag_finished.connect(self._on_drag_finished, Qt.ConnectionType.QueuedConnection)
        layout.addWidget(self.list, 1)

        buttons = QHBoxLayout()
        self._add_button(buttons, "New", self.session.add_blank_layer)
        self._add_button(buttons, "Copy", self.session.duplicate_active)
        self._add_button(buttons, "Delete", self.session.delete_active)
        layout.addLayout(buttons)
        buttons2 = QHBoxLayout()
        self._add_button(buttons2, "Up", lambda: self.session.move_active(1))
        self._add_button(buttons2, "Down", lambda: self.session.move_active(-1))
        self._add_button(
            buttons2,
            "Merge down",
            self.session.merge_down_active,
            "Bake this layer into the one below it. Other layers stay as they are. "
            "Flatten image, in the Layer menu, collapses the whole stack.",
        )
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

    def _add_button(self, layout: QHBoxLayout, text: str, callback, tip: str = "") -> None:
        button = QPushButton(text)
        if tip:
            button.setToolTip(tip)
        button.clicked.connect(callback)
        layout.addWidget(button)

    def refresh(self) -> None:
        self._building = True
        # Block the list widget, not its model.
        # 1. clear() and addItem() would otherwise emit currentRowChanged and
        #    itemChanged, which this panel treats as a click or a visibility toggle.
        # 2. The view lays rows out only from the model's reset and insert
        #    signals. Blocking those leaves the default Background row, and
        #    every layer opened later, in the model but not on screen.
        self.list.blockSignals(True)
        self.list.clear()
        layers = self.session.document.layers
        # The list shows the front of the stack first, which is the last model entry.
        for layer in reversed(layers):
            item = QListWidgetItem(self._label(layer))
            item.setData(Qt.ItemDataRole.UserRole, layer.id)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsDragEnabled
                | Qt.ItemFlag.ItemIsDropEnabled
            )
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

    def _on_drag_started(self) -> None:
        self._reordering = True

    def _on_drag_finished(self) -> None:
        if self._building:
            self._reordering = False
            return
        # The list is already in the dropped order. The document still has
        # the previous order until the apply below.
        self._pending_visual_ids = [
            self.list.item(index).data(Qt.ItemDataRole.UserRole) for index in range(self.list.count())
        ]
        self._reordering = True
        self._apply_pending_drop()

    def _apply_pending_drop(self) -> None:
        visual_ids = self._pending_visual_ids
        self._pending_visual_ids = None
        try:
            if not visual_ids:
                return
            model_ids = [layer.id for layer in self.session.document.layers]
            move = stack_move(model_ids, visual_ids)
            if move is not None:
                old_index, new_index = move
                self.session.move_layer(old_index, new_index)
        finally:
            self._reordering = False
            # Rebuild from the document so a rejected drop snaps back, and
            # labels stay in sync when no window listener refreshes us.
            self.refresh()

    def _on_row(self, row: int) -> None:
        if self._building or self._reordering or row < 0:
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

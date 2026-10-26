"""Tool options beside the canvas.

The form is built from the active tool's option list. It writes session
options and does not touch pixels itself.
"""

from __future__ import annotations

from PyQt6.QtCore import QSize, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pixedit.paint import brush_preview_image
from pixedit.textutil import group_fonts
from pixedit.ui.canvas import pil_to_pixmap
from pixedit.ui.icons import icon_named


def wrap_degrees(degrees: float) -> int:
    """Nearest angle in -180..180. 200° and -160° are the same turn."""
    wrapped = (float(degrees) + 180.0) % 360.0 - 180.0
    rounded = int(round(wrapped))
    if rounded > 180:
        rounded -= 360
    elif rounded < -180:
        rounded += 360
    return rounded


def _track_column(widget: QWidget) -> QWidget:
    """Let the form assign the field width, instead of the control's contents.

    A Preferred or Minimum policy keeps the widget at its own hint. The
    longest font name, or a spin box wide enough for three digits, then
    becomes the options panel's minimum and sticks out of the tool column.
    Ignored drops that hint so the field column can shrink and grow with
    the splitter. The vertical policy stays, so a slider does not grow tall.
    """
    widget.setSizePolicy(QSizePolicy.Policy.Ignored, widget.sizePolicy().verticalPolicy())
    widget.setMinimumWidth(0)
    return widget


class ColumnComboBox(QComboBox):
    """Closed box follows the tool column. The open list stays readable.

    The default combo sizes itself to its longest entry. A font family is
    wider than the tool column, and that hint was locking the options panel.
    The closed box uses a short minimum instead. Opening the list widens it
    to the longest entry so the name can still be read.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(0)
        _track_column(self)
        self.currentTextChanged.connect(self._sync_tooltip)

    def setCurrentIndex(self, index: int) -> None:
        super().setCurrentIndex(index)
        # A stored font is applied with signals blocked, so the text signal
        # does not run. The tip still has to name what the closed box shows.
        self._sync_tooltip(self.currentText())

    def _sync_tooltip(self, text: str) -> None:
        self.setToolTip(text)

    def showPopup(self) -> None:
        view = self.view()
        text_width = view.sizeHintForColumn(0)
        # The list scrolls, so the longest name needs the bar's width too.
        # Without it the last characters sit under the scrollbar.
        needed = text_width + view.verticalScrollBar().sizeHint().width()
        if text_width > 0:
            view.setMinimumWidth(needed)
        super().showPopup()
        popup = view.window()
        # An empty menu never opens. The view's window is then this window,
        # and resizing it would change the tool column.
        if popup is None or popup is self.window() or needed <= self.width():
            return
        frame = popup.frameWidth() * 2
        popup.resize(max(popup.width(), needed + frame), popup.height())


# Command icons use the full width of the tool column. Choice icons share
# that row with their own caption, and grow up to the same 44px square.
# 32px is the old field size. 28px is the smallest square that still keeps
# four pictures beside "Mode" on one row when the tool column is at its
# 228px minimum and the options scrollbar is showing.
_ACTION_SIDE = 44
_ACTION_ICON = 26
_CHOICE_MIN = 32
_CHOICE_MAX = 44
_CHOICE_WRAP = 28
_CHOICE_ICON_MIN = 20
_CHOICE_ICON_MAX = 26
_CHOICE_GAP = 4
_LABEL_GAP = 6


def _choice_span(count: int, side: int) -> int:
    """Pixels covered by ``count`` squares of ``side``, including the gaps."""
    if count <= 0:
        return 0
    return count * side + (count - 1) * _CHOICE_GAP


def _choice_icon_px(side: int) -> int:
    """Glyph grows with the square, from 20px at 32px up to 26px at 44px."""
    if side <= _CHOICE_MIN:
        return _CHOICE_ICON_MIN
    if side >= _CHOICE_MAX:
        return _CHOICE_ICON_MAX
    span = _CHOICE_MAX - _CHOICE_MIN
    grown = (side - _CHOICE_MIN) / span
    return int(round(_CHOICE_ICON_MIN + grown * (_CHOICE_ICON_MAX - _CHOICE_ICON_MIN)))


def _icon_button(label: str, icon_name: str, tip: str = "", *, side: int, icon_px: int) -> QToolButton:
    """Square picture button. The words stay in the tooltip and the accessible name."""
    button = QToolButton()
    button.setObjectName("iconButton")
    button.setIcon(icon_named(icon_name))
    button.setIconSize(QSize(icon_px, icon_px))
    button.setFixedSize(side, side)
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
    button.setAutoRaise(False)
    button.setAccessibleName(label)
    button.setToolTip(tip or label)
    button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return button


class IconChoice(QWidget):
    """Exclusive picture buttons. The stored value is the choice id.

    The row spans the options panel. The caption is only as wide as its own
    text, and the squares sit in the space immediately to its right. They
    grow together up to the command-button size while they still share one
    row. A long slider label elsewhere in the form cannot push them apart.
    """

    changed = pyqtSignal(str)

    def __init__(self, caption: str, choices: list[tuple[str, str, str]]) -> None:
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.label = QLabel(caption, self)
        self._buttons: list[QToolButton] = []
        self._by_value: dict[str, QToolButton] = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for value, label, icon_name in choices:
            button = _icon_button(label, icon_name, side=_CHOICE_MIN, icon_px=_CHOICE_ICON_MIN)
            button.setParent(self)
            button.setCheckable(True)
            button.setProperty("choice", value)
            self._buttons.append(button)
            self._by_value[value] = button
            self._group.addButton(button)
        self._group.buttonClicked.connect(self._emit)
        _track_column(self)
        self.setMinimumHeight(_CHOICE_MIN)

    def _emit(self, button: QAbstractButton) -> None:
        # A second click on the current picture must not clear the choice.
        if not button.isChecked():
            self._group.blockSignals(True)
            button.setChecked(True)
            self._group.blockSignals(False)
        self.changed.emit(str(button.property("choice")))

    def labels(self) -> list[str]:
        return [button.accessibleName() for button in self._buttons]

    def value(self) -> str:
        button = self._group.checkedButton()
        if button is None:
            return ""
        return str(button.property("choice"))

    def set_value(self, value: str) -> None:
        button = self._by_value.get(str(value))
        if button is None:
            return
        self._group.blockSignals(True)
        button.setChecked(True)
        self._group.blockSignals(False)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._height_for(width)

    def minimumSizeHint(self) -> QSize:
        return QSize(0, self._height_for(self.width()))

    def sizeHint(self) -> QSize:
        return QSize(0, self._height_for(self.width()))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place()

    def _caption_width(self) -> int:
        self.label.ensurePolished()
        return max(0, self.label.sizeHint().width())

    def _metrics(self, width: int) -> tuple[int, int, int, int]:
        """Return ``(side, columns, button_x, label_y_beside)``.

        Purpose: keep the caption next to the pictures, on one row when the
        column allows.

        1. The caption takes only its own text width. Squares use what remains.
        2. Grow every square up to 44px while they still share one row.
        3. Stay on that one row down to 28px. The minimum tool column, once
           the scrollbar is taken out, is about that wide for four pictures.
        4. Below 28px, wrap 32px squares in the remaining width.
        5. If one 32px square does not fit beside the caption, the caption
           moves above and the squares use the full row.

        ``label_y_beside`` is 0 when the caption sits on the first button row,
        and -1 when the caption is on its own line above the squares.
        """
        count = len(self._buttons)
        # No width yet: report one row, so the size hint does not stack the squares.
        if count == 0 or width <= 0:
            return _CHOICE_MIN, max(count, 1), 0, 0
        caption = self._caption_width()
        gap = _LABEL_GAP if caption else 0
        room = width - caption - gap
        if room >= _CHOICE_MIN:
            gaps = (count - 1) * _CHOICE_GAP
            side = (room - gaps) // count
            if side >= _CHOICE_WRAP:
                return min(side, _CHOICE_MAX), count, caption + gap, 0
            columns = 1
            while columns < count and _choice_span(columns + 1, _CHOICE_MIN) <= room:
                columns += 1
            return _CHOICE_MIN, columns, caption + gap, 0
        columns = 1
        while columns < count and _choice_span(columns + 1, _CHOICE_MIN) <= width:
            columns += 1
        return _CHOICE_MIN, max(1, columns), 0, -1

    def _height_for(self, width: int) -> int:
        count = len(self._buttons)
        if count == 0:
            return _CHOICE_MIN
        # The first hint arrives before the form assigns a width. 44px is the
        # tallest square, so that pass cannot clip a button that later grows.
        if width <= 0:
            return _CHOICE_MAX
        side, columns, _button_x, label_above = self._metrics(width)
        rows = (count + columns - 1) // columns
        buttons = _choice_span(rows, side)
        if label_above:
            caption_h = max(self.label.sizeHint().height(), 0)
            return caption_h + _CHOICE_GAP + buttons
        return max(buttons, self.label.sizeHint().height(), _CHOICE_MIN)

    def _place(self) -> None:
        count = len(self._buttons)
        width = self.width()
        # Before the form assigns a row width there is nothing to measure against.
        if count == 0 or width <= 0:
            return
        side, columns, button_x, label_above = self._metrics(width)
        icon_px = _choice_icon_px(side)
        caption_w = self._caption_width()
        caption_h = self.label.sizeHint().height()
        if label_above:
            self.label.setGeometry(0, 0, caption_w, caption_h)
            button_y = caption_h + _CHOICE_GAP
        else:
            button_y = 0
            self.label.setGeometry(0, max(0, (side - caption_h) // 2), caption_w, caption_h)
        self.label.show()
        rows = (count + columns - 1) // columns
        needed = button_y + _choice_span(rows, side)
        if not label_above:
            needed = max(needed, caption_h, _CHOICE_MIN)
        step = side + _CHOICE_GAP
        for index, button in enumerate(self._buttons):
            button.setIconSize(QSize(icon_px, icon_px))
            button.setFixedSize(side, side)
            button.setGeometry(button_x + (index % columns) * step, button_y + (index // columns) * step, side, side)
            button.show()
        if self.minimumHeight() != needed:
            self.setMinimumHeight(needed)
            self.updateGeometry()


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
        self.action_buttons: list[QAbstractButton] = []
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
        # Vertical minimum stays so a short viewport scrolls instead of
        # squashing the rows. The font menu may be narrower than its longest
        # name; that name is readable in the open list.
        outer.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.heading = QLabel("Tool")
        self.heading.setObjectName("section")
        self.hint = QLabel("")
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        self.hint.hide()
        outer.addWidget(self.heading)
        self.form_host = QWidget()
        self.form = QFormLayout(self.form_host)
        self.form.setContentsMargins(0, 4, 0, 0)
        self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.form.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        outer.addWidget(self.form_host)
        self.actions_host = QWidget()
        self.actions_layout = QVBoxLayout(self.actions_host)
        self.actions_layout.setContentsMargins(0, 4, 0, 0)
        self.actions_layout.setSpacing(6)
        self.actions_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        outer.addWidget(self.actions_host)
        # Spare height stays between the controls and the description, so
        # sliders and buttons stay at the top. A short viewport collapses
        # this gap and the description is the last thing you scroll to.
        outer.addStretch(1)
        outer.addWidget(self.hint)

    def set_tool(self, tool) -> None:
        self.brush_preview = None
        self._brush_erase = False
        self._loading = True
        try:
            self.heading.setText(tool.label)
            self.hint.setText(tool.hint or "")
            self.hint.setVisible(bool(tool.hint))
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
                if isinstance(widget, IconChoice):
                    # The caption is inside the row. A spanning row keeps a long
                    # slider label from widening the gap in front of the pictures.
                    self.form.addRow(widget)
                    label = widget.label
                else:
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
            self._sync_crop_ratio()
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
                elif isinstance(widget, IconChoice):
                    widget.set_value("" if value is None else str(value))
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
        self._sync_crop_ratio()

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

    def _make_action_button(self, spec: dict) -> QAbstractButton:
        label = str(spec["label"])
        shortcut = spec.get("shortcut")
        tip = f"{label} ({shortcut})" if shortcut else label
        icon_name = spec.get("icon")
        if icon_name:
            button: QAbstractButton = _icon_button(label, str(icon_name), tip, side=_ACTION_SIDE, icon_px=_ACTION_ICON)
            button.icon_only = True
        else:
            # Ignored horizontally so a pair does not force the panel to the sum
            # of the two labels. The row still gives each button its text width
            # through the stretch factor below.
            text_button = QPushButton(label)
            text_button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            text_button.setToolTip(tip)
            text_button.setAccessibleName(label)
            text_button.icon_only = False
            button = text_button
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
        # Text commands stop at two per row, and only when both full labels
        # fit. A third label would be crushed, and one label wider than the
        # column still gets its own row so the splitter cannot clip it.
        # Icon buttons have no label width, so they keep packing until the
        # next square would pass the visible column. new_row starts a row
        # anyway, which keeps Scale and Rotate under the quarter-turns.
        gap = 6
        pending: list[tuple[QAbstractButton, int, bool]] = []
        for spec in actions:
            button = self._make_action_button(spec)
            button.setParent(self)
            button.ensurePolished()
            if getattr(button, "icon_only", False):
                width = _ACTION_SIDE
            else:
                width = button.sizeHint().width()
            pending.append((button, width, bool(spec.get("new_row"))))
        rows: list[list[tuple[QAbstractButton, int]]] = []
        current: list[tuple[QAbstractButton, int]] = []
        used = 0
        for button, width, new_row in pending:
            extra = gap if current else 0
            row_has_text = any(not getattr(placed, "icon_only", False) for placed, _width in current)
            this_is_text = not getattr(button, "icon_only", False)
            text_capped = bool(current) and len(current) >= 2 and (this_is_text or row_has_text)
            overflow = bool(current) and used + extra + width > budget
            if current and (new_row or text_capped or overflow):
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
            icons = all(getattr(button, "icon_only", False) for button, _width in row_buttons)
            for button, width in row_buttons:
                widest = max(widest, width)
                tallest = max(tallest, button.sizeHint().height())
                # Squares stay at _ACTION_SIDE. Text buttons share the row by label width.
                row.addWidget(button, 0 if icons else max(1, width))
                self.action_buttons.append(button)
            # A trailing stretch keeps the squares together. Without it, Qt
            # pours the spare column width into the gaps and the icons drift apart.
            if icons:
                row.addStretch(1)
            # Text floor is the wider label, not the sum. The sum would lock
            # the options widget at the two-column width and keep the right
            # button under the splitter. Icon squares cannot shrink, so their
            # floor is the whole row; four of them are still narrower than
            # the tool column. The height floor keeps an Ignored button
            # from collapsing the row.
            if icons:
                row_host.setMinimumWidth(sum(width for _button, width in row_buttons) + gap * (len(row_buttons) - 1))
            else:
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

    def _sync_crop_ratio(self) -> None:
        """Custom is the only proportion that needs its own width and height."""
        if "crop_ratio" not in self._specs:
            return
        custom = str(self.session.option("crop_ratio", "free")) == "custom"
        for key in ("crop_ratio_w", "crop_ratio_h"):
            widget = self._widgets.get(key)
            if widget is not None:
                self.form.setRowVisible(widget, custom)

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
            return _track_column(slider)
        if kind == "choice" and spec.get("icons"):
            return self._make_icon_choice(spec)
        if kind == "choice":
            combo = ColumnComboBox()
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
                if key == "crop_ratio":
                    self._sync_crop_ratio()
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
                # Shape, select, and mask change the canvas pointer.
                if self._on_option is not None:
                    self._on_option(key, value)

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

    def _make_icon_choice(self, spec: dict) -> IconChoice:
        choices = [(str(choice[0]), str(choice[1]), str(choice[2])) for choice in spec["choices"]]
        widget = IconChoice(str(spec["label"]), choices)
        widget.set_value(str(self.session.option(spec["key"], spec.get("default"))))
        key = str(spec["key"])

        def on_choice(value: str, key: str = key) -> None:
            # Checking the stored picture while the panel is built is not an edit.
            if self._loading:
                return
            self.session.options[key] = value
            if key == "text_align":
                self._apply_text_style(transient=False)
            # Shape, select, and mask change the canvas pointer.
            if self._on_option is not None:
                self._on_option(key, value)

        widget.changed.connect(on_choice)
        return widget

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
        combo = ColumnComboBox()
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
        combo = ColumnComboBox()
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
        return _track_column(box)

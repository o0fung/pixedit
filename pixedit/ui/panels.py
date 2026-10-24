"""Tool options and the layer stack. Both talk to the session, not to pixels directly."""

from __future__ import annotations

from PyQt6.QtCore import QSize, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLayout,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStyledItemDelegate,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pixedit.model import BLEND_MODES, layers_leaving, mask_target
from pixedit.paint import brush_preview_image
from pixedit.textutil import group_fonts
from pixedit.ui.canvas import pil_to_pixmap
from pixedit.ui.icons import icon_named
from pixedit.ui.thumbs import THUMB_SIZE, checker_image, layer_thumb, text_head


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

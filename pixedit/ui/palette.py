"""Foreground color palette for the lower-left column.

The square is saturation across and brightness down. The strip under it is
hue. Dragging writes ``session.foreground`` live and tells the window when
the drag is still moving, so a text layer can preview without an undo step.
Recent swatches update only when the drag, a typed hex value, or a
Color-tool pick finishes.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QSlider, QVBoxLayout, QWidget

_RECENT_LIMIT = 8


def parse_hex(text: str) -> tuple[int, int, int, int | None] | None:
    """Parse ``#RGB``, ``#RRGGBB``, or ``#RRGGBBAA``. Missing alpha is ``None``."""
    raw = text.strip()
    if raw.startswith("#"):
        raw = raw[1:]
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) not in (6, 8) or any(ch not in "0123456789abcdefABCDEF" for ch in raw):
        return None
    value = int(raw, 16)
    if len(raw) == 6:
        return (value >> 16) & 255, (value >> 8) & 255, value & 255, None
    return (value >> 24) & 255, (value >> 16) & 255, (value >> 8) & 255, value & 255


def _rgb_to_hsv(red: int, green: int, blue: int) -> tuple[float, float, float]:
    rn, gn, bn = red / 255.0, green / 255.0, blue / 255.0
    peak = max(rn, gn, bn)
    floor = min(rn, gn, bn)
    span = peak - floor
    if span <= 1e-8:
        hue = 0.0
    elif peak == rn:
        hue = ((gn - bn) / span) % 6.0
    elif peak == gn:
        hue = (bn - rn) / span + 2.0
    else:
        hue = (rn - gn) / span + 4.0
    hue = (hue / 6.0) % 1.0
    sat = 0.0 if peak <= 1e-8 else span / peak
    return hue, sat, peak


def _hsv_to_rgb(hue: float, sat: float, val: float) -> tuple[int, int, int]:
    """One HSV color, channels 0..255. ``hue`` is 0..1."""
    sector = (hue % 1.0) * 6.0
    index = int(sector) % 6
    frac = sector - int(sector)
    peak = val
    rest = val * (1.0 - sat)
    fade = val * (1.0 - frac * sat)
    rise = val * (1.0 - (1.0 - frac) * sat)
    table = (
        (peak, rise, rest),
        (fade, peak, rest),
        (rest, peak, rise),
        (rest, fade, peak),
        (rise, rest, peak),
        (peak, rest, fade),
    )
    red, green, blue = table[index]
    return int(round(red * 255)), int(round(green * 255)), int(round(blue * 255))


def _rgb_image(array: np.ndarray) -> QImage:
    height, width, _channels = array.shape
    packed = np.ascontiguousarray(array)
    image = QImage(packed.data, width, height, width * 3, QImage.Format.Format_RGB888)
    return image.copy()


def _sv_array(hue: float, width: int, height: int) -> np.ndarray:
    sat = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    val = np.linspace(1.0, 0.0, height, dtype=np.float32)[:, None]
    sector = (hue % 1.0) * 6.0
    index = int(sector) % 6
    frac = sector - int(sector)
    # val is one column. Mixing it with sat (one row) expands to the full
    # field. peak has to be expanded too, or stacking fails when the field
    # is wider than a single pixel.
    peak = np.broadcast_to(val, (height, width))
    rest = val * (1.0 - sat)
    fade = val * (1.0 - frac * sat)
    rise = val * (1.0 - (1.0 - frac) * sat)
    channels = {
        0: (peak, rise, rest),
        1: (fade, peak, rest),
        2: (rest, peak, rise),
        3: (rest, fade, peak),
        4: (rise, rest, peak),
        5: (peak, rest, fade),
    }[index]
    stacked = np.stack(channels, axis=-1)
    return np.clip(np.rint(stacked * 255.0), 0, 255).astype(np.uint8)


def _hue_array(width: int, height: int) -> np.ndarray:
    hues = np.linspace(0.0, 1.0, width, endpoint=False, dtype=np.float32)
    row = np.zeros((width, 3), dtype=np.uint8)
    for x, hue in enumerate(hues):
        row[x] = _hsv_to_rgb(float(hue), 1.0, 1.0)
    return np.broadcast_to(row, (height, width, 3)).copy()


class _Field(QWidget):
    """A painted color field. The palette turns a click into a foreground color."""

    def __init__(self, palette: ColorPalette, kind: str, height: int) -> None:
        super().__init__(palette)
        self.palette = palette
        self.kind = kind
        self._image: QImage | None = None
        self._pressed = False
        self.setFixedHeight(height)
        self.setMouseTracking(False)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_image(self, image: QImage) -> None:
        self._image = image
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.palette.rebuild_fields()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        if self._image is not None and not self._image.isNull():
            painter.drawImage(0, 0, self._image)
        else:
            painter.fillRect(self.rect(), QColor("#10141a"))
        if self.kind == "sv":
            x = self.palette.sat * max(0, self.width() - 1)
            y = (1.0 - self.palette.val) * max(0, self.height() - 1)
            self._draw_ring(painter, x, y)
        else:
            x = self.palette.hue * max(0, self.width() - 1)
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.drawLine(int(round(x)), 0, int(round(x)), self.height())

    def _draw_ring(self, painter: QPainter, x: float, y: float) -> None:
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#111111"), 1))
        painter.drawEllipse(QPointF(x, y), 6.0, 6.0)
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.drawEllipse(QPointF(x, y), 4.5, 4.5)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._pressed = True
        self.palette._drag = self.kind
        self._pick(event.position().x(), event.position().y(), transient=True)

    def mouseMoveEvent(self, event) -> None:
        if not self._pressed:
            return
        self._pick(event.position().x(), event.position().y(), transient=True)

    def mouseReleaseEvent(self, event) -> None:
        if not self._pressed:
            return
        self._pressed = False
        # Keep ``_drag`` set through the commit so the window's refresh does
        # not rebuild the field from a rounded color and jump the marker.
        self._pick(event.position().x(), event.position().y(), transient=False)
        self.palette._drag = None
        self.palette.note_recent()

    def _pick(self, x: float, y: float, *, transient: bool) -> None:
        if self.kind == "sv":
            width = max(1, self.width() - 1)
            height = max(1, self.height() - 1)
            sat = min(1.0, max(0.0, x / width))
            val = min(1.0, max(0.0, 1.0 - (y / height)))
            self.palette.set_hsv(self.palette.hue, sat, val, transient=transient)
            return
        width = max(1, self.width() - 1)
        hue = min(0.999, max(0.0, x / width))
        self.palette.set_hsv(hue, self.palette.sat, self.palette.val, transient=transient)


class ColorPalette(QWidget):
    def __init__(self, session, on_color=None) -> None:
        super().__init__()
        self.session = session
        self._on_color = on_color
        self.hue = 0.0
        self.sat = 0.0
        self.val = 0.0
        self._drag: str | None = None
        self._syncing = False
        self._recent: list[tuple[int, int, int, int]] = []
        self._sv_key: tuple[int, int, int] | None = None
        self._hue_key: int | None = None
        self.setObjectName("palette")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(4)
        self.sv_pad = _Field(self, "sv", 72)
        self.hue_pad = _Field(self, "hue", 12)
        layout.addWidget(self.sv_pad)
        layout.addWidget(self.hue_pad)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.chip = QWidget()
        self.chip.setFixedSize(28, 28)
        self.hex_edit = QLineEdit()
        self.hex_edit.setPlaceholderText("#RRGGBB")
        self.hex_edit.setMaxLength(9)
        self.hex_edit.setToolTip("Foreground color. Type #RGB, #RRGGBB, or #RRGGBBAA.")
        self.hex_edit.editingFinished.connect(self._commit_hex)
        row.addWidget(self.chip)
        row.addWidget(self.hex_edit, 1)
        layout.addLayout(row)

        self.opacity_label = QLabel("Opacity  100%")
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setToolTip("Opacity of the foreground color. Brush, shape, fill, and text use it.")
        self.opacity_slider.sliderPressed.connect(self._opacity_pressed)
        self.opacity_slider.valueChanged.connect(self._opacity_moved)
        self.opacity_slider.sliderReleased.connect(self._opacity_released)
        opacity_row = QHBoxLayout()
        opacity_row.setSpacing(6)
        opacity_row.addWidget(self.opacity_label)
        opacity_row.addWidget(self.opacity_slider, 1)
        layout.addLayout(opacity_row)

        recent_row = QHBoxLayout()
        recent_row.setSpacing(4)
        self.recent_buttons: list[QPushButton] = []
        for index in range(_RECENT_LIMIT):
            button = QPushButton()
            button.setFixedSize(22, 22)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip("Recent color")
            button.clicked.connect(lambda _checked=False, slot=index: self._pick_recent(slot))
            recent_row.addWidget(button)
            self.recent_buttons.append(button)
        recent_row.addStretch(1)
        layout.addLayout(recent_row)

        self.sync_from_session()
        self.note_recent()

    def sync_from_session(self) -> None:
        """Match the widgets to ``session.foreground`` without restyling a layer."""
        if self._drag:
            return
        red, green, blue, alpha = (int(channel) for channel in self.session.foreground)
        self._apply_rgb(red, green, blue, alpha, emit=False)

    def note_recent(self) -> None:
        """Remember the current foreground once a pick has finished."""
        color = tuple(int(channel) for channel in self.session.foreground)
        if len(color) != 4:
            return
        if self._recent and self._recent[0] == color:
            self._paint_recent()
            return
        self._recent = [color, *[item for item in self._recent if item != color]]
        del self._recent[_RECENT_LIMIT:]
        self._paint_recent()

    def set_hsv(self, hue: float, sat: float, val: float, *, transient: bool) -> None:
        # The caller passes the hue to keep. Dragging into gray still sends
        # the current hue, so the strip does not jump to red.
        self.hue = float(hue) % 1.0
        self.sat = min(1.0, max(0.0, float(sat)))
        self.val = min(1.0, max(0.0, float(val)))
        red, green, blue = _hsv_to_rgb(self.hue, self.sat, self.val)
        _red, _green, _blue, alpha = (int(channel) for channel in self.session.foreground)
        self.session.foreground = (red, green, blue, alpha)
        self._refresh()
        self._emit(transient=transient)

    def rebuild_fields(self) -> None:
        self._refresh_images(force=True)

    def _commit_hex(self) -> None:
        parsed = parse_hex(self.hex_edit.text())
        if parsed is None:
            self._show_hex()
            return
        red, green, blue, alpha = parsed
        if alpha is None:
            alpha = int(self.session.foreground[3])
        self._apply_rgb(red, green, blue, alpha, emit=True)
        self.note_recent()

    def _pick_recent(self, index: int) -> None:
        if index >= len(self._recent):
            return
        red, green, blue, alpha = self._recent[index]
        self._apply_rgb(red, green, blue, alpha, emit=True)
        self.note_recent()

    def _opacity_pressed(self) -> None:
        self._drag = "opacity"

    def _opacity_moved(self, percent: int) -> None:
        if self._syncing:
            return
        self._write_alpha(int(percent))
        # A keyboard nudge is not a drag, so it commits immediately. A mouse
        # drag previews and commits from sliderReleased.
        if self.opacity_slider.isSliderDown():
            self._emit(transient=True)
            return
        self._emit(transient=False)
        self.note_recent()

    def _opacity_released(self) -> None:
        if self._syncing:
            self._drag = None
            return
        self._write_alpha(int(self.opacity_slider.value()))
        self._emit(transient=False)
        self._drag = None
        self.note_recent()

    def _apply_rgb(self, red: int, green: int, blue: int, alpha: int, *, emit: bool) -> None:
        hue, sat, val = _rgb_to_hsv(red, green, blue)
        if sat > 0.001:
            self.hue = hue
        self.sat = sat
        self.val = val
        self.session.foreground = (int(red), int(green), int(blue), int(alpha))
        self._refresh()
        if emit:
            self._emit(transient=False)

    def _write_alpha(self, percent: int) -> None:
        red, green, blue, _alpha = (int(channel) for channel in self.session.foreground)
        alpha = int(round(percent * 255 / 100))
        self.session.foreground = (red, green, blue, alpha)
        self.opacity_label.setText(f"Opacity  {percent}%")
        self._show_hex()
        self._paint_chip()

    def _refresh(self) -> None:
        self._syncing = True
        percent = int(round(self.session.foreground[3] * 100 / 255))
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(percent)
        self.opacity_slider.blockSignals(False)
        self.opacity_label.setText(f"Opacity  {percent}%")
        self._syncing = False
        self._show_hex()
        self._paint_chip()
        self._refresh_images(force=False)
        self.sv_pad.update()
        self.hue_pad.update()

    def _show_hex(self) -> None:
        if self.hex_edit.hasFocus():
            return
        red, green, blue, alpha = (int(channel) for channel in self.session.foreground)
        text = f"#{red:02X}{green:02X}{blue:02X}" if alpha == 255 else f"#{red:02X}{green:02X}{blue:02X}{alpha:02X}"
        if self.hex_edit.text() != text:
            self.hex_edit.setText(text)

    def _paint_chip(self) -> None:
        red, green, blue, alpha = (int(channel) for channel in self.session.foreground)
        self.chip.setStyleSheet(
            f"background: rgba({red}, {green}, {blue}, {alpha}); border: 1px solid #343e4a; border-radius: 6px;"
        )

    def _paint_recent(self) -> None:
        for index, button in enumerate(self.recent_buttons):
            if index >= len(self._recent):
                button.setEnabled(False)
                button.setStyleSheet("background: #10141a; border: 1px solid #343e4a; padding: 0;")
                continue
            red, green, blue, alpha = self._recent[index]
            button.setEnabled(True)
            button.setStyleSheet(
                f"background: rgba({red}, {green}, {blue}, {alpha}); border: 1px solid #343e4a; padding: 0;"
            )

    def _refresh_images(self, *, force: bool) -> None:
        sv_width = max(1, self.sv_pad.width())
        sv_height = max(1, self.sv_pad.height())
        hue_bucket = int(round(self.hue * 360))
        sv_key = (sv_width, sv_height, hue_bucket)
        if force or sv_key != self._sv_key:
            if sv_width > 1 and sv_height > 1:
                self.sv_pad.set_image(_rgb_image(_sv_array(self.hue, sv_width, sv_height)))
                self._sv_key = sv_key
        hue_width = max(1, self.hue_pad.width())
        if force or hue_width != self._hue_key:
            if hue_width > 1:
                self.hue_pad.set_image(_rgb_image(_hue_array(hue_width, max(1, self.hue_pad.height()))))
                self._hue_key = hue_width

    def _emit(self, *, transient: bool) -> None:
        if self._on_color is not None:
            self._on_color(transient=transient)

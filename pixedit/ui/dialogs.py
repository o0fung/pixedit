"""Modal editors: new canvas, text, PDF import, live filter previews, and About."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pixedit import __version__

from pixedit.adjust import apply_masked
from pixedit.geom import (
    MAX_EDGE,
    inches_from_pixels,
    linked_dpi,
    linked_inches,
    linked_percent,
    linked_pixels,
    percent_from_pixels,
    placed_box,
    selection_for_layer,
)
from pixedit.proxy import downscale, preview_scale, set_preview_source
from pixedit.model import MutateLayerCommand, TextInfo, apply_visual, visual_state
from pixedit.presets import (
    PAPER_DPI,
    PRESETS,
    RESOLUTIONS,
    SCREEN_DPI,
    page_label,
    preset_by_id,
    preset_size,
)


class NewDocumentDialog(QDialog):
    """Blank canvas. A template fills the pixel size; typing a side leaves it.

    Custom starts at 1280 × 800 and 72 dpi, which is what OK does if nothing
    is changed. Paper templates (A4 and the other sheets) start at 300 dpi
    and follow orientation. Screen templates stay at their listed pixels and
    72 dpi. One lock covers the write-back: filling the spin boxes from a
    template must not look like a typed size and flip the template to Custom.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New canvas")
        self._lock = False
        # Tracks the last template kind so entering paper from a screen or
        # custom size can start at 300 dpi, while switching A4 to A3 keeps
        # a resolution the user already chose.
        self._kind = "custom"
        form = QFormLayout(self)
        self.template = QComboBox()
        for preset in PRESETS:
            self.template.addItem(preset.label, preset.id)
        self.orientation = QComboBox()
        self.orientation.addItem("Portrait", "portrait")
        self.orientation.addItem("Landscape", "landscape")
        self.orientation.setEnabled(False)
        self.resolution = QComboBox()
        for dpi in RESOLUTIONS:
            self.resolution.addItem(f"{dpi} dpi", dpi)
        self.width = QSpinBox()
        self.width.setRange(1, MAX_EDGE)
        self.width.setValue(1280)
        self.height = QSpinBox()
        self.height.setRange(1, MAX_EDGE)
        self.height.setValue(800)
        self.summary = QLabel("1280 × 800 px")
        self.summary.setWordWrap(True)
        self.background = QComboBox()
        self.background.addItems(["white", "black", "transparent"])
        form.addRow("Template", self.template)
        form.addRow("Orientation", self.orientation)
        form.addRow("Resolution", self.resolution)
        form.addRow("Width", self.width)
        form.addRow("Height", self.height)
        form.addRow(self.summary)
        form.addRow("Background", self.background)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.template.currentIndexChanged.connect(self._on_template)
        self.orientation.currentIndexChanged.connect(self._on_preset_option)
        self.resolution.currentIndexChanged.connect(self._on_preset_option)
        self.width.valueChanged.connect(self._on_typed_size)
        self.height.valueChanged.connect(self._on_typed_size)

    def values(self) -> tuple[int, int, str, float]:
        width = self.width.value()
        height = self.height.value()
        return width, height, self.background.currentText(), float(self.resolution.currentData())

    def _on_template(self, _index: int = 0) -> None:
        """Apply the template the user just picked.

        Paper entered from anywhere else starts at 300 dpi, then the page
        is converted to pixels. A later paper-to-paper change keeps the
        resolution already showing. Screen sizes force 72 dpi and ignore
        orientation. Choosing Custom restores 1280 × 800 at 72 dpi.
        """
        if self._lock:
            return
        preset = preset_by_id(self.template.currentData())
        previous = self._kind
        self._lock = True
        try:
            if preset.kind == "paper":
                self.orientation.setEnabled(True)
                self.resolution.setEnabled(True)
                if previous != "paper":
                    self._select_dpi(int(PAPER_DPI))
                self._fill_from_preset(preset.id)
            elif preset.kind == "screen":
                self.orientation.setEnabled(False)
                self.resolution.setEnabled(False)
                self._select_dpi(int(SCREEN_DPI))
                self._fill_from_preset(preset.id)
            else:
                self.orientation.setEnabled(False)
                self.resolution.setEnabled(True)
                self._select_dpi(int(SCREEN_DPI))
                width, height, _dpi = preset_size("custom", "portrait", SCREEN_DPI)
                self.width.setValue(width)
                self.height.setValue(height)
            self._kind = preset.kind
        finally:
            self._lock = False
        self._refresh_summary()

    def _on_preset_option(self, _index: int = 0) -> None:
        """Orientation or resolution changed.

        A paper page is measured again, so A4 at 72 dpi is the smaller pixel
        grid and landscape swaps the sides. Custom keeps the typed pixels and
        only the stored resolution changes. Screen presets disable both
        controls, so this does not resize them.
        """
        if self._lock:
            return
        if self._kind != "paper":
            self._refresh_summary()
            return
        self._lock = True
        try:
            self._fill_from_preset(self.template.currentData())
        finally:
            self._lock = False
        self._refresh_summary()

    def _on_typed_size(self, _value: int = 0) -> None:
        """A typed width or height wins over the template.

        The combo shows Custom, but the numbers and the resolution stay as
        they are. The lock keeps that combo change from restoring 1280 × 800.
        """
        if self._lock:
            return
        self._lock = True
        try:
            index = self.template.findData("custom")
            if index >= 0:
                self.template.setCurrentIndex(index)
            self.orientation.setEnabled(False)
            self.resolution.setEnabled(True)
            self._kind = "custom"
        finally:
            self._lock = False
        self._refresh_summary()

    def _fill_from_preset(self, preset_id: str) -> None:
        width, height, _dpi = preset_size(
            preset_id,
            self.orientation.currentData(),
            float(self.resolution.currentData()),
        )
        self.width.setValue(width)
        self.height.setValue(height)

    def _select_dpi(self, dpi: int) -> None:
        index = self.resolution.findData(dpi)
        if index >= 0:
            self.resolution.setCurrentIndex(index)

    def _refresh_summary(self) -> None:
        pixels = f"{self.width.value()} × {self.height.value()} px"
        if self._kind != "paper":
            self.summary.setText(pixels)
            return
        page = page_label(self.template.currentData(), self.orientation.currentData())
        side = self.orientation.currentData()
        name = preset_by_id(self.template.currentData()).label
        self.summary.setText(f"{name} {side} · {page} · {pixels}")


class PdfImportDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import PDF")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Each page is its own canvas, in one group. The canvas shows one page at a time."))
        form = QFormLayout()
        self.dpi = QComboBox()
        self.dpi.addItem("72 DPI  (screen)", 1.0)
        self.dpi.addItem("144 DPI  (sharp)", 2.0)
        self.dpi.addItem("216 DPI  (print)", 3.0)
        # 72 DPI is the default. 144 and 216 keep more pixels, and the file is larger.
        self.dpi.setCurrentIndex(0)
        form.addRow("Render", self.dpi)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def scale(self) -> float:
        return float(self.dpi.currentData())


class RasterSizeDialog(QDialog):
    """Choose how many pixels to keep when the file is larger than the screen."""

    def __init__(self, parent, width: int, height: int) -> None:
        super().__init__(parent)
        self.setWindowTitle("Open image")
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                f"This image is {width} × {height} px. "
                "A shorter edge uses less memory. Full size keeps every pixel."
            )
        )
        self.choice = QComboBox()
        self.choice.addItem("Full size", None)
        longest = max(int(width), int(height))
        for edge in (4096, 2048, 1024):
            if edge < longest:
                self.choice.addItem(f"Longest edge {edge} px", edge)
        layout.addWidget(self.choice)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def max_edge(self) -> int | None:
        value = self.choice.currentData()
        return None if value is None else int(value)


class TextDialog(QDialog):
    """The words only. Font, style, align, spacing, rotation, and color live on the tool panel."""

    def __init__(self, initial: TextInfo, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Text")
        self._initial = initial
        layout = QVBoxLayout(self)
        self.text = QPlainTextEdit()
        self.text.setPlainText(initial.text)
        self.text.setMinimumHeight(80)
        layout.addWidget(self.text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(420, 180)

    def result_info(self) -> TextInfo:
        style = self._initial
        return TextInfo(
            text=self.text.toPlainText(),
            font_path=style.font_path,
            font_family=style.font_family,
            size=style.size,
            color=style.color,
            rotation=style.rotation,
            align=style.align,
            line_spacing=style.line_spacing,
        )


# Row-major: corners, sides, and the center. The label is the tooltip.
_ANCHORS = (
    ("nw", "Top left"),
    ("n", "Top"),
    ("ne", "Top right"),
    ("w", "Left"),
    ("c", "Center"),
    ("e", "Right"),
    ("sw", "Bottom left"),
    ("s", "Bottom"),
    ("se", "Bottom right"),
)
_ANCHOR_CELL = 24
_ANCHOR_GAP = 1
# A local sheet, so the toolbar's rounded tool buttons do not restyle the cells.
_ANCHOR_STYLE = """
QWidget#anchorMatrix { background: #8ea0b0; }
QToolButton#anchorCell {
    background: #12161b;
    border: none;
    border-radius: 0;
    padding: 0;
    margin: 0;
}
QToolButton#anchorCell:hover { background: #1d4254; }
QToolButton#anchorCell:checked { background: #f4f7f8; }
QToolButton#anchorCell:checked:hover { background: #ffffff; }
"""


def _dialog_buttons(parent: QDialog) -> QDialogButtonBox:
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(parent.accept)
    buttons.rejected.connect(parent.reject)
    return buttons


class ProportionSizeForm(QWidget):
    """Percent and pixel size. 100% is the size this form was opened with.

    Keep proportions links the axes. A percent edit applies that percent to
    both. A pixel edit keeps the original aspect. One lock covers the write
    back, so updating the pixel field does not run again and replace a typed
    50% with the percent implied by a rounded pixel.
    """

    dpiEdited = pyqtSignal(float)

    def __init__(self, width: int, height: int, *, inches: bool = False, dpi: float = 72.0) -> None:
        super().__init__()
        self._width0 = max(1, int(width))
        self._height0 = max(1, int(height))
        self._dpi = max(1.0, float(dpi))
        self._resample = True
        self._lock = False
        self.width_percent = self._percent_box()
        self.height_percent = self._percent_box()
        self.width_px = self._pixel_box(self._width0)
        self.height_px = self._pixel_box(self._height0)
        self.width_in = self._inch_box() if inches else None
        self.height_in = self._inch_box() if inches else None
        self.keep = QCheckBox("Keep proportions")
        self.keep.setChecked(True)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Width", self._axis_row(self.width_percent, self.width_px, self.width_in))
        form.addRow("Height", self._axis_row(self.height_percent, self.height_px, self.height_in))
        form.addRow(self.keep)
        self._show_inches()
        self.width_percent.valueChanged.connect(lambda _value: self._on_percent("width"))
        self.height_percent.valueChanged.connect(lambda _value: self._on_percent("height"))
        self.width_px.valueChanged.connect(lambda _value: self._on_pixels("width"))
        self.height_px.valueChanged.connect(lambda _value: self._on_pixels("height"))
        if self.width_in is not None and self.height_in is not None:
            self.width_in.valueChanged.connect(lambda _value: self._on_inches("width"))
            self.height_in.valueChanged.connect(lambda _value: self._on_inches("height"))
        self.keep.toggled.connect(self._on_keep)

    def pixel_size(self) -> tuple[int, int]:
        return self.width_px.value(), self.height_px.value()

    @property
    def dpi(self) -> float:
        return self._dpi

    def set_resample(self, enabled: bool) -> None:
        """Percent and pixels edit the grid. Inches stay available either way.

        Turning resample off throws away a pending pixel edit and shows the
        size the document has now. Those fields are the file's pixels again.
        """
        self._resample = bool(enabled)
        for widget in (self.width_percent, self.height_percent, self.width_px, self.height_px):
            widget.setEnabled(self._resample)
        if not self._resample:
            self._show(self._width0, self._height0, 100.0, 100.0)

    def set_dpi(self, dpi: float, *, keep_inches: bool) -> None:
        if self._lock:
            return
        width_px, height_px, width_in, height_in, dpi = linked_dpi(
            self.width_px.value(),
            self.height_px.value(),
            self._dpi,
            dpi,
            resample=keep_inches and self._resample,
        )
        if keep_inches and self._resample:
            width_percent = percent_from_pixels(self._width0, width_px)
            height_percent = percent_from_pixels(self._height0, height_px)
        else:
            width_percent = self.width_percent.value()
            height_percent = self.height_percent.value()
        self._show(width_px, height_px, width_percent, height_percent, width_in, height_in, dpi)

    def _percent_box(self) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setDecimals(1)
        box.setRange(0.1, 1600000.0)
        box.setSingleStep(1.0)
        box.setSuffix(" %")
        box.setValue(100.0)
        box.setKeyboardTracking(False)
        return box

    def _pixel_box(self, value: int) -> QSpinBox:
        box = QSpinBox()
        box.setRange(1, 16000)
        box.setSuffix(" px")
        box.setValue(int(value))
        box.setKeyboardTracking(False)
        return box

    def _inch_box(self) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setDecimals(3)
        box.setRange(0.001, 16000.0)
        box.setSingleStep(0.1)
        box.setSuffix(" in")
        box.setKeyboardTracking(False)
        return box

    def _axis_row(self, percent: QWidget, pixels: QWidget, inches: QWidget | None) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(percent, 1)
        layout.addWidget(pixels, 1)
        if inches is not None:
            layout.addWidget(inches, 1)
        return row

    def _on_percent(self, axis: str) -> None:
        if self._lock:
            return
        width_px, height_px, width_percent, height_percent = linked_percent(
            self._width0,
            self._height0,
            self.width_percent.value(),
            self.height_percent.value(),
            axis=axis,
            keep=self.keep.isChecked(),
        )
        self._show(width_px, height_px, width_percent, height_percent)

    def _on_pixels(self, axis: str) -> None:
        if self._lock:
            return
        width_px, height_px, width_percent, height_percent = linked_pixels(
            self._width0,
            self._height0,
            self.width_px.value(),
            self.height_px.value(),
            axis=axis,
            keep=self.keep.isChecked(),
        )
        self._show(width_px, height_px, width_percent, height_percent)

    def _on_inches(self, axis: str) -> None:
        if self._lock or self.width_in is None or self.height_in is None:
            return
        width_px, height_px, width_in, height_in, dpi = linked_inches(
            self._width0,
            self._height0,
            self.width_px.value(),
            self.height_px.value(),
            self.width_in.value(),
            self.height_in.value(),
            self._dpi,
            axis=axis,
            keep=self.keep.isChecked(),
            resample=self._resample,
        )
        if self._resample:
            width_percent = percent_from_pixels(self._width0, width_px)
            height_percent = percent_from_pixels(self._height0, height_px)
        else:
            width_percent = self.width_percent.value()
            height_percent = self.height_percent.value()
        self._show(width_px, height_px, width_percent, height_percent, width_in, height_in, dpi)

    def _on_keep(self, checked: bool) -> None:
        # Turning the link on snaps height back to the original aspect.
        if self._lock or not checked:
            return
        self._on_pixels("width")

    def _show(
        self,
        width_px: int,
        height_px: int,
        width_percent: float,
        height_percent: float,
        width_in: float | None = None,
        height_in: float | None = None,
        dpi: float | None = None,
    ) -> None:
        self._lock = True
        try:
            self.width_px.setValue(int(width_px))
            self.height_px.setValue(int(height_px))
            self.width_percent.setValue(float(width_percent))
            self.height_percent.setValue(float(height_percent))
            if dpi is not None:
                self._dpi = float(dpi)
            if width_in is None or height_in is None:
                self._show_inches()
            elif self.width_in is not None and self.height_in is not None:
                self.width_in.setValue(float(width_in))
                self.height_in.setValue(float(height_in))
            if dpi is not None:
                self.dpiEdited.emit(self._dpi)
        finally:
            self._lock = False

    def _show_inches(self) -> None:
        if self.width_in is None or self.height_in is None:
            return
        self.width_in.setValue(inches_from_pixels(self.width_px.value(), self._dpi))
        self.height_in.setValue(inches_from_pixels(self.height_px.value(), self._dpi))


class AnchorPicker(QWidget):
    """3×3 anchor matrix. Corners, sides, and the center can each stay put.

    The filled cell is the selection. The other cells stay empty squares, in
    the same arrangement as Photoshop's canvas-size anchor.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("anchorMatrix")
        self.setStyleSheet(_ANCHOR_STYLE)
        self.setToolTip("The filled cell stays put. Pick a corner, a side, or the center.")
        side = _ANCHOR_CELL * 3 + _ANCHOR_GAP * 2
        self.setFixedSize(side, side)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(_ANCHOR_GAP)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for index, (key, tip) in enumerate(_ANCHORS):
            button = QToolButton()
            button.setObjectName("anchorCell")
            button.setCheckable(True)
            button.setText("")
            button.setToolTip(tip)
            button.setAccessibleName(tip)
            button.setFixedSize(_ANCHOR_CELL, _ANCHOR_CELL)
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            button.setProperty("anchor", key)
            button.setAutoRaise(False)
            if key == "c":
                button.setChecked(True)
            self._group.addButton(button)
            grid.addWidget(button, index // 3, index % 3)
            grid.setRowStretch(index // 3, 0)
            grid.setColumnStretch(index % 3, 0)

    def anchor(self) -> str:
        button = self._group.checkedButton()
        if button is None:
            return "c"
        return str(button.property("anchor"))


class ScaleDialog(QDialog):
    def __init__(self, width: int, height: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Scale layer")
        layout = QVBoxLayout(self)
        self.form = ProportionSizeForm(width, height)
        layout.addWidget(self.form)
        layout.addWidget(_dialog_buttons(self))
        self.resize(420, 160)

    def pixel_size(self) -> tuple[int, int]:
        return self.form.pixel_size()


class CanvasSizeDialog(QDialog):
    def __init__(self, width: int, height: int, foreground: tuple[int, ...], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Canvas size")
        color = tuple(int(channel) for channel in foreground[:4])
        self._foreground = color if len(color) == 4 else (*color[:3], 255)
        layout = QVBoxLayout(self)
        self.form = ProportionSizeForm(width, height)
        layout.addWidget(self.form)
        options = QFormLayout()
        anchor_host = QWidget()
        anchor_row = QHBoxLayout(anchor_host)
        anchor_row.setContentsMargins(0, 0, 0, 0)
        self.anchor_picker = AnchorPicker()
        anchor_row.addWidget(self.anchor_picker, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        anchor_row.addStretch(1)
        options.addRow("Anchor", anchor_host)
        self.fill = QComboBox()
        self.fill.addItem("Transparent", None)
        self.fill.addItem("White", (255, 255, 255, 255))
        self.fill.addItem("Black", (0, 0, 0, 255))
        self.fill.addItem("Foreground", "foreground")
        options.addRow("New area", self.fill)
        layout.addLayout(options)
        layout.addWidget(_dialog_buttons(self))
        self.setMinimumWidth(480)
        self.adjustSize()

    def values(self) -> tuple[int, int, str, tuple[int, int, int, int] | None]:
        width, height = self.form.pixel_size()
        chosen = self.fill.currentData()
        if chosen == "foreground":
            fill: tuple[int, int, int, int] | None = self._foreground
        elif chosen is None:
            fill = None
        else:
            fill = tuple(int(channel) for channel in chosen)
        return width, height, self.anchor_picker.anchor(), fill


class ImageSizeDialog(QDialog):
    def __init__(self, width: int, height: int, dpi: float, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Image size")
        layout = QVBoxLayout(self)
        self.form = ProportionSizeForm(width, height, inches=True, dpi=dpi)
        layout.addWidget(self.form)
        options = QFormLayout()
        self.resolution = QDoubleSpinBox()
        self.resolution.setRange(1, 2400)
        self.resolution.setDecimals(2)
        self.resolution.setSuffix(" dpi")
        self.resolution.setValue(float(dpi))
        self.resolution.setKeyboardTracking(False)
        self.resolution.setToolTip("Pixels per inch. With resample off, this changes the printed size only.")
        options.addRow("Resolution", self.resolution)
        self.resample = QCheckBox("Resample")
        self.resample.setChecked(True)
        self.resample.setToolTip("On: rebuild the pixels. Off: keep the pixels and only change the resolution.")
        options.addRow(self.resample)
        self.interpolation = QComboBox()
        for key, label in (
            ("nearest", "Nearest"),
            ("bilinear", "Bilinear"),
            ("bicubic", "Bicubic"),
            ("lanczos", "Lanczos"),
        ):
            self.interpolation.addItem(label, key)
        self.interpolation.setCurrentIndex(max(0, self.interpolation.findData("lanczos")))
        options.addRow("Interpolation", self.interpolation)
        layout.addLayout(options)
        layout.addWidget(_dialog_buttons(self))
        self.resolution.valueChanged.connect(self._on_resolution)
        self.resample.toggled.connect(self._on_resample)
        self.form.dpiEdited.connect(self._on_dpi_from_form)
        self.resize(560, 280)

    def _on_resolution(self, value: float) -> None:
        self.form.set_dpi(float(value), keep_inches=self.resample.isChecked())

    def _on_dpi_from_form(self, dpi: float) -> None:
        self.resolution.blockSignals(True)
        self.resolution.setValue(float(dpi))
        self.resolution.blockSignals(False)

    def _on_resample(self, checked: bool) -> None:
        self.form.set_resample(bool(checked))
        self.interpolation.setEnabled(bool(checked))

    def values(self) -> tuple[int, int, float, str, bool]:
        width, height = self.form.pixel_size()
        method = str(self.interpolation.currentData() or "lanczos")
        return width, height, float(self.resolution.value()), method, self.resample.isChecked()


class RotateDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Rotate layer")
        form = QFormLayout(self)
        self.degrees = QSpinBox()
        self.degrees.setRange(-360, 360)
        self.degrees.setValue(0)
        form.addRow("Degrees clockwise", self.degrees)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)


class LiveFilterDialog(QDialog):
    """Slider dialog that previews on the active layer and commits one undo step.

    Preview writes straight onto the layer image. Cancel restores the snapshot
    taken when the dialog opened. OK restores that snapshot and commits the
    preview through the history, so undo returns to the pre-dialog pixels even
    though the user already saw the result.
    """

    def __init__(self, parent, session, title: str, specs: list[dict], apply_fn) -> None:
        super().__init__(parent)
        self.session = session
        self.apply_fn = apply_fn
        self.layer = session.active_layer()
        self.before = visual_state(self.layer)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        self._sliders: dict[str, QSlider] = {}
        form = QFormLayout()
        for spec in specs:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(int(spec["min"]), int(spec["max"]))
            slider.setValue(int(spec["value"]))
            label = QLabel(self._caption(spec, slider.value()))
            slider.valueChanged.connect(lambda value, spec=spec, label=label: self._on_slide(spec, label, value))
            self._sliders[spec["key"]] = slider
            form.addRow(label, slider)
            self._labels = getattr(self, "_labels", {})
            self._labels[spec["key"]] = (label, spec)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._preview(final=False)

    def _caption(self, spec: dict, value: int) -> str:
        scale = spec.get("scale")
        shown = value * scale if scale else value
        if scale:
            return f"{spec['label']}  {shown:.2f}"
        return f"{spec['label']}  {shown}"

    def _on_slide(self, spec, label, value: int) -> None:
        label.setText(self._caption(spec, value))
        self._preview(final=False)

    def values(self) -> dict:
        data = {}
        for key, slider in self._sliders.items():
            spec = self._labels[key][1]
            value = slider.value() * spec.get("scale", 1)
            data[key] = value
        return data

    def _preview(self, *, final: bool) -> None:
        source = self.before["image"]
        selection = selection_for_layer(self.session.document, self.layer)
        zoom = float(getattr(self.session, "view_zoom", 1.0) or 1.0)
        scale = 1.0 if final else preview_scale(source.width, source.height, zoom)
        if scale >= 1.0:
            set_preview_source(self.layer, None)
            edited = self.apply_fn(source, self.values())
            self.layer.image = apply_masked(source, edited, selection)
        else:
            small = downscale(source, scale)
            small_selection = None if selection is None else selection.resize(small.size, Image.Resampling.NEAREST)
            edited = self.apply_fn(small, self.values())
            self.layer.image = source
            set_preview_source(self.layer, apply_masked(small, edited, small_selection))
        self.layer.kind = "raster"
        self.layer.text = None
        doc = self.session.document
        self.session.preview(placed_box(self.layer, doc.width, doc.height))

    def accept(self) -> None:
        self._preview(final=True)
        new_state = visual_state(self.layer)
        new_state["kind"] = "raster"
        new_state["text"] = None
        apply_visual(self.layer, self.before)
        self.session.commit(MutateLayerCommand(self.layer, new_state, name=self.windowTitle()))
        super().accept()

    def reject(self) -> None:
        set_preview_source(self.layer, None)
        apply_visual(self.layer, self.before)
        self.session.preview()
        super().reject()


class ShortcutsDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Keyboard shortcuts")
        layout = QVBoxLayout(self)
        text = QLabel(
            "\n".join(
                [
                    "Ctrl+N  New          Ctrl+O  Open          Ctrl+W  Close",
                    "Ctrl+S  Save project Ctrl+Shift+S  Save as",
                    "Ctrl+E  Export flattened image",
                    "Ctrl+Z  Undo         Ctrl+Shift+Z or Ctrl+Y  Redo",
                    "Ctrl+A  Select all   Ctrl+D or Esc  Deselect   Ctrl+Shift+I  Invert",
                    "Ctrl+X  Cut selection to a new layer",
                    "Ctrl+C  Copy selection to a new layer",
                    "Ctrl+Shift+C  Copy to the clipboard    Ctrl+V  Paste",
                    "Delete or Backspace  Clear pixels inside the selection",
                    "Select options  Feather  Soften the next selection. Select → Feather… uses that radius",
                    "Ctrl+Shift+O  Place an image as new layers (File → Place, Layer → Place image)",
                    "Ctrl+0  Fit          Ctrl+1  100%       Ctrl++ / Ctrl+-  Zoom",
                    "V Transform   B Brush   E Eraser   T Text   R Mask   S Select",
                    "C Sizes & Crop   I Image & Eyedrop   G Fill   U Shape   H Hand   Z Zoom",
                    "[ and ]  Brush size     Space  Pan     Wheel  Zoom at cursor",
                    "Escape  Cancel the current drag, or deselect",
                    "Toolbar  New, Open, Close, Save, Save As, Place, Export, Show folder, and About",
                ]
            )
        )
        text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
        self.resize(560, 280)


def _about_markdown() -> str:
    """Guide text shipped beside the stylesheet. The version is not in this file."""
    path = Path(__file__).resolve().parent.parent / "resources" / "about.md"
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return "PixEdit is a layered desktop image editor."


class AboutDialog(QDialog):
    """Name, version, and the short guide in ``resources/about.md``.

    The version comes from ``__version__``, so editing the guide does not
    have to track the package version.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About PixEdit")
        layout = QVBoxLayout(self)
        title = QLabel("PixEdit")
        title.setObjectName("aboutTitle")
        self.version = QLabel(__version__)
        self.version.setObjectName("aboutVersion")
        self.version.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(title)
        layout.addWidget(self.version)
        self.body = QTextBrowser()
        self.body.setObjectName("aboutBody")
        self.body.setOpenLinks(False)
        self.body.setOpenExternalLinks(False)
        # Markdown becomes HTML. The default stylesheet colors headings to
        # match the dark chrome after that conversion.
        self.body.document().setDefaultStyleSheet(
            "h2 { color: #d8f6ff; font-size: 15px; }"
            "p, li { color: #e7edf2; }"
            "code { color: #d8f6ff; }"
        )
        self.body.setMarkdown(_about_markdown())
        layout.addWidget(self.body, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
        self.resize(560, 480)

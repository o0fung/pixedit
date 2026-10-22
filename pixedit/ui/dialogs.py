"""Modal editors: new canvas, text, PDF import, and live filter previews."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from pixedit.adjust import apply_masked
from pixedit.geom import selection_for_layer
from pixedit.model import MutateLayerCommand, TextInfo, apply_visual, visual_state
from pixedit.textutil import discover_fonts


class NewDocumentDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New canvas")
        form = QFormLayout(self)
        self.width = QSpinBox()
        self.width.setRange(1, 16000)
        self.width.setValue(1280)
        self.height = QSpinBox()
        self.height.setRange(1, 16000)
        self.height.setValue(800)
        self.background = QComboBox()
        self.background.addItems(["white", "black", "transparent"])
        form.addRow("Width", self.width)
        form.addRow("Height", self.height)
        form.addRow("Background", self.background)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> tuple[int, int, str]:
        return self.width.value(), self.height.value(), self.background.currentText()


class PdfImportDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import PDF")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Each page becomes a layer. Only the first page starts visible."))
        form = QFormLayout()
        self.dpi = QComboBox()
        self.dpi.addItem("72 DPI  (screen)", 1.0)
        self.dpi.addItem("144 DPI  (sharp)", 2.0)
        self.dpi.addItem("216 DPI  (print)", 3.0)
        self.dpi.setCurrentIndex(1)
        form.addRow("Render", self.dpi)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def scale(self) -> float:
        return float(self.dpi.currentData())


class TextDialog(QDialog):
    def __init__(self, initial: TextInfo, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Text")
        self._color = tuple(initial.color)
        layout = QVBoxLayout(self)
        self.text = QPlainTextEdit()
        self.text.setPlainText(initial.text)
        self.text.setMinimumHeight(80)
        layout.addWidget(self.text)
        form = QFormLayout()
        self.font = QComboBox()
        self._fonts = discover_fonts()
        if not self._fonts:
            self.font.addItem("Default", None)
        else:
            for family, style, path in self._fonts:
                self.font.addItem(f"{family} {style}".strip(), path)
            if initial.font_path:
                index = self.font.findData(initial.font_path)
                if index >= 0:
                    self.font.setCurrentIndex(index)
        self.size = QSpinBox()
        self.size.setRange(1, 512)
        self.size.setValue(int(initial.size))
        self.rotation = QSlider(Qt.Orientation.Horizontal)
        self.rotation.setRange(-180, 180)
        self.rotation.setValue(int(round(initial.rotation)))
        self.rotation_label = QLabel(f"{self.rotation.value()}°")
        self.rotation.valueChanged.connect(lambda value: self.rotation_label.setText(f"{value}°"))
        rotation_row = QHBoxLayout()
        rotation_row.addWidget(self.rotation)
        rotation_row.addWidget(self.rotation_label)
        self.color_button = QPushButton("Color")
        self.color_button.clicked.connect(self._pick_color)
        self._paint_color_button()
        form.addRow("Font", self.font)
        form.addRow("Size", self.size)
        form.addRow("Rotation", rotation_row)
        form.addRow("Color", self.color_button)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(420, 320)

    def _pick_color(self) -> None:
        current = QColor(*self._color)
        picked = QColorDialog.getColor(current, self, "Text color", QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if not picked.isValid():
            return
        self._color = (picked.red(), picked.green(), picked.blue(), picked.alpha())
        self._paint_color_button()

    def _paint_color_button(self) -> None:
        r, g, b, _a = self._color
        self.color_button.setStyleSheet(f"background: rgb({r}, {g}, {b}); color: white;")

    def result_info(self) -> TextInfo:
        path = self.font.currentData()
        label = self.font.currentText()
        family = label.split(" ")[0] if label else "Default"
        return TextInfo(
            text=self.text.toPlainText(),
            font_path=path,
            font_family=family,
            size=self.size.value(),
            color=self._color,
            rotation=float(self.rotation.value()),
        )


class ScaleDialog(QDialog):
    def __init__(self, width: int, height: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Scale layer")
        self._aspect = width / max(1, height)
        self._lock = False
        form = QFormLayout(self)
        self.width = QSpinBox()
        self.height = QSpinBox()
        for box, value in ((self.width, width), (self.height, height)):
            box.setRange(1, 16000)
            box.setValue(value)
        self.keep = QCheckBox("Keep aspect ratio")
        self.keep.setChecked(True)
        self.width.valueChanged.connect(self._width_changed)
        self.height.valueChanged.connect(self._height_changed)
        form.addRow("Width", self.width)
        form.addRow("Height", self.height)
        form.addRow(self.keep)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _width_changed(self, value: int) -> None:
        if not self.keep.isChecked() or self._lock:
            return
        self._lock = True
        self.height.setValue(max(1, int(round(value / self._aspect))))
        self._lock = False

    def _height_changed(self, value: int) -> None:
        if not self.keep.isChecked() or self._lock:
            return
        self._lock = True
        self.width.setValue(max(1, int(round(value * self._aspect))))
        self._lock = False

    def pixel_size(self) -> tuple[int, int]:
        return self.width.value(), self.height.value()


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
        self._preview()

    def _caption(self, spec: dict, value: int) -> str:
        scale = spec.get("scale")
        shown = value * scale if scale else value
        if scale:
            return f"{spec['label']}  {shown:.2f}"
        return f"{spec['label']}  {shown}"

    def _on_slide(self, spec, label, value: int) -> None:
        label.setText(self._caption(spec, value))
        self._preview()

    def values(self) -> dict:
        data = {}
        for key, slider in self._sliders.items():
            spec = self._labels[key][1]
            value = slider.value() * spec.get("scale", 1)
            data[key] = value
        return data

    def _preview(self) -> None:
        edited = self.apply_fn(self.before["image"], self.values())
        masked = apply_masked(self.before["image"], edited, selection_for_layer(self.session.document, self.layer))
        self.layer.image = masked
        self.layer.kind = "raster"
        self.layer.text = None
        self.session.preview()

    def accept(self) -> None:
        self._preview()
        new_state = visual_state(self.layer)
        new_state["kind"] = "raster"
        new_state["text"] = None
        apply_visual(self.layer, self.before)
        self.session.commit(MutateLayerCommand(self.layer, new_state, name=self.windowTitle()))
        super().accept()

    def reject(self) -> None:
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
                    "Ctrl+N  New          Ctrl+O  Open",
                    "Ctrl+S  Save project Ctrl+Shift+S  Save as",
                    "Ctrl+E  Export flattened image",
                    "Ctrl+Z  Undo         Ctrl+Shift+Z or Ctrl+Y  Redo",
                    "Ctrl+A  Select all   Ctrl+D or Esc  Deselect   Ctrl+Shift+I  Invert",
                    "Ctrl+X  Cut selection to a new layer",
                    "Ctrl+C  Copy selection to a new layer",
                    "Delete or Backspace  Clear pixels inside the selection",
                    "Ctrl+Shift+O  Place an image as new layers (File → Place, Layer → Place image)",
                    "Ctrl+0  Fit          Ctrl+1  100%       Ctrl++ / Ctrl+-  Zoom",
                    "V Move   B Brush   E Eraser   T Text   R Redact   M Select",
                    "C Crop   G Fill    I Eyedropper   U Shape   H Hand   Z Zoom",
                    "[ and ]  Brush size     Space  Pan     Wheel  Zoom at cursor",
                    "Escape  Cancel the current drag, or deselect",
                    "Toolbar  Save, Open, Place, and Export (same as the File menu)",
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

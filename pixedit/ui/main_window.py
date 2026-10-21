"""Main window: menus, tool strip, canvas, and the layer stack."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QColor, QIcon, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QButtonGroup,
    QColorDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pixedit import __version__
from pixedit.adjust import blur, brightness_contrast, desaturate, hue_saturation, invert, levels, pixelate, sharpen
from pixedit.geom import alpha_bbox
from pixedit.io import export_image, load_path, save_project
from pixedit.plugins.api import get_api
from pixedit.plugins.loader import load_plugins
from pixedit.session import Session
from pixedit.tools.base import Tool, builtin_tools
from pixedit.ui.canvas import CanvasView
from pixedit.ui.dialogs import (
    LiveFilterDialog,
    NewDocumentDialog,
    PdfImportDialog,
    RotateDialog,
    ScaleDialog,
    ShortcutsDialog,
    TextDialog,
)
from pixedit.ui.panels import LayerPanel, ToolOptions

_OPEN_FILTER = "Images and projects (*.png *.jpg *.jpeg *.webp *.tif *.tiff *.bmp *.gif *.pdf *.pix);;All files (*)"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        load_plugins()
        self.session = Session()
        self.session.add_listener(self._on_session)
        self.session.set_message_handler(self._toast)
        self.session.set_text_prompter(self._prompt_text)
        self._owned_actions: list[QAction] = []
        self._owned_shortcuts: list[QShortcut] = []
        self.tools: list[Tool] = []
        self.tool_by_id: dict[str, Tool] = {}
        self.current: Tool | None = None
        self._buttons: dict[str, QToolButton] = {}
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._cursor = (0.0, 0.0)
        self._build()
        self._rebuild_tools()
        self._build_menus()
        self._install_shortcuts()
        self.setWindowTitle("PixEdit")
        self.setWindowIcon(_app_icon())
        self.resize(1280, 820)
        self._on_session("change")
        QTimer.singleShot(0, self.canvas.fit)
        QTimer.singleShot(0, self.canvas.setFocus)

    def _build(self) -> None:
        self.setStatusBar(QStatusBar())
        shell = QWidget()
        self.setCentralWidget(shell)
        row = QHBoxLayout(shell)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        left = QFrame()
        left.setObjectName("panel")
        left.setFixedWidth(248)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        self.tool_host = QWidget()
        self.tool_layout = QVBoxLayout(self.tool_host)
        self.tool_layout.setContentsMargins(0, 0, 0, 0)
        self.tool_layout.setSpacing(4)
        left_layout.addWidget(self.tool_host)
        self.swatch = QPushButton("Foreground")
        self.swatch.clicked.connect(self._choose_color)
        left_layout.addWidget(self.swatch)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.options = ToolOptions(self.session)
        scroll.setWidget(self.options)
        left_layout.addWidget(scroll, 1)

        self.canvas = CanvasView(self.session, self._current_tool, self._on_cursor)
        self.layers = LayerPanel(self.session, self._rename_layer)
        right = QFrame()
        right.setObjectName("layers")
        right.setMinimumWidth(240)
        right.setMaximumWidth(360)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.layers)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self.canvas)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        row.addWidget(splitter)
        self._update_swatch()

    def _rebuild_tools(self) -> None:
        current_id = self.current.id if self.current else "move"
        builtin_ids = {tool.id for tool in builtin_tools()}
        extras = [tool for tool in get_api().tools if tool.id not in builtin_ids]
        self.tools = [*builtin_tools(), *extras]
        self.tool_by_id = {tool.id: tool for tool in self.tools}
        while self.tool_layout.count():
            item = self.tool_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for button in list(self._buttons.values()):
            self._button_group.removeButton(button)
        self._buttons.clear()
        for tool in self.tools:
            button = QToolButton()
            button.setText(tool.label)
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            tip = tool.label if not tool.shortcut else f"{tool.label}  ({tool.shortcut})"
            button.setToolTip(tip)
            button.clicked.connect(lambda _checked=False, chosen=tool: self.set_tool(chosen))
            self._button_group.addButton(button)
            self.tool_layout.addWidget(button)
            self._buttons[tool.id] = button
        self.tool_layout.addStretch(1)
        self.set_tool(self.tool_by_id.get(current_id, self.tools[0]))

    def _build_menus(self) -> None:
        for action in self._owned_actions:
            action.setShortcut(QKeySequence())
            action.deleteLater()
        self._owned_actions = []
        bar = self.menuBar()
        bar.clear()
        file_menu = bar.addMenu("File")
        edit_menu = bar.addMenu("Edit")
        image_menu = bar.addMenu("Image")
        layer_menu = bar.addMenu("Layer")
        select_menu = bar.addMenu("Select")
        filter_menu = bar.addMenu("Filter")
        view_menu = bar.addMenu("View")
        plugin_menu = bar.addMenu("Plugins")
        help_menu = bar.addMenu("Help")

        self._action(file_menu, "New…", self._new, QKeySequence.StandardKey.New)
        self._action(file_menu, "Open…", self._open, QKeySequence.StandardKey.Open)
        file_menu.addSeparator()
        self._action(file_menu, "Save project", self._save, QKeySequence.StandardKey.Save)
        self._action(file_menu, "Save project as…", self._save_as, QKeySequence("Ctrl+Shift+S"))
        self._action(file_menu, "Export flattened…", self._export, QKeySequence("Ctrl+E"))
        file_menu.addSeparator()
        self._action(file_menu, "Quit", self.close, QKeySequence.StandardKey.Quit)

        self.undo_action = self._action(edit_menu, "Undo", self.session.undo, QKeySequence.StandardKey.Undo)
        self.redo_action = self._action(edit_menu, "Redo", self.session.redo, QKeySequence.StandardKey.Redo)
        edit_menu.addSeparator()
        self._action(edit_menu, "Deselect", self.session.deselect, QKeySequence("Ctrl+D"))

        self._action(image_menu, "Crop to selection", self._crop_selection)
        self._action(image_menu, "Trim transparent edges", self.session.trim)
        image_menu.addSeparator()
        self._action(image_menu, "Rotate canvas 90° clockwise", lambda: self.session.transform_canvas("rotate-cw"))
        self._action(image_menu, "Rotate canvas 90° counter-clockwise", lambda: self.session.transform_canvas("rotate-ccw"))
        self._action(image_menu, "Flip canvas horizontal", lambda: self.session.transform_canvas("flip-h"))
        self._action(image_menu, "Flip canvas vertical", lambda: self.session.transform_canvas("flip-v"))
        image_menu.addSeparator()
        self._action(image_menu, "Brightness / Contrast…", lambda: self._filter(
            "Brightness / Contrast",
            [
                {"key": "brightness", "label": "Brightness", "min": -100, "max": 100, "value": 0},
                {"key": "contrast", "label": "Contrast", "min": -100, "max": 100, "value": 0},
            ],
            lambda image, values: brightness_contrast(image, values["brightness"], values["contrast"]),
        ))
        self._action(image_menu, "Levels…", lambda: self._filter(
            "Levels",
            [
                {"key": "in_black", "label": "Input black", "min": 0, "max": 254, "value": 0},
                {"key": "in_white", "label": "Input white", "min": 1, "max": 255, "value": 255},
                {"key": "gamma", "label": "Gamma", "min": 10, "max": 300, "value": 100, "scale": 0.01},
            ],
            lambda image, values: levels(image, values["in_black"], max(values["in_black"] + 1, values["in_white"]), values["gamma"]),
        ))
        self._action(image_menu, "Hue / Saturation…", lambda: self._filter(
            "Hue / Saturation",
            [
                {"key": "hue", "label": "Hue", "min": -180, "max": 180, "value": 0},
                {"key": "sat", "label": "Saturation", "min": -100, "max": 100, "value": 0},
                {"key": "light", "label": "Lightness", "min": -100, "max": 100, "value": 0},
            ],
            lambda image, values: hue_saturation(image, values["hue"], values["sat"], values["light"]),
        ))
        self._action(image_menu, "Desaturate", lambda: self.session.run_filter("Desaturate", desaturate))
        self._action(image_menu, "Invert", lambda: self.session.run_filter("Invert", invert))

        self._action(layer_menu, "New layer", self.session.add_blank_layer, QKeySequence("Ctrl+Shift+N"))
        self._action(layer_menu, "Duplicate", self.session.duplicate_active, QKeySequence("Ctrl+J"))
        self._action(layer_menu, "Delete", self.session.delete_active)
        self._action(layer_menu, "Rename…", self._rename_layer)
        layer_menu.addSeparator()
        self._action(layer_menu, "Merge down", self.session.merge_down_active, QKeySequence("Ctrl+Shift+E"))
        self._action(layer_menu, "Flatten image", self.session.flatten)
        layer_menu.addSeparator()
        self._action(layer_menu, "Bring forward", lambda: self.session.move_active(1))
        self._action(layer_menu, "Send backward", lambda: self.session.move_active(-1))
        layer_menu.addSeparator()
        self._action(layer_menu, "Scale…", self._scale_layer)
        self._action(layer_menu, "Rotate…", self._rotate_layer)
        self._action(layer_menu, "Flip horizontal", lambda: self.session.transform_active("flip-h"))
        self._action(layer_menu, "Flip vertical", lambda: self.session.transform_active("flip-v"))

        self._action(select_menu, "All", self.session.select_all, QKeySequence.StandardKey.SelectAll)
        self._action(select_menu, "Deselect", self.session.deselect, QKeySequence("Ctrl+D"))
        self._action(select_menu, "Invert", self.session.invert_selection, QKeySequence("Ctrl+Shift+I"))
        select_menu.addSeparator()
        self._action(select_menu, "Rectangular", lambda: self._select_mode("rect"))
        self._action(select_menu, "Elliptical", lambda: self._select_mode("ellipse"))
        self._action(select_menu, "Freehand", lambda: self._select_mode("freehand"))
        self._action(select_menu, "Magic wand", lambda: self._select_mode("wand"))

        self._action(filter_menu, "Gaussian blur…", lambda: self._filter(
            "Gaussian blur",
            [{"key": "radius", "label": "Radius", "min": 1, "max": 40, "value": 4}],
            lambda image, values: blur(image, values["radius"]),
        ))
        self._action(filter_menu, "Sharpen", lambda: self.session.run_filter("Sharpen", sharpen))
        self._action(filter_menu, "Pixelate…", lambda: self._filter(
            "Pixelate",
            [{"key": "block", "label": "Block", "min": 2, "max": 64, "value": 8}],
            lambda image, values: pixelate(image, int(values["block"])),
        ))
        filter_menu.addSeparator()
        for command in get_api().commands:
            if command.menu != "Filter":
                continue
            action = self._action(filter_menu, command.label, lambda _checked=False, cb=command.callback: self._run_plugin(cb))
            if command.shortcut:
                action.setShortcut(QKeySequence(command.shortcut))

        self._action(view_menu, "Zoom in", lambda: self.canvas.zoom_at(self.canvas.width() / 2, self.canvas.height() / 2, 1.25), QKeySequence.StandardKey.ZoomIn)
        self._action(view_menu, "Zoom out", lambda: self.canvas.zoom_at(self.canvas.width() / 2, self.canvas.height() / 2, 0.8), QKeySequence.StandardKey.ZoomOut)
        self._action(view_menu, "Fit", self.canvas.fit, QKeySequence("Ctrl+0"))
        self._action(view_menu, "Actual pixels", self.canvas.zoom_100, QKeySequence("Ctrl+1"))
        view_menu.addSeparator()
        rulers = self._action(view_menu, "Rulers", self._toggle_rulers)
        rulers.setCheckable(True)
        rulers.setChecked(bool(self.session.option("rulers", True)))
        self._rulers_action = rulers

        self._action(plugin_menu, "Reload plugins", self._reload_plugins)
        plugin_menu.addSeparator()
        if not get_api().commands and not get_api().tools:
            empty = plugin_menu.addAction("No extra plugins loaded")
            empty.setEnabled(False)
        for command in get_api().commands:
            self._action(
                plugin_menu,
                f"{command.label}  ({command.menu})",
                lambda _checked=False, cb=command.callback: self._run_plugin(cb),
            )

        self._action(help_menu, "Keyboard shortcuts", lambda: ShortcutsDialog(self).exec())
        self._action(help_menu, "About PixEdit", self._about)

    def _action(self, menu, label: str, slot, shortcut=None) -> QAction:
        action = QAction(label, self)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.triggered.connect(lambda _checked=False: slot())
        menu.addAction(action)
        self._owned_actions.append(action)
        return action

    def _install_shortcuts(self) -> None:
        for shortcut in self._owned_shortcuts:
            shortcut.setEnabled(False)
            shortcut.deleteLater()
        self._owned_shortcuts = []
        redo_alt = QShortcut(QKeySequence("Ctrl+Y"), self)
        redo_alt.activated.connect(self.session.redo)
        self._owned_shortcuts.append(redo_alt)
        for tool in self.tools:
            if not tool.shortcut or len(tool.shortcut) != 1:
                continue
            shortcut = QShortcut(QKeySequence(tool.shortcut), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(lambda chosen=tool: self.set_tool(chosen))
            self._owned_shortcuts.append(shortcut)
        smaller = QShortcut(QKeySequence("["), self)
        smaller.activated.connect(lambda: self._bump_brush(-2))
        larger = QShortcut(QKeySequence("]"), self)
        larger.activated.connect(lambda: self._bump_brush(2))
        self._owned_shortcuts.extend((smaller, larger))

    def _current_tool(self) -> Tool | None:
        return self.current

    def set_tool(self, tool: Tool) -> None:
        if self.current is not None and self.current is not tool:
            self.current.pointer_cancel(self.session)
        self.current = tool
        button = self._buttons.get(tool.id)
        if button is not None:
            button.setChecked(True)
        self.options.set_tool(tool)
        self.canvas.set_tool_cursor(tool.cursor)
        self.statusBar().showMessage(tool.hint or tool.label)

    def _on_session(self, kind: str) -> None:
        self.canvas.invalidate()
        self._update_swatch()
        self._update_status()
        if kind == "preview":
            return
        self.layers.refresh()
        self._update_title()
        self._update_history_actions()

    def _update_history_actions(self) -> None:
        if hasattr(self, "undo_action"):
            self.undo_action.setEnabled(self.session.history.can_undo())
            undo = self.session.history.undo_stack[-1].name if self.session.history.can_undo() else ""
            self.undo_action.setText(f"Undo {undo}".strip())
            self.redo_action.setEnabled(self.session.history.can_redo())
            redo = self.session.history.redo_stack[-1].name if self.session.history.can_redo() else ""
            self.redo_action.setText(f"Redo {redo}".strip())

    def _update_title(self) -> None:
        if self.session.path:
            name = Path(self.session.path).name
        else:
            name = "Untitled"
        mark = " •" if self.session.dirty else ""
        self.setWindowTitle(f"PixEdit — {name}{mark}")

    def _update_swatch(self) -> None:
        r, g, b, a = self.session.foreground
        self.swatch.setText(f"  #{r:02X}{g:02X}{b:02X}")
        self.swatch.setStyleSheet(
            f"background: rgba({r}, {g}, {b}, {a}); color: {'#111' if r + g + b > 380 else '#fff'}; text-align: left;"
        )

    def _update_status(self) -> None:
        doc = self.session.document
        x, y = self._cursor
        tool = self.current.label if self.current else ""
        zoom = int(round(self.canvas.zoom * 100))
        self.statusBar().showMessage(
            f"{doc.width} × {doc.height} px    {x:.0f}, {y:.0f}    {zoom}%    {tool}"
        )

    def _on_cursor(self, x: float, y: float, zoom: float) -> None:
        self._cursor = (x, y)
        self._update_status()

    def _toast(self, text: str) -> None:
        self.statusBar().showMessage(text, 4000)

    def _choose_color(self) -> None:
        current = QColor(*self.session.foreground)
        picked = QColorDialog.getColor(current, self, "Foreground", QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if not picked.isValid():
            return
        self.session.foreground = (picked.red(), picked.green(), picked.blue(), picked.alpha())
        self._update_swatch()

    def _prompt_text(self, initial):
        dialog = TextDialog(initial, self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return None
        return dialog.result_info()

    def _confirm_discard(self) -> bool:
        if not self.session.dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            "Save the current project before continuing?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            return self._save()
        return True

    def _new(self) -> None:
        if not self._confirm_discard():
            return
        dialog = NewDocumentDialog(self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        width, height, background = dialog.values()
        self.session.new_canvas(width, height, background)
        QTimer.singleShot(0, self.canvas.fit)

    def _open(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open", "", _OPEN_FILTER)
        if path:
            self.open_path(path)

    def open_path(self, path: str) -> None:
        pdf_scale = 2.0
        if Path(path).suffix.lower() == ".pdf":
            dialog = PdfImportDialog(self)
            if dialog.exec() != dialog.DialogCode.Accepted:
                return
            pdf_scale = dialog.scale()
        try:
            document = load_path(path, pdf_scale=pdf_scale)
        except Exception as exc:
            QMessageBox.warning(self, "Open failed", f"Could not open {path}\n\n{exc}")
            return
        project = Path(path).suffix.lower() == ".pix"
        self.session.replace_document(document, path if project else None)
        QTimer.singleShot(0, self.canvas.fit)
        self.statusBar().showMessage(f"Opened {Path(path).name}")

    def _save(self) -> bool:
        if not self.session.path:
            return self._save_as()
        try:
            save_project(self.session.document, self.session.path)
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return False
        self.session.mark_clean()
        self.statusBar().showMessage(f"Saved {self.session.path}")
        return True

    def _save_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(self, "Save project", "untitled.pix", "PixEdit project (*.pix)")
        if not path:
            return False
        if not path.lower().endswith(".pix"):
            path += ".pix"
        self.session.path = path
        return self._save()

    def _export(self) -> None:
        path, selected = QFileDialog.getSaveFileName(
            self,
            "Export flattened image",
            "export.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;WebP (*.webp)",
        )
        if not path:
            return
        suffix = Path(path).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            if "JPEG" in selected:
                path += ".jpg"
            elif "WebP" in selected:
                path += ".webp"
            else:
                path += ".png"
        try:
            export_image(self.session.document, path)
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(f"Exported {path}")

    def _rename_layer(self) -> None:
        layer = self.session.active_layer()
        if layer is None:
            return
        name, ok = QInputDialog.getText(self, "Rename layer", "Name", text=layer.name)
        if ok:
            self.session.rename_active(name)

    def _scale_layer(self) -> None:
        layer = self.session.active_layer()
        if layer is None:
            return
        bitmap = layer.bitmap()
        dialog = ScaleDialog(bitmap.width, bitmap.height, self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        width, height = dialog.pixel_size()
        self.session.transform_active("scale", width=width, height=height)

    def _rotate_layer(self) -> None:
        if self.session.active_layer() is None:
            return
        dialog = RotateDialog(self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        self.session.transform_active("rotate", degrees=dialog.degrees.value())

    def _crop_selection(self) -> None:
        selection = self.session.document.selection
        if selection is None:
            self._toast("Draw a selection first.")
            return
        box = alpha_bbox(selection)
        if box is None:
            self._toast("The selection is empty.")
            return
        left, top, right, bottom = box
        self.session.crop_to(left, top, right - left, bottom - top)

    def _select_mode(self, mode: str) -> None:
        self.session.options["select_mode"] = mode
        tool = self.tool_by_id.get("select")
        if tool is not None:
            self.set_tool(tool)

    def _filter(self, title: str, specs: list[dict], apply_fn) -> None:
        layer = self.session.active_layer()
        if layer is None or layer.kind == "redaction":
            self._toast("Choose a raster or text layer.")
            return
        LiveFilterDialog(self, self.session, title, specs, apply_fn).exec()

    def _run_plugin(self, callback) -> None:
        try:
            callback(self.session)
        except Exception as exc:
            QMessageBox.warning(self, "Plugin", f"{type(exc).__name__}: {exc}")

    def _reload_plugins(self) -> None:
        load_plugins(force=True)
        self._rebuild_tools()
        self._build_menus()
        self._install_shortcuts()
        self._toast("Plugins reloaded")

    def _toggle_rulers(self) -> None:
        self.session.options["rulers"] = not bool(self.session.option("rulers", True))
        if hasattr(self, "_rulers_action"):
            self._rulers_action.setChecked(bool(self.session.option("rulers", True)))
        self.canvas.update()

    def _bump_brush(self, delta: int) -> None:
        size = int(self.session.option("brush_size", 24)) + delta
        size = max(1, min(400, size))
        self.session.options["brush_size"] = size
        self.options.sync("brush_size", size)
        self._toast(f"Brush size {size}")

    def _about(self) -> None:
        QMessageBox.information(
            self,
            "About PixEdit",
            f"PixEdit {__version__}\n\nA layered image editor.\nLaunch it with the pix command.",
        )

    def closeEvent(self, event) -> None:
        if self._confirm_discard():
            event.accept()
        else:
            event.ignore()


def _app_icon() -> QIcon:
    pixmap = QPixmap(64, 64)
    pixmap.fill(QColor("#143544"))
    return QIcon(pixmap)

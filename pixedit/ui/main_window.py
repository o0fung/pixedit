"""Main window: menus, tool strip, canvas, and the layer stack."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSize, Qt, QTimer, QUrl
from PyQt6.QtGui import QAction, QColor, QDesktopServices, QIcon, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pixedit import __version__
from pixedit.adjust import blur, brightness_contrast, desaturate, hue_saturation, invert, levels, pixelate, sharpen
from pixedit.frames import apply_layer_frame, export_animated_gif, frame_count, frame_kind
from pixedit.io import (
    EXPORT_SUFFIXES,
    export_filter_for_suffix,
    export_filter_string,
    export_image,
    extension_for_export_filter,
    load_path,
    save_project,
)
from pixedit.model import empty_document
from pixedit.session.layers import apply_pix_group_names
from pixedit.output_folder import (
    dialog_start,
    ensure_output_folder,
    output_folder,
    pin_native_dialog_directory,
    set_output_folder,
)
from pixedit.plugins.api import get_api
from pixedit.plugins.loader import load_plugins
from pixedit.session import Session
from pixedit.session.core import START_HINT
from pixedit.tools.base import Tool, builtin_tools
from pixedit.ui.canvas import CanvasView, _is_clear_shortcut
from pixedit.ui.dialogs import (
    AboutDialog,
    CanvasSizeDialog,
    ImageSizeDialog,
    LiveFilterDialog,
    NewDocumentDialog,
    PdfImportDialog,
    RotateDialog,
    ScaleDialog,
    ShortcutsDialog,
    TextDialog,
)
from pixedit.ui.icons import icon_for_tool, icon_named
from pixedit.ui.layers_panel import LayerPanel
from pixedit.ui.palette import ColorPalette
from pixedit.ui.tool_options import ToolOptions

_OPEN_FILTER = "Images and projects (*.png *.jpg *.jpeg *.webp *.tif *.tiff *.bmp *.gif *.pdf *.pix);;All files (*)"
_PLACE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.tif *.tiff *.bmp *.gif *.pdf);;All files (*)"

# Toolbar clusters. Anything not listed (Marker, other plugins) follows in its own group.
_TOOL_GROUPS = (
    ("transform", "hand", "zoom", "select"),
    ("brush", "eraser", "fill", "shape", "text"),
    ("color", "crop", "mask"),
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        load_plugins()
        self.session = Session(empty_document())
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
        status = QStatusBar()
        self.setStatusBar(status)
        # Permanent readout. Toasts use showMessage, which covers this label
        # for a few seconds and then reveals it again. Writing the readout
        # through showMessage used to wipe those toasts on the next mouse move.
        self._status_label = QLabel()
        self._status_label.setObjectName("statusInfo")
        status.addWidget(self._status_label, 1)
        # Permanent widgets stay visible while showMessage covers the readout.
        # A click opens About; the document name in the title stays version-free.
        self._version_label = _VersionLabel(__version__)
        status.addPermanentWidget(self._version_label)
        # File actions sit in their own strip. The left column stays the
        # canvas tools (Move, Brush, and the rest), grouped separately.
        self.file_toolbar = QToolBar("File")
        self.file_toolbar.setObjectName("fileToolbar")
        self.file_toolbar.setMovable(False)
        self.file_toolbar.setFloatable(False)
        self.file_toolbar.setIconSize(QSize(22, 22))
        self.file_toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.file_toolbar)
        shell = QWidget()
        self.setCentralWidget(shell)
        row = QHBoxLayout(shell)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        left = QFrame()
        left.setObjectName("panel")
        # The longest command label needs about 193px. The frame and the
        # options panel each add 8px of margin on both sides, so 228 keeps
        # that label inside the column. 248 is only the starting width.
        left.setMinimumWidth(228)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        self.tool_host = QWidget()
        self.tool_layout = QVBoxLayout(self.tool_host)
        self.tool_layout.setContentsMargins(0, 0, 0, 0)
        self.tool_layout.setSpacing(4)
        left_layout.addWidget(self.tool_host)
        self.palette = ColorPalette(self.session, self._on_palette_color)
        left_layout.addWidget(self.palette)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.options = ToolOptions(self.session, self._on_tool_option, self._on_panel_action)
        scroll.setWidget(self.options)
        left_layout.addWidget(scroll, 1)

        self.canvas = CanvasView(self.session, self._current_tool, self._on_cursor, self._drop_event)
        self.setAcceptDrops(True)
        self._playing = False
        self._play_index = 0
        self._play_kind: str | None = None
        self._play_saved: list | None = None
        self._play_return: tuple | None = None
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_play)
        self.layers = LayerPanel(self.session)
        right = QFrame()
        right.setObjectName("layers")
        right.setMinimumWidth(240)
        right.setMaximumWidth(360)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.layers)

        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(left)
        splitter.addWidget(self.canvas)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([248, 800, 280])
        row.addWidget(splitter)

    def _rebuild_tools(self) -> None:
        current_id = self.current.id if self.current else "transform"
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
        # Transform/Hand/Zoom/Select, then paint tools, then Image & Eyedrop,
        # Sizes & Crop, and Mask, then Marker and any other plugins. A line
        # separates groups.
        grouped: list[list[Tool]] = []
        placed: set[str] = set()
        for ids in _TOOL_GROUPS:
            group = [self.tool_by_id[tool_id] for tool_id in ids if tool_id in self.tool_by_id]
            if group:
                grouped.append(group)
                placed.update(tool.id for tool in group)
        extras = [tool for tool in self.tools if tool.id not in placed]
        if extras:
            grouped.append(extras)
        for index, group in enumerate(grouped):
            if index:
                self.tool_layout.addWidget(_tool_separator())
            for tool in group:
                button = QToolButton()
                caption = tool.button_caption()
                # A single & is a mnemonic and disappears. && is the visible ampersand.
                button.setText(caption.replace("&", "&&"))
                button.setIcon(icon_for_tool(tool.id))
                button.setIconSize(QSize(22, 22))
                button.setCheckable(True)
                button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
                button.setFixedHeight(30)
                button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                button.setAccessibleName(caption)
                button.setToolTip(f"{caption}\n{tool.hint}" if tool.hint else caption)
                button.clicked.connect(lambda _checked=False, chosen=tool: self.set_tool(chosen))
                self._button_group.addButton(button)
                self.tool_layout.addWidget(button)
                self._buttons[tool.id] = button
        self.tool_layout.addStretch(1)
        self.set_tool(self.tool_by_id.get(current_id, self.tools[0]))

    def _build_menus(self) -> None:
        # Drop the old action objects before building new ones. The file
        # strip holds the same objects as the File menu, so clear it first.
        if hasattr(self, "file_toolbar"):
            self.file_toolbar.clear()
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

        self._new_action = self._file_action(
            file_menu,
            "New…",
            "New",
            self._new,
            QKeySequence.StandardKey.New,
            "new",
            "Start a blank document",
        )
        self._open_action = self._file_action(
            file_menu,
            "Open…",
            "Open",
            self._open,
            QKeySequence.StandardKey.Open,
            "open",
            "Open a file and replace the current document",
        )
        self._place_action = self._file_action(
            file_menu,
            "Place…",
            "Place",
            self._place,
            QKeySequence("Ctrl+Shift+O"),
            "place",
            "Place an image as new layers on the current document",
        )
        file_menu.addSeparator()
        self._save_action = self._file_action(
            file_menu,
            "Save project",
            "Save",
            self._save,
            QKeySequence.StandardKey.Save,
            "save",
            "Save the project to its .pix file",
        )
        self._save_as_action = self._file_action(
            file_menu,
            "Save project as…",
            "Save As",
            self._save_as,
            QKeySequence("Ctrl+Shift+S"),
            "save-as",
            "Save the project to a new .pix file",
        )
        self._export_action = self._file_action(
            file_menu,
            "Export flattened…",
            "Export",
            self._export,
            QKeySequence("Ctrl+E"),
            "export",
            "Export a flattened PNG, JPEG, WebP, TIFF, BMP, GIF, or PDF",
        )
        self._action(file_menu, "Export animated GIF…", self._export_gif)
        file_menu.addSeparator()
        self._show_folder_action = self._file_action(
            file_menu,
            "Show output folder",
            "Show folder",
            self._show_output_folder,
            None,
            "folder",
            "Show the output folder",
        )
        self._choose_folder_action = self._action(file_menu, "Choose output folder…", self._choose_output_folder)
        file_menu.addSeparator()
        self._close_action = self._file_action(
            file_menu,
            "Close",
            "Close",
            self.close,
            QKeySequence.StandardKey.Close,
            "close",
            "Close PixEdit",
        )
        self._action(file_menu, "Quit", self.close, QKeySequence.StandardKey.Quit)
        self.undo_action = self._action(edit_menu, "Undo", self.session.undo, QKeySequence.StandardKey.Undo)
        self.redo_action = self._action(edit_menu, "Redo", self.session.redo, QKeySequence.StandardKey.Redo)
        self._prepare_history_button(self.undo_action, "Undo", "undo")
        self._prepare_history_button(self.redo_action, "Redo", "redo")
        self._update_output_folder_tooltip()
        self._update_history_actions()

        edit_menu.addSeparator()
        # One action shared by Edit, Select, and Layer so Ctrl+X / Ctrl+C fire once.
        # Line edits accept the shortcut override and still cut or copy text.
        self._cut_action = self._make_action(
            "Cut selection to new layer", self.session.cut_selection_to_layer, QKeySequence("Ctrl+X")
        )
        self._copy_action = self._make_action(
            "Copy selection to new layer", self.session.copy_selection_to_layer, QKeySequence("Ctrl+C")
        )
        self._deselect_action = self._make_action("Deselect", self.session.deselect, QKeySequence("Ctrl+D"))
        # Shared by Edit, Select, and Layer. The icon is set once on the action.
        self._cut_action.setIcon(icon_named("cut-to-layer"))
        self._copy_action.setIcon(icon_named("copy-to-layer"))
        self._deselect_action.setIcon(icon_named("deselect"))
        edit_menu.addAction(self._cut_action)
        edit_menu.addAction(self._copy_action)
        edit_menu.addAction(self._deselect_action)
        # WidgetShortcut displays Delete in the menu. It only fires when the
        # window itself has focus, so a line edit still receives the key.
        # Canvas and this window handle Delete while the user is editing.
        clear_action = self._action(
            edit_menu, "Clear selection contents", self.session.clear_selection_contents, icon="clear-pixels"
        )
        clear_action.setShortcut(QKeySequence(Qt.Key.Key_Delete))
        clear_action.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)

        self._action(image_menu, "Crop to selection", self._crop_selection, icon="crop-to-selection")
        self._action(image_menu, "Trim transparent edges", self.session.trim)
        self._action(image_menu, "Canvas size…", self._canvas_size)
        self._action(image_menu, "Image size…", self._image_size)
        image_menu.addSeparator()
        self._action(image_menu, "Rotate canvas 90° clockwise", lambda: self.session.transform_canvas("rotate-cw"), icon="rotate-cw")
        self._action(image_menu, "Rotate canvas 90° counter-clockwise", lambda: self.session.transform_canvas("rotate-ccw"), icon="rotate-ccw")
        self._action(image_menu, "Flip canvas horizontal", lambda: self.session.transform_canvas("flip-h"), icon="flip-h")
        self._action(image_menu, "Flip canvas vertical", lambda: self.session.transform_canvas("flip-v"), icon="flip-v")
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
        self._action(layer_menu, "New group", self.session.add_group)
        self._action(layer_menu, "Delete group", self.session.delete_active_group)
        self._action(layer_menu, "Place image…", self._place)
        self._action(layer_menu, "Duplicate", self.session.duplicate_active, QKeySequence("Ctrl+J"))
        self._action(layer_menu, "Delete", self.session.delete_active)
        self._action(layer_menu, "Rename…", self._rename_layer)
        layer_menu.addSeparator()
        layer_menu.addAction(self._cut_action)
        layer_menu.addAction(self._copy_action)
        layer_menu.addSeparator()
        self._action(layer_menu, "Apply mask", self.session.apply_active_mask)
        self._action(
            layer_menu, "Merge down", self.session.merge_down_active, QKeySequence("Ctrl+Shift+E"), icon="layer-merge-down"
        )
        self._action(layer_menu, "Flatten image", self.session.flatten)
        layer_menu.addSeparator()
        self._action(layer_menu, "Bring forward", lambda: self.session.move_active(1))
        self._action(layer_menu, "Send backward", lambda: self.session.move_active(-1))
        layer_menu.addSeparator()
        self._action(layer_menu, "Scale…", self._scale_layer, icon="scale")
        self._action(layer_menu, "Rotate…", self._rotate_layer, icon="rotate")
        self._action(layer_menu, "Flip horizontal", self.session.flip_active_horizontal, icon="flip-h")
        self._action(layer_menu, "Flip vertical", self.session.flip_active_vertical, icon="flip-v")

        self._action(select_menu, "All", self.session.select_all, QKeySequence.StandardKey.SelectAll, icon="select-all")
        select_menu.addAction(self._deselect_action)
        self._action(select_menu, "Invert", self.session.invert_selection, QKeySequence("Ctrl+Shift+I"), icon="select-invert")
        select_menu.addSeparator()
        select_menu.addAction(self._cut_action)
        select_menu.addAction(self._copy_action)
        select_menu.addSeparator()
        self._action(select_menu, "Rectangular", lambda: self._select_mode("rect"), icon="select-rect")
        self._action(select_menu, "Elliptical", lambda: self._select_mode("ellipse"), icon="select-ellipse")
        self._action(select_menu, "Freehand", lambda: self._select_mode("freehand"), icon="select-freehand")
        self._action(select_menu, "Magic wand", lambda: self._select_mode("wand"), icon="geom-wand")

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

        self._action(view_menu, "Zoom in", lambda: self.canvas.zoom_at(self.canvas.width() / 2, self.canvas.height() / 2, 1.25), QKeySequence.StandardKey.ZoomIn, icon="zoom")
        self._action(view_menu, "Zoom out", lambda: self.canvas.zoom_at(self.canvas.width() / 2, self.canvas.height() / 2, 0.8), QKeySequence.StandardKey.ZoomOut, icon="zoom-out")
        self._action(view_menu, "Fit", self.canvas.fit, QKeySequence("Ctrl+0"), icon="zoom-fit")
        self._action(view_menu, "Actual pixels", self.canvas.zoom_100, QKeySequence("Ctrl+1"))
        view_menu.addSeparator()
        rulers = self._action(view_menu, "Rulers", self._toggle_rulers)
        rulers.setCheckable(True)
        rulers.setChecked(bool(self.session.option("rulers", True)))
        self._rulers_action = rulers
        snap = self._action(view_menu, "Snap to guides", self._toggle_snap)
        snap.setCheckable(True)
        snap.setChecked(bool(self.session.option("snap", True)))
        self._snap_action = snap
        view_menu.addSeparator()
        self._action(view_menu, "Play frames", self._toggle_play)
        self._action(view_menu, "Stop playback", self._stop_play)

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
        # AboutRole moves this item into the application menu on macOS.
        # On Windows and Linux the role is ignored and it stays under Help.
        # The same action is the About button beside Show folder. The menu
        # keeps the full name; the strip uses the short caption.
        self._about_action = self._action(help_menu, "About PixEdit", self._about)
        self._about_action.setMenuRole(QAction.MenuRole.AboutRole)
        self._about_action.setIcon(icon_named("about"))
        self._about_action.setIconText("About")
        self._about_action.setToolTip("About PixEdit")
        self._about_action.setStatusTip("About PixEdit")
        self._populate_file_toolbar()

    def _file_action(self, menu, label: str, short_label: str, slot, shortcut, icon_name: str, summary: str) -> QAction:
        """A File-menu action that also appears on the file toolbar.

        ``short_label`` is the button caption. The menu keeps ``label``.
        The tooltip includes the shortcut so the strip matches the menu.
        """
        action = self._action(menu, label, slot, shortcut)
        action.setIconText(short_label)
        action.setIcon(icon_named(icon_name))
        keys = action.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
        action.setToolTip(f"{summary} ({keys})" if keys else summary)
        action.setStatusTip(action.toolTip())
        return action

    def _prepare_history_button(self, action: QAction, short: str, icon_name: str) -> None:
        """Toolbar caption and icon. The menu text is set separately and can name the step."""
        action.setIcon(icon_named(icon_name))
        action.setIconText(short)

    def _populate_file_toolbar(self) -> None:
        # New, Open, Close, Save, Save As, Place, Export, then Undo and Redo,
        # then Show folder and About. Undo and Redo are the Edit-menu actions,
        # so each shortcut fires once. About is the Help-menu action. Choosing
        # a new folder stays in the menu.
        self.file_toolbar.clear()
        for action in (
            self._new_action,
            self._open_action,
            self._close_action,
            self._save_action,
            self._save_as_action,
            self._place_action,
            self._export_action,
        ):
            self.file_toolbar.addAction(action)
        self.file_toolbar.addSeparator()
        self.file_toolbar.addAction(self.undo_action)
        self.file_toolbar.addAction(self.redo_action)
        self.file_toolbar.addSeparator()
        self.file_toolbar.addAction(self._show_folder_action)
        self.file_toolbar.addAction(self._about_action)

    def _update_output_folder_tooltip(self) -> None:
        folder = output_folder()
        show_tip = f"Show the output folder ({folder})"
        self._show_folder_action.setToolTip(show_tip)
        self._show_folder_action.setStatusTip(show_tip)
        choose_tip = f"Choose where Open, Place, Save, and Export start ({folder})"
        self._choose_folder_action.setToolTip(choose_tip)
        self._choose_folder_action.setStatusTip(choose_tip)

    def _file_dialog_start(self, filename: str | None = None) -> str:
        """Directory argument for Open, Place, Save As, Export, and Choose.

        Flow:
        1. Create the configured folder when it is missing.
        2. If that fails, say so and start in the home folder instead.
        3. On macOS, point the Finder panel at that directory. It would
           otherwise reopen the last place the user browsed.
        """
        folder = ensure_output_folder()
        if folder is None:
            self._toast(
                f"Could not create the output folder ({output_folder()}). Starting in your home folder."
            )
            folder = Path.home()
        pin_native_dialog_directory(folder)
        return dialog_start(filename, folder=folder)

    def _show_output_folder(self) -> None:
        folder = ensure_output_folder()
        if folder is None:
            self._toast(f"Could not create the output folder ({output_folder()}).")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
            self._toast(f"Could not open {folder}")

    def _choose_output_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose output folder", self._file_dialog_start())
        if not chosen:
            return
        set_output_folder(chosen)
        self._update_output_folder_tooltip()
        self._toast(f"Output folder set to {chosen}")

    def _make_action(self, label: str, slot, shortcut=None) -> QAction:
        action = QAction(label, self)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.triggered.connect(lambda _checked=False: slot())
        self._owned_actions.append(action)
        return action

    def _action(self, menu, label: str, slot, shortcut=None, icon: str | None = None) -> QAction:
        action = self._make_action(label, slot, shortcut)
        if icon:
            action.setIcon(icon_named(icon))
        menu.addAction(action)
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
        # The options panel is rebuilt below. Remember the keyboard first.
        # A focused slider or spin box is destroyed with the old panel, and
        # Qt would move the keyboard onto the color code. The code box itself
        # should also give way, so the next letter is a tool shortcut.
        # The layer list sits outside that panel and keeps the keyboard.
        focus = self.focusWidget()
        return_to_canvas = focus is self.palette.hex_edit or (
            focus is not None and self.options.isAncestorOf(focus)
        )
        self.current = tool
        button = self._buttons.get(tool.id)
        if button is not None:
            button.setChecked(True)
        self.options.set_tool(tool)
        self.canvas.refresh_tool_cursor()
        self._toast(tool.hint or tool.label)
        self._sync_text_panel()
        if return_to_canvas:
            self.canvas.setFocus()

    def _on_session(self, kind: str) -> None:
        if kind == "overlay":
            # The rubber band moved and the photo did not. Rebuilding the
            # composite here would stall the cursor poll that is collecting points.
            self.canvas.update()
            return
        bounds = self.session.preview_bounds if kind == "preview" else None
        self.canvas.invalidate(bounds)
        self._update_status()
        if kind == "preview":
            # Choosing a layer in the stack only previews. The text panel still
            # has to show that layer's font and color. A style drag sets
            # text_edit_in_progress, and the sync below leaves it alone.
            self._sync_text_panel()
            return
        # An eyedropper pick on Image & Eyedrop is a finished color, so it can
        # recolor the active text layer. Every other change refreshes the palette
        # from the session, then loads the active text layer into the panel.
        if kind == "color":
            self.palette.sync_from_session()
            self.palette.note_recent()
            self._restyle_text(transient=False)
            self.options.refresh_brush_preview()
        else:
            self.palette.sync_from_session()
        self.layers.refresh()
        self._update_title()
        self._update_history_actions()
        self.options.update_actions()
        if kind != "color":
            self._sync_text_panel()
        if self.current is not None and self.current.id == "color":
            self.options.reload_from_session()
            self.options.update_actions()

    def _update_history_actions(self) -> None:
        if hasattr(self, "undo_action"):
            undo = self.session.history.undo_stack[-1].name if self.session.history.can_undo() else ""
            self._sync_history_action(self.undo_action, "Undo", self.session.history.can_undo(), undo)
            redo = self.session.history.redo_stack[-1].name if self.session.history.can_redo() else ""
            self._sync_history_action(self.redo_action, "Redo", self.session.history.can_redo(), redo)

    def _sync_history_action(self, action: QAction, verb: str, enabled: bool, step: str) -> None:
        """Menu text names the step. The toolbar button stays the short word.

        ``setText`` does not change ``iconText``, but setting it again keeps a
        later menu update from widening the top bar if the two ever diverge.
        """
        action.setEnabled(enabled)
        action.setText(f"{verb} {step}".strip())
        action.setIconText(verb)
        keys = action.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
        summary = f"{verb} {step}".strip()
        tip = f"{summary} ({keys})" if keys else summary
        action.setToolTip(tip)
        action.setStatusTip(tip)

    def _update_title(self) -> None:
        # A saved .pix project wins. Otherwise show the opened image or PDF
        # name. A new canvas has neither and stays Untitled. With no page yet
        # the title is only the app name.
        if not self.session.document.groups:
            self.setWindowTitle("PixEdit")
            return
        if self.session.path:
            name = Path(self.session.path).name
        elif self.session.source_name:
            name = self.session.source_name
        else:
            name = "Untitled"
        mark = " •" if self.session.dirty else ""
        self.setWindowTitle(f"PixEdit — {name}{mark}")

    def _suggested_project_name(self) -> str:
        """Filename Save As should offer.

        An existing project keeps its name. An opened image or PDF uses that
        stem, so photo.png suggests photo.pix. A new canvas stays untitled.pix.
        """
        if self.session.path:
            return Path(self.session.path).name
        if self.session.source_name:
            stem = Path(self.session.source_name).stem
            if stem:
                return f"{stem}.pix"
        return "untitled.pix"

    def _export_suggestion(self) -> tuple[str, str]:
        """Filename and filter the Export dialog should open with.

        photo.png stays PNG, scan.pdf stays PDF. A .pix project uses its stem
        plus .png. A new canvas stays export.png.
        """
        source = self.session.path or self.session.source_name
        if not source:
            return "export.png", export_filter_for_suffix(".png")
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix in EXPORT_SUFFIXES and path.stem:
            return f"{path.stem}{suffix}", export_filter_for_suffix(suffix)
        stem = path.stem or "export"
        return f"{stem}.png", export_filter_for_suffix(".png")

    def _on_palette_color(self, transient: bool) -> None:
        self.options.refresh_brush_preview()
        self._restyle_text(transient=transient)
        if self.current is not None and self.current.id == "mask":
            self.session.recolor_solid_mask(transient=transient)

    def _restyle_text(self, *, transient: bool) -> None:
        if self.current is None or self.current.id != "text":
            return
        self.session.apply_text_style(transient=transient)

    def _sync_text_panel(self) -> None:
        """Show the active text layer's style in the panel without redrawing it."""
        if self.current is None or self.current.id != "text":
            return
        if self.session.text_edit_in_progress():
            return
        layer = self.session.active_layer()
        if layer is None or layer.kind != "text" or layer.text is None:
            return
        self.session.adopt_text_style(layer.text)
        self.palette.sync_from_session()
        self.options.reload_from_session()

    def _on_panel_action(self, action_id: str) -> None:
        if action_id == "zoom_in":
            self.canvas.zoom_at(self.canvas.width() / 2, self.canvas.height() / 2, 1.25)
            return
        if action_id == "zoom_out":
            self.canvas.zoom_at(self.canvas.width() / 2, self.canvas.height() / 2, 0.8)
            return
        if action_id == "zoom_fit":
            self.canvas.fit()
            return
        if action_id == "zoom_100":
            self.canvas.zoom_100()
            return
        if action_id == "scale_layer":
            self._scale_layer()
            return
        if action_id == "canvas_size":
            self._canvas_size()
            return
        if action_id == "image_size":
            self._image_size()
            return
        if action_id == "rotate_layer":
            self._rotate_layer()
            return
        if action_id == "apply_color":
            self.session.commit_color_adjust()
            return
        if action_id == "reset_color":
            self.session.cancel_color_adjust()
            return
        method = getattr(self.session, action_id, None)
        if method is not None:
            method()

    def _on_tool_option(self, key: str, _value) -> None:
        if key in {"brush_size", "shape_kind", "select_mode", "mask_shape"}:
            self.canvas.refresh_tool_cursor()

    def _update_status(self) -> None:
        doc = self.session.document
        if not doc.groups:
            self._status_label.setText(START_HINT)
            return
        x, y = self._cursor
        tool = self.current.label if self.current else ""
        zoom = int(round(self.canvas.zoom * 100))
        index = doc.group_index(doc.active_group_id) + 1
        if len(doc.groups) > 1:
            page = f"{doc.group.name}  {index} of {len(doc.groups)} · "
        else:
            page = f"{doc.group.name} · "
        self._status_label.setText(
            f"{page}{doc.width} × {doc.height} px · {_format_dpi(doc.dpi)} dpi    {x:.0f}, {y:.0f}    {zoom}%    {tool}"
        )

    def _on_cursor(self, x: float, y: float, zoom: float) -> None:
        self._cursor = (x, y)
        self._update_status()

    def _toast(self, text: str) -> None:
        self.statusBar().showMessage(text, 4000)

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
        self._stop_play()
        dialog = NewDocumentDialog(self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        width, height, background, dpi = dialog.values()
        self.session.new_canvas(width, height, background, dpi)
        QTimer.singleShot(0, self.canvas.fit)

    def _open(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open", self._file_dialog_start(), _OPEN_FILTER)
        if path:
            self.open_path(path)

    def _place(self) -> None:
        """Add an image as new layers. The open document stays in place."""
        if not self.session.require_page():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Place image", self._file_dialog_start(), _PLACE_FILTER)
        if not path:
            return
        self._place_path(path)

    def _place_path(self, path: str) -> None:
        """Place one file the same way File → Place does, without the chooser."""
        pdf_scale = 2.0
        if Path(path).suffix.lower() == ".pdf":
            dialog = PdfImportDialog(self)
            if dialog.exec() != dialog.DialogCode.Accepted:
                return
            pdf_scale = dialog.scale()
        self._stop_play()
        try:
            incoming = load_path(path, pdf_scale=pdf_scale)
        except Exception as exc:
            QMessageBox.warning(self, "Place failed", f"Could not place {path}\n\n{exc}")
            return
        if Path(path).suffix.lower() == ".pdf":
            self.session.place_groups(incoming.groups)
        else:
            self.session.place_layers(incoming.layers)

    def open_path(self, path: str) -> None:
        pdf_scale = 2.0
        if Path(path).suffix.lower() == ".pdf":
            dialog = PdfImportDialog(self)
            if dialog.exec() != dialog.DialogCode.Accepted:
                return
            pdf_scale = dialog.scale()
        self._stop_play()
        try:
            document = load_path(path, pdf_scale=pdf_scale)
        except Exception as exc:
            QMessageBox.warning(self, "Open failed", f"Could not open {path}\n\n{exc}")
            return
        # A .pix file is the save path. An image or PDF only lends its filename,
        # so Save still asks for a project and does not overwrite the source.
        project = Path(path).suffix.lower() == ".pix"
        if project:
            apply_pix_group_names(document.groups, path)
        self.session.replace_document(
            document,
            path if project else None,
            None if project else Path(path).name,
        )
        QTimer.singleShot(0, self.canvas.fit)
        self._toast(f"Opened {Path(path).name}")

    def _save(self) -> bool:
        if not self.session.require_page():
            return False
        self._stop_play()
        if not self.session.path:
            return self._save_as()
        try:
            save_project(self.session.document, self.session.path)
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return False
        self.session.mark_clean()
        self._toast(f"Saved {self.session.path}")
        return True

    def _save_as(self) -> bool:
        if not self.session.require_page():
            return False
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save project",
            self._file_dialog_start(self._suggested_project_name()),
            "PixEdit project (*.pix)",
        )
        if not path:
            return False
        if not path.lower().endswith(".pix"):
            path += ".pix"
        self.session.path = path
        return self._save()

    def _export(self) -> None:
        if not self.session.require_page():
            return
        self._stop_play()
        filename, selected_filter = self._export_suggestion()
        path, selected = QFileDialog.getSaveFileName(
            self,
            "Export flattened image",
            self._file_dialog_start(filename),
            export_filter_string(),
            selected_filter,
        )
        if not path:
            return
        # The filename's extension wins. A name with no known extension takes
        # the extension of the filter the user left selected.
        suffix = Path(path).suffix.lower()
        if suffix not in EXPORT_SUFFIXES:
            path += extension_for_export_filter(selected)
        try:
            export_image(self.session.document, path)
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        note = f"Exported {path}"
        if Path(path).suffix.lower() != ".pdf" and len(self.session.document.groups) > 1:
            note = (
                f"Exported {path}. This file is {self.session.document.group.name}, "
                "the page you are viewing. Export PDF to write every page."
            )
        self._toast(note)

    def _rename_layer(self) -> None:
        self.layers.rename_selection()

    def _scale_layer(self) -> None:
        if not self.session.require_page():
            return
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
        if not self.session.require_page():
            return
        if self.session.active_layer() is None:
            return
        dialog = RotateDialog(self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        self.session.transform_active("rotate", degrees=dialog.degrees.value())

    def _crop_selection(self) -> None:
        self.session.crop_to_selection()

    def _canvas_size(self) -> None:
        if not self.session.require_page():
            return
        document = self.session.document
        dialog = CanvasSizeDialog(document.width, document.height, tuple(self.session.foreground), self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        width, height, anchor, fill = dialog.values()
        self.session.resize_canvas(width, height, anchor, fill)

    def _image_size(self) -> None:
        if not self.session.require_page():
            return
        document = self.session.document
        dialog = ImageSizeDialog(document.width, document.height, document.dpi, self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        width, height, dpi, method, resample = dialog.values()
        self.session.resize_image(width, height, dpi, resample, method)

    def _select_mode(self, mode: str) -> None:
        self.session.options["select_mode"] = mode
        tool = self.tool_by_id.get("select")
        if tool is not None:
            self.set_tool(tool)

    def _filter(self, title: str, specs: list[dict], apply_fn) -> None:
        if not self.session.require_page():
            return
        self.session.cancel_color_adjust()
        layer = self.session.active_layer()
        if layer is None or layer.kind == "redaction":
            self._toast("Choose a raster or text layer.")
            return
        LiveFilterDialog(self, self.session, title, specs, apply_fn).exec()

    def _run_plugin(self, callback) -> None:
        if not self.session.require_page():
            return
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

    def _toggle_snap(self) -> None:
        self.session.options["snap"] = not bool(self.session.option("snap", True))
        if hasattr(self, "_snap_action"):
            self._snap_action.setChecked(bool(self.session.option("snap", True)))
        state = "on" if self.session.option("snap", True) else "off"
        self._toast(f"Snap to guides is {state}.")

    def dragEnterEvent(self, event) -> None:
        if paths_from_drop(event):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event) -> None:
        self._drop_event(event)

    def _drop_event(self, event) -> None:
        paths = paths_from_drop(event)
        if not paths:
            event.ignore()
            return
        self._drop_paths(paths)
        event.acceptProposedAction()

    def _drop_paths(self, paths: list[str]) -> None:
        """Place dropped files on the open document.

        An image or PDF lands on the page on screen. A ``.pix`` project is
        appended as new groups at the end. The document already open stays
        open, including its save path. With no page yet, the first file
        opens a document and any further files in that drop are placed on it.
        """
        self._stop_play()
        pending = list(paths)
        # Nothing is open, so the first file becomes the document.
        if pending and not self.session.document.groups:
            first = pending.pop(0)
            self.open_path(first)
            if not self.session.document.groups:
                return
        for path in pending:
            if Path(path).suffix.lower() == ".pix":
                self._place_project(path)
            else:
                self._place_path(path)

    def _place_project(self, path: str) -> None:
        """Append each page of a ``.pix`` file as a group at the end."""
        try:
            incoming = load_path(path)
        except Exception as exc:
            QMessageBox.warning(self, "Place failed", f"Could not place {path}\n\n{exc}")
            return
        apply_pix_group_names(incoming.groups, path)
        self.session.place_groups(incoming.groups)

    def _toggle_play(self) -> None:
        if self._playing:
            self._stop_play()
            return
        if not self.session.require_page():
            return
        kind = frame_kind(self.session.document)
        if kind is None:
            self._toast("Nothing to play. Open a multi-page file, or a GIF whose frames are hidden layers.")
            return
        self._playing = True
        self._play_kind = kind
        self._play_index = 0
        self._play_return = (
            self.session.document.active_group_id,
            self.canvas.zoom,
            self.canvas.offset_x,
            self.canvas.offset_y,
        )
        self.canvas.letterbox = kind == "pages"
        if kind == "layers":
            self._play_saved = [(layer, layer.visible) for layer in self.session.document.layers]
        self._show_play_frame()
        self._play_timer.start(max(20, int(self.session.option("frame_delay", 100))))

    def _advance_play(self) -> None:
        count = frame_count(self.session.document)
        if not self._playing or count < 2:
            self._stop_play()
            return
        self._play_index = (self._play_index + 1) % count
        self._show_play_frame()

    def _show_play_frame(self) -> None:
        document = self.session.document
        if self._play_kind == "pages":
            group = document.groups[self._play_index % len(document.groups)]
            self.session.activate_group(group.id)
            return
        apply_layer_frame(document, self._play_index)
        self.session.notify("change")

    def _stop_play(self) -> None:
        was = self._playing
        self._play_timer.stop()
        self._playing = False
        self._play_kind = None
        self.canvas.letterbox = False
        saved = self._play_saved
        self._play_saved = None
        if saved is not None:
            for layer, visible in saved:
                layer.visible = visible
        return_to = self._play_return
        self._play_return = None
        if return_to is not None:
            group_id, zoom, offset_x, offset_y = return_to
            if (
                self.session.document.group_by_id(group_id) is not None
                and self.session.document.active_group_id != group_id
            ):
                self.session.activate_group(group_id)
            self.canvas.zoom = zoom
            self.canvas.offset_x = offset_x
            self.canvas.offset_y = offset_y
            self.session.view_zoom = zoom
        if was:
            self.session.notify("change")
            self.canvas.update()

    def _export_gif(self) -> None:
        if not self.session.require_page():
            return
        self._stop_play()
        delay, ok = QInputDialog.getInt(
            self,
            "Animated GIF",
            "Frame delay (ms)",
            int(self.session.option("frame_delay", 100)),
            20,
            2000,
            10,
        )
        if not ok:
            return
        self.session.options["frame_delay"] = int(delay)
        path, _selected = QFileDialog.getSaveFileName(
            self,
            "Export animated GIF",
            self._file_dialog_start("animation.gif"),
            "GIF (*.gif)",
        )
        if not path:
            return
        if not path.lower().endswith(".gif"):
            path += ".gif"
        try:
            notes = export_animated_gif(self.session.document, path, int(delay))
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        note = f"Exported {path}"
        if notes:
            note = f"{note}. Padded {', '.join(notes)} onto the largest frame."
        self._toast(note)

    def _bump_brush(self, delta: int) -> None:
        size = int(self.session.option("brush_size", 24)) + delta
        size = max(1, min(400, size))
        self.session.options["brush_size"] = size
        self.options.sync("brush_size", size)
        self.canvas.refresh_tool_cursor()
        self._toast(f"Brush size {size}")

    def _about(self) -> None:
        AboutDialog(self).exec()

    def keyPressEvent(self, event) -> None:
        # Reaches here when the focused child does not accept the key (the
        # canvas does accept it). Line edits consume Delete themselves.
        if event.key() == Qt.Key.Key_Alt and not event.isAutoRepeat():
            self.canvas.set_alt_held(True)
        if event.key() == Qt.Key.Key_Space:
            # The layer list ignores Space so a row checkbox does not flip.
            # The canvas owns hold-Space as pan. A text field accepts the key
            # before it gets here, so typing a space is unchanged.
            self.canvas.keyPressEvent(event)
            event.accept()
            return
        if _is_clear_shortcut(event):
            self.session.clear_selection_contents()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and not event.isAutoRepeat():
            tool = self.current
            busy = bool(tool.pointer_cancel(self.session)) if tool is not None else False
            if not busy and self.session.document.selection is not None:
                self.session.deselect()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Alt and not event.isAutoRepeat():
            self.canvas.set_alt_held(False)
        if event.key() == Qt.Key.Key_Space:
            self.canvas.keyReleaseEvent(event)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def closeEvent(self, event) -> None:
        if self._confirm_discard():
            event.accept()
        else:
            event.ignore()


class _VersionLabel(QLabel):
    """Status-bar version. A click opens About without disturbing the readout."""

    def __init__(self, version: str, parent=None) -> None:
        super().__init__(version, parent)
        self.setObjectName("statusVersion")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"PixEdit {version}")
        self.setAccessibleName(f"PixEdit {version}")

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            opener = getattr(self.window(), "_about", None)
            if opener is not None:
                opener()
        super().mouseReleaseEvent(event)


def paths_from_drop(event) -> list[str]:
    """Local file paths on a drag event. Empty when the drag is not files."""
    mime = event.mimeData()
    if mime is None or not mime.hasUrls():
        return []
    return [url.toLocalFile() for url in mime.urls() if url.isLocalFile()]


def _tool_separator() -> QFrame:
    line = QFrame()
    line.setObjectName("toolSeparator")
    line.setFixedHeight(1)
    line.setAccessibleName("Tool group separator")
    return line


def _format_dpi(dpi: float) -> str:
    value = float(dpi)
    if abs(value - round(value)) < 0.05:
        return str(int(round(value)))
    return f"{value:.1f}"


def _app_icon() -> QIcon:
    pixmap = QPixmap(64, 64)
    pixmap.fill(QColor("#143544"))
    return QIcon(pixmap)

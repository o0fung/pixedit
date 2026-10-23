"""Console script and a headless window smoke test."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from pixedit.cli import main


def test_version_and_help():
    with pytest.raises(SystemExit) as version:
        main(["--version"])
    assert version.value.code == 0
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0


def test_check_constructs_window():
    assert main(["--check"]) == 0


def test_status_version_and_about_guide():
    """The version stays on the status bar, and About includes the short guide."""
    from PyQt6.QtWidgets import QApplication

    from pixedit import __version__
    from pixedit.ui.dialogs import AboutDialog
    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    assert window._version_label.text() == __version__
    dialog = AboutDialog(window)
    assert dialog.version.text() == __version__
    guide = dialog.body.toPlainText()
    assert "Save writes a" in guide
    assert ".pix" in guide
    window.close()
    app.processEvents()


def test_window_menus_plugins_and_open(tmp_path: Path):
    from pixedit.ui.main_window import MainWindow
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    titles = [action.text() for action in window.menuBar().actions()]
    assert "File" in titles and "Layer" in titles and "Filter" in titles
    labels = []
    for action in window.menuBar().actions():
        menu = action.menu()
        if menu is None:
            continue
        labels.extend(child.text() for child in menu.actions())
    assert any("Vignette" in label for label in labels)
    assert any("Trim" in label for label in labels)
    assert any("Clear selection contents" in label for label in labels)
    assert "example.marker" in window.tool_by_id
    assert "mask" in window.tool_by_id
    for tool_id, button in window._buttons.items():
        assert not button.icon().isNull(), tool_id
        tool = window.tool_by_id[tool_id]
        assert button.text() == tool.button_caption()
        if tool.shortcut:
            assert button.text() == f"{tool.label} ({tool.shortcut})"
        else:
            assert button.text() == tool.label
        tip = button.toolTip()
        assert tool.label in tip
        if tool.shortcut:
            assert tool.shortcut in tip
        if tool.hint:
            assert tool.hint in tip
    path = tmp_path / "photo.png"
    Image.new("RGBA", (12, 9), (20, 30, 40, 255)).save(path)
    window.open_path(str(path))
    app.processEvents()
    assert window.session.document.width == 12
    assert window.session.document.layers[0].image.getpixel((0, 0))[:3] == (20, 30, 40)
    window.session.undo()
    window.session.redo()
    window.close()
    app.processEvents()


def test_canvas_mouse_move_reports_cursor():
    """A hover must report canvas coordinates. Passing ``pos.y`` without calling it aborted the process."""
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    app.processEvents()
    local = QPointF(120, 90)
    move = QMouseEvent(
        QEvent.Type.MouseMove,
        local,
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(window.canvas, move)
    app.processEvents()
    expected_x, expected_y = window.canvas.widget_to_canvas(local.x(), local.y())
    assert window._cursor == pytest.approx((expected_x, expected_y))
    assert f"{expected_x:.0f}, {expected_y:.0f}" in window._status_label.text()
    window.close()
    app.processEvents()


def test_delete_clears_selection_pixels():
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QKeyEvent
    from PyQt6.QtWidgets import QApplication

    from pixedit.tools.base import PointerEvent
    from pixedit.tools.builtins import SelectTool
    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    app.processEvents()
    window.session.new_canvas(24, 24, "white")
    window.session.options["select_mode"] = "rect"
    tool = SelectTool()
    tool.pointer_down(window.session, PointerEvent(4, 4))
    tool.pointer_up(window.session, PointerEvent(12, 12))
    window.canvas.setFocus()
    app.processEvents()
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(window.canvas, event)
    app.processEvents()
    image = window.session.active_layer().image
    assert image.getpixel((6, 6))[3] == 0
    assert image.getpixel((1, 1))[3] == 255
    assert image.size == (24, 24)
    window.session.dirty = False
    window.close()
    app.processEvents()


def _menu(window, title: str):
    for action in window.menuBar().actions():
        if action.text() == title and action.menu() is not None:
            return action.menu()
    raise AssertionError(title)


def _shortcut_text(action, native: bool) -> str:
    from PyQt6.QtGui import QKeySequence

    form = QKeySequence.SequenceFormat.NativeText if native else QKeySequence.SequenceFormat.PortableText
    return action.shortcut().toString(form)


def test_file_toolbar_and_tool_groups():
    from PyQt6.QtGui import QKeySequence
    from PyQt6.QtWidgets import QApplication, QToolButton

    from pixedit.output_folder import output_folder
    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(1400, 700)
    window.show()
    app.processEvents()
    kinds = ["|" if action.isSeparator() else action.iconText() for action in window.file_toolbar.actions()]
    assert kinds == [
        "New",
        "Open",
        "Close",
        "Save",
        "Save As",
        "Place",
        "Export",
        "|",
        "Undo",
        "Redo",
        "|",
        "Show folder",
    ]
    actions = [action for action in window.file_toolbar.actions() if not action.isSeparator()]
    file_actions = [action for action in actions if action not in (window.undo_action, window.redo_action)]
    assert [action.text() for action in file_actions] == [
        "New…",
        "Open…",
        "Close",
        "Save project",
        "Save project as…",
        "Place…",
        "Export flattened…",
        "Show output folder",
    ]
    file_menu = _menu(window, "File")
    by_text = {action.text(): action for action in file_menu.actions()}
    folder = str(output_folder())
    for action in file_actions:
        assert by_text[action.text()] is action
        assert not action.icon().isNull()
        if action is window._show_folder_action:
            assert folder in action.toolTip()
            assert action.shortcut().isEmpty()
        else:
            # The tooltip shows the platform shortcut, ⌘ on macOS and Ctrl elsewhere.
            assert _shortcut_text(action, native=True) in action.toolTip()
    visible = [action.text() for action in file_menu.actions() if action.text()]
    assert visible[-4:] == ["Show output folder", "Choose output folder…", "Close", "Quit"]
    assert window._choose_folder_action not in window.file_toolbar.actions()
    assert folder in window._choose_folder_action.toolTip()
    assert _shortcut_text(window._new_action, native=False) == "Ctrl+N"
    assert _shortcut_text(window._open_action, native=False) == "Ctrl+O"
    # Offscreen Qt reports Close as Ctrl+F4. The Mac app uses ⌘W.
    # Either string is StandardKey.Close.
    close_keys = QKeySequence(QKeySequence.StandardKey.Close).toString(QKeySequence.SequenceFormat.PortableText)
    assert _shortcut_text(window._close_action, native=False) == close_keys
    assert _shortcut_text(window._save_action, native=False) == "Ctrl+S"
    assert _shortcut_text(window._save_as_action, native=False) == "Ctrl+Shift+S"
    assert _shortcut_text(window._place_action, native=False) == "Ctrl+Shift+O"
    assert _shortcut_text(window._export_action, native=False) == "Ctrl+E"
    assert window.file_toolbar.isVisible()
    buttons = []
    for action in window.file_toolbar.actions():
        if action.isSeparator():
            continue
        buttons.extend(
            widget for widget in action.associatedObjects() if isinstance(widget, QToolButton) and widget.text()
        )
    assert {button.text() for button in buttons} == {
        "New",
        "Open",
        "Close",
        "Save",
        "Save As",
        "Place",
        "Export",
        "Undo",
        "Redo",
        "Show folder",
    }
    assert not window.undo_action.icon().isNull()
    assert not window.redo_action.icon().isNull()
    assert window.undo_action.iconText() == "Undo"
    assert window.redo_action.iconText() == "Redo"
    assert not window.undo_action.isEnabled()
    assert not window.redo_action.isEnabled()
    assert _shortcut_text(window.undo_action, native=True) in window.undo_action.toolTip()
    assert _shortcut_text(window.redo_action, native=True) in window.redo_action.toolTip()
    edit_by_text = {action.text(): action for action in _menu(window, "Edit").actions()}
    assert edit_by_text["Undo"] is window.undo_action
    assert edit_by_text["Redo"] is window.redo_action
    # The menu names the step. The button stays the short word.
    window.session.set_opacity(window.session.active_layer(), 0.4)
    app.processEvents()
    assert window.undo_action.text() == "Undo Layer opacity"
    assert window.undo_action.iconText() == "Undo"
    assert window.undo_action.isEnabled()
    assert "Layer opacity" in window.undo_action.toolTip()
    buttons = []
    for action in window.file_toolbar.actions():
        if action.isSeparator():
            continue
        buttons.extend(
            widget for widget in action.associatedObjects() if isinstance(widget, QToolButton) and widget.text()
        )
    assert "Undo" in {button.text() for button in buttons}
    assert "Undo Layer opacity" not in {button.text() for button in buttons}
    window.session.undo()
    app.processEvents()
    assert not window.undo_action.isEnabled()
    assert window.redo_action.text() == "Redo Layer opacity"
    assert window.redo_action.iconText() == "Redo"
    assert window.redo_action.isEnabled()

    layer_labels = [action.text() for action in _menu(window, "Layer").actions()]
    select_labels = [action.text() for action in _menu(window, "Select").actions()]
    edit_labels = [action.text() for action in _menu(window, "Edit").actions()]
    for label in ("Cut selection to new layer", "Copy selection to new layer"):
        assert label in edit_labels
        assert label in select_labels
        assert label in layer_labels
    assert "Deselect" in edit_labels
    assert "Deselect" in select_labels
    assert "Place image…" in layer_labels
    assert "Place…" in [action.text() for action in file_menu.actions()]

    order = []
    for index in range(window.tool_layout.count()):
        widget = window.tool_layout.itemAt(index).widget()
        if widget is None:
            continue
        if widget.objectName() == "toolSeparator":
            order.append("|")
        elif isinstance(widget, QToolButton):
            order.append(widget.text())
    assert order == [
        "Transform (V)",
        "Hand (H)",
        "Zoom (Z)",
        "Select (M)",
        "Color (I)",
        "|",
        "Brush (B)",
        "Eraser (E)",
        "Fill (G)",
        "Shape (U)",
        "Text (T)",
        "|",
        "Crop (C)",
        "Mask (R)",
        "|",
        "Marker",
    ]
    window.session.dirty = False
    window.close()
    app.processEvents()


def test_opacity_slider_and_brush_cursor():
    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.canvas import brush_circle_cursor
    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    cursor = brush_circle_cursor(30)
    pixmap = cursor.pixmap()
    assert not pixmap.isNull()
    ratio = pixmap.devicePixelRatio() or 1.0
    assert pixmap.width() / ratio >= 30
    hotspot = cursor.hotSpot()
    assert abs(hotspot.x() - (pixmap.width() / ratio) / 2) <= 2
    assert abs(hotspot.y() - (pixmap.height() / ratio) / 2) <= 2

    window = MainWindow()
    window.resize(900, 700)
    window.show()
    app.processEvents()
    window.palette.opacity_slider.setValue(40)
    app.processEvents()
    assert window.session.foreground[3] == round(40 * 255 / 100)
    assert "40%" in window.palette.opacity_label.text()
    assert window.palette.hex_edit.text().startswith("#")

    window.set_tool(window.tool_by_id["eraser"])
    window.session.options["brush_size"] = 40
    window.options.sync("brush_size", 40)
    assert window.options.brush_preview is not None
    eraser_preview = window.options.brush_preview.pixmap().toImage()
    assert eraser_preview.pixelColor(2, 2).alpha() > eraser_preview.pixelColor(88, 48).alpha()
    window.canvas.zoom = 2.0
    window.canvas.refresh_tool_cursor()
    brush_cursor = window.canvas.cursor()
    brush_pixmap = brush_cursor.pixmap()
    brush_ratio = brush_pixmap.devicePixelRatio() or 1.0
    assert brush_pixmap.width() / brush_ratio >= 70
    window.set_tool(window.tool_by_id["transform"])
    assert window.canvas.cursor().pixmap().isNull()
    assert window.options.brush_preview is None

    window.set_tool(window.tool_by_id["brush"])
    window.session.foreground = (255, 0, 0, 255)
    window.options._widgets["brush_size"].setValue(48)
    window.options._widgets["brush_hardness"].setValue(100)
    app.processEvents()
    assert window.options.brush_preview is not None
    hard_red = window.options.brush_preview.pixmap().toImage().pixelColor(108, 48).red()
    window.options._widgets["brush_hardness"].setValue(0)
    app.processEvents()
    soft_red = window.options.brush_preview.pixmap().toImage().pixelColor(108, 48).red()
    assert hard_red > soft_red + 40
    window.session.foreground = (20, 24, 28, 255)
    window.session.options["brush_size"] = 10
    window.canvas.zoom = 2.0
    window.canvas.refresh_tool_cursor()
    small = window.canvas.cursor().pixmap().width()
    window._bump_brush(30)
    large = window.canvas.cursor().pixmap().width()
    assert large > small + 20
    window.session.dirty = False
    window.close()
    app.processEvents()


def test_escape_deselects_when_idle_and_cancels_a_shape_first():
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QKeyEvent
    from PyQt6.QtWidgets import QApplication

    from pixedit.tools.base import PointerEvent
    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    app.processEvents()
    window.session.new_canvas(40, 40, "transparent")
    from PIL import Image

    mask = Image.new("L", (40, 40), 0)
    mask.paste(255, (4, 4, 16, 16))
    window.session.set_selection(mask)
    window.set_tool(window.tool_by_id["shape"])
    window.session.foreground = (255, 0, 0, 255)
    window.session.options["shape_kind"] = "rect"
    window.session.options["shape_filled"] = True
    shape = window.current
    shape.pointer_down(window.session, PointerEvent(8, 8))
    shape.pointer_move(window.session, PointerEvent(28, 24))
    # The selection clips the preview, so sample a pixel that actually painted.
    layer = window.session.active_layer()
    box = layer.image.getchannel("A").getbbox()
    assert box is not None
    sample_x = layer.x + box[0]
    sample_y = layer.y + box[1]
    assert layer.image.getpixel((box[0], box[1]))[0] > 200
    window.canvas.setFocus()
    app.processEvents()
    escape = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(window.canvas, escape)
    app.processEvents()
    assert window.session.document.selection is not None
    assert window.session.active_layer().image.getpixel((sample_x, sample_y))[3] == 0
    QApplication.sendEvent(window.canvas, escape)
    app.processEvents()
    assert window.session.document.selection is None
    window.session.dirty = False
    window.close()
    app.processEvents()


def test_parse_hex():
    from pixedit.ui.palette import parse_hex

    assert parse_hex("#0F0") == (0, 255, 0, None)
    assert parse_hex("#112233") == (0x11, 0x22, 0x33, None)
    assert parse_hex("11223380") == (0x11, 0x22, 0x33, 0x80)
    assert parse_hex("nope") is None


def _mouse_click(widget, x: float, y: float) -> None:
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtWidgets import QApplication

    pos = QPointF(x, y)
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        pos,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        pos,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(widget, press)
    QApplication.sendEvent(widget, release)


def test_palette_text_select_and_zoom_panels():
    from PyQt6.QtWidgets import QApplication

    from pixedit.model import TextInfo
    from pixedit.tools.base import PointerEvent
    from pixedit.tools.builtins import ColorTool
    from pixedit.ui.dialogs import TextDialog
    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(1100, 800)
    window.show()
    app.processEvents()

    ColorTool().pointer_down(window.session, PointerEvent(4, 4))
    app.processEvents()
    assert window.session.foreground[:3] == (255, 255, 255)
    assert window.palette.hex_edit.text().upper() == "#FFFFFF"

    pad = window.palette.sv_pad
    pad.resize(100, 120)
    app.processEvents()
    window.palette.set_hsv(0.0, 0.0, 0.0, transient=False)
    _mouse_click(pad, pad.width() - 1, 1)
    red, green, blue, _alpha = window.session.foreground
    assert red > 240 and green < 30 and blue < 30

    window.palette.note_recent()
    red_color = tuple(window.session.foreground)
    window.palette.set_hsv(0.33, 1.0, 1.0, transient=False)
    window.palette.note_recent()
    window.palette.recent_buttons[1].click()
    app.processEvents()
    assert tuple(window.session.foreground) == red_color

    window.palette.hex_edit.setText("#0000FF")
    window.palette.hex_edit.editingFinished.emit()
    assert window.session.foreground[:3] == (0, 0, 255)

    dialog = TextDialog(TextInfo("Hi", "/font.ttf", "Demo", 30, (1, 2, 3, 4), 12, "justify", 18))
    dialog.text.setPlainText("Yo")
    kept = dialog.result_info()
    assert kept.text == "Yo"
    assert kept.font_path == "/font.ttf"
    assert kept.font_family == "Demo"
    assert kept.size == 30
    assert kept.color == (1, 2, 3, 4)
    assert kept.rotation == 12
    assert kept.align == "justify"
    assert kept.line_spacing == 18

    window.session.add_text(TextInfo("Hi", None, "Default", 28, (255, 0, 0, 255), 200), 8, 8)
    window.set_tool(window.tool_by_id["text"])
    app.processEvents()
    assert window.options._labels["text_font"].text() == "Font"
    assert window.options._labels["text_style"].text() == "Style"
    assert window.options._labels["text_align"].text() == "Align"
    assert window.options._widgets["text_align"].findData("justify") >= 0
    assert window.options._labels["text_size"].text() == "Size"
    assert window.options._labels["text_line_spacing"].text().startswith("Line spacing")
    assert window.options._labels["text_rotation"].text().startswith("Rotation")
    assert "°" in window.options._labels["text_rotation"].text()
    assert window.session.options["text_rotation"] == 200
    assert window.session.active_layer().text.rotation == 200
    assert window.session.foreground[:3] == (255, 0, 0)
    steps = len(window.session.history.undo_stack)
    window.session.apply_text_style(transient=False)
    assert len(window.session.history.undo_stack) == steps
    layer = window.session.active_layer()
    center = (layer.x + layer.image.width / 2, layer.y + layer.image.height / 2)
    window.options._widgets["text_size"].setValue(72)
    app.processEvents()
    layer = window.session.active_layer()
    assert layer.text is not None and layer.text.size == 72
    assert abs((layer.x + layer.image.width / 2) - center[0]) <= 1
    assert abs((layer.y + layer.image.height / 2) - center[1]) <= 1
    window.session.undo()
    assert window.session.active_layer().text.size == 28

    window.palette.set_hsv(0.33, 1.0, 1.0, transient=False)
    app.processEvents()
    assert window.session.active_layer().text.color[1] > 200
    window.session.undo()
    assert window.session.active_layer().text.color[0] == 255

    window.set_tool(window.tool_by_id["brush"])
    kept_color = window.session.active_layer().text.color
    window.palette.set_hsv(0.0, 1.0, 1.0, transient=False)
    assert window.session.active_layer().text.color == kept_color
    assert window.session.foreground[0] > 240

    window.set_tool(window.tool_by_id["select"])
    app.processEvents()
    labels = [button.text() for button in window.options.action_buttons]
    assert labels == [
        "Select all",
        "Deselect",
        "Invert",
        "Cut to layer",
        "Copy to layer",
        "Clear pixels",
        "Crop to selection",
    ]
    by_label = {button.text(): button for button in window.options.action_buttons}
    assert by_label["Select all"].isEnabled()
    assert not by_label["Deselect"].isEnabled()
    assert not by_label["Crop to selection"].isEnabled()
    window.session.select_all()
    app.processEvents()
    assert by_label["Deselect"].isEnabled()
    by_label["Deselect"].click()
    app.processEvents()
    assert window.session.document.selection is None

    from PIL import Image

    def _select_rect() -> None:
        window.session.new_canvas(20, 20, "white")
        mask = Image.new("L", (20, 20), 0)
        mask.paste(255, (2, 3, 10, 12))
        window.session.set_selection(mask)

    _select_rect()
    window._crop_selection()
    menu_size = (window.session.document.width, window.session.document.height)
    _select_rect()
    window.set_tool(window.tool_by_id["select"])
    by_label = {button.text(): button for button in window.options.action_buttons}
    assert by_label["Crop to selection"].isEnabled()
    by_label["Crop to selection"].click()
    app.processEvents()
    assert (window.session.document.width, window.session.document.height) == menu_size == (8, 9)

    window.set_tool(window.tool_by_id["zoom"])
    app.processEvents()
    zoom_labels = [button.text() for button in window.options.action_buttons]
    assert zoom_labels == ["Zoom in", "Zoom out", "Fit", "100%"]
    zoom_buttons = {button.text(): button for button in window.options.action_buttons}
    window.canvas.zoom = 2.0
    zoom_buttons["100%"].click()
    assert window.canvas.zoom == 1.0
    zoom_buttons["Zoom in"].click()
    assert window.canvas.zoom > 1.0
    zoom_buttons["Fit"].click()
    assert window.canvas.zoom > 0

    window.session.dirty = False
    window.close()
    app.processEvents()


def test_transform_color_and_mask_panels():
    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(1100, 800)
    window.show()
    app.processEvents()

    window.set_tool(window.tool_by_id["transform"])
    labels = [button.text() for button in window.options.action_buttons]
    assert "Flip horizontal" in labels
    assert "Flip vertical" in labels
    assert "Rotate 90° clockwise" in labels
    assert "Rotate 90° counter-clockwise" in labels
    assert "Scale…" in labels
    assert "Rotate…" in labels
    buttons = {button.text(): button for button in window.options.action_buttons}
    buttons["Flip horizontal"].click()
    app.processEvents()
    assert window.session.history.undo_stack[-1].name == "Flip horizontal"

    window.set_tool(window.tool_by_id["color"])
    apply = next(button for button in window.options.action_buttons if button.text() == "Apply")
    reset = next(button for button in window.options.action_buttons if button.text() == "Reset")
    assert not apply.isEnabled()
    assert not reset.isEnabled()

    window.set_tool(window.tool_by_id["mask"])
    combo = window.options._widgets["mask_kind"]
    assert [combo.itemText(index) for index in range(combo.count())] == [
        "Pixelate",
        "Blur",
        "Solid",
        "Noise",
        "Desaturate",
        "Invert",
        "Dim",
        "Sharpen",
    ]
    assert window.options._widgets["mask_amount"].isVisible()
    assert window.options._labels["mask_amount"].text().startswith("Block")
    combo.setCurrentText("Solid")
    app.processEvents()
    assert not window.options._widgets["mask_amount"].isVisible()
    combo.setCurrentText("Blur")
    app.processEvents()
    assert window.options._widgets["mask_amount"].isVisible()
    assert window.options._labels["mask_amount"].text().startswith("Radius")

    window.session.dirty = False
    window.close()
    app.processEvents()


def test_close_quits_instead_of_opening_a_canvas():
    """Close ends the app. A clean picture closes with no dialog and stays as it was."""
    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    window.session.new_canvas(17, 19, "white")
    document = window.session.document
    assert not window.session.dirty
    assert "Close PixEdit" in window._close_action.toolTip()

    window._close_action.trigger()
    app.processEvents()

    assert not window.isVisible()
    assert window.session.document is document
    assert document.width == 17 and document.height == 19


def test_close_cancel_keeps_unsaved_picture(monkeypatch):
    """Unsaved work asks first. Cancel leaves the window and the picture in place."""
    from PyQt6.QtWidgets import QApplication, QMessageBox

    from pixedit.ui import main_window as main_window_module
    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    window.session.new_canvas(21, 13, "white")
    window.session.dirty = True
    document = window.session.document

    def cancel(*_args, **_kwargs):
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(main_window_module.QMessageBox, "question", cancel)
    window._close_action.trigger()
    app.processEvents()

    assert window.isVisible()
    assert window.session.document is document
    assert window.session.dirty
    assert document.width == 21 and document.height == 13
    window.session.dirty = False
    window.close()
    app.processEvents()


def test_transform_buttons_stay_inside_the_tool_column():
    """Long Transform labels stay inside the column, and the splitter can widen it."""
    from PyQt6.QtWidgets import QSplitter

    from pixedit.app import create_app
    from pixedit.ui.main_window import MainWindow

    app = create_app()
    window = MainWindow()
    window.resize(1280, 820)
    window.show()
    app.processEvents()
    splitter = window.findChild(QSplitter)
    left = splitter.widget(0)
    assert left.width() == 248
    window.set_tool(window.tool_by_id["transform"])
    app.processEvents()
    buttons = {button.text(): button for button in window.options.action_buttons}

    def row_top(button) -> int:
        return button.mapTo(left, button.rect().topLeft()).y()

    # The styled flip labels are wider than half of 248, so each has its own row.
    # The two rotate commands need still more, and Scale… / Rotate… fit together.
    assert row_top(buttons["Flip horizontal"]) != row_top(buttons["Flip vertical"])
    assert row_top(buttons["Rotate 90° clockwise"]) != row_top(buttons["Rotate 90° counter-clockwise"])
    assert row_top(buttons["Scale…"]) == row_top(buttons["Rotate…"])
    _assert_buttons_inside(window.options.action_buttons, left)
    splitter.setSizes([420, 500, 280])
    app.processEvents()
    assert left.width() > 248
    buttons = {button.text(): button for button in window.options.action_buttons}
    assert row_top(buttons["Flip horizontal"]) == row_top(buttons["Flip vertical"])
    assert row_top(buttons["Rotate 90° clockwise"]) == row_top(buttons["Rotate 90° counter-clockwise"])
    _assert_buttons_inside(window.options.action_buttons, left)
    window.session.dirty = False
    window.close()
    app.processEvents()


def _assert_buttons_inside(buttons, left) -> None:
    boxes = []
    for button in buttons:
        top = button.mapTo(left, button.rect().topLeft())
        bottom = button.mapTo(left, button.rect().bottomRight())
        assert top.x() >= 0, button.text()
        assert bottom.x() < left.width(), button.text()
        assert bottom.y() > top.y(), button.text()
        assert button.width() >= button.sizeHint().width(), button.text()
        boxes.append((top.x(), top.y(), bottom.x(), bottom.y(), button.text()))
    for index, box in enumerate(boxes):
        for other in boxes[index + 1 :]:
            separated = box[2] < other[0] or other[2] < box[0] or box[3] < other[1] or other[3] < box[1]
            assert separated, (box[4], other[4])


def test_layer_list_draws_background_and_opened_project(tmp_path):
    """The default Background row is drawn, and every layer of an opened project is too."""
    from PIL import Image
    from PyQt6.QtWidgets import QApplication

    from pixedit.factory import raster_layer, redaction_layer, text_layer
    from pixedit.io import save_project
    from pixedit.model import Document, TextInfo
    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(1280, 820)
    window.show()
    app.processEvents()
    layers = window.layers.list
    assert layers.count() == 1
    assert layers.item(0).text().startswith("Background")
    assert layers.visualItemRect(layers.item(0)).height() > 0

    document = Document(
        40,
        30,
        [
            raster_layer("Photo", Image.new("RGBA", (40, 30), (10, 20, 30, 255))),
            text_layer("Caption", TextInfo("Hi", None, "Default", 18, (255, 0, 0, 255)), 2, 2),
            redaction_layer("Blur", Image.new("L", (40, 30), 0), "blur", 8),
        ],
    )
    path = tmp_path / "stack.pix"
    save_project(document, str(path))
    window.open_path(str(path))
    app.processEvents()
    names = []
    for index in range(layers.count()):
        item = layers.item(index)
        assert layers.visualItemRect(item).height() > 0
        names.append(item.text().split()[0])
    assert names == ["Blur", "Caption", "Photo"]
    window.session.dirty = False
    window.close()
    app.processEvents()

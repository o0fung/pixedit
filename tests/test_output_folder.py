"""Output folder path, persistence, and the macOS file-panel pin."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pixedit.output_folder import (
    _PANEL_KEYS,
    _panel_bookmark,
    _panel_directory,
    _set_panel_bookmark,
    default_output_folder,
    dialog_start,
    ensure_output_folder,
    output_folder,
    pin_native_dialog_directory,
    set_output_folder,
)


def _app():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_default_folder_and_round_trip(tmp_path: Path):
    _app()
    assert default_output_folder().parts[-2:] == ("output", "PixEdit")
    assert output_folder() == default_output_folder()
    custom = tmp_path / "exports"
    set_output_folder(custom)
    assert output_folder() == custom
    # Naming the dialog start does not create the folder. The dialog does.
    assert not custom.exists()
    assert dialog_start() == str(custom)
    assert dialog_start("untitled.pix") == str(custom / "untitled.pix")
    assert dialog_start("export.png") == str(custom / "export.png")
    assert ensure_output_folder() == custom
    assert custom.is_dir()


def test_removed_folder_is_created_again(tmp_path: Path):
    _app()
    custom = tmp_path / "gone"
    set_output_folder(custom)
    custom.mkdir()
    custom.rmdir()
    assert ensure_output_folder() == custom
    assert custom.is_dir()


def test_file_blocking_the_folder_is_rejected(tmp_path: Path):
    _app()
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("x")
    set_output_folder(blocker)
    assert ensure_output_folder() is None


def test_pin_skips_other_platforms_and_missing_folders(tmp_path: Path, monkeypatch):
    def refuse(folder: Path) -> None:
        raise AssertionError(folder)

    monkeypatch.setattr("pixedit.output_folder.sys.platform", "linux")
    monkeypatch.setattr("pixedit.output_folder._write_macos_panel_directory", refuse)
    pin_native_dialog_directory(tmp_path)

    monkeypatch.setattr("pixedit.output_folder.sys.platform", "darwin")
    called: list[Path] = []
    monkeypatch.setattr("pixedit.output_folder._write_macos_panel_directory", called.append)
    pin_native_dialog_directory(tmp_path / "missing")
    assert called == []


def test_window_dialogs_start_in_the_output_folder(tmp_path: Path, monkeypatch):
    from pixedit.ui.main_window import MainWindow

    chosen = tmp_path / "chosen"
    set_output_folder(chosen)
    pinned: list[Path] = []
    monkeypatch.setattr("pixedit.ui.main_window.pin_native_dialog_directory", pinned.append)
    app = _app()
    window = MainWindow()
    assert app is not None
    assert window._file_dialog_start() == str(chosen)
    assert window._file_dialog_start("untitled.pix") == str(chosen / "untitled.pix")
    assert window._file_dialog_start("export.png") == str(chosen / "export.png")
    assert chosen.is_dir()
    assert pinned == [chosen, chosen, chosen]

    opened: list[str] = []
    monkeypatch.setattr(
        "pixedit.ui.main_window.QDesktopServices.openUrl",
        lambda url: opened.append(url.toLocalFile()) or True,
    )
    window._show_output_folder()
    assert opened == [str(chosen)]


def test_dialog_falls_back_to_home_when_the_folder_cannot_be_created(tmp_path: Path, monkeypatch):
    from pixedit.ui.main_window import MainWindow

    blocker = tmp_path / "blocked"
    blocker.write_text("x")
    set_output_folder(blocker)
    pinned: list[Path] = []
    monkeypatch.setattr("pixedit.ui.main_window.pin_native_dialog_directory", pinned.append)
    app = _app()
    window = MainWindow()
    assert app is not None
    assert window._file_dialog_start("export.png") == str(Path.home() / "export.png")
    assert pinned == [Path.home()]
    assert "Could not create the output folder" in window.statusBar().currentMessage()
    window._show_output_folder()
    assert "Could not create the output folder" in window.statusBar().currentMessage()
    assert pinned == [Path.home()]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS file panel")
def test_pin_points_the_finder_panel_at_the_folder(tmp_path: Path):
    """Point the panel at a temp folder, then put the previous bookmarks back.

    The keys live in this Python process's preferences, which is the same
    file other ``python`` launches read. The restore runs even when the
    assertion fails.
    """
    folder = tmp_path / "pinned"
    folder.mkdir()
    before = {key: _panel_bookmark(key) for key in _PANEL_KEYS}
    mismatch = ""
    try:
        pin_native_dialog_directory(folder)
        for key in _PANEL_KEYS:
            opened = _panel_directory(key)
            if opened is None or Path(opened).resolve() != folder.resolve():
                mismatch = f"{key} opened {opened}"
                break
    finally:
        for key, blob in before.items():
            _set_panel_bookmark(key, blob)
    assert {key: _panel_bookmark(key) for key in _PANEL_KEYS} == before
    assert mismatch == ""

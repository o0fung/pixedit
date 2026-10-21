"""Qt application entry used by the ``pix`` console script."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import QApplication

from pixedit import __version__


def _stylesheet() -> str:
    path = Path(__file__).resolve().parent / "resources" / "style.qss"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return ""


def _icon() -> QIcon:
    pixmap = QPixmap(32, 32)
    pixmap.fill(QColor("#143544"))
    return QIcon(pixmap)


def create_app() -> QApplication:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("PixEdit")
    app.setApplicationVersion(__version__)
    app.setStyle("Fusion")
    style = _stylesheet()
    if style:
        app.setStyleSheet(style)
    app.setWindowIcon(_icon())
    return app


def run(path: str | None = None) -> int:
    """Show the editor and enter the Qt event loop."""
    from pixedit.ui.main_window import MainWindow

    app = create_app()
    window = MainWindow()
    window.show()
    if path:
        window.open_path(path)
    return int(app.exec())


def check() -> int:
    """Construct the window offscreen and return. Used by ``pix --check``."""
    from pixedit.ui.main_window import MainWindow

    app = create_app()
    window = MainWindow()
    window.show()
    app.processEvents()
    if window.session.document.width < 1 or not window.tool_by_id:
        return 1
    window.close()
    app.processEvents()
    return 0

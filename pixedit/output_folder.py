"""Where Open, Place, Save As, and Export dialogs start.

The folder is remembered in an ini file. Tests point ``PIXEDIT_CONFIG`` at a
temporary file so they do not read or write the real one. The directory is
created the first time a dialog needs it, not when the window opens.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

from PyQt6.QtCore import QSettings, QStandardPaths

_SETTINGS_KEY = "outputDirectory"
_CONFIG_ENV = "PIXEDIT_CONFIG"
# AppKit's open/save panel reopens whichever of these was written last.
# The current key is NSOSPLastRootDirectory; NSNavLastRootDirectory is the
# older name still consulted on some systems.
_PANEL_KEYS = ("NSOSPLastRootDirectory", "NSNavLastRootDirectory")


def default_output_folder() -> Path:
    """``Documents/output/PixEdit`` under the platform Documents directory."""
    documents = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
    root = Path(documents) if documents else Path.home() / "Documents"
    return root / "output" / "PixEdit"


def output_folder() -> Path:
    """Configured folder, or the default. Does not create it."""
    raw = _settings().value(_SETTINGS_KEY, "")
    if isinstance(raw, str) and raw.strip():
        folder = Path(raw.strip()).expanduser()
        if not folder.is_absolute():
            folder = Path.cwd() / folder
        return folder
    return default_output_folder()


def set_output_folder(path: str | Path) -> None:
    """Remember ``path``. A relative path is stored from the current directory."""
    folder = Path(path).expanduser()
    if not folder.is_absolute():
        folder = Path.cwd() / folder
    config = Path(_config_path())
    config.parent.mkdir(parents=True, exist_ok=True)
    settings = _settings()
    settings.setValue(_SETTINGS_KEY, str(folder))
    settings.sync()


def ensure_output_folder() -> Path | None:
    """Create the configured folder, including any missing parents.

    A folder the user removed after choosing it is created again on the
    next dialog. Returns None when the path is a file, or cannot be created,
    so the caller can fall back to the home directory.
    """
    folder = output_folder()
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    if not folder.is_dir():
        return None
    return folder


def dialog_start(filename: str | None = None, folder: Path | None = None) -> str:
    """Path a file dialog should open.

    Open and Place pass no filename and get the folder. Save As and Export
    pass a suggested filename, such as ``photo.pix`` or ``photo.png``. A new
    canvas still uses ``untitled.pix`` and ``export.png``.
    """
    root = output_folder() if folder is None else folder
    if filename:
        return str(root / filename)
    return str(root)


def pin_native_dialog_directory(folder: Path) -> None:
    """Make the macOS Finder dialog open in ``folder``.

    The native panel ignores QFileDialog's directory and reopens the last
    place the user browsed. Windows and Linux honor that directory, so this
    is a no-op there. ``folder`` has to exist; a missing path is skipped
    rather than writing a bookmark the panel cannot resolve.
    """
    if sys.platform != "darwin" or not folder.is_dir():
        return
    try:
        _write_macos_panel_directory(folder)
    except Exception:
        # The dialog still receives the directory argument. A failure here
        # only means Finder may reopen the previous folder.
        return


def _config_path() -> str:
    override = os.environ.get(_CONFIG_ENV)
    if override:
        return override
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation)
    folder = Path(base) if base else Path.home() / ".config" / "PixEdit"
    return str(folder / "pixedit.ini")


def _settings() -> QSettings:
    return QSettings(_config_path(), QSettings.Format.IniFormat)


def _write_macos_panel_directory(folder: Path) -> None:
    """Store ``folder`` where NSOpenPanel reads its starting directory.

    ``setURL:forKey:`` writes a bookmark, which is the form those keys
    already use. Both the current and the legacy key are updated so the
    panel cannot prefer a stale one.
    """
    bridge = _PanelBridge.shared()
    pool = bridge.autorelease_pool()
    try:
        url = bridge.directory_url(folder)
        defaults = bridge.standard_defaults()
        for key in _PANEL_KEYS:
            bridge.set_url(defaults, url, key)
        bridge.synchronize(defaults)
    finally:
        bridge.drain(pool)


def _panel_bookmark(key: str) -> bytes | None:
    """Raw bookmark for ``key``, or None when the panel has no saved directory."""
    bridge = _PanelBridge.shared()
    pool = bridge.autorelease_pool()
    try:
        data = bridge.object_for_key(bridge.standard_defaults(), key)
        if data is None:
            return None
        return bridge.data_bytes(data)
    finally:
        bridge.drain(pool)


def _set_panel_bookmark(key: str, blob: bytes | None) -> None:
    """Put back a bookmark captured by ``_panel_bookmark``. None removes the key."""
    bridge = _PanelBridge.shared()
    pool = bridge.autorelease_pool()
    try:
        defaults = bridge.standard_defaults()
        if blob is None:
            bridge.remove_key(defaults, key)
        else:
            bridge.set_data(defaults, key, blob)
        bridge.synchronize(defaults)
    finally:
        bridge.drain(pool)


def _panel_directory(key: str = _PANEL_KEYS[0]) -> str | None:
    """Folder the panel would open, after resolving its bookmark."""
    bridge = _PanelBridge.shared()
    pool = bridge.autorelease_pool()
    try:
        return bridge.url_path(bridge.standard_defaults(), key)
    finally:
        bridge.drain(pool)


class _PanelBridge:
    """NSUserDefaults calls for the Finder open/save panel.

    PyObjC is not a dependency. Each ``objc_msgSend`` signature is declared
    on its own, which is what arm64 requires.
    """

    _instance: _PanelBridge | None = None

    def __init__(self) -> None:
        lib = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        lib.objc_getClass.restype = ctypes.c_void_p
        lib.objc_getClass.argtypes = [ctypes.c_char_p]
        lib.sel_registerName.restype = ctypes.c_void_p
        lib.sel_registerName.argtypes = [ctypes.c_char_p]
        self._lib = lib
        self._sels: dict[bytes, ctypes.c_void_p] = {}
        self._sends: dict[tuple, ctypes._CFuncPtr] = {}
        self._nsstring = lib.objc_getClass(b"NSString")
        self._nsurl = lib.objc_getClass(b"NSURL")
        self._nsdata = lib.objc_getClass(b"NSData")
        self._defaults_class = lib.objc_getClass(b"NSUserDefaults")
        self._pool_class = lib.objc_getClass(b"NSAutoreleasePool")

    @classmethod
    def shared(cls) -> _PanelBridge:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _sel(self, name: bytes) -> ctypes.c_void_p:
        selector = self._sels.get(name)
        if selector is None:
            selector = self._lib.sel_registerName(name)
            self._sels[name] = selector
        return selector

    def _send(self, restype, *argtypes):
        key = (restype, argtypes)
        send = self._sends.get(key)
        if send is None:
            send = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)(
                ("objc_msgSend", self._lib)
            )
            self._sends[key] = send
        return send

    def autorelease_pool(self) -> ctypes.c_void_p:
        return self._send(ctypes.c_void_p)(self._pool_class, self._sel(b"new"))

    def drain(self, pool: ctypes.c_void_p) -> None:
        self._send(None)(pool, self._sel(b"drain"))

    def standard_defaults(self) -> ctypes.c_void_p:
        return self._send(ctypes.c_void_p)(self._defaults_class, self._sel(b"standardUserDefaults"))

    def nsstring(self, text: str) -> ctypes.c_void_p:
        return self._send(ctypes.c_void_p, ctypes.c_char_p)(
            self._nsstring,
            self._sel(b"stringWithUTF8String:"),
            text.encode("utf-8"),
        )

    def directory_url(self, folder: Path) -> ctypes.c_void_p:
        return self._send(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool)(
            self._nsurl,
            self._sel(b"fileURLWithPath:isDirectory:"),
            self.nsstring(str(folder)),
            True,
        )

    def set_url(self, defaults: ctypes.c_void_p, url: ctypes.c_void_p, key: str) -> None:
        self._send(None, ctypes.c_void_p, ctypes.c_void_p)(
            defaults,
            self._sel(b"setURL:forKey:"),
            url,
            self.nsstring(key),
        )

    def object_for_key(self, defaults: ctypes.c_void_p, key: str) -> ctypes.c_void_p | None:
        found = self._send(ctypes.c_void_p, ctypes.c_void_p)(
            defaults,
            self._sel(b"objectForKey:"),
            self.nsstring(key),
        )
        return found or None

    def data_bytes(self, data: ctypes.c_void_p) -> bytes:
        size = self._send(ctypes.c_ulong)(data, self._sel(b"length"))
        raw = self._send(ctypes.c_void_p)(data, self._sel(b"bytes"))
        if not raw or not size:
            return b""
        return ctypes.string_at(raw, size)

    def set_data(self, defaults: ctypes.c_void_p, key: str, blob: bytes) -> None:
        # ``blob`` is copied into NSData before this buffer goes away.
        buffer = ctypes.create_string_buffer(blob)
        data = self._send(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)(
            self._nsdata,
            self._sel(b"dataWithBytes:length:"),
            ctypes.cast(buffer, ctypes.c_void_p),
            len(blob),
        )
        self._send(None, ctypes.c_void_p, ctypes.c_void_p)(
            defaults,
            self._sel(b"setObject:forKey:"),
            data,
            self.nsstring(key),
        )

    def remove_key(self, defaults: ctypes.c_void_p, key: str) -> None:
        self._send(None, ctypes.c_void_p)(
            defaults,
            self._sel(b"removeObjectForKey:"),
            self.nsstring(key),
        )

    def url_path(self, defaults: ctypes.c_void_p, key: str) -> str | None:
        url = self._send(ctypes.c_void_p, ctypes.c_void_p)(
            defaults,
            self._sel(b"URLForKey:"),
            self.nsstring(key),
        )
        if not url:
            return None
        path = self._send(ctypes.c_void_p)(url, self._sel(b"path"))
        if not path:
            return None
        raw = self._send(ctypes.c_char_p)(path, self._sel(b"UTF8String"))
        if not raw:
            return None
        return raw.decode("utf-8")

    def synchronize(self, defaults: ctypes.c_void_p) -> None:
        self._send(ctypes.c_bool)(defaults, self._sel(b"synchronize"))

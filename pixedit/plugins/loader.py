"""Load built-in, entry-point, and user plugins exactly once per process."""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

from pixedit.plugins.api import PluginAPI, get_api, reset_api

_loaded = False


def user_plugin_dir() -> Path:
    return Path.home() / ".pixedit" / "plugins"


def load_plugins(force: bool = False) -> PluginAPI:
    """Discover plugins.

    Order:

    1. Distribution entry points in the ``pixedit.plugins`` group.
    2. The shipped example, if that entry point was not installed (running
       from a checkout before ``pip install -e .``).
    3. ``~/.pixedit/plugins/*.py`` modules that define ``register(api)``.

    Failures are printed and skipped so one bad plugin cannot block the app.
    """
    global _loaded
    if _loaded and not force:
        return get_api()
    if force:
        reset_api()
    api = get_api()
    _load_entry_points(api)
    if not api.has("example.vignette"):
        _safe_call("builtin example", _load_builtin_example, api)
    _load_user_plugins(api)
    _loaded = True
    return api


def reset_plugins() -> PluginAPI:
    global _loaded
    _loaded = False
    return reset_api()


def _safe_call(name: str, func, *args) -> None:
    try:
        func(*args)
    except Exception:
        print(f"PixEdit plugin {name} failed:\n{traceback.format_exc()}", file=sys.stderr)


def _load_entry_points(api: PluginAPI) -> None:
    try:
        from importlib.metadata import entry_points
    except ImportError:
        return
    try:
        group = entry_points(group="pixedit.plugins")
    except TypeError:
        group = entry_points().get("pixedit.plugins", [])
    for entry in group:
        def _run(ep=entry) -> None:
            register = ep.load()
            register(api)

        _safe_call(entry.name, _run)


def _load_builtin_example(api: PluginAPI) -> None:
    from pixedit.plugins.examples.basic import register

    register(api)


def _load_user_plugins(api: PluginAPI) -> None:
    folder = user_plugin_dir()
    if not folder.is_dir():
        return
    for path in sorted(folder.glob("*.py")):
        def _run(plugin_path: Path = path) -> None:
            module_name = f"pixedit_user_{plugin_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load {plugin_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            register = getattr(module, "register", None)
            if register is None:
                raise AttributeError(f"{plugin_path.name} has no register(api) function")
            register(api)

        _safe_call(path.name, _run)

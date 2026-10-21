"""The small API plugins use to add commands and canvas tools.

A plugin module exposes ``register(api)``. The editor calls it at startup for:

* the ``pixedit.plugins`` entry-point group
* every ``*.py`` file in ``~/.pixedit/plugins``

``callback(session)`` receives the editing session (document, active layer,
``commit``, foreground color, options). Interactive tools subclass
``pixedit.tools.base.Tool``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pixedit.tools.base import Tool


@dataclass
class MenuCommand:
    command_id: str
    label: str
    menu: str
    callback: Callable[[Any], None]
    shortcut: str | None = None


class PluginAPI:
    def __init__(self) -> None:
        self.commands: list[MenuCommand] = []
        self.tools: list[Tool] = []
        self._ids: set[str] = set()

    def add_command(
        self,
        *,
        command_id: str,
        label: str,
        callback: Callable[[Any], None],
        menu: str = "Filter",
        shortcut: str | None = None,
    ) -> None:
        """Add a menu action. ``callback(session)`` should edit via ``session.commit``."""
        if command_id in self._ids:
            return
        self._ids.add(command_id)
        self.commands.append(
            MenuCommand(
                command_id=command_id,
                label=label,
                menu=menu,
                callback=callback,
                shortcut=shortcut,
            )
        )

    def add_tool(self, tool: Tool) -> None:
        """Add a canvas tool. It shows up in the toolbar after the built-in tools."""
        if tool.id in self._ids:
            return
        self._ids.add(tool.id)
        self.tools.append(tool)

    def has(self, item_id: str) -> bool:
        return item_id in self._ids


_api = PluginAPI()


def get_api() -> PluginAPI:
    return _api


def reset_api() -> PluginAPI:
    global _api
    _api = PluginAPI()
    return _api

"""Plugin registry. See ``pixedit.plugins.api`` and the example in ``examples``."""

from pixedit.plugins.api import PluginAPI, get_api
from pixedit.plugins.loader import load_plugins, reset_plugins

__all__ = ["PluginAPI", "get_api", "load_plugins", "reset_plugins"]

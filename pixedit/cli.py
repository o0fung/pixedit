"""Console entry point.

``pix`` launches the GUI, the same shape as ``rr`` launching ``rr_app``:
a short ``[project.scripts]`` command whose ``main`` imports the Qt app only
after argument parsing, and reports startup failures instead of dying silently.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from pixedit import __version__


def _runtime_log_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "pixedit"
    return Path.home() / ".pixedit" / "logs"


def _write_startup_log(title: str, message: str, exc: BaseException | None = None) -> Path | None:
    try:
        log_dir = _runtime_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = log_dir / f"startup-{stamp}.log"
        lines = [f"{title}\n", f"{message}\n"]
        if exc is not None:
            lines.append("\nTraceback:\n")
            lines.append("".join(traceback.format_exception(exc)))
        path.write_text("".join(lines), encoding="utf-8")
        return path
    except Exception:
        return None


def _missing_dep_hint(mod: str) -> str:
    return (
        f"Missing dependency: {mod}\n\n"
        "From a local checkout:\n"
        "  python -m venv .venv\n"
        "  source .venv/bin/activate\n"
        "  pip install -e .\n"
    )


def _display_available() -> bool:
    if os.environ.get("QT_QPA_PLATFORM"):
        return True
    if sys.platform in {"win32", "darwin"}:
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pix",
        description="Launch the PixEdit layered image editor.",
    )
    parser.add_argument("path", nargs="?", help="Image, PDF, or .pix project to open")
    parser.add_argument("--version", action="version", version=f"pixedit {__version__}")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Construct the window and exit. Uses the offscreen Qt platform when needed.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Do not write bytecode into the checkout when the console script starts.
    sys.dont_write_bytecode = True
    args = _parse_args(argv)

    if args.check:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from pixedit.app import check
        except ModuleNotFoundError as exc:
            print(_missing_dep_hint(exc.name or "unknown"), file=sys.stderr)
            return 1
        try:
            return int(check())
        except Exception as exc:
            log_path = _write_startup_log("PixEdit check failed", "GUI failed to construct.", exc)
            print(f"PixEdit check failed: {exc}", file=sys.stderr)
            if log_path:
                print(f"Details: {log_path}", file=sys.stderr)
            return 1

    if not _display_available():
        print(
            "PixEdit needs a display to open the editor.\n"
            "On a headless machine, run `pix --check` to construct the UI offscreen,\n"
            "or `pytest` to exercise the document, layer, and redaction code.",
            file=sys.stderr,
        )
        return 1

    try:
        from pixedit.app import run
    except ModuleNotFoundError as exc:
        message = _missing_dep_hint(exc.name or "unknown")
        log_path = _write_startup_log("PixEdit failed to start", message, exc)
        print(message, file=sys.stderr)
        if log_path:
            print(f"Details: {log_path}", file=sys.stderr)
        return 1
    except Exception as exc:
        log_path = _write_startup_log("PixEdit failed to initialize", "Unexpected startup error.", exc)
        print(f"PixEdit failed to initialize: {exc}", file=sys.stderr)
        if log_path:
            print(f"Details: {log_path}", file=sys.stderr)
        return 1

    try:
        return int(run(args.path) or 0)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        log_path = _write_startup_log("PixEdit crashed on launch", "Unexpected error while starting the GUI.", exc)
        print(f"PixEdit crashed on launch: {exc}", file=sys.stderr)
        if log_path:
            print(f"Details: {log_path}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

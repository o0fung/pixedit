"""Allow ``python -m pixedit`` in addition to the ``pix`` console script."""

from __future__ import annotations

import sys

from pixedit.cli import main

if __name__ == "__main__":
    sys.exit(main())

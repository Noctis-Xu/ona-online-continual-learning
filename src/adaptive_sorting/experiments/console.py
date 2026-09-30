"""Small terminal helpers shared by experiment entry points."""

from __future__ import annotations

import sys
from typing import Callable


RESET = "\033[0m"
BOLD_RED = "\033[1;31m"


def run_cli(main: Callable[[], None]) -> None:
    try:
        main()
    except Exception as exc:
        print(f"{BOLD_RED}ERROR{RESET}: {exc}", file=sys.stderr)
        raise

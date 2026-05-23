"""
Compatibility launcher.

Keeps `python main_loop_closure.py` working while the implementation lives in `src/main_loop_closure.py`.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    runpy.run_path(str(src / "main_loop_closure.py"), run_name="__main__")


if __name__ == "__main__":
    main()


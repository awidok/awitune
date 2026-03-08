import sys
from pathlib import Path

# Allow running as `python __main__.py` from inside awitune/
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "awitune"

from .lib.cli import main

main()

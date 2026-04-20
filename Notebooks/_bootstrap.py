"""Bootstrap for notebooks in this folder.

Usage from any notebook in Notebooks/:

    from _bootstrap import sqx_tool
    sqx_tool.newproject(sqx_tool.argparse.Namespace(
        symbol_dukascopy="XAUUSD_darwinex",
        symbol_darwinex="XAUUSD_dx_darwinex",
        timeframe="H4",
        direction="Long",
    ))
"""

import sys
from pathlib import Path

_SRC = (Path(__file__).resolve().parent.parent / "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import sqx_tool  # noqa: E402,F401

sqx_tool.configure_logging()

"""Bootstrap for notebooks.

Lives in src/ and is importable because start_notebook.bat puts src/ on
PYTHONPATH. Usage from any notebook:

    from _bootstrap import sqx_tool
    sqx_tool.newproject(sqx_tool.argparse.Namespace(
        symbol_dukascopy="XAUUSD_darwinex",
        symbol_darwinex="XAUUSD_dx_darwinex",
        timeframe="H4",
        direction="Long",
    ))
"""

from pathlib import Path
from typing import Any

import sqx_tool
import update_instruments_db
import calculate_spreads
from calculate_spreads import (
    csv_to_daily_spread_cache,
    load_daily_spread_cache,
    compute_spread_px_py_from_cache,
)

_SRC = Path(__file__).resolve().parent

__all__ = [
    "sqx_tool",
    "update_instruments_db",
    "calculate_spreads",
    "csv_to_daily_spread_cache",
    "load_daily_spread_cache",
    "compute_spread_px_py_from_cache",
    "mq5_instruments_script",
    "update_instruments",
]

sqx_tool.configure_logging()

mq5_instruments_script: dict[str, Path] = {
    'ex5': _SRC / 'mq5' / 'DX_Update_SQX_Instruments_information.ex5',
    'mq5': _SRC / 'mq5' / 'DX_Update_SQX_Instruments_information.mq5'
}


def update_instruments(
    xml_path: str | Path,
    db_path: str | Path | None = None,
    broker_name: str = "darwinex",
    broker_id: int = 4,
) -> list[dict[str, Any]]:
    """Upsert INSTRUMENTS rows from a broker XML into the SQX SQLite DB.

    ``db_path`` defaults to the symbols DB resolved from config.ini
    (``sqx_tool.SETTINGS.symbols_db``).

    Usage from a notebook:

        from _bootstrap import update_instruments
        update_instruments("Updated Instrument information.xml")
    """
    xml_path_p = Path(xml_path)
    db_path_p = Path(db_path) if db_path is not None else sqx_tool.SETTINGS.symbols_db
    rows = list(update_instruments_db.parse_xml(
        xml_path_p, broker_suffix=broker_name, broker_id=broker_id,
    ))
    update_instruments_db.upsert_rows(db_path_p, rows)
    print(f"{len(rows)} symbols processed.")
    return rows
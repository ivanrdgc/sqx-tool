#!/usr/bin/env python3
"""
update_instruments_db.py

Bring the StrategyQuant X INSTRUMENTS table up to date from the instrument
information exported by the DX_Update_SQX_Instruments_information MT5 script.

Run it with no arguments (see update_instruments_db.bat / .sh):

  * If MetaTrader has already written an "Updated SQX Instruments Information
    (…)" folder under MQL5/Files, every XML found there is imported into the
    SQX symbols DB.
  * Otherwise the MT5 script is copied into MQL5/Scripts so it can be run from
    MetaTrader, and nothing is imported.

Both locations come from config.ini at the repo root (SQX_PATH / MT5_PATH).
"""
import argparse
import html
import shutil
import sqlite3
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Iterator

# The embedded Python ships a ._pth file, which runs the interpreter in
# isolated mode and keeps the script's own directory off sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sqx_tool

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

_SRC = Path(__file__).resolve().parent

#: Copied into MQL5/Scripts when MT5 has not exported anything yet.
MQ5_SCRIPT_FILES = (
    _SRC / "mq5" / "DX_Update_SQX_Instruments_information.ex5",
    _SRC / "mq5" / "DX_Update_SQX_Instruments_information.mq5",
)

#: MT5 writes one folder per broker profile under MQL5/Files, e.g.
#: "Updated SQX Instruments Information (Imported from (Darwinex-Live) (TimeOffset=0) (USD Account))"
EXPORT_DIR_GLOB = "Updated SQX Instruments Information*"
EXPORT_XML_REL = Path("StrategyQuantX") / "Updated Instrument information.xml"

#: Naming/id convention applied to every imported symbol.
DEFAULT_BROKER_NAME = "darwinex"
DEFAULT_BROKER_ID = 4

#: XML → DB column mapping (xml_attribute -> db_column, type)
FIELD_MAP: dict[str, tuple[str, Callable[[str], Any]]] = {
    "instrument":            ("INSTRUMENT",          str),
    "description":           ("DESCRIPTION",         str),
    "pointValue":            ("POINTVALUE",          float),
    "tickSize":              ("TICKSIZE",            float),
    "tickStep":              ("TICKSTEP",            float),
    "defaultSpread":         ("DEFAULTSPREAD",       float),
    "commissions":           ("COMMISSIONS",         str),     # will unescape later
    "dataType":              ("DATATYPE",            int),
    "exchange":              ("EXCHANGE",            str),
    "country":               ("COUNTRY",             str),
    "sector":                ("SECTOR",              str),
    "defaultSlippage":       ("DEFAULTSLIPPAGE",     float),
    "swap":                  ("SWAP",                str),     # will unescape later
    "orderSizeMultiplier":   ("ORDERSIZEMULTIPLIER", float),
    "orderSizeStep":         ("ORDERSIZESTEP",       float),
    "broker":                ("BROKER_ID",           int),
    "minDistance":           ("MIN_DISTANCE",        float),
}

# Columns that have sensible defaults even when the attribute is missing
DEFAULTS: dict[str, Any] = {
    "DEFAULTSPREAD":       0.0,
    "DEFAULTSLIPPAGE":     0.0,
    "SWAP":                None,
    "ORDERSIZEMULTIPLIER": 1.0,
    "ORDERSIZESTEP":       0.0,
    "BROKER_ID":           -1,
    "MIN_DISTANCE":        0.0,
    "EXCHANGE":            None,
    "COUNTRY":             None,
    "SECTOR":              None,
}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def parse_xml(
    path: Path,
    broker_suffix: str = "",
    broker_id: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield dicts ready for insertion into the DB."""
    tree = ET.parse(path)
    for elem in tree.findall(".//InstrumentInfo"):
        row: dict[str, Any] = {}

        for xml_attr, (db_col, coercer) in FIELD_MAP.items():
            raw_val = elem.attrib.get(xml_attr)

            # If value is missing use DEFAULTS or leave None
            if raw_val is None:
                row[db_col] = DEFAULTS.get(db_col)
                continue

            # html-unescape long XML strings so they are stored exactly as XML
            if db_col in {"COMMISSIONS", "SWAP"}:
                raw_val = html.unescape(raw_val)

            try:
                row[db_col] = coercer(raw_val)
            except ValueError as exc:
                raise ValueError(
                    f"Unable to convert '{raw_val}' for column {db_col!r}"
                ) from exc

        # Apply broker-specific overrides / naming rules
        if broker_suffix:
            row["INSTRUMENT"] = f"{row['INSTRUMENT']}_{broker_suffix}"
        if broker_id is not None:
            row["BROKER_ID"] = broker_id

        yield row


def upsert_rows(db_path: Path, rows: list[dict[str, Any]]) -> None:
    """Insert or update rows using SQLite’s ON CONFLICT syntax."""
    if not rows:
        return

    # All rows share the same set of columns
    columns: list[str] = list(rows[0].keys())
    placeholders = ", ".join(["?"] * len(columns))
    cols_joined = ", ".join(columns)
    update_set = ", ".join(f"{col}=excluded.{col}" for col in columns if col != "INSTRUMENT")

    sql = (
        f"INSERT INTO INSTRUMENTS ({cols_joined}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT(INSTRUMENT) DO UPDATE SET {update_set};"
    )

    with sqlite3.connect(db_path) as con:
        con.executemany(sql, ([row[col] for col in columns] for row in rows))
        con.commit()


# ---------------------------------------------------------------------------
# MetaTrader 5 integration
# ---------------------------------------------------------------------------

def find_exports(mt5_path: Path) -> list[Path]:
    """Return every instrument XML the MT5 script has exported, sorted by path."""
    files_dir = mt5_path / "MQL5" / "Files"
    if not files_dir.is_dir():
        return []
    return sorted(
        xml
        for export_dir in files_dir.glob(EXPORT_DIR_GLOB)
        if (xml := export_dir / EXPORT_XML_REL).is_file()
    )


def deploy_mq5_script(mt5_path: Path) -> Path:
    """Copy the instruments-export script into MT5's Scripts folder."""
    scripts_dir = mt5_path / "MQL5" / "Scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    for src in MQ5_SCRIPT_FILES:
        if not src.is_file():
            raise FileNotFoundError(f"MT5 script missing from the distribution: {src}")
        shutil.copy2(src, scripts_dir)
    return scripts_dir


def import_export(
    xml_path: Path,
    db_path: Path,
    broker_name: str = DEFAULT_BROKER_NAME,
    broker_id: int = DEFAULT_BROKER_ID,
) -> int:
    """Upsert one exported XML into the symbols DB, returning the row count."""
    rows = list(parse_xml(xml_path, broker_suffix=broker_name, broker_id=broker_id))
    upsert_rows(db_path, rows)
    return len(rows)


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------

def _pause() -> None:
    """Keep the console open when launched by double-clicking the .bat."""
    try:
        input("\nYou can close this window…")
    except EOFError:
        pass  # not interactive (piped or redirected) – nothing to wait for


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import MT5-exported instrument information into the SQX symbols DB."
    )
    parser.add_argument("--xml", type=Path, help="Import this XML instead of searching MT5")
    parser.add_argument("--db", type=Path, help="SQLite database (default: from config.ini SQX_PATH)")
    parser.add_argument("--broker_name", default=DEFAULT_BROKER_NAME,
                        help="Suffix appended to every symbol (default: %(default)s)")
    parser.add_argument("--broker_id", type=int, default=DEFAULT_BROKER_ID,
                        help="Value for the BROKER_ID column (default: %(default)s)")
    args = parser.parse_args(argv)

    mt5_path = sqx_tool.SETTINGS.mt5_path
    db_path = args.db or sqx_tool.SETTINGS.symbols_db

    print("=== Update SQX Instruments ===\n")
    print(f"MT5 : {mt5_path}")
    print(f"DB  : {db_path}\n")

    # ---- 1. No export yet → deploy the MT5 script and stop -----------------
    exports = [args.xml] if args.xml else find_exports(mt5_path)
    if not exports:
        if not mt5_path.is_dir():
            print(f"MetaTrader folder not found at {mt5_path}.")
            print("Check MT5_PATH in config.ini.")
            return 1

        scripts_dir = deploy_mq5_script(mt5_path)
        print("No exported instrument information found under MQL5/Files.")
        print(f"Copied DX_Update_SQX_Instruments_information to:\n  {scripts_dir}\n")
        print("Next steps:")
        print("  1. Open MetaTrader 5 and refresh the Navigator (right-click, Refresh)")
        print("  2. Run the DX_Update_SQX_Instruments_information script on any chart")
        print("  3. Close StrategyQuant X, then run this tool again")
        return 0

    # ---- 2. Export present → import it into the symbols DB -----------------
    if not db_path.is_file():
        print(f"Can't access the symbols DB at {db_path}.")
        print("Check SQX_PATH in config.ini.")
        return 1

    print("StrategyQuant X must be closed while the database is updated.\n")
    print(f"Found {len(exports)} export(s):\n")

    total = 0
    for xml_path in exports:
        count = import_export(xml_path, db_path, args.broker_name, args.broker_id)
        total += count
        print(f"  {count:5d} symbols  {xml_path.parent.parent.name}")

    print(f"\n{total} symbols processed into {db_path.name}.")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as e:
        print(f"Error: {e}")
        _pause()
        sys.exit(1)
    _pause()
    sys.exit(code)

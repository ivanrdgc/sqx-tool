#!/usr/bin/env python3
"""
update_instruments.py

Read a Darwinex-style XML file and bring an SQLite table called INSTRUMENTS
up-to-date.  Run:

    python update_instruments.py instruments.xml data.db --broker_name darwinex --broker_id 4
"""
import argparse
import html
import sqlite3
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

#: XML → DB column mapping (xml_attribute -> db_column, type)
FIELD_MAP = {
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
DEFAULTS = {
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

def parse_xml(path: Path, broker_suffix: str = "", broker_id: int | None = None):
    """Yield dicts ready for insertion into the DB."""
    tree = ET.parse(path)
    for elem in tree.findall(".//InstrumentInfo"):
        row: dict[str, object] = {}

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


def upsert_rows(db_path: Path, rows: list[dict]):
    """Insert or update rows using SQLite’s ON CONFLICT syntax."""
    if not rows:
        return

    # All rows share the same set of columns
    columns = list(rows[0].keys())
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
# CLI wrapper
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Update INSTRUMENTS table from XML.")
    parser.add_argument("xml_path", type=Path, help="Path to XML file exported by the broker")
    parser.add_argument("db_path",  type=Path, help="SQLite database (*.db)")
    parser.add_argument("--broker_name", default="", help="Suffix to append to the symbol (e.g. 'darwinex')")
    parser.add_argument("--broker_id",   type=int, help="Value for BROKER_ID column")
    args = parser.parse_args(argv)

    # 1) parse the XML
    rows = list(parse_xml(args.xml_path, broker_suffix=args.broker_name, broker_id=args.broker_id))

    # 2) upsert into SQLite
    upsert_rows(args.db_path, rows)

    print(f"{len(rows)} symbols processed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"Error: {e}")

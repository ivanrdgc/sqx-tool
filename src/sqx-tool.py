#!/usr/bin/env python3

"""
sqx-tool.py: StrategyQuant X project helper tool

- Scaffolds new SQX projects from a template
- Removes ExitAfterBars blocks from .sqx files
- Provides an interactive CLI for non-technical users
- Handles all file and directory management, XML patching, and logging
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
#  Standard library imports
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import base64
import concurrent.futures as cf
import io
import logging
import os
import platform
import re
import shutil
import sqlite3
import sys
from datetime import datetime, date, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import cast, Callable, Optional, Tuple, Any, List, Union
import zipfile
import multiprocessing as mp
import xml.etree.ElementTree as ET

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Settings:
    """All file‑system locations & naming conventions live here."""
    
    script_dir: Path = Path(__file__).resolve().parent
    template_path_ramon: Path = (script_dir / "Templates" / "Ramon_Mensual.cfx").resolve()
    template_path_hc: Path = (script_dir / "Templates" / "Hobbiecode.cfx").resolve()
    projects_base: Path = (script_dir / "../Projects").resolve()
    log_file: Path = (script_dir / "sqx-tool.log").resolve()
    unix_sh: Path = (script_dir / "../run.sh").resolve()
    # Default log level when no -v/-q flags are provided.
    # One of: "trace", "debug", "info", "warning", "error", "critical"
    default_log_level: str = "info"
    
    project_dir_tpl: str = "{symbol}/{timestamp}_{symbol}_{timeframe}_{direction}"
    file_prefix_tpl: str = "{symbol} {timeframe} {direction}"
    
    # symbols_db: Path = Path("D:/SQX/user/data/data.db").resolve()
    # symbols_db: Path = Path("/home/user/SQX/user/data/data.db").resolve()
    symbols_db: Path = (script_dir / "../../user/data/data.db").resolve()

SETTINGS = Settings()

# Helper to check for symbols DB existence

def require_symbols_db() -> None:
    """Exit with a message if the symbols DB is not accessible."""
    if not SETTINGS.symbols_db.is_file():
        print(f"Can't access symbols_db at {SETTINGS.symbols_db}.")
        input("\nYou can close this window…")
        sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
#  Logging helpers
# ─────────────────────────────────────────────────────────────────────────────

# Add a TRACE level below DEBUG for ultra-verbose logging
TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")

def _trace(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, msg, args, **kwargs)

logging.Logger.trace = _trace  # type: ignore[attr-defined]


def configure_logging(verbosity: int = 0, quiet: int = 0) -> None:
    """Configure logging to file with multiple verbosity levels.

    - quiet: 0=default, 1=WARNING, 2=ERROR, 3=CRITICAL
    - verbosity: 0=INFO, 1=DEBUG, >=2=TRACE
    """
    # Quiet overrides verbosity; otherwise derive from flags or Settings.default_log_level
    if quiet > 0:
        quiet = min(quiet, 3)
        level = [logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL][quiet]
    elif verbosity > 0:
        level = logging.DEBUG if verbosity == 1 else TRACE_LEVEL
    else:
        level_map = {
            "trace": TRACE_LEVEL,
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }
        level = level_map.get(SETTINGS.default_log_level.lower(), logging.INFO)

    fh = logging.FileHandler(SETTINGS.log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(process)d  %(levelname)8s  %(name)s – %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    # Only file handler; never log to stdout
    logging.basicConfig(level=level, handlers=[fh], force=True)

# Module logger shortcuts
logger = logging.getLogger(__name__)
log = logger.info

# Replaced below with module-level logger after configure_logging

# ─────────────────────────────────────────────────────────────────────────────
#  XMLMutator type for patching XML
# ─────────────────────────────────────────────────────────────────────────────

# Accepts any arguments, matching patch() usage and all mutator signatures
XMLMutator = Callable[..., None]

# ─────────────────────────────────────────────────────────────────────────────
#  ZipEditor
# ─────────────────────────────────────────────────────────────────────────────

class ZipEditor:
    """Utility for reading, patching, and writing zip archives containing XML files."""
    def __init__(self, zip_path: Path) -> None:
        self.zip_path = zip_path
        with zipfile.ZipFile(zip_path, "r") as zin:
            self._files = {i.filename: zin.read(i.filename) for i in zin.infolist()}

    def patch(self, entry_name: str, mutator: XMLMutator, *args, **kwargs) -> None:
        if entry_name not in self._files:
            logger.debug("Entry %s not found in zip archive.", entry_name)
            return
        root = ET.fromstring(self._files[entry_name])
        mutator(root, *args, **kwargs)
        buf = io.BytesIO()
        ET.ElementTree(root).write(buf, encoding="utf-8", xml_declaration=True)
        self._files[entry_name] = buf.getvalue()

    def write(self, dest: Optional[Path] = None) -> Path:
        target = dest or self.zip_path
        tmp = target.with_suffix(".tmp")
        with zipfile.ZipFile(tmp, "w") as zout:
            for fn, data in self._files.items():
                zout.writestr(fn, data)
        os.replace(tmp, target)
        return target

# ─────────────────────────────────────────────────────────────────────────────
#  Symbol helpers (TZ‑aware)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SymbolInfo:
    first_date: Optional[datetime]
    last_date:  Optional[datetime]
    spread:     Optional[float]
    commission: Optional[str]
    swap:       Optional[str]


def symbol_exists(key: str) -> bool:
    db = SETTINGS.symbols_db
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        return con.execute("SELECT 1 FROM DATA WHERE SYMBOL = ?", (key,)).fetchone() is not None


def get_symbol_info(key: str) -> SymbolInfo:
    db = SETTINGS.symbols_db
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT DATEFROM, DATETO, i.DEFAULTSPREAD, i.COMMISSIONS, i.SWAP "
        "FROM DATA d JOIN INSTRUMENTS i ON i.INSTRUMENT = d.INSTRUMENT WHERE d.SYMBOL = ?",
        (key,),
    ).fetchone()
    if row is None:
        logging.warning(f"No symbol info found for key: {key}")
        return SymbolInfo(None, None, None, None, None)

    first_dt = datetime.fromtimestamp(row["DATEFROM"] / 1000, timezone.utc)
    last_dt  = datetime.fromtimestamp(row["DATETO"]  / 1000, timezone.utc)

    return SymbolInfo(first_dt, last_dt, row["DEFAULTSPREAD"], row["COMMISSIONS"], row["SWAP"])

# ─────────────────────────────────────────────────────────────────────────────
#  remove_eab implementation
# ─────────────────────────────────────────────────────────────────────────────

def _strip_eab_single(args: Tuple[Path, Path]) -> Tuple[str, str]:
    """Worker function run in a *separate* process.

    Removes the *ExitAfterBars* building block from an ``.sqx`` file and writes
    the result to *dst_dir*.  Returns a tuple ``(basename, status)`` for
    progress reporting.
    """
    src, dst_dir = args
    dst = dst_dir / src.name
    xml_name = "strategy_Portfolio.xml"

    logging.debug("_strip_eab_single(src=%s, dst=%s)", src, dst)

    # ---- 1. read the inner XML -------------------------------------------
    try:
        with zipfile.ZipFile(src, "r") as z:
            xml_data = z.read(xml_name)
    except (KeyError, zipfile.BadZipFile):
        return src.name, "no strategy_Portfolio.xml – skipped"

    # ---- 2. parse + mutate -----------------------------------------------
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError:
        return src.name, "XML parse error – skipped"

    # 2a. collect ids to remove
    ids_to_remove: list[str] = []
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "Param" and child.get("key") == "#ExitAfterBars.ExitAfterBars#":
                if child.text:
                    ids_to_remove.append(child.text)
                parent.remove(child)

    # 2b. purge matching <variable> blocks
    vars_parent = root.find(".//Variables")
    if ids_to_remove and vars_parent is not None:
        for var in list(vars_parent):
            id_elem = var.find("id")
            if id_elem is not None and id_elem.text in ids_to_remove:
                vars_parent.remove(var)

    # 2c. prune lone Boolean <Item>
    for signal in root.findall(".//signals/signal"):
        items = [e for e in signal if e.tag == "Item"]
        if len(items) == 1 and items[0].get("key") == "Boolean":
            signal.remove(items[0])

    # 2d. write back to new archive
    buf = io.BytesIO()
    ET.ElementTree(root).write(buf, encoding="utf-8", xml_declaration=True)

    try:
        with zipfile.ZipFile(dst, "w") as zout, zipfile.ZipFile(src, "r") as zin:
            for item in zin.infolist():
                data = buf.getvalue() if item.filename == xml_name else zin.read(item.filename)
                zout.writestr(item, data)
    except Exception as exc:
        logging.error(f"Error writing zip file {dst}: {exc}")
        return src.name, f"ERROR: {exc}"

    return src.name, "ok"


def remove_eab(args: argparse.Namespace) -> None:
    """CLI entry point for **remove_eab** sub-command."""
    logging.debug("remove_eab(args=%s)", args)

    src_dir = Path(args.path_from).expanduser().resolve()
    dst_dir = Path(args.path_to).expanduser().resolve()

    dst_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in src_dir.iterdir() if p.suffix.lower() == ".sqx"]
    log("found %d .sqx files in %s", len(files), src_dir)
    logger.trace("Files to process: %s", [str(f) for f in files])

    tasks: List[Tuple[Path, Path]] = [(p, dst_dir) for p in files]

    with cf.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for basename, status in pool.map(_strip_eab_single, tasks):
            log("  %-30s %s", basename, status)
            logger.trace("Processed %s: %s", basename, status)


def remove_eab_b64(args: argparse.Namespace) -> None:
    """CLI entry point for **remove_eab_b64** sub-command.
    
    Works exactly like remove_eab but accepts base64-encoded paths.
    Decodes the paths and delegates to the core remove_eab functionality.
    """
    logger.debug("remove_eab_b64(args=%s)", args)
    
    try:
        # Decode base64 strings to get the actual paths
        path_from_decoded = base64.b64decode(args.path_from_b64).decode('utf-8')
        path_to_decoded = base64.b64decode(args.path_to_b64).decode('utf-8')
        
        logger.debug("Decoded paths - from: %s, to: %s", path_from_decoded, path_to_decoded)
        
        # Create a new args object with decoded paths
        decoded_args = argparse.Namespace(
            path_from=path_from_decoded,
            path_to=path_to_decoded,
            jobs=args.jobs
        )
        
        # Delegate to the existing remove_eab function
        remove_eab(decoded_args)
        
    except Exception as e:
        logging.error("Error in remove_eab_b64: %s", str(e))
        print(f"Error decoding base64 paths: {e}")
        raise

# ─────────────────────────────────────────────────────────────────────────────
#  newproject implementation
# ─────────────────────────────────────────────────────────────────────────────

def newproject(args: argparse.Namespace) -> None:
    """Scaffold a brand-new *StrategyQuant X* project from the template."""
    require_symbols_db()
    logger.debug("newproject(args=%s)", args)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    template: Path = Settings.template_path_ramon if args.template == "Ramon" else Settings.template_path_hc
    symbol_dukascopy: str = args.symbol_dukascopy
    symbol_darwinex: Optional[str] = args.symbol_darwinex or None
    timeframe: str = args.timeframe.upper()
    direction: str = args.direction.capitalize()

    logging.info("Creating new project for %s (%s, %s) – direction=%s",
                 symbol_dukascopy, symbol_darwinex or "no DWX", timeframe, direction)

    # ---- 1. Preconditions --------------------------------------------------
    for sym in filter(None, (symbol_dukascopy, symbol_darwinex)):
        if not symbol_exists(sym):
            logging.error("symbol '%s' not found in symbols DB – aborting", sym)
            print(f"symbol '{sym}' not found in symbols DB – aborting")
            return

    # ---- 2. Fetch symbol metadata -----------------------------------------
    sym_info = get_symbol_info(symbol_dukascopy)
    sym_2_info = get_symbol_info(symbol_darwinex) if symbol_darwinex else sym_info

    if not template.is_file():
        logging.error("template not found in %s", template)
        print(f"template not found in {template}")
        return

    # ---- 3. Create project directory tree ---------------------------------
    project_rel = SETTINGS.project_dir_tpl.format(
        symbol=symbol_dukascopy,
        timestamp=ts,
        timeframe=timeframe,
        direction=direction,
    )
    project_dir = SETTINGS.projects_base / project_rel
    project_dir.mkdir(parents=True, exist_ok=True)
    log("project dir: %s", project_dir)
    logger.debug("Created project directory: %s", project_dir)

    for sub in [
        "01 - Edge",
        "02 - To Strategy",
        "03 - Strategy",
        "04 - MQL/MQL5",
        "04 - MQL/MQL4",
    ]:
        (project_dir / sub).mkdir(parents=True, exist_ok=True)
        logger.trace("Created subdir %s", sub)

    # ---- 4. Copy template --------------------------------------------------
    dest_cfx = project_dir / f"{project_dir.name}.cfx"
    shutil.copyfile(template, dest_cfx)
    log("copied template to %s", dest_cfx)
    logger.debug("Copied template from %s to %s", template, dest_cfx)

    # ----------------------------------------------------------------------
    #  Helper mutators – declared *inside* newproject so they can capture
    #  surrounding variables without global state.
    # ----------------------------------------------------------------------

    def patch_config(root: ET.Element) -> None:
        logger.debug("patch_config() – setting project name → %s", project_dir.name)
        root.set("name", project_dir.name)

    def patch_load_from_files(root: ET.Element, subdir: Path) -> None:
        node = root.find(".//LoadFromFiles/SourceDirectory")
        if node is not None:
            logger.trace("patch_load_from_files() – %s", subdir)
            node.text = str(subdir.resolve())

    def patch_save_to_files(root: ET.Element,
                            sqx_dir: Optional[Path] = None,
                            sc_dir:  Optional[Path] = None) -> None:
        if sqx_dir is not None:
            node = root.find(".//SaveToFiles/DestDirectorySqx")
            if node is not None:
                logger.trace("patch_save_to_files() – sqx_dir=%s", sqx_dir)
                node.text = str(sqx_dir.resolve())
        if sc_dir is not None:
            node_sc = root.find(".//SaveToFiles/DestDirectorySC")
            if node_sc is not None:
                logger.trace("patch_save_to_files() – sc_dir=%s", sc_dir)
                node_sc.text = str(sc_dir.resolve())

    def patch_call_external(root: ET.Element, command: str) -> None:
        call = root.find(".//CallExternalScript/File")
        if call is not None:
            logger.debug("patch_call_external() – cmd=%s", command)
            if platform.system() == "Windows":
                # Windows: use sys.executable directly
                call.text = sys.executable
                call.set("params", f'"{Path(__file__).resolve()}" {command}')
            else:
                # Linux / Mac
                call.text = str(Settings.unix_sh)
                call.set("params",  command)

    def patch_market_side(root: ET.Element) -> None:
        ms = root.find(".//MarketSides")
        if ms is not None:
            logger.debug("patch_market_side() → %s", direction.lower())
            ms.set("type", direction.lower())

    def patch_setup(root: ET.Element) -> None:
        setup = root.find(".//Setups/Setup")
        if setup is None:
            return
        # Chart -------------------------------------------------------------
        chart = setup.find("Chart")
        if chart is not None:
            logger.debug("patch_setup.chart() – %s @ %s", symbol_dukascopy, timeframe)
            chart.set("symbol", symbol_dukascopy)
            chart.set("timeframe", timeframe)
            if sym_info.spread is not None:
                chart.set("spread", str(sym_info.spread))

        # Commissions -------------------------------------------------------
        if sym_info.commission:
            comm_parent = setup.find("Commissions")
            if comm_parent is None:
                comm_parent = ET.SubElement(setup, "Commissions")
            comm_parent.clear()
            try:
                comm_parent.append(ET.fromstring(sym_info.commission))
            except ET.ParseError as exc:
                logging.warning("bad commission XML for %s: %s", symbol_dukascopy, exc)

        # Swap --------------------------------------------------------------
        if sym_info.swap:
            old = setup.find("Swap")
            if old is not None:
                setup.remove(old)
            try:
                setup.append(ET.fromstring(sym_info.swap))
            except ET.ParseError as exc:
                logging.warning("bad swap XML for %s: %s", symbol_dukascopy, exc)

    def patch_dates(
        root: ET.Element,
        in_from: Union[datetime, str],
        in_to: Union[datetime, str],
        oos_spans: Optional[List[Tuple[Union[datetime, str], Union[datetime, str], Optional[str]]]] = None,
    ) -> None:
        """Rewrite <Setup> date range and <OutOfSample> block in *root*.

        Notes
        -----
        * *oos_spans* accepts tuples of length 2 → basic OOS    ``(from, to)``
          or length 3 → annotated            ``(from, to, type)`` where *type*
          is usually ``'oos'`` or ``'isv'``.  Unknown/None → treated as ``oos``.
        """
        logger.debug("patch_dates(in_from=%s, in_to=%s, spans=%s)", in_from, in_to, oos_spans)

        def _fmt(d: Union[datetime, date, str]) -> str:
            return d.strftime("%Y.%m.%d") if isinstance(d, (datetime, date)) else str(d)

        def _to_dt(d: Union[datetime, date, str]) -> datetime:
            if isinstance(d, datetime):
                return d
            if isinstance(d, date):
                return datetime.combine(d, datetime.min.time())
            return datetime.strptime(str(d), "%Y.%m.%d")

        ranges: List[Tuple[str, str, str]] = []
        # Robust type narrowing for static checkers
        if oos_spans is None:
            oos_spans_list: List[Tuple[Union[datetime, str], Union[datetime, str], Optional[str]]] = []
        else:
            oos_spans_list = oos_spans
        for tup in oos_spans_list:
            if len(tup) == 2:
                tup = cast(Tuple[Union[datetime, str], Union[datetime, str]], tup)
                fr, to = tup
                typ = "oos"
            elif len(tup) == 3:
                fr, to, typ = tup
                typ = typ or "oos"
            else:
                raise ValueError("oos_spans tuples must have length 2 or 3")
            ranges.append((_fmt(fr), _fmt(to), typ.lower()))

        ranges.sort(key=lambda r: _to_dt(r[0]))

        # <Setup …>
        setup = root.find(".//Data/Setups/Setup")
        if setup is not None:
            setup.set("dateFrom", _fmt(in_from))
            setup.set("dateTo",   _fmt(in_to))

        # Rebuild <OutOfSample>
        data_node = root.find(".//Data")
        if data_node is None:
            return
        if (old := data_node.find("OutOfSample")) is not None:
            data_node.remove(old)
        if ranges:
            new_oos = ET.SubElement(data_node, "OutOfSample", {"showGraph": "false"})
            for fr, to, typ in ranges:
                attribs = {"dateFrom": fr, "dateTo": to}
                if typ != "oos":
                    attribs["type"] = typ
                ET.SubElement(new_oos, "Range", attribs)

    def patch_other_markets(root: ET.Element, end_date: Optional[datetime] = None) -> None:
        setups = root.findall(".//RetestOnAdditionalMarkets/Settings/Setups/Setup")
        logger.debug("patch_other_markets() – found %d setups", len(setups))

        # Defensive: ensure sym_info.last_date and sym_2_info.last_date are not None
        if sym_info.last_date is None or sym_2_info.last_date is None:
            logging.warning("Cannot determine last_date for one of the symbols; skipping patch_other_markets.")
            return

        if end_date is None:
            end_date = min(sym_info.last_date, sym_2_info.last_date)

        if len(setups) >= 2:
            chart0 = setups[0].find("Chart")
            if chart0 is not None:
                chart0.set("symbol", symbol_dukascopy)
            if sym_2_info.first_date is not None:
                setups[0].set("dateFrom", sym_2_info.first_date.strftime("%Y.%m.%d"))
            if end_date is not None:
                setups[0].set("dateTo", end_date.strftime("%Y.%m.%d"))

            chart1 = setups[1].find("Chart")
            if chart1 is not None:
                chart1.set("symbol", symbol_darwinex if symbol_darwinex is not None else symbol_dukascopy)
            if sym_2_info.first_date is not None:
                setups[1].set("dateFrom", sym_2_info.first_date.strftime("%Y.%m.%d"))
            if end_date is not None:
                setups[1].set("dateTo", end_date.strftime("%Y.%m.%d"))

    def patch_rhp_spread(root: ET.Element, spread: float) -> None:
        logger.debug("patch_rhp_spread(spread=%s)", spread)
        spread_el = root.find(".//RetestWithHigherPrecision/Settings/Spread")
        if spread_el is not None:
            spread_el.text = str(spread)

    def patch_disable_task(root: ET.Element, xml_name: str) -> None:
        for task in root.findall(".//Tasks/Task"):
            if task.get("taskXMLFile") == xml_name:
                logger.debug("patch_disable_task() – disabling %s", xml_name)
                task.set("active", "false")

    def patch_input_databank(root: ET.Element, new_value: str = "E-OOS1") -> None:
        for db in root.findall(".//Databank[@label='Input databank'][@name='Input']"):
            if db.get("value") != new_value:
                logger.debug("patch_input_databank() – %s → %s", db.get("value"), new_value)
                db.set("value", new_value)

    def patch_mc_spread(root: ET.Element, min_spread: float, max_spread: float) -> None:
        if sym_info.spread is None:
            return

        params = root.find(
            ".//MonteCarloRetest/Settings/Methods/"
            "Method[@type='RandomizeSpread']/Params"
        )
        if params is None:
            return

        for p in params.findall("Param"):
            key = p.get("key")
            if key == "Min":
                p.text = str(min_spread)
            elif key == "Max":
                p.text = str(max_spread)

    def patch_hbp_spread(root: ET.Element, spread: float) -> None:
        """Patch the spread value in HigherPrecision settings."""
        logging.debug("patch_hbp_spread(spread=%s)", spread)
        spread_el = root.find(".//RetestWithHigherPrecision/Settings/Spread")
        if spread_el is not None:
            spread_el.text = str(spread)
        else:
            logging.warning("patch_hbp_spread: Spread element not found in RetestWithHigherPrecision/Settings")

    # ------------------------------------------------------------------
    #  Perform all mutations in a *single* ZipEditor instance ---------------
    # ------------------------------------------------------------------
    editor = ZipEditor(dest_cfx)

    # Config & global market side
    editor.patch("config.xml", patch_config)

    if template == Settings.template_path_ramon:
        # Build tasks -----------------------------------------------------------
        for i in range(1, 3):
            xml = f"Build-Task{i}.xml"
            editor.patch(xml, patch_market_side)
            editor.patch(xml, patch_setup)

        # Retest tasks ----------------------------------------------------------
        for i in range(1, 14):
            xml = f"Retest-Task{i}.xml"
            editor.patch(xml, patch_setup)

        # Other markets / Spread / Dates ---------------------------------------
        editor.patch("Retest-Task2.xml", patch_other_markets, datetime(2025, 1, 1))
        editor.patch("Retest-Task8.xml", patch_other_markets)

        if symbol_darwinex is None:
            editor.patch("config.xml", patch_disable_task, "Retest-Task2.xml")
            editor.patch("config.xml", patch_disable_task, "Retest-Task8.xml")
            editor.patch("Retest-Task3.xml", patch_input_databank, "E-OOS1")
            editor.patch("Retest-Task9.xml", patch_input_databank, "S-OOS1")

        if sym_info.spread is not None:
            editor.patch("Retest-Task3.xml", patch_rhp_spread, sym_info.spread * 2)
            editor.patch("Retest-Task9.xml", patch_rhp_spread, sym_info.spread * 3)
        else:
            logging.warning("sym_info.spread is None; skipping patch_rhp_spread.")

        # Save / Load folders ---------------------------------------------------
        editor.patch("SaveToFiles-Task1.xml", patch_save_to_files, project_dir / "01 - Edge")
        editor.patch("SaveToFiles-Task2.xml", patch_save_to_files, project_dir / "03 - Strategy")
        editor.patch("SaveToFiles-Task3.xml", patch_save_to_files, None, project_dir / "04 - MQL/MQL5")
        editor.patch("SaveToFiles-Task4.xml", patch_save_to_files, None, project_dir / "04 - MQL/MQL4")

        editor.patch("LoadFromFiles-Task1.xml", patch_load_from_files, project_dir / "02 - To Strategy")

        # External script -------------------------------------------------------
        # Encode paths as base64 to avoid quoting issues on Unix
        path_from_b64 = base64.b64encode(str((project_dir / '01 - Edge').resolve()).encode('utf-8')).decode('utf-8')
        path_to_b64 = base64.b64encode(str((project_dir / '02 - To Strategy').resolve()).encode('utf-8')).decode('utf-8')
        cmd = f"remove_eab_b64 {path_from_b64} {path_to_b64}"
        editor.patch("CallExternalScript-Task1.xml", patch_call_external, cmd)

        # Rename files in all project directories
        file_prefix = SETTINGS.file_prefix_tpl.format(
            symbol=symbol_dukascopy,
            timeframe=timeframe,
            direction=direction,
        )
        # Encode prefix and directories as base64 to avoid quoting issues on Unix
        prefix_b64 = base64.b64encode(file_prefix.encode('utf-8')).decode('utf-8')
        dirs_b64 = [
            base64.b64encode(str((project_dir / '01 - Edge').resolve()).encode('utf-8')).decode('utf-8'),
            base64.b64encode(str((project_dir / '02 - To Strategy').resolve()).encode('utf-8')).decode('utf-8'),
            base64.b64encode(str((project_dir / '03 - Strategy').resolve()).encode('utf-8')).decode('utf-8'),
            base64.b64encode(str((project_dir / '04 - MQL/MQL5').resolve()).encode('utf-8')).decode('utf-8'),
            base64.b64encode(str((project_dir / '04 - MQL/MQL4').resolve()).encode('utf-8')).decode('utf-8')
        ]
        rename_cmd = f"rename_files_b64 {prefix_b64} {' '.join(dirs_b64)}"
        editor.patch("CallExternalScript-Task2.xml", patch_call_external, rename_cmd)

        # Date ranges -----------------------------------------------------------
        build_start = datetime(2018, 1, 1)
        build_end = datetime(2025, 1, 1)

        if sym_info.first_date is not None:
            retest_start = max(datetime(2008, 1, 1, tzinfo=timezone.utc), sym_info.first_date)
        else:
            logging.warning("sym_info.first_date is None; using default retest_start.")
            retest_start = datetime(2008, 1, 1, tzinfo=timezone.utc)
        retest_end_edge = datetime(2025, 1, 1)
        retest_end_strategy = sym_info.last_date if sym_info.last_date is not None else datetime(2025, 1, 1)

        oos_ranges_edge = [(retest_start, datetime(2018, 1, 1))]
        oos_ranges_strategy = [
            (retest_start, datetime(2018, 1, 1)),
            (datetime(2025, 1, 1), sym_info.last_date, "isv") if sym_info.last_date is not None else (datetime(2025, 1, 1), datetime(2025, 1, 1), "isv"),
        ]

        for i in range(1, 3):
            xml = f"Build-Task{i}.xml"
            editor.patch(xml, patch_dates, build_start, build_end)

        for i in range(1, 7):
            xml = f"Retest-Task{i}.xml"
            editor.patch(xml, patch_dates, retest_start, retest_end_edge, oos_ranges_edge)

        for i in range(7, 14):
            xml = f"Retest-Task{i}.xml"
            editor.patch(xml, patch_dates, retest_start, retest_end_strategy, oos_ranges_strategy)

    elif template == Settings.template_path_hc:
        # Build tasks -----------------------------------------------------------
        for i in range(1, 4):
            xml = f"Build-Task{i}.xml"
            editor.patch(xml, patch_market_side)
            editor.patch(xml, patch_setup)

        # Retest tasks ----------------------------------------------------------
        for i in range(1, 14):
            xml = f"Retest-Task{i}.xml"
            editor.patch(xml, patch_setup)

        # Other markets / Spread / Dates ---------------------------------------
        editor.patch("Retest-Task3.xml", patch_other_markets)
        editor.patch("Retest-Task9.xml", patch_other_markets)

        if symbol_darwinex is None:
            editor.patch("config.xml", patch_disable_task, "Retest-Task3.xml")
            editor.patch("config.xml", patch_disable_task, "Retest-Task9.xml")
            editor.patch("Retest-Task4.xml", patch_input_databank, "E-MonteCarlo")
            editor.patch("Retest-Task10.xml", patch_input_databank, "S-MonteCarlo")

        if sym_info.spread is not None:
            editor.patch("Retest-Task2.xml", patch_mc_spread, sym_info.spread, sym_info.spread * 3)
            editor.patch("Retest-Task8.xml", patch_mc_spread, sym_info.spread, sym_info.spread * 3)
            
            # Patch HigherPrecision spread for specific tasks
            editor.patch("Retest-Task1.xml", patch_hbp_spread, sym_info.spread)
            editor.patch("Retest-Task7.xml", patch_hbp_spread, sym_info.spread)

        else:
            logging.warning("sym_info.spread is None; skipping patch_mc_spread and patch_hbp_spread.")

        # Save / Load folders ---------------------------------------------------
        editor.patch("SaveToFiles-Task1.xml", patch_save_to_files, project_dir / "01 - Edge")
        editor.patch("SaveToFiles-Task2.xml", patch_save_to_files, project_dir / "03 - Strategy")
        editor.patch("SaveToFiles-Task3.xml", patch_save_to_files, None, project_dir / "04 - MQL/MQL5")
        editor.patch("SaveToFiles-Task4.xml", patch_save_to_files, None, project_dir / "04 - MQL/MQL4")

        editor.patch("LoadFromFiles-Task1.xml", patch_load_from_files, project_dir / "02 - To Strategy")

        # External script -------------------------------------------------------
        # Encode paths as base64 to avoid quoting issues on Unix
        path_from_b64 = base64.b64encode(str((project_dir / '01 - Edge').resolve()).encode('utf-8')).decode('utf-8')
        path_to_b64 = base64.b64encode(str((project_dir / '02 - To Strategy').resolve()).encode('utf-8')).decode('utf-8')
        cmd = f"remove_eab_b64 {path_from_b64} {path_to_b64}"
        editor.patch("CallExternalScript-Task1.xml", patch_call_external, cmd)

        # Rename files in all project directories
        file_prefix = SETTINGS.file_prefix_tpl.format(
            symbol=symbol_dukascopy,
            timeframe=timeframe,
            direction=direction,
        )
        # Encode prefix and directories as base64 to avoid quoting issues on Unix
        prefix_b64 = base64.b64encode(file_prefix.encode('utf-8')).decode('utf-8')
        dirs_b64 = [
            base64.b64encode(str((project_dir / '01 - Edge').resolve()).encode('utf-8')).decode('utf-8'),
            base64.b64encode(str((project_dir / '02 - To Strategy').resolve()).encode('utf-8')).decode('utf-8'),
            base64.b64encode(str((project_dir / '03 - Strategy').resolve()).encode('utf-8')).decode('utf-8'),
            base64.b64encode(str((project_dir / '04 - MQL/MQL5').resolve()).encode('utf-8')).decode('utf-8'),
            base64.b64encode(str((project_dir / '04 - MQL/MQL4').resolve()).encode('utf-8')).decode('utf-8')
        ]
        rename_cmd = f"rename_files_b64 {prefix_b64} {' '.join(dirs_b64)}"
        editor.patch("CallExternalScript-Task2.xml", patch_call_external, rename_cmd)

        # Date ranges -----------------------------------------------------------
        build_start = datetime(2006, 1, 1) if sym_info.first_date < datetime(2010, 1, 1, tzinfo=timezone.utc) else sym_info.first_date
        build_end = datetime(2020, 1, 1) if sym_info.first_date < datetime(2010, 1, 1, tzinfo=timezone.utc) else datetime(2021, 1, 1)

        retest_start = build_start
        retest_end = sym_info.last_date

        oos_ranges = [(build_end, retest_end)]

        for i in range(1, 4):
            xml = f"Build-Task{i}.xml"
            editor.patch(xml, patch_dates, build_start, build_end)

        for i in range(1, 7):
            xml = f"Retest-Task{i}.xml"
            editor.patch(xml, patch_dates, retest_start, retest_end, oos_ranges)

        for i in range(7, 14):
            xml = f"Retest-Task{i}.xml"
            editor.patch(xml, patch_dates, retest_start, retest_end, oos_ranges)

    # ---- finally write out -------------------------------------------------
    editor.write()
    log("project %s created successfully", project_dir.name)
    logging.debug(f"Project {project_dir.name} created successfully at {project_dir}")

# ─────────────────────────────────────────────────────────────────────────────
#  Interactive wizard & CLI helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Question:
    key: str
    prompt: str
    validate: Callable[[str], bool]        # str → bool
    transform: Callable[[str], Any] = str # str → Any
    error: str = "→ invalid value\n"
    default: Optional[str] = None


def ask(questions: List[Question]) -> dict[str, object]:
    """Prompt until every question is answered with a *valid* value."""
    logging.debug("ask() – %d questions", len(questions))
    answers: dict[str, object] = {}
    for q in questions:
        while True:
            raw = input(q.prompt).strip()
            if raw == "" and q.default is not None:
                answers[q.key] = q.default
                break
            if q.validate(raw):
                answers[q.key] = q.transform(raw)
                break
            print(q.error)
    logging.debug("Collected answers: %s", answers)
    return answers


def launch_cli() -> None:
    """Interactive wizard for non-technical users."""
    require_symbols_db()
    QUESTIONS = [
        Question(
            key="template",
            prompt="Project Template (R=Ramon / H=Hobbiecode): ",
            validate=lambda s: s.upper() in {"R", "RAMON", "RAMÓN", "H", "HOBBIECODE"},
            transform=lambda s: "Ramon" if s.upper().startswith("R") else "Hobbiecode",
        ),
        Question(
            key="symbol_dukascopy",
            prompt="Symbol Name with Dukascopy data (e.g. EURUSD_dukascopy_darwinex): ",
            validate=lambda s: symbol_exists(s),
            error="→ invalid symbol\n",
        ),
        Question(
            key="symbol_darwinex",
            prompt="[Optional] Symbol Name with Darwinex data (e.g. EURUSD_darwinex_darwinex): ",
            validate=lambda s: s == "" or symbol_exists(s),
            error="→ invalid symbol\n",
            default="",
        ),
        Question(
            key="timeframe",
            prompt="Time-frame (e.g. H4, H1, D1): ",
            validate=lambda s: bool(re.fullmatch(r"[A-Z]\d?", s.strip(), re.I)),
            transform=str.upper,
        ),
        Question(
            key="direction",
            prompt="Direction  (L=Long / S=Short): ",
            validate=lambda s: s.upper() in {"L", "LONG", "S", "SHORT"},
            transform=lambda s: "Long" if s.upper().startswith("L") else "Short",
        ),
    ]

    print("=== Create a New SQX Project === (Ctrl-C to exit)\n")
    try:
        answers = ask(QUESTIONS)
        # Fire the project generator
        import argparse
        newproject(argparse.Namespace(**answers))
        print("\nDone.")
        print("\nYou can close this window…")
        input()
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        print("\nYou can close this window…\n")

# ─────────────────────────────────────────────────────────────────────────────
#  rename_files implementation
# ─────────────────────────────────────────────────────────────────────────────

def rename_files(args: argparse.Namespace) -> None:
    """Rename all *.sqx, *.mq4, *.mq5 files in given directories according to prefix and format rules."""
    prefix = args.prefix.strip()
    directories = args.directories
    logging.debug(f"rename_files(prefix={prefix}, directories={directories})")
    exts = {'.sqx', '.mq4', '.mq5'}
    # Regex for Format 1: Strategy XXX.sqx/mq4/mq5
    re_fmt1 = re.compile(r'^Strategy (.+?)(\.[sm]q[45x])$')
    # Regex for Format 2: Strategy XXX - Improved YYY.sqx/mq4/mq5
    re_fmt2 = re.compile(r'^Strategy (.+?) - Improved (.+?)(\.[sm]q[45x])$')
    for dir_path in directories:
        dir_path = Path(dir_path).expanduser().resolve()
        if not dir_path.is_dir():
            logging.warning(f"Directory not found: {dir_path}")
            continue
        for file in dir_path.iterdir():
            if not file.is_file() or file.suffix.lower() not in exts:
                continue
            m2 = re_fmt2.match(file.name)
            m1 = re_fmt1.match(file.name)
            if m2:
                # Format 2
                xxx, yyy, ext = m2.groups()
                new_name = f"{prefix} {xxx} - {yyy}{ext}"
            elif m1:
                # Format 1
                xxx, ext = m1.groups()
                new_name = f"{prefix} {xxx}{ext}"
            else:
                logging.info(f"Skipping file (no match): {file}")
                continue
            new_path = file.with_name(new_name)
            if new_path.exists():
                logging.warning(f"Target file already exists, skipping: {new_path}")
                continue
            logging.info(f"Renaming {file} -> {new_path}")
            file.rename(new_path)


def rename_files_b64(args: argparse.Namespace) -> None:
    """CLI entry point for **rename_files_b64** sub-command.
    
    Works exactly like rename_files but accepts base64-encoded prefix and directories.
    Decodes the base64 strings and delegates to the core rename_files functionality.
    """
    logging.debug("rename_files_b64(args=%s)", args)
    
    try:
        # Decode base64 strings to get the actual values
        prefix_decoded = base64.b64decode(args.prefix_b64).decode('utf-8')
        directories_decoded = [base64.b64decode(d).decode('utf-8') for d in args.directories_b64]
        
        logging.info("Decoded prefix: %s", prefix_decoded)
        logging.info("Decoded directories: %s", directories_decoded)
        
        # Create a new args object with decoded values
        decoded_args = argparse.Namespace(
            prefix=prefix_decoded,
            directories=directories_decoded
        )
        
        # Delegate to the existing rename_files function
        rename_files(decoded_args)
        
    except Exception as e:
        logging.error("Error in rename_files_b64: %s", str(e))
        print(f"Error decoding base64 parameters: {e}")
        raise

# ─────────────────────────────────────────────────────────────────────────────
#  CLI boilerplate – sub-commands & argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Configure basic logging first to capture all startup information
    configure_logging(0, 0)  # Start with basic logging, will be reconfigured later
    
    # Log script startup with all arguments
    logging.info("=== SQX Tool Started ===")
    logging.info("Script started at: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logging.info("Command line arguments: %s", sys.argv)
    logging.info("Working directory: %s", os.getcwd())
    logging.info("Script location: %s", __file__)
    
    parser = argparse.ArgumentParser(description="StrategyQuantX helper tool")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase log verbosity (-v=DEBUG, -vv=TRACE)"
    )
    parser.add_argument(
        "-q", "--quiet", action="count", default=0,
        help="Decrease log verbosity (-q=WARNING, -qq=ERROR, -qqq=CRITICAL)"
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    # remove_eab ------------------------------------------------------------
    p_eab = subparsers.add_parser("remove_eab", help="strip ExitAfterBars from .sqx files")
    p_eab.add_argument("path_from")
    p_eab.add_argument("path_to")
    p_eab.add_argument("-j", "--jobs", type=int, help="workers (default: CPU count)")
    p_eab.set_defaults(func=remove_eab)

    # remove_eab_b64 --------------------------------------------------------
    p_eab_b64 = subparsers.add_parser("remove_eab_b64", help="strip ExitAfterBars from .sqx files using base64-encoded paths")
    p_eab_b64.add_argument("path_from_b64", help="base64-encoded source directory path")
    p_eab_b64.add_argument("path_to_b64", help="base64-encoded destination directory path")
    p_eab_b64.add_argument("-j", "--jobs", type=int, help="workers (default: CPU count)")
    p_eab_b64.set_defaults(func=remove_eab_b64)

    # newproject ------------------------------------------------------------
    p_new = subparsers.add_parser("newproject", help="scaffold a new SQX project")
    p_new.add_argument("template", choices=["Ramon", "Hobbiecode"])
    p_new.add_argument("symbol_dukascopy")
    p_new.add_argument("symbol_darwinex")
    p_new.add_argument("timeframe")
    p_new.add_argument("direction", choices=["Long", "Short"])
    p_new.set_defaults(func=newproject)

    # rename_files ----------------------------------------------------------
    p_rename = subparsers.add_parser("rename_files", help="rename strategy files with a prefix in given directories")
    p_rename.add_argument("prefix", help="Prefix to use for renamed files")
    p_rename.add_argument("directories", nargs='+', help="Directories to process")
    p_rename.set_defaults(func=rename_files)

    # rename_files_b64 ------------------------------------------------------
    p_rename_b64 = subparsers.add_parser("rename_files_b64", help="rename strategy files with a prefix using base64-encoded parameters")
    p_rename_b64.add_argument("prefix_b64", help="base64-encoded prefix to use for renamed files")
    p_rename_b64.add_argument("directories_b64", nargs='+', help="base64-encoded directories to process")
    p_rename_b64.set_defaults(func=rename_files_b64)

    try:
        args = parser.parse_args()
        # Reconfigure logging with the correct verbosity/quiet settings
        configure_logging(args.verbose, args.quiet)
        
        logging.info("Successfully parsed CLI arguments: %s", args)
        logging.info("Command: %s", args.command or "interactive_mode")

        if not args.command:
            logging.info("No command specified, launching interactive CLI")
            launch_cli()
        else:
            # Default for jobs → CPU count when not specified
            if args.command in ("remove_eab", "remove_eab_b64") and args.jobs is None:
                args.jobs = os.cpu_count() or 1
                logging.info("Set jobs to CPU count: %s", args.jobs)
            logging.info("Executing command: %s", args.command)
            args.func(args)
            logging.info("Command completed successfully: %s", args.command)
            
    except SystemExit as e:
        # This catches argparse errors and help/version requests
        if e.code == 2:  # Invalid arguments
            logging.error("Invalid command line arguments provided")
            logging.error("Arguments were: %s", sys.argv)
        elif e.code == 0:  # Help or version requested
            logging.info("Help or version information requested")
        else:
            logging.error("SystemExit with code %s", e.code)
        raise
    except Exception as e:
        logging.error("Unexpected error during argument parsing or execution: %s", str(e))
        logging.error("Arguments were: %s", sys.argv)
        raise
    finally:
        logging.info("=== SQX Tool Finished ===")

# ─────────────────────────────────────────────────────────────────────────────
#  Guard – freeze-support for pyinstaller, spawn on Windows
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    main()

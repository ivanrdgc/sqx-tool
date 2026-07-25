#!/usr/bin/env python3

"""
sqx_tool.py: StrategyQuant X project helper tool

- Scaffolds new SQX projects from a template
- Provides an interactive CLI for non-technical users
- Handles all file and directory management, XML patching, and logging
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
#  Standard library imports
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import configparser
import io
import logging
import os
import re
import sqlite3
import sys
from datetime import datetime, date, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, Any, List, Union
import zipfile
import xml.etree.ElementTree as ET

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

def _load_config_path(repo_root: Path, key: str, fallback: str) -> Path:
    """Read a path setting from the [sqx] section of config.ini at the repo root.

    Falls back to *fallback* when the file or key is missing. Backslashes in
    the value are normalized so Path behaves sensibly on non-Windows hosts.
    """
    config_path = repo_root / "config.ini"
    parser = configparser.ConfigParser()
    parser.read(config_path, encoding="utf-8")
    raw = parser.get("sqx", key, fallback=fallback)
    if os.name != "nt":
        raw = raw.replace("\\", "/")
    return Path(raw)


_SCRIPT_DIR = Path(__file__).resolve().parent
_SQX_PATH = _load_config_path(_SCRIPT_DIR.parent, "SQX_PATH", r"C:\SQX_143")
_MT5_PATH = _load_config_path(_SCRIPT_DIR.parent, "MT5_PATH", r"C:\MetaTrader\MetaTrader 5")


@dataclass(frozen=True)
class Settings:
    """All file‑system locations & naming conventions live here."""

    script_dir: Path = _SCRIPT_DIR
    template_dir: Path = (_SCRIPT_DIR / "Template").resolve()
    projects_base: Path = (_SCRIPT_DIR / "../Projects").resolve()
    log_file: Path = (_SCRIPT_DIR / "sqx_tool.log").resolve()
    # Default log level when no -v/-q flags are provided.
    # One of: "trace", "debug", "info", "warning", "error", "critical"
    default_log_level: str = "info"

    project_dir_tpl: str = "{symbol}/{timestamp}_{symbol}_{timeframe}_{direction}"
    # Prefix handed to the RenameStrategies custom analysis, so strategies are
    # named e.g. "XAUUSD H1 Long 12345" instead of "Strategy 12345".
    strategy_prefix_tpl: str = "{symbol} {timeframe} {direction}"

    sqx_path: Path = _SQX_PATH
    symbols_db: Path = (_SQX_PATH / "user" / "data" / "data.db")
    mt5_path: Path = _MT5_PATH

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
    """Utility for reading, patching, and writing zip archives containing XML files.

    Accepts either a zip archive or a directory of extracted files as source.
    """
    def __init__(self, source: Path) -> None:
        self.zip_path = source
        self._files: dict[str, bytes] = {}
        if source.is_dir():
            for p in sorted(source.rglob("*")):
                if p.is_file():
                    entry = p.relative_to(source).as_posix()
                    self._files[entry] = p.read_bytes()
        else:
            with zipfile.ZipFile(source, "r") as zin:
                self._files = {i.filename: zin.read(i.filename) for i in zin.infolist()}

    def patch(self, entry_name: str, mutator: XMLMutator, *args: Any, **kwargs: Any) -> None:
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
#  Path cleaning helper
# ─────────────────────────────────────────────────────────────────────────────

def _clean_path(path_str: str) -> str:
    """Clean a path string by removing quotes and trailing flags.
    
    On Windows, paths with spaces can sometimes include trailing quotes or flags
    if not properly quoted in the command line. This function strips those.
    """
    # Strip leading/trailing whitespace
    path_str = path_str.strip()
    # Remove surrounding quotes if present
    if (path_str.startswith('"') and path_str.endswith('"')) or \
       (path_str.startswith("'") and path_str.endswith("'")):
        path_str = path_str[1:-1]
    
    # Remove any trailing flags that might have been accidentally included
    # (e.g., " -vv", " -v", " -q", etc.) - do this before removing quotes
    # to handle cases like: 'path" -vv'
    path_str = re.sub(r'\s+-[vq]+$', '', path_str)
    
    # Remove any trailing quote that might be left (after flag removal)
    path_str = path_str.rstrip('"').rstrip("'")
    
    # Strip trailing whitespace, but preserve trailing backslashes/slashes
    # by temporarily removing them, stripping, then adding back
    trailing_slash = path_str.endswith('\\') or path_str.endswith('/')
    if trailing_slash:
        path_str = path_str[:-1].rstrip() + path_str[-1]
    else:
        path_str = path_str.rstrip()
    
    return path_str

# ─────────────────────────────────────────────────────────────────────────────
#  newproject implementation
# ─────────────────────────────────────────────────────────────────────────────

def newproject(args: argparse.Namespace) -> None:
    """Scaffold a brand-new *StrategyQuant X* project from the template."""
    require_symbols_db()
    logger.debug("newproject(args=%s)", args)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    template: Path = SETTINGS.template_dir
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

    # Raw symbol, without the _darwinex/_dx suffix, e.g. "EURUSD" from
    # "EURUSD_darwinex". Drives the project/folder name and the strategy prefix.
    base_symbol = symbol_dukascopy.split("_")[0]

    if not template.is_dir():
        logging.error("template not found in %s", template)
        print(f"template not found in {template}")
        return

    # ---- 3. Create project directory tree ---------------------------------
    project_rel = SETTINGS.project_dir_tpl.format(
        symbol=base_symbol,
        timestamp=ts,
        timeframe=timeframe,
        direction=direction,
    )
    project_dir = SETTINGS.projects_base / project_rel
    project_dir.mkdir(parents=True, exist_ok=True)
    log("project dir: %s", project_dir)
    logger.debug("Created project directory: %s", project_dir)

    subdirs = (
        "01 - E-Build", "02 - E-Retests", "03 - E-Final",
        "04 - S-Build", "05 - S-Retests 1", "06 - S-Retests 2", "07 - S-Final",
        "08 - S-Final Demo", "09 - S-Darwinex",
    )
    for sub in subdirs:
        (project_dir / sub).mkdir(parents=True, exist_ok=True)
        logger.log(TRACE_LEVEL, "Created subdir %s", sub)

    # ---- 4. Prepare output .cfx path --------------------------------------
    dest_cfx = project_dir / f"{project_dir.name}.cfx"
    logger.debug("Project .cfx will be built at %s from template dir %s", dest_cfx, template)
    project_dirs = [(project_dir / sub).resolve() for sub in subdirs]
    (e_build_dir, e_retests_dir, e_final_dir, s_build_dir, s_retests_1_dir, s_retests_2_dir,
     s_final_dir, s_final_demo_dir, s_darwinex_dir) = project_dirs

    # ----------------------------------------------------------------------
    #  Helper mutators – declared *inside* newproject so they can capture
    #  surrounding variables without global state.
    # ----------------------------------------------------------------------

    def patch_config(root: ET.Element) -> None:
        logger.debug("patch_config() – setting project name → %s", project_dir.name)
        root.set("name", project_dir.name)

    def patch_save_to_files(root: ET.Element,
                            sqx_dir: Optional[Path] = None,
                            sc_dir:  Optional[Path] = None) -> None:
        if sqx_dir is not None:
            node = root.find(".//SaveToFiles/DestDirectorySqx")
            if node is not None:
                logger.log(TRACE_LEVEL, "patch_save_to_files() – sqx_dir=%s", sqx_dir)
                node.text = str(sqx_dir.resolve())
        if sc_dir is not None:
            node_sc = root.find(".//SaveToFiles/DestDirectorySC")
            if node_sc is not None:
                logger.log(TRACE_LEVEL, "patch_save_to_files() – sc_dir=%s", sc_dir)
                node_sc.text = str(sc_dir.resolve())

    def patch_market_side(root: ET.Element) -> None:
        ms = root.find(".//MarketSides")
        if ms is not None:
            logger.debug("patch_market_side() → %s", direction.lower())
            ms.set("type", direction.lower())

    def patch_setup(
        root: ET.Element,
        symbol: str = symbol_dukascopy,
        info: SymbolInfo = sym_info,
        use_swap: bool = True,
        use_commission: bool = True,
        use_spread: bool = True,
    ) -> None:
        setup = root.find(".//Setups/Setup")
        if setup is None:
            return
        # Chart -------------------------------------------------------------
        chart = setup.find("Chart")
        if chart is not None:
            logger.debug("patch_setup.chart() – %s @ %s", symbol, timeframe)
            chart.set("symbol", symbol)
            chart.set("timeframe", timeframe)
            if info.spread is not None and use_spread:
                chart.set("spread", str(info.spread))
            else:
                chart.set("spread", str(0))

        # Commissions -------------------------------------------------------
        comm_parent = setup.find("Commissions")
        if comm_parent is None:
            comm_parent = ET.SubElement(setup, "Commissions")
        comm_parent.clear()
        comm_xml = info.commission if (info.commission and use_commission) else '<Method type="None" use="true"><Params/></Method>'
        try:
            comm_parent.append(ET.fromstring(comm_xml))
        except ET.ParseError as exc:
            logging.warning("bad commission XML for %s: %s", symbol, exc)

        # Swap --------------------------------------------------------------
        old = setup.find("Swap")
        if old is not None:
            setup.remove(old)
        swap_xml = info.swap if (info.swap and use_swap) else '<Swap use="false" />'
        try:
            setup.append(ET.fromstring(swap_xml))
        except ET.ParseError as exc:
            logging.warning("bad swap XML for %s: %s", symbol, exc)

    def patch_dates(
        root: ET.Element,
        in_from: Union[datetime, str],
        in_to: Union[datetime, str],
        oos_spans: Optional[List[Tuple[Union[datetime, str], Union[datetime, str], Optional[str]]]] = None,
    ) -> None:
        """Rewrite <Setup> date range and <OutOfSample> block in *root*.

        *oos_spans* is a list of ``(from, to, type)`` tuples where *type* is
        usually ``'oos'`` or ``'isv'``. ``None`` is treated as ``'oos'``.
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
        for fr, to, typ in oos_spans or []:
            ranges.append((_fmt(fr), _fmt(to), (typ or "oos").lower()))

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
            setups[0].set("dateTo", end_date.strftime("%Y.%m.%d"))

            chart1 = setups[1].find("Chart")
            if chart1 is not None:
                chart1.set("symbol", symbol_darwinex if symbol_darwinex is not None else symbol_dukascopy)
            if sym_2_info.first_date is not None:
                setups[1].set("dateFrom", sym_2_info.first_date.strftime("%Y.%m.%d"))
            setups[1].set("dateTo", end_date.strftime("%Y.%m.%d"))

    def patch_custom_analysis(root: ET.Element, method: str, input_args: str) -> None:
        """Set the inputArgs of a <CustomAnalysis> block driving *method*.

        Only blocks already declaring that method are touched, so a task that
        uses a different custom analysis - or none at all - is left alone.
        """
        found = 0
        for node in root.iter("CustomAnalysis"):
            if node.get("method") != method:
                continue
            logger.debug("patch_custom_analysis() – %s inputArgs=%s", method, input_args)
            node.set("inputArgs", input_args)
            found += 1
        if found == 0:
            logging.warning(
                "no <CustomAnalysis method='%s'> block found – strategies will keep "
                "their default names", method,
            )

    def patch_disable_crosscheck(root: ET.Element, name: str) -> None:
        """Turn off a single cross-check inside a Retest task's <CrossChecks>.

        Used for the additional-markets comparison when no Darwinex symbol is
        given: the rest of the retest, including its other cross-checks, still
        runs, and the task's databank chain is untouched (the check lives inside
        the retest rather than being a standalone task).
        """
        found = 0
        for cc in root.findall(f".//CrossChecks/{name}"):
            logger.debug("patch_disable_crosscheck() – disabling %s", name)
            cc.set("use", "false")
            found += 1
        if found == 0:
            logging.warning("no <%s> cross-check found to disable", name)

    # ------------------------------------------------------------------
    #  Perform all mutations in a *single* ZipEditor instance ---------------
    # ------------------------------------------------------------------
    editor = ZipEditor(SETTINGS.template_dir)

    # Date ranges -----------------------------------------------------------
    if sym_info.first_date is None or sym_info.last_date is None or sym_2_info.first_date is None or sym_2_info.last_date is None:
        logging.error("symbol '%s' has no date range in symbols DB – aborting", symbol_dukascopy)
        print(f"symbol '{symbol_dukascopy}' has no date range in symbols DB – aborting")
        return

    first_day = max(datetime(2010, 1, 1, tzinfo=timezone.utc), sym_info.first_date)
    last_day = sym_info.last_date

    first_day_dx = max(datetime(2010, 1, 1, tzinfo=timezone.utc), sym_2_info.first_date)
    last_day_dx = sym_2_info.last_date

    build_start = max(datetime(2019, 1, 1, tzinfo=timezone.utc), first_day)
    build_end = min(datetime(2025, 7, 1, tzinfo=timezone.utc), last_day)

    retest_start = max(datetime(2010, 1, 1, tzinfo=timezone.utc), first_day)
    retest_start_dx = max(datetime(2010, 1, 1, tzinfo=timezone.utc), first_day_dx)
    retest_end = build_end
    retest_end_final = last_day
    retest_end_final_dx = last_day_dx

    oos_ranges: List[Tuple[Union[datetime, str], Union[datetime, str], Optional[str]]] = [
        (retest_start, build_start, "oos"),
    ]
    oos_ranges_final: List[Tuple[Union[datetime, str], Union[datetime, str], Optional[str]]] = [
        (retest_start, build_start, "oos"),
        (build_end, retest_end_final, "oos"),
    ]
    oos_ranges_final_dx: List[Tuple[Union[datetime, str], Union[datetime, str], Optional[str]]] = [
        (build_end, retest_end_final_dx, "oos"),
    ]
    if build_start > retest_start_dx:
        oos_ranges_final_dx.insert(0, (retest_start_dx, build_start, "oos"))

    # Build tasks: date range ----------------------------------------------
    for i in range(1, 3):
        editor.patch(f"Build-Task{i}.xml", patch_dates, build_start, build_end)

    # Retest tasks: date range. Numbering follows config.xml execution order:
    #   1 E-Retests        OOS retest + cross-checks   normal OOS
    #   2 E-Final          final retest                final OOS
    #   3 S-Retests 1      OOS retest + cross-checks   normal OOS
    #   4 S-Retests 2      advanced cross-checks       normal OOS
    #   5 S-Clean Strategy final retest                final OOS
    #   6 S-Darwinex Tick  tick retest on Darwinex     Darwinex final OOS
    editor.patch("Retest-Task1.xml", patch_dates, retest_start,    retest_end,          oos_ranges)
    editor.patch("Retest-Task2.xml", patch_dates, retest_start,    retest_end_final,    oos_ranges_final)
    editor.patch("Retest-Task3.xml", patch_dates, retest_start,    retest_end,          oos_ranges)
    editor.patch("Retest-Task4.xml", patch_dates, retest_start,    retest_end,          oos_ranges)
    editor.patch("Retest-Task5.xml", patch_dates, retest_start,    retest_end_final,    oos_ranges_final)
    editor.patch("Retest-Task6.xml", patch_dates, retest_start_dx, retest_end_final_dx, oos_ranges_final_dx)

    # Config name
    editor.patch("config.xml", patch_config)

    # Build tasks: market side + main-chart setup --------------------------
    for i in range(1, 3):
        editor.patch(f"Build-Task{i}.xml", patch_market_side)
        editor.patch(f"Build-Task{i}.xml", patch_setup, symbol_dukascopy, sym_info, False, False, False)

    # Retest tasks: main-chart setup ---------------------------------------
    #   E/S retests, E-Final and S-Clean Strategy all trade the Dukascopy
    #   symbol; S-Darwinex Tick uses the Darwinex symbol (with real swap/
    #   commission/spread), falling back to the Dukascopy symbol when none.
    editor.patch("Retest-Task1.xml", patch_setup, symbol_dukascopy, sym_info, False, False, False)
    editor.patch("Retest-Task2.xml", patch_setup, symbol_dukascopy, sym_info, False, False, False)
    editor.patch("Retest-Task3.xml", patch_setup, symbol_dukascopy, sym_info, False, False, False)
    editor.patch("Retest-Task4.xml", patch_setup, symbol_dukascopy, sym_info, False, False, False)
    editor.patch("Retest-Task5.xml", patch_setup, symbol_dukascopy, sym_info, False, False, False)
    editor.patch("Retest-Task6.xml", patch_setup, symbol_darwinex or symbol_dukascopy, sym_2_info)

    # "Other markets" is now a cross-check inside E-Retests and S-Retests
    # (merged from the old standalone tasks), so its two extra market setups
    # are patched there rather than in a task of their own.
    editor.patch("Retest-Task1.xml", patch_other_markets, build_end)
    editor.patch("Retest-Task3.xml", patch_other_markets, build_end)

    # Without a Darwinex symbol the additional-market comparison is
    # meaningless, so disable just that cross-check – the rest of each retest,
    # including its other cross-checks, still runs.
    if symbol_darwinex is None:
        editor.patch("Retest-Task1.xml", patch_disable_crosscheck, "RetestOnAdditionalMarkets")
        editor.patch("Retest-Task3.xml", patch_disable_crosscheck, "RetestOnAdditionalMarkets")

    # Strategy naming -------------------------------------------------------
    # RenameStrategies turns "Strategy 12345" into "XAUUSD H1 Long 12345" at
    # E-Build. Later tasks leave inputArgs empty: by then the name already
    # carries the prefix and only "Improved" needs dropping.
    strategy_prefix = SETTINGS.strategy_prefix_tpl.format(
        symbol=base_symbol,
        timeframe=timeframe,
        direction=direction,
    )
    editor.patch("Build-Task1.xml", patch_custom_analysis, "RenameStrategies", strategy_prefix)
    log("strategy prefix: %s", strategy_prefix)

    # Save folders – one per SaveToFiles task, matching its title ----------
    editor.patch("SaveToFiles-Task1.xml", patch_save_to_files, e_build_dir)       # E-Build Save
    editor.patch("SaveToFiles-Task2.xml", patch_save_to_files, e_retests_dir)     # E-Retests Save
    editor.patch("SaveToFiles-Task3.xml", patch_save_to_files, e_final_dir)       # E-Save Final
    editor.patch("SaveToFiles-Task4.xml", patch_save_to_files, s_build_dir)       # S-Build Save
    editor.patch("SaveToFiles-Task5.xml", patch_save_to_files, s_retests_1_dir)   # S-Retests 1 Save
    editor.patch("SaveToFiles-Task6.xml", patch_save_to_files, s_retests_2_dir)   # S-Retests 2 Save
    editor.patch("SaveToFiles-Task7.xml", patch_save_to_files, s_final_dir)       # S-Save Final
    editor.patch("SaveToFiles-Task8.xml", patch_save_to_files, s_final_demo_dir)  # S-Save Final Demo
    editor.patch("SaveToFiles-Task9.xml", patch_save_to_files, s_darwinex_dir)    # S-Save Darwinex

    # ---- finally write out -------------------------------------------------
    editor.write(dest_cfx)
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
            key="symbol",
            prompt="Symbol (e.g. XAUUSD): ",
            validate=lambda s: symbol_exists(f"{s}_darwinex"),
            error="→ invalid symbol (no *_darwinex entry in DB)\n",
        ),
        Question(
            key="timeframe",
            prompt="Time-frame (e.g. H4, H1, D1): ",
            validate=lambda s: bool(re.fullmatch(r"[A-Z]\d+", s.strip(), re.I)),
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
        symbol = str(answers.pop("symbol"))
        answers["symbol_dukascopy"] = f"{symbol}_darwinex"
        dx_candidate = f"{symbol}_dx_darwinex"
        answers["symbol_darwinex"] = dx_candidate if symbol_exists(dx_candidate) else ""
        newproject(argparse.Namespace(**answers))
        print("\nDone.")
        print("\nYou can close this window…")
        input()
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        print("\nYou can close this window…\n")

# ─────────────────────────────────────────────────────────────────────────────
#  remove_duplicates implementation
# ─────────────────────────────────────────────────────────────────────────────

def remove_duplicate_files(src_dir: Path, dest_dir: Path) -> None:
    """Remove all files from src_dir that are present in dest_dir (matching by filename).
    
    Args:
        src_dir: Source directory containing files to potentially remove
        dest_dir: Destination directory to check for matching filenames
    """
    src_dir = Path(src_dir).expanduser().resolve()
    dest_dir = Path(dest_dir).expanduser().resolve()
    
    logging.debug("remove_duplicate_files(src_dir=%s, dest_dir=%s)", src_dir, dest_dir)
    
    if not src_dir.is_dir():
        logging.warning("Source directory not found: %s", src_dir)
        return
    
    if not dest_dir.is_dir():
        logging.warning("Destination directory not found: %s", dest_dir)
        return
    
    # Get all filenames in dest_dir
    dest_files = {f.name for f in dest_dir.iterdir() if f.is_file()}
    log("Found %d files in destination directory %s", len(dest_files), dest_dir)
    logger.log(TRACE_LEVEL, "Destination files: %s", sorted(dest_files))
    
    # Find and remove matching files from src_dir
    removed_count = 0
    for file in src_dir.iterdir():
        if not file.is_file():
            continue
        if file.name in dest_files:
            try:
                file.unlink()
                removed_count += 1
                log("Removed %s (exists in %s)", file.name, dest_dir)
                logger.debug("Deleted file: %s", file)
            except Exception as exc:
                logging.error("Error removing file %s: %s", file, exc)
    
    log("Removed %d duplicate file(s) from %s", removed_count, src_dir)


def remove_duplicates(args: argparse.Namespace) -> None:
    """CLI entry point for **remove_duplicates** sub-command."""
    logging.debug("remove_duplicates(args=%s)", args)
    
    # Handle case where paths might be combined into one argument (Windows quoting issues)
    paths = args.paths if hasattr(args, 'paths') else [args.src_dir, args.dest_dir]
    
    # If we only got one argument, try to split it intelligently
    if len(paths) == 1:
        # Try to split on patterns like: path" path or path' path
        # Look for quote followed by space and what looks like a path
        combined = paths[0]
        logger.debug("Attempting to split combined path argument: %s", combined)
        
        # Try to find a split point: quote (possibly after backslash) followed by space and a path-like string
        # Pattern: ...Param\" .\path or ...Param" .\path
        match = re.search(r'\\?["\']\s+(.+)$', combined)
        if match:
            # Split at the quote - find the position of the quote
            quote_pos = combined.rfind('"')
            if quote_pos == -1:
                quote_pos = combined.rfind("'")
            if quote_pos > 0:
                src_dir_str = combined[:quote_pos].rstrip()
                dest_dir_str = match.group(1)
                logger.debug("Split at quote: src=%s, dest=%s", src_dir_str, dest_dir_str)
            else:
                # Quote found but at start, which is odd - try the match group approach
                src_dir_str = combined[:match.start()].rstrip().rstrip('"').rstrip("'")
                dest_dir_str = match.group(1)
        else:
            # Try splitting on a pattern like: ...Param\ ...WFM
            # Look for a space followed by a path-like pattern (starts with . or letter, has backslashes)
            match = re.search(r'\s+([.\\a-zA-Z].*)$', combined)
            if match:
                # Estimate split point - find the last space before what looks like a second path
                split_pos = combined.rfind(' ', 0, match.start())
                if split_pos > 0:
                    src_dir_str = combined[:split_pos].rstrip()
                    dest_dir_str = match.group(1)
                    logger.debug("Split at space pattern: src=%s, dest=%s", src_dir_str, dest_dir_str)
                else:
                    # Can't split intelligently
                    raise ValueError(f"Could not parse combined paths. Got: {combined}")
            else:
                raise ValueError(f"Expected 2 paths but got 1 combined argument: {combined}")
    elif len(paths) == 2:
        src_dir_str = paths[0]
        dest_dir_str = paths[1]
    else:
        # More than 2 - combine extras into dest_dir
        src_dir_str = paths[0]
        dest_dir_str = ' '.join(paths[1:])
    
    # Clean paths to handle Windows quoting issues
    src_dir_str = _clean_path(src_dir_str)
    dest_dir_str = _clean_path(dest_dir_str)
    
    src_dir = Path(src_dir_str).expanduser().resolve()
    dest_dir = Path(dest_dir_str).expanduser().resolve()
    
    remove_duplicate_files(src_dir, dest_dir)

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

    # newproject ------------------------------------------------------------
    p_new = subparsers.add_parser("newproject", help="scaffold a new SQX project")
    p_new.add_argument("symbol_dukascopy")
    p_new.add_argument("symbol_darwinex")
    p_new.add_argument("timeframe")
    p_new.add_argument("direction", choices=["Long", "Short"])
    p_new.set_defaults(func=newproject)

    # remove_duplicates ------------------------------------------------------
    p_remove_dup = subparsers.add_parser("remove_duplicates", help="remove files from src_dir that exist in dest_dir (matching by filename)")
    # Use nargs='+' to handle cases where paths might be combined due to Windows quoting issues
    p_remove_dup.add_argument("paths", nargs='+', help="source directory and destination directory (2 paths, or 1 combined path that will be split)")
    p_remove_dup.set_defaults(func=remove_duplicates)

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

if __name__ == "__main__":
    main()

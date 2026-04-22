"""
calculate_spreads.py

Build and query a per-day spread cache from a tick CSV.

Two CSV layouts are supported (with or without a header row):
  - ``DateTime, Spread``               (spread precomputed)
  - ``DateTime, Bid, Ask, Volume``     (spread derived as Ask - Bid; Volume ignored)

Use ``csv_to_daily_spread_cache`` to produce a pickle cache of
``{ Day -> np.array(sorted daily spreads) }``, then
``compute_spread_px_py_from_cache`` to compute pX(pY) statistics.
"""
import pickle
import time
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm


def count_lines(path: str | Path, buffer_size: int = 1024 * 1024) -> int:
    with open(path, "rb") as f:
        return sum(buf.count(b"\n") for buf in iter(lambda: f.read(buffer_size), b""))


DATETIME_FORMATS: tuple[str, ...] = ("%Y%m%d %H:%M:%S.%f", "%Y%m%d %H:%M:%S")


def _match_datetime_format(value: str) -> str | None:
    for fmt in DATETIME_FORMATS:
        try:
            pd.to_datetime(value, format=fmt)
            return fmt
        except (ValueError, TypeError):
            continue
    return None


def csv_has_header(csv_path: str | Path) -> bool:
    """
    Detect whether the CSV has a header row by attempting to parse the
    first field of the first non-empty line as a DateTime.
    """
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                first_field = line.split(",", 1)[0].strip().strip('"').strip("'")
                return _match_datetime_format(first_field) is None
    return False


def detect_datetime_format(csv_path: str | Path, has_header: bool) -> str:
    """
    Return the first format from DATETIME_FORMATS that parses the first
    data row's DateTime field.
    """
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        if has_header:
            next(f, None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            first_field = line.split(",", 1)[0].strip().strip('"').strip("'")
            fmt = _match_datetime_format(first_field)
            if fmt is not None:
                return fmt
            break
    raise ValueError(
        f"Could not detect DateTime format in {csv_path}. "
        f"Expected one of: {DATETIME_FORMATS}"
    )


CsvSchema = Literal["spread", "bid_ask"]


def detect_csv_schema(csv_path: str | Path, has_header: bool) -> CsvSchema:
    """
    Return ``"spread"`` for ``DateTime, Spread`` CSVs or ``"bid_ask"`` for
    ``DateTime, Bid, Ask[, Volume]`` CSVs.

    Detection uses the header column names when present, otherwise falls
    back to the field count of the first data row.
    """
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        if has_header:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cols = [c.strip().strip('"').strip("'").lower() for c in line.split(",")]
                if "bid" in cols and "ask" in cols:
                    return "bid_ask"
                if "spread" in cols:
                    return "spread"
                break
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_cols = len(line.split(","))
            return "bid_ask" if n_cols >= 3 else "spread"
    raise ValueError(f"Could not detect CSV schema in {csv_path}")


def csv_to_daily_spread_cache(
    csv_path: str | Path,
    cache_path: str | Path,
    chunksize: int = 2_000_000,
    sort_values: bool = True,
) -> dict[pd.Timestamp, NDArray[np.float32]]:
    """
    Read CSV in chunks and build a pickle cache:
        { Day -> np.array(sorted daily spreads) }

    Accepts CSVs with or without a header row, in either layout:
      - ``DateTime, Spread``
      - ``DateTime, Bid, Ask[, Volume]``  (Spread is computed as Ask - Bid;
        Volume, if present, is ignored)

    This is optimized for later pX(pY) calculations.
    """
    csv_path = Path(csv_path)
    cache_path = Path(cache_path)
    tmp_path = cache_path.with_suffix(".tmp.pkl")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    has_header = csv_has_header(csv_path)
    datetime_format = detect_datetime_format(csv_path, has_header)
    schema = detect_csv_schema(csv_path, has_header)
    tqdm.write(f"Header detected: {has_header}")
    tqdm.write(f"DateTime format: {datetime_format}")
    tqdm.write(f"CSV schema: {schema}")

    tqdm.write("Counting CSV rows...")
    t_count_0 = time.perf_counter()
    total_lines = count_lines(csv_path)
    total_rows = max(total_lines - (1 if has_header else 0), 0)
    total_chunks = (total_rows + chunksize - 1) // chunksize
    t_count_1 = time.perf_counter()

    tqdm.write(f"Rows: {total_rows:,}")
    tqdm.write(f"Chunks: {total_chunks:,}")
    tqdm.write(f"Line count time: {t_count_1 - t_count_0:.2f} s")

    if cache_path.exists():
        tqdm.write(f"Removing existing cache: {cache_path}")
        cache_path.unlink()

    if tmp_path.exists():
        tqdm.write(f"Removing temp cache: {tmp_path}")
        tmp_path.unlink()

    day_data: dict[Any, list[NDArray[np.float32]]] = {}
    rows_read = 0
    rows_kept = 0
    rows_dropped = 0

    t0 = time.perf_counter()

    if schema == "bid_ask":
        usecols = [0, 1, 2]
        names = ["DateTime", "Bid", "Ask"]
    else:
        usecols = [0, 1]
        names = ["DateTime", "Spread"]

    reader = pd.read_csv(
        csv_path,
        usecols=usecols,
        chunksize=chunksize,
        header=0 if has_header else None,
        names=names,
    )

    for raw_chunk in tqdm(reader, total=total_chunks, desc="CSV -> chunk parse", unit="chunk"):
        chunk = cast(pd.DataFrame, raw_chunk)
        rows_read += len(chunk)

        chunk["DateTime"] = pd.to_datetime(
            chunk["DateTime"],
            format=datetime_format,
            errors="coerce",
        )

        if schema == "bid_ask":
            bid = pd.to_numeric(chunk["Bid"], errors="coerce")
            ask = pd.to_numeric(chunk["Ask"], errors="coerce")
            chunk["Spread"] = (ask - bid).astype("float32")
        else:
            chunk["Spread"] = pd.to_numeric(chunk["Spread"], errors="coerce").astype("float32")

        before = len(chunk)
        chunk = chunk.dropna(subset=["DateTime", "Spread"])
        rows_dropped += before - len(chunk)

        if chunk.empty:
            continue

        chunk["Day"] = chunk["DateTime"].dt.floor("D")
        rows_kept += len(chunk)

        for day, group in chunk.groupby("Day", sort=False):
            arr: NDArray[np.float32] = group["Spread"].to_numpy(dtype=np.float32, copy=True)
            day_data.setdefault(day, []).append(arr)

    daily_cache: dict[pd.Timestamp, NDArray[np.float32]] = {}

    for day in tqdm(sorted(day_data.keys()), desc="Merging daily arrays", unit="day"):
        values: NDArray[np.float32] = np.concatenate(day_data[day]).astype(np.float32, copy=False)
        if sort_values:
            values.sort()
        daily_cache[pd.Timestamp(day)] = values

    tqdm.write("Saving pickle cache...")
    with open(tmp_path, "wb") as f:
        pickle.dump(daily_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    tmp_path.replace(cache_path)

    t1 = time.perf_counter()

    tqdm.write(f"Rows read: {rows_read:,}")
    tqdm.write(f"Rows kept: {rows_kept:,}")
    tqdm.write(f"Rows dropped: {rows_dropped:,}")
    tqdm.write(f"Days cached: {len(daily_cache):,}")
    tqdm.write(f"Output file: {cache_path}")
    tqdm.write(f"Build time: {t1 - t0:.2f} s")

    return daily_cache


def load_daily_spread_cache(
    cache_path: str | Path,
) -> dict[pd.Timestamp, NDArray[np.float32]]:
    cache_path = Path(cache_path)
    tqdm.write(f"Loading cache: {cache_path}")
    with open(cache_path, "rb") as f:
        return cast(dict[pd.Timestamp, NDArray[np.float32]], pickle.load(f))


def compute_spread_px_py_from_cache(
    daily_cache: dict[pd.Timestamp, NDArray[np.float32]],
    daily_percentile: float,
    total_percentile: float,
) -> tuple[float, pd.DataFrame]:
    """
    Compute pX(pY) from daily cache.

    Example:
        daily_percentile=90, total_percentile=70 -> p70(p90)
    """
    days: list[pd.Timestamp] = sorted(daily_cache.keys())
    daily_vals: NDArray[np.float32] = np.empty(len(days), dtype=np.float32)

    for i, day in enumerate(tqdm(days, desc=f"Daily p{daily_percentile}", unit="day")):
        values = daily_cache[day]
        daily_vals[i] = np.percentile(values, daily_percentile)

    tqdm.write(f"Computing p{total_percentile}(p{daily_percentile})...")
    final_value = float(np.percentile(daily_vals, total_percentile))

    daily_df = pd.DataFrame({
        "Day": days,
        f"p{daily_percentile}": daily_vals,
    })

    return final_value, daily_df

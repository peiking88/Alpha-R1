"""TDengine direct data loader.

Tries ``taosws`` in-process; if unavailable (not on PyPI), falls back to a
subprocess that runs the query in the system Python where ``taosws`` is
installed.  The subprocess exchange data via ``.npy`` files in a temp dir.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from .schema import code_to_instrument

# Detect whether taosws is importable in this interpreter
try:
    import taosws  # noqa: F401
    _HAS_TAOS = True
except ImportError:
    _HAS_TAOS = False


def _connect():
    import taosws
    from .schema import TDENGINE_URL, TDENGINE_DB
    conn = taosws.connect(TDENGINE_URL)
    conn.query(f"USE {TDENGINE_DB}")
    return conn


def _query(conn, sql: str) -> list[dict]:
    r = conn.query(sql)
    cols = [d.name() for d in r.fields]
    return [dict(zip(cols, row)) for row in r]


def _naive_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Normalize a calendar to tz-naive plain dates.

    TDengine daily bars carry an intraday timestamp (e.g. ``15:00+08:00``);
    downstream date comparisons (decision days, windows) use plain dates.
    """
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_instruments(market: str = "all") -> list[str]:
    if _HAS_TAOS:
        return _load_instruments_proc(market)
    return _load_instruments_sub(market)


def load_calendar(start: str, end: str) -> pd.DatetimeIndex:
    if _HAS_TAOS:
        return _load_calendar_proc(start, end)
    return _load_calendar_sub(start, end)


def load_ohlcv(instruments: list[str], start: str, end: str) -> dict:
    if _HAS_TAOS:
        return _load_ohlcv_proc(instruments, start, end)
    return _load_ohlcv_sub(instruments, start, end)


def load_realtime_bar(instrument: str) -> dict:
    if _HAS_TAOS:
        return _load_realtime_proc(instrument)
    return _load_realtime_sub(instrument)


# ---------------------------------------------------------------------------
# In-process implementation (taosws available)
# ---------------------------------------------------------------------------

def _load_instruments_proc(market: str = "all") -> list[str]:
    conn = _connect()
    try:
        # bj (北交所) excluded: extreme low-liquidity outliers distort factor scores
        rows = _query(conn, "SELECT market, code FROM stock_name WHERE market <> 'bj'")
    finally:
        conn.close()
    return sorted(set(code_to_instrument(r["market"], r["code"]) for r in rows))


def _load_calendar_proc(start: str, end: str) -> pd.DatetimeIndex:
    conn = _connect()
    try:
        rows = _query(
            conn,
            f"SELECT DISTINCT ts FROM kline WHERE cycle='1d' "
            f"AND ts >= '{start}' AND ts < '{end}' ORDER BY ts",
        )
    finally:
        conn.close()
    return _naive_dates(pd.DatetimeIndex([pd.Timestamp(r["ts"]) for r in rows]))


def _load_ohlcv_proc(instruments: list[str], start: str, end: str) -> dict:
    conn = _connect()
    try:
        clauses = " OR ".join(
            f"(market='{i[:2].lower()}' AND code='{i[2:]}')" for i in instruments
        )
        fields_sql = "open, high, low, close, volume, amount"
        sql = (
            f"SELECT ts, market, code, {fields_sql} "
            f"FROM kline WHERE cycle='1d' AND ts >= '{start}' AND ts < '{end}' "
            f"AND ({clauses}) ORDER BY market, code, ts"
        )
        rows = _query(conn, sql)
    finally:
        conn.close()
    return _rows_to_panel(rows, instruments, start, end)


def _load_realtime_proc(instrument: str) -> dict:
    conn = _connect()
    try:
        m, c = instrument[:2].lower(), instrument[2:]
        rows = _query(conn, f"SELECT ts, open, high, low, close, volume, amount "
                           f"FROM k_{m}{c}_1d ORDER BY ts DESC LIMIT 1")
    finally:
        conn.close()
    if not rows:
        return {}
    r = rows[0]
    v = float(r["volume"]); a = float(r["amount"])
    return {
        "timestamp": str(r["ts"])[:10],
        "open": float(r["open"]), "high": float(r["high"]),
        "low": float(r["low"]), "close": float(r["close"]),
        "volume": v, "amount": a, "vwap": a / v if v > 0 else None,
    }


def _rows_to_panel(rows, instruments, start, end):
    if not rows:
        return _empty_panel(instruments, start, end)
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_localize(None)
    df["ts"] = df["ts"].dt.normalize()
    df["instrument"] = df.apply(lambda r: code_to_instrument(r["market"], r["code"]), axis=1)
    calendar = pd.DatetimeIndex(sorted(df["ts"].unique()))
    panels = {}
    for field in ["open", "high", "low", "close", "volume", "amount"]:
        pivot = df.pivot_table(index="ts", columns="instrument", values=field, aggfunc="first")
        pivot = pivot.reindex(index=calendar, columns=instruments)
        panels[field] = pivot.values.astype(np.float32)
    vol = panels["volume"]; amt = panels["amount"]
    with np.errstate(divide="ignore", invalid="ignore"):
        panels["vwap"] = np.where(vol > 0, amt / vol, np.nan).astype(np.float32)
    panels["calendar"] = calendar
    panels["instruments"] = instruments
    return panels


def _empty_panel(instruments, start, end):
    cal = load_calendar(start, end)
    T, N = len(cal), len(instruments)
    return {
        **{f: np.full((T, N), np.nan, dtype=np.float32) for f in ["open", "high", "low", "close", "volume", "amount"]},
        "vwap": np.full((T, N), np.nan, dtype=np.float32),
        "calendar": cal,
        "instruments": instruments,
    }


# ---------------------------------------------------------------------------
# Subprocess implementation (taosws NOT available — delegate to system Python)
# ---------------------------------------------------------------------------

def _load_instruments_sub(market: str = "all") -> list[str]:
    from .tdx_subprocess import load_instruments as _load
    return _load(market)


def _load_calendar_sub(start: str, end: str) -> pd.DatetimeIndex:
    from .tdx_subprocess import load_calendar as _load
    return _load(start, end)


def _load_ohlcv_sub(instruments: list[str], start: str, end: str) -> dict:
    from .tdx_subprocess import load_ohlcv as _load
    return _load(instruments, start, end)


def _load_realtime_sub(instrument: str) -> dict:
    from .tdx_subprocess import load_realtime_bar as _load
    return _load(instrument)

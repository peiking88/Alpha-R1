#!/usr/bin/env python3
"""Fetch daily OHLCV from TDengine and write Alpha-R1 stock_data CSVs.

Modeled on tdx-cpp/scripts/leverage-risk.py (taosws) and the backward-adjustment
logic from peiking88/Kronos/scripts/tdx_export_from_tdengine.py.

Per-instrument tables:
  k_{market}{code}_1d   — raw OHLCV
  a_{market}{code}      — adjust events (fenhong/peigujia/songzhuangu/peigu)

The backward-adjustment factor (后复权) is computed from ``a_`` events so the
most recent bar has factor=1.0 and historical bars are scaled up. Output CSV
carries the factor column for qlib's dump_bin.

Usage:
    python scripts/fetch_stock_data.py                       # full universe, default window
    python scripts/fetch_stock_data.py --start 2020-01-01 --end 2024-12-31
    python scripts/fetch_stock_data.py --categories ashare,etf --limit 50
    python scripts/fetch_stock_data.py --workers 8          # parallel fetch
    python scripts/fetch_stock_data.py --json                # report counts only

Output:
    data/stock_data/{MARKET}{code}.csv   (e.g. SH600000.csv)
    columns: date,open,high,low,close,volume,factor
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import taosws

TDENGINE_URL = os.environ.get("TDENGINE_URL", "taosws://root:taosdata@localhost:6041")
TDENGINE_DB = os.environ.get("TDENGINE_DB", "tdx")
OUTPUT_DIR = os.environ.get(
    "ALPHA_R1_CSV_DIR",
    str(Path(__file__).resolve().parent.parent / "data" / "stock_data"),
)
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2024-12-31"
MIN_BARS = 100  # skip instruments with fewer bars in window


def connect():
    conn = taosws.connect(TDENGINE_URL)
    conn.query(f"USE {TDENGINE_DB}")
    return conn


# ---------------------------------------------------------------------------
# Universe classification
# ---------------------------------------------------------------------------

def classify(market, code, name):
    """Return 'index' | 'ashare' | 'etf' | None."""
    if "ETF" in name or "基金" in name:
        return "etf"
    if market == "sh" and code.startswith("5"):
        return "etf"
    if market == "sz" and code.startswith(("15", "16")):
        return "etf"

    if market == "sh" and code.startswith(("000", "999")):
        return "index"
    if market == "sz" and code.startswith(("399", "395")):
        return "index"
    if code.startswith(("88", "899")):
        return "index"

    if market == "sh" and code.startswith(("60", "68")):
        return "ashare"
    if market == "sz" and code.startswith(("00", "30")):
        return "ashare"
    if market == "bj" and code.startswith(("43", "83", "87", "920")):
        return "ashare"
    return None


def fetch_universe(conn, categories):
    """Return [(market, code, name, category)] filtered to the requested categories."""
    r = conn.query("SELECT market, code, name FROM stock_name")
    cols = [d.name() for d in r.fields]
    rows = [dict(zip(cols, row)) for row in r]
    want = set(categories)
    out = []
    for row in rows:
        cat = classify(row["market"], row["code"], row["name"])
        if cat in want:
            out.append((row["market"], row["code"], row["name"], cat))
    return out


# ---------------------------------------------------------------------------
# Per-instrument fetch + backward adjustment (Kronos logic)
# ---------------------------------------------------------------------------

def query_symbol(symbol, start, end):
    """Fetch OHLCV + adjust events for one symbol. Returns (df_raw, events).

    df_raw: DataFrame indexed by date with open/high/low/close/vol/amt (float64).
    events: list of adjust-event dicts sorted by date ascending.
    """
    conn = connect()
    try:
        r = conn.query(
            f"SELECT ts, open, high, low, close, volume, amount "
            f"FROM k_{symbol}_1d WHERE ts >= '{start}' AND ts < '{end}' ORDER BY ts"
        )
        rows = list(r)
        if len(rows) < MIN_BARS:
            return None, None
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "vol", "amt"])
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
        df = df.set_index("ts").sort_index().astype(np.float64)

        events = []
        try:
            r2 = conn.query(
                f"SELECT ts, fenhong, peigujia, songzhuangu, peigu "
                f"FROM a_{symbol} ORDER BY ts"
            )
            for row in r2:
                ts, fh, pj, sz, pg = row
                fh, pj, sz, pg = float(fh), float(pj), float(sz), float(pg)
                if fh > 0 or sz > 0 or pg > 0:
                    events.append({
                        "date": pd.Timestamp(ts).tz_localize(None),
                        "fenhong": fh,
                        "peigujia": pj,
                        "songzhuangu": sz,
                        "peigu": pg,
                    })
        except Exception:
            pass  # no adjust table -> factor stays 1.0
        return df, events
    finally:
        conn.close()


def compute_back_adjust_factor(df, events):
    """Backward-adjustment factor (后复权), Kronos formula.

    factor[today] = 1.0; for each event going back, factor[:event_idx] *= multiplier.
    multiplier = C*(1+S+P) / (C - D + P*Pp)
    """
    n = len(df)
    factor = np.ones(n, dtype=np.float64)
    if not events:
        return factor
    events_sorted = sorted(events, key=lambda e: e["date"])
    df_dates = df.index.values
    raw_close = df["close"].values
    for evt in events_sorted:
        evt_date = np.datetime64(evt["date"])
        event_idx = int(np.searchsorted(df_dates, evt_date))
        if event_idx >= n or event_idx == 0:
            continue
        C_before = raw_close[event_idx - 1]
        if C_before <= 0:
            continue
        D = evt["fenhong"] / 10.0
        S = evt["songzhuangu"] / 10.0
        P = evt["peigu"] / 10.0
        Pp = evt["peigujia"]
        denominator = C_before - D + P * Pp
        if denominator <= 0:
            continue
        multiplier = C_before * (1.0 + S + P) / denominator
        if abs(multiplier - 1.0) < 1e-12:
            continue
        factor[:event_idx] *= multiplier
    return factor


def process_one(args_tuple):
    """Worker: fetch + adjust + write CSV for one instrument. Returns (symbol, status, n_rows)."""
    market, code, name, cat, start, end, out_dir = args_tuple
    symbol = f"{market}{code}"
    try:
        df, events = query_symbol(symbol, start, end)
        if df is None:
            return symbol, "empty", 0
        factor = compute_back_adjust_factor(df, events)
        df_adj = df.copy()
        for col in ("open", "high", "low", "close"):
            df_adj[col] = df[col].values * factor
        # write CSV
        lines = ["date,open,high,low,close,volume,factor"]
        for ts, row in df_adj.iterrows():
            if None in (row.open, row.high, row.low, row.close, row.vol):
                continue
            lines.append(
                f"{ts.strftime('%Y-%m-%d')},{row.open},{row.high},{row.low},{row.close},{row.vol},{factor[df.index.get_loc(ts)]}"
            )
        path = Path(out_dir) / f"{market.upper()}{code}.csv"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return symbol, "ok", len(df_adj)
    except Exception as e:
        return symbol, f"error: {e}", 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Fetch daily OHLCV + 后复权 from TDengine → stock_data CSV")
    ap.add_argument("--start", default=DEFAULT_START, help="start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", default=DEFAULT_END, help="end date YYYY-MM-DD (exclusive)")
    ap.add_argument("--categories", default="index,ashare,etf",
                    help="comma-separated: index,ashare,etf (default: all three)")
    ap.add_argument("--output-dir", default=OUTPUT_DIR, help="CSV output directory")
    ap.add_argument("--workers", type=int, default=4, help="parallel fetch threads (default: 4)")
    ap.add_argument("--limit", type=int, metavar="N", help="only fetch the first N instruments (debug)")
    ap.add_argument("--json", action="store_true", help="print universe counts as JSON and exit")
    args = ap.parse_args()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    conn = connect()
    print(f"[fetch] universe from {TDENGINE_DB}.stock_name ...")
    universe = fetch_universe(conn, categories)
    conn.close()
    print(f"[fetch] {len(universe)} instruments match {categories}")

    if args.json:
        import json
        counts = {c: sum(1 for _, _, _, cat in universe if cat == c) for c in categories}
        print(json.dumps({"total": len(universe), "by_category": counts}, ensure_ascii=False))
        return

    if args.limit:
        universe = universe[: args.limit]
        print(f"[fetch] --limit {args.limit}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(m, c, n, cat, args.start, args.end, str(out_dir)) for m, c, n, cat in universe]

    written = 0
    empty = 0
    errors = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            sym, status, n_rows = future.result()
            if status == "ok":
                written += 1
            elif status == "empty":
                empty += 1
            else:
                errors += 1
                if errors <= 5:
                    print(f"  [error] {sym}: {status}")
            if i % 200 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  written={written} empty={empty} errors={errors}")

    dt = time.time() - t0
    print(f"[fetch] done: {written} CSVs in {out_dir}, {empty} empty, {errors} errors, {dt:.1f}s")
    print(f"[fetch] workers={args.workers}, window=[{args.start}, {args.end})")


if __name__ == "__main__":
    main()

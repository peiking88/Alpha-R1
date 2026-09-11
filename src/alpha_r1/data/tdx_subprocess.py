"""TDengine data loader via subprocess (system Python with taosws).

The system Python has ``taosws`` installed but it is not on PyPI, so we run
the queries in a subprocess and exchange data via numpy ``.npy`` files.
This keeps the quantaalpha env (torch/transformers) clean of TDengine deps.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Path to the system Python interpreter that has taosws
_SYS_PYTHON = "/usr/bin/python3"
# Path to the helper script (created at the bottom of this module)
_HELPER = Path(__file__).parent / "_tdx_query.py"


def _ensure_helper():
    """Write the helper script if it doesn't exist."""
    if _HELPER.exists():
        return
    _HELPER.write_text(HELPER_SRC)


def _run_helper(args: dict) -> dict:
    """Run the helper with JSON args, return JSON result."""
    _ensure_helper()
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(args, f)
        argfile = f.name
    try:
        r = subprocess.run(
            [_SYS_PYTHON, str(_HELPER), argfile],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            raise RuntimeError(f"tdx helper failed: {r.stderr[-500:]}")
        return json.loads(r.stdout)
    finally:
        Path(argfile).unlink(missing_ok=True)


def load_instruments(market: str = "all") -> list[str]:
    d = _run_helper({"cmd": "instruments", "market": market})
    return d["instruments"]


def load_calendar(start: str, end: str) -> pd.DatetimeIndex:
    d = _run_helper({"cmd": "calendar", "start": start, "end": end})
    return pd.DatetimeIndex(d["calendar"])


def load_ohlcv(instruments: list[str], start: str, end: str) -> dict:
    """Load OHLCV panels via subprocess.  Returns [T, N] float32 arrays."""
    with tempfile.TemporaryDirectory() as tmpdir:
        arg = {"cmd": "ohlcv", "instruments": instruments, "start": start, "end": end, "outdir": tmpdir}
        _run_helper(arg)
        # Load arrays written by the helper
        panels = {}
        for field in ["open", "high", "low", "close", "volume", "amount", "vwap"]:
            p = Path(tmpdir) / f"{field}.npy"
            if p.exists():
                panels[field] = np.load(str(p))
        cal_path = Path(tmpdir) / "calendar.json"
        panels["calendar"] = pd.DatetimeIndex(json.loads(cal_path.read_text()))
        panels["instruments"] = instruments
    return panels


def load_realtime_bar(instrument: str) -> dict:
    return _run_helper({"cmd": "realtime", "instrument": instrument})


# ---------------------------------------------------------------------------
# Helper script source (runs in system Python with taosws)
# ---------------------------------------------------------------------------
HELPER_SRC = '''#!/usr/bin/env python3
"""Helper: run TDengine queries, write results to disk."""
import json
import sys
import numpy as np
import pandas as pd
import taosws

TDENGINE_URL = "taosws://root:taosdata@localhost:6041"
TDENGINE_DB = "tdx"

def connect():
    conn = taosws.connect(TDENGINE_URL)
    conn.query(f"USE {TDENGINE_DB}")
    return conn

def query_all(conn, sql):
    r = conn.query(sql)
    cols = [d.name() for d in r.fields]
    return [dict(zip(cols, row)) for row in r]

def main():
    args = json.load(open(sys.argv[1]))
    cmd = args["cmd"]
    conn = connect()

    if cmd == "instruments":
        rows = query_all(conn, "SELECT market, code FROM stock_name")
        inst = sorted(set(f"{r['market'].upper()}{r['code']}" for r in rows))
        print(json.dumps({"instruments": inst}))

    elif cmd == "calendar":
        rows = query_all(conn, f"SELECT DISTINCT ts FROM kline WHERE cycle=\\'1d\\' AND ts >= \\'{args['start']}\\' AND ts < \\'{args['end']}\\' ORDER BY ts")
        cal = [str(r["ts"])[:10] for r in rows]
        print(json.dumps({"calendar": cal}))

    elif cmd == "ohlcv":
        instruments = args["instruments"]
        start, end = args["start"], args["end"]
        clauses = " OR ".join(f"(market='{i[:2].lower()}' AND code='{i[2:]}')" for i in instruments)
        fields = "open,high,low,close,volume,amount"
        sql = f"SELECT ts,market,code,{fields} FROM kline WHERE cycle='1d' AND ts >= '{start}' AND ts < '{end}' AND ({clauses}) ORDER BY market,code,ts"
        rows = query_all(conn, sql)
        outdir = args["outdir"]

        if not rows:
            # Write empty arrays
            for f in ["open","high","low","close","volume","amount","vwap"]:
                np.save(f"{outdir}/{f}.npy", np.zeros((0, len(instruments)), dtype=np.float32))
            json.dump([], open(f"{outdir}/calendar.json", "w"))
            print(json.dumps({"status": "ok", "rows": 0}))
            return

        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"])
        df["instrument"] = df.apply(lambda r: f"{r['market'].upper()}{r['code']}", axis=1)
        calendar = pd.DatetimeIndex(sorted(df["ts"].unique()))
        json.dump([str(d)[:10] for d in calendar], open(f"{outdir}/calendar.json", "w"))

        for field in ["open","high","low","close","volume","amount"]:
            pivot = df.pivot_table(index="ts", columns="instrument", values=field, aggfunc="first")
            pivot = pivot.reindex(index=calendar, columns=instruments)
            np.save(f"{outdir}/{field}.npy", pivot.values.astype(np.float32))

        # vwap = amount / volume
        vol = df.pivot_table(index="ts", columns="instrument", values="volume", aggfunc="first").reindex(index=calendar, columns=instruments).values.astype(np.float32)
        amt = df.pivot_table(index="ts", columns="instrument", values="amount", aggfunc="first").reindex(index=calendar, columns=instruments).values.astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            vwap = np.where(vol > 0, amt / vol, np.nan).astype(np.float32)
        np.save(f"{outdir}/vwap.npy", vwap)
        print(json.dumps({"status": "ok", "rows": len(rows)}))

    elif cmd == "realtime":
        inst = args["instrument"]
        m, c = inst[:2].lower(), inst[2:]
        rows = query_all(conn, f"SELECT ts,open,high,low,close,volume,amount FROM k_{m}{c}_1d ORDER BY ts DESC LIMIT 1")
        if rows:
            r = rows[0]
            v = float(r["volume"]); a = float(r["amount"])
            print(json.dumps({"timestamp": str(r["ts"])[:10], "open": float(r["open"]), "high": float(r["high"]), "low": float(r["low"]), "close": float(r["close"]), "volume": v, "amount": a, "vwap": a/v if v > 0 else None}))
        else:
            print(json.dumps({}))

    conn.close()

if __name__ == "__main__":
    main()
'''

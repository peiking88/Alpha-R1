#!/usr/bin/env python3
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
        rows = query_all(conn, f"SELECT DISTINCT ts FROM kline WHERE cycle=\'1d\' AND ts >= \'{args['start']}\' AND ts < \'{args['end']}\' ORDER BY ts")
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

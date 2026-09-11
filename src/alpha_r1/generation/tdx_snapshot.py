"""TDengine index snapshot fetcher — extracted from market-analysis.py.

Fetches daily index OHLCV from the ``kline`` supertable and renders a compact
market snapshot text, suitable as input to an LLM market-memory prompt.
"""

from datetime import date, timedelta

# Major indices: code → (display name, market tag).
_INDEX_INFO = {
    "000001": ("上证指数", "sh"),
    "399001": ("深证成指", "sz"),
    "399006": ("创业板指", "sz"),
    "000688": ("科创50", "sh"),
    "000300": ("沪深300", "sh"),
    "000905": ("中证500", "sh"),
    "000852": ("中证1000", "sh"),
}


def fetch_index_snapshot(conn, day=None):
    """Return [(name, close, chg, pct, open, high, low, amount_yi)] for the trading day.

    ``conn`` is an active taosws connection with the database already USEd.
    ``day`` is a ``date`` (default: today). Uses the previous close to compute
    the true change. Days with no data return an empty list.
    """
    day = day or date.today()
    yesterday = day - timedelta(days=1)
    next_day = day + timedelta(days=1)
    day_s = day.strftime("%Y-%m-%d")
    yday_s = yesterday.strftime("%Y-%m-%d")
    next_s = next_day.strftime("%Y-%m-%d")
    conditions = " OR ".join(
        f"(code='{c}' AND market='{m}')" for c, (_, m) in _INDEX_INFO.items()
    )
    sql = (
        f"SELECT ts, open, high, low, close, amount, code "
        f"FROM kline WHERE cycle='1d' AND ts >= '{yday_s}' AND ts < '{next_s}' "
        f"AND ({conditions}) ORDER BY code, ts"
    )
    r = conn.query(sql)
    cols = [d.name() for d in r.fields]
    rows = [dict(zip(cols, row)) for row in r]

    by_code = {}
    for row in rows:
        by_code.setdefault(row["code"], []).append(row)

    out = []
    for code, entries in by_code.items():
        name, _ = _INDEX_INFO.get(code, (code, ""))
        entries.sort(key=lambda x: x["ts"])
        today_entry = entries[-1]
        pre_close = float(entries[-2]["close"]) if len(entries) >= 2 else None
        try:
            c = float(today_entry["close"])
            chg = c - pre_close if pre_close else 0.0
            pct = (chg / pre_close * 100) if pre_close else 0.0
            amt_yi = float(today_entry["amount"]) / 1e8
            out.append((name, c, chg, pct, float(today_entry["open"]),
                        float(today_entry["high"]), float(today_entry["low"]), amt_yi))
        except (ValueError, ZeroDivisionError, TypeError):
            continue
    return out


def format_snapshot(snapshot):
    """Render index snapshot rows as a compact text block."""
    if not snapshot:
        return "(no index data)"
    lines = []
    for name, c, chg, pct, op, hi, lo, amt in snapshot:
        lines.append(
            f"{name} {c:.2f}（{chg:+.2f}, {pct:+.2f}%）, "
            f"今开 {op:.2f}, 最高 {hi:.2f}, 最低 {lo:.2f}, "
            f"成交额 {amt:.2f} 亿"
        )
    return "\n".join(lines)

#!/usr/bin/env python3
"""Generate daily market-memory input files from TDengine index snapshots.

Adapts the index-snapshot logic from tdx-cpp/scripts/market-analysis.py into
the ``data/price_market/`` / ``data/news/`` txt format that
``build_market_memory.py`` consumes. This replaces manual txt preparation with
an automated pull from real market data.

Output (per trading day):
    data/price_market/YYYY-MM-DD_23:59:59.txt   (index snapshot)
    data/news/YYYY-MM-DD.txt                     (placeholder / web news, if --with-news)

Usage:
    python scripts/gen_market_memory_input.py                       # default window
    python scripts/gen_market_memory_input.py --start 2023-01-01 --end 2024-12-31
    python scripts/gen_market_memory_input.py --with-news            # also scrape web news
"""

import argparse
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import taosws

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_r1.generation.tdx_snapshot import fetch_index_snapshot, format_snapshot

try:
    from alpha_r1.generation.cls_news import fetch_telegraph, format_news
    _HAS_CLS = True
except ImportError:
    _HAS_CLS = False

TDENGINE_URL = os.environ.get("TDENGINE_URL", "taosws://root:taosdata@localhost:6041")
TDENGINE_DB = os.environ.get("TDENGINE_DB", "tdx")
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2024-12-31"


def connect():
    conn = taosws.connect(TDENGINE_URL)
    conn.query(f"USE {TDENGINE_DB}")
    return conn


def trading_days(start, end):
    """List dates for weekdays in [start, end)."""
    days = []
    d = start
    while d < end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def write_price_market(path, snapshot_text, day):
    header = f"日期: {day}\n主要指数行情（收盘）：\n"
    Path(path).write_text(header + snapshot_text + "\n", encoding="utf-8")


def write_news_placeholder(path, day):
    """Without --with-news this is a stub noting no news."""
    Path(path).write_text(f"日期: {day}\n（无新闻数据，仅基于指数行情生成市场记忆）\n",
                         encoding="utf-8")


def write_news(path, day, items):
    """Write real 财联社 telegraph news."""
    Path(path).write_text(format_news(day, items) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Generate market-memory input txt from TDengine")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--price-dir", default="data/price_market")
    ap.add_argument("--news-dir", default="data/news")
    ap.add_argument("--with-news", action="store_true",
                    help="scrape 财联社 telegraph news (requires playwright/chromium)")
    args = ap.parse_args()

    if args.with_news and not _HAS_CLS:
        sys.exit("cls_news requires playwright: pip install playwright")

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    days = trading_days(start, end)
    print(f"[gen] {len(days)} trading days in [{args.start}, {args.end})")

    price_dir = Path(args.price_dir)
    news_dir = Path(args.news_dir)
    price_dir.mkdir(parents=True, exist_ok=True)
    news_dir.mkdir(parents=True, exist_ok=True)

    conn = connect()
    # Fetch real news once (财联社 telegraph only serves the current day).
    news_items = []
    if args.with_news:
        print("[gen] fetching 财联社 telegraph news ...")
        news_items = fetch_telegraph(max_items=100)
        print(f"[gen] got {len(news_items)} telegraph items")
    written = 0
    missing = 0
    t0 = time.time()
    for day in days:
        ds = day.strftime("%Y-%m-%d")
        snapshot = fetch_index_snapshot(conn, day)
        snap_text = format_snapshot(snapshot)
        if snapshot:
            write_price_market(price_dir / f"{ds}_23:59:59.txt", snap_text, ds)
            written += 1
        else:
            missing += 1
        if args.with_news and news_items:
            write_news(news_dir / f"{ds}.txt", ds, news_items)
        else:
            write_news_placeholder(news_dir / f"{ds}.txt", ds)

    dt = time.time() - t0
    print(f"[gen] done: {written} days with index data, {missing} without, {dt:.1f}s")
    print(f"[gen] price_market -> {price_dir}")
    print(f"[gen] news         -> {news_dir}")
    print("[gen] next: python scripts/build_market_memory.py")


if __name__ == "__main__":
    main()

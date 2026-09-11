#!/usr/bin/env python3
"""TDengine direct + GPU backtest CLI.

Usage:
    python scripts/run_realtime_backtest.py --zxg --alphas 1,2,3
    python scripts/run_realtime_backtest.py --zxg --device cuda
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_r1.backtest.realtime_backtest import run_factor_backtest
from alpha_r1.backtest.universe import load_zxg
from alpha_r1.factors.alpha101 import parse_alpha_spec


def main():
    parser = argparse.ArgumentParser(description="TDengine direct + GPU backtest")
    parser.add_argument("--alphas", default="all",
                        help="factor subset: 'all', '001-101' or '001,005' (default: all)")
    parser.add_argument("--config", default="configs/backtest.yaml")
    parser.add_argument("--zxg", nargs="?", const="/home/li/.local/share/tdxcfv/drive_c/tc/T0002/blocknew/zxg.blk",
                        help="use 自选股 universe from zxg.blk")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())

    custom = None
    if args.zxg:
        zxg = load_zxg(args.zxg)
        # Intersect with instruments that have data in TDengine
        from alpha_r1.data import load_instruments
        available = set(load_instruments())
        custom = [c for c in zxg if c in available]
        print(f"[backtest] zxg universe: {len(zxg)} in blk, {len(custom)} available in TDengine")
        config["market"] = "custom"

    names = parse_alpha_spec(args.alphas)
    print(f"[backtest] backtesting {len(names)} factors on {args.device} "
          f"({config['start_date']}..{config['end_date']})")
    run_factor_backtest(names, config, custom_instruments=custom, device=args.device)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""GPU-accelerated Alpha101 factor backtest (PyTorch CUDA).

Usage:
    python scripts/run_torch_backtest.py                       # all instruments, GPU
    python scripts/run_torch_backtest.py --zxg                 # 自选股 zxg.blk
    python scripts/run_torch_backtest.py --zxg --alphas 1,2,3  # subset
    python scripts/run_torch_backtest.py --device cpu          # force CPU
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_r1.backtest.torch_backtest import backtest_factors_torch
from alpha_r1.backtest.universe import load_zxg
from alpha_r1.factors.alpha101 import parse_alpha_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", default="all",
                        help="factor subset: 'all', '001-101' or '001,005' (default: all)")
    parser.add_argument("--config", default="configs/backtest.yaml")
    parser.add_argument("--zxg", nargs="?", const="/home/li/.local/share/tdxcfv/drive_c/tc/T0002/blocknew/zxg.blk",
                        help="use 自选股 universe from zxg.blk")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="torch device (default: cuda)")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())

    custom = None
    if args.zxg:
        zxg = load_zxg(args.zxg)
        all_txt = Path(config["qlib_data_dir"]).expanduser() / "instruments" / "all.txt"
        available = {line.split("\t")[0] for line in all_txt.read_text().splitlines() if line.strip()}
        custom = [c for c in zxg if c in available]
        print(f"[torch] zxg universe: {len(zxg)} in blk, {len(custom)} available in qlib data")
        config["market"] = "custom"

    names = parse_alpha_spec(args.alphas)
    print(f"[torch] backtesting {len(names)} factors on {args.device} "
          f"({config['start_date']}..{config['end_date']})")
    backtest_factors_torch(names, config, custom_instruments=custom, device=args.device)


if __name__ == "__main__":
    main()

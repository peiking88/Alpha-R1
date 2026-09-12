#!/usr/bin/env python3
"""Estimate the fixed linear return model (paper Appendix F).

OLS of forward H-day returns on cross-sectionally standardized Alpha101
factor values over the historical window, written as betas.csv (consumed by
scripts/run_strategy_backtest.py and training/reward.py).

Data comes straight from TDengine via ``alpha_r1.data`` (no qlib / CSV stage);
factors are computed on the GPU with ``torch_factors.compute_factors``.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_r1.data import load_ohlcv, load_instruments
from alpha_r1.backtest.torch_factors import compute_factors
from alpha_r1.backtest.linear_model import estimate_betas, save_betas
from alpha_r1.factors.alpha101 import parse_alpha_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/strategy.yaml")
    parser.add_argument("--alphas", default="all",
                        help="factor subset: 'all', '001-101' or '001,005' (default: all)")
    parser.add_argument("--start-date", default=None,
                        help="estimation window start (default: config train_start_date)")
    parser.add_argument("--end-date", default=None,
                        help="estimation window end (default: config train_end_date)")
    parser.add_argument("--output", default="result/linear_model/betas.csv")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    start = args.start_date or config["train_start_date"]
    end = args.end_date or config["train_end_date"]
    holding = config.get("holding_days", 5)

    names = parse_alpha_spec(args.alphas)
    instruments = load_instruments(config.get("market", "all"))

    print(f"[linear-model] loading OHLCV for {len(instruments)} instruments "
          f"({start}..{end}) ...")
    panels = load_ohlcv(instruments, start, end)
    calendar = panels["calendar"]
    cols = panels["instruments"]
    print(f"[linear-model] data shape: {panels['close'].shape} (T x N), "
          f"{len(calendar)} days")

    F = compute_factors(
        torch.as_tensor(panels["close"], device=args.device),
        torch.as_tensor(panels["high"], device=args.device),
        torch.as_tensor(panels["low"], device=args.device),
        torch.as_tensor(panels["open"], device=args.device),
        torch.as_tensor(panels["volume"], device=args.device),
        torch.as_tensor(panels["vwap"], device=args.device),
        device=args.device,
    )

    factor_panels = {}
    for n in names:
        if n not in F:
            print(f"[linear-model] WARN {n}: not computable, skipped")
            continue
        factor_panels[n] = pd.DataFrame(
            F[n].float().cpu().numpy(), index=calendar, columns=cols)

    # Forward H-day returns: close[t+H] / close[t] - 1 (same as qlib Ref($close,-H))
    close_df = pd.DataFrame(panels["close"], index=calendar, columns=cols)
    fwd = close_df.shift(-holding) / close_df - 1

    betas, intercept = estimate_betas(factor_panels, fwd)
    save_betas(betas, intercept, args.output)
    print(f"[linear-model] {len(factor_panels)} betas (intercept={intercept:.6f}) -> {args.output}")
    top = betas.abs().nlargest(5).index
    for n in top:
        print(f"  {n}: {betas[n]:+.6f}")


if __name__ == "__main__":
    main()

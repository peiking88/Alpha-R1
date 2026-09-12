#!/usr/bin/env python3
"""Run the end-to-end strategy backtest on parsed Alpha-R1 selections.

Input: selections.json (date -> factor list) from scripts/parse_outputs.py,
plus betas.csv from scripts/train_linear_model.py. Output: metrics JSON and
daily NAV CSV under ``configs/strategy.yaml: output_dir``.

Passing a directory as --selections runs every *.json in it as one round and
additionally writes an "average" result computed on the day-by-day arithmetic
mean of the rounds' daily returns (the paper's multi-round aggregation).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_r1.backtest import performance_metrics, run_strategy, save_strategy_result


def _load_selections(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", required=True,
                        help="selections.json file, or a directory of per-round files")
    parser.add_argument("--betas", required=True,
                        help="betas.csv from scripts/train_linear_model.py")
    parser.add_argument("--config", default="configs/strategy.yaml")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="torch device for factor computation (default: cuda; "
                             "use cpu when the full universe exceeds GPU memory)")
    parser.add_argument("--output-dir", default=None,
                        help="override configs/strategy.yaml output_dir")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    config["device"] = args.device
    output_dir = Path(args.output_dir or config.get("output_dir", "result/strategy_backtest"))
    rf = config.get("risk_free_rate", 0.0)

    src = Path(args.selections)
    files = sorted(src.glob("*.json")) if src.is_dir() else [src]
    if not files:
        raise FileNotFoundError(f"no selections json found at {src}")

    daily_returns = []
    for f in files:
        result = run_strategy(_load_selections(f), args.betas, config)
        name = f.stem if src.is_dir() else "backtest_result"
        save_strategy_result(result, output_dir / ("rounds" if src.is_dir() else ""), name)
        daily = pd.DataFrame(result["daily"])
        daily_returns.append(daily.set_index(pd.to_datetime(daily["date"]))["return"])

    if len(daily_returns) > 1:
        avg = pd.concat(daily_returns, axis=1).mean(axis=1)
        metrics = performance_metrics(avg, rf)
        save_strategy_result({"metrics": metrics,
                              "daily": [{"date": str(d.date()), "return": float(r)}
                                        for d, r in avg.items()]},
                             output_dir, "average")
        print(f"[strategy] {len(files)} rounds averaged -> {output_dir / 'average.json'}")


if __name__ == "__main__":
    main()

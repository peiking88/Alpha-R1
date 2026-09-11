"""Single-factor backtesting on qlib (paper Section 3.1.3).

Each Alpha101 factor is backtested independently over the historical window:
factor values are computed with the qlib expression engine, daily IC / RankIC
against forward returns are measured, and a daily-rebalanced top-k equal-weight
long portfolio is simulated. The resulting performance vector P_i (returns,
volatility, IC statistics, yearly breakdown for decay analysis) is saved as
``result/alpha_backtest/alphaNNN.json`` and feeds factor description
generation (paper Section 3.2.1).
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..factors.alpha101 import ALPHA101
from .alpha101_qlib import QLIB_EXPRESSIONS, custom_ops_config, set_cs_universe, clear_panel_cache
from .data import init_qlib


def _json_safe(obj):
    """Recursively replace non-finite floats with None for strict JSON output."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _load_instruments(qlib_dir: str, custom: list[str] | None = None) -> list[str]:
    """Read instrument list from qlib's instruments/all.txt or a custom list.

    ``D.instruments()`` returns a dict in newer qlib, so we read directly.
    """
    if custom:
        return list(custom)
    all_txt = Path(qlib_dir).expanduser() / "instruments" / "all.txt"
    instruments = []
    for line in all_txt.read_text().splitlines():
        if line.strip():
            instruments.append(line.split("\t")[0])
    return instruments


def _load_batch(D, instruments, expressions, start, end):
    """Load a batch of expressions in one shot. Returns DataFrame or raises."""
    df = D.features(instruments, expressions, start_time=start, end_time=end)
    return df


def _load_panels(names: list[str], qlib_dir: str, start: str, end: str,
                 ic_horizon: int, batch_size: int = 10,
                 custom_instruments: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load factor panels and forward returns as datetime x instrument frames.

    Loads factors in batches for speed; if a batch fails (sparse instrument
    crashing a rolling op), retries each factor in that batch individually.
    """
    from qlib.data import D

    expressions = [QLIB_EXPRESSIONS[n] for n in names]
    fwd_expr = f"Ref($close, -{ic_horizon}) / $close - 1"
    instruments = _load_instruments(qlib_dir, custom=custom_instruments)

    # Filter to instruments with full-window coverage: some listings are
    # sparse (newly listed, suspended) and crash qlib's rolling ops with
    # empty-series broadcast errors.
    if custom_instruments is None:
        close_df = D.features(instruments, ["$close"], start_time=start, end_time=end)
        counts = close_df.iloc[:, 0].groupby(level="instrument").count()
        threshold = max(252, int(counts.quantile(0.2)))
        valid = counts[counts >= threshold].index.tolist()
        if len(valid) < len(instruments):
            print(f"[backtest] filtered instruments {len(instruments)} -> {len(valid)} "
                  f"(sparse-data drop, threshold={threshold} days)")
            instruments = valid

    # Batch load with per-factor fallback.
    frames: dict[str, pd.DataFrame] = {}
    for i in range(0, len(names), batch_size):
        batch_names = names[i:i + batch_size]
        batch_exprs = [QLIB_EXPRESSIONS[n] for n in batch_names]
        try:
            df = _load_batch(D, instruments, batch_exprs, start, end)
            for name, expr in zip(batch_names, batch_exprs):
                frames[name] = df[expr].unstack(level="instrument")
        except Exception:
            # Batch failed — retry each factor individually to isolate the culprit.
            for name, expr in zip(batch_names, batch_exprs):
                try:
                    df = _load_batch(D, instruments, [expr], start, end)
                    frames[name] = df.iloc[:, 0].unstack(level="instrument")
                except Exception as e:
                    print(f"[backtest] SKIP {name}: {e}")

    fwd_df = D.features(instruments, [fwd_expr], start_time=start, end_time=end)
    fwd_panel = fwd_df.iloc[:, 0].unstack(level="instrument")
    return frames, fwd_panel


def _ic_stats(factor: pd.DataFrame, fwd: pd.DataFrame) -> dict:
    """Daily cross-sectional IC / RankIC between factor and forward returns."""
    factor, fwd = factor.align(fwd, join="inner")
    ics, rank_ics = [], []
    for date in factor.index:
        f = factor.loc[date]
        r = fwd.loc[date]
        valid = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        if valid.sum() < 10:
            continue
        fv, rv = f[valid], r[valid]
        if fv.std() == 0 or rv.std() == 0:
            continue
        ics.append(float(np.corrcoef(fv, rv)[0, 1]))
        rank_ics.append(float(fv.corr(rv, method="spearman")))
    ic = pd.Series(ics, dtype=float)
    ic = ic[np.isfinite(ic)]
    ric = pd.Series(rank_ics, dtype=float)
    ric = ric[np.isfinite(ric)]

    def _ir(series: pd.Series) -> float | None:
        if len(series) == 0:
            return None
        std = series.std()
        if not np.isfinite(std) or std == 0:
            return None
        return float(series.mean() / std)

    return {
        "mean_ic": float(ic.mean()) if len(ic) else None,
        "ic_std": float(ic.std()) if len(ic) else None,
        "ic_ir": _ir(ic),
        "ic_win_rate": float((ic > 0).mean()) if len(ic) else None,
        "mean_rank_ic": float(ric.mean()) if len(ric) else None,
        "rank_ic_std": float(ric.std()) if len(ric) else None,
        "rank_ic_ir": _ir(ric),
        "rank_ic_win_rate": float((ric > 0).mean()) if len(ric) else None,
        "ic_days": int(len(ic)),
    }


def _benchmark_stats(portfolio: dict, benchmark_ret: pd.Series) -> None:
    """Add benchmark and excess-return statistics to a portfolio result."""
    rets_index = pd.to_datetime(list(portfolio["nav_series"].keys()))
    bench = benchmark_ret.reindex(rets_index).dropna()
    if len(bench) == 0:
        return
    bench_nav = float((1 + bench).prod())
    bench_ann = bench_nav ** (252 / len(bench)) - 1
    portfolio["benchmark_annual_return"] = float(bench_ann)
    if portfolio.get("annual_return") is not None:
        portfolio["excess_annual_return"] = float(portfolio["annual_return"] - bench_ann)


def _load_benchmark_ret(benchmark: str, start: str, end: str) -> pd.Series | None:
    """Daily returns of the benchmark instrument, or None if unavailable."""
    from qlib.data import D

    try:
        df = D.features([benchmark], ["$close / Ref($close, 1) - 1"],
                        start_time=start, end_time=end)
    except Exception as e:
        print(f"[backtest] benchmark {benchmark} unavailable: {e}")
        return None
    if df.empty:
        return None
    return df.iloc[:, 0].droplevel("instrument")


def _topk_backtest(factor: pd.DataFrame, fwd: pd.DataFrame, topk: int, fee: float) -> dict:
    """Daily-rebalanced top-k equal-weight long portfolio from factor signal."""
    factor, fwd = factor.align(fwd, join="inner")
    daily_ret, daily_turnover = [], []
    prev_holdings: set = set()
    for date in factor.index:
        f = factor.loc[date].dropna()
        f = f[np.isfinite(f)]
        if len(f) < topk:
            # Days with too few valid names are skipped (no position taken);
            # skipped days are simply absent from the return series.
            prev_holdings = set()
            continue
        holdings = set(f.nlargest(topk).index)
        r = fwd.loc[date, list(holdings)].dropna()
        if len(r) == 0:
            prev_holdings = holdings
            continue
        turnover = 1.0 if not prev_holdings else 1 - len(holdings & prev_holdings) / topk
        ret = float(r.mean()) - fee * turnover * 2  # sell old + buy new
        daily_ret.append((str(date.date()), ret))
        daily_turnover.append(turnover)
        prev_holdings = holdings

    if not daily_ret:
        return {}

    rets = pd.Series(dict(daily_ret)).sort_index()
    nav = (1 + rets).cumprod()
    ann_ret = float(nav.iloc[-1] ** (252 / len(rets)) - 1)
    ann_vol = float(rets.std() * math.sqrt(252))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else None
    drawdown = (nav / nav.cummax() - 1).min()

    yearly = {}
    for year, group in rets.groupby(pd.to_datetime(rets.index).year):
        y_nav = float((1 + group).prod())
        yearly[str(year)] = {
            "return": y_nav - 1,
            "sharpe": float(group.mean() / group.std() * math.sqrt(252)) if group.std() > 0 else None,
            "days": int(len(group)),
        }

    return {
        "total_return": float(nav.iloc[-1] - 1),
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": float(drawdown),
        "avg_daily_turnover": float(np.mean(daily_turnover)) if daily_turnover else None,
        "total_trading_days": int(len(rets)),
        "yearly_breakdown": yearly,
        "nav_series": {d: float(v) for d, v in nav.items()},
    }


def backtest_factors(names: list[str], config: dict) -> dict[str, dict]:
    """Run single-factor backtests and persist one JSON per factor.

    Args:
        names: factor names (e.g. ``["alpha001", ...]``).
        config: parsed ``configs/backtest.yaml``.

    Returns:
        Mapping of factor name -> performance vector P_i.
    """
    names = [n.lower() for n in names]
    invalid = [n for n in names if n not in ALPHA101]
    if invalid:
        raise ValueError(f"unknown factor names: {invalid}")

    set_cs_universe(config["market"])
    init_qlib(config["qlib_data_dir"], custom_ops=custom_ops_config())

    clear_panel_cache()
    frames, fwd_panel = _load_panels(
        names, config["qlib_data_dir"], config["start_date"], config["end_date"],
        config.get("ic_horizon", 1),
        batch_size=config.get("batch_size", 10),
        custom_instruments=config.get("custom_instruments"),
    )
    benchmark_ret = None
    if config.get("benchmark"):
        benchmark_ret = _load_benchmark_ret(
            config["benchmark"], config["start_date"], config["end_date"])

    output_dir = Path(config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for name in names:
        try:
            portfolio = _topk_backtest(frames[name], fwd_panel,
                                       config.get("topk", 50), config.get("fee", 0.001))
            if portfolio and benchmark_ret is not None:
                _benchmark_stats(portfolio, benchmark_ret)
            p_i = {
                "factor": name,
                "formula": ALPHA101[name],
                "qlib_expression": QLIB_EXPRESSIONS[name],
                "market": config["market"],
                "window": {"start": config["start_date"], "end": config["end_date"]},
                "topk": config.get("topk", 50),
                "fee": config.get("fee", 0.001),
                "ic": _ic_stats(frames[name], fwd_panel),
                "portfolio": portfolio,
            }
        except Exception as e:
            print(f"[backtest] {name} FAILED: {e}")
            continue
        out_path = output_dir / f"{name}.json"
        out_path.write_text(json.dumps(_json_safe(p_i), ensure_ascii=False, indent=2))
        print(f"[backtest] {name} -> {out_path}")
        results[name] = p_i
    return results

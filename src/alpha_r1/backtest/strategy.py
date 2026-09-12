"""End-to-end strategy backtest (paper Section 3.3 and Appendix F).

Turns parsed Alpha-R1 selections (date -> factor list) into a tradable
long-only portfolio following the paper's execution protocol:

- fixed linear model scores each stock; the top N form an equal-weighted
  long-only portfolio;
- slot rotation: capital is split into H slots and only slot ``t mod H`` is
  rebalanced on day t (daily turnover 1/H);
- fills are approximated with daily VWAP ($vwap, falling back to $close);
- 10 bps fees on both sides; unfilled cash earns the risk-free rate;
- optional limit-lock filter (daily-bar approximation, off by default).

Decision day t uses factor values of the previous trading day t-1.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .linear_model import load_betas, score_stocks

TRADING_DAYS_PER_YEAR = 252


class SlotBacktest:
    """Slot-rotation portfolio engine operating on precomputed panels."""

    def __init__(self, holding_days: int = 5, fee: float = 0.001,
                 rf_annual: float = 0.0, reject_limit_locked: bool = False,
                 limit_threshold: float = 0.098):
        self.H = holding_days
        self.fee = fee
        self.rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1
        self.reject_limit_locked = reject_limit_locked
        self.limit_threshold = limit_threshold

    def run(self, calendar: pd.DatetimeIndex, picks: dict[pd.Timestamp, list[str]],
            close: pd.DataFrame, vwap: pd.DataFrame | None = None) -> pd.DataFrame:
        """Simulate the portfolio. Returns per-day nav/return/turnover.

        Args:
        calendar: trading days to simulate (datetime index).
        picks: decision date -> list of stocks to buy that day. On calendar
            days absent from ``picks`` the due slot simply holds its existing
            positions instead of liquidating to cash.
        close / vwap: datetime x instrument price panels; vwap is the
            execution price (falls back to close where absent).
        """
        exec_px = vwap.reindex_like(close).fillna(close) if vwap is not None else close
        prev_close = close.shift(1)
        last_px = pd.Series(np.nan, index=close.columns)
        slots = [{"cash": 1.0 / self.H, "holdings": {}} for _ in range(self.H)]

        records = []
        for i, date in enumerate(calendar):
            slot = slots[i % self.H]
            px, cp, pc = exec_px.loc[date], close.loc[date], prev_close.loc[date]
            last_px = cp.fillna(last_px)
            sold = 0.0
            traded_today = date in picks  # no new signal: the slot holds its positions

            if traded_today:
                for s, shares in list(slot["holdings"].items()):
                    p = px.get(s)
                    if not np.isfinite(p) or p <= 0:
                        p = last_px.get(s)
                    if not np.isfinite(p):
                        continue  # no price at all: keep the position
                    if (self.reject_limit_locked and np.isfinite(pc.get(s))
                            and np.isfinite(cp.get(s))
                            and cp[s] / pc[s] - 1 <= -self.limit_threshold):
                        continue  # locked at lower limit: defer the sell
                    slot["cash"] += shares * p * (1 - self.fee)
                    sold += shares * p
                    del slot["holdings"][s]

            for sl in slots:
                sl["cash"] *= 1 + self.rf_daily

            bought = 0.0
            stocks = [s for s in picks.get(date, [])
                      if np.isfinite(px.get(s, np.nan)) and px[s] > 0]
            if self.reject_limit_locked:
                stocks = [s for s in stocks
                          if not (np.isfinite(pc.get(s, np.nan)) and np.isfinite(cp.get(s, np.nan))
                                  and cp[s] / pc[s] - 1 >= self.limit_threshold)]
            if traded_today and stocks and slot["cash"] > 0:
                per_stock = slot["cash"] / len(stocks)
                for s in stocks:
                    slot["holdings"][s] = per_stock * (1 - self.fee) / px[s]
                bought = slot["cash"]
                slot["cash"] = 0.0

            nav = sum(sl["cash"] + sum(sh * last_px.get(s, 0.0)
                                       for s, sh in sl["holdings"].items())
                      for sl in slots)
            records.append((date, nav, (sold + bought) / 2))

        df = pd.DataFrame(records, columns=["date", "nav", "traded"]).set_index("date")
        df["return"] = df["nav"].pct_change().fillna(0.0)
        df["turnover"] = df["traded"] / df["nav"].shift(1).fillna(1.0)
        return df[["nav", "return", "turnover"]]


def performance_metrics(daily_ret: pd.Series, rf_annual: float = 0.0,
                        benchmark_ret: pd.Series | None = None) -> dict:
    """Metric set of the paper's evaluation appendix (K=252).

    SR is the excess Sharpe ratio (AR - rf) / Vol; MDD is a positive magnitude.
    """
    daily_ret = daily_ret.dropna()
    if len(daily_ret) == 0:
        return {}
    nav = (1 + daily_ret).cumprod()
    ann_ret = float(nav.iloc[-1] ** (TRADING_DAYS_PER_YEAR / len(daily_ret)) - 1)
    ann_vol = float(daily_ret.std() * math.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = (ann_ret - rf_annual) / ann_vol if ann_vol > 0 else None
    mdd = float(-(nav / nav.cummax() - 1).min())
    downside = daily_ret - rf_annual / TRADING_DAYS_PER_YEAR
    downside_std = downside[downside < 0].std() * math.sqrt(TRADING_DAYS_PER_YEAR)
    sortino = (ann_ret - rf_annual) / downside_std if downside_std and downside_std > 0 else None

    out = {
        "total_return": float(nav.iloc[-1] - 1),
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": mdd,
        "calmar_ratio": ann_ret / mdd if mdd > 0 else None,
        "win_rate": float((daily_ret > 0).mean()),
        "trading_days": int(len(daily_ret)),
        "nav_series": {str(d.date()): float(v) for d, v in nav.items()},
    }
    if benchmark_ret is not None:
        bench = benchmark_ret.reindex(daily_ret.index).fillna(0.0)
        active = daily_ret - bench
        active_std = active.std()
        bench_metrics = performance_metrics(bench, rf_annual)
        out["benchmark_annual_return"] = bench_metrics.get("annual_return")
        out["excess_annual_return"] = ann_ret - bench_metrics["annual_return"] \
            if bench_metrics.get("annual_return") is not None else None
        out["information_ratio"] = float(active.mean() / active_std * math.sqrt(TRADING_DAYS_PER_YEAR)) \
            if active_std and active_std > 0 else None
        out["benchmark_nav_series"] = bench_metrics.get("nav_series")
    return out


def _parse_selection_dates(selections: dict) -> dict[pd.Timestamp, list[str]]:
    return {pd.Timestamp(str(d)): list(factors) for d, factors in selections.items()}


def run_strategy(selections: dict[str, list[str]], betas_path: str, config: dict) -> dict:
    """Run the end-to-end backtest for one selections mapping.

    Uses the unified data loader (TDengine/Parquet) and GPU factor
    computation; no qlib expression engine involved.
    """
    from ..data import load_ohlcv, load_calendar
    from .torch_factors import compute_factors

    betas, intercept = load_betas(betas_path)
    selections = _parse_selection_dates(selections)
    top_n = config.get("top_n", 10)

    # Load instruments and data
    if config.get("custom_instruments"):
        instruments = config["custom_instruments"]
    else:
        from ..data import load_instruments
        instruments = load_instruments(config.get("market", "all"))

    start, end = config["start_date"], config["end_date"]
    print(f"[strategy] loading OHLCV for {len(instruments)} instruments ...")
    panels = load_ohlcv(instruments, start, end)
    cal = panels["calendar"]
    close = panels["close"]

    decision_days = sorted(
        pd.Timestamp(str(d)) for d in selections
        if pd.Timestamp(start) <= pd.Timestamp(str(d)) <= pd.Timestamp(end)
        and pd.Timestamp(str(d)) in cal
    )
    if not decision_days:
        raise ValueError("no selection dates fall inside the backtest window/calendar")

    factor_names = sorted({f for factors in selections.values() for f in factors})

    # Compute factors on GPU
    device = config.get("device", "cuda")
    print(f"[strategy] computing {len(factor_names)} factors on {device} ...")
    F = compute_factors(
        *(torch.as_tensor(panels[f], device=device) for f in ["close", "high", "low", "open", "volume", "vwap"]),
        device=device,
    )

    # Build factor panels as DataFrames
    inst_list = instruments if isinstance(instruments, list) else list(instruments)
    factor_panels = {f: pd.DataFrame(F[f].cpu().numpy(), index=cal, columns=inst_list)
                     for f in factor_names if f in F}
    close_df = pd.DataFrame(close, index=cal, columns=inst_list)

    picks: dict[pd.Timestamp, list[str]] = {}
    for day in decision_days:
        prev_idx = max(cal.get_loc(day) - 1, 0)
        day_values = pd.DataFrame({f: factor_panels[f].iloc[prev_idx]
                                   for f in selections[day] if f in factor_panels})
        scores = score_stocks(day_values, selections[day], betas, intercept)
        if scores is not None:
            picks[day] = list(scores.nlargest(top_n).index)

    sim_cal = cal[(cal >= decision_days[0]) & (cal <= end)]
    holding = config.get("holding_days", 5)
    engine = SlotBacktest(
        holding_days=holding,
        fee=config.get("fee", 0.001),
        rf_annual=config.get("risk_free_rate", 0.0),
        reject_limit_locked=config.get("reject_limit_locked", False),
        limit_threshold=config.get("limit_threshold", 0.098),
    )
    vwap_df = pd.DataFrame(panels["vwap"], index=cal, columns=inst_list) if panels.get("vwap") is not None else None
    result = engine.run(sim_cal, picks, close_df, vwap_df)

    benchmark_ret = None
    if config.get("benchmark"):
        try:
            # Load benchmark via the unified data loader
            from ..data import load_ohlcv
            bench_panels = load_ohlcv([config["benchmark"]], str(sim_cal[0]), str(sim_cal[-1]))
            bench_close = pd.Series(bench_panels["close"][:, 0], index=bench_panels["calendar"])
            benchmark_ret = bench_close.pct_change()
        except Exception as e:
            print(f"[strategy] benchmark {config['benchmark']} unavailable: {e}")

    metrics = performance_metrics(result["return"], config.get("risk_free_rate", 0.0),
                                  benchmark_ret)
    return {
        "metrics": metrics,
        "daily": result.reset_index().assign(date=lambda d: d["date"].astype(str)).to_dict("records"),
        "picks": {str(d.date()): stocks for d, stocks in picks.items()},
        "config_snapshot": {k: config.get(k) for k in
                            ("market", "benchmark", "top_n", "holding_days", "fee",
                             "risk_free_rate", "start_date", "end_date")},
    }


def save_strategy_result(result: dict, output_dir: str | Path, name: str = "backtest_result") -> None:
    """Write metrics JSON and daily NAV CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "daily"},
                   ensure_ascii=False, indent=2, default=str))
    pd.DataFrame(result["daily"]).to_csv(output_dir / f"{name}_daily.csv", index=False)
    print(f"[strategy] -> {output_dir / (name + '.json')}")

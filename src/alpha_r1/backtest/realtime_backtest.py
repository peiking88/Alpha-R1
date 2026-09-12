"""Real-time / historical backtest engine — TDengine direct + GPU factors.

Self-contained engine that loads data via the unified ``alpha_r1.data``
interface and computes factors with ``torch_factors.py`` on the GPU.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from ..data import load_ohlcv, load_instruments
from .torch_factors import compute_factors


def _json_safe(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _ic_stats(factor: np.ndarray, fwd: np.ndarray) -> dict:
    """Daily cross-sectional IC / RankIC."""
    T = min(len(factor), len(fwd))
    ics, rank_ics = [], []
    for t in range(T):
        f = factor[t]; r = fwd[t]
        valid = np.isfinite(f) & np.isfinite(r)
        if valid.sum() < 10:
            continue
        fv, rv = f[valid], r[valid]
        if fv.std() == 0 or rv.std() == 0:
            continue
        ics.append(float(np.corrcoef(fv, rv)[0, 1]))
        try:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(fv, rv)
            rank_ics.append(float(rho))
        except Exception:
            pass
    ic = np.array(ics)
    ric = np.array(rank_ics)
    ic = ic[np.isfinite(ic)]
    ric = ric[np.isfinite(ric)]

    def _ir(s):
        if len(s) == 0: return None
        std = s.std()
        if not np.isfinite(std) or std == 0: return None
        return float(s.mean() / std)

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


def _topk_backtest(factor: np.ndarray, fwd: np.ndarray, topk: int, fee: float) -> dict:
    """Daily-rebalanced top-k equal-weight long portfolio."""
    T = min(len(factor), len(fwd))
    daily_ret, daily_turnover = [], []
    prev_holdings = set()
    for t in range(T):
        f = factor[t]
        valid = np.isfinite(f)
        if valid.sum() < topk:
            prev_holdings = set()
            continue
        idx = np.where(valid)[0]
        fv = f[idx]
        top_idx = idx[np.argsort(fv)[-topk:]]
        holdings = set(top_idx)
        r = fwd[t]
        r = r[list(holdings)]
        r = r[np.isfinite(r)]
        if len(r) == 0:
            prev_holdings = holdings
            continue
        turnover = 1.0 if not prev_holdings else 1 - len(holdings & prev_holdings) / topk
        ret = float(r.mean()) - fee * turnover * 2
        daily_ret.append(ret)
        daily_turnover.append(turnover)
        prev_holdings = holdings
    if not daily_ret:
        return {}
    rets = np.array(daily_ret)
    nav = np.cumprod(1 + rets)
    ann_ret = float(nav[-1] ** (252 / len(rets)) - 1)
    ann_vol = float(rets.std() * math.sqrt(252))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else None
    drawdown = float((nav / np.maximum.accumulate(nav) - 1).min())
    yearly = {}
    chunk = 252
    for i in range(0, len(rets), chunk):
        yr_rets = rets[i:i + chunk]
        y_nav = float((1 + yr_rets).prod())
        yearly[str(i // chunk)] = {
            "return": y_nav - 1,
            "sharpe": float(yr_rets.mean() / yr_rets.std() * math.sqrt(252)) if yr_rets.std() > 0 else None,
            "days": int(len(yr_rets)),
        }
    return {
        "total_return": float(nav[-1] - 1),
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": drawdown,
        "avg_daily_turnover": float(np.mean(daily_turnover)) if daily_turnover else None,
        "total_trading_days": int(len(rets)),
        "yearly_breakdown": yearly,
    }


def run_factor_backtest(
    names: list[str],
    config: dict,
    custom_instruments: list[str] | None = None,
    device: str = "cuda",
) -> dict[str, dict]:
    """Run single-factor backtests using TDengine data + GPU factors.

    Args:
        names: factor names (e.g. ``["alpha001", ...]``).
        config: parsed ``configs/backtest.yaml``.
        custom_instruments: optional instrument list. If None, the full
            universe from the data source is used.
        device: torch device.
    """
    names = [n.lower() for n in names]

    instruments = custom_instruments or load_instruments(config.get("market", "all"))

    print(f"[backtest] loading OHLCV for {len(instruments)} instruments ...")
    panels = load_ohlcv(instruments, config["start_date"], config["end_date"])
    calendar = panels["calendar"]
    close = panels["close"]
    print(f"[backtest] data shape: {close.shape} (T x N), {len(calendar)} days")

    # GPU tensors
    close_t = torch.as_tensor(close, device=device)
    high_t = torch.as_tensor(panels["high"], device=device)
    low_t = torch.as_tensor(panels["low"], device=device)
    open_t = torch.as_tensor(panels["open"], device=device)
    volume_t = torch.as_tensor(panels["volume"], device=device)
    vwap_t = torch.as_tensor(panels["vwap"], device=device)

    print(f"[backtest] computing {len(names)} factors on {device} ...")
    F = compute_factors(close_t, high_t, low_t, open_t, volume_t, vwap_t, device=device)

    # Forward returns
    fwd = close[1:] / close[:-1] - 1  # [T-1, N]

    output_dir = Path(config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name in names:
        if name not in F:
            continue
        factor = F[name][:-1].cpu().numpy()  # align with fwd
        if not np.any(np.isfinite(factor)):
            print(f"[backtest] SKIP {name}: all NaN")
            continue
        portfolio = _topk_backtest(factor, fwd, config.get("topk", 50), config.get("fee", 0.001))
        p_i = {
            "factor": name,
            "market": config.get("market", "custom"),
            "window": {"start": config["start_date"], "end": config["end_date"]},
            "topk": config.get("topk", 50),
            "fee": config.get("fee", 0.001),
            "ic": _ic_stats(factor, fwd),
            "portfolio": portfolio,
        }
        out_path = output_dir / f"{name}.json"
        out_path.write_text(json.dumps(_json_safe(p_i), ensure_ascii=False, indent=2))
        print(f"[backtest] {name} -> {out_path}")
        results[name] = p_i
    return results

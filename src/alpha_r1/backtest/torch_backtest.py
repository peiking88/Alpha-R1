"""GPU-accelerated single-factor backtest engine (PyTorch).

Replaces qlib's CPU expression engine with vectorized CUDA tensor ops.
Loads OHLCV once, computes all 82 Alpha101 factors in parallel on the GPU,
then evaluates IC/RankIC and a daily-rebalanced top-k portfolio.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import init_qlib
from .torch_factors import compute_factors


def _json_safe(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _load_ohlcv(qlib_dir: str, instruments: list[str], start: str, end: str):
    """Load OHLCV + vwap from qlib into numpy arrays [T, N]."""
    from qlib.data import D

    fields = ["$close", "$high", "$low", "$open", "$volume", "$vwap"]
    df = D.features(instruments, fields, start_time=start, end_time=end)
    # Align to a common (datetime, instrument) index
    parts = []
    for f in fields:
        s = df[f].unstack(level="instrument")
        parts.append(s)
    # Use the union of indices; forward-fill is NOT done (NaN = no data)
    idx = parts[0].index
    cols = parts[0].columns
    arrays = []
    for s in parts:
        s = s.reindex(index=idx, columns=cols)
        arrays.append(s.values.astype(np.float32))
    return arrays, list(idx), list(cols)


def _ic_stats(factor: np.ndarray, fwd: np.ndarray) -> dict:
    """Daily cross-sectional IC / RankIC."""
    T = min(len(factor), len(fwd))
    ics, rank_ics = [], []
    for t in range(T):
        f = factor[t]
        r = fwd[t]
        valid = np.isfinite(f) & np.isfinite(r)
        if valid.sum() < 10:
            continue
        fv, rv = f[valid], r[valid]
        if fv.std() == 0 or rv.std() == 0:
            continue
        ics.append(float(np.corrcoef(fv, rv)[0, 1]))
        # Spearman rank IC
        from scipy.stats import spearmanr
        rho, _ = spearmanr(fv, rv)
        rank_ics.append(float(rho))
    ic = np.array(ics)
    ric = np.array(rank_ics)
    ic = ic[np.isfinite(ic)]
    ric = ric[np.isfinite(ric)]

    def _ir(s):
        if len(s) == 0:
            return None
        std = s.std()
        if not np.isfinite(std) or std == 0:
            return None
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
        # nlargest
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
    # approximate yearly by chunking
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


def backtest_factors_torch(
    names: list[str],
    config: dict,
    custom_instruments: list[str] | None = None,
    device: str = "cuda",
) -> dict[str, dict]:
    """Run GPU backtest for the given factors.

    Args:
        names: factor names.
        config: parsed backtest.yaml.
        custom_instruments: optional instrument list (e.g. zxg).
        device: torch device.
    """
    names = [n.lower() for n in names]
    qlib_dir = str(Path(config["qlib_data_dir"]).expanduser())
    init_qlib(qlib_dir, kernels=1)

    if custom_instruments:
        instruments = custom_instruments
    else:
        # Read from qlib's all.txt
        all_txt = Path(qlib_dir) / "instruments" / "all.txt"
        instruments = [line.split("\t")[0] for line in all_txt.read_text().splitlines() if line.strip()]

    print(f"[torch] loading OHLCV for {len(instruments)} instruments ...")
    arrays, dates, cols = _load_ohlcv(qlib_dir, instruments, config["start_date"], config["end_date"])
    close_np, high_np, low_np, open_np, vol_np, vwap_np = arrays
    print(f"[torch] data shape: {close_np.shape} (T x N), {len(dates)} days")

    # Move to GPU
    close = torch.as_tensor(close_np, device=device)
    high = torch.as_tensor(high_np, device=device)
    low = torch.as_tensor(low_np, device=device)
    open_ = torch.as_tensor(open_np, device=device)
    volume = torch.as_tensor(vol_np, device=device)
    vwap = torch.as_tensor(vwap_np, device=device)

    print(f"[torch] computing {len(names)} factors on {device} ...")
    F = compute_factors(close, high, low, open_, volume, vwap, device=device)
    print(f"[torch] computed {len(F)} factors")

    # Forward returns
    fwd = close[1:] / close[:-1] - 1  # [T-1, N]
    fwd_np = fwd.cpu().numpy()

    output_dir = Path(config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name in names:
        if name not in F:
            continue
        factor = F[name]
        # Align factor with forward returns (factor today -> return tomorrow)
        factor_aligned = factor[:-1]  # [T-1, N]
        factor_np = factor_aligned.cpu().numpy()
        if not np.any(np.isfinite(factor_np)):
            print(f"[torch] SKIP {name}: all NaN")
            continue
        portfolio = _topk_backtest(factor_np, fwd_np, config.get("topk", 50), config.get("fee", 0.001))
        p_i = {
            "factor": name,
            "market": config.get("market", "custom"),
            "window": {"start": config["start_date"], "end": config["end_date"]},
            "topk": config.get("topk", 50),
            "fee": config.get("fee", 0.001),
            "ic": _ic_stats(factor_np, fwd_np),
            "portfolio": portfolio,
        }
        out_path = output_dir / f"{name}.json"
        out_path.write_text(json.dumps(_json_safe(p_i), ensure_ascii=False, indent=2))
        print(f"[torch] {name} -> {out_path}")
        results[name] = p_i
    return results

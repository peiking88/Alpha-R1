"""PyTorch GPU implementation of Alpha101 factors.

Evaluates all 82 Alpha101 factors as vectorized tensor operations on the GPU.
Input: float32 tensors of shape [T, N] (time x instruments) on CUDA.
Output: dict of factor tensors, each [T, N] aligned to the input calendar.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Rolling-window primitives via unfold
# ---------------------------------------------------------------------------

def _unfold(x: torch.Tensor, W: int) -> torch.Tensor:
    """Rolling window view of x [T, N] -> [T, N, W].

    Output has the SAME length as input.  For the first W-1 rows the window
    is padded with NaN (warm-up period).
    """
    T, N = x.shape
    if W <= 1:
        return x.unsqueeze(2)  # [T, N, 1]
    # Pad W-1 NaN rows at the front, then unfold
    pad = torch.full((W - 1, N), float("nan"), dtype=x.dtype, device=x.device)
    xpad = torch.cat([pad, x], dim=0)  # [T+W-1, N]
    return xpad.unfold(0, W, 1)  # [T, N, W]


def _align_front(x: torch.Tensor, W: int, value: torch.Tensor) -> torch.Tensor:
    """No-op: _unfold already returns [T, N, W] aligned to input length."""
    return value


def _nanmean(x: torch.Tensor, dim: int) -> torch.Tensor:
    valid = ~torch.isnan(x)
    return torch.where(valid, x, torch.zeros_like(x)).sum(dim=dim) / valid.float().sum(dim=dim).clamp(min=1)


def _nansum(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.where(~torch.isnan(x), x, torch.zeros_like(x)).sum(dim=dim)


def _nanmin(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.where(~torch.isnan(x), x, torch.full_like(x, float("inf"))).min(dim=dim).values


def _nanmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.where(~torch.isnan(x), x, torch.full_like(x, float("-inf"))).max(dim=dim).values


def ts_mean(x: torch.Tensor, W: int) -> torch.Tensor:
    return _align_front(x, W, _nanmean(_unfold(x, W), dim=2))


def ts_sum(x: torch.Tensor, W: int) -> torch.Tensor:
    return _align_front(x, W, _nansum(_unfold(x, W), dim=2))


def _nanstd(x: torch.Tensor, dim: int, correction: int = 0) -> torch.Tensor:
    """NaN-aware standard deviation (PyTorch has no nanstd)."""
    valid = ~torch.isnan(x)
    # Count valid per reduction group
    n = valid.float().sum(dim=dim)
    # Replace NaN with 0 for mean computation
    x0 = torch.where(valid, x, torch.zeros_like(x))
    mean = x0.sum(dim=dim) / n.clamp(min=1)
    # Expand mean back
    shape = list(x.shape)
    shape[dim] = 1
    mean_exp = mean.unsqueeze(dim)
    var = torch.where(valid, (x - mean_exp) ** 2, torch.zeros_like(x)).sum(dim=dim) / (n - correction).clamp(min=1)
    return var.sqrt()


def ts_std(x: torch.Tensor, W: int) -> torch.Tensor:
    return _align_front(x, W, _nanstd(_unfold(x, W), dim=2, correction=0))


def ts_min(x: torch.Tensor, W: int) -> torch.Tensor:
    return _align_front(x, W, _nanmin(_unfold(x, W), dim=2))


def ts_max(x: torch.Tensor, W: int) -> torch.Tensor:
    return _align_front(x, W, _nanmax(_unfold(x, W), dim=2))


def ts_argmax(x: torch.Tensor, W: int) -> torch.Tensor:
    """Days since N-day maximum (1 = today)."""
    w = _unfold(x, W)
    ax = torch.nan_to_num(w, nan=-1e30).argmax(dim=2)
    return _align_front(x, W, (W - 1 - ax).float())


def ts_argmin(x: torch.Tensor, W: int) -> torch.Tensor:
    w = _unfold(x, W)
    amin = torch.nan_to_num(w, nan=1e30).argmin(dim=2)
    return _align_front(x, W, (W - 1 - amin).float())


def ts_rank(x: torch.Tensor, W: int) -> torch.Tensor:
    """Time-series percentile rank of the most recent value within window."""
    w = _unfold(x, W)
    today = w[:, :, -1:]
    return _align_front(x, W, _nanmean((w <= today).float(), dim=2))


def delay(x: torch.Tensor, D: int) -> torch.Tensor:
    if D <= 0:
        return x
    pad = torch.full((D,) + x.shape[1:], float("nan"), dtype=x.dtype, device=x.device)
    return torch.cat([pad, x[:-D]], dim=0)


def delta(x: torch.Tensor, D: int) -> torch.Tensor:
    return x - delay(x, D)


# ---------------------------------------------------------------------------
# Cross-sectional primitives
# ---------------------------------------------------------------------------

def cs_rank(x: torch.Tensor) -> torch.Tensor:
    """Cross-sectional percentile rank per day (deterministic across GPU/CPU)."""
    # Handle boolean input (from comparison ops)
    if x.dtype == torch.bool:
        x = x.float()
    N = x.shape[1]
    valid = ~torch.isnan(x)
    filled = torch.where(valid, x, torch.tensor(float("inf"), dtype=x.dtype, device=x.device))
    # Tie-breaker: add a tiny position-dependent offset so argsort is deterministic
    # across GPU/CPU even when values are equal.  Offset 1e-6 is far below factor precision.
    tie_break = torch.arange(N, dtype=torch.float32, device=x.device).view(1, N) * 1e-6
    rank = torch.argsort(torch.argsort(filled + tie_break, dim=1, stable=True), dim=1, stable=True).float()
    result = rank / max(N - 1, 1)
    return torch.where(valid, result, torch.tensor(float("nan"), dtype=x.dtype, device=x.device))


def cs_scale(x: torch.Tensor) -> torch.Tensor:
    """Cross-sectional scaling so sum(|x|) == 1 per day."""
    s = x.abs().nansum(dim=1, keepdim=True).clamp(min=1e-12)
    return x / s


# ---------------------------------------------------------------------------
# Correlation / covariance (time-series)
# ---------------------------------------------------------------------------

def _ts_correl(x: torch.Tensor, y: torch.Tensor, W: int) -> torch.Tensor:
    wx = _unfold(x, W)
    wy = _unfold(y, W)
    mx = wx.nanmean(dim=2, keepdim=True)
    my = wy.nanmean(dim=2, keepdim=True)
    dx = wx - mx
    dy = wy - my
    valid = (~torch.isnan(dx)) & (~torch.isnan(dy))
    dx = torch.where(valid, dx, torch.zeros_like(dx))
    dy = torch.where(valid, dy, torch.zeros_like(dy))
    num = (dx * dy).nansum(dim=2)
    den = (dx.pow(2).nansum(dim=2).clamp(min=1e-12) * dy.pow(2).nansum(dim=2).clamp(min=1e-12)).sqrt()
    return _align_front(x, W, (num / den).nan_to_num(0.0))


def correlation(x: torch.Tensor, y: torch.Tensor, W: int) -> torch.Tensor:
    return _ts_correl(x, y, W)


def covariance(x: torch.Tensor, y: torch.Tensor, W: int) -> torch.Tensor:
    wx = _unfold(x, W)
    wy = _unfold(y, W)
    mx = wx.nanmean(dim=2, keepdim=True)
    my = wy.nanmean(dim=2, keepdim=True)
    dx = wx - mx
    dy = wy - my
    valid = (~torch.isnan(dx)) & (~torch.isnan(dy))
    dx = torch.where(valid, dx, torch.zeros_like(dx))
    dy = torch.where(valid, dy, torch.zeros_like(dy))
    nvalid = valid.float().sum(dim=2).clamp(min=1)
    return _align_front(x, W, (dx * dy).sum(dim=2) / nvalid)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def signed_power(x: torch.Tensor, a: float) -> torch.Tensor:
    return torch.sign(x) * x.abs().pow(a)


def decay_linear(x: torch.Tensor, W: int) -> torch.Tensor:
    """Linearly weighted moving average, oldest weight 1, newest W."""
    w = _unfold(x, W)
    weights = torch.arange(1, W + 1, dtype=x.dtype, device=x.device).view(1, 1, W)
    valid = ~torch.isnan(w)
    w0 = torch.where(valid, w, torch.zeros_like(w))
    wt = weights * valid.float()
    wt_sum = wt.sum(dim=2).clamp(min=1e-12)
    return _align_front(x, W, (w0 * weights).sum(dim=2) / wt_sum)


def _adv(volume: torch.Tensor, W: int) -> torch.Tensor:
    return ts_mean(volume, W)


# ---------------------------------------------------------------------------
# Factor evaluation
# ---------------------------------------------------------------------------

def _ensure_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def compute_factors(close, high, low, open_, volume, vwap, device="cuda"):
    """Evaluate all 82 Alpha101 factors on the GPU.

    Args:
        close/high/low/open_/volume/vwap: array-like [T, N].
        device: torch device (``"cuda"`` or ``"cpu"``).

    Returns:
        dict mapping factor name -> float32 tensor [T, N].
    """
    close = _ensure_tensor(close, device)
    high = _ensure_tensor(high, device)
    low = _ensure_tensor(low, device) if low is not None else close.clone()
    open_ = _ensure_tensor(open_, device)
    volume = _ensure_tensor(volume, device)
    vwap = _ensure_tensor(vwap, device)

    returns = delta(close, 1) / delay(close, 1).clamp(min=1e-6)
    adv5 = _adv(volume, 5)
    adv10 = _adv(volume, 10)
    adv15 = _adv(volume, 15)
    adv20 = _adv(volume, 20)
    adv30 = _adv(volume, 30)
    adv40 = _adv(volume, 40)
    adv50 = _adv(volume, 50)
    adv60 = _adv(volume, 60)
    adv120 = _adv(volume, 120)
    adv180 = _adv(volume, 180)

    F = {}
    F["alpha001"] = cs_rank(ts_argmax(signed_power(torch.where(returns < 0, ts_std(returns, 20), close), 2.0), 5)) - 0.5
    F["alpha002"] = correlation(cs_rank(delta(torch.log(volume.clamp(min=1e-6)), 2)), cs_rank((close - open_) / open_.clamp(min=1e-6)), 6)
    F["alpha003"] = -1 * correlation(cs_rank(open_), cs_rank(volume), 10)
    F["alpha004"] = -1 * ts_rank(cs_rank(low), 9)
    F["alpha005"] = cs_rank(open_ - ts_sum(vwap, 10) / 10) * cs_rank(close - vwap)
    F["alpha006"] = -1 * correlation(open_, volume, 10)
    F["alpha007"] = torch.where(adv20 < volume, -1 * ts_rank(delta(close, 7).abs(), 60) * torch.sign(delta(close, 7)), torch.tensor(-1.0, device=device))
    F["alpha008"] = cs_rank(ts_sum(open_, 5) * ts_sum(returns, 5) - delay(ts_sum(open_, 5) * ts_sum(returns, 5), 10))
    F["alpha009"] = delta(close, 1)
    F["alpha010"] = cs_rank(torch.where(0 < ts_min(delta(close, 1), 4), delta(close, 1), torch.where(ts_max(delta(close, 1), 4) < 0, delta(close, 1), delta(close, 1))))
    F["alpha011"] = (cs_rank(ts_max(vwap - close, 3)) + cs_rank(ts_min(vwap - close, 3))) * cs_rank(delta(volume, 3)) * torch.sign(close - vwap)
    F["alpha012"] = torch.sign(delta(volume, 1)) * delta(close, 1)
    F["alpha013"] = (covariance(cs_rank(close), cs_rank(volume), 5) - 0.5) * torch.sign(delta(close, 5))
    F["alpha014"] = (-1 * cs_rank(delta(returns, 3))) * correlation(open_, volume, 10) * (-1 * torch.sign(delta(close, 10)))
    F["alpha015"] = -1 * ts_sum(cs_rank(correlation(cs_rank(high), cs_rank(volume), 3)), 3)
    F["alpha016"] = -1 * cs_rank(covariance(cs_rank(high), cs_rank(volume), 5))
    F["alpha017"] = ((-1 * cs_rank(ts_rank(close, 10))) * cs_rank(delta(delta(close, 1), 1))) * cs_rank(ts_rank(volume / adv20.clamp(min=1e-6), 5)) * (-1 * torch.sign(delta(close, 5)))
    F["alpha018"] = -1 * cs_rank(ts_std((close - open_).abs(), 5) + (close - open_) + correlation(close, open_, 10))
    F["alpha019"] = (-1 * torch.sign((close - delay(close, 7)) + delta(close, 7))) * (1 + cs_rank(1 + ts_sum(returns, 250)))
    F["alpha020"] = (-1 * cs_rank(open_ - delay(high, 1))) * cs_rank(open_ - delay(close, 1)) * cs_rank(open_ - delay(low, 1))
    F["alpha021"] = torch.where(0 < ts_min(delta(close, 1), 5), delta(close, 1), torch.where(ts_max(delta(close, 1), 5) < 0, delta(close, 1), -1 * delta(close, 1)))
    F["alpha022"] = -1 * (delta(correlation(high, volume, 5), 5) * cs_rank(ts_std(close, 20)))
    F["alpha023"] = torch.where(ts_sum(high, 20) / 20 < high, delta(high, 2), torch.zeros_like(close))
    F["alpha024"] = (close - delay(close, 5)) / delay(close, 5).clamp(min=1e-6)
    F["alpha025"] = cs_rank((-1 * returns * adv20 * vwap * (high - close)) / (ts_sum(adv20, 20) / 20 * ts_sum(high - close, 20) / 20).clamp(min=1e-6))
    F["alpha026"] = ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)
    F["alpha027"] = torch.where(0.5 > cs_rank(ts_sum(correlation(cs_rank(volume), cs_rank(vwap), 6), 2) / 2.0), torch.tensor(-1.0, device=device), torch.tensor(1.0, device=device))
    F["alpha028"] = cs_scale(correlation(adv20, low, 5) + (high + low) / 2 - close)
    F["alpha029"] = torch.min(
        cs_rank(cs_scale(torch.log(ts_min(cs_rank(cs_rank(-1 * cs_rank(delta(close - 1, 5)))), 2).clamp(min=1e-6)))),
        torch.tensor(5.0, device=device)
    ) + ts_rank(delay(-1 * returns, 6), 5)
    F["alpha030"] = torch.sign(close - delay(close, 1)) * (
        1.0 - cs_rank(torch.sign(close - delay(close, 1)) + torch.sign(delay(close, 1) - delay(close, 2)) + torch.sign(delay(close, 2) - delay(close, 3)))
    ) * ts_sum(volume, 5) / ts_sum(volume, 20).clamp(min=1e-6)
    F["alpha031"] = (
        cs_rank(cs_rank(cs_rank(decay_linear(-1 * cs_rank(cs_rank(delta(close, 10))), 10))))
        + cs_rank(-1 * delta(close, 3))
        + torch.sign(cs_scale(correlation(adv20, low, 12)))
        + torch.sign(delta(close, 5))
    )
    F["alpha032"] = cs_scale(ts_sum(close, 7) / 7 - close) + 20 * cs_scale(correlation(vwap, delay(close, 5), 230))
    F["alpha033"] = cs_rank((1 - open_ / close.clamp(min=1e-6)).pow(1))
    F["alpha034"] = cs_rank(1 - cs_rank(ts_std(returns, 2) / ts_std(returns, 5).clamp(min=1e-6))) + (1 - cs_rank(delta(close, 1)))
    F["alpha035"] = ts_rank(volume, 32) * (1 - ts_rank(close + high - low, 16)) * (1 - ts_rank(returns, 32))
    F["alpha036"] = (
        cs_rank(correlation(close - open_, delay(volume, 1), 15)) * 2.21
        + cs_rank(open_ - close) * 0.7
        + cs_rank(ts_rank(delay(-1 * returns, 6), 5)) * 0.73
        + cs_rank(correlation(vwap, adv20, 6).abs())
        + cs_rank(((ts_sum(close, 200) / 200 - open_) * (close - open_))) * 0.6
    )
    F["alpha037"] = cs_rank(correlation(delay(close - open_, 1), close, 200)) + cs_rank(close - open_)
    F["alpha038"] = cs_rank(ts_rank(close, 10)) * cs_rank(close / open_.clamp(min=1e-6))
    F["alpha039"] = (-1 * cs_rank(delta(close, 7) * (1 - cs_rank(decay_linear(volume / adv20.clamp(min=1e-6), 9))))) * (1 + cs_rank(ts_sum(returns, 250))) * cs_rank(-delta(close, 7))
    F["alpha040"] = (-1 * cs_rank(ts_std(high, 10))) * correlation(high, volume, 10) * cs_rank(ts_std(close, 10))
    F["alpha041"] = (high * low).pow(0.5) - vwap
    F["alpha042"] = cs_rank(vwap - close) / cs_rank(vwap + close).clamp(min=1e-6)
    F["alpha043"] = ts_rank(volume / adv20.clamp(min=1e-6), 20) * ts_rank(-1 * delta(close, 7), 8)
    F["alpha044"] = -1 * correlation(high, cs_rank(volume), 5)
    F["alpha045"] = cs_rank(ts_sum(delay(close, 5), 20) / 20) * correlation(close, volume, 2) * cs_rank(correlation(ts_sum(close, 5), ts_sum(close, 20), 2))
    F["alpha046"] = -cs_rank(
        ((delay(close, 10) - delay(close, 20)) / delay(close, 20).clamp(min=1e-6))
        - ((close - delay(close, 10)) / delay(close, 10).clamp(min=1e-6))
    )
    F["alpha047"] = (
        (cs_rank(1 / close.clamp(min=1e-6)) * volume / adv20.clamp(min=1e-6))
        * ((high * cs_rank(high - close)) / (ts_sum(high, 5) / 5).clamp(min=1e-6))
        - cs_rank(vwap - delay(vwap, 5))
    )
    F["alpha049"] = torch.where(
        (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) > 0.1,
        torch.tensor(1.0, device=device),
        -1 * (close - delay(close, 1)),
    )
    F["alpha050"] = ts_max(cs_rank(correlation(cs_rank(volume), cs_rank(vwap), 5)), 5)
    F["alpha051"] = torch.where(
        (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) > 0.05,
        torch.tensor(1.0, device=device),
        -1 * (close - delay(close, 1)),
    )
    F["alpha052"] = (-1 * ts_min(low, 5) + delay(ts_min(low, 5), 5)) * cs_rank((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220) * ts_rank(volume, 5)
    F["alpha053"] = delta(((close - low) - (high - close)) / (close - low).clamp(min=1e-6), 9)
    F["alpha054"] = (-1 * (low - close) * open_.pow(5)) / ((low - high) * close.pow(5).clamp(min=1e-6)) * cs_rank(ts_rank(volume, 5))
    F["alpha055"] = -1 * correlation(
        cs_rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)).clamp(min=1e-6)),
        cs_rank(volume), 6
    ) * cs_rank(ts_std(close, 20))
    F["alpha057"] = (close - vwap) / decay_linear(cs_rank(ts_argmax(close, 30)), 2).clamp(min=1e-6)
    F["alpha060"] = 2 * cs_scale(cs_rank((((close - low) - (high - close)) / (high - low).clamp(min=1e-6)) * volume)) - cs_scale(cs_rank(ts_argmax(close, 10)))
    F["alpha061"] = torch.sign(cs_rank(vwap - ts_min(vwap, 16)) - cs_rank(correlation(vwap, adv180, 18)))
    F["alpha062"] = (cs_rank(correlation(vwap, ts_sum(adv20, 22), 10)) - cs_rank(((cs_rank(open_) + cs_rank(open_)) < (cs_rank((high + low) / 2) + cs_rank(high))))) * (-ts_sum(returns, 5))
    F["alpha064"] = (
        cs_rank(correlation(ts_sum(open_ * 0.178404 + low * (1 - 0.178404), 13), ts_sum(adv120, 13), 17))
        - cs_rank(delta(((high + low) / 2 * 0.178404 + vwap * (1 - 0.178404)), 4))
    ) * -1
    F["alpha065"] = cs_rank(correlation(vwap, ts_sum(adv60, 9), 6)) - cs_rank(open_ - ts_min(open_, 14))
    F["alpha066"] = (
        cs_rank(decay_linear(delta(vwap, 4), 7))
        + ts_rank(decay_linear(-(((low - vwap) / (open_ - (high + low) / 2)).clamp(min=1e-6)), 11), 7)
    )
    F["alpha068"] = cs_rank(delta(close * 0.518371 + low * (1 - 0.518371), 1)) - ts_rank(correlation(cs_rank(high), cs_rank(adv15), 9), 14)
    F["alpha071"] = torch.max(
        cs_rank(decay_linear(correlation(ts_rank(close, 3), ts_rank(adv180, 12), 18), 4)),
        cs_rank(decay_linear((cs_rank(low + open_ - 2 * vwap)).pow(2), 16)),
    )
    F["alpha072"] = cs_rank(decay_linear(correlation((high + low) / 2, adv40, 9), 10)) - cs_rank(decay_linear(-correlation(ts_rank(vwap, 4), ts_rank(volume, 18), 7), 3))
    F["alpha073"] = torch.max(
        cs_rank(decay_linear(-delta(vwap, 5), 3)),
        ts_rank(decay_linear(-delta(open_ * 0.147155 + low * (1 - 0.147155), 2) / (open_ * 0.147155 + low * (1 - 0.147155)).clamp(min=1e-6), 3), 17),
    )
    F["alpha074"] = (
        cs_rank(correlation(cs_rank(high * 0.0261661 + vwap * (1 - 0.0261661)), cs_rank(volume), 11))
        - cs_rank(correlation(close, ts_sum(adv30, 37), 15))
    ) / (
        cs_rank(correlation(cs_rank(high * 0.0261661 + vwap * (1 - 0.0261661)), cs_rank(volume), 11))
        + cs_rank(correlation(close, ts_sum(adv30, 37), 15))
        + 1e-6
    )
    F["alpha075"] = (
        cs_rank(correlation(vwap, volume, 4))
        - cs_rank(correlation(cs_rank(low), cs_rank(adv50), 12))
    ) * (-ts_rank(close, 20) + 0.5)
    F["alpha077"] = (
        cs_rank(decay_linear(vwap - (high + low) / 2, 20))
        + cs_rank(decay_linear(-correlation((high + low) / 2, adv40, 3), 6))
    )
    F["alpha078"] = (
        cs_rank(correlation(ts_mean(low * 0.352233 + vwap * 0.647767, 20), ts_mean(adv40, 20), 7))
        + cs_rank(correlation(vwap, volume, 6))
    )
    F["alpha081"] = cs_rank(correlation(vwap, ts_sum(adv10, 50), 8)) - cs_rank(correlation(cs_rank(vwap), cs_rank(volume), 5))
    F["alpha083"] = (
        cs_rank(delay((high - low) / (ts_sum(close, 5) / 5), 2))
        * cs_rank(cs_rank(volume))
        * (vwap - close)
    ) / ((high - low) / (ts_sum(close, 5) / 5)).clamp(min=1e-6)
    F["alpha084"] = signed_power(ts_rank(vwap - ts_max(vwap, 15), 21), delta(close, 5).abs() + 1)
    F["alpha085"] = (
        cs_rank(correlation(high * 0.876703 + close * (1 - 0.876703), adv30, 10))
        ** cs_rank(correlation(ts_rank((high + low) / 2, 4), ts_rank(volume, 10), 7))
    ) * torch.sign(ts_sum(returns, 5))
    F["alpha086"] = (cs_rank(close - vwap) - ts_rank(correlation(close, ts_sum(adv20, 15), 6), 20)) * -1
    F["alpha088"] = torch.min(
        cs_rank(decay_linear(-((cs_rank(open_) + cs_rank(low)) - (cs_rank(high) + cs_rank(close))), 8)),
        ts_rank(decay_linear(-correlation(ts_rank(close, 8), ts_rank(adv60, 21), 8), 7), 3),
    )
    F["alpha092"] = torch.max(
        ts_rank(decay_linear(((high + low) / 2 + close > low + open_).float(), 15), 19),
        ts_rank(decay_linear(correlation(cs_rank(-low.clamp(min=1e-6)), cs_rank(adv30), 8), 7), 7),
    )
    F["alpha094"] = cs_rank(vwap - ts_min(vwap, 12)) * correlation(ts_rank(vwap, 20), ts_rank(adv60, 4), 18)
    F["alpha095"] = torch.sign(
        ts_rank(cs_rank(correlation(ts_sum((high + low) / 2, 20), ts_sum(adv40, 20), 13)).pow(5), 12)
        - cs_rank(open_ - ts_min(open_, 12))
    )
    F["alpha096"] = torch.max(
        ts_rank(decay_linear(correlation(cs_rank(vwap), cs_rank(volume), 4), 4), 8),
        ts_rank(decay_linear(ts_argmax(correlation(ts_rank(close, 7), ts_rank(adv60, 4), 4), 13), 14), 13),
    )
    F["alpha098"] = (
        cs_rank(decay_linear(correlation(vwap, ts_sum(adv5, 26), 5), 7))
        + cs_rank(decay_linear(-ts_rank(ts_argmin(correlation(cs_rank(open_), cs_rank(adv15), 20), 9), 7), 8))
    )
    F["alpha099"] = (
        correlation(ts_sum((high + low) / 2, 20), ts_sum(adv60, 20), 9)
        - correlation(low, volume, 6)
    ) * -1
    F["alpha101"] = (close - open_) / ((high - low) + 0.001)

    # Clean up: drop the temporary rank_cov artifact if present
    F.pop("rank_cov", None)
    return F
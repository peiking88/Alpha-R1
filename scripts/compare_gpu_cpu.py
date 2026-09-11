#!/usr/bin/env python3
"""Fast GPU vs CPU (PyTorch) + PyTorch vs qlib equivalence check.

Strategy:
  1. GPU vs CPU (both PyTorch): verifies the PyTorch implementation is device-agnostic.
  2. PyTorch vs qlib: pick 3 representative factors on 3 instruments (index/stock/fund),
     compute with both engines, compare values.

For the qlib comparison we pre-filter to the 3 instruments so qlib doesn't waste
time on the full 8255-stock universe.
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_r1.backtest.data import init_qlib
from alpha_r1.backtest.torch_factors import compute_factors

try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_ohlcv(qlib_dir, instruments, start, end):
    from qlib.data import D
    fields = ["$close", "$high", "$low", "$open", "$volume", "$vwap"]
    df = D.features(instruments, fields, start_time=start, end_time=end)
    parts = []
    for f in fields:
        s = df[f].unstack(level="instrument")
        parts.append(s)
    idx = parts[0].index
    cols = parts[0].columns
    arrays = []
    for s in parts:
        s = s.reindex(index=idx, columns=cols)
        arrays.append(s.values.astype(np.float32))
    return arrays, list(idx), list(cols)


def compute_qlib_subset(qlib_dir, instruments, start, end, names):
    """Compute a subset of factors with qlib, only for the given instruments."""
    from qlib.data import D
    from alpha_r1.backtest.alpha101_qlib import QLIB_EXPRESSIONS, register_custom_ops, set_cs_universe
    set_cs_universe("all")
    register_custom_ops()
    F = {}
    for name in names:
        expr = QLIB_EXPRESSIONS[name]
        try:
            f = D.features(instruments, [expr], start_time=start, end_time=end)
            F[name] = f.iloc[:, 0].unstack(level="instrument").values
        except Exception as e:
            print(f"  qlib {name} failed: {e}")
    return F


def main():
    qlib_dir = str(Path.home() / ".qlib/qlib_data/alpha_r1")
    init_qlib(qlib_dir, kernels=1)

    # Pick: index, stock, fund (all must have data in qlib)
    instruments = ["SH000688", "SH600769", "SH501001"]
    start, end = "2023-01-01", "2024-12-31"

    print(f"Loading OHLCV for {instruments} ...")
    arrays, dates, cols = load_ohlcv(qlib_dir, instruments, start, end)
    close_np, high_np, low_np, open_np, vol_np, vwap_np = arrays
    print(f"Data shape: {close_np.shape} (T x N)")

    # ---- 1. GPU vs CPU (both PyTorch) ----
    print("\n=== 1. GPU vs CPU (PyTorch) ===")
    F_gpu = compute_factors(
        torch.as_tensor(close_np, device="cuda"),
        torch.as_tensor(high_np, device="cuda"),
        torch.as_tensor(low_np, device="cuda"),
        torch.as_tensor(open_np, device="cuda"),
        torch.as_tensor(vol_np, device="cuda"),
        torch.as_tensor(vwap_np, device="cuda"),
        device="cuda",
    )
    F_cpu = compute_factors(
        torch.as_tensor(close_np, device="cpu"),
        torch.as_tensor(high_np, device="cpu"),
        torch.as_tensor(low_np, device="cpu"),
        torch.as_tensor(open_np, device="cpu"),
        torch.as_tensor(vol_np, device="cpu"),
        torch.as_tensor(vwap_np, device="cpu"),
        device="cpu",
    )
    gpu_np = {k: v.cpu().numpy() for k, v in F_gpu.items()}
    cpu_np = {k: v.numpy() for k, v in F_cpu.items()}

    print(f"{'factor':<12} {'max_abs_diff':>14} {'match':>8}")
    print("-" * 40)
    mismatch = []
    for name in sorted(gpu_np.keys()):
        if name not in cpu_np:
            continue
        g, c = gpu_np[name], cpu_np[name]
        valid = np.isfinite(g) & np.isfinite(c)
        if valid.sum() == 0:
            continue
        diff = np.abs(g[valid] - c[valid]).max()
        ok = diff < 1e-4
        if not ok:
            mismatch.append(name)
        print(f"{name:<12} {diff:>14.6e} {'OK' if ok else 'DIFF':>8}")
    print(f"\nGPU vs CPU: {len(set(gpu_np) & set(cpu_np))} factors compared, {len(mismatch)} mismatches")
    if mismatch:
        print(f"Mismatches: {mismatch[:10]}")

    # ---- 2. PyTorch vs qlib (subset) ----
    # Pick 3 representative factors: simple (alpha009), medium (alpha001), complex (alpha036)
    subset = ["alpha001", "alpha009", "alpha036"]
    print(f"\n=== 2. PyTorch vs qlib (factors: {subset}) ===")
    F_qlib = compute_qlib_subset(qlib_dir, instruments, start, end, subset)

    print(f"{'factor':<12} {'max_abs_diff':>14} {'match':>8}")
    print("-" * 40)
    qlib_mismatch = []
    for name in subset:
        if name not in F_qlib or name not in gpu_np:
            print(f"{name:<12} {'(missing)':>14}")
            continue
        pt = gpu_np[name]
        ql = F_qlib[name]
        T = min(pt.shape[0], ql.shape[0])
        N = min(pt.shape[1], ql.shape[1])
        pt = pt[:T, :N]
        ql = ql[:T, :N]
        valid = np.isfinite(pt) & np.isfinite(ql)
        if valid.sum() == 0:
            print(f"{name:<12} {'(all NaN)':>14}")
            continue
        diff = np.abs(pt[valid] - ql[valid]).max()
        ok = diff < 1e-3
        if not ok:
            qlib_mismatch.append(name)
        print(f"{name:<12} {diff:>14.6e} {'OK' if ok else 'DIFF':>8}")
    print(f"\nPyTorch vs qlib: {len(subset)} factors compared, {len(qlib_mismatch)} mismatches")


if __name__ == "__main__":
    main()

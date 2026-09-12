#!/usr/bin/env python3
"""Fast GPU vs CPU (PyTorch) device-agnostic check for torch_factors.

Computes every implemented Alpha101 factor with both engines on a small
instrument set loaded from TDengine and compares values.
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_r1.data import load_ohlcv
from alpha_r1.backtest.torch_factors import compute_factors


def main():
    instruments = ["SH000688", "SH600769", "SH501001"]
    start, end = "2023-01-01", "2024-12-31"

    print(f"Loading OHLCV for {instruments} ...")
    panels = load_ohlcv(instruments, start, end)
    arrays = [panels[f] for f in ["close", "high", "low", "open", "volume", "vwap"]]
    print(f"Data shape: {panels['close'].shape} (T x N)")

    print("\n=== GPU vs CPU (PyTorch) ===")
    F_gpu = compute_factors(*(torch.as_tensor(a, device="cuda") for a in arrays), device="cuda")
    F_cpu = compute_factors(*(torch.as_tensor(a, device="cpu") for a in arrays), device="cpu")
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


if __name__ == "__main__":
    main()

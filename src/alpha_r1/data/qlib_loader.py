"""qlib-backed data loader (legacy, for backward compatibility)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_ohlcv(instruments: list[str], start: str, end: str):
    """Load OHLCV from qlib binary format."""
    from alpha_r1.backtest.data import init_qlib
    import os
    qlib_dir = os.path.expanduser("~/.qlib/qlib_data/alpha_r1")
    init_qlib(qlib_dir, kernels=1)
    from qlib.data import D

    fields = ["$open", "$high", "$low", "$close", "$volume", "$amount"]
    df = D.features(instruments, fields, start_time=start, end_time=end)
    calendar = pd.DatetimeIndex(sorted(df.index.get_level_values("datetime").unique()))
    N = len(instruments)
    T = len(calendar)
    panels = {}
    for f in fields:
        pivot = df[f].unstack(level="instrument")
        pivot = pivot.reindex(index=calendar, columns=instruments)
        panels[f.lstrip("$")] = pivot.values.astype(np.float32)
    # vwap
    vol = panels["volume"]
    amt = panels["amount"]
    with np.errstate(divide="ignore", invalid="ignore"):
        panels["vwap"] = np.where(vol > 0, amt / vol, np.nan).astype(np.float32)
    panels["calendar"] = calendar
    panels["instruments"] = instruments
    return panels


def load_calendar(start: str, end: str):
    from alpha_r1.backtest.data import init_qlib
    import os
    qlib_dir = os.path.expanduser("~/.qlib/qlib_data/alpha_r1")
    init_qlib(qlib_dir, kernels=1)
    from qlib.data import D
    return pd.DatetimeIndex(D.calendar(start_time=start, end_time=end, freq="day"))


def load_instruments(market: str = "all") -> list[str]:
    from alpha_r1.backtest.data import init_qlib
    import os
    qlib_dir = os.path.expanduser("~/.qlib/qlib_data/alpha_r1")
    init_qlib(qlib_dir, kernels=1)
    from qlib.data import D
    inst = D.instruments(market=market)
    if isinstance(inst, dict):
        # flatten
        all_inst = []
        for v in inst.values():
            all_inst.extend(v)
        return sorted(set(all_inst))
    return list(inst)

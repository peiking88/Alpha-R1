"""Unified data loading interface — TDengine direct / Parquet / qlib.

Switch backends via ``source`` argument or ``ALPHA_R1_DATA_SOURCE`` env var.
"""

from __future__ import annotations

import os
from typing import Literal

Source = Literal["tdengine", "parquet", "qlib"]

_DEFAULT_SOURCE = os.environ.get("ALPHA_R1_DATA_SOURCE", "tdengine")


def load_ohlcv(
    instruments: list[str],
    start: str,
    end: str,
    source: Source = _DEFAULT_SOURCE,
) -> dict[str, list[str] | list | list[str]]:
    """Load OHLCV + vwap for the given instruments and date window.

    Returns:
        ``{"close": array, "high": array, "low": array, "open": array,
          "volume": array, "vwap": array, "calendar": DatetimeIndex,
          "instruments": list[str]}``
        Arrays are ``float32 [T, N]`` aligned to a common calendar.
    """
    if source == "tdengine":
        from .tdx_loader import load_ohlcv as _load
    elif source == "parquet":
        from .parquet_loader import load_ohlcv as _load
    elif source == "qlib":
        from .qlib_loader import load_ohlcv as _load
    else:
        raise ValueError(f"unknown data source: {source}")
    return _load(instruments, start, end)


def load_calendar(start: str, end: str, source: Source = _DEFAULT_SOURCE):
    """Return a ``DatetimeIndex`` of trading days in [start, end)."""
    if source == "tdengine":
        from .tdx_loader import load_calendar as _load
    elif source == "parquet":
        from .parquet_loader import load_calendar as _load
    elif source == "qlib":
        from .qlib_loader import load_calendar as _load
    else:
        raise ValueError(f"unknown data source: {source}")
    return _load(start, end)


def load_instruments(market: str = "all", source: Source = _DEFAULT_SOURCE) -> list[str]:
    """Return instrument list (with SH/SZ/BJ prefix)."""
    if source == "tdengine":
        from .tdx_loader import load_instruments as _load
    elif source == "parquet":
        from .parquet_loader import load_instruments as _load
    elif source == "qlib":
        from .qlib_loader import load_instruments as _load
    else:
        raise ValueError(f"unknown data source: {source}")
    return _load(market)


def load_realtime_bar(instrument: str, source: Source = _DEFAULT_SOURCE) -> dict:
    """Load the latest bar for real-time use."""
    if source == "tdengine":
        from .tdx_loader import load_realtime_bar as _load
    else:
        raise ValueError(f"source {source} does not support realtime bar")
    return _load(instrument)

"""Parquet offline data loader (placeholder)."""

from __future__ import annotations


def load_ohlcv(instruments: list[str], start: str, end: str):
    raise NotImplementedError("Parquet loader not yet implemented")


def load_calendar(start: str, end: str):
    raise NotImplementedError("Parquet loader not yet implemented")


def load_instruments(market: str = "all") -> list[str]:
    raise NotImplementedError("Parquet loader not yet implemented")

"""Shared schema constants for data loading."""

import os

# ---------------------------------------------------------------------------
# TDengine connection
# ---------------------------------------------------------------------------
TDENGINE_URL = os.environ.get("TDENGINE_URL", "taosws://root:taosdata@localhost:6041")
TDENGINE_DB = os.environ.get("TDENGINE_DB", "tdx")

# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------
OHLCV_FIELDS = ["open", "high", "low", "close", "volume", "amount"]
FIELD_TO_QLIB = {
    "open": "$open",
    "high": "$high",
    "low": "$low",
    "close": "$close",
    "volume": "$volume",
    "amount": "$amount",
    "vwap": "$vwap",
}

# ---------------------------------------------------------------------------
# Instrument classification (通达信 / TDengine conventions)
# ---------------------------------------------------------------------------
# market prefix in TDengine tbname: sh / sz / bj
# instrument id in our code: SH / SZ / BJ (uppercase)


def code_to_instrument(market: str, code: str) -> str:
    """``('sh', '600000')`` → ``'SH600000'``."""
    return f"{market.upper()}{code}"


def instrument_to_code(inst: str) -> tuple[str, str]:
    """``'SH600000'`` → ``('sh', '600000')``."""
    return inst[:2].lower(), inst[2:]


def table_name(market: str, code: str, cycle: str = "1d") -> str:
    """TDengine sub-table name: ``k_sh600000_1d``."""
    return f"k_{market}{code}_{cycle}"

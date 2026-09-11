"""Custom qlib expression operators for WorldQuant Alpha101 semantics.

Loaded by qlib through its ``custom_ops`` config mechanism, so every class
must live at module level (qlib resolves ``{"class": ..., "module_path": ...}``
via getattr on the module, including in spawned worker processes).

Differences from qlib built-ins:
- ``CSRank`` / ``CSScale`` are cross-sectional (per day across instruments);
  qlib's ``Rank`` is time-series and matches WorldQuant ``ts_rank``.
- ``TsArgMax`` / ``TsArgMin`` follow the WorldQuant convention: days since the
  window extremum occurred (1 = today).
- ``DecayLinear`` weights the window linearly (oldest weight 1, newest N).

Cross-sectional ops evaluate the inner expression for the whole universe via
``D.features``; this requires single-process evaluation (``kernels=1``), which
``init_qlib`` sets by default.
"""

import numpy as np
import pandas as pd
from qlib.data.base import Expression, ExpressionOps

_CS_UNIVERSE = "all"
_PANEL_CACHE: dict = {}


def set_cs_universe(market: str) -> None:
    global _CS_UNIVERSE
    _CS_UNIVERSE = market


def clear_panel_cache() -> None:
    _PANEL_CACHE.clear()


def _load_panel(feature, start_index: int, end_index: int, freq: str) -> pd.DataFrame:
    """Evaluate an expression for the whole universe; datetime x instrument.

    The panel is reindexed to the exact calendar slice so that row i always
    corresponds to calendar position ``start_index + i`` even when some dates
    have no in-span instruments.
    """
    from qlib.data import D

    key = (str(feature), start_index, end_index, freq)
    if key not in _PANEL_CACHE:
        cal = D.calendar(freq=freq)
        end_index = min(end_index, len(cal) - 1)  # right-extended exprs may exceed the calendar
        instruments = D.instruments(market=_CS_UNIVERSE)
        df = D.features(instruments, [str(feature)],
                        start_time=cal[start_index], end_time=cal[end_index], freq=freq)
        panel = df.iloc[:, 0].unstack(level="instrument")
        _PANEL_CACHE[key] = panel.reindex(pd.DatetimeIndex(cal[start_index:end_index + 1]))
    return _PANEL_CACHE[key]


class _CSOp(ExpressionOps):
    """Base for cross-sectional ops: panel transform + RangeIndex reassembly."""

    def __init__(self, feature):
        self.feature = feature

    def __neg__(self):
        # Support unary minus in expressions like `-1 * Delta(...)` which qlib's
        # parser may rewrite as `-(...)` applied to the op instance.
        return _NegOp(self)

    def __str__(self):
        return f"{type(self).__name__}({self.feature})"

    def _transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def _load_internal(self, instrument, start_index, end_index, *args):
        panel = self._transform(_load_panel(self.feature, start_index, end_index, *args))
        if instrument in panel.columns:
            values = panel[instrument].to_numpy()
        else:
            values = np.full(len(panel.index), np.nan)
        return pd.Series(values, index=pd.RangeIndex(start_index, start_index + len(values)))

    def get_longest_back_rolling(self):
        return self.feature.get_longest_back_rolling()

    def get_extended_window_size(self):
        return self.feature.get_extended_window_size()


class CSRank(_CSOp):
    """Cross-sectional percentile rank (WorldQuant ``rank``)."""

    def _transform(self, panel):
        return panel.rank(axis=1, pct=True)


class CSScale(_CSOp):
    """Cross-sectional scaling to sum(abs(x)) == 1 (WorldQuant ``scale``)."""

    def _transform(self, panel):
        return panel.div(panel.abs().sum(axis=1).replace(0, np.nan), axis=0)


class _RollingWindowOp(ExpressionOps):
    """Base for rolling-window ops with a scalar window (floats rounded to ints)."""

    def __init__(self, feature, N):
        self.feature = feature
        self.N = max(1, int(round(float(N))))

    def __str__(self):
        return f"{type(self).__name__}({self.feature},{self.N})"

    def get_longest_back_rolling(self):
        return self.feature.get_longest_back_rolling() + self.N - 1

    def get_extended_window_size(self):
        lft, rght = self.feature.get_extended_window_size()
        return lft + self.N - 1, rght


class DecayLinear(_RollingWindowOp):
    """Linearly decaying weighted moving average (WorldQuant ``decay_linear``).

    The first ``N - 1`` values use partial windows (qlib house style), and
    NaN inputs propagate into the weighted sum.
    """

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        weights = np.arange(1, self.N + 1, dtype=float)

        def _wavg(window):
            w = weights[-len(window):]
            return float(np.dot(window, w) / w.sum())

        return series.rolling(self.N, min_periods=1).apply(_wavg, raw=True)


class TsArgMax(_RollingWindowOp):
    """Days since the N-day maximum occurred (1 = today); partial windows at the start."""

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        return series.rolling(self.N, min_periods=1).apply(
            lambda w: float(len(w) - np.argmax(w)), raw=True)


class TsArgMin(_RollingWindowOp):
    """Days since the N-day minimum occurred (1 = today); partial windows at the start."""

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        return series.rolling(self.N, min_periods=1).apply(
            lambda w: float(len(w) - np.argmin(w)), raw=True)


class _NegOp(ExpressionOps):
    """Unary minus wrapper: ``-X`` for any expression op."""

    def __init__(self, feature):
        self.feature = feature

    def __neg__(self):
        return self.feature  # --x == x

    def __str__(self):
        return f"-{self.feature}"

    def _load_internal(self, instrument, start_index, end_index, *args):
        return -self.feature.load(instrument, start_index, end_index, *args)

    def get_longest_back_rolling(self):
        return self.feature.get_longest_back_rolling()

    def get_extended_window_size(self):
        return self.feature.get_extended_window_size()


class SignedPower(ExpressionOps):
    """WorldQuant ``SignedPower(x, a)`` = sign(x) * |x| ** a; a may be an expression."""

    def __init__(self, feature, exponent):
        self.feature = feature
        self.exponent = exponent

    def __str__(self):
        return f"SignedPower({self.feature},{self.exponent})"

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if isinstance(self.exponent, Expression):
            exp = self.exponent.load(instrument, start_index, end_index, *args)
        else:
            exp = float(self.exponent)
        return np.sign(series) * np.abs(series) ** exp

    def get_longest_back_rolling(self):
        left = self.feature.get_longest_back_rolling()
        if isinstance(self.exponent, Expression):
            return max(left, self.exponent.get_longest_back_rolling())
        return left

    def get_extended_window_size(self):
        lft, rght = self.feature.get_extended_window_size()
        if isinstance(self.exponent, Expression):
            el, er = self.exponent.get_extended_window_size()
            return max(lft, el), max(rght, er)
        return lft, rght


CUSTOM_OPS = [CSRank, CSScale, DecayLinear, SignedPower, TsArgMax, TsArgMin]


# ---------------------------------------------------------------------------
# Patch qlib's built-in ExpressionOps to support unary minus.
# Alpha101 expressions use ``-1 * Op(...)``; qlib's parser can rewrite that
# as a unary minus on the op instance.  Built-in ops (Delta, Sign, ...) don't
# implement ``__neg__``, which raises
# ``TypeError: bad operand type for unary -: 'Delta'``.  Monkey-patching the
# base class fixes every built-in op at once.
# ---------------------------------------------------------------------------
def _patch_builtin_neg() -> None:
    from qlib.data.base import ExpressionOps

    if getattr(ExpressionOps, "__neg__", None) is not None:
        return

    def _neg(self):
        return _NegOp(self)

    ExpressionOps.__neg__ = _neg  # type: ignore[attr-defined]


_patch_builtin_neg()

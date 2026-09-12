"""Fixed linear return model: beta estimation and inference-time scoring.

Follows the paper's fixed-linear-model specification (Appendix F): OLS on a
pooled cross-sectional panel over a historical window (2020-2023), with the
forward H-day stock return as the dependent variable. Factor values are
cross-sectionally winsorized (1%/99%), z-scored and median-filled per trading
day before entering the regression, and the same transform is applied at
inference time. Only selected factors contribute to the score.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def winsorize_zscore(s: pd.Series, q: float = 0.01) -> pd.Series:
    """Winsorize at the q/(1-q) quantiles, z-score, then median-fill NaN."""
    s = s.clip(s.quantile(q), s.quantile(1 - q))
    std = s.std()
    z = (s - s.mean()) / std if std > 0 else s * 0.0
    return z.fillna(z.median()).fillna(0.0)


def standardize_panel(panel: pd.DataFrame, q: float = 0.01) -> pd.DataFrame:
    """Row-wise (per trading day) winsorize + z-score + median fill.

    ``panel`` is datetime x instrument.
    """
    return panel.apply(lambda row: winsorize_zscore(row, q), axis=1)


def estimate_betas(factor_panels: dict[str, pd.DataFrame],
                   fwd_returns: pd.DataFrame) -> tuple[pd.Series, float]:
    """Pooled cross-sectional OLS of forward returns on standardized factors.

    Args:
        factor_panels: factor name -> datetime x instrument raw values.
        fwd_returns: datetime x instrument forward H-day returns.

    Returns:
        (betas, intercept): coefficients indexed by factor name.
    """
    names = sorted(factor_panels)
    X = pd.DataFrame({n: standardize_panel(factor_panels[n]).stack() for n in names})
    y = fwd_returns.stack().rename("y")
    data = X.join(y, how="inner").dropna(subset=["y"])
    if len(data) < len(names) + 10:
        raise ValueError(f"too few pooled observations ({len(data)}) to estimate {len(names)} betas")
    A = np.column_stack([np.ones(len(data)), data[names].to_numpy()])
    coef, *_ = np.linalg.lstsq(A, data["y"].to_numpy(), rcond=None)
    return pd.Series(coef[1:], index=names), float(coef[0])


def score_stocks(day_values: pd.DataFrame, selected: list[str],
                 betas: pd.Series, intercept: float = 0.0) -> pd.Series | None:
    """Score one cross-section: dot(beta, standardized factor values) + intercept.

    ``day_values`` is instrument x factor raw values for a single day.
    Unselected factors contribute zero weight. Returns None when none of the
    selected factors is available.
    """
    available = [f for f in selected if f in day_values.columns and f in betas.index]
    if not available:
        return None
    z = day_values[available].apply(winsorize_zscore)
    return z.mul(betas[available], axis=1).sum(axis=1) + intercept


def save_betas(betas: pd.Series, intercept: float, path: str) -> None:
    """Write the CSV consumed by strategy backtests and training/reward.py."""
    out = pd.DataFrame({"factor": list(betas.index) + ["_intercept"],
                        "beta": list(betas.values) + [intercept]})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def load_betas(path: str) -> tuple[pd.Series, float]:
    """Inverse of :func:`save_betas`; returns (betas, intercept)."""
    df = pd.read_csv(path).set_index("factor")["beta"]
    intercept = float(df.pop("_intercept")) if "_intercept" in df.index else 0.0
    return df, intercept

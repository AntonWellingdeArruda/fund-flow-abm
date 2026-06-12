"""Reshape the long modeling frame into matrices the models consume.

The data layer (Phase 1) emits a long, lag-disciplined frame: one row per
(period, category). VAR and the baselines want a WIDE flow matrix
(periods × categories). Exogenous predictors are the LAGGED macro columns,
already shifted by the feature layer so there is no contemporaneous leakage.
"""
from __future__ import annotations

import pandas as pd

# Lag columns that are NOT exogenous macro regressors: own-flow AR lags and the
# latent regime label (kept out of exog because regime is unobserved in production).
_NON_EXOG_LAGS = frozenset({
    "net_flow_brl_lag1", "redemption_gross_brl_lag1", "regime_lag1",
})


def exog_columns(frame: pd.DataFrame) -> list[str]:
    """Discover lagged-macro exogenous regressors present in the frame.

    Generic across data sources: any '*_lag1' column that is not an own-flow
    AR lag or the regime label. Synthetic supplies the full macro set; a real
    adapter (e.g. BCB) may supply a subset — both work unchanged.
    """
    return [
        c for c in frame.columns
        if c.endswith("_lag1") and c not in _NON_EXOG_LAGS
    ]


def wide_flows(frame: pd.DataFrame, value: str = "net_flow_brl") -> pd.DataFrame:
    """Pivot to a (period × category) matrix of the chosen flow column.

    Index is a monthly PeriodIndex (sorted); columns are categories.
    """
    wide = frame.pivot(index="period", columns="category", values=value)
    wide.index = pd.PeriodIndex(wide.index, freq="M")
    wide = wide.sort_index()
    wide.columns.name = None
    return wide


def wide_exog(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Per-period matrix of lagged macro exogenous regressors.

    Macro is identical across categories within a period, so we take the
    first row per period. Index aligns with wide_flows().
    """
    cols = columns or exog_columns(frame)
    macro = (
        frame[["period", *cols]]
        .drop_duplicates(subset="period")
        .set_index("period")
    )
    macro.index = pd.PeriodIndex(macro.index, freq="M")
    return macro.sort_index()

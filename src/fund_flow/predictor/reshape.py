"""Reshape the long modeling frame into matrices the models consume.

The data layer (Phase 1) emits a long, lag-disciplined frame: one row per
(period, category). VAR and the baselines want a WIDE flow matrix
(periods × categories). Exogenous predictors are the LAGGED macro columns,
already shifted by the feature layer so there is no contemporaneous leakage.
"""
from __future__ import annotations

import pandas as pd

# Lagged macro columns used as exogenous regressors (all already _lag1).
EXOG_COLUMNS = [
    "selic_rate_lag1", "delta_selic_lag1", "ipca_monthly_lag1",
    "ibovespa_return_lag1", "dxy_return_lag1", "ust_10y_lag1",
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
    cols = columns or EXOG_COLUMNS
    macro = (
        frame[["period", *cols]]
        .drop_duplicates(subset="period")
        .set_index("period")
    )
    macro.index = pd.PeriodIndex(macro.index, freq="M")
    return macro.sort_index()

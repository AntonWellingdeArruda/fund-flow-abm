"""Walk-forward, out-of-sample backtest (CLAUDE.md §7).

Expanding window: for each test period t from `min_train` onward, fit each
model on flows[:t] (+ aligned exog) and forecast flow[t]. Errors are pooled
across categories and periods. NO in-sample evaluation, ever.

A candidate model is acceptable only if it beats BOTH naive baselines
(random walk, AR(1)) on out-of-sample RMSE.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fund_flow.predictor.base import Predictor


@dataclass
class BacktestResult:
    name: str
    rmse: float
    mae: float
    n_forecasts: int
    errors: pd.DataFrame  # period × category forecast errors (actual − pred)

    def summary_row(self) -> dict:
        return {"model": self.name, "rmse": self.rmse,
                "mae": self.mae, "n_forecasts": self.n_forecasts}


def walk_forward(
    model_factory,
    flows: pd.DataFrame,        # (period × category), PeriodIndex
    exog: pd.DataFrame | None,  # (period × lagged-macro), aligned to flows
    min_train: int,
    name: str | None = None,
    window: int | None = None,
    test_start: int | None = None,
    test_end: int | None = None,
) -> BacktestResult:
    """Run a walk-forward, out-of-sample backtest for one model.

    model_factory: zero-arg callable returning a fresh Predictor each step
    (so no state leaks between training windows).

    window: if None, the training set EXPANDS from the start (flows[:t]); if an
    int, training uses only the most recent `window` months (a ROLLING window,
    flows[t-window:t]). A rolling window discards old data, which can help when
    the flow process shifts across regimes (e.g. high- vs low-Selic) and hurt
    when more history simply means a better-estimated model.

    test_start / test_end: restrict the FORECAST points to that index range
    (still training only on data strictly before each point). Lets you score a
    single sub-period — e.g. only forecasts made during a high-rate regime —
    without leaking future data into training.
    """
    flows = flows.sort_index()
    if exog is not None:
        exog = exog.reindex(flows.index)

    T = len(flows)
    if min_train >= T:
        raise ValueError(f"min_train={min_train} too large for {T} periods")

    lo = max(min_train, test_start if test_start is not None else min_train)
    hi = min(T, test_end if test_end is not None else T)

    err_rows = []
    for t in range(lo, hi):
        train_lo = max(0, t - window) if window else 0
        # A rolling window must still respect the minimum training length.
        if window and t - train_lo < min_train:
            train_lo = max(0, t - min_train)
        train_flows = flows.iloc[train_lo:t]
        train_exog = exog.iloc[train_lo:t] if exog is not None else None
        next_exog = exog.iloc[t] if exog is not None else None

        model = model_factory()
        model.fit(train_flows, train_exog)
        pred = model.predict_next(next_exog)

        actual = flows.iloc[t]
        err = actual - pred.reindex(actual.index)
        err_rows.append(err.rename(flows.index[t]))

    errors = pd.DataFrame(err_rows)
    flat = errors.to_numpy().ravel()
    rmse = float(np.sqrt(np.mean(flat ** 2)))
    mae = float(np.mean(np.abs(flat)))
    label = name or getattr(model_factory(), "name", "model")
    return BacktestResult(label, rmse, mae, len(flat), errors)


def compare(results: list[BacktestResult]) -> pd.DataFrame:
    """Tidy comparison table sorted by RMSE (best first)."""
    df = pd.DataFrame([r.summary_row() for r in results])
    return df.sort_values("rmse").reset_index(drop=True)


def per_category_rmse(result: BacktestResult) -> pd.Series:
    """Out-of-sample RMSE per category (one value per flow column).

    The pooled RMSE can hide where a predictor actually helps. A credit-spread
    signal, for instance, should improve Crédito Privado / Renda Fixa far more
    than the 6-category average, so we slice the same OOS errors by category.
    """
    return np.sqrt((result.errors ** 2).mean(axis=0)).rename(result.name)


def per_category_table(results: list[BacktestResult]) -> pd.DataFrame:
    """category × model table of OOS RMSE (column per model)."""
    return pd.concat([per_category_rmse(r) for r in results], axis=1)

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
) -> BacktestResult:
    """Run an expanding-window backtest for one model.

    model_factory: zero-arg callable returning a fresh Predictor each step
    (so no state leaks between training windows).
    """
    flows = flows.sort_index()
    if exog is not None:
        exog = exog.reindex(flows.index)

    T = len(flows)
    if min_train >= T:
        raise ValueError(f"min_train={min_train} too large for {T} periods")

    err_rows = []
    for t in range(min_train, T):
        train_flows = flows.iloc[:t]
        train_exog = exog.iloc[:t] if exog is not None else None
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

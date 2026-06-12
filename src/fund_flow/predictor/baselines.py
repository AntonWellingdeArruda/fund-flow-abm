"""Naive baselines — the bars the real predictor MUST beat (CLAUDE.md §7).

A model that cannot out-of-sample beat these is not adding value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fund_flow.predictor.base import Predictor


class RandomWalkPredictor(Predictor):
    """Forecast next flow = last observed flow, per category."""

    name = "random_walk"

    def fit(self, flows, exog=None):
        self._last = flows.iloc[-1].copy()
        return self

    def predict_next(self, next_exog=None):
        return self._last.copy()


class AR1Predictor(Predictor):
    """Per-category AR(1): flow[t] = c + phi * flow[t-1].

    Fitted independently per category by closed-form OLS. Categories are
    modeled separately here on purpose — the JOINT model is the VAR; AR(1)
    is the univariate baseline it must beat.
    """

    name = "ar1"

    def fit(self, flows, exog=None):
        self._params: dict[str, tuple[float, float]] = {}
        self._last: dict[str, float] = {}
        for col in flows.columns:
            y = flows[col].to_numpy()
            x_prev = y[:-1]
            y_next = y[1:]
            # OLS of y_next on [1, x_prev]
            X = np.column_stack([np.ones_like(x_prev), x_prev])
            coef, *_ = np.linalg.lstsq(X, y_next, rcond=None)
            self._params[col] = (float(coef[0]), float(coef[1]))
            self._last[col] = float(y[-1])
        self._columns = list(flows.columns)
        return self

    def predict_next(self, next_exog=None):
        out = {}
        for col in self._columns:
            c, phi = self._params[col]
            out[col] = c + phi * self._last[col]
        return pd.Series(out)

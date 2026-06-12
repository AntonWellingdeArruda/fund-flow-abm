"""Joint VAR(X) across the 6 Anbima categories.

Each category's next flow is regressed on the lagged flows of ALL categories
(capturing rotation between categories, CLAUDE.md §3.3) plus lagged macro
exogenous regressors. Estimated equation-by-equation with a standardized
ridge penalty — "regularized linear" per §3.7, since the sample is small and
OLS overfits a 6-equation system with this many regressors.

Lag discipline: exog rows are the data layer's *_lag1 macro, i.e. values
realized at t-1 used to predict flow[t] — no contemporaneous leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fund_flow.predictor.base import Predictor


class VARXPredictor(Predictor):
    name = "varx"

    def __init__(self, order: int = 1, ridge_alpha: float = 1.0):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.ridge_alpha = ridge_alpha

    def _build_design(
        self, flows: np.ndarray, exog: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, Y) for rows t = order .. T-1."""
        T, k = flows.shape
        p = self.order
        rows_X, rows_Y = [], []
        for t in range(p, T):
            feat = []
            for lag in range(1, p + 1):
                feat.extend(flows[t - lag])
            if exog is not None:
                feat.extend(exog[t])
            rows_X.append(feat)
            rows_Y.append(flows[t])
        return np.asarray(rows_X, float), np.asarray(rows_Y, float)

    def fit(self, flows, exog=None):
        self._columns = list(flows.columns)
        self._exog_cols = list(exog.columns) if exog is not None else []
        flows_arr = flows.to_numpy(float)
        exog_arr = exog.to_numpy(float) if exog is not None else None

        X, Y = self._build_design(flows_arr, exog_arr)

        # Standardize features and targets so the ridge penalty is comparable.
        self._xbar = X.mean(axis=0)
        self._xstd = X.std(axis=0)
        self._xstd[self._xstd == 0] = 1.0
        self._ybar = Y.mean(axis=0)
        self._ystd = Y.std(axis=0)
        self._ystd[self._ystd == 0] = 1.0

        Xs = (X - self._xbar) / self._xstd
        Ys = (Y - self._ybar) / self._ystd

        n_features = Xs.shape[1]
        gram = Xs.T @ Xs + self.ridge_alpha * np.eye(n_features)
        self._beta = np.linalg.solve(gram, Xs.T @ Ys)  # (n_features × k)

        # Keep the last `order` flow rows for one-step-ahead forecasting.
        self._history = flows_arr[-self.order:].copy()
        return self

    def predict_next(self, next_exog=None):
        feat = []
        # most-recent lag first (flow[t-1], flow[t-2], ...)
        for lag in range(1, self.order + 1):
            feat.extend(self._history[-lag])
        if self._exog_cols:
            if next_exog is None:
                raise ValueError("VARXPredictor was fit with exog; next_exog required")
            feat.extend(float(next_exog[c]) for c in self._exog_cols)

        x = np.asarray(feat, float)
        xs = (x - self._xbar) / self._xstd
        ys = xs @ self._beta
        y = self._ybar + ys * self._ystd
        return pd.Series(y, index=self._columns)

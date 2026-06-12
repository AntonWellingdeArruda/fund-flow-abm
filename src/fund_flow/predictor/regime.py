"""Regime-switching predictor (CLAUDE.md §3.4) — added AFTER the baseline.

Per-category 2-regime Markov-switching AR(1) with regime-specific means
(Selic high vs low behave differently). statsmodels has no Markov-switching
*VAR*, so the regime model is univariate per category; the joint model is the
VARX. One-step forecast marginalizes over the predicted next-period regime:

    E[y_{t+1}] = Σ_j  P(regime_{t+1}=j | data_t) · (const_j + φ · y_t)

If EM fitting fails to converge for a category, that category falls back to a
closed-form AR(1) so the backtest stays robust.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_autoregression import (
    MarkovAutoregression,
)

from fund_flow.predictor.base import Predictor


def _ar1_fallback(y: np.ndarray) -> tuple[float, float, float]:
    """Closed-form AR(1): returns (const, phi, last)."""
    x_prev, y_next = y[:-1], y[1:]
    X = np.column_stack([np.ones_like(x_prev), x_prev])
    coef, *_ = np.linalg.lstsq(X, y_next, rcond=None)
    return float(coef[0]), float(coef[1]), float(y[-1])


class MarkovRegimePredictor(Predictor):
    name = "markov_regime"

    def __init__(self, k_regimes: int = 2):
        self.k_regimes = k_regimes

    def _forecast_one(self, y: np.ndarray) -> float:
        """One-step-ahead forecast for a single category series.

        statsmodels MarkovAutoregression uses Hamilton's mean-deviation form:
            (y_t - mu_{S_t}) = phi (y_{t-1} - mu_{S_{t-1}}) + e_t
        so the one-step conditional mean depends on both the current and the
        previous regime, marginalized with the filtered + transition probs.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = MarkovAutoregression(
                    y, k_regimes=self.k_regimes, order=1, trend="c",
                    switching_ar=False, switching_trend=True,
                )
                res = model.fit(disp=False)

            params = dict(zip(res.model.param_names, np.asarray(res.params)))
            mu = [params[f"const[{j}]"] for j in range(self.k_regimes)]
            ar = params["ar.L1"]

            # trans[i, j] = P(S_{t+1}=i | S_t=j); columns sum to 1
            trans = np.asarray(res.regime_transition).squeeze()
            filt = np.asarray(res.filtered_marginal_probabilities[-1])  # P(S_t=j | data)
            last = y[-1]

            e = 0.0
            for i in range(self.k_regimes):
                for j in range(self.k_regimes):
                    e += trans[i, j] * filt[j] * (mu[i] + ar * (last - mu[j]))
            return float(e)
        except Exception:
            c, phi, last = _ar1_fallback(y)
            return c + phi * last

    def fit(self, flows, exog=None):
        self._columns = list(flows.columns)
        self._series = {c: flows[c].to_numpy(float) for c in self._columns}
        return self

    def predict_next(self, next_exog=None):
        out = {c: self._forecast_one(self._series[c]) for c in self._columns}
        return pd.Series(out)

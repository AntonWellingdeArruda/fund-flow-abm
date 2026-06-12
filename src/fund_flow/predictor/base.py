"""Common predictor interface for the walk-forward backtest.

Every model fits on a wide flow matrix (periods × categories) plus optional
lagged exogenous macro, and produces a one-step-ahead forecast: a Series
indexed by category for the period immediately after the training window.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Predictor(ABC):
    name: str = "predictor"

    @abstractmethod
    def fit(
        self,
        flows: pd.DataFrame,            # (period × category)
        exog: pd.DataFrame | None = None,  # (period × lagged-macro), aligned to flows
    ) -> "Predictor":
        ...

    @abstractmethod
    def predict_next(self, next_exog: pd.Series | None = None) -> pd.Series:
        """One-step-ahead forecast (category → net_flow_brl).

        next_exog carries the lagged-macro row for the forecast period
        (already known at fit time, since it is lagged). Models without
        exogenous inputs ignore it.
        """
        ...

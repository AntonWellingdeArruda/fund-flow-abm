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

    def coefficients(self) -> pd.DataFrame:
        """Standardized ridge weights: index = predictor, columns = category.

        Because X and Y were standardized before fitting, magnitudes are directly
        comparable ACROSS predictors and categories — e.g. you can read off
        whether Crédito Privado loads more on fed_funds than on sp500_return.
        These are the weights the model LEARNED; they are not set by hand.
        """
        names: list[str] = []
        for lag in range(1, self.order + 1):
            names.extend(f"{c}_flow_lag{lag}" for c in self._columns)
        names.extend(self._exog_cols)
        return pd.DataFrame(self._beta, index=names, columns=self._columns)

    def _predict_from_design(self, X: np.ndarray) -> np.ndarray:
        """Predicted (n × k) flows for a raw design matrix X (n × n_features)."""
        xs = (X - self._xbar) / self._xstd
        return self._ybar + (xs @ self._beta) * self._ystd

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


class VARXCVPredictor(Predictor):
    """VARX whose ridge α is chosen by NESTED cross-validation.

    On each fit, the last `val_months` of the *training* window are held out as
    an inner validation block; α is the value minimizing inner-validation RMSE,
    then the model is refit on the full training window with that α. Because α is
    selected using only training data, the outer walk-forward test points stay
    untouched — this is the honest way to "search for the best weighting" without
    snooping the test set (CLAUDE.md §7).
    """

    name = "varx_cv"
    DEFAULT_ALPHAS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0)

    def __init__(self, order: int = 1, alphas=None, val_months: int = 24):
        self.order = order
        self.alphas = tuple(alphas) if alphas else self.DEFAULT_ALPHAS
        self.val_months = val_months
        self.chosen_alpha_: float | None = None

    def fit(self, flows, exog=None):
        T = len(flows)
        v = min(self.val_months, max(6, T // 5))
        inner_T = T - v
        # Not enough inner-train history → skip CV, use a sensible default.
        if inner_T <= self.order + 2:
            self.chosen_alpha_ = 1.0
        else:
            f_arr = flows.to_numpy(float)
            e_arr = exog.to_numpy(float) if exog is not None else None
            best_rmse, best_a = np.inf, 1.0
            for a in self.alphas:
                m = VARXPredictor(self.order, a).fit(
                    flows.iloc[:inner_T],
                    exog.iloc[:inner_T] if exog is not None else None,
                )
                X_all, Y_all = m._build_design(f_arr, e_arr)  # rows order..T-1
                X_val, Y_val = X_all[-v:], Y_all[-v:]
                pred = m._predict_from_design(X_val)
                rmse = float(np.sqrt(np.mean((Y_val - pred) ** 2)))
                if rmse < best_rmse:
                    best_rmse, best_a = rmse, a
            self.chosen_alpha_ = best_a

        self._model = VARXPredictor(self.order, self.chosen_alpha_).fit(flows, exog)
        self._columns = self._model._columns
        return self

    def predict_next(self, next_exog=None):
        return self._model.predict_next(next_exog)

    def coefficients(self) -> pd.DataFrame:
        return self._model.coefficients()


class PerCategorySelectPredictor(Predictor):
    """Per-category model selection by inner validation.

    Candidates per category: AR(1), VARX-CV, and — when Phase-2.6 `signals` are
    supplied — VARX-CV-excess. For each category the candidate with lowest
    inner-validation RMSE is chosen; inner validation holds out the last
    `val_months` of the training window (the same split VARXCVPredictor uses for
    its α search), so the outer OOS test points are never touched during selection.

    After selection each sub-model is refit on the full training window, then
    predict_next calls the per-category winner.
    """

    name = "per_cat_select"

    def __init__(self, order: int = 1, val_months: int = 24, signals: dict | None = None):
        self.order = order
        self.val_months = val_months
        self.signals = signals or {}
        self.selected_: dict[str, str] = {}  # category → "ar1" | "varxcv" | "excess"

    def fit(self, flows, exog=None):
        from fund_flow.predictor.baselines import AR1Predictor  # avoid circular import

        self._columns = list(flows.columns)
        T = len(flows)
        v = min(self.val_months, max(6, T // 5))
        inner_T = T - v

        if inner_T <= self.order + 2:
            # Not enough inner history — default to VARXCV for all categories
            self.selected_ = {c: "varxcv" for c in self._columns}
        else:
            inner_exog = exog.iloc[:inner_T] if exog is not None else None
            ar1_inner = AR1Predictor().fit(flows.iloc[:inner_T], None)
            varxcv_inner = VARXCVPredictor(self.order).fit(flows.iloc[:inner_T], inner_exog)

            # AR(1) validation predictions: apply the fitted (c, phi) to *actual*
            # lagged flows in the validation window (one-step-ahead, no refitting).
            flows_arr = flows.to_numpy(float)
            k = len(self._columns)
            pred_ar1 = np.zeros((v, k))
            for i, t in enumerate(range(inner_T, T)):
                prev = flows_arr[t - 1]
                for j, col in enumerate(self._columns):
                    c, phi = ar1_inner._params[col]
                    pred_ar1[i, j] = c + phi * prev[j]

            # VARXCV validation predictions: build the full design matrix, take val rows.
            exog_arr = exog.to_numpy(float) if exog is not None else None
            X_all, Y_all = varxcv_inner._model._build_design(flows_arr, exog_arr)
            Y_val = Y_all[-v:]
            pred_varxcv = varxcv_inner._model._predict_from_design(X_all[-v:])

            cand_rmse = {
                "ar1": np.sqrt(np.mean((Y_val - pred_ar1) ** 2, axis=0)),
                "varxcv": np.sqrt(np.mean((Y_val - pred_varxcv) ** 2, axis=0)),
            }
            if self.signals:
                excess_inner = VARXExcessCVPredictor(
                    self.order, signals=self.signals).fit(flows.iloc[:inner_T], inner_exog)
                val_periods = list(flows.index[inner_T:T])
                pred_ex = excess_inner.predict_rows(flows, exog, val_periods)[self._columns]
                cand_rmse["excess"] = np.sqrt(np.mean(
                    (Y_val - pred_ex.to_numpy()) ** 2, axis=0))

            self.selected_ = {}
            for j, col in enumerate(self._columns):
                self.selected_[col] = min(cand_rmse, key=lambda m: cand_rmse[m][j])

        # Refit the needed sub-models on the FULL training window.
        self._ar1 = AR1Predictor().fit(flows, None)
        self._varxcv = VARXCVPredictor(self.order).fit(flows, exog)
        self._excess = (VARXExcessCVPredictor(self.order, signals=self.signals).fit(flows, exog)
                        if self.signals else None)
        return self

    def predict_next(self, next_exog=None):
        preds = {
            "ar1": self._ar1.predict_next(None),
            "varxcv": self._varxcv.predict_next(next_exog),
        }
        if self._excess is not None:
            preds["excess"] = self._excess.predict_next(next_exog)
        out = {col: preds[self.selected_.get(col, "varxcv")][col] for col in self._columns}
        return pd.Series(out)


def _ridge_fit_1d(X: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    """Standardized ridge of one target on X. Returns the params for _ridge_pred_1d."""
    xbar = X.mean(axis=0)
    xstd = X.std(axis=0)
    xstd[xstd == 0] = 1.0
    ybar = float(y.mean())
    ystd = float(y.std()) or 1.0
    Xs = (X - xbar) / xstd
    ys = (y - ybar) / ystd
    gram = Xs.T @ Xs + alpha * np.eye(Xs.shape[1])
    beta = np.linalg.solve(gram, Xs.T @ ys)
    return {"beta": beta, "xbar": xbar, "xstd": xstd, "ybar": ybar, "ystd": ystd}


def _ridge_pred_1d(X: np.ndarray, p: dict) -> np.ndarray:
    return p["ybar"] + ((X - p["xbar"]) / p["xstd"] @ p["beta"]) * p["ystd"]


class VARXExcessCVPredictor(Predictor):
    """Joint ridge VAR augmented with each category's OWN excess-return signal.

    Each category c's flow equation is regressed on [all-category flow lags |
    macro exog | (c's own excess return at a per-category window w_c)]. The window
    w_c ∈ {none, 1m, 3m, 6m} and a system-wide ridge α are chosen by NESTED
    cross-validation inside each training window (the outer test point is never
    seen). When w_c = none for every category, this is exactly the base VARX-CV —
    so the "no category degrades" guardrail is structurally near-automatic and the
    Δ vs base is attributable to the excess feature (Phase 2.6).

    The per-category excess signals are passed via `signals` (closure), each a
    (period × category) DataFrame of the LAGGED excess return (excess_ret_*m_lag1).
    Signals may be NaN (Cambial, early periods, sparse benchmarks): a window is
    eligible for a category only if ≥ `min_valid` of its training rows are present,
    and remaining NaNs are imputed with the training-window mean (≈0, neutral).
    """

    name = "varx_excess_cv"
    DEFAULT_ALPHAS = VARXCVPredictor.DEFAULT_ALPHAS
    WINDOWS: tuple = (None, 1, 3, 6)

    def __init__(self, order: int = 1, signals: dict | None = None, alphas=None,
                 val_months: int = 24, min_valid: float = 0.6):
        self.order = order
        self.signals = signals or {}
        self.alphas = tuple(alphas) if alphas else self.DEFAULT_ALPHAS
        self.val_months = val_months
        self.min_valid = min_valid
        self.chosen_alpha_: float | None = None
        self.windows_: dict[str, int | None] = {}

    # -- helpers ---------------------------------------------------------- #
    def _aligned_signals(self, design_periods) -> dict[int, np.ndarray]:
        """{window: (n_design × k) array} of the excess signal on the design rows."""
        out = {}
        for w, df in self.signals.items():
            out[w] = (df.reindex(design_periods)[self._columns].to_numpy(float)
                      if df is not None else None)
        return out

    @staticmethod
    def _augment(Xbase: np.ndarray, col: np.ndarray | None, fill: float):
        if col is None:
            return Xbase
        c = np.where(np.isfinite(col), col, fill).reshape(-1, 1)
        return np.hstack([Xbase, c])

    def _candidate_cols(self, sig: dict, w, j, rows):
        return None if w is None or sig.get(w) is None else sig[w][rows, j]

    # -- fit -------------------------------------------------------------- #
    def fit(self, flows, exog=None):
        self._columns = list(flows.columns)
        flows_arr = flows.to_numpy(float)
        exog_arr = exog.to_numpy(float) if exog is not None else None
        self._design_periods = list(flows.index[self.order:])
        Xbase, Y = VARXPredictor(self.order)._build_design(flows_arr, exog_arr)
        sig = self._aligned_signals(self._design_periods)

        n = len(Y)
        v = min(self.val_months, max(6, n // 5))
        inner = n - v
        if inner <= self.order + 2 or not self.signals:
            # too little inner history (or no signals) → behave as plain VARX-CV
            self.chosen_alpha_ = self._pick_alpha_no_signal(Xbase, Y, inner, v)
            self.windows_ = {c: None for c in self._columns}
        else:
            self._choose(Xbase, Y, sig, inner, v)

        # refit on the full training window with the chosen α + per-category window
        self._params: dict[str, dict] = {}
        self._fill: dict[str, float] = {}
        rows_all = np.arange(n)
        for c, j in [(c, i) for i, c in enumerate(self._columns)]:
            w = self.windows_.get(c)
            col = self._candidate_cols(sig, w, j, rows_all)
            fill = float(np.nanmean(col)) if (col is not None and np.isfinite(col).any()) else 0.0
            self._fill[c] = fill
            Xc = self._augment(Xbase, col, fill)
            self._params[c] = _ridge_fit_1d(Xc, Y[:, j], self.chosen_alpha_)

        self._history = flows_arr[-self.order:].copy()
        self._exog_cols = list(exog.columns) if exog is not None else []
        self._last_period = flows.index[-1]
        return self

    def _pick_alpha_no_signal(self, Xbase, Y, inner, v) -> float:
        if inner <= self.order + 2:
            return 1.0
        Xtr, Xval = Xbase[:inner], Xbase[inner:]
        best_rmse, best_a = np.inf, 1.0
        for a in self.alphas:
            sse = 0.0
            for j in range(Y.shape[1]):
                p = _ridge_fit_1d(Xtr, Y[:inner, j], a)
                sse += float(np.sum((Y[inner:, j] - _ridge_pred_1d(Xval, p)) ** 2))
            if sse < best_rmse:
                best_rmse, best_a = sse, a
        return best_a

    def _choose(self, Xbase, Y, sig, inner, v) -> None:
        tr = np.arange(inner)
        val = np.arange(inner, len(Y))
        Xtr_b, Xval_b = Xbase[:inner], Xbase[inner:]
        best = (np.inf, 1.0, {c: None for c in self._columns})
        for a in self.alphas:
            total, wins = 0.0, {}
            for j, c in enumerate(self._columns):
                best_c = (np.inf, None)
                for w in self.WINDOWS:
                    col_tr = self._candidate_cols(sig, w, j, tr)
                    if col_tr is not None and np.isfinite(col_tr).mean() < self.min_valid:
                        continue  # signal too sparse for this category → ineligible
                    fill = (float(np.nanmean(col_tr))
                            if (col_tr is not None and np.isfinite(col_tr).any()) else 0.0)
                    Xtr_c = self._augment(Xtr_b, col_tr, fill)
                    Xval_c = self._augment(Xval_b, self._candidate_cols(sig, w, j, val), fill)
                    p = _ridge_fit_1d(Xtr_c, Y[:inner, j], a)
                    sse = float(np.sum((Y[inner:, j] - _ridge_pred_1d(Xval_c, p)) ** 2))
                    if sse < best_c[0]:
                        best_c = (sse, w)
                total += best_c[0]
                wins[c] = best_c[1]
            if total < best[0]:
                best = (total, a, wins)
        self.chosen_alpha_ = best[1]
        self.windows_ = best[2]

    # -- predict ---------------------------------------------------------- #
    def predict_next(self, next_exog=None):
        feat = []
        for lag in range(1, self.order + 1):
            feat.extend(self._history[-lag])
        if self._exog_cols:
            if next_exog is None:
                raise ValueError("VARXExcessCVPredictor was fit with exog; next_exog required")
            feat.extend(float(next_exog[c]) for c in self._exog_cols)
        base = np.asarray(feat, float)

        next_period = self._last_period + 1
        out = {}
        for j, c in enumerate(self._columns):
            w = self.windows_.get(c)
            if w is None or self.signals.get(w) is None:
                x = base
            else:
                df = self.signals[w]
                sval = (df.at[next_period, c]
                        if (next_period in df.index and c in df.columns) else np.nan)
                if not np.isfinite(sval):
                    sval = self._fill[c]
                x = np.concatenate([base, [sval]])
            out[c] = float(_ridge_pred_1d(x.reshape(1, -1), self._params[c])[0])
        return pd.Series(out)

    def predict_rows(self, flows, exog, periods) -> pd.DataFrame:
        """One-step-ahead predictions for the given design `periods`, using ACTUAL
        lagged flows + signals (no recursion). For inner-CV / diagnostics — applies
        the already-fitted per-category params to those rows' augmented designs.
        """
        flows_arr = flows.to_numpy(float)
        exog_arr = exog.to_numpy(float) if exog is not None else None
        design_periods = list(flows.index[self.order:])
        Xbase, _ = VARXPredictor(self.order)._build_design(flows_arr, exog_arr)
        sig = self._aligned_signals(design_periods)
        pos = {p: i for i, p in enumerate(design_periods)}
        rows = np.array([pos[p] for p in periods])
        out = {}
        for j, c in enumerate(self._columns):
            w = self.windows_.get(c)
            col = self._candidate_cols(sig, w, j, rows)
            Xc = self._augment(Xbase[rows], col, self._fill[c])
            out[c] = _ridge_pred_1d(Xc, self._params[c])
        return pd.DataFrame(out, index=list(periods))[self._columns]

import numpy as np
import pandas as pd
import pytest

from fund_flow.data.dataset import build_dataset
from fund_flow.predictor.backtest import (
    BacktestResult,
    compare,
    per_category_rmse,
    per_category_table,
    walk_forward,
)
from fund_flow.predictor.baselines import AR1Predictor, RandomWalkPredictor
from fund_flow.predictor.reshape import wide_exog, wide_flows
from fund_flow.predictor.var_model import VARXPredictor


@pytest.fixture(scope="module")
def matrices(cfg):
    frame = build_dataset(cfg)
    return wide_flows(frame), wide_exog(frame)


class TestReshape:
    def test_wide_flows_shape(self, cfg, matrices):
        flows, _ = matrices
        n_months = cfg["synthetic"]["predictor"]["n_months"]
        assert flows.shape == (n_months - 1, len(cfg["categories"]))
        assert isinstance(flows.index, pd.PeriodIndex)

    def test_exog_columns_all_lagged(self, matrices):
        _, exog = matrices
        assert all(c.endswith("_lag1") for c in exog.columns)

    def test_flows_and_exog_aligned(self, matrices):
        flows, exog = matrices
        assert flows.index.equals(exog.index)


class TestBaselines:
    def test_random_walk_predicts_last(self, matrices):
        flows, _ = matrices
        m = RandomWalkPredictor().fit(flows.iloc[:50])
        pred = m.predict_next()
        pd.testing.assert_series_equal(pred, flows.iloc[49], check_names=False)

    def test_ar1_returns_all_categories(self, matrices):
        flows, _ = matrices
        m = AR1Predictor().fit(flows.iloc[:50])
        pred = m.predict_next()
        assert set(pred.index) == set(flows.columns)
        assert not pred.isnull().any()


class TestVARX:
    def test_order_validation(self):
        with pytest.raises(ValueError, match="order"):
            VARXPredictor(order=0)

    def test_predict_shape(self, matrices):
        flows, exog = matrices
        m = VARXPredictor(order=1, ridge_alpha=1.0).fit(flows.iloc[:50], exog.iloc[:50])
        pred = m.predict_next(exog.iloc[50])
        assert set(pred.index) == set(flows.columns)
        assert not pred.isnull().any()

    def test_requires_exog_when_fitted_with_exog(self, matrices):
        flows, exog = matrices
        m = VARXPredictor(order=1).fit(flows.iloc[:50], exog.iloc[:50])
        with pytest.raises(ValueError, match="next_exog required"):
            m.predict_next(None)


class TestWalkForward:
    def test_no_lookahead_expanding_window(self, matrices):
        """Forecast for period t must not depend on data at/after t.

        Verified by checking the random-walk forecast at each step equals the
        value at t-1 (the last training point), never the actual at t.
        """
        flows, _ = matrices
        res = walk_forward(lambda: RandomWalkPredictor(), flows, None, min_train=36)
        # reconstruct: error[t] = actual[t] - actual[t-1] for random walk
        for period in res.errors.index:
            pos = flows.index.get_loc(period)
            expected_err = flows.iloc[pos] - flows.iloc[pos - 1]
            np.testing.assert_allclose(
                res.errors.loc[period].values, expected_err.values, rtol=1e-9
            )

    def test_min_train_too_large_raises(self, matrices):
        flows, _ = matrices
        with pytest.raises(ValueError, match="min_train"):
            walk_forward(lambda: RandomWalkPredictor(), flows, None, min_train=10_000)

    def test_n_forecasts_count(self, matrices):
        flows, _ = matrices
        res = walk_forward(lambda: RandomWalkPredictor(), flows, None, min_train=36)
        expected = (len(flows) - 36) * flows.shape[1]
        assert res.n_forecasts == expected


class TestRegime:
    def test_single_forecast_valid(self, matrices):
        from fund_flow.predictor.regime import MarkovRegimePredictor
        flows, _ = matrices
        m = MarkovRegimePredictor().fit(flows.iloc[:60])
        pred = m.predict_next()
        assert set(pred.index) == set(flows.columns)
        assert not pred.isnull().any()
        assert np.isfinite(pred.values).all()

    def test_fallback_on_degenerate_series(self):
        from fund_flow.predictor.regime import MarkovRegimePredictor
        # constant series → MS fit degenerate → AR(1) fallback, must not crash
        flows = pd.DataFrame({
            "Ações": [1.0] * 40,
            "Renda Fixa": np.arange(40, dtype=float),
        })
        pred = MarkovRegimePredictor().fit(flows).predict_next()
        assert np.isfinite(pred.values).all()


class TestAcceptanceGate:
    """CLAUDE.md §7 — the delivered predictor must beat BOTH naive baselines."""

    def test_varx_beats_baselines_out_of_sample(self, matrices):
        flows, exog = matrices
        min_train = 36
        rw = walk_forward(lambda: RandomWalkPredictor(), flows, None, min_train)
        ar1 = walk_forward(lambda: AR1Predictor(), flows, None, min_train)
        varx = walk_forward(
            lambda: VARXPredictor(order=1, ridge_alpha=1.0), flows, exog, min_train
        )
        assert varx.rmse < rw.rmse, "VARX must beat random walk"
        assert varx.rmse < ar1.rmse, "VARX must beat AR(1)"

    def test_compare_sorted_by_rmse(self, matrices):
        flows, exog = matrices
        results = [
            walk_forward(lambda: RandomWalkPredictor(), flows, None, 36, "rw"),
            walk_forward(lambda: VARXPredictor(ridge_alpha=1.0), flows, exog, 36, "varx"),
        ]
        tbl = compare(results)
        assert tbl["rmse"].is_monotonic_increasing


class TestVARXExtensions:
    def test_coefficients_labeled(self, matrices):
        flows, exog = matrices
        m = VARXPredictor(order=1, ridge_alpha=1.0).fit(flows.iloc[:60], exog.iloc[:60])
        c = m.coefficients()
        assert list(c.columns) == list(flows.columns)            # per-category
        assert c.shape[0] == flows.shape[1] + exog.shape[1]      # flow lags + exog
        assert any(name.endswith("_flow_lag1") for name in c.index)
        assert set(exog.columns).issubset(set(c.index))

    def test_predict_from_design_matches_predict_next(self, matrices):
        flows, exog = matrices
        m = VARXPredictor(order=1, ridge_alpha=1.0).fit(flows.iloc[:60], exog.iloc[:60])
        X, _ = m._build_design(flows.iloc[:61].to_numpy(float),
                               exog.iloc[:61].to_numpy(float))
        from_design = m._predict_from_design(X[-1:])[0]
        nxt = m.predict_next(exog.iloc[60]).to_numpy()
        assert np.allclose(from_design, nxt)

    def test_varxcv_selects_alpha_and_predicts(self, matrices):
        from fund_flow.predictor.var_model import VARXCVPredictor
        flows, exog = matrices
        m = VARXCVPredictor(order=1, alphas=[0.1, 1.0, 10.0], val_months=12).fit(
            flows.iloc[:80], exog.iloc[:80])
        assert m.chosen_alpha_ in (0.1, 1.0, 10.0)
        pred = m.predict_next(exog.iloc[80])
        assert not pred.isnull().any()
        assert list(pred.index) == list(flows.columns)

    def test_varxcv_defaults_alpha_on_tiny_sample(self, matrices):
        from fund_flow.predictor.var_model import VARXCVPredictor
        flows, exog = matrices
        m = VARXCVPredictor(order=1, val_months=24).fit(
            flows.iloc[:6], exog.iloc[:6])          # no room for inner-train
        assert m.chosen_alpha_ == 1.0


class TestPerCategory:
    def _result(self, errors):
        return BacktestResult("m", rmse=0.0, mae=0.0,
                              n_forecasts=errors.size, errors=errors)

    def test_per_category_rmse(self):
        errors = pd.DataFrame(
            {"Crédito Privado": [3.0, -4.0], "Renda Fixa": [0.0, 0.0]},
            index=["2024-01", "2024-02"],
        )
        r = per_category_rmse(self._result(errors))
        assert r["Crédito Privado"] == pytest.approx((12.5) ** 0.5)  # sqrt(mean[9,16])
        assert r["Renda Fixa"] == pytest.approx(0.0)
        assert r.name == "m"

    def test_per_category_table_one_col_per_model(self):
        errors = pd.DataFrame({"A": [1.0, -1.0], "B": [2.0, 0.0]})
        a = BacktestResult("ar1", 0, 0, 4, errors)
        b = BacktestResult("varx", 0, 0, 4, errors)
        tbl = per_category_table([a, b])
        assert list(tbl.columns) == ["ar1", "varx"]
        assert list(tbl.index) == ["A", "B"]


class TestPerCategorySelect:
    def test_selects_per_category_and_predicts(self, matrices):
        from fund_flow.predictor.var_model import PerCategorySelectPredictor

        flows, exog = matrices
        m = PerCategorySelectPredictor(order=1, val_months=12).fit(
            flows.iloc[:80], exog.iloc[:80]
        )
        # Each category gets a selection
        assert set(m.selected_.keys()) == set(flows.columns)
        assert all(v in ("ar1", "varxcv") for v in m.selected_.values())
        # Prediction has correct shape and no NaN
        pred = m.predict_next(exog.iloc[80])
        assert list(pred.index) == list(flows.columns)
        assert not pred.isnull().any()

    def test_fallback_on_tiny_sample(self, matrices):
        from fund_flow.predictor.var_model import PerCategorySelectPredictor

        flows, exog = matrices
        # Tiny sample — no room for inner split → default all to varxcv
        m = PerCategorySelectPredictor(order=1, val_months=24).fit(
            flows.iloc[:6], exog.iloc[:6]
        )
        assert all(v == "varxcv" for v in m.selected_.values())

    def test_walk_forward_integration(self, matrices):
        from fund_flow.predictor.var_model import PerCategorySelectPredictor

        flows, exog = matrices
        res = walk_forward(
            lambda: PerCategorySelectPredictor(order=1),
            flows, exog, min_train=36, name="per_cat",
        )
        assert res.rmse > 0
        assert res.n_forecasts == (len(flows) - 36) * flows.shape[1]

    def test_selects_excess_candidate_when_signal_drives_flow(self):
        from fund_flow.predictor.var_model import PerCategorySelectPredictor

        rng = np.random.default_rng(3)
        n = 100
        periods = pd.period_range("2010-01", periods=n, freq="M")
        cats = ["A", "B"]
        sigA = rng.normal(0, 1, n)
        flows = pd.DataFrame(
            {"A": 3.0 * sigA + rng.normal(0, 0.05, n), "B": rng.normal(0, 1, n)},
            index=periods)
        signals = {1: pd.DataFrame(np.c_[sigA, rng.normal(0, 1, n)],
                                   index=periods, columns=cats)}
        m = PerCategorySelectPredictor(order=1, signals=signals).fit(flows, None)
        assert m.selected_["A"] == "excess"   # the predictive signal is picked
        pred = m.predict_next(None)
        assert not pred.isnull().any()


class TestVARXExcessCV:
    def _signals_frames(self, periods, cats, arr_by_w):
        return {w: pd.DataFrame(a, index=periods, columns=cats)
                for w, a in arr_by_w.items()}

    def test_no_signals_all_windows_none(self, matrices):
        from fund_flow.predictor.var_model import VARXExcessCVPredictor
        flows, exog = matrices
        m = VARXExcessCVPredictor(order=1, signals={}).fit(flows.iloc[:80], exog.iloc[:80])
        assert set(m.windows_.values()) == {None}
        pred = m.predict_next(exog.iloc[80])
        assert list(pred.index) == list(flows.columns)
        assert not pred.isnull().any()

    def test_selects_window_when_signal_drives_flow(self):
        from fund_flow.predictor.var_model import VARXExcessCVPredictor
        rng = np.random.default_rng(0)
        n = 90
        periods = pd.period_range("2010-01", periods=n, freq="M")
        cats = ["A", "B"]
        # signal_A drives A's flow; B is pure noise. Signal is already lag-1 aligned
        # (signal at period t == info usable to predict flow[t]).
        sigA = rng.normal(0, 1, n)
        sigB = rng.normal(0, 1, n)
        flowA = 3.0 * sigA + rng.normal(0, 0.05, n)   # A depends ONLY on its signal
        flowB = rng.normal(0, 1, n)
        flows = pd.DataFrame({"A": flowA, "B": flowB}, index=periods)
        signals = self._signals_frames(periods, cats, {1: np.c_[sigA, sigB]})
        m = VARXExcessCVPredictor(order=1, signals=signals,
                                  alphas=[0.1, 1.0, 10.0], val_months=24).fit(flows, None)
        assert m.windows_["A"] == 1            # the predictive signal is selected
        pred = m.predict_next(None)
        assert np.isfinite(pred["A"]) and np.isfinite(pred["B"])

    def test_handles_nan_signal_rows(self):
        from fund_flow.predictor.var_model import VARXExcessCVPredictor
        rng = np.random.default_rng(1)
        n = 80
        periods = pd.period_range("2010-01", periods=n, freq="M")
        cats = ["A", "B"]
        sig = rng.normal(0, 1, (n, 2))
        sig[:20, 0] = np.nan          # A's signal absent early (e.g. pre-ETF benchmark)
        flows = pd.DataFrame(rng.normal(0, 1, (n, 2)), index=periods, columns=cats)
        signals = self._signals_frames(periods, cats, {3: sig})
        m = VARXExcessCVPredictor(order=1, signals=signals, val_months=20).fit(flows, None)
        pred = m.predict_next(None)
        assert not pred.isnull().any()

    def test_walk_forward_with_signals(self, matrices):
        from fund_flow.predictor.var_model import VARXExcessCVPredictor
        flows, exog = matrices
        cats = list(flows.columns)
        rng = np.random.default_rng(2)
        sig = pd.DataFrame(rng.normal(0, 1, (len(flows), len(cats))),
                           index=flows.index, columns=cats)
        res = walk_forward(
            lambda: VARXExcessCVPredictor(order=1, signals={1: sig}),
            flows, exog, min_train=36, name="excess")
        assert res.rmse > 0
        assert res.n_forecasts == (len(flows) - 36) * flows.shape[1]


class TestWalkForwardWindow:
    def test_rolling_window_limits_train_size(self, matrices, monkeypatch):
        flows, _ = matrices
        seen = []

        class Spy(AR1Predictor):
            def fit(self, f, e=None):
                seen.append(len(f))
                return super().fit(f, e)

        walk_forward(lambda: Spy(), flows, None, min_train=24, window=24)
        # Every training window is capped at the rolling size, never expanding.
        assert max(seen) == 24
        assert min(seen) >= 24

    def test_expanding_window_grows(self, matrices):
        flows, _ = matrices
        seen = []

        class Spy(AR1Predictor):
            def fit(self, f, e=None):
                seen.append(len(f))
                return super().fit(f, e)

        walk_forward(lambda: Spy(), flows, None, min_train=24)  # window=None
        assert seen[0] == 24 and seen[-1] > seen[0]  # expands

    def test_test_window_restricts_forecasts(self, matrices):
        flows, _ = matrices
        res = walk_forward(lambda: AR1Predictor(), flows, None, min_train=24,
                           test_start=30, test_end=40)
        assert len(res.errors) == 10  # forecasts only for t in [30, 40)

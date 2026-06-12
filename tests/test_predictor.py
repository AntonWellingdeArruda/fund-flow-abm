import numpy as np
import pandas as pd
import pytest

from fund_flow.data.dataset import build_dataset
from fund_flow.predictor.backtest import compare, walk_forward
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

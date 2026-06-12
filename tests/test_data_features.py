import numpy as np
import pandas as pd
import pytest

from fund_flow.data.cleaning import clean_flows, clean_macro
from fund_flow.data.dataset import build_dataset
from fund_flow.data.features import (
    TARGET_COLUMNS,
    build_model_frame,
    feature_columns,
)
from fund_flow.data.sources import SyntheticFlowSource, SyntheticMacroSource

# Contemporaneous macro/return columns that must NEVER appear as predictors.
_CONTEMPORANEOUS_MACRO = [
    "selic_rate", "delta_selic", "ipca_monthly",
    "ibovespa_return", "dxy_return", "ust_10y",
]


@pytest.fixture
def flows(cfg):
    return clean_flows(SyntheticFlowSource(cfg).load())


@pytest.fixture
def macro(cfg):
    return clean_macro(SyntheticMacroSource(cfg).load())


@pytest.fixture
def frame(cfg, flows, macro):
    return build_model_frame(flows, macro, exempt=cfg["come_cotas_exempt"], n_lags=1)


class TestFrameShape:
    def test_no_nan(self, frame):
        assert not frame.isnull().any().any()

    def test_warmup_dropped(self, cfg, frame):
        n_months = cfg["synthetic"]["predictor"]["n_months"]
        n_cats = len(cfg["categories"])
        # one warmup period dropped per category at n_lags=1
        assert len(frame) == (n_months - 1) * n_cats

    def test_targets_present(self, frame):
        for t in TARGET_COLUMNS:
            assert t in frame.columns

    def test_own_flow_ar_feature_present(self, frame):
        assert "net_flow_brl_lag1" in frame.columns
        assert "redemption_gross_brl_lag1" in frame.columns


class TestLagDiscipline:
    """CLAUDE.md §3.2 — no contemporaneous returns as predictors."""

    def test_contemporaneous_macro_dropped_from_frame(self, frame):
        for col in _CONTEMPORANEOUS_MACRO:
            assert col not in frame.columns, \
                f"Contemporaneous {col!r} must be dropped to prevent leakage"

    def test_no_contemporaneous_macro_in_feature_columns(self, frame):
        feats = feature_columns(frame)
        for col in _CONTEMPORANEOUS_MACRO:
            assert col not in feats

    def test_feature_columns_are_lagged_or_control(self, frame):
        feats = feature_columns(frame)
        for f in feats:
            assert f == "come_cotas" or "_lag" in f, \
                f"Feature {f!r} is neither lagged nor the come-cotas control"

    def test_lag_is_honest(self, cfg, flows, macro, frame):
        # selic_rate_lag1 at period t must equal raw macro selic_rate at t-1.
        macro_sorted = macro.sort_values("period").reset_index(drop=True)
        period_to_selic = dict(zip(macro_sorted["period"], macro_sorted["selic_rate"]))
        periods = macro_sorted["period"].tolist()
        prev_period = {periods[i]: periods[i - 1] for i in range(1, len(periods))}

        acoes = frame[frame["category"] == "Ações"]
        for _, row in acoes.iterrows():
            expected = period_to_selic[prev_period[row["period"]]]
            assert row["selic_rate_lag1"] == pytest.approx(expected)


class TestComeCotasControl:
    def test_present_as_feature(self, frame):
        assert "come_cotas" in feature_columns(frame)

    def test_exempt_categories_never_true(self, cfg, frame):
        for cat in cfg["come_cotas_exempt"]:
            sub = frame[frame["category"] == cat]
            assert not sub["come_cotas"].any()


class TestEconomicStructureSurvives:
    def test_lead_lag_ibov_acoes_in_features(self, frame):
        acoes = frame[frame["category"] == "Ações"]
        corr = np.corrcoef(
            acoes["ibovespa_return_lag1"].values,
            acoes["net_flow_brl"].values,
        )[0, 1]
        assert corr > 0, \
            f"Lagged Ibovespa→Ações flow correlation {corr:.3f} should be positive"

    def test_flow_autocorr_via_lag_feature(self, frame):
        rf = frame[frame["category"] == "Renda Fixa"]
        corr = np.corrcoef(
            rf["net_flow_brl_lag1"].values,
            rf["net_flow_brl"].values,
        )[0, 1]
        assert corr > 0.2


class TestBuildDatasetOrchestrator:
    def test_end_to_end(self, cfg):
        ds = build_dataset(cfg)
        assert not ds.isnull().any().any()
        assert "net_flow_brl" in ds.columns
        assert "ibovespa_return_lag1" in ds.columns

    def test_deterministic(self, cfg):
        a = build_dataset(cfg)
        b = build_dataset(cfg)
        pd.testing.assert_frame_equal(a, b)

    def test_n_lags_validation(self, cfg, flows, macro):
        with pytest.raises(ValueError, match="n_lags"):
            build_model_frame(flows, macro, exempt=cfg["come_cotas_exempt"], n_lags=0)

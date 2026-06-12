import numpy as np
import pandas as pd
import pytest

from fund_flow.data.cleaning import clean_flows, clean_macro
from fund_flow.data.sources import (
    SyntheticFlowSource,
    SyntheticMacroSource,
)


@pytest.fixture
def raw_flows(cfg):
    return SyntheticFlowSource(cfg).load()


@pytest.fixture
def raw_macro(cfg):
    return SyntheticMacroSource(cfg).load()


class TestCleanFlows:
    def test_passes_clean_data(self, raw_flows):
        out = clean_flows(raw_flows)
        assert len(out) == len(raw_flows)
        assert not out.isnull().any().any()

    def test_sorted_by_category_then_period(self, raw_flows):
        out = clean_flows(raw_flows)
        for _, grp in out.groupby("category"):
            periods = grp["period"].tolist()
            assert periods == sorted(periods)

    def test_dedups(self, raw_flows):
        dup = pd.concat([raw_flows, raw_flows.iloc[:5]], ignore_index=True)
        out = clean_flows(dup)
        assert not out.duplicated(subset=["period", "category"]).any()

    def test_rejects_unknown_category(self, raw_flows):
        bad = raw_flows.copy()
        bad.loc[0, "category"] = "Tesouro Direto"
        with pytest.raises(ValueError, match="unknown categories"):
            clean_flows(bad)

    def test_rejects_malformed_period(self, raw_flows):
        bad = raw_flows.copy()
        bad.loc[0, "period"] = "2024/01"
        with pytest.raises(ValueError, match="malformed periods"):
            clean_flows(bad)

    def test_rejects_negative_gross(self, raw_flows):
        bad = raw_flows.copy()
        bad.loc[0, "redemption_gross_brl"] = -1.0
        with pytest.raises(ValueError, match="negative redemption_gross_brl"):
            clean_flows(bad)

    def test_rejects_missing_column(self, raw_flows):
        bad = raw_flows.drop(columns=["net_flow_brl"])
        with pytest.raises(ValueError, match="missing required columns"):
            clean_flows(bad)


class TestCleanMacro:
    def test_passes_clean_data(self, raw_macro):
        out = clean_macro(raw_macro)
        assert len(out) == len(raw_macro)
        assert not out.isnull().any().any()

    def test_sorted_by_period(self, raw_macro):
        out = clean_macro(raw_macro)
        assert out["period"].tolist() == sorted(out["period"].tolist())

    def test_forward_fills_missing(self, raw_macro):
        gapped = raw_macro.copy()
        gapped.loc[3, "selic_rate"] = np.nan
        out = clean_macro(gapped)
        assert not out["selic_rate"].isnull().any()
        # filled value equals the prior period's realized value
        assert out.loc[3, "selic_rate"] == out.loc[2, "selic_rate"]

    def test_regime_is_int(self, raw_macro):
        out = clean_macro(raw_macro)
        assert out["regime"].dtype == int

    def test_dedups_periods(self, raw_macro):
        dup = pd.concat([raw_macro, raw_macro.iloc[:3]], ignore_index=True)
        out = clean_macro(dup)
        assert not out["period"].duplicated().any()

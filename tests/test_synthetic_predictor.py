import numpy as np
import pandas as pd
import pytest

from fund_flow.synthetic.predictor_data import generate_predictor_panel


class TestPredicatorShape:
    def test_row_count(self, cfg, predictor_df):
        n_months = cfg["synthetic"]["predictor"]["n_months"]
        n_cats = len(cfg["categories"])
        assert len(predictor_df) == n_months * n_cats

    def test_required_columns(self, predictor_df):
        expected = {
            "period", "category", "net_flow_brl", "redemption_gross_brl",
            "selic_rate", "delta_selic", "ipca_monthly", "ibovespa_return",
            "dxy_return", "ust_10y", "regime", "come_cotas",
        }
        assert expected.issubset(set(predictor_df.columns))

    def test_no_nan(self, predictor_df):
        assert not predictor_df.isnull().any().any()

    def test_all_categories_present(self, cfg, predictor_df):
        assert set(predictor_df["category"].unique()) == set(cfg["categories"])

    def test_gross_redemption_non_negative(self, predictor_df):
        assert (predictor_df["redemption_gross_brl"] >= 0).all()


class TestComeCotasFlag:
    def test_only_may_and_november(self, predictor_df):
        cc_rows = predictor_df[predictor_df["come_cotas"]]
        months = cc_rows["period"].apply(
            lambda p: pd.Period(p, freq="M").month
        ).unique()
        assert set(months).issubset({5, 11})

    def test_exempt_categories_never_flagged(self, cfg, predictor_df):
        for cat in cfg["come_cotas_exempt"]:
            cat_rows = predictor_df[predictor_df["category"] == cat]
            assert not cat_rows["come_cotas"].any(), \
                f"{cat} should be exempt from come_cotas"

    def test_non_exempt_flagged_in_may(self, cfg, predictor_df):
        non_exempt = [c for c in cfg["categories"]
                      if c not in cfg["come_cotas_exempt"]]
        may_rows = predictor_df[
            predictor_df["period"].apply(lambda p: pd.Period(p, freq="M").month == 5)
        ]
        for cat in non_exempt:
            cat_may = may_rows[may_rows["category"] == cat]
            assert cat_may["come_cotas"].all(), \
                f"{cat} should have come_cotas=True in May"


class TestEconomicStructure:
    """Phase 2 validation requires these properties to be detectable."""

    def test_flow_autocorrelation_rf(self, predictor_df):
        rf = (
            predictor_df[predictor_df["category"] == "Renda Fixa"]
            .sort_values("period")["net_flow_brl"]
        )
        ac = rf.autocorr(lag=1)
        assert ac > 0.2, f"Renda Fixa flow autocorrelation {ac:.3f} too low"

    def test_flow_autocorrelation_cp(self, predictor_df):
        cp = (
            predictor_df[predictor_df["category"] == "Crédito Privado"]
            .sort_values("period")["net_flow_brl"]
        )
        ac = cp.autocorr(lag=1)
        assert ac > 0.2, f"Crédito Privado flow autocorrelation {ac:.3f} too low"

    def test_regime_rotation_sign(self, predictor_df):
        high_selic = predictor_df[predictor_df["regime"] == 1]
        low_selic = predictor_df[predictor_df["regime"] == 0]

        rf_high = high_selic[high_selic["category"] == "Renda Fixa"]["net_flow_brl"].mean()
        rf_low = low_selic[low_selic["category"] == "Renda Fixa"]["net_flow_brl"].mean()
        assert rf_high > rf_low, "Renda Fixa should attract more flows in high-Selic regime"

        acoes_high = high_selic[high_selic["category"] == "Ações"]["net_flow_brl"].mean()
        acoes_low = low_selic[low_selic["category"] == "Ações"]["net_flow_brl"].mean()
        assert acoes_high < acoes_low, "Ações should lose flows in high-Selic regime"

    def test_lead_lag_ibov_acoes(self, predictor_df):
        acoes = (
            predictor_df[predictor_df["category"] == "Ações"]
            .sort_values("period")
            .reset_index(drop=True)
        )
        # lagged ibovespa_return[t] should positively predict net_flow_brl[t+1]
        ibov_lag = acoes["ibovespa_return"].iloc[:-1].values
        flow_next = acoes["net_flow_brl"].iloc[1:].values
        corr = np.corrcoef(ibov_lag, flow_next)[0, 1]
        assert corr > 0, f"Lagged Ibovespa→Ações flow correlation {corr:.3f} should be positive"

    def test_deterministic(self, cfg):
        df1 = generate_predictor_panel(cfg)
        df2 = generate_predictor_panel(cfg)
        pd.testing.assert_frame_equal(df1, df2)

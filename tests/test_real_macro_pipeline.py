"""The pipeline must run on a real-shaped macro SUBSET (BCB columns only),
not just the full synthetic schema. Uses an offline BCB-shaped frame so it
needs no network.
"""
import numpy as np
import pandas as pd

from fund_flow.data.cleaning import clean_flows, clean_macro
from fund_flow.data.features import build_model_frame, feature_columns
from fund_flow.data.sources import SyntheticFlowSource
from fund_flow.predictor.reshape import exog_columns, wide_exog, wide_flows
from fund_flow.predictor.var_model import VARXPredictor


def _bcb_shaped_macro(periods):
    """A macro frame with exactly the columns BcbMacroSource provides."""
    rng = np.random.default_rng(0)
    n = len(periods)
    return pd.DataFrame({
        "period": [str(p) for p in periods],
        "selic_rate": np.linspace(0.06, 0.1375, n),
        "delta_selic": rng.normal(0, 0.002, n),
        "ipca_monthly": rng.uniform(0.001, 0.009, n),
        "usdbrl_return": rng.normal(0, 0.03, n),
    })


def test_subset_macro_flows_end_to_end(cfg):
    flows = clean_flows(SyntheticFlowSource(cfg).load())
    periods = pd.PeriodIndex(sorted(flows["period"].unique()), freq="M")
    macro = clean_macro(_bcb_shaped_macro(periods))

    frame = build_model_frame(flows, macro, exempt=cfg["come_cotas_exempt"], n_lags=1)

    # exog discovered from the subset: the 4 BCB lags, no regime/ibov/ust.
    exog = exog_columns(frame)
    assert set(exog) == {
        "selic_rate_lag1", "delta_selic_lag1",
        "ipca_monthly_lag1", "usdbrl_return_lag1",
    }
    # No contemporaneous macro leaked in.
    for col in ("selic_rate", "ipca_monthly", "usdbrl_return"):
        assert col not in feature_columns(frame)
    assert not frame.isnull().any().any()


def test_varx_fits_on_subset_macro(cfg):
    flows = clean_flows(SyntheticFlowSource(cfg).load())
    periods = pd.PeriodIndex(sorted(flows["period"].unique()), freq="M")
    macro = clean_macro(_bcb_shaped_macro(periods))
    frame = build_model_frame(flows, macro, exempt=cfg["come_cotas_exempt"])

    wf = wide_flows(frame)
    we = wide_exog(frame)              # discovers the 4 BCB exog columns
    assert we.shape[1] == 4

    m = VARXPredictor(order=1, ridge_alpha=1.0).fit(wf.iloc[:50], we.iloc[:50])
    pred = m.predict_next(we.iloc[50])
    assert not pred.isnull().any()

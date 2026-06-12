"""Orchestrator: source → clean → features → modeling frame.

Single entrypoint the Predictor (Phase 2) consumes. Defaults to synthetic
sources; swap in Csv*/real adapters without touching downstream code.
"""
from __future__ import annotations

import pandas as pd

from fund_flow.data.cleaning import clean_flows, clean_macro
from fund_flow.data.features import build_model_frame
from fund_flow.data.sources import (
    FlowSource,
    MacroSource,
    SyntheticFlowSource,
    SyntheticMacroSource,
)


def build_dataset(
    cfg: dict,
    flow_source: FlowSource | None = None,
    macro_source: MacroSource | None = None,
    n_lags: int = 1,
) -> pd.DataFrame:
    """Run the full data layer and return the analysis-ready modeling frame.

    Sources default to the synthetic backends driven by cfg.
    """
    flow_source = flow_source or SyntheticFlowSource(cfg)
    macro_source = macro_source or SyntheticMacroSource(cfg)

    flows = clean_flows(flow_source.load())
    macro = clean_macro(macro_source.load())

    return build_model_frame(
        flows, macro, exempt=cfg["come_cotas_exempt"], n_lags=n_lags
    )

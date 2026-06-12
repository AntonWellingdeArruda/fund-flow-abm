"""Feature engineering with strict lag discipline (CLAUDE.md §3.2).

The model predicts period t's flow using ONLY information realized by t-1
(macro/returns) plus calendar controls known ex ante (come-cotas). To make
this structural rather than conventional, contemporaneous macro/return columns
are DROPPED from the modeling frame — only lagged versions survive as
predictors, so a model literally cannot be fed ibovespa_return[t] to predict
net_flow_brl[t].

Targets (net_flow_brl[t], redemption_gross_brl[t]) and the realized regime
label (regime[t], used by Phase 2's regime-switching model / for evaluation)
are kept as non-predictor columns and are excluded from feature_columns().
"""
from __future__ import annotations

import pandas as pd

from fund_flow.utils.calendar import is_come_cotas

TARGET_COLUMNS = ["net_flow_brl", "redemption_gross_brl"]

# Own-flow columns lagged into AR predictors (and kept contemporaneous as targets).
_FLOW_LAG_SOURCES = ["net_flow_brl", "redemption_gross_brl"]

# Latent-state label: lagged into a feature, kept contemporaneous as metadata
# (the regime-switching model infers it; it is never a contemporaneous predictor).
_REGIME = "regime"


def build_model_frame(
    flows: pd.DataFrame,
    macro: pd.DataFrame,
    exempt: list[str],
    n_lags: int = 1,
) -> pd.DataFrame:
    """Merge cleaned flows + macro into a tidy, lag-disciplined modeling frame.

    One row per (period, category). Drops the first n_lags periods per category
    (no lag available). Returns no NaN.
    """
    if n_lags < 1:
        raise ValueError("n_lags must be >= 1")

    # Macro lag sources are discovered from whatever the macro table provides
    # (synthetic supplies the full set; a real adapter may supply a subset),
    # so the pipeline is not tied to the synthetic column list.
    macro_sources = [c for c in macro.columns if c != "period"]
    lag_sources = _FLOW_LAG_SOURCES + macro_sources

    merged = flows.merge(macro, on="period", how="inner")
    merged = merged.sort_values(["category", "period"]).reset_index(drop=True)

    grouped = merged.groupby("category", sort=False)
    for col in lag_sources:
        for k in range(1, n_lags + 1):
            merged[f"{col}_lag{k}"] = grouped[col].shift(k)

    # come-cotas: calendar control known ex ante → legitimate at period t.
    merged["come_cotas"] = [
        is_come_cotas(p, c, exempt)
        for p, c in zip(merged["period"], merged["category"])
    ]

    # Drop warmup rows that lack the deepest lag.
    deepest = [f"{col}_lag{n_lags}" for col in lag_sources]
    merged = merged.dropna(subset=deepest).reset_index(drop=True)

    # Structurally drop contemporaneous macro/return predictors (keep regime
    # label as metadata if present) to prevent look-ahead leakage (§3.2).
    drop_contemporaneous = [c for c in macro_sources if c != _REGIME]
    merged = merged.drop(columns=drop_contemporaneous)

    return merged


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Predictor columns: all lagged features + the come-cotas control.

    Excludes targets, keys, and the realized regime metadata label.
    """
    cols = [c for c in frame.columns if c.endswith(tuple(
        f"_lag{k}" for k in range(1, 10)
    ))]
    if "come_cotas" in frame.columns:
        cols.append("come_cotas")
    return cols

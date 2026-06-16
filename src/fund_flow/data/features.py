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
    extra_lag_sources: list[str] | None = None,
) -> pd.DataFrame:
    """Merge cleaned flows + macro into a tidy, lag-disciplined modeling frame.

    One row per (period, category). Drops the first n_lags periods per category
    (no lag available). Returns no NaN in the REQUIRED predictors.

    extra_lag_sources: PER-CATEGORY columns already present in `flows` (Phase 2.6
    excess returns). They are lagged per category and their contemporaneous form
    is dropped, exactly like macro — BUT they are *optional*: they may be NaN
    (Cambial, early periods, pre-2018 IMA-B funds) and are deliberately excluded
    from the warm-up dropna, so a sparse excess signal never deletes the flow rows
    the base models need.
    """
    if n_lags < 1:
        raise ValueError("n_lags must be >= 1")

    # Macro lag sources are discovered from whatever the macro table provides
    # (synthetic supplies the full set; a real adapter may supply a subset),
    # so the pipeline is not tied to the synthetic column list.
    macro_sources = [c for c in macro.columns if c != "period"]
    extra = [c for c in (extra_lag_sources or []) if c in flows.columns]
    required_lag = _FLOW_LAG_SOURCES + macro_sources
    all_lag = required_lag + extra

    merged = flows.merge(macro, on="period", how="inner")
    merged = merged.sort_values(["category", "period"]).reset_index(drop=True)

    grouped = merged.groupby("category", sort=False)
    for col in all_lag:
        for k in range(1, n_lags + 1):
            merged[f"{col}_lag{k}"] = grouped[col].shift(k)

    # come-cotas: calendar control known ex ante → legitimate at period t.
    merged["come_cotas"] = [
        is_come_cotas(p, c, exempt)
        for p, c in zip(merged["period"], merged["category"])
    ]

    # Drop warm-up rows that lack the deepest REQUIRED lag (excess signal excluded
    # so its NaNs do not prune flow rows).
    deepest = [f"{col}_lag{n_lags}" for col in required_lag]
    merged = merged.dropna(subset=deepest).reset_index(drop=True)

    # Structurally drop contemporaneous macro/return + excess predictors (keep
    # regime label as metadata if present) to prevent look-ahead leakage (§3.2).
    drop_contemporaneous = [c for c in macro_sources if c != _REGIME] + extra
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

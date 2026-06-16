"""Assemble + cache the REAL data panel (flows + macro) for the predictor.

The raw sources are slow to hit live (CVM informe archives, BCB SGS with retry,
yfinance). The webapp and repeated backtests need fast, repeatable access, so
the assembled monthly macro panel is cached to a gitignored CSV. Flows are
already cached by scripts/fetch_cvm_*.py to `flows_real.csv`.

Macro columns (all cover the full ~2004+ flow window):
    selic_rate, delta_selic, ipca_monthly, usdbrl_return   (BCB, no key)
    fed_funds, ust_10y, vix                                (FRED, deep history)
    ibovespa_return, sp500_return, dxy_return               (Yahoo, 1985/1993+)

The EMBI+ credit spread (embi_spread, delta_embi) comes from IPEAdata and is
OPT-IN (include_ipea=True): it is discontinued at 2024-07, so adding it to the
inner-joined panel truncates the window there.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from fund_flow.data.sources import (
    BcbMacroSource,
    CompositeMacroSource,
    FredMacroSource,
    IpeaMacroSource,
    YahooMacroSource,
)

DATA_DIR = Path("data")
MACRO_CACHE = DATA_DIR / "macro_real.csv"
EXCESS_CACHE = DATA_DIR / "category_excess.csv"

# Phase-2.6 per-category excess-return windows (raw / contemporaneous; the feature
# layer lags them to *_lag1). Trailing means END at t; build_model_frame's shift(1)
# turns them into windows ending at t-1, per CLAUDE.md §3 lag discipline.
EXCESS_WINDOWS = {"excess_ret_1m": 1, "excess_ret_3m": 3, "excess_ret_6m": 6}

# Which macro variables come from which source — exposed so a UI can group the
# per-predictor toggles by origin.
MACRO_GROUPS = {
    "BCB (Brazil)": ["selic_rate", "delta_selic", "ipca_monthly", "usdbrl_return"],
    "Credit spread (IPEA EMBI)": ["embi_spread", "delta_embi"],
    "FRED (US rates)": ["fed_funds", "ust_10y"],
    "Risk / volatility": ["vix"],
    "Yahoo (markets)": ["ibovespa_return", "sp500_return", "dxy_return"],
}


def build_real_macro_source(
    start: str,
    end: str,
    include_fred: bool = True,
    include_yahoo: bool = True,
    include_ipea: bool = False,
) -> CompositeMacroSource:
    """Composite of the macro sources that cover the flow window.

    Unlike FRED's S&P 500 / broad-dollar (which truncate to ~2016), the default
    sources all reach back past 2004, so the inner join keeps the whole history.
    include_ipea adds the EMBI+ credit spread — opt-in because that series is
    discontinued at 2024-07 and would truncate the inner-joined panel there.
    """
    sources = [BcbMacroSource(start=start, end=end, timeout=90.0)]
    if include_fred:
        sources.append(FredMacroSource(start=start, end=end))
    if include_yahoo:
        sources.append(YahooMacroSource(start=start, end=end))
    if include_ipea:
        sources.append(IpeaMacroSource(start=start, end=end))
    return CompositeMacroSource(sources)


def load_real_macro(
    start: str, end: str, refresh: bool = False, cache: Path = MACRO_CACHE
) -> pd.DataFrame:
    """Return the assembled real macro panel, using the on-disk cache if it
    already covers [start, end]. Set refresh=True to force a live re-fetch.
    """
    if cache.exists() and not refresh:
        df = pd.read_csv(cache, dtype={"period": str})
        # The macro panel legitimately starts ~1 month after `start` (monthly
        # returns lose the first observation), so allow a 1-month grace on the
        # lower bound — otherwise the cache never validates and we re-fetch live.
        if not df.empty:
            cmin = pd.Period(df["period"].min(), "M")
            cmax = pd.Period(df["period"].max(), "M")
            if cmin <= pd.Period(start, "M") + 1 and cmax >= pd.Period(end, "M"):
                return df[(df["period"] >= start) & (df["period"] <= end)].reset_index(drop=True)

    df = build_real_macro_source(start, end).load()
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def flow_period_range(flows_csv: str) -> tuple[str, str]:
    """(min, max) period in a flows CSV, e.g. ('2004-04', '2025-12')."""
    p = pd.read_csv(flows_csv, dtype={"period": str})["period"]
    return str(p.min()), str(p.max())


def attach_excess_windows(
    flows: pd.DataFrame, excess_csv: Path = EXCESS_CACHE
) -> tuple[pd.DataFrame, list[str]]:
    """Merge per-category excess-return windows into the long flows frame.

    Returns (flows_with_windows, extra_lag_source_names). If the excess cache is
    absent, returns the flows unchanged and an empty list — so the predictor
    pipeline degrades gracefully to macro-only when Phase 2.6 data isn't built.
    Trailing means are computed per category on the realized ER_c_t (ending at t).
    """
    if not excess_csv.exists():
        return flows, []
    ex = pd.read_csv(excess_csv, dtype={"period": str})
    ex = ex.sort_values(["category", "period"]).reset_index(drop=True)
    g = ex.groupby("category", sort=False)["cat_excess_return"]
    for col, win in EXCESS_WINDOWS.items():
        ex[col] = g.transform(lambda s, w=win: s.rolling(w, min_periods=1).mean())
    keep = ["period", "category", *EXCESS_WINDOWS]
    flows = flows.merge(ex[keep], on=["period", "category"], how="left")
    return flows, list(EXCESS_WINDOWS)


def build_real_frame(
    flows_csv: str, cfg: dict, refresh: bool = False, n_lags: int = 1,
    excess_csv: Path = EXCESS_CACHE,
) -> pd.DataFrame:
    """Analysis-ready modeling frame from real flows CSV + cached real macro.

    Mirrors data.dataset.build_dataset but pulls macro from the on-disk cache
    (fast, repeatable) instead of re-fetching live. Used by the webapp. If the
    Phase-2.6 excess cache exists, per-category excess-return windows are merged
    in and lagged alongside the macro (they become 'excess_ret_*m_lag1').
    """
    from fund_flow.data.cleaning import clean_flows, clean_macro
    from fund_flow.data.features import build_model_frame
    from fund_flow.data.sources import CsvFlowSource

    start, end = flow_period_range(flows_csv)
    flows = clean_flows(CsvFlowSource(flows_csv).load())
    flows, extra = attach_excess_windows(flows, excess_csv)
    macro = clean_macro(load_real_macro(start, end, refresh=refresh))
    return build_model_frame(
        flows, macro, exempt=cfg["come_cotas_exempt"], n_lags=n_lags,
        extra_lag_sources=extra,
    )

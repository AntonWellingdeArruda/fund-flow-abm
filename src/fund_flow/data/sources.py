"""Ingestion adapters for the data layer.

Two raw tables flow through the pipeline:
  - FLOWS:  long format, one row per (period, category):
            period, category, net_flow_brl, redemption_gross_brl
  - MACRO:  wide format, one row per period:
            period, selic_rate, delta_selic, ipca_monthly,
            ibovespa_return, dxy_return, ust_10y, regime

Phase 1 ships synthetic + CSV adapters. Real public-source adapters
(Anbima captação líquida, BCB SGS, B3) plug in behind the same Protocols;
stubs document the public endpoints without making network calls so the
pipeline stays testable offline (CLAUDE.md §5).
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd

from fund_flow.synthetic.predictor_data import generate_predictor_panel

FLOW_COLUMNS = ["period", "category", "net_flow_brl", "redemption_gross_brl"]
MACRO_COLUMNS = [
    "period", "selic_rate", "delta_selic", "ipca_monthly",
    "ibovespa_return", "dxy_return", "ust_10y", "regime",
]


class FlowSource(Protocol):
    def load(self) -> pd.DataFrame:
        """Return raw flows: long format with FLOW_COLUMNS."""
        ...


class MacroSource(Protocol):
    def load(self) -> pd.DataFrame:
        """Return raw macro: wide format with MACRO_COLUMNS, one row per period."""
        ...


# --- Synthetic backends (default; reuse the Phase 0 ground-truth generator) ---

class SyntheticFlowSource:
    """Slices the synthetic panel into a raw flows table."""

    def __init__(self, cfg: dict):
        self._cfg = cfg

    def load(self) -> pd.DataFrame:
        panel = generate_predictor_panel(self._cfg)
        return panel[FLOW_COLUMNS].copy()


class SyntheticMacroSource:
    """Slices the synthetic panel into a raw macro table (one row per period).

    Macro values are identical across categories within a period in the
    synthetic panel, so we de-duplicate on period.
    """

    def __init__(self, cfg: dict):
        self._cfg = cfg

    def load(self) -> pd.DataFrame:
        panel = generate_predictor_panel(self._cfg)
        macro = panel[MACRO_COLUMNS].drop_duplicates(subset="period")
        return macro.reset_index(drop=True)


# --- CSV backends (for approved-environment local data) ---

class CsvFlowSource:
    def __init__(self, path: str):
        self._path = path

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self._path)
        return df[FLOW_COLUMNS].copy()


class CsvMacroSource:
    def __init__(self, path: str):
        self._path = path

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self._path)
        return df[MACRO_COLUMNS].copy()


# --- Real public-source stubs (documented seams; no network calls in Phase 1) ---

class AnbimaFlowSource:
    """ANBIMA captação líquida by category via the authenticated API.

    OAuth2 is fully wired (see fund_flow.data.anbima.AnbimaClient). The DATA
    endpoint path for captação líquida depends on the subscribed API product
    and must be supplied via `data_path`; once known, load() fetches and
    reshapes it to FLOW_COLUMNS.
    """

    def __init__(self, data_path: str | None = None, client=None):
        self._data_path = data_path
        self._client = client

    def _make_client(self):
        from fund_flow.config import get_secret
        from fund_flow.data.anbima import AnbimaClient
        return AnbimaClient(
            get_secret("ANBIMA_CLIENT_ID"),
            get_secret("ANBIMA_CLIENT_SECRET"),
        )

    def load(self) -> pd.DataFrame:
        if self._data_path is None:
            raise NotImplementedError(
                "ANBIMA OAuth is wired, but the captação-líquida data endpoint "
                "is unknown. Provide AnbimaFlowSource(data_path=...) once the "
                "API product / path is confirmed; then map the response to "
                f"{FLOW_COLUMNS}."
            )
        client = self._client or self._make_client()
        raw = client.get(self._data_path)
        # Reshaping to FLOW_COLUMNS is endpoint-specific; implement once the
        # response schema is known.
        raise NotImplementedError(
            f"Fetched ANBIMA payload from {self._data_path!r}; add reshape to "
            f"{FLOW_COLUMNS} for this endpoint's schema. Sample keys: "
            f"{list(raw)[:8] if isinstance(raw, dict) else type(raw).__name__}"
        )


class FredMacroSource:
    """Real US macro from FRED (needs a free API key in FRED_API_KEY).

    Returns: period, ust_10y, fed_funds, sp500_return, usd_broad_return.
    """

    def __init__(self, start: str = "2010-01", end: str | None = None,
                 api_key: str | None = None, timeout: float = 30.0):
        self.start = start
        self.end = end
        self.api_key = api_key
        self.timeout = timeout

    def load(self) -> pd.DataFrame:
        from fund_flow.config import get_secret
        from fund_flow.data.bcb import to_monthly_last
        from fund_flow.data.fred import (
            FED_FUNDS,
            SP500,
            UST_10Y,
            USD_BROAD,
            fetch_fred,
        )

        key = self.api_key or get_secret("FRED_API_KEY")
        start_iso = pd.Period(self.start, freq="M").start_time.date().isoformat()
        end_iso = (
            pd.Period(self.end, freq="M").end_time.date().isoformat()
            if self.end else None
        )

        def fetch(sid):
            return to_monthly_last(fetch_fred(sid, key, start_iso, end_iso, self.timeout))

        ust = fetch(UST_10Y) / 100.0          # % → fraction
        ff = fetch(FED_FUNDS) / 100.0
        sp = fetch(SP500)
        usd = fetch(USD_BROAD)

        df = pd.DataFrame({
            "ust_10y": ust,
            "fed_funds": ff,
            "sp500_return": sp.pct_change(),
            "usd_broad_return": usd.pct_change(),
        }).dropna()
        df.index = df.index.astype(str)
        return df.reset_index(names="period")


class IbovespaSource:
    """Real Ibovespa monthly return via yfinance (^BVSP). Returns:
        period, ibovespa_return

    Isolated behind the MacroSource seam — if Yahoo's unofficial endpoint
    breaks, drop this source from the composite and the rest still works.
    """

    def __init__(self, start: str = "2010-01", end: str | None = None):
        self.start = start
        self.end = end

    def load(self) -> pd.DataFrame:
        from fund_flow.data.bcb import to_monthly_last
        from fund_flow.data.yahoo import IBOVESPA, fetch_yahoo_close

        start_iso = pd.Period(self.start, freq="M").start_time.date().isoformat()
        end_iso = (
            pd.Period(self.end, freq="M").end_time.date().isoformat()
            if self.end else None
        )
        close = to_monthly_last(fetch_yahoo_close(IBOVESPA, start_iso, end_iso))
        df = pd.DataFrame({"ibovespa_return": close.pct_change()}).dropna()
        df.index = df.index.astype(str)
        return df.reset_index(names="period")


class CompositeMacroSource:
    """Merge several MacroSources on `period` into one real macro table.

    e.g. CompositeMacroSource([BcbMacroSource(...), FredMacroSource(...)])
    yields the combined Brazil + US macro frame. The pipeline is
    macro-column-agnostic, so any merged column set flows through.
    """

    def __init__(self, sources: list[MacroSource], how: str = "inner"):
        if not sources:
            raise ValueError("CompositeMacroSource needs at least one source")
        self._sources = sources
        self._how = how

    def load(self) -> pd.DataFrame:
        merged: pd.DataFrame | None = None
        for src in self._sources:
            df = src.load()
            merged = df if merged is None else merged.merge(df, on="period", how=self._how)
        return merged.sort_values("period").reset_index(drop=True)


class BcbMacroSource:
    """Real Brazilian macro from the public BCB SGS API (no key required).

    Sources Selic (432), IPCA (433), and USD/BRL (1), aligns them to monthly
    periods, and returns the columns it can genuinely provide:
        period, selic_rate, delta_selic, ipca_monthly, usdbrl_return

    This is a deliberate SUBSET of the synthetic schema — it omits
    ibovespa_return / ust_10y (need B3/FRED) and the latent `regime` label.
    The pipeline is macro-column-agnostic, so this subset flows through
    cleaning → features → predictor unchanged.
    """

    def __init__(self, start: str = "2010-01", end: str | None = None,
                 timeout: float = 30.0):
        self.start = start
        self.end = end
        self.timeout = timeout

    def load(self) -> pd.DataFrame:
        from fund_flow.data.bcb import (
            IPCA_MONTHLY,
            SELIC_TARGET,
            USD_BRL,
            fetch_sgs,
            to_monthly_last,
        )

        start_iso = pd.Period(self.start, freq="M").start_time.date().isoformat()
        end_iso = (
            pd.Period(self.end, freq="M").end_time.date().isoformat()
            if self.end else None
        )

        selic = to_monthly_last(
            fetch_sgs(SELIC_TARGET, start_iso, end_iso, self.timeout)
        ) / 100.0                      # % a.a. → fraction
        ipca = to_monthly_last(
            fetch_sgs(IPCA_MONTHLY, start_iso, end_iso, self.timeout)
        ) / 100.0                      # % a.m. → fraction
        usd = to_monthly_last(
            fetch_sgs(USD_BRL, start_iso, end_iso, self.timeout)
        )

        df = pd.DataFrame({
            "selic_rate": selic,
            "delta_selic": selic.diff(),
            "ipca_monthly": ipca,
            "usdbrl_return": usd.pct_change(),
        }).dropna()

        df.index = df.index.astype(str)         # PeriodIndex → "YYYY-MM"
        return df.reset_index(names="period")[MACRO_COLUMNS_BCB]


# Columns the real BCB adapter provides (subset of the synthetic schema).
MACRO_COLUMNS_BCB = [
    "period", "selic_rate", "delta_selic", "ipca_monthly", "usdbrl_return",
]

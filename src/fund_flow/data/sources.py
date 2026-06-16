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


class CvmFlowSource:
    """Real fund flows from CVM Informe Diário, aggregated to Anbima categories.

    For each month in [start, end], downloads the informe, maps each fund's CNPJ
    to an Anbima category via `mapper`, and sums to monthly per-category flows:
        net_flow_brl         = Σ(CAPTC_DIA − RESG_DIA)
        redemption_gross_brl = Σ(RESG_DIA)
    (gross subscriptions are recoverable as net + redemption_gross.)

    `scale` divides BRL values (e.g. 1e9 → R$ billions). Funds whose CNPJ is not
    in the mapper are dropped. CVM may 403 datacenter IPs — run from an allowed
    network.
    """

    def __init__(self, start: str, end: str, mapper, scale: float = 1.0,
                 timeout: float = 60.0):
        self.start = start
        self.end = end
        self.mapper = mapper
        self.scale = scale
        self.timeout = timeout

    def _aggregate_month(self, informe, cnpj_to_cat, period_str: str) -> list[dict]:
        informe = informe.copy()
        informe["category"] = informe["cnpj"].map(cnpj_to_cat)
        informe = informe.dropna(subset=["category"])
        grp = informe.groupby("category").agg(
            captacao=("captacao", "sum"),
            resgate=("resgate", "sum"),
        )
        return [
            {
                "period": period_str,
                "category": category,
                "net_flow_brl": (r["captacao"] - r["resgate"]) / self.scale,
                "redemption_gross_brl": r["resgate"] / self.scale,
            }
            for category, r in grp.iterrows()
        ]

    def load(self) -> pd.DataFrame:
        from fund_flow.data.cvm import (
            LAST_HIST_YEAR,
            fetch_informe_diario,
            fetch_informe_year,
        )

        cnpj_to_cat = self.mapper.mapping()
        months = pd.period_range(self.start, self.end, freq="M")
        rows = []
        # Older months come from one yearly HIST archive each; fetch it once per
        # year and reuse across its months rather than re-downloading per month.
        hist_cache: dict[int, dict[str, "pd.DataFrame"]] = {}
        for m in months:
            ym = f"{m.year}{m.month:02d}"
            if m.year <= LAST_HIST_YEAR:
                if m.year not in hist_cache:
                    hist_cache = {m.year: fetch_informe_year(m.year, self.timeout)}
                informe = hist_cache[m.year].get(ym)
                if informe is None:
                    continue
            else:
                informe = fetch_informe_diario(ym, self.timeout)
            rows.extend(self._aggregate_month(informe, cnpj_to_cat, str(m)))
        return pd.DataFrame(rows, columns=FLOW_COLUMNS)


class FredMacroSource:
    """Real US macro from FRED (needs a free API key in FRED_API_KEY).

    Returns: period, fed_funds, ust_10y, vix — all with deep history
    (FEDFUNDS 1954, DGS10 1962, VIXCLS 1990), so they cover the full flow window.
    VIX is the CBOE volatility index (risk-off proxy), kept as a level.

    FRED's S&P 500 (SP500) is licensed with a rolling ~10-year window and the
    broad-dollar index (DTWEXBGS) starts only in 2006; including either here
    would truncate an inner-joined macro panel to ~2016. Source the S&P 500 and
    dollar index from YahooMacroSource instead (^GSPC / DX-Y.NYB, both 1985+).
    """

    def __init__(self, start: str = "2004-01", end: str | None = None,
                 api_key: str | None = None, timeout: float = 30.0):
        self.start = start
        self.end = end
        self.api_key = api_key
        self.timeout = timeout

    def load(self) -> pd.DataFrame:
        from fund_flow.config import get_secret
        from fund_flow.data.bcb import to_monthly_last
        from fund_flow.data.fred import FED_FUNDS, UST_10Y, VIX, fetch_fred

        key = self.api_key or get_secret("FRED_API_KEY")
        start_iso = pd.Period(self.start, freq="M").start_time.date().isoformat()
        end_iso = (
            pd.Period(self.end, freq="M").end_time.date().isoformat()
            if self.end else None
        )

        def fetch(sid):
            return to_monthly_last(fetch_fred(sid, key, start_iso, end_iso, self.timeout))

        df = pd.DataFrame({
            "fed_funds": fetch(FED_FUNDS) / 100.0,     # % → fraction
            "ust_10y": fetch(UST_10Y) / 100.0,
            "vix": fetch(VIX),                          # index level (risk-off)
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


class YahooMacroSource:
    """Real market returns via yfinance — monthly % change of each ticker.

    Defaults to the three indices with deep history that FRED can't serve over
    the full flow window:
        ibovespa_return  ^BVSP      (1993+)
        sp500_return     ^GSPC      (1985+)
        dxy_return       DX-Y.NYB   (1985+)  US dollar index

    Pass a custom {column_name: ticker} map to fetch a different set. Isolated
    behind the MacroSource seam — if Yahoo's unofficial endpoint breaks, drop
    this source from the composite and the rest still works.
    """

    DEFAULT_TICKERS = {
        "ibovespa_return": "^BVSP",
        "sp500_return": "^GSPC",
        "dxy_return": "DX-Y.NYB",
    }

    def __init__(self, tickers: dict[str, str] | None = None,
                 start: str = "2004-01", end: str | None = None):
        self.tickers = tickers or dict(self.DEFAULT_TICKERS)
        self.start = start
        self.end = end

    def load(self) -> pd.DataFrame:
        from fund_flow.data.bcb import to_monthly_last
        from fund_flow.data.yahoo import fetch_yahoo_close

        start_iso = pd.Period(self.start, freq="M").start_time.date().isoformat()
        end_iso = (
            pd.Period(self.end, freq="M").end_time.date().isoformat()
            if self.end else None
        )
        cols = {}
        for name, ticker in self.tickers.items():
            close = to_monthly_last(fetch_yahoo_close(ticker, start_iso, end_iso))
            cols[name] = close.pct_change()
        df = pd.DataFrame(cols).dropna()
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
    The credit-spread lever (EMBI+) is NOT here: it is not on SGS, so it comes
    from IPEAdata via IpeaMacroSource. The pipeline is macro-column-agnostic,
    so this subset flows through cleaning → features → predictor unchanged.
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


class IpeaMacroSource:
    """EMBI+ Risco-Brasil sovereign credit spread from IPEAdata (free, no key).

    EMBI+ Brazil (JP Morgan) is THE Brazilian systematic credit-risk premium —
    the closest free, long-history proxy for the debenture-vs-government spread
    that drives Crédito Privado / Renda Fixa flows. It is daily back to 1994 but
    the free series is DISCONTINUED at 2024-07, so an inner-joined macro panel
    truncates there. Use it for backtests / ABM calibration, not live forecasts
    past mid-2024.

    Emits:
        period, embi_spread (bps → fraction, level), delta_embi (monthly change)
    """

    def __init__(self, start: str | None = None, end: str | None = None,
                 timeout: float = 60.0):
        self.start = start
        self.end = end
        self.timeout = timeout

    def load(self) -> pd.DataFrame:
        from fund_flow.data.bcb import to_monthly_last
        from fund_flow.data.ipea import EMBI_BRAZIL, fetch_ipea_series

        embi = to_monthly_last(
            fetch_ipea_series(EMBI_BRAZIL, self.timeout)
        ) / 10000.0                     # bps → fraction (300 bps = 0.03)

        df = pd.DataFrame({
            "embi_spread": embi,                # sovereign credit-risk level
            "delta_embi": embi.diff(),          # monthly change = risk-off impulse
        }).dropna()
        df.index = df.index.astype(str)         # PeriodIndex → "YYYY-MM"
        df = df.reset_index(names="period")
        if self.start is not None:
            df = df[df["period"] >= self.start]
        if self.end is not None:
            df = df[df["period"] <= self.end]
        return df[MACRO_COLUMNS_IPEA].reset_index(drop=True)


# Columns the IPEAdata credit-spread adapter provides.
MACRO_COLUMNS_IPEA = ["period", "embi_spread", "delta_embi"]

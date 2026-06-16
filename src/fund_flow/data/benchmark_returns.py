"""Monthly total-return series per canonical benchmark ID (Phase 2.6, Step 2).

One monthly total return per ID, all from free/public sources already in the
pipeline. Where no clean free total-return series exists, a documented PROXY is
used and recorded in the returned `sources` map (real vs proxy is never hidden).

Coverage of the canonical IDs:
  CDI, SELIC   — BCB SGS 12 / 11 daily rates, compounded within each month.
  IPCA         — BCB SGS 433 monthly %, /100.
  IBOV         — Yahoo ^BVSP month-end pct_change.
  SP500_BRL    — Yahoo ^GSPC (USD) combined with BRL/USD (SGS 1) FX return.
  MSCI_WORLD_BRL — Yahoo URTH (fallback EFA) in USD × FX  [proxy: ETF].
  IMA-B, IMA-B5 — Yahoo B3 ETFs IMAB11.SA / B5P211.SA   [proxy: ETF, ~2018+].
  IDA-IPCA     — proxy: IMA-B (no free IDA total-return series).
  IBRX100      — proxy: IBOV (IBrX not on a free API).
  IHFA         — proxy: CDI (ANBIMA IHFA has no free endpoint).
  ZERO         — 0 every month (excluded funds; never a fabricated benchmark).

The conversion helpers are pure and unit-tested offline; the live fetch is
covered by a `@pytest.mark.live` range sanity check.
"""
from __future__ import annotations

import pandas as pd

from fund_flow.data.bcb import IPCA_MONTHLY, USD_BRL, fetch_sgs, to_monthly_last
from fund_flow.data.yahoo import YahooFetchError, fetch_yahoo_close

CDI_SGS = 12      # Taxa de juros - CDI, % a.d. (daily)
SELIC_SGS = 11    # Taxa de juros - Selic, % a.d. (daily)

CANONICAL_IDS = (
    "CDI", "SELIC", "IPCA", "IBOV", "IBRX100", "IMA-B", "IMA-B5",
    "IDA-IPCA", "SP500_BRL", "MSCI_WORLD_BRL", "IHFA", "ZERO",
)


# --------------------------------------------------------------------------- #
# Pure conversion helpers (offline-testable)
# --------------------------------------------------------------------------- #
def monthly_return_from_daily_rate(daily_pct: pd.Series) -> pd.Series:
    """Compound a daily % rate (e.g. CDI 0.0393 %/day) into a monthly total return.

    Monthly factor = Π_d (1 + r_d/100) over the business days in the month; return
    = factor − 1. Indexed by a monthly PeriodIndex.
    """
    if daily_pct.empty:
        return pd.Series(dtype=float)
    f = 1.0 + daily_pct.astype(float) / 100.0
    g = f.groupby(daily_pct.index.to_period("M")).prod() - 1.0
    g.index = pd.PeriodIndex(g.index, freq="M")
    return g.sort_index()


def monthly_return_from_prices(close: pd.Series) -> pd.Series:
    """Month-end last price → monthly simple return (PeriodIndex)."""
    return to_monthly_last(close).pct_change()


def combine_local_fx(usd_return: pd.Series, fx_return: pd.Series) -> pd.Series:
    """USD-denominated asset return + BRL/USD FX return → BRL total return.

    (1 + r_usd)(1 + r_fx) − 1, aligned on the monthly index.
    """
    df = pd.concat([usd_return.rename("a"), fx_return.rename("fx")], axis=1).dropna()
    return (1 + df["a"]) * (1 + df["fx"]) - 1


# --------------------------------------------------------------------------- #
# Live builder
# --------------------------------------------------------------------------- #
def _empty(s) -> bool:
    return s is None or not hasattr(s, "dropna") or s.dropna().empty


def _try(label_holder: dict, key: str, builder, source_desc: str, fallback=None,
         fallback_desc: str | None = None):
    """Run `builder`; on failure use `fallback` (a Series or a zero-arg callable,
    evaluated lazily) and record the source actually used."""
    try:
        s = builder()
        if _empty(s):
            raise ValueError("empty series")
        label_holder[key] = source_desc
        return s
    except Exception as exc:  # noqa: BLE001
        if fallback is None:
            label_holder[key] = f"UNAVAILABLE ({type(exc).__name__})"
            return pd.Series(dtype=float)
        fb = fallback() if callable(fallback) else fallback
        if _empty(fb):
            label_holder[key] = f"UNAVAILABLE ({type(exc).__name__}; fallback empty)"
            return pd.Series(dtype=float)
        label_holder[key] = fallback_desc or f"proxy ({source_desc} failed)"
        return fb


def fetch_benchmark_returns(
    start: str | None = None, end: str | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Build the monthly benchmark-return panel and the source-used map.

    Returns (returns_df indexed by monthly PeriodIndex with one column per
    canonical ID, sources: {ID -> human-readable source/proxy description}).
    """
    src: dict[str, str] = {}
    cols: dict[str, pd.Series] = {}

    # Brazilian cash benchmarks (full history, real series).
    cols["CDI"] = _try(src, "CDI",
                       lambda: monthly_return_from_daily_rate(fetch_sgs(CDI_SGS, start, end)),
                       "BCB SGS 12 (CDI daily), compounded monthly")
    cols["SELIC"] = _try(src, "SELIC",
                         lambda: monthly_return_from_daily_rate(fetch_sgs(SELIC_SGS, start, end)),
                         "BCB SGS 11 (Selic daily), compounded monthly")
    cols["IPCA"] = _try(src, "IPCA",
                        lambda: (to_monthly_last(fetch_sgs(IPCA_MONTHLY, start, end)) / 100.0),
                        "BCB SGS 433 (IPCA % a.m.) / 100")

    # FX (BRL per USD) monthly return, reused for USD-denominated assets.
    fx = _try(src, "_fx",
              lambda: monthly_return_from_prices(fetch_sgs(USD_BRL, start, end)),
              "BCB SGS 1 (BRL/USD)")
    src.pop("_fx", None)

    # Brazilian equity (real series).
    ibov = _try(src, "IBOV",
                lambda: monthly_return_from_prices(fetch_yahoo_close("^BVSP", start, end)),
                "Yahoo ^BVSP month-end")
    cols["IBOV"] = ibov
    # IBrX-100: no free API → proxy IBOV.
    cols["IBRX100"] = ibov.copy()
    src["IBRX100"] = "proxy: IBOV (IBrX-100 has no free total-return API)"

    # US / global equity in BRL.
    cols["SP500_BRL"] = _try(
        src, "SP500_BRL",
        lambda: combine_local_fx(
            monthly_return_from_prices(fetch_yahoo_close("^GSPC", start, end)), fx),
        "Yahoo ^GSPC (USD) × BRL/USD FX")
    cols["MSCI_WORLD_BRL"] = _try(
        src, "MSCI_WORLD_BRL",
        lambda: combine_local_fx(
            monthly_return_from_prices(fetch_yahoo_close("URTH", start, end)), fx),
        "proxy: Yahoo URTH ETF (USD) × FX",
        fallback=lambda: combine_local_fx(
            monthly_return_from_prices(fetch_yahoo_close("EFA", start, end)), fx),
        fallback_desc="proxy: Yahoo EFA ETF (USD) × FX (URTH unavailable)")

    # IPCA-linked bond indices via B3 ETFs (proxy, ~2018+).
    imab = _try(src, "IMA-B",
                lambda: monthly_return_from_prices(fetch_yahoo_close("IMAB11.SA", start, end)),
                "proxy: Yahoo IMAB11.SA ETF month-end",
                fallback=cols["IPCA"],
                fallback_desc="proxy: IPCA (IMAB11.SA ETF unavailable)")
    cols["IMA-B"] = imab
    cols["IMA-B5"] = _try(
        src, "IMA-B5",
        lambda: monthly_return_from_prices(fetch_yahoo_close("B5P211.SA", start, end)),
        "proxy: Yahoo B5P211.SA ETF month-end",
        fallback=imab, fallback_desc="proxy: IMA-B (B5P211.SA ETF unavailable)")

    # IDA-IPCA: no free total-return series → proxy IMA-B.
    cols["IDA-IPCA"] = imab.copy()
    src["IDA-IPCA"] = "proxy: IMA-B (no free IDA-IPCA total-return series)"
    # IHFA: ANBIMA hedge-fund index, no free endpoint → conservative CDI proxy.
    cols["IHFA"] = cols["CDI"].copy()
    src["IHFA"] = "proxy: CDI (ANBIMA IHFA has no free API)"

    df = pd.concat(cols, axis=1).sort_index()
    df["ZERO"] = 0.0
    src["ZERO"] = "constant 0 (excluded funds)"
    return df, src

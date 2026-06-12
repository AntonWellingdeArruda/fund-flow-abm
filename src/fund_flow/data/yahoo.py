"""Yahoo Finance via yfinance — UNOFFICIAL community scraper.

Yahoo offers no official free API; yfinance scrapes Yahoo's internal endpoints
and can break without notice when Yahoo changes them. It is therefore isolated
behind this thin module + the MacroSource seam, so a breakage here cannot
affect BCB/FRED ingestion. Used for Ibovespa (^BVSP), which neither FRED nor a
free B3 API provides.
"""
from __future__ import annotations

import pandas as pd

IBOVESPA = "^BVSP"


class YahooFetchError(RuntimeError):
    """Raised when a yfinance download fails or returns nothing."""


def fetch_yahoo_close(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """Daily adjusted close as a float Series (DatetimeIndex)."""
    try:
        import yfinance as yf
        df = yf.download(
            ticker, start=start, end=end, interval="1d",
            progress=False, auto_adjust=True, threads=False,
        )
    except Exception as exc:  # yfinance raises a variety of network errors
        raise YahooFetchError(f"yfinance download failed for {ticker}: {exc}") from exc

    if df is None or df.empty:
        raise YahooFetchError(f"yfinance returned no data for {ticker}")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):     # single-ticker MultiIndex columns
        close = close.iloc[:, 0]
    close = close.astype(float).sort_index()
    close.index = pd.to_datetime(close.index)
    close.index.name = "date"
    return close

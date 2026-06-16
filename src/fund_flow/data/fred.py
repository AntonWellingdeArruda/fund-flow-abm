"""Real US macro from the FRED API (St. Louis Fed). Requires a free API key.

Endpoint:
    https://api.stlouisfed.org/fred/series/observations
        ?series_id=...&api_key=...&file_type=json&observation_start=YYYY-MM-DD

Missing observations come back as ".". HTTP and parsing are split so the
parser is unit-testable offline.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from fund_flow.data.bcb import to_monthly_last  # shared month-end resample

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

# Series used by the macro adapter.
UST_10Y = "DGS10"          # 10-Year Treasury constant maturity, % (daily)
FED_FUNDS = "FEDFUNDS"     # Effective federal funds rate, % (monthly)
VIX = "VIXCLS"             # CBOE Volatility Index, level (daily, 1990+) — risk-off
SP500 = "SP500"            # S&P 500 index level (daily, ~10y history)
USD_BROAD = "DTWEXBGS"     # Nominal broad USD index (daily) — DXY-style proxy


class FredFetchError(RuntimeError):
    """Raised when a FRED request fails (network, HTTP, or bad payload)."""


def _parse_fred(payload: dict) -> pd.Series:
    """Parse FRED observations JSON into a float Series (DatetimeIndex).

    '.' values (FRED's missing marker) become NaN.
    """
    obs = payload.get("observations", [])
    if not obs:
        return pd.Series(dtype=float)
    dates = pd.to_datetime([o["date"] for o in obs])
    values = pd.to_numeric(
        [o["value"] if o["value"] != "." else None for o in obs],
        errors="coerce",
    )
    s = pd.Series(values, index=dates).sort_index().dropna()
    s.index.name = "date"
    return s


def _http_get_json(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FredFetchError(f"FRED request failed: {url}\n{exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FredFetchError(f"FRED returned non-JSON payload from {url}") from exc


def fetch_fred(
    series_id: str,
    api_key: str,
    start: str | None = None,
    end: str | None = None,
    timeout: float = 30.0,
) -> pd.Series:
    """Fetch one FRED series as a float Series (DatetimeIndex)."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    if start:
        params["observation_start"] = start
    if end:
        params["observation_end"] = end
    url = FRED_URL + "?" + urllib.parse.urlencode(params)
    return _parse_fred(_http_get_json(url, timeout))

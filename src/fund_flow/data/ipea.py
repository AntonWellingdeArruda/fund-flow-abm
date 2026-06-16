"""Real IPEAdata client — free Brazilian macro series, no API key.

The one series we need here is EMBI+ Risco-Brasil (JP Morgan's Brazil sovereign
spread), which is the closest free, long-history proxy for the credit premium
that drives Crédito Privado / Renda Fixa flows. It is NOT on BCB SGS — it lives
on IPEAdata's OData4 API:

    http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='JPM366_EMBI366')

The endpoint returns the WHOLE series (~10k daily rows, bps) as JSON rows of
{SERCODIGO, VALDATA, VALVALOR, ...}; we filter/resample downstream. The free
series is daily from 1994-04 but DISCONTINUED at 2024-07.

HTTP and parsing are split so the parser is unit-testable offline.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import pandas as pd

IPEA_URL = (
    "http://www.ipeadata.gov.br/api/odata4/"
    "ValoresSerie(SERCODIGO='{code}')"
)

EMBI_BRAZIL = "JPM366_EMBI366"   # EMBI+ Risco-Brasil, bps, daily (1994-04 .. 2024-07)


class IpeaFetchError(RuntimeError):
    """Raised when an IPEAdata request fails (network, HTTP, or bad payload)."""


_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 Chrome/124 Safari/537.36",
}


def _parse_ipea(value: list[dict]) -> pd.Series:
    """Parse IPEAdata OData rows into a float Series indexed by date.

    Rows look like {'VALDATA': '1994-04-29T00:00:00-03:00', 'VALVALOR': 1120.0}.
    Null VALVALOR (non-trading days) are dropped; the tz-aware timestamp is
    reduced to a tz-naive calendar date so it merges cleanly with the BCB/FRED
    month-end series.
    """
    rows = [r for r in value if r.get("VALVALOR") is not None]
    if not rows:
        return pd.Series(dtype=float)
    dates = pd.to_datetime([r["VALDATA"][:10] for r in rows])  # tz-naive date
    values = pd.to_numeric([r["VALVALOR"] for r in rows], errors="coerce")
    s = pd.Series(values, index=dates).sort_index()
    s.index.name = "date"
    return s


def fetch_ipea_series(code: str, timeout: float = 60.0, retries: int = 3) -> pd.Series:
    """Fetch one IPEAdata series as a float Series (DatetimeIndex), with retry.

    The OData endpoint serves the full history in one response; date filtering
    is done downstream. Transient network failures back off and retry.
    """
    url = IPEA_URL.format(code=code)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            return _parse_ipea(json.loads(raw)["value"])
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError, KeyError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    raise IpeaFetchError(
        f"IPEAdata request failed after {retries} tries: {url}\n{last}"
    ) from last

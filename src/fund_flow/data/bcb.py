"""Real BCB SGS client — public Brazilian Central Bank time series, no API key.

Endpoint:
    https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json
    [&dataInicial=DD/MM/YYYY&dataFinal=DD/MM/YYYY]

Daily series (e.g. Selic target 432, USD/BRL 1) are capped at ~10 years per
request, so fetch_sgs chunks long ranges. Monthly series (IPCA 433, Selic
accumulated 4189) have no such cap.

HTTP and parsing are split so the parser is unit-testable offline.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import date

import pandas as pd

SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"

# Series codes used by the macro adapter.
SELIC_TARGET = 432       # Selic meta, % a.a., daily
IPCA_MONTHLY = 433       # IPCA, % a.m., monthly
USD_BRL = 1              # Dólar comercial (venda), daily


class BcbFetchError(RuntimeError):
    """Raised when a BCB SGS request fails (network, HTTP, or bad payload)."""


def _parse_sgs(payload: list[dict]) -> pd.Series:
    """Parse SGS JSON rows ([{'data': 'DD/MM/YYYY', 'valor': '1.23'}, ...]).

    Returns a float Series indexed by a sorted DatetimeIndex.
    """
    if not payload:
        return pd.Series(dtype=float)
    dates = pd.to_datetime([row["data"] for row in payload], format="%d/%m/%Y")
    values = pd.to_numeric([row["valor"] for row in payload], errors="coerce")
    s = pd.Series(values, index=dates).sort_index()
    s.index.name = "date"
    return s


def _http_get_json(url: str, timeout: float) -> list[dict]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise BcbFetchError(f"BCB request failed: {url}\n{exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BcbFetchError(f"BCB returned non-JSON payload from {url}") from exc


def _chunks(start: date, end: date, years: int = 10):
    """Yield (chunk_start, chunk_end) windows of at most `years` years."""
    cur = start
    while cur <= end:
        stop = min(date(cur.year + years, cur.month, 1), end)
        yield cur, stop
        # advance one day past stop
        stop_ts = pd.Timestamp(stop) + pd.Timedelta(days=1)
        cur = stop_ts.date()


def fetch_sgs(
    code: int,
    start: str | None = None,
    end: str | None = None,
    timeout: float = 30.0,
) -> pd.Series:
    """Fetch one SGS series as a float Series (DatetimeIndex).

    start/end are ISO 'YYYY-MM-DD' strings. Long ranges are chunked into
    ≤10-year windows so daily series do not hit the API cap.
    """
    if start is None and end is None:
        url = SGS_URL.format(code=code) + "?formato=json"
        return _parse_sgs(_http_get_json(url, timeout))

    start_d = pd.Timestamp(start or "1990-01-01").date()
    end_d = pd.Timestamp(end or date.today().isoformat()).date()

    parts: list[pd.Series] = []
    for cs, ce in _chunks(start_d, end_d):
        url = (
            SGS_URL.format(code=code)
            + f"?formato=json&dataInicial={cs.strftime('%d/%m/%Y')}"
            + f"&dataFinal={ce.strftime('%d/%m/%Y')}"
        )
        parts.append(_parse_sgs(_http_get_json(url, timeout)))
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts).sort_index()[lambda s: ~s.index.duplicated()]


def to_monthly_last(s: pd.Series) -> pd.Series:
    """Resample a (possibly daily) series to month-end last observation,
    indexed by a monthly PeriodIndex."""
    if s.empty:
        return s
    monthly = s.resample("ME").last()
    monthly.index = pd.PeriodIndex(monthly.index, freq="M")
    return monthly

from __future__ import annotations

import pandas as pd


def is_come_cotas(period: str, category: str, exempt: list[str]) -> bool:
    """True for May and November periods for non-exempt categories."""
    p = pd.Period(period, freq="M")
    return p.month in (5, 11) and category not in exempt


def business_days_in_month(period: str) -> int:
    """Count of business days in a YYYY-MM period."""
    p = pd.Period(period, freq="M")
    dates = pd.bdate_range(p.start_time, p.end_time)
    return len(dates)


def add_business_days(date: str, n: int) -> str:
    """Return ISO date string for date + n business days."""
    ts = pd.Timestamp(date)
    result = ts + pd.offsets.BDay(n)
    return result.strftime("%Y-%m-%d")

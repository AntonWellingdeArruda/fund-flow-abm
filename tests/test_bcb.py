import pandas as pd
import pytest

from fund_flow.data.bcb import (
    BcbFetchError,
    SELIC_TARGET,
    _chunks,
    _parse_sgs,
    fetch_sgs,
    to_monthly_last,
)

# A recorded SGS payload (daily Selic-target style rows) for offline parsing.
SAMPLE = [
    {"data": "02/01/2023", "valor": "13.75"},
    {"data": "03/01/2023", "valor": "13.75"},
    {"data": "01/02/2023", "valor": "13.75"},
    {"data": "15/02/2023", "valor": "13.75"},
]


class TestParse:
    def test_parses_dates_and_values(self):
        s = _parse_sgs(SAMPLE)
        assert isinstance(s.index, pd.DatetimeIndex)
        assert s.iloc[0] == pytest.approx(13.75)
        assert s.index[0] == pd.Timestamp("2023-01-02")

    def test_sorted(self):
        s = _parse_sgs(SAMPLE)
        assert s.index.is_monotonic_increasing

    def test_empty_payload(self):
        s = _parse_sgs([])
        assert s.empty


class TestMonthly:
    def test_month_end_last(self):
        s = _parse_sgs(SAMPLE)
        m = to_monthly_last(s)
        assert isinstance(m.index, pd.PeriodIndex)
        # two distinct months in the sample
        assert len(m) == 2
        assert str(m.index[0]) == "2023-01"


class TestChunks:
    def test_single_window_within_10y(self):
        from datetime import date
        wins = list(_chunks(date(2020, 1, 1), date(2022, 1, 1)))
        assert len(wins) == 1

    def test_splits_long_range(self):
        from datetime import date
        wins = list(_chunks(date(2000, 1, 1), date(2025, 1, 1)))
        assert len(wins) >= 3
        # windows must be contiguous and cover the range
        assert wins[0][0] == date(2000, 1, 1)
        assert wins[-1][1] == date(2025, 1, 1)


# --- Network-gated live test: skips cleanly when offline ---

@pytest.fixture(scope="module")
def online():
    try:
        s = fetch_sgs(SELIC_TARGET, start="2024-01-01", end="2024-03-31", timeout=15)
        if s.empty:
            pytest.skip("BCB returned empty payload")
        return True
    except BcbFetchError:
        pytest.skip("BCB SGS unreachable (offline)")


@pytest.mark.live
class TestLive:
    def test_real_selic_in_plausible_range(self, online):
        s = fetch_sgs(SELIC_TARGET, start="2024-01-01", end="2024-03-31", timeout=15)
        m = to_monthly_last(s) / 100.0
        # Selic target was ~10.5–11.75% in early 2024
        assert (m > 0.08).all() and (m < 0.15).all()

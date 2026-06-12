import pandas as pd
import pytest

from fund_flow.config import has_secret
from fund_flow.data.fred import UST_10Y, FredFetchError, _parse_fred, fetch_fred

SAMPLE = {
    "observations": [
        {"date": "2024-03-01", "value": "4.20"},
        {"date": "2024-04-01", "value": "."},      # FRED missing marker
        {"date": "2024-05-01", "value": "4.48"},
    ]
}


class TestParse:
    def test_drops_missing_marker(self):
        s = _parse_fred(SAMPLE)
        assert len(s) == 2                          # the "." row is dropped
        assert s.iloc[0] == pytest.approx(4.20)

    def test_datetime_index_sorted(self):
        s = _parse_fred(SAMPLE)
        assert isinstance(s.index, pd.DatetimeIndex)
        assert s.index.is_monotonic_increasing

    def test_empty(self):
        assert _parse_fred({"observations": []}).empty


@pytest.fixture(scope="module")
def fred_online():
    if not has_secret("FRED_API_KEY"):
        pytest.skip("FRED_API_KEY not set")
    from fund_flow.config import get_secret
    try:
        s = fetch_fred(UST_10Y, get_secret("FRED_API_KEY"),
                       start="2024-01-01", end="2024-03-31", timeout=15)
        if s.empty:
            pytest.skip("FRED returned empty")
        return get_secret("FRED_API_KEY")
    except FredFetchError:
        pytest.skip("FRED unreachable (offline)")


class TestLive:
    def test_ust10y_plausible(self, fred_online):
        s = fetch_fred(UST_10Y, fred_online, start="2024-01-01", end="2024-03-31")
        # 10Y yield was ~3.8–4.4% in early 2024
        assert (s > 3.0).all() and (s < 5.5).all()

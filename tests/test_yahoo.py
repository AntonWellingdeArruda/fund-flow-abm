import numpy as np
import pandas as pd
import pytest

from fund_flow.data import sources, yahoo
from fund_flow.data.sources import IbovespaSource
from fund_flow.data.yahoo import IBOVESPA, YahooFetchError, fetch_yahoo_close


class TestIbovespaSourceOffline:
    def test_builds_monthly_return(self, monkeypatch):
        # fake ~4 months of daily closes
        idx = pd.date_range("2024-01-01", "2024-04-30", freq="B")
        close = pd.Series(np.linspace(120000, 130000, len(idx)), index=idx)
        monkeypatch.setattr(sources, "fetch_yahoo_close", lambda *a, **k: close,
                            raising=False)
        # IbovespaSource imports fetch_yahoo_close inside load(); patch the module
        monkeypatch.setattr(yahoo, "fetch_yahoo_close", lambda *a, **k: close)

        df = IbovespaSource(start="2024-01", end="2024-04").load()
        assert list(df.columns) == ["period", "ibovespa_return"]
        assert not df.isnull().any().any()
        # rising series → positive returns
        assert (df["ibovespa_return"] > 0).all()


# --- network-gated live test ---

@pytest.fixture(scope="module")
def yahoo_online():
    try:
        s = fetch_yahoo_close(IBOVESPA, start="2024-01-01", end="2024-03-31")
        if s.empty:
            pytest.skip("yfinance returned empty")
        return True
    except YahooFetchError:
        pytest.skip("yfinance/Yahoo unreachable")


@pytest.mark.live
class TestLive:
    def test_real_ibovespa_level(self, yahoo_online):
        s = fetch_yahoo_close(IBOVESPA, start="2024-01-01", end="2024-03-31")
        # Ibovespa traded ~120k–135k in early 2024
        assert (s > 100_000).all() and (s < 150_000).all()

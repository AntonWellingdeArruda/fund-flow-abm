"""Offline tests for the extended macro sources + the real-data cache."""
import numpy as np
import pandas as pd
import pytest

from fund_flow.data import ipea, yahoo
from fund_flow.data.ipea import _parse_ipea
from fund_flow.data.sources import (
    MACRO_COLUMNS_IPEA,
    IpeaMacroSource,
    YahooMacroSource,
)


class TestIpeaEmbi:
    """IPEAdata EMBI+ parsing and the credit-spread macro adapter."""

    def test_parse_drops_nulls_and_strips_tz(self):
        value = [
            {"VALDATA": "2004-05-31T00:00:00-03:00", "VALVALOR": 500.0},
            {"VALDATA": "2004-06-30T00:00:00-03:00", "VALVALOR": None},  # non-trading
            {"VALDATA": "2004-07-30T00:00:00-03:00", "VALVALOR": 520.0},
        ]
        s = _parse_ipea(value)
        assert len(s) == 2                      # null row dropped
        assert s.index.tz is None               # tz stripped for clean merges
        assert s.iloc[0] == pytest.approx(500.0)

    def test_source_emits_embi_scaled(self, monkeypatch):
        idx = pd.date_range("2004-05-01", periods=4, freq="MS")
        series = pd.Series([500.0, 520.0, 480.0, 510.0], index=idx)  # bps
        monkeypatch.setattr(ipea, "fetch_ipea_series", lambda *a, **k: series)

        df = IpeaMacroSource(start="2004-06", end="2004-08").load()

        assert list(df.columns) == MACRO_COLUMNS_IPEA
        # 2004-05 dropped (delta NaN), then bounded to >= 2004-06
        assert df["period"].tolist() == ["2004-06", "2004-07", "2004-08"]
        jun = df.iloc[0]
        assert jun["embi_spread"] == pytest.approx(0.052)   # 520 bps → fraction
        assert jun["delta_embi"] == pytest.approx(0.002)    # (520-500)/10000
        assert not df.isnull().any().any()


class TestYahooMacroSource:
    def test_multi_ticker_monthly_returns(self, monkeypatch):
        idx = pd.date_range("2024-01-01", "2024-05-31", freq="B")

        def fake_close(ticker, start=None, end=None):
            # distinct rising level per ticker so columns are independent
            base = {"^BVSP": 120000, "^GSPC": 4800, "DX-Y.NYB": 100}[ticker]
            return pd.Series(np.linspace(base, base * 1.05, len(idx)), index=idx)

        monkeypatch.setattr(yahoo, "fetch_yahoo_close", fake_close)

        df = YahooMacroSource(start="2024-01", end="2024-05").load()
        assert list(df.columns) == [
            "period", "ibovespa_return", "sp500_return", "dxy_return",
        ]
        assert not df.isnull().any().any()
        # rising series → positive monthly returns everywhere
        assert (df.drop(columns="period") > 0).all().all()

    def test_custom_ticker_map(self, monkeypatch):
        idx = pd.date_range("2024-01-01", "2024-04-30", freq="B")
        monkeypatch.setattr(
            yahoo, "fetch_yahoo_close",
            lambda *a, **k: pd.Series(np.linspace(100, 110, len(idx)), index=idx),
        )
        df = YahooMacroSource({"sp500_return": "^GSPC"},
                              start="2024-01", end="2024-04").load()
        assert list(df.columns) == ["period", "sp500_return"]


class TestRealMacroCache:
    def test_uses_cache_when_covering(self, tmp_path, monkeypatch):
        import fund_flow.realdata as rd

        cache = tmp_path / "macro.csv"
        cached = pd.DataFrame({
            "period": ["2004-05", "2004-06", "2004-07"],
            "selic_rate": [0.16, 0.16, 0.16],
            "fed_funds": [0.01, 0.01, 0.01],
        })
        cached.to_csv(cache, index=False)

        def boom(*a, **k):
            raise AssertionError("should not fetch live when cache covers range")

        monkeypatch.setattr(rd, "build_real_macro_source", boom)
        out = rd.load_real_macro("2004-05", "2004-07", cache=cache)
        assert list(out["period"]) == ["2004-05", "2004-06", "2004-07"]

    def test_refetches_when_cache_misses(self, tmp_path, monkeypatch):
        import fund_flow.realdata as rd

        cache = tmp_path / "macro.csv"  # does not exist yet

        class FakeSrc:
            def load(self):
                return pd.DataFrame({"period": ["2004-05"], "selic_rate": [0.16]})

        monkeypatch.setattr(rd, "build_real_macro_source", lambda *a, **k: FakeSrc())
        out = rd.load_real_macro("2004-05", "2004-05", cache=cache)
        assert cache.exists()                       # cache written
        assert out["period"].tolist() == ["2004-05"]

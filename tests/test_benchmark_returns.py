"""Tests for benchmark return series (Phase 2.6, Step 2).

Conversion helpers are pure (offline); the full live fetch is a `@pytest.mark.live`
range sanity check.
"""
import numpy as np
import pandas as pd
import pytest

from fund_flow.data.benchmark_returns import (
    CANONICAL_IDS,
    combine_local_fx,
    fetch_benchmark_returns,
    monthly_return_from_daily_rate,
    monthly_return_from_prices,
)


class TestDailyRateCompounding:
    def test_constant_rate_compounds(self):
        # all business days within June 2024 at 0.04%/day → (1.0004**n - 1)
        idx = pd.bdate_range("2024-06-03", "2024-06-28")
        s = pd.Series(0.04, index=idx)
        out = monthly_return_from_daily_rate(s)
        assert len(out) == 1
        assert out.iloc[0] == pytest.approx(1.0004 ** len(idx) - 1, rel=1e-9)
        assert isinstance(out.index, pd.PeriodIndex)

    def test_cdi_scale_is_sane(self):
        # ~0.039%/day over a month ≈ 0.8%/month, NOT 0.5+ (guards the /100 bug)
        idx = pd.bdate_range("2024-06-03", periods=20)
        out = monthly_return_from_daily_rate(pd.Series(0.0393, index=idx))
        assert 0.001 < out.iloc[0] < 0.02

    def test_empty(self):
        assert monthly_return_from_daily_rate(pd.Series(dtype=float)).empty


class TestPriceReturns:
    def test_month_end_pct_change(self):
        idx = pd.to_datetime(["2024-05-31", "2024-06-28"])
        out = monthly_return_from_prices(pd.Series([100.0, 110.0], index=idx))
        assert out.iloc[-1] == pytest.approx(0.10)


class TestCombineFx:
    def test_compounds_asset_and_fx(self):
        idx = pd.PeriodIndex(["2024-06"], freq="M")
        usd = pd.Series([0.05], index=idx)   # +5% in USD
        fx = pd.Series([0.02], index=idx)    # BRL/USD +2%
        out = combine_local_fx(usd, fx)
        assert out.iloc[0] == pytest.approx(1.05 * 1.02 - 1)

    def test_alignment_dropna(self):
        usd = pd.Series([0.05, 0.01], index=pd.PeriodIndex(["2024-06", "2024-07"], freq="M"))
        fx = pd.Series([0.02], index=pd.PeriodIndex(["2024-06"], freq="M"))
        out = combine_local_fx(usd, fx)
        assert list(out.index.astype(str)) == ["2024-06"]


@pytest.mark.live
class TestLiveBenchmarkReturns:
    def test_all_ids_present_and_in_range(self):
        df, sources = fetch_benchmark_returns("2015-01-01", "2024-12-31")
        for cid in CANONICAL_IDS:
            assert cid in df.columns, f"missing {cid}"
            assert cid in sources
        # every realised monthly return is a sane fraction
        vals = df.drop(columns=["ZERO"]).to_numpy()
        finite = vals[np.isfinite(vals)]
        assert (finite > -0.5).all() and (finite < 0.5).all()
        # CDI should be positive and small most months
        cdi = df["CDI"].dropna()
        assert (cdi > 0).mean() > 0.95
        assert cdi.median() < 0.02

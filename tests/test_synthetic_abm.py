import numpy as np
import pytest

from fund_flow.schema import VALID_CATEGORIES, FlowShock
from fund_flow.synthetic.abm_data import generate_synthetic_book, stress_flow_shock


class TestSyntheticBook:
    def test_illiquid_holdings_non_negative(self, abm_book):
        assert (abm_book.illiquid_holdings.values >= 0).all()

    def test_overlap_invariant(self, cfg, abm_book):
        min_overlap = cfg["synthetic"]["abm"]["min_overlap"]
        holders_per_instrument = (abm_book.illiquid_holdings > 0).sum(axis=0)
        assert (holders_per_instrument >= min_overlap).all(), \
            f"Some instruments held by fewer than {min_overlap} funds"

    def test_nav_identity(self, cfg, abm_book):
        buffer_pct = cfg["synthetic"]["abm"]["liquid_buffer_pct"]
        illiquid_total = abm_book.illiquid_holdings.sum(axis=1)
        expected_illiquid = abm_book.nav * (1.0 - buffer_pct)
        assert np.allclose(illiquid_total, expected_illiquid, rtol=1e-9), \
            "NAV identity violated: nav*(1-buffer_pct) != illiquid_holdings.sum(axis=1)"

    def test_liquid_buffer_non_negative(self, abm_book):
        assert (abm_book.liquid_buffer_brl >= 0).all()

    def test_liquid_buffer_not_in_illiquid_holdings(self, abm_book):
        for col in abm_book.illiquid_holdings.columns:
            assert "liquid" not in col.lower() and "LFT" not in col, \
                f"Column {col!r} looks like a liquid asset in illiquid_holdings"

    def test_nav_equals_illiquid_plus_buffer(self, abm_book):
        recomputed = abm_book.illiquid_holdings.sum(axis=1) + abm_book.liquid_buffer_brl
        assert np.allclose(abm_book.nav.values, recomputed.values, rtol=1e-9)

    def test_fund_categories_valid(self, abm_book):
        for fund, cat in abm_book.fund_categories.items():
            assert cat in VALID_CATEGORIES, \
                f"Fund {fund!r} has unknown category {cat!r}"

    def test_fund_names_match_index(self, abm_book):
        assert list(abm_book.illiquid_holdings.index) == abm_book.fund_names
        assert list(abm_book.nav.index) == abm_book.fund_names

    def test_deterministic(self, cfg):
        b1 = generate_synthetic_book(cfg)
        b2 = generate_synthetic_book(cfg)
        assert b1.fund_names == b2.fund_names
        assert np.allclose(b1.illiquid_holdings.values, b2.illiquid_holdings.values)


class TestStressFlowShock:
    def test_returns_three_shocks(self, cfg):
        shocks = stress_flow_shock(cfg)
        assert len(shocks) == 3

    def test_all_shocks_valid_flow_shock(self, cfg):
        for s in stress_flow_shock(cfg):
            assert isinstance(s, FlowShock)
            assert s.category == "Crédito Privado"
            assert s.redemption_gross_brl > 0
            assert s.net_flow_brl < 0

    def test_consecutive_periods(self, cfg):
        from pandas import Period
        shocks = stress_flow_shock(cfg)
        periods = [Period(s.period, freq="M") for s in shocks]
        for i in range(1, len(periods)):
            assert periods[i] == periods[i - 1] + 1, \
                "Stress shocks must be in consecutive monthly periods"

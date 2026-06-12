import pytest

from fund_flow.schema import VALID_CATEGORIES, FlowShock, FundContext


class TestFlowShock:
    def test_happy_path_all_categories(self):
        for cat in VALID_CATEGORIES:
            s = FlowShock(period="2024-05", category=cat,
                          net_flow_brl=-1e9, redemption_gross_brl=2e9)
            assert s.category == cat

    def test_period_format_valid(self):
        FlowShock(period="2020-01", category="Ações",
                  net_flow_brl=0.0, redemption_gross_brl=0.0)

    @pytest.mark.parametrize("bad_period", ["2024/01", "2024-1", "01-2024", "202401"])
    def test_period_format_rejected(self, bad_period):
        with pytest.raises(Exception):
            FlowShock(period=bad_period, category="Ações",
                      net_flow_brl=0.0, redemption_gross_brl=0.0)

    def test_unknown_category_raises(self):
        with pytest.raises(ValueError, match="Unknown category"):
            FlowShock(period="2024-01", category="Tesouro Direto",
                      net_flow_brl=0.0, redemption_gross_brl=0.0)

    def test_negative_gross_redemption_raises(self):
        with pytest.raises(ValueError, match="redemption_gross_brl"):
            FlowShock(period="2024-01", category="Ações",
                      net_flow_brl=0.0, redemption_gross_brl=-1.0)

    def test_frozen(self):
        s = FlowShock(period="2024-01", category="Ações",
                      net_flow_brl=0.0, redemption_gross_brl=1.0)
        with pytest.raises(Exception):
            s.net_flow_brl = 99.0  # type: ignore[misc]

    def test_net_flow_can_be_negative(self):
        s = FlowShock(period="2024-01", category="Renda Fixa",
                      net_flow_brl=-5e9, redemption_gross_brl=6e9)
        assert s.net_flow_brl < 0


class TestFundContext:
    def _make(self, **overrides):
        defaults = dict(
            name="FundA", category="Renda Fixa", nav_brl=1e9,
            liquid_ratio=0.05, drawdown=0.0,
            institutional_fraction=0.5, redemption_sensitivity=1.0, d_plus=1,
        )
        defaults.update(overrides)
        return FundContext(**defaults)

    def test_happy_path(self):
        ctx = self._make()
        assert ctx.nav_brl == 1e9

    def test_negative_ratio_raises(self):
        with pytest.raises(ValueError):
            self._make(liquid_ratio=-0.01)

    def test_negative_drawdown_raises(self):
        with pytest.raises(ValueError):
            self._make(drawdown=-0.1)

    def test_zero_nav_raises(self):
        with pytest.raises(ValueError):
            self._make(nav_brl=0.0)

    def test_negative_d_plus_raises(self):
        with pytest.raises(ValueError):
            self._make(d_plus=-1)

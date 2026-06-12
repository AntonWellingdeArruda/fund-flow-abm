import pytest

from fund_flow.allocation import NavProportionalAllocation
from fund_flow.schema import FlowShock, FundContext


def make_context(name, category, nav_brl, **kwargs):
    defaults = dict(liquid_ratio=0.05, drawdown=0.0,
                    institutional_fraction=0.5, redemption_sensitivity=1.0, d_plus=1)
    defaults.update(kwargs)
    return FundContext(name=name, category=category, nav_brl=nav_brl, **defaults)


@pytest.fixture
def shock():
    return FlowShock(
        period="2024-05",
        category="Crédito Privado",
        net_flow_brl=-8e9,
        redemption_gross_brl=10e9,
    )


@pytest.fixture
def contexts():
    return {
        "FundA": make_context("FundA", "Crédito Privado", 4e9),
        "FundB": make_context("FundB", "Crédito Privado", 6e9),
        "FundC": make_context("FundC", "Renda Fixa", 5e9),  # different category
    }


class TestNavProportionalAllocation:
    def test_sums_to_gross_redemption(self, shock, contexts):
        alloc = NavProportionalAllocation()
        result = alloc.allocate(shock, contexts)
        total = sum(result.values())
        assert abs(total - shock.redemption_gross_brl) < 1e-9

    def test_all_values_non_negative(self, shock, contexts):
        alloc = NavProportionalAllocation()
        result = alloc.allocate(shock, contexts)
        assert all(v >= 0 for v in result.values())

    def test_only_category_funds_included(self, shock, contexts):
        alloc = NavProportionalAllocation()
        result = alloc.allocate(shock, contexts)
        assert set(result.keys()) == {"FundA", "FundB"}
        assert "FundC" not in result

    def test_proportional_to_nav(self, shock, contexts):
        alloc = NavProportionalAllocation()
        result = alloc.allocate(shock, contexts)
        # FundA NAV=4bn, FundB NAV=6bn → ratio 40:60
        assert abs(result["FundA"] / result["FundB"] - 4 / 6) < 1e-9

    def test_empty_category_returns_empty(self, shock):
        alloc = NavProportionalAllocation()
        no_cp = {"FundX": make_context("FundX", "Ações", 1e9)}
        result = alloc.allocate(shock, no_cp)
        assert result == {}


class TestFundContextsBridge:
    def test_fund_contexts_keys_match_fund_names(self, abm_book):
        ctx = abm_book.fund_contexts()
        assert set(ctx.keys()) == set(abm_book.fund_names)

    def test_all_contexts_valid(self, abm_book):
        for ctx in abm_book.fund_contexts().values():
            # FundContext.__post_init__ would have raised if invalid
            assert ctx.nav_brl > 0
            assert 0 <= ctx.liquid_ratio <= 1

    # abm_book fixture injected via conftest
    @pytest.fixture
    def abm_book(self, cfg):
        from fund_flow.synthetic.abm_data import generate_synthetic_book
        return generate_synthetic_book(cfg)

    @pytest.fixture
    def cfg(self):
        import yaml
        with open("config/default.yaml", encoding="utf-8") as f:
            return yaml.safe_load(f)

"""Synthetic credit-fund complex for ABM development and testing.

Generates a SyntheticBook with:
  - Instruments: debentures, CRI/CRA, FIDC quotas (generic illiquid credit).
  - Overlapping holdings: each instrument held by min_overlap..8 funds.
  - Separate liquid buffer (LFT/cash): sold first, non-contagious.
  - NAV identity: nav = illiquid_total / (1 - liquid_buffer_pct).
  - FundContext bridge to the allocation layer.

RNG namespace: [seed, 1] — independent from predictor_data's [seed, 0].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fund_flow.schema import VALID_CATEGORIES, FlowShock, FundContext


@dataclass
class Instrument:
    name: str
    instrument_type: str      # "debenture" | "CRI_CRA" | "FIDC_quota"
    market_depth_brl: float   # average daily tradeable volume (BRL)
    impact_exponent: float = 0.5  # sqrt price impact (immutable per CLAUDE.md §3.5)


@dataclass
class SyntheticBook:
    """Synthetic credit-fund complex shared book.

    NAV identity: nav * (1 - liquid_buffer_pct) == illiquid_holdings.sum(axis=1)
    liquid_buffer_brl is NOT in illiquid_holdings and is NOT subject to
    mark-to-market contagion — it is sold first when a fund faces redemptions.
    """
    fund_names: list[str]
    fund_categories: pd.Series         # fund_name → Anbima category
    instruments: list[Instrument]
    illiquid_holdings: pd.DataFrame    # (n_funds × n_instruments), BRL notional
    liquid_buffer_brl: pd.Series       # per fund, BRL
    nav: pd.Series                     # illiquid_holdings.sum(axis=1) + liquid_buffer_brl
    d_plus: pd.Series                  # settlement lag per fund (int, business days)
    institutional_fraction: pd.Series  # Phase 0 default = 0.5
    redemption_sensitivity: pd.Series  # Phase 0 default = 1.0

    def fund_contexts(self) -> dict[str, FundContext]:
        """Build FundContext dict for the allocation layer (Phase 0 defaults)."""
        liquid_ratio = self.liquid_buffer_brl / self.nav
        return {
            name: FundContext(
                name=name,
                category=str(self.fund_categories[name]),
                nav_brl=float(self.nav[name]),
                liquid_ratio=float(liquid_ratio[name]),
                drawdown=0.0,
                institutional_fraction=float(self.institutional_fraction[name]),
                redemption_sensitivity=float(self.redemption_sensitivity[name]),
                d_plus=int(self.d_plus[name]),
            )
            for name in self.fund_names
        }


def generate_synthetic_book(cfg: dict) -> SyntheticBook:
    """Return a SyntheticBook consistent with cfg."""
    rng = np.random.default_rng([cfg["seed"], 1])
    ac = cfg["synthetic"]["abm"]
    categories: list[str] = cfg["categories"]

    n_funds: int = ac["n_funds"]
    n_instruments: int = ac["n_instruments"]
    min_overlap: int = ac["min_overlap"]
    buffer_pct: float = ac["liquid_buffer_pct"]
    d_plus_opts: list[int] = ac["d_plus_windows"]
    mix: dict[str, float] = ac["instrument_mix"]

    fund_names = [f"Fund_{chr(65 + i)}" for i in range(n_funds)]

    # Assign funds to Anbima categories (at least one fund per category)
    credit_categories = [c for c in categories if c in VALID_CATEGORIES]
    fund_cats_list: list[str] = list(credit_categories)  # one per category first
    remaining = n_funds - len(fund_cats_list)
    fund_cats_list += rng.choice(credit_categories, size=remaining).tolist()
    rng.shuffle(fund_cats_list)
    fund_categories = pd.Series(fund_cats_list, index=fund_names, name="category")

    # Build instruments with type distribution from mix
    type_names = list(mix.keys())
    type_probs = np.array([mix[t] for t in type_names])
    type_probs /= type_probs.sum()
    inst_types = rng.choice(type_names, size=n_instruments, p=type_probs)

    instruments: list[Instrument] = []
    for j, itype in enumerate(inst_types):
        depth = float(rng.uniform(5e6, 50e6))  # BRL 5M..50M daily depth
        instruments.append(Instrument(
            name=f"{itype.upper()}_{j + 1:03d}",
            instrument_type=itype,
            market_depth_brl=depth,
        ))

    # Build holdings matrix: each instrument held by min_overlap..min(n_funds,8) funds
    holdings_arr = np.zeros((n_funds, n_instruments))
    max_overlap = min(n_funds, 8)
    for j in range(n_instruments):
        n_holders = int(rng.integers(min_overlap, max_overlap + 1))
        holders = rng.choice(n_funds, size=n_holders, replace=False)
        for h in holders:
            holdings_arr[h, j] = rng.uniform(1e6, 20e6)  # BRL 1M..20M notional

    illiquid_holdings = pd.DataFrame(
        holdings_arr,
        index=fund_names,
        columns=[inst.name for inst in instruments],
    )

    # NAV identity: nav * (1 - buffer_pct) = illiquid_total
    illiquid_total = illiquid_holdings.sum(axis=1)
    nav = illiquid_total / (1.0 - buffer_pct)
    liquid_buffer_brl = nav * buffer_pct

    # Settlement lags and behavioural defaults
    d_plus = pd.Series(
        rng.choice(d_plus_opts, size=n_funds).tolist(),
        index=fund_names, name="d_plus",
    )
    institutional_fraction = pd.Series(
        [0.5] * n_funds, index=fund_names, name="institutional_fraction"
    )
    redemption_sensitivity = pd.Series(
        [1.0] * n_funds, index=fund_names, name="redemption_sensitivity"
    )

    return SyntheticBook(
        fund_names=fund_names,
        fund_categories=fund_categories,
        instruments=instruments,
        illiquid_holdings=illiquid_holdings,
        liquid_buffer_brl=liquid_buffer_brl,
        nav=nav,
        d_plus=d_plus,
        institutional_fraction=institutional_fraction,
        redemption_sensitivity=redemption_sensitivity,
    )


def stress_flow_shock(cfg: dict) -> list[FlowShock]:
    """Canonical stress: 20% net redemption in Crédito Privado, 3 consecutive months.

    Returns category-level FlowShock objects. Use NavProportionalAllocation
    (or a behavioural variant) to distribute to individual funds.
    """
    pc = cfg["synthetic"]["predictor"]
    start = pd.Period(pc["start_date"], freq="M") + 60  # start mid-series

    shocks = []
    for i in range(3):
        period = str(start + i)
        # Representative category-level NAV for Crédito Privado (synthetic scale)
        category_nav_proxy = 50_000_000_000.0  # BRL 50 bn
        gross = category_nav_proxy * 0.20
        shocks.append(FlowShock(
            period=period,
            category="Crédito Privado",
            net_flow_brl=-gross * 0.85,   # net ≈ 85% of gross (some inflows remain)
            redemption_gross_brl=gross,
        ))
    return shocks

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

_PERIOD_RE = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])$")

VALID_CATEGORIES: frozenset[str] = frozenset([
    "Ações",
    "Multimercado",
    "Renda Fixa",
    "Crédito Privado",
    "Previdência",
    "Cambial",
])


@dataclass(frozen=True)
class FlowShock:
    """Category-level flow impulse emitted by the Predictor, consumed by the ABM.

    The ABM receives redemption_gross_brl as forced-selling pressure.
    net_flow_brl is available for reporting; it must NOT feed back into
    the Predictor's training set (circularity).
    """
    period: str                  # "YYYY-MM"
    category: str                # must be in VALID_CATEGORIES
    net_flow_brl: float          # captação líquida; negative = net redemption
    redemption_gross_brl: float  # gross outflows; always >= 0

    def __post_init__(self) -> None:
        if not _PERIOD_RE.match(self.period):
            raise ValueError(f"period must be YYYY-MM, got {self.period!r}")
        if self.category not in VALID_CATEGORIES:
            raise ValueError(f"Unknown category: {self.category!r}")
        if self.redemption_gross_brl < 0:
            raise ValueError(
                f"redemption_gross_brl must be >= 0, got {self.redemption_gross_brl}"
            )


@dataclass
class FundContext:
    """Per-fund state carrier for the allocation layer and (Phase 3) fund agents.

    Phase 0 populates behavioural fields with neutral defaults.
    Phase 3 calibrates them from agent state.
    """
    name: str
    category: str                    # Anbima category
    nav_brl: float                   # current NAV in BRL
    liquid_ratio: float              # liquid_buffer / nav; sold before illiquid assets
    drawdown: float                  # NAV drawdown from peak (0..1); Phase 0 = 0.0
    institutional_fraction: float    # fraction of units held by institutional investors
    redemption_sensitivity: float    # multiplier on base redemption rate; Phase 0 = 1.0
    d_plus: int                      # settlement lag in business days

    def __post_init__(self) -> None:
        for field in ("liquid_ratio", "drawdown",
                      "institutional_fraction", "redemption_sensitivity"):
            val = getattr(self, field)
            if val < 0:
                raise ValueError(f"{field} must be >= 0, got {val}")
        if self.nav_brl <= 0:
            raise ValueError(f"nav_brl must be > 0, got {self.nav_brl}")
        if self.d_plus < 0:
            raise ValueError(f"d_plus must be >= 0, got {self.d_plus}")

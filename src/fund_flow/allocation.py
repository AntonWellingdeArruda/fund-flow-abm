from __future__ import annotations

from typing import Protocol

from fund_flow.schema import FlowShock, FundContext


class AllocationRule(Protocol):
    """Maps a category-level FlowShock to per-fund gross redemptions.

    Implementors receive the full FundContext for every fund in the shocked
    category, enabling behavioural allocators (Phase 3) to condition on
    drawdown, share-class mix, and redemption sensitivity without any
    contract change.
    """

    def allocate(
        self,
        shock: FlowShock,
        fund_contexts: dict[str, FundContext],
    ) -> dict[str, float]:
        """Return {fund_name: gross_redemption_brl} for funds in shock.category.

        Postconditions (caller may assert):
          - Keys are exactly the funds whose category == shock.category.
          - Values are all >= 0.
          - Values sum to shock.redemption_gross_brl within 1e-9.
        """
        ...


class NavProportionalAllocation:
    """Degenerate case: allocates proportionally by NAV.

    Ignores drawdown, share-class mix, and redemption_sensitivity — a valid
    special case of the AllocationRule Protocol. Phase 3 swaps in a
    behavioural implementation without touching the Protocol signature.
    """

    def allocate(
        self,
        shock: FlowShock,
        fund_contexts: dict[str, FundContext],
    ) -> dict[str, float]:
        peers = {
            k: v for k, v in fund_contexts.items()
            if v.category == shock.category
        }
        if not peers:
            return {}
        total_nav = sum(v.nav_brl for v in peers.values())
        return {
            k: (v.nav_brl / total_nav) * shock.redemption_gross_brl
            for k, v in peers.items()
        }

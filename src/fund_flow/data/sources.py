"""Ingestion adapters for the data layer.

Two raw tables flow through the pipeline:
  - FLOWS:  long format, one row per (period, category):
            period, category, net_flow_brl, redemption_gross_brl
  - MACRO:  wide format, one row per period:
            period, selic_rate, delta_selic, ipca_monthly,
            ibovespa_return, dxy_return, ust_10y, regime

Phase 1 ships synthetic + CSV adapters. Real public-source adapters
(Anbima captação líquida, BCB SGS, B3) plug in behind the same Protocols;
stubs document the public endpoints without making network calls so the
pipeline stays testable offline (CLAUDE.md §5).
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd

from fund_flow.synthetic.predictor_data import generate_predictor_panel

FLOW_COLUMNS = ["period", "category", "net_flow_brl", "redemption_gross_brl"]
MACRO_COLUMNS = [
    "period", "selic_rate", "delta_selic", "ipca_monthly",
    "ibovespa_return", "dxy_return", "ust_10y", "regime",
]


class FlowSource(Protocol):
    def load(self) -> pd.DataFrame:
        """Return raw flows: long format with FLOW_COLUMNS."""
        ...


class MacroSource(Protocol):
    def load(self) -> pd.DataFrame:
        """Return raw macro: wide format with MACRO_COLUMNS, one row per period."""
        ...


# --- Synthetic backends (default; reuse the Phase 0 ground-truth generator) ---

class SyntheticFlowSource:
    """Slices the synthetic panel into a raw flows table."""

    def __init__(self, cfg: dict):
        self._cfg = cfg

    def load(self) -> pd.DataFrame:
        panel = generate_predictor_panel(self._cfg)
        return panel[FLOW_COLUMNS].copy()


class SyntheticMacroSource:
    """Slices the synthetic panel into a raw macro table (one row per period).

    Macro values are identical across categories within a period in the
    synthetic panel, so we de-duplicate on period.
    """

    def __init__(self, cfg: dict):
        self._cfg = cfg

    def load(self) -> pd.DataFrame:
        panel = generate_predictor_panel(self._cfg)
        macro = panel[MACRO_COLUMNS].drop_duplicates(subset="period")
        return macro.reset_index(drop=True)


# --- CSV backends (for approved-environment local data) ---

class CsvFlowSource:
    def __init__(self, path: str):
        self._path = path

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self._path)
        return df[FLOW_COLUMNS].copy()


class CsvMacroSource:
    def __init__(self, path: str):
        self._path = path

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self._path)
        return df[MACRO_COLUMNS].copy()


# --- Real public-source stubs (documented seams; no network calls in Phase 1) ---

class AnbimaFlowSource:
    """Public Anbima captação líquida by category.

    Endpoint: Anbima publishes monthly consolidated industry statistics
    (Consolidado Histórico de Fundos) as downloadable spreadsheets. Implement
    `load()` to fetch + reshape to FLOW_COLUMNS once the environment is
    approved for that data class (CLAUDE.md §5).
    """

    def load(self) -> pd.DataFrame:  # pragma: no cover - intentional stub
        raise NotImplementedError(
            "AnbimaFlowSource is a documented seam. Use SyntheticFlowSource or "
            "CsvFlowSource until a public/approved Anbima feed is wired in."
        )


class BcbMacroSource:
    """Public BCB SGS series (Selic, DI curve, IPCA) + B3/US market data.

    Endpoint: BCB SGS REST API (api.bcb.gov.br/dados/serie/...). Implement
    `load()` to fetch each series, align to monthly periods, and reshape to
    MACRO_COLUMNS once approved.
    """

    def load(self) -> pd.DataFrame:  # pragma: no cover - intentional stub
        raise NotImplementedError(
            "BcbMacroSource is a documented seam. Use SyntheticMacroSource or "
            "CsvMacroSource until a public/approved BCB feed is wired in."
        )

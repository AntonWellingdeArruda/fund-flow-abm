"""Cleaning: validate, sort, dedup, coerce, and flag/fill missing values.

Cleaning is deliberately conservative — it raises on structural problems
(unknown category, malformed period, missing required column) rather than
silently coercing, so data-quality issues surface early.
"""
from __future__ import annotations

import re

import pandas as pd

from fund_flow.data.sources import FLOW_COLUMNS, MACRO_COLUMNS
from fund_flow.schema import VALID_CATEGORIES

_PERIOD_RE = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])$")


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def _validate_periods(periods: pd.Series, name: str) -> None:
    bad = periods[~periods.astype(str).str.match(_PERIOD_RE)]
    if len(bad) > 0:
        raise ValueError(f"{name} has malformed periods: {bad.unique().tolist()[:5]}")


def clean_flows(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, sort, and dedup the raw flows table."""
    _require_columns(df, FLOW_COLUMNS, "flows")
    df = df[FLOW_COLUMNS].copy()

    _validate_periods(df["period"], "flows")

    unknown = set(df["category"].unique()) - VALID_CATEGORIES
    if unknown:
        raise ValueError(f"flows has unknown categories: {sorted(unknown)}")

    df["net_flow_brl"] = pd.to_numeric(df["net_flow_brl"], errors="coerce")
    df["redemption_gross_brl"] = pd.to_numeric(df["redemption_gross_brl"], errors="coerce")

    if df[["net_flow_brl", "redemption_gross_brl"]].isnull().any().any():
        raise ValueError("flows has non-numeric net_flow_brl / redemption_gross_brl")
    if (df["redemption_gross_brl"] < 0).any():
        raise ValueError("flows has negative redemption_gross_brl")

    df = df.drop_duplicates(subset=["period", "category"])
    df = df.sort_values(["category", "period"]).reset_index(drop=True)
    return df


def clean_macro(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, sort, dedup, and forward-fill the raw macro table.

    Accepts any macro column set (synthetic supplies the full schema; a real
    adapter may supply a subset) provided it has 'period' plus ≥1 indicator.
    'regime' is optional (synthetic-only latent label). Missing observations
    are forward-filled (honest for a slow-moving monthly series); leading NaN
    is back-filled so the first period is usable.
    """
    _require_columns(df, ["period"], "macro")
    df = df.copy()

    indicator_cols = [c for c in df.columns if c != "period"]
    if not indicator_cols:
        raise ValueError("macro must have at least one indicator column")

    _validate_periods(df["period"], "macro")

    df = df.drop_duplicates(subset="period")
    df = df.sort_values("period").reset_index(drop=True)

    if df["period"].duplicated().any():
        raise ValueError("macro has duplicate periods after dedup")

    numeric_cols = [c for c in indicator_cols if c != "regime"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    if "regime" in df.columns:
        df["regime"] = df["regime"].astype(int)

    if df[numeric_cols].isnull().any().any():
        raise ValueError("macro still has NaN after fill")
    return df

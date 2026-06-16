#!/usr/bin/env python3
"""Fetch the FULL CVM flow history (HIST yearly archives + monthly files) and
report ANBIMA-category AUM coverage per year.

One download pass: aggregates net/redemption flows to the 6 categories AND, for
each month, measures what fraction of total fund AUM (VL_PATRIM_LIQ) the registry
mapper resolves to a category. Coverage degrades in older years because CVM's
current class registry only retains currently/recently registered classes.

Usage (run from a CVM-allowed network):
    PYTHONPATH=src python3 scripts/fetch_cvm_history.py \
        --start-year 2000 --end-year 2025 --out flows_real.csv
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from fund_flow.data.categories import CvmRegistroClasseCategoryMapper
from fund_flow.data.cvm import (
    LAST_HIST_YEAR,
    fetch_informe_diario,
    fetch_informe_year,
)
from fund_flow.data.sources import FLOW_COLUMNS


def _aggregate_month(informe, cnpj_to_cat, period_str, scale):
    """(flow rows for the 6 categories, total_pl, matched_pl) for one month."""
    informe = informe.copy()
    informe["category"] = informe["cnpj"].map(cnpj_to_cat)
    total_pl = informe["pl"].sum()
    matched = informe.dropna(subset=["category"])
    matched_pl = matched["pl"].sum()
    grp = matched.groupby("category").agg(
        captacao=("captacao", "sum"), resgate=("resgate", "sum")
    )
    rows = [
        {
            "period": period_str,
            "category": cat,
            "net_flow_brl": (r["captacao"] - r["resgate"]) / scale,
            "redemption_gross_brl": r["resgate"] / scale,
        }
        for cat, r in grp.iterrows()
    ]
    return rows, total_pl, matched_pl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2000)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--scale", type=float, default=1e9)
    ap.add_argument("--out", default="flows_real.csv")
    args = ap.parse_args()

    print("Building category map from CVM RCVM-175 class registry …")
    cnpj_to_cat = CvmRegistroClasseCategoryMapper().mapping()
    print(f"  {len(cnpj_to_cat)} classes mapped to a category\n")

    rows: list[dict] = []
    cov: dict[int, list[float]] = {}  # year -> [matched_pl_sum, total_pl_sum]
    t0 = time.time()

    for year in range(args.start_year, args.end_year + 1):
        try:
            if year <= LAST_HIST_YEAR:
                months = fetch_informe_year(year)            # one yearly archive
            else:
                months = {
                    f"{year}{mth:02d}": fetch_informe_diario(f"{year}{mth:02d}")
                    for mth in range(1, 13)
                }
        except Exception as exc:                             # noqa: BLE001
            print(f"{year}: skipped ({type(exc).__name__}: {exc})")
            continue

        acc = cov.setdefault(year, [0.0, 0.0])
        for ym in sorted(months):
            period_str = f"{ym[:4]}-{ym[4:]}"
            mrows, total_pl, matched_pl = _aggregate_month(
                months[ym], cnpj_to_cat, period_str, args.scale
            )
            rows.extend(mrows)
            acc[0] += matched_pl
            acc[1] += total_pl
        pct = 100 * acc[0] / acc[1] if acc[1] else 0.0
        print(f"{year}: {len(months):2d} months | AUM coverage {pct:5.1f}%  "
              f"({time.time() - t0:5.0f}s elapsed)", flush=True)

    flows = pd.DataFrame(rows, columns=FLOW_COLUMNS)
    flows.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}: {len(flows)} rows, "
          f"{flows['period'].nunique()} months "
          f"({flows['period'].min()} … {flows['period'].max()})")

    print("\nAUM coverage by year (PL-weighted):")
    for year in sorted(cov):
        m, t = cov[year]
        print(f"  {year}: {100 * m / t if t else 0:5.1f}%")


if __name__ == "__main__":
    main()

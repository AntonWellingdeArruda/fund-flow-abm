#!/usr/bin/env python3
"""Fetch fund-level quotas and build category-level excess returns (Phase 2.6).

One download pass over the CVM informe diário (HIST yearly archives + monthly
files): retains each fund's month-end quota + PL, computes fund monthly returns,
subtracts each fund's declared-benchmark return, applies the exclusion filters,
and AUM-weights to a (period, category) excess return ER_c_t.

Writes data/category_excess.csv [period, category, cat_excess_return, n_funds,
aum_brl] and reports, per year, what fraction of total month-end AUM survives into
the excess panel (the real survivorship/coverage gauge).

Usage (run from a CVM-allowed network):
    PYTHONPATH=src python3 scripts/fetch_excess_returns.py \
        --start-year 2004 --end-year 2025 --out data/category_excess.csv
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from fund_flow.data.benchmark_mapper import canonical_benchmark
from fund_flow.data.benchmark_returns import fetch_benchmark_returns
from fund_flow.data.categories import map_anbima_classificacao
from fund_flow.data.cvm import (
    LAST_HIST_YEAR,
    fetch_informe_diario,
    fetch_informe_year,
    fetch_registro_classe,
)
from fund_flow.data.excess_returns import (
    aggregate_category_excess,
    chain_returns,
    month_end_stats,
)
from fund_flow.schema import VALID_CATEGORIES


def build_registry_maps():
    """One registry fetch → (cnpj→category, cnpj→benchmark_id)."""
    reg = fetch_registro_classe()
    cat_of, bench_of = {}, {}
    for r in reg.itertuples(index=False):
        anbima = None if pd.isna(r.anbima) else str(r.anbima)
        cat = map_anbima_classificacao(anbima)
        if cat not in VALID_CATEGORIES:
            continue
        ind = None if pd.isna(r.indicador) else str(r.indicador)
        nome = None if pd.isna(r.nome) else str(r.nome)
        cat_of[r.cnpj] = cat
        bench_of[r.cnpj] = canonical_benchmark(ind, nome, anbima)
    return cat_of, bench_of


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2004)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--out", default="data/category_excess.csv")
    ap.add_argument("--min-days", type=int, default=15)
    ap.add_argument("--min-pl", type=float, default=1e6)
    args = ap.parse_args()

    print("Building registry maps (category + benchmark per CNPJ) …")
    cat_of, bench_of = build_registry_maps()
    print(f"  {len(cat_of):,} classes mapped to a category\n")

    print("Fetching benchmark return series …")
    bench_returns, sources = fetch_benchmark_returns(
        f"{args.start_year}-01-01", f"{args.end_year}-12-31")
    for k, v in sources.items():
        print(f"  {k:16s} {v}")
    print()

    stats_by_period: dict[str, pd.DataFrame] = {}
    total_pl_by_year: dict[int, float] = {}
    t0 = time.time()

    for year in range(args.start_year, args.end_year + 1):
        try:
            if year <= LAST_HIST_YEAR:
                months = fetch_informe_year(year)
            else:
                months = {
                    f"{year}{mth:02d}": fetch_informe_diario(f"{year}{mth:02d}")
                    for mth in range(1, 13)
                }
        except Exception as exc:  # noqa: BLE001
            print(f"{year}: skipped ({type(exc).__name__}: {exc})")
            continue

        for ym in sorted(months):
            period = f"{ym[:4]}-{ym[4:]}"
            st = month_end_stats(months[ym], period)
            stats_by_period[period] = st
            total_pl_by_year[year] = total_pl_by_year.get(year, 0.0) + st["pl"].sum()
        print(f"{year}: {len(months):2d} months  ({time.time() - t0:5.0f}s)", flush=True)

    print("\nChaining fund returns + aggregating category excess …")
    fund_ret = chain_returns(stats_by_period)
    cat_excess = aggregate_category_excess(
        fund_ret, bench_of, cat_of, bench_returns,
        min_days=args.min_days, min_pl=args.min_pl)

    out = cat_excess[["period", "category", "cat_excess_return", "n_funds", "aum_brl"]]
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}: {len(out)} rows, {out['period'].nunique()} months "
          f"({out['period'].min()} … {out['period'].max()})")

    # survivorship / coverage gauge: excess-panel AUM vs total month-end AUM, by year
    print("\nExcess-panel AUM coverage by year (panel AUM ÷ total month-end AUM):")
    out["_year"] = out["period"].str.slice(0, 4).astype(int)
    panel_by_year = out.groupby("_year")["aum_brl"].sum()
    for year in sorted(total_pl_by_year):
        tot = total_pl_by_year[year]
        # total_pl summed across months; panel aum_brl also summed across months
        pan = panel_by_year.get(year, 0.0)
        print(f"  {year}: {100 * pan / tot if tot else 0:5.1f}%")

    print("\nPer-category month counts in the excess panel:")
    print(out.groupby("category")["period"].nunique().to_string())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 2.6 Step-1 STOP GATE: benchmark-mapping AUM coverage.

For each sampled month, join the RCVM-175 registry (CNPJ → category + canonical
benchmark + which tier produced it) to the informe diário's month-end PL (AUM),
and report per category: % of AUM whose benchmark is usable (≠ ZERO), the tier
breakdown (declared / name / anbima-default), and the top unmapped declared
strings by AUM.

Coverage target: < 10% AUM in ZERO per category. Do NOT proceed to Step 2 if any
category is < 90% mapped — bring this table to the user first.

Run:  PYTHONPATH=src .venv/bin/python3 scripts/audit_benchmark_coverage.py [YYYYMM ...]
Default months sample recent periods (fast, monthly endpoint).
"""
from __future__ import annotations

import sys
from collections import defaultdict

import pandas as pd

from fund_flow.data.benchmark_mapper import canonical_benchmark_traced
from fund_flow.data.categories import map_anbima_classificacao
from fund_flow.data.cvm import (
    LAST_HIST_YEAR,
    fetch_informe_diario,
    fetch_informe_year,
    fetch_registro_classe,
)
from fund_flow.schema import VALID_CATEGORIES

DEFAULT_MONTHS = ["202512", "202406", "202206"]


def _clean(v) -> str | None:
    return None if v is None or pd.isna(v) else str(v)


def build_maps():
    """Registry → per-cnpj {category, benchmark_id, tier, raw_declared}."""
    reg = fetch_registro_classe()
    cat_of, bench_of, tier_of, raw_of = {}, {}, {}, {}
    for r in reg.itertuples(index=False):
        cat = map_anbima_classificacao(_clean(r.anbima))
        if cat not in VALID_CATEGORIES:
            continue
        cid, tier = canonical_benchmark_traced(
            _clean(r.indicador), _clean(r.nome), _clean(r.anbima))
        cat_of[r.cnpj] = cat
        bench_of[r.cnpj] = cid
        tier_of[r.cnpj] = tier
        raw_of[r.cnpj] = _clean(r.indicador) or "(blank)"
    return cat_of, bench_of, tier_of, raw_of


def fetch_month(ym: str) -> pd.DataFrame:
    year = int(ym[:4])
    if year <= LAST_HIST_YEAR:
        return fetch_informe_year(year)[ym]
    return fetch_informe_diario(ym)


def month_end_pl(informe: pd.DataFrame) -> dict[str, float]:
    """Last-observed PL per cnpj within the month (AUM weight)."""
    df = informe.dropna(subset=["pl"])
    idx = df.groupby("cnpj")["date"].idxmax()
    last = df.loc[idx]
    return dict(zip(last["cnpj"], last["pl"]))


def main() -> None:
    months = sys.argv[1:] or DEFAULT_MONTHS
    print(f"Building registry maps … (benchmark + category per CNPJ)")
    cat_of, bench_of, tier_of, raw_of = build_maps()
    print(f"registry: {len(cat_of):,} CNPJs mapped to a category\n")

    unmapped_aum: dict[str, float] = defaultdict(float)  # raw declared → AUM in ZERO

    for ym in months:
        print("=" * 78)
        print(f"MONTH {ym}")
        print("=" * 78)
        try:
            informe = fetch_month(ym)
        except Exception as exc:  # noqa: BLE001
            print(f"  (skip {ym}: {exc})")
            continue
        pl = month_end_pl(informe)

        # per category: total AUM, AUM with usable benchmark, tier AUM
        tot = defaultdict(float)
        usable = defaultdict(float)
        tier_aum = defaultdict(lambda: defaultdict(float))
        for cnpj, aum in pl.items():
            cat = cat_of.get(cnpj)
            if cat is None or not (aum > 0):
                continue
            tot[cat] += aum
            cid = bench_of.get(cnpj, "ZERO")
            if cid != "ZERO":
                usable[cat] += aum
                tier_aum[cat][tier_of.get(cnpj, "?")] += aum
            else:
                unmapped_aum[raw_of.get(cnpj, "(blank)")] += aum

        rows = []
        for cat in VALID_CATEGORIES:
            t = tot[cat]
            if t <= 0:
                continue
            cov = 100 * usable[cat] / t
            decl = 100 * tier_aum[cat]["declared"] / t
            name = 100 * tier_aum[cat]["name"] / t
            anb = 100 * tier_aum[cat]["anbima"] / t
            rows.append({"category": cat, "AUM_Rbi": round(t / 1e9, 1),
                         "mapped_%": round(cov, 1), "declared_%": round(decl, 1),
                         "name_%": round(name, 1), "anbima_%": round(anb, 1)})
        df = pd.DataFrame(rows)
        gtot = df["AUM_Rbi"].sum()
        gmapped = (df["mapped_%"] * df["AUM_Rbi"]).sum() / gtot if gtot else 0
        print(df.to_string(index=False))
        print(f"  POOLED mapped: {gmapped:.1f}%  (registry-matched AUM only)\n")

    print("=" * 78)
    print("TOP 20 UNMAPPED (ZERO) DECLARED STRINGS BY AUM (summed across months)")
    print("=" * 78)
    top = sorted(unmapped_aum.items(), key=lambda kv: kv[1], reverse=True)[:20]
    for raw, aum in top:
        print(f"  R$ {aum/1e9:8.1f} bi   {raw}")


if __name__ == "__main__":
    main()

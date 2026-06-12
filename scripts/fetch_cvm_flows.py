#!/usr/bin/env python3
"""Fetch REAL fund flows from CVM Informe Diário, categorized to Anbima classes.

Usage (run from a network CVM allows — it 403s some datacenter IPs):
    PYTHONPATH=src python3 scripts/fetch_cvm_flows.py --start 2024-01 --end 2024-06
        [--mapper registro|anbima|cvm]   # default: registro
        [--scale 1e9]                    # R$ billions
        [--out flows_real.csv]

--mapper registro : CVM RCVM-175 class registry (free, no auth). Full ANBIMA
                    taxonomy incl. Crédito Privado + Previdência; joins the
                    informe at share-class level → ~94% AUM coverage. BEST.
--mapper anbima   : official ANBIMA Fundos v2 API (needs production access;
                    falls back to CVM cadastro on AnbimaError).
--mapper cvm      : CVM cad_fi.csv CLASSE only (coarse; no Crédito Privado /
                    Previdência split, and only ~2.5% AUM under RCVM 175).
"""
from __future__ import annotations

import argparse

from fund_flow.data.categories import (
    AnbimaCategoryMapper,
    CvmCadastroCategoryMapper,
    CvmRegistroClasseCategoryMapper,
)
from fund_flow.data.cvm import fetch_cadastro
from fund_flow.data.sources import CvmFlowSource


def _cvm_mapper():
    print("Building category map from CVM cad_fi.csv …")
    return CvmCadastroCategoryMapper(fetch_cadastro())


def _registro_mapper():
    print("Building category map from CVM RCVM-175 class registry …")
    return CvmRegistroClasseCategoryMapper()


def build_mapper(kind: str):
    if kind == "registro":
        return _registro_mapper()
    if kind == "cvm":
        return _cvm_mapper()
    # anbima (with graceful fallback)
    from fund_flow.config import get_secret
    from fund_flow.data.anbima import AnbimaClient, AnbimaError
    try:
        client = AnbimaClient(
            get_secret("ANBIMA_CLIENT_ID"), get_secret("ANBIMA_CLIENT_SECRET")
        )
        mapper = AnbimaCategoryMapper(client)
        m = mapper.mapping()              # triggers the API call(s) now
        print(f"ANBIMA category map: {len(m)} funds")

        class _Pre:                       # wrap the already-built dict
            def mapping(self_inner):
                return m
        return _Pre()
    except AnbimaError as exc:
        print(f"ANBIMA mapping unavailable ({exc}); falling back to CVM cadastro.")
        return _cvm_mapper()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)   # YYYY-MM
    ap.add_argument("--end", required=True)
    ap.add_argument(
        "--mapper", choices=["registro", "anbima", "cvm"], default="registro"
    )
    ap.add_argument("--scale", type=float, default=1e9)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mapper = build_mapper(args.mapper)
    print(f"Fetching CVM flows {args.start} … {args.end} (scale={args.scale:g}) …")
    flows = CvmFlowSource(args.start, args.end, mapper, scale=args.scale).load()

    print(f"\n{len(flows)} (period × category) rows\n")
    print(flows.to_string(index=False))
    pivot = flows.pivot(index="period", columns="category", values="net_flow_brl")
    print("\nnet_flow_brl by category:\n", pivot.to_string())

    if args.out:
        flows.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fetch REAL Brazilian macro from the public BCB SGS API (no key required).

Usage:
    PYTHONPATH=src python scripts/fetch_bcb_macro.py [--start 2010-01]
                                                     [--end 2024-12]
                                                     [--out macro_bcb.csv]

Pulls Selic (432), IPCA (433), USD/BRL (1), aligns to monthly periods, and
prints the tail. With --out, also writes a CSV that CsvMacroSource can load.
Note: output is gitignored (*.csv) — real data stays out of the repo (§5).
"""
from __future__ import annotations

import argparse

from fund_flow.data.cleaning import clean_macro
from fund_flow.data.sources import BcbMacroSource


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"Fetching real BCB macro {args.start} … {args.end or 'today'}")
    raw = BcbMacroSource(start=args.start, end=args.end).load()
    macro = clean_macro(raw)

    print(f"\n{len(macro)} monthly rows, columns: {list(macro.columns)}\n")
    print(macro.tail(12).to_string(index=False))

    if args.out:
        macro.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

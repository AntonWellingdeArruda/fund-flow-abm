#!/usr/bin/env python3
"""Fetch the combined REAL macro panel: BCB (Brazil) + FRED (US).

Usage:
    PYTHONPATH=src python scripts/fetch_real_macro.py [--start 2010-01]
                                                      [--end 2024-12]
                                                      [--out macro_real.csv]
                                                      [--bcb-only]

BCB (no key): Selic, IPCA, USD/BRL.   FRED (needs FRED_API_KEY in .env):
UST 10Y, fed funds, S&P 500 return, broad-USD return. Output CSV is gitignored.
"""
from __future__ import annotations

import argparse

from fund_flow.config import has_secret
from fund_flow.data.cleaning import clean_macro
from fund_flow.data.sources import (
    BcbMacroSource,
    CompositeMacroSource,
    FredMacroSource,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bcb-only", action="store_true")
    args = ap.parse_args()

    sources = [BcbMacroSource(start=args.start, end=args.end)]
    if not args.bcb_only:
        if not has_secret("FRED_API_KEY"):
            print("FRED_API_KEY not set — falling back to BCB only.")
        else:
            sources.append(FredMacroSource(start=args.start, end=args.end))

    print(f"Fetching real macro {args.start} … {args.end or 'today'} "
          f"from {len(sources)} source(s)")
    macro = clean_macro(CompositeMacroSource(sources).load())

    print(f"\n{len(macro)} monthly rows, columns: {list(macro.columns)}\n")
    print(macro.tail(12).to_string(index=False))

    if args.out:
        macro.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Walk-forward backtest of the Predictor vs naive baselines (CLAUDE.md §7).

Usage:
    # synthetic (default — data we control):
    PYTHONPATH=src python3 scripts/run_backtest.py [--min-train 36] [--no-regime]

    # REAL data: CVM flows CSV + live BCB/FRED/Ibovespa macro:
    PYTHONPATH=src python3 scripts/run_backtest.py \
        --flows flows_real.csv --macro real

Reports out-of-sample RMSE/MAE for: random walk, AR(1), joint VARX, and
(optionally) the Markov regime model. The acceptance gate is that the joint
VARX beats BOTH naive baselines.
"""
from __future__ import annotations

import argparse
import time
import warnings

import pandas as pd
import yaml

from fund_flow.data.dataset import build_dataset
from fund_flow.data.sources import CsvFlowSource
from fund_flow.predictor.backtest import (
    compare,
    per_category_rmse,
    per_category_table,
    walk_forward,
)
from fund_flow.predictor.baselines import AR1Predictor, RandomWalkPredictor
from fund_flow.predictor.reshape import wide_exog, wide_flows
from fund_flow.predictor.var_model import VARXCVPredictor, VARXPredictor
from fund_flow.realdata import build_real_frame, build_real_macro_source, flow_period_range


def _build_real_sources(flows_path: str, with_embi: bool = False):
    """CsvFlowSource(flows_path) + the full real macro panel bounded to its
    period range. BCB + FRED + Yahoo cover the full ~2004+ window; with_embi
    adds the IPEAdata EMBI+ credit spread, which truncates the inner-joined
    panel at 2024-07 (the series is discontinued there).
    """
    start, end = flow_period_range(flows_path)
    macro = build_real_macro_source(start, end, include_ipea=with_embi)
    return CsvFlowSource(flows_path), macro


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--min-train", type=int, default=36)
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--no-regime", action="store_true",
                    help="skip the slow Markov regime model")
    ap.add_argument("--flows", default=None,
                    help="CVM flows CSV (real data); omit for synthetic")
    ap.add_argument("--macro", choices=["synthetic", "real", "cached"],
                    default="synthetic",
                    help="'real' = live BCB+FRED+Yahoo (full 2004+); "
                         "'cached' = the on-disk macro_real.csv panel; needs --flows")
    ap.add_argument("--with-embi", action="store_true",
                    help="add the IPEAdata EMBI+ credit spread (real macro only); "
                         "truncates the panel at 2024-07 where the series ends")
    ap.add_argument("--cv", action="store_true",
                    help="VARX picks ridge alpha by nested CV inside each train "
                         "window (the §7-honest method; alpha never sees the test point)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.macro == "cached":
        if not args.flows:
            ap.error("--macro cached requires --flows")
        print(f"CACHED real macro panel + flows={args.flows}\n")
        frame = build_real_frame(args.flows, cfg)
    elif args.flows or args.macro == "real":
        if not args.flows:
            ap.error("--macro real requires --flows (real macro aligns to flow periods)")
        embi_note = " + IPEA EMBI (panel ends 2024-07)" if args.with_embi else ""
        print(f"REAL data: flows={args.flows}, macro=BCB+FRED+Yahoo{embi_note} (live)\n")
        flow_source, macro_source = _build_real_sources(args.flows, args.with_embi)
        frame = build_dataset(cfg, flow_source=flow_source, macro_source=macro_source)
    else:
        frame = build_dataset(cfg)
    flows = wide_flows(frame)
    exog = wide_exog(frame)
    mt = args.min_train

    # Credit-spread columns (present only with real macro) — the lever under
    # test. We fit VARX twice off the SAME panel (with and without them) so the
    # per-category ablation is apples-to-apples.
    credit_cols = [c for c in exog.columns if "embi" in c or "icc" in c]
    exog_nocredit = exog.drop(columns=credit_cols) if credit_cols else None

    print(f"flows matrix {flows.shape}, exog {exog.shape}, min_train={mt}")
    if credit_cols:
        print(f"credit-spread exog: {credit_cols}")
    print()

    t0 = time.time()

    def varx():
        if args.cv:
            return VARXCVPredictor(order=1)
        return VARXPredictor(order=1, ridge_alpha=args.ridge_alpha)

    rw_res = walk_forward(lambda: RandomWalkPredictor(), flows, None, mt, "random_walk")
    ar1_res = walk_forward(lambda: AR1Predictor(), flows, None, mt, "ar1")
    varx_res = walk_forward(varx, flows, exog, mt, "varx_full")
    results = [rw_res, ar1_res, varx_res]

    varx_nc_res = None
    if credit_cols:
        varx_nc_res = walk_forward(varx, flows, exog_nocredit, mt, "varx_nocredit")
        results.append(varx_nc_res)

    if not args.no_regime:
        from fund_flow.predictor.regime import MarkovRegimePredictor
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results.append(
                walk_forward(lambda: MarkovRegimePredictor(), flows, None, mt,
                             "markov_regime")
            )

    print("POOLED OOS (all categories):")
    print(compare(results).to_string(index=False))

    # Per-category OOS — where a credit signal should actually show up.
    cat_models = [ar1_res, varx_res] + ([varx_nc_res] if varx_nc_res else [])
    print("\nPER-CATEGORY OOS RMSE:")
    print(per_category_table(cat_models).round(3).to_string())

    # Headline gate (VARX with full panel vs both baselines, pooled).
    rw, ar1, vx = rw_res.rmse, ar1_res.rmse, varx_res.rmse
    print(f"\nVARX_full vs random_walk: {(1 - vx / rw) * 100:+.1f}%  (beats: {vx < rw})")
    print(f"VARX_full vs ar1:         {(1 - vx / ar1) * 100:+.1f}%  (beats: {vx < ar1})")
    print("ACCEPTANCE GATE (VARX beats both baselines): "
          f"{'PASS' if (vx < rw and vx < ar1) else 'FAIL'}")

    # Credit ablation: did the spreads help, and where?
    if varx_nc_res is not None:
        full = per_category_rmse(varx_res)
        nc = per_category_rmse(varx_nc_res)
        ab = pd.DataFrame({
            "varx_nocredit": nc,
            "varx_full": full,
            "credit_gain_%": (1 - full / nc) * 100,   # +% = spreads helped
        }).round(3)
        print("\nCREDIT ABLATION (VARX_full vs VARX_nocredit, per category):")
        print(ab.to_string())
        print(f"pooled credit gain: {(1 - vx / varx_nc_res.rmse) * 100:+.2f}%")

    print(f"\nbacktest time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

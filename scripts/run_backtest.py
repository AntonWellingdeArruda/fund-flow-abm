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
from fund_flow.data.sources import (
    BcbMacroSource,
    CompositeMacroSource,
    CsvFlowSource,
    FredMacroSource,
    IbovespaSource,
)
from fund_flow.predictor.backtest import compare, walk_forward
from fund_flow.predictor.baselines import AR1Predictor, RandomWalkPredictor
from fund_flow.predictor.reshape import wide_exog, wide_flows
from fund_flow.predictor.var_model import VARXPredictor


def _build_real_sources(flows_path: str):
    """CsvFlowSource(flows_path) + live BCB+FRED+Ibovespa macro bounded to its
    period range. Returns (flow_source, macro_source)."""
    flow_source = CsvFlowSource(flows_path)
    periods = pd.read_csv(flows_path)["period"]
    start, end = str(periods.min()), str(periods.max())
    macro_source = CompositeMacroSource([
        BcbMacroSource(start=start, end=end),
        FredMacroSource(start=start, end=end),
        IbovespaSource(start=start, end=end),
    ])
    return flow_source, macro_source


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--min-train", type=int, default=36)
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--no-regime", action="store_true",
                    help="skip the slow Markov regime model")
    ap.add_argument("--flows", default=None,
                    help="CVM flows CSV (real data); omit for synthetic")
    ap.add_argument("--macro", choices=["synthetic", "real"], default="synthetic",
                    help="'real' = live BCB+FRED+Ibovespa; needs --flows")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.flows or args.macro == "real":
        if not args.flows:
            ap.error("--macro real requires --flows (real macro aligns to flow periods)")
        print(f"REAL data: flows={args.flows}, macro=BCB+FRED+Ibovespa (live)\n")
        flow_source, macro_source = _build_real_sources(args.flows)
        frame = build_dataset(cfg, flow_source=flow_source, macro_source=macro_source)
    else:
        frame = build_dataset(cfg)
    flows = wide_flows(frame)
    exog = wide_exog(frame)
    mt = args.min_train

    print(f"flows matrix {flows.shape}, exog {exog.shape}, min_train={mt}\n")

    t0 = time.time()
    results = [
        walk_forward(lambda: RandomWalkPredictor(), flows, None, mt, "random_walk"),
        walk_forward(lambda: AR1Predictor(), flows, None, mt, "ar1 (baseline)"),
        walk_forward(
            lambda: VARXPredictor(order=1, ridge_alpha=args.ridge_alpha),
            flows, exog, mt, "varx (joint)",
        ),
    ]
    if not args.no_regime:
        from fund_flow.predictor.regime import MarkovRegimePredictor
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results.append(
                walk_forward(lambda: MarkovRegimePredictor(), flows, None, mt,
                             "markov_regime")
            )

    tbl = compare(results)
    print(tbl.to_string(index=False))

    rmse = dict(zip(tbl["model"], tbl["rmse"]))
    rw, ar1 = rmse["random_walk"], rmse["ar1 (baseline)"]
    vx = rmse["varx (joint)"]
    print(f"\nVARX vs random_walk: {(1 - vx / rw) * 100:+.1f}% RMSE  "
          f"(beats: {vx < rw})")
    print(f"VARX vs ar1:         {(1 - vx / ar1) * 100:+.1f}% RMSE  "
          f"(beats: {vx < ar1})")
    gate = vx < rw and vx < ar1
    print(f"\nACCEPTANCE GATE (VARX beats both baselines): "
          f"{'PASS' if gate else 'FAIL'}")
    print(f"backtest time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

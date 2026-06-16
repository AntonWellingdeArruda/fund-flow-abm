#!/usr/bin/env python3
"""Over/under-fitting diagnostics for the VARX predictor on the real series.

Run:
    PYTHONPATH=src .venv/bin/python3 scripts/diagnose_model.py [--flows flows_real.csv]

Reports four honest diagnostics (CLAUDE.md §7):
  1. In-sample vs out-of-sample RMSE gap per α — a wide gap that shrinks as α
     grows is the signature of overfitting.
  2. The OOS α-path — purely diagnostic. Picking α off this curve would snoop
     the test set; the deployed model picks α by NESTED CV instead.
  3. Nested-CV VARX vs AR(1) — the honest, deployable comparison.
  4. Learned standardized weights per category — e.g. does Crédito Privado load
     more on rates (fed_funds / ust_10y / selic) than on sp500_return?
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml

from fund_flow.predictor.backtest import walk_forward
from fund_flow.predictor.baselines import AR1Predictor
from fund_flow.predictor.reshape import wide_exog, wide_flows
from fund_flow.predictor.var_model import VARXCVPredictor, VARXPredictor
from fund_flow.realdata import build_real_frame


def insample_rmse(flows: pd.DataFrame, exog: pd.DataFrame, alpha: float) -> float:
    m = VARXPredictor(order=1, ridge_alpha=alpha).fit(flows, exog)
    X, Y = m._build_design(flows.to_numpy(float), exog.to_numpy(float))
    pred = m._predict_from_design(X)
    return float(np.sqrt(np.mean((Y - pred) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", default="flows_real.csv")
    ap.add_argument("--min-train", type=int, default=36)
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config/default.yaml"))
    frame = build_real_frame(args.flows, cfg)
    flows, exog = wide_flows(frame), wide_exog(frame)
    mt = args.min_train
    print(f"{len(flows)} months, {exog.shape[1]} predictors, min_train={mt}\n")

    # --- 1 & 2: in-sample vs OOS RMSE across the ridge path ----------------- #
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
    print("ridge α   in-sample   out-of-sample   gap")
    rows = []
    for a in alphas:
        ins = insample_rmse(flows, exog, a)
        oos = walk_forward(lambda a=a: VARXPredictor(1, a), flows, exog, mt).rmse
        rows.append((a, ins, oos))
        print(f"{a:7.1f}   {ins:9.3f}   {oos:13.3f}   {oos - ins:+.3f}")
    best_oos_a = min(rows, key=lambda r: r[2])[0]
    print(f"\n(diagnostic only) α with lowest OOS = {best_oos_a} — selecting on "
          "this would snoop the test set; use nested CV instead.")

    # --- 3: honest deployable comparison ----------------------------------- #
    ar1 = walk_forward(lambda: AR1Predictor(), flows, None, mt, "ar1").rmse
    cv = walk_forward(lambda: VARXCVPredictor(order=1), flows, exog, mt, "varx_cv")
    print(f"\nAR(1)              OOS RMSE = {ar1:.3f}")
    print(f"VARX (nested-CV α) OOS RMSE = {cv.rmse:.3f}  "
          f"({(1 - cv.rmse / ar1) * 100:+.1f}% vs AR1)")

    # --- 4: learned per-category weights (macro predictors only) ----------- #
    full = VARXPredictor(order=1, ridge_alpha=best_oos_a).fit(flows, exog)
    coefs = full.coefficients()
    macro = coefs.loc[[c for c in coefs.index if not c.endswith("_flow_lag1")]]
    print("\nStandardized macro weights per category "
          f"(VARX α={best_oos_a}, |w| comparable across rows/cols):")
    with pd.option_context("display.float_format", lambda x: f"{x:+.2f}"):
        print(macro.to_string())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Two honest experiments on the REAL panel, to settle open predictor questions.

E1. Does decomposing flows into gross components help?
    The VARX currently models only NET flow (its own lags) + macro; lagged GROSS
    redemptions are dropped (reshape._NON_EXOG_LAGS). Redemption waves are known
    to cluster, so lagged gross redemptions might lead next month's net flow.
    We test a JOINT VAR on the stacked (net, redemption) vector — each net-flow
    equation then sees every category's lagged net flow AND lagged redemptions —
    and score OOS on the NET columns only, vs net-only VARX and AR(1).

E2. Does the training window matter — expanding vs rolling, and which sub-period?
    The backtest trains expanding-from-2004. If the flow process shifts across
    Selic regimes, a rolling window (recent months only) could forecast better.

Run:  PYTHONPATH=src .venv/bin/python3 scripts/experiment_predictors.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from fund_flow.predictor.backtest import per_category_rmse, walk_forward
from fund_flow.predictor.baselines import AR1Predictor
from fund_flow.predictor.reshape import wide_exog, wide_flows
from fund_flow.predictor.var_model import VARXCVPredictor, VARXPredictor
from fund_flow.realdata import build_real_frame

CFG = yaml.safe_load(open("config/default.yaml", encoding="utf-8"))
FLOWS = "flows_real.csv"
MT = 36


def _net_rmse(result, net_cols) -> pd.Series:
    """Per-category OOS RMSE restricted to the NET-flow columns."""
    e = result.errors[net_cols]
    return np.sqrt((e ** 2).mean(axis=0))


def main() -> None:
    frame = build_real_frame(FLOWS, CFG)
    net = wide_flows(frame, "net_flow_brl")                 # period × 6
    red = wide_flows(frame, "redemption_gross_brl")         # period × 6
    exog = wide_exog(frame)                                  # period × macro

    cats = list(net.columns)
    net_cols = cats
    # Stacked (net, redemption) wide matrix for the joint VAR.
    red2 = red.add_suffix("__red")
    joint = pd.concat([net, red2], axis=1)

    print(f"panel: {net.shape[0]} months {net.index[0]}..{net.index[-1]}, "
          f"{len(cats)} categories, exog {exog.shape[1]} macro cols, min_train={MT}\n")

    # ---------------- E1: gross-flow decomposition ------------------------- #
    print("=" * 72)
    print("E1  GROSS-FLOW DECOMPOSITION (per-category OOS RMSE on NET flow)")
    print("=" * 72)
    ar1 = walk_forward(lambda: AR1Predictor(), net, None, MT, "ar1")
    vx_net = walk_forward(lambda: VARXPredictor(1, 1.0), net, exog, MT, "varx_net")
    vx_joint = walk_forward(lambda: VARXPredictor(1, 1.0), joint, exog, MT, "varx_joint")
    # auto-alpha versions (the honest, no-snoop comparison)
    cv_net = walk_forward(lambda: VARXCVPredictor(1), net, exog, MT, "varxcv_net")
    cv_joint = walk_forward(lambda: VARXCVPredictor(1), joint, exog, MT, "varxcv_joint")

    tbl = pd.DataFrame({
        "ar1": per_category_rmse(ar1),
        "varx_net": per_category_rmse(vx_net),
        "varx_joint": _net_rmse(vx_joint, net_cols),
        "varxcv_net": per_category_rmse(cv_net),
        "varxcv_joint": _net_rmse(cv_joint, net_cols),
    })
    tbl.loc["POOLED"] = np.sqrt((tbl ** 2).mean())  # rough pooled summary
    print(tbl.round(3).to_string())
    g = (1 - _net_rmse(cv_joint, net_cols) / per_category_rmse(cv_net)) * 100
    print("\ngross-decomp gain over net-only (varxcv), per category (%):")
    print(g.round(2).to_string())

    # E1b: parsimonious test — does each category's OWN lagged redemption add
    # anything beyond its own lagged net flow? (Avoids the 12-dim VAR's bloat.)
    # Per-category expanding OOS, OLS: net[t] ~ net[t-1]  vs  net[t-1] + red[t-1].
    print("\nE1b PARSIMONIOUS (own net-lag vs +own redemption-lag), OOS RMSE:")
    rows = []
    for c in cats:
        y = net[c].to_numpy()
        nl = net[c].shift(1).to_numpy()
        rl = red[c].shift(1).to_numpy()
        e1, e2 = [], []
        for t in range(MT, len(y)):
            tr = slice(1, t)  # rows 1..t-1 have valid lags
            X1 = np.c_[np.ones(t - 1), nl[tr]]
            X2 = np.c_[np.ones(t - 1), nl[tr], rl[tr]]
            b1 = np.linalg.lstsq(X1, y[tr], rcond=None)[0]
            b2 = np.linalg.lstsq(X2, y[tr], rcond=None)[0]
            e1.append(y[t] - np.array([1, nl[t]]) @ b1)
            e2.append(y[t] - np.array([1, nl[t], rl[t]]) @ b2)
        r1 = float(np.sqrt(np.mean(np.array(e1) ** 2)))
        r2 = float(np.sqrt(np.mean(np.array(e2) ** 2)))
        rows.append({"category": c, "net_lag": round(r1, 3),
                     "+red_lag": round(r2, 3),
                     "gain_%": round((1 - r2 / r1) * 100, 2)})
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------------- E2: expanding vs rolling window ---------------------- #
    print("\n" + "=" * 72)
    print("E2  TRAINING WINDOW: expanding vs rolling (pooled OOS RMSE)")
    print("=" * 72)
    rows = []
    for label, win in [("expanding", None), ("rolling-120", 120),
                       ("rolling-84", 84), ("rolling-60", 60), ("rolling-48", 48)]:
        a = walk_forward(lambda: AR1Predictor(), net, None, MT, "ar1", window=win)
        v = walk_forward(lambda: VARXCVPredictor(1), net, exog, MT, "varxcv", window=win)
        rows.append({"window": label, "ar1_rmse": a.rmse, "varxcv_rmse": v.rmse})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # ---------------- E2b: sub-period (does WHERE we test matter?) --------- #
    print("\n" + "=" * 72)
    print("E2b SUB-PERIOD OOS (expanding train, AR1 vs varxcv, by forecast era)")
    print("=" * 72)
    T = len(net)
    thirds = [("early", MT, MT + (T - MT) // 3),
              ("mid", MT + (T - MT) // 3, MT + 2 * (T - MT) // 3),
              ("late", MT + 2 * (T - MT) // 3, T)]
    rows = []
    for label, s, e in thirds:
        a = walk_forward(lambda: AR1Predictor(), net, None, MT, "ar1",
                         test_start=s, test_end=e)
        v = walk_forward(lambda: VARXCVPredictor(1), net, exog, MT, "varxcv",
                         test_start=s, test_end=e)
        rows.append({"era": label, "periods": f"{net.index[s]}..{net.index[e-1]}",
                     "ar1_rmse": a.rmse, "varxcv_rmse": v.rmse,
                     "varx_gain_%": (1 - v.rmse / a.rmse) * 100})
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

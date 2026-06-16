#!/usr/bin/env python3
"""Phase 2.6 Step-5 STOP GATE: does the per-category excess-return signal help?

Walk-forward OOS backtest comparing, per category:
    AR(1)            — naive baseline
    VARX-CV-base     — joint ridge VAR, macro exog only (no excess)
    VARX-CV-excess   — same + each category's OWN excess return (window chosen
                       per category by inner CV)

PRIMARY   — Ações and/or Multimercado improve OOS vs VARX-CV-base
GUARDRAIL — no category degrades > 1% vs base; pooled does not deteriorate

Run:  PYTHONPATH=src .venv/bin/python3 scripts/backtest_excess.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from fund_flow.predictor.backtest import per_category_rmse, walk_forward
from fund_flow.predictor.baselines import AR1Predictor
from fund_flow.predictor.reshape import wide_category_signal, wide_exog, wide_flows
from fund_flow.predictor.var_model import VARXCVPredictor, VARXExcessCVPredictor
from fund_flow.realdata import EXCESS_WINDOWS, build_real_frame

CFG = yaml.safe_load(open("config/default.yaml", encoding="utf-8"))
FLOWS = "flows_real.csv"
MT = 36
TARGETS = ("Ações", "Multimercado")


def main() -> None:
    frame = build_real_frame(FLOWS, CFG)
    flows = wide_flows(frame)
    exog = wide_exog(frame)
    # per-category lagged excess signals, keyed by window length (1/3/6 months)
    signals = {}
    for col, win in EXCESS_WINDOWS.items():
        sig = wide_category_signal(frame, f"{col}_lag1")
        if sig is not None:
            signals[win] = sig
    if not signals:
        raise SystemExit("No excess signals in frame — run scripts/fetch_excess_returns.py first.")

    cov = {w: float(np.isfinite(s.to_numpy()).mean()) for w, s in signals.items()}
    print(f"panel: {flows.shape[0]} months {flows.index[0]}..{flows.index[-1]}, "
          f"signals present (non-NaN frac): {cov}\n")

    ar1 = walk_forward(lambda: AR1Predictor(), flows, None, MT, "ar1")
    base = walk_forward(lambda: VARXCVPredictor(1), flows, exog, MT, "varxcv_base")
    excess = walk_forward(
        lambda: VARXExcessCVPredictor(1, signals=signals), flows, exog, MT, "varxcv_excess")

    tbl = pd.DataFrame({
        "AR(1)": per_category_rmse(ar1),
        "VARX-CV-base": per_category_rmse(base),
        "VARX-CV-excess": per_category_rmse(excess),
    })
    tbl["Δ vs base %"] = (1 - tbl["VARX-CV-excess"] / tbl["VARX-CV-base"]) * 100
    pooled = pd.Series({
        "AR(1)": ar1.rmse, "VARX-CV-base": base.rmse, "VARX-CV-excess": excess.rmse,
        "Δ vs base %": (1 - excess.rmse / base.rmse) * 100,
    }, name="POOLED")
    tbl = pd.concat([tbl, pooled.to_frame().T])

    print("=" * 74)
    print("PER-CATEGORY OOS RMSE  (excess vs base; + = excess better)")
    print("=" * 74)
    print(tbl.round(3).to_string())

    # selected windows (illustrative, fit on full panel — in-sample selection)
    fm = VARXExcessCVPredictor(1, signals=signals).fit(flows, exog)
    print("\nselected window per category (full-panel fit, illustrative):")
    print({c: fm.windows_[c] for c in flows.columns}, "alpha:", fm.chosen_alpha_)

    # acceptance
    print("\n" + "=" * 74)
    primary = {c: tbl.loc[c, "Δ vs base %"] for c in TARGETS if c in tbl.index}
    primary_ok = any(v > 0 for v in primary.values())
    worst = tbl.loc[[c for c in flows.columns], "Δ vs base %"].min()
    guard_ok = worst > -1.0 and pooled["Δ vs base %"] >= 0
    print(f"PRIMARY  (Ações/Multimercado improve): {primary} → {'PASS' if primary_ok else 'FAIL'}")
    print(f"GUARDRAIL (no cat < -1%, pooled ≥ 0):  worst={worst:+.2f}%, "
          f"pooled={pooled['Δ vs base %']:+.2f}% → {'PASS' if guard_ok else 'FAIL'}")
    print("VERDICT:", "ADD excess to PerCategorySelect" if (primary_ok and guard_ok)
          else "document 'tried, did not help' and proceed to Phase 3")


if __name__ == "__main__":
    main()

"""Local research dashboard for the Fund-Flow Predictor → ABM project.

Run (from the project root):
    PYTHONPATH=src .venv/bin/python3 -m streamlit run app/streamlit_app.py

Predictor tab: pick which macro predictors feed the model, choose the model and
its regularization, run the walk-forward out-of-sample backtest, and read the
leaderboard / actual-vs-predicted plots. ABM tab: preview the synthetic fund
complex and stress scenario the Phase-3 engine will simulate.

Honesty notes (CLAUDE.md §7):
  * The backtest is strictly walk-forward, out-of-sample. No in-sample scores.
  * A statistical model LEARNS its predictor weights from data — you don't set
    them by hand. The honest knobs exposed here are (a) which predictors to
    include and (b) the ridge shrinkage (how much weight predictors may take).
  * Real flows pre-2004 are excluded (CVM registry coverage < 50%).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# Make `src/` importable whether or not the package is installed.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fund_flow.predictor.backtest import compare, walk_forward
from fund_flow.predictor.baselines import AR1Predictor, RandomWalkPredictor
from fund_flow.predictor.reshape import (
    exog_columns,
    wide_category_signal,
    wide_exog,
    wide_flows,
)
from fund_flow.predictor.var_model import (
    PerCategorySelectPredictor,
    VARXCVPredictor,
    VARXExcessCVPredictor,
    VARXPredictor,
)
from fund_flow.realdata import EXCESS_WINDOWS

st.set_page_config(page_title="Fund-Flow Predictor", layout="wide")

CONFIG = ROOT / "config" / "default.yaml"
FLOWS_CSV = ROOT / "flows_real.csv"


@st.cache_data(show_spinner=False)
def _cfg() -> dict:
    with open(CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data(show_spinner="Building dataset…")
def get_frame(source: str, refresh: bool = False) -> pd.DataFrame:
    """Modeling frame for the chosen source. Cached across reruns."""
    cfg = _cfg()
    if source == "Real (CVM + macro)":
        from fund_flow.realdata import build_real_frame
        return build_real_frame(str(FLOWS_CSV), cfg, refresh=refresh)
    from fund_flow.data.dataset import build_dataset
    return build_dataset(cfg)


def predictor_label(col: str) -> str:
    """'selic_rate_lag1' -> 'selic_rate' for display."""
    return col[:-5] if col.endswith("_lag1") else col


def _diverging_bg(val: float, limit: float) -> str:
    """RdBu-style cell background WITHOUT matplotlib (Styler.background_gradient
    needs it; we avoid the dependency). Negative -> red, positive -> blue, with
    intensity scaled to ``limit``. Returns a CSS string for Styler.map."""
    if pd.isna(val):
        return ""
    t = max(-1.0, min(1.0, float(val) / (limit or 1.0)))
    a = abs(t)
    fade = int(round(255 * (1 - a)))
    if t < 0:                       # red side
        r, g, b = 255, fade, fade
    else:                           # blue side
        r, g, b = fade, fade, 255
    return f"background-color: rgb({r},{g},{b}); color: black"


# --------------------------------------------------------------------------- #
# Predictor tab
# --------------------------------------------------------------------------- #
def predictor_tab() -> None:
    st.header("Captação-líquida predictor — walk-forward backtest")

    with st.sidebar:
        st.subheader("Data")
        source = st.radio(
            "Source", ["Real (CVM + macro)", "Synthetic"], index=0,
            help="Real = CVM fund flows + BCB/FRED/Yahoo macro (cached locally).",
        )
        refresh = False
        if source.startswith("Real"):
            refresh = st.button("↻ Refresh macro cache (slow, re-fetches live)")

    try:
        frame = get_frame(source, refresh=refresh)
    except FileNotFoundError:
        st.error(f"{FLOWS_CSV.name} not found. Fetch it first:\n\n"
                 "`PYTHONPATH=src .venv/bin/python3 scripts/fetch_cvm_history.py "
                 "--start-year 2004 --end-year 2025`")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not build dataset: {exc}")
        return

    all_exog = exog_columns(frame)
    periods = sorted(frame["period"].unique())
    st.caption(f"**{len(periods)} months** · {periods[0]} → {periods[-1]} · "
               f"{len(all_exog)} candidate predictors · 6 categories")

    # ---- sidebar: predictors, models, hyperparameters --------------------- #
    with st.sidebar:
        st.subheader("Predictors (lagged 1 month)")
        try:
            from fund_flow.realdata import MACRO_GROUPS
            groups = MACRO_GROUPS
        except Exception:  # noqa: BLE001
            groups = {}
        selected: list[str] = []
        grouped_bases = {b for v in groups.values() for b in v}
        # grouped toggles
        for group, bases in groups.items():
            present = [f"{b}_lag1" for b in bases if f"{b}_lag1" in all_exog]
            if not present:
                continue
            st.markdown(f"*{group}*")
            for col in present:
                if st.checkbox(predictor_label(col), value=True, key=f"p_{col}"):
                    selected.append(col)
        # any predictors not covered by the known groups (e.g. synthetic extras)
        other = [c for c in all_exog if predictor_label(c) not in grouped_bases]
        if other:
            st.markdown("*Other*")
            for col in other:
                if st.checkbox(predictor_label(col), value=True, key=f"p_{col}"):
                    selected.append(col)

        # Phase-2.6 per-category predictor. Unlike the macro predictors above (one
        # shared value per month), this is PER-CATEGORY: each category's equation
        # sees only its OWN excess return, so it rides a dedicated design column,
        # not the shared exog matrix — only the VARX-CV model can consume it.
        has_excess = any(f"{c}_lag1" in frame.columns for c in EXCESS_WINDOWS)
        st.markdown("*Performance signal (per-category)*")
        use_excess = st.checkbox(
            "Excess return vs own benchmark", value=has_excess, disabled=not has_excess,
            help="Each category's AUM-weighted excess return vs its declared benchmark "
                 "(window 1/3/6m chosen per category by inner-CV). Per-category, not shared "
                 "like macro — consumed by the VARX-CV model (and per-category-select). "
                 "Toggle off and re-run for the no-signal baseline. "
                 + ("" if has_excess else "Requires data/category_excess.csv — not built yet."))

        st.subheader("Models")
        run_rw = st.checkbox("Random walk (baseline)", value=True)
        run_ar1 = st.checkbox("AR(1) (baseline)", value=True)
        run_varx = st.checkbox("VARX (joint, ridge — manual α)", value=True)
        run_varxcv = st.checkbox("VARX (auto-α, nested CV)", value=True,
                                 help="Picks ridge α by inner validation inside each "
                                      "training window — honest, no test-set snooping. "
                                      "Consumes the per-category performance signal when "
                                      "that predictor is toggled on above.")
        run_percat = st.checkbox("Per-category select (AR1 / VARX-CV by inner-CV)", value=False,
                                 help="For each category independently picks AR(1) or VARX-CV "
                                      "by inner-validation RMSE (the VARX-CV candidate uses the "
                                      "performance signal when that predictor is on). Honest — "
                                      "selection uses only training data.")
        run_markov = st.checkbox("Markov regime (slow)", value=False)

        st.subheader("Hyperparameters")
        ridge_alpha = st.select_slider(
            "VARX ridge α (shrinkage)",
            options=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0], value=1.0,
            help="Higher α shrinks predictor weights toward 0 (more regularization). "
                 "Eyeballing the test RMSE to pick α snoops the test set — use auto-α.",
        )
        # ---- two-handle evaluation window slider ----------------------------- #
        MIN_WARMUP = 36  # months of history required before first evaluation
        period_strs = [str(p) for p in periods]
        _eval_opts = period_strs[MIN_WARMUP:]
        _bt_range = st.select_slider(
            "Backtest evaluation window",
            options=_eval_opts,
            value=(_eval_opts[0], period_strs[-1]),
            help=f"Left handle = first month the model is evaluated on "
                 f"(minimum {MIN_WARMUP} months warm-up enforced). "
                 "Right handle = last month evaluated. "
                 "Training always uses all data strictly before each forecast date.",
        )
        eval_start_str, eval_end_str = _bt_range
        eval_start_idx = period_strs.index(eval_start_str)
        eval_end_idx = period_strs.index(eval_end_str) + 1  # exclusive upper bound
        min_train = eval_start_idx   # warm-up = everything before the eval window

        win_mode = st.radio(
            "Training window",
            ["Expanding (all history)", "Rolling (recent only)"], index=0,
            help="Expanding = train on everything up to each forecast date. "
                 "Rolling = train on only the most recent N months. (Backtests "
                 "say expanding wins — more history beats recency here.)",
        )
        window = None
        if win_mode.startswith("Rolling"):
            max_window = max(12, eval_start_idx)
            window = st.slider("Rolling window (months)", 12, max_window,
                               min(60, max_window))
        go = st.button("▶ Run backtest", type="primary")
        # Show reset whenever a run exists OR is being produced this pass (`go`),
        # since the sidebar renders before the `if go:` block stores the run.
        if go or st.session_state.get("run") is not None:
            if st.button("↩ Reset to signal table"):
                st.session_state.pop("run", None)
                st.rerun()

    # ---- run the backtest only when the button is clicked ----------------- #
    # Streamlit re-runs the whole script on every widget change. The result
    # selectboxes below must NOT recompute (or reset the view), so the run is
    # done once on click and stashed in session_state; the display reads from
    # there on subsequent reruns.
    if go:
        flows = wide_flows(frame)
        exog = wide_exog(frame, columns=selected) if selected else None
        # Phase-2.6 per-category excess signals (period × category), keyed by window —
        # only built when the user toggled the performance-signal predictor on.
        signals = {}
        if use_excess:
            for col, win in EXCESS_WINDOWS.items():
                sig = wide_category_signal(frame, f"{col}_lag1")
                if sig is not None:
                    signals[win] = sig

        factories = []
        if run_rw:
            factories.append((lambda: RandomWalkPredictor(), None, "random_walk"))
        if run_ar1:
            factories.append((lambda: AR1Predictor(), None, "ar1 (baseline)"))
        if run_varx:
            a = ridge_alpha
            factories.append((lambda a=a: VARXPredictor(order=1, ridge_alpha=a),
                              exog, f"varx (α={ridge_alpha})"))
        if run_varxcv:
            # The auto-α VARX consumes the per-category performance signal when that
            # predictor is on (→ excess-aware variant); otherwise the shared-exog VARX-CV.
            if signals:
                factories.append((lambda s=signals: VARXExcessCVPredictor(order=1, signals=s),
                                  exog, "varx (auto-α + perf signal)"))
            else:
                factories.append((lambda: VARXCVPredictor(order=1), exog, "varx (auto-α)"))
        if run_percat:
            factories.append((lambda s=signals: PerCategorySelectPredictor(order=1, signals=s),
                              exog, "per_cat_select (AR1/VARXCV)"))
        if run_markov:
            from fund_flow.predictor.regime import MarkovRegimePredictor
            factories.append((lambda: MarkovRegimePredictor(), None, "markov_regime"))

        if not factories:
            st.warning("Select at least one model.")
            return

        results = []
        prog = st.progress(0.0, text="Running walk-forward backtest…")
        for i, (fac, ex, name) in enumerate(factories):
            results.append(walk_forward(
                fac, flows, ex, min_train, name,
                window=window,
                test_start=eval_start_idx,
                test_end=eval_end_idx,
            ))
            prog.progress((i + 1) / len(factories), text=f"Done: {name}")
        prog.empty()

        st.session_state["run"] = {
            "results": {r.name: r for r in results},
            "order": [r.name for r in results],
            "tbl": compare(results),
            "flows": flows,
            "exog": exog,
            "selected": selected,
            "ridge_alpha": ridge_alpha,
            # Exactly what produced this leaderboard — so a result is reproducible
            # and you can tell why two runs differ.
            "config": {
                "n_months": len(periods),
                "full_span": f"{periods[0]} → {periods[-1]}",
                "eval_window": f"{eval_start_str} → {eval_end_str}",
                "eval_n": eval_end_idx - eval_start_idx,
                "warmup_months": eval_start_idx,
                "window": f"rolling-{window}" if window else "expanding",
                "ridge_alpha": ridge_alpha,
                "models": [name for _, _, name in factories],
                "predictors": ([predictor_label(c) for c in selected]
                               + (["excess-return (per-category)"] if signals else []))
                              or ["(none)"],
            },
        }

    run = st.session_state.get("run")
    if run is None:
        st.info("The table below is the **lag-1 correlation scan** — a model-free "
                "look at which predictors carry signal. It is **not** a backtest "
                "result. Pick predictors / models in the sidebar, then **Run "
                "backtest** for the out-of-sample leaderboard.")
        _signal_strength(frame, selected)
        return

    tbl = run["tbl"]
    flows = run["flows"]

    # ---- reproducibility banner: exactly what produced this leaderboard --- #
    rc = run.get("config")
    if rc:
        st.caption(
            f"**This run** · eval window **{rc['eval_window']}** "
            f"({rc['eval_n']} months, {rc['warmup_months']} months warm-up) · "
            f"train window **{rc['window']}** · VARX α **{rc['ridge_alpha']}** · "
            f"models: {', '.join(rc['models'])} · "
            f"predictors: {', '.join(rc['predictors'])}. "
            "Change a sidebar knob and re-run to compare; **Reset** returns to the "
            "signal table."
        )

    # ---- leaderboard + verdict ------------------------------------------- #
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Leaderboard (out-of-sample)")
        st.dataframe(tbl.style.format({"rmse": "{:.3f}", "mae": "{:.3f}"}),
                     width="stretch")
    with right:
        st.subheader("RMSE")
        st.bar_chart(tbl.set_index("model")["rmse"], horizontal=True)

    rmse = dict(zip(tbl["model"], tbl["rmse"]))
    if "ar1 (baseline)" in rmse:
        best = tbl.iloc[0]
        ar1 = rmse["ar1 (baseline)"]
        varx_keys = [m for m in rmse if m.startswith("varx")]
        if varx_keys:
            bestvarx = min(varx_keys, key=lambda k: rmse[k])
            d = (1 - rmse[bestvarx] / ar1) * 100
            verb = "beats" if d > 0 else "loses to"
            (st.success if d > 0 else st.warning)(
                f"Best VARX (**{bestvarx}**) {verb} AR(1) by {abs(d):.1f}% RMSE "
                f"out-of-sample. Overall best: **{best['model']}** "
                f"(RMSE {best['rmse']:.3f})."
            )

    # ---- actual vs predicted --------------------------------------------- #
    st.subheader("Actual vs predicted")
    model_names = run["order"]
    pick_model = st.selectbox("Model", model_names,
                              index=model_names.index(tbl.iloc[0]["model"]))
    cat = st.selectbox("Category", list(flows.columns))
    res = run["results"][pick_model]
    err = res.errors[cat]
    actual = flows[cat].reindex(err.index)
    chart_df = pd.DataFrame({
        "actual": actual,
        "predicted": actual - err,
    })
    chart_df.index = chart_df.index.astype(str)
    # actual = blue, predicted = orange
    st.line_chart(chart_df, color=["#1f77b4", "#ff7f0e"])
    st.caption("Predicted = actual − out-of-sample error, per the walk-forward run.")

    # ---- learned per-category weights ------------------------------------ #
    if run["selected"]:
        st.subheader("Learned weights (standardized ridge coefficients)")
        st.caption("⚠️ **In-sample** fit on the full sample — this is an explanatory "
                   "view of what the model leans on, NOT predictive performance. "
                   "Big, tidy-looking weights here do **not** mean better forecasts; "
                   "the honest score is the out-of-sample leaderboard above. "
                   "Magnitudes are comparable across rows and columns (e.g. each "
                   "category's loading on rates vs vix). These are LEARNED, not hand-set.")
        fit_a = run["ridge_alpha"]
        m = VARXPredictor(order=1, ridge_alpha=fit_a).fit(flows, run["exog"])
        coefs = m.coefficients()
        macro = coefs.loc[[c for c in coefs.index if not c.endswith("_flow_lag1")]]
        st.dataframe(
            macro.style.format("{:+.2f}").map(
                lambda v: _diverging_bg(v, 0.25)),
            width="stretch",
        )
        st.caption(f"VARX α={fit_a}. Re-run with a different α slider value to see "
                   "how shrinkage pulls weights toward 0.")


def _signal_strength(frame: pd.DataFrame, selected: list[str]) -> None:
    """Model-agnostic lag-1 correlation of each predictor with next-month flow,
    shown before a run so the user can see which predictors carry signal."""
    if not selected:
        return
    st.subheader("Predictor signal (lag-1 correlation with next-month flow)")
    st.caption("Correlation only — not causal. Helps decide which predictors to include.")
    wide = wide_flows(frame)
    ex = wide_exog(frame, columns=selected).reindex(wide.index)
    rows = {}
    for col in selected:
        rows[predictor_label(col)] = {
            cat: wide[cat].corr(ex[col]) for cat in wide.columns
        }
    corr = pd.DataFrame(rows).T
    st.dataframe(corr.style.format("{:+.2f}").map(
        lambda v: _diverging_bg(v, 0.5)), width="stretch")


# --------------------------------------------------------------------------- #
# ABM tab (preview — engine lands in Phase 3)
# --------------------------------------------------------------------------- #
def abm_tab() -> None:
    st.header("Credit-fund-complex ABM — scenario preview")
    st.info("The ABM **simulation engine is Phase 3** (not built yet). This tab "
            "previews the synthetic fund complex and the stress FlowShock the "
            "engine will run, with the knobs you'll later use to drive it.")

    cfg = dict(_cfg())
    abm = dict(cfg["synthetic"]["abm"])
    with st.sidebar:
        st.subheader("ABM book parameters")
        abm["n_funds"] = st.slider("Funds", 4, 40, int(abm["n_funds"]))
        abm["n_instruments"] = st.slider("Instruments", 10, 150,
                                         int(abm["n_instruments"]))
        abm["min_overlap"] = st.slider("Min funds per instrument (overlap)", 1, 10,
                                       int(abm["min_overlap"]))
        abm["liquid_buffer_pct"] = st.slider("Liquid buffer (% of NAV)", 0.0, 0.30,
                                             float(abm["liquid_buffer_pct"]))
    cfg["synthetic"] = dict(cfg["synthetic"]); cfg["synthetic"]["abm"] = abm

    try:
        from fund_flow.synthetic.abm_data import (
            generate_synthetic_book,
            stress_flow_shock,
        )
        book = generate_synthetic_book(cfg)
        shocks = stress_flow_shock(cfg)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not build synthetic book: {exc}")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Funds", len(book.fund_names))
    c2.metric("Instruments", len(book.instruments))
    total_nav = float(book.nav.sum())
    c3.metric("Total NAV (R$)", f"{total_nav:,.0f}")

    st.subheader("NAV by fund")
    nav = book.nav.copy()
    nav.index = [str(i) for i in nav.index]
    st.bar_chart(nav)

    st.subheader("Stress scenario — FlowShock the ABM will consume")
    sh = pd.DataFrame([{
        "period": s.period, "category": s.category,
        "net_flow_brl": s.net_flow_brl,
        "redemption_gross_brl": s.redemption_gross_brl,
    } for s in shocks])
    st.dataframe(sh, width="stretch")
    st.caption("Phase 3 will route this shock through the allocation layer → "
               "sqrt market impact → mark-to-market contagion across the shared book.")


# --------------------------------------------------------------------------- #
# Methodology & findings report
# --------------------------------------------------------------------------- #
def report_tab() -> None:
    st.header("Methodology & findings")
    st.caption("Why this model, these data, these predictors — and an honest "
               "account of what we tried that did NOT work. All numbers below are "
               "walk-forward, out-of-sample (CLAUDE.md §7).")

    st.markdown(r"""
### 1. What we are forecasting, and the two-stage design
The target is **captação líquida** (net new money = inflows − outflows) per Anbima
category, one month ahead. This forecast is the **impulse** consumed by a separate
structural **ABM** that simulates how a redemption shock propagates across a complex
of credit funds sharing a debenture book. The split is deliberate: a statistical
model is the right tool for *how much* money moves; an agent-based model is the right
tool for *what that flow does to a fragile system*. Data flows **one way**
(predictor → ABM); feeding simulated flows back into training would be circular.

### 2. Data sources — and why each
| Layer | Source | Span | Why this source |
|---|---|---|---|
| **Flows** | CVM Informe Diário (RCVM-175) | 2004-04 → 2025-12, 6 categories | Free, no auth; gives **gross** aportes *and* resgates per fund per day → serves both the predictor (aggregate) and the ABM (fund-level). Anbima's API never exposed a flows endpoint. |
| **Rates / FX / prices** | BCB SGS · FRED · Yahoo | full window | Free, deep history. Selic/IPCA/USD (BCB), Fed funds/UST-10y/**VIX** (FRED), Ibovespa/S&P 500/DXY (Yahoo). |
| **Credit spread** | IPEAdata EMBI+ (`JPM366_EMBI366`) | 1994 → 2024-07 | The genuine sovereign credit premium. **Opt-in** — tested, did not help (§5). |
| **Fund excess returns** (Phase 2.6) | CVM `VL_QUOTA` per fund vs **declared** benchmark (`Indicador_Desempenho`) | 2004 → 2025, 263 mo | AUM-weighted per-category excess return vs each fund's OWN benchmark. The performance-chasing signal — the first lever to beat the base OOS (§5). |

**Benchmark return series (per canonical ID).** Each fund's declared benchmark is
mapped to a canonical ID and a monthly total-return series: CDI/SELIC (BCB SGS 12/11,
compounded), IPCA (SGS 433), IBOV (Yahoo ^BVSP), SP500-in-BRL (^GSPC × BRL/USD FX);
and documented **proxies** — IMA-B/IMA-B5 via B3 ETFs IMAB11.SA/B5P211.SA (~2019+),
MSCI-World via URTH, IBRX100→IBOV, IHFA→CDI, IDA→IMA-B. Funds whose benchmark can't be
resolved (`ZERO`: Cambial, ~62% of equity that declares none after the *tightened* Ações
rule) are **excluded**, never given a fabricated benchmark. Coverage of excess-panel AUM
rises from ~42% (2004, survivorship) to ~93% (2025).

**Caveats we do not hide.** (a) *Survivorship*: the CVM class registry only lists
surviving classes, so pre-2004 coverage is < 50% — we floor the sample at 2004.
(b) *come-cotas* (May/Nov withholding) is seasonality, not signal — it is a control,
never a predictor. (c) Internal positions are out of scope; everything here is public.

### 3. Predictors and the lag discipline
Every predictor is **lagged at least one month**: we forecast flow *t* from macro/returns
realized at *t-1*, plus the come-cotas calendar dummy known ex-ante. Contemporaneous
returns are *structurally dropped* from the modeling frame so they cannot leak in. This
is not pedantry — **flows and returns are jointly determined** (reflexivity), so using
contemporaneous returns would bake in a non-causal correlation
([Warther 1995](https://www.sciencedirect.com/science/article/abs/pii/0304405X95008272)).

Two kinds of predictor feed the model: **shared macro** (one value per month, broadcast to
every category — Selic, VIX, …) and one **per-category predictor** added in Phase 2.6:

* **Per-category excess return (performance signal).** Each category's AUM-weighted excess
  return vs each fund's **own declared benchmark** — a different value per category in the
  same month (windows 1/3/6m, chosen per category by inner-CV). The economic rationale is
  **performance-chasing**: investors steer money toward categories that recently beat their
  benchmark ([Sirri & Tufano 1998](https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00066)).
  We deliberately use each fund's *own* benchmark, not one index per category: within Ações,
  international funds are benchmarked to S&P 500 / MSCI, not Ibovespa, and imposing Ibovespa
  universally injects noise exactly where the signal should be cleanest. Because it is
  per-category, it cannot ride the shared-exog matrix — it enters as a dedicated design
  column in each category's own equation (so only the joint VARX consumes it).
""")

    st.markdown(r"""
### 4. Models — and why these, not others
* **Random walk & AR(1)** — mandatory naive baselines. A candidate is only "good" if
  it beats both out-of-sample.
* **VARX (joint ridge VAR)** — all six categories in one system so the model can learn
  **rotation** between categories (money leaving Multimercado for Renda Fixa). Estimated
  with a **standardized ridge penalty** because OLS overfits a 6-equation system on a
  ~260-month sample. Shrinkage of many weak predictors is the textbook fix
  (De Mol, Giannone & Reichlin 2008; the Minnesota/Bayesian-VAR tradition, Litterman 1986).
* **VARX-CV** — picks the ridge α by **nested cross-validation inside each training
  window**, so α never sees the test set (the honest alternative to eyeballing test RMSE).
* **Markov regime-switching** — Selic high vs low are different worlds; a two-state model
  (Hamilton 1989) lets parameters switch.
* **VARX-CV with the performance signal (Phase 2.6)** — *not a new model*: the same
  VARX-CV, fed the **per-category excess-return predictor** from §3. Each category's equation
  gains one column — its own excess return at the inner-CV-selected window — so the comparison
  "VARX-CV-base vs VARX-CV-excess" in §5 is an **ablation of that predictor**, not two
  different estimators. In the dashboard it is a predictor toggle, not a model checkbox.
* **Why no deep learning.** ML's documented wins in finance come from *huge* cross-sections
  ([Gu, Kelly & Xiu 2020](https://academic.oup.com/rfs/article/33/5/2223/5758276): thousands
  of stocks × decades). Our sample is **6 series × ~260 months**. Neural nets would memorize
  noise; this is exactly the regime where regularized linear models dominate.

### 5. Findings — honest, out-of-sample
**(a) Flows are persistence-dominated; AR(1) is very hard to beat.** The best joint model
(VARX-CV, net-only) beats AR(1) by **+0.4%** pooled RMSE — real but tiny — and *loses* to
AR(1) on Crédito Privado. This is the
[Welch & Goyal (2008)](https://academic.oup.com/rfs/article-abstract/21/4/1455/1565737)
lesson in miniature: in-sample richness, but out-of-sample the naive baseline wins.

**(b) Macro adds a small, economically-sensible increment.** VIX is the single best macro
driver (Crédito Privado loads negatively; Renda Fixa positively — flight-to-safety).
Loading up on more macro at low shrinkage *hurts* OOS; the best α is high (≈50).

**(c) Things we tried that did NOT work** (each made OOS worse, not better):

| Idea tested | Result (OOS) | Verdict |
|---|---|---|
| Sovereign **credit spread** (EMBI+) | pooled **+0.03%**; Crédito Privado **−0.16%** | No help — collinear with VIX |
| **Gross-flow decomposition** (joint net+redemption VAR) | **−1% to −12%** per category | Hurts — adds parameters, not signal |
| Gross decomposition, parsimonious (own redemption-lag) | **−0.7% to −2.6%** | Hurts — net-flow lag already carries it |
| **Rolling / shifted** training window (48–120 mo) | up to **−4.6%** vs expanding | Hurts — more history beats recency |

**(d) What DID work — the performance-chasing excess-return signal (Phase 2.6).** Giving
each category its OWN AUM-weighted excess return vs its declared benchmark is the **first
lever in this project to beat the base predictor out-of-sample**. Per-category OOS RMSE,
263 months, vs **VARX-CV-base** (macro only):

| Category | AR(1) | VARX-CV-base | VARX-CV-excess | Δ vs base |
|---|--:|--:|--:|--:|
| Ações | 3.354 | 3.430 | 3.403 | **+0.76%** |
| Cambial | 0.397 | 0.400 | 0.402 | −0.50% |
| Crédito Privado | 20.294 | 21.019 | 20.724 | **+1.41%** |
| Multimercado | 8.525 | 8.919 | 8.716 | **+2.27%** |
| Previdência | 4.442 | 4.406 | 4.408 | −0.05% |
| Renda Fixa | 25.344 | 24.422 | 24.351 | +0.29% |
| **POOLED** | 13.892 | 13.839 | **13.700**… 13.721 | **+0.85%** |

Inner-CV selects the window per category (Ações **3m**, Multimercado & Crédito Privado
**1m**; others **none**). The gains land exactly where performance-chasing is strongest
(Multimercado, Crédito Privado, Ações), consistent with Sirri & Tufano (1998).

### 6. So what *is* better? The honest verdict
**The excess-return predictor is a genuine, modest improvement — the binding constraint
loosened, but only a little.** Feeding it to the VARX-CV (pooled **13.72**) beats both the
same model without it (13.84) and AR(1) (13.89). Two deployments, documented without picking
a forced winner:

* **Joint VARX-CV + signal — best pooled.** The VARX-CV consuming the predictor in every
  equation where inner-CV finds it helps. Cleanest single impulse generator (+0.85% vs the
  no-signal VARX-CV, +1.2% vs AR(1)).
* **Per-category select (AR(1) vs VARX-CV ± signal) — most conservative.** Picks the best
  candidate per category on training data only. **Honest caveat:** AR(1) still *individually*
  wins Ações/Multimercado/Crédito Privado (e.g. Multimercado AR(1) 8.525 < 8.716 with the
  signal), so selection mostly falls back to AR(1) there — its value is narrower than the
  +2.3% Multimercado number on the joint model suggests.

The earlier negative levers (EMBI, gross decomposition, rolling windows) still stand as
tried-and-failed. And the highest-value work remains **structural — the ABM**:
flow-driven fire sales and mark-to-market contagion
([Coval & Stafford 2007](https://www.sciencedirect.com/science/article/abs/pii/S0304405X07001158);
Shleifer & Vishny 2011) under a **square-root price-impact** law
([Tóth et al. 2011](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.1.021006);
Almgren et al. 2005). One untested low-priority lever: *idiosyncratic* credit-event flags
for Crédito Privado (e.g. the Jan-2023 Americanas / Light defaults) that a sovereign spread
is blind to.

**Limitations of the excess signal (not hidden).** It rests partly on assumption: ~62% of
Ações AUM declares no benchmark, and after the *tightened* rule (no fabricated IBOV for
active equity) Ações excess-panel coverage is ~55% recent / ~44% in 2010; early-year
coverage is thinner from survivorship (the registry is a current snapshot). The signal is
real, but it is read against declared-or-defaulted benchmarks, not a perfect counterfactual.

### References
- Warther, V. A. (1995). *Aggregate mutual fund flows and security returns.* Journal of Financial Economics 39, 209–235.
- Welch, I. & Goyal, A. (2008). *A Comprehensive Look at the Empirical Performance of Equity Premium Prediction.* Review of Financial Studies 21(4), 1455–1508.
- Sirri, E. R. & Tufano, P. (1998). *Costly Search and Mutual Fund Flows.* Journal of Finance 53(5), 1589–1622.
- Hamilton, J. D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle.* Econometrica 57(2), 357–384.
- Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions.* Journal of Business & Economic Statistics 4(1), 25–38.
- De Mol, C., Giannone, D. & Reichlin, L. (2008). *Forecasting using a large number of predictors.* Journal of Econometrics 146(2), 318–328.
- Gu, S., Kelly, B. & Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning.* Review of Financial Studies 33(5), 2223–2273.
- Coval, J. & Stafford, E. (2007). *Asset fire sales (and purchases) in equity markets.* Journal of Financial Economics 86, 479–512.
- Shleifer, A. & Vishny, R. (2011). *Fire Sales in Finance and Macroeconomics.* Journal of Economic Perspectives 25(1), 29–48.
- Tóth, B. et al. (2011). *Anomalous price impact and the critical nature of liquidity.* Physical Review X 1, 021006.
- Almgren, R., Thum, C., Hauptmann, E. & Li, H. (2005). *Direct estimation of equity market impact.* Risk 18(7), 57–62.
""")


# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("📈 Fund-Flow Predictor → Credit-Fund-Complex ABM")
    tab1, tab2, tab3 = st.tabs(
        ["Predictor", "ABM (preview)", "Methodology & findings"])
    with tab1:
        predictor_tab()
    with tab2:
        abm_tab()
    with tab3:
        report_tab()


main()

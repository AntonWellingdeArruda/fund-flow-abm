"""Synthetic macro + fund-flow panel for Predictor development and testing.

Economic structure embedded (required for Phase 2 validation):
  - Flow autocorrelation: AR(1) within each category (ar1_flow_coeff).
  - Returns lead flows: lagged Ibovespa → next-period Ações flow;
    lagged Δselic → next-period Renda Fixa flow.
  - Regime rotation: in high-Selic regime, flows shift from
    Ações/Multimercado toward Renda Fixa/Crédito Privado.
  - come_cotas flag: control variable only; never used as a flow adjustment.

RNG namespace: [seed, 0] — independent from abm_data's [seed, 1].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fund_flow.utils.calendar import is_come_cotas


# Regime rotation: mean monthly flow (BRL bn) per category per regime.
# Positive = net inflow; negative = net redemption.
_REGIME_ROTATION: dict[int, dict[str, float]] = {
    0: {  # low-Selic regime: equity / multi attract flows
        "Ações": 4.0,
        "Multimercado": 3.5,
        "Renda Fixa": 1.0,
        "Crédito Privado": 1.5,
        "Previdência": 2.0,
        "Cambial": 0.5,
    },
    1: {  # high-Selic regime: fixed-income / credit attract flows
        "Ações": -1.5,
        "Multimercado": 0.5,
        "Renda Fixa": 5.0,
        "Crédito Privado": 3.5,
        "Previdência": 2.0,
        "Cambial": 0.5,
    },
}

# Noise std per category (BRL bn/month)
_NOISE_STD: dict[str, float] = {
    "Ações": 2.5,
    "Multimercado": 2.0,
    "Renda Fixa": 2.0,
    "Crédito Privado": 1.5,
    "Previdência": 1.0,
    "Cambial": 0.8,
}

# Macro beta: which category's flow responds to which lagged macro signal.
# Only two are wired per plan; others have beta=0 (pure noise + rotation).
_MACRO_BETAS: dict[str, tuple[str, float]] = {
    "Ações": ("ibovespa_return", 0.0),       # overridden by ibov_to_acoes_lag_beta
    "Renda Fixa": ("delta_selic", 0.0),      # overridden by selic_to_rf_lag_beta
}


def generate_predictor_panel(cfg: dict) -> pd.DataFrame:
    """Return long-format DataFrame with one row per (period, category).

    Columns: period, category, net_flow_brl, redemption_gross_brl,
             selic_rate, delta_selic, ipca_monthly, ibovespa_return,
             dxy_return, ust_10y, regime, come_cotas.
    """
    rng = np.random.default_rng([cfg["seed"], 0])
    pc = cfg["synthetic"]["predictor"]
    categories: list[str] = cfg["categories"]
    exempt: list[str] = cfg["come_cotas_exempt"]

    n_months: int = pc["n_months"]
    ar1: float = pc["ar1_flow_coeff"]
    ibov_beta: float = pc["ibov_to_acoes_lag_beta"]
    selic_rf_beta: float = pc["selic_to_rf_lag_beta"]

    periods = pd.period_range(start=pc["start_date"], periods=n_months, freq="M")

    # --- Macro simulation ---
    macro_rng, flow_rng = rng.spawn(2)

    # Regime chain: 2-state Markov with ~18-month average duration
    trans = np.array([[0.944, 0.056], [0.056, 0.944]])
    regimes = np.empty(n_months, dtype=int)
    regimes[0] = 1  # start in high-Selic
    for t in range(1, n_months):
        regimes[t] = macro_rng.choice([0, 1], p=trans[regimes[t - 1]])

    selic_high = pc["selic_high_mean"]
    selic_low = pc["selic_low_mean"]
    selic_rate = np.where(regimes == 1, selic_high, selic_low)
    selic_rate += macro_rng.normal(0, 0.005, n_months)
    selic_rate = np.clip(selic_rate, 0.02, 0.20)
    delta_selic = np.diff(selic_rate, prepend=selic_rate[0])

    ipca_monthly = 0.004 + 0.12 * selic_rate + macro_rng.normal(0, 0.002, n_months)
    ibovespa_return = (
        np.where(regimes == 0, 0.012, -0.005)
        + macro_rng.normal(0, 0.05, n_months)
    )
    dxy_return = macro_rng.normal(0, 0.015, n_months)
    ust_10y = 0.04 + macro_rng.normal(0, 0.005, n_months)

    # --- Flow simulation ---
    flow_net = {cat: np.zeros(n_months) for cat in categories}
    gross_out = {cat: np.zeros(n_months) for cat in categories}

    for t in range(n_months):
        for cat in categories:
            regime_mean = _REGIME_ROTATION[regimes[t]][cat]

            # Macro lag signal (t-1 values already computed above)
            lag_t = max(t - 1, 0)
            macro_signal = 0.0
            if cat == "Ações":
                macro_signal = ibov_beta * ibovespa_return[lag_t]
            elif cat == "Renda Fixa":
                macro_signal = selic_rf_beta * delta_selic[lag_t]

            ar_term = ar1 * flow_net[cat][t - 1] if t > 0 else 0.0
            noise = flow_rng.normal(0, _NOISE_STD[cat])
            flow_net[cat][t] = regime_mean + ar_term + macro_signal + noise

            # Gross redemptions: always positive; modelled as a fraction of NAV proxy
            # Inflow months have small mechanical redemptions; outflow months larger.
            base_gross = max(abs(flow_net[cat][t]) * 0.6 + abs(noise) * 0.3, 0.1)
            gross_out[cat][t] = base_gross

    # --- Assemble long-format DataFrame ---
    rows = []
    for i, period in enumerate(periods):
        period_str = str(period)
        for cat in categories:
            rows.append({
                "period": period_str,
                "category": cat,
                "net_flow_brl": flow_net[cat][i],
                "redemption_gross_brl": gross_out[cat][i],
                "selic_rate": selic_rate[i],
                "delta_selic": delta_selic[i],
                "ipca_monthly": ipca_monthly[i],
                "ibovespa_return": ibovespa_return[i],
                "dxy_return": dxy_return[i],
                "ust_10y": ust_10y[i],
                "regime": int(regimes[i]),
                "come_cotas": is_come_cotas(period_str, cat, exempt),
            })

    return pd.DataFrame(rows).reset_index(drop=True)

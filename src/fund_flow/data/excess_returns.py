"""Fund-level and category-level excess returns (Phase 2.6, Step 3).

Pipeline (all pure / offline-testable; the network I/O lives in
scripts/fetch_excess_returns.py):

  1. month_end_stats(informe, period) — per fund, the month-end quota + PL and the
     number of distinct days it reported a quota that month.
  2. chain_returns(stats_by_period) — fund monthly return from consecutive
     month-end quotas: r_f_t = quota_t / quota_{t-1} − 1 (only when t-1 is the
     immediately preceding month; gaps break the chain → NaN).
  3. aggregate_category_excess(...) — subtract each fund's benchmark return, apply
     exclusion filters, and AUM-weight to a (period, category) category excess
     return ER_c_t.

Exclusion filters (a fund-month is dropped if any holds):
  * benchmark maps to ZERO (no usable benchmark)
  * fewer than `min_days` (default 15) distinct quota observations in the month
  * month-end PL < `min_pl` (default R$ 1M) — micro funds are noise
  * |fund return| > `max_abs_ret` (default 0.5) — a quota jump that large is
    almost surely a data artefact (split / quota reset), not a real monthly return
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _period_ord(period_str: pd.Series) -> pd.Series:
    """'YYYY-MM' → integer month ordinal (year*12 + month) for gap detection."""
    p = pd.PeriodIndex(period_str, freq="M")
    return pd.Series(p.year * 12 + p.month, index=period_str.index)


def month_end_stats(informe: pd.DataFrame, period: str) -> pd.DataFrame:
    """Per fund: month-end quota + PL and #days with a quota, for one month.

    `informe` is one month of parsed informe-diário rows [cnpj, date, ..., pl,
    quota]. Rows without a quota are ignored for the month-end pick and day count.
    """
    df = informe.dropna(subset=["quota"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["period", "cnpj", "quota", "pl", "n_days"])
    df["_d"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["_d"])
    last_idx = df.groupby("cnpj")["_d"].idxmax()
    last = df.loc[last_idx, ["cnpj", "quota", "pl"]].reset_index(drop=True)
    n_days = df.groupby("cnpj")["_d"].nunique().rename("n_days")
    out = last.merge(n_days, on="cnpj")
    out.insert(0, "period", period)
    return out[["period", "cnpj", "quota", "pl", "n_days"]]


def chain_returns(stats_by_period: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Chain month-end quotas into fund monthly returns.

    Returns long [period, cnpj, ret, pl, n_days]; `ret` is NaN when the previous
    month-end quota is missing or not the immediately preceding month.
    """
    if not stats_by_period:
        return pd.DataFrame(columns=["period", "cnpj", "ret", "pl", "n_days"])
    allrows = pd.concat(stats_by_period.values(), ignore_index=True)
    allrows = allrows.sort_values(["cnpj", "period"]).reset_index(drop=True)
    g = allrows.groupby("cnpj", sort=False)
    prev_quota = g["quota"].shift(1)
    prev_ord = g["period"].shift(1)
    cur_ord = _period_ord(allrows["period"])
    prev_ord_num = _period_ord(prev_ord.fillna(allrows["period"]))
    consecutive = (cur_ord.to_numpy() - prev_ord_num.to_numpy()) == 1

    ret = allrows["quota"] / prev_quota - 1.0
    ret = ret.where(consecutive)
    out = allrows[["period", "cnpj", "pl", "n_days"]].copy()
    out["ret"] = ret.to_numpy()
    return out[["period", "cnpj", "ret", "pl", "n_days"]]


def aggregate_category_excess(
    returns_df: pd.DataFrame,            # [period, cnpj, ret, pl, n_days]
    benchmark_of: dict[str, str],        # cnpj → canonical benchmark ID
    category_of: dict[str, str],         # cnpj → Anbima category
    bench_returns: pd.DataFrame,         # PeriodIndex × benchmark-ID columns
    min_days: int = 15,
    min_pl: float = 1e6,
    max_abs_ret: float = 0.5,
) -> pd.DataFrame:
    """AUM-weighted category excess returns ER_c_t (period, category, value, …).

    Output columns: period, category, cat_excess_return, n_funds, aum_brl.
    """
    df = returns_df.dropna(subset=["ret"]).copy()
    df = df[(df["n_days"] >= min_days) & (df["pl"] >= min_pl)
            & (df["ret"].abs() <= max_abs_ret)]
    df["category"] = df["cnpj"].map(category_of)
    df["bench_id"] = df["cnpj"].map(benchmark_of)
    df = df.dropna(subset=["category", "bench_id"])
    df = df[df["bench_id"] != "ZERO"]
    if df.empty:
        return pd.DataFrame(
            columns=["period", "category", "cat_excess_return", "n_funds", "aum_brl"])

    # benchmark return per (period, bench_id) via a long merge
    blong = bench_returns.stack().rename("bench_ret").reset_index()
    blong.columns = ["period", "bench_id", "bench_ret"]
    blong["period"] = blong["period"].astype(str)
    df = df.merge(blong, on=["period", "bench_id"], how="left")
    df = df.dropna(subset=["bench_ret"])
    df["er"] = df["ret"] - df["bench_ret"]

    def _wavg(x: pd.DataFrame) -> pd.Series:
        w = x["pl"].to_numpy(float)
        return pd.Series({
            "cat_excess_return": float(np.average(x["er"].to_numpy(float), weights=w)),
            "n_funds": int(len(x)),
            "aum_brl": float(w.sum()),
        })

    out = (df.groupby(["period", "category"], sort=True)
             .apply(_wavg, include_groups=False)
             .reset_index())
    return out

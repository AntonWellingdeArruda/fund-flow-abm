"""Offline tests for the fund/category excess-return pipeline (Phase 2.6, Step 3)."""
import numpy as np
import pandas as pd
import pytest

from fund_flow.data.excess_returns import (
    aggregate_category_excess,
    chain_returns,
    month_end_stats,
)


def _informe(rows):
    return pd.DataFrame(rows, columns=["cnpj", "date", "captacao", "resgate", "pl", "quota"])


class TestMonthEndStats:
    def test_picks_last_quota_and_counts_days(self):
        inf = _informe([
            ("A", "2024-04-01", 0, 0, 1e7, 1.00),
            ("A", "2024-04-15", 0, 0, 1.1e7, 1.05),
            ("A", "2024-04-30", 0, 0, 1.2e7, 1.10),   # month-end
            ("B", "2024-04-10", 0, 0, 5e6, 2.00),
        ])
        out = month_end_stats(inf, "2024-04").set_index("cnpj")
        assert out.loc["A", "quota"] == pytest.approx(1.10)   # last by date
        assert out.loc["A", "pl"] == pytest.approx(1.2e7)
        assert out.loc["A", "n_days"] == 3
        assert out.loc["B", "n_days"] == 1

    def test_ignores_rows_without_quota(self):
        inf = _informe([("A", "2024-04-01", 0, 0, 1e7, np.nan)])
        assert month_end_stats(inf, "2024-04").empty


class TestChainReturns:
    def test_consecutive_month_return(self):
        stats = {
            "2024-04": month_end_stats(_informe([("A", "2024-04-30", 0, 0, 1e7, 1.00)]), "2024-04"),
            "2024-05": month_end_stats(_informe([("A", "2024-05-31", 0, 0, 1e7, 1.10)]), "2024-05"),
        }
        out = chain_returns(stats).set_index("period")
        assert np.isnan(out.loc["2024-04", "ret"])          # no prior month
        assert out.loc["2024-05", "ret"] == pytest.approx(0.10)

    def test_gap_breaks_chain(self):
        # April then June (May missing) → June return is NaN (no look-ahead bridge)
        stats = {
            "2024-04": month_end_stats(_informe([("A", "2024-04-30", 0, 0, 1e7, 1.00)]), "2024-04"),
            "2024-06": month_end_stats(_informe([("A", "2024-06-30", 0, 0, 1e7, 1.20)]), "2024-06"),
        }
        out = chain_returns(stats).set_index("period")
        assert np.isnan(out.loc["2024-06", "ret"])


class TestAggregateCategoryExcess:
    def _bench(self):
        idx = pd.PeriodIndex(["2024-05"], freq="M")
        return pd.DataFrame({"IBOV": [0.04], "CDI": [0.008], "ZERO": [0.0]}, index=idx)

    def test_aum_weighted_excess_and_exclusions(self):
        returns_df = pd.DataFrame([
            # cnpj, period, ret, pl, n_days
            ("A", "2024-05", 0.10, 9e9, 22),   # Ações vs IBOV → er +0.06, huge AUM
            ("B", "2024-05", 0.05, 1e9, 22),   # Ações vs IBOV → er +0.01, small AUM
            ("C", "2024-05", 0.20, 5e8, 22),   # PL < 1M? no (5e8) but ZERO bench → excluded
            ("D", "2024-05", 0.30, 1e5, 22),   # PL < 1M → excluded
            ("E", "2024-05", 0.90, 9e9, 22),   # |ret|>0.5 → excluded (data artefact)
            ("F", "2024-05", 0.02, 9e9, 5),    # n_days < 15 → excluded
        ], columns=["cnpj", "period", "ret", "pl", "n_days"])
        bench_of = {"A": "IBOV", "B": "IBOV", "C": "ZERO", "D": "IBOV", "E": "IBOV", "F": "IBOV"}
        cat_of = {k: "Ações" for k in "ABCDEF"}
        out = aggregate_category_excess(returns_df, bench_of, cat_of, self._bench())
        assert list(out["category"]) == ["Ações"]
        row = out.iloc[0]
        assert row["n_funds"] == 2                  # only A and B survive
        # AUM-weighted er: (0.06*9e9 + 0.01*1e9)/1e10 = 0.055
        assert row["cat_excess_return"] == pytest.approx(0.055, abs=1e-9)
        assert row["aum_brl"] == pytest.approx(1e10)

    def test_no_nan_in_output(self):
        returns_df = pd.DataFrame([("A", "2024-05", 0.10, 9e9, 22)],
                                  columns=["cnpj", "period", "ret", "pl", "n_days"])
        out = aggregate_category_excess(
            returns_df, {"A": "IBOV"}, {"A": "Ações"}, self._bench())
        assert not out.isna().any().any()

    def test_missing_benchmark_month_excluded(self):
        # benchmark has no row for this fund's period → fund-month dropped, no crash
        returns_df = pd.DataFrame([("A", "2024-09", 0.10, 9e9, 22)],
                                  columns=["cnpj", "period", "ret", "pl", "n_days"])
        out = aggregate_category_excess(
            returns_df, {"A": "IBOV"}, {"A": "Ações"}, self._bench())
        assert out.empty

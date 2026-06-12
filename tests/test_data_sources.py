import pandas as pd
import pytest

from fund_flow.data.sources import (
    FLOW_COLUMNS,
    MACRO_COLUMNS,
    AnbimaFlowSource,
    CsvFlowSource,
    CsvMacroSource,
    SyntheticFlowSource,
    SyntheticMacroSource,
)


class TestSyntheticSources:
    def test_flow_source_columns(self, cfg):
        df = SyntheticFlowSource(cfg).load()
        assert list(df.columns) == FLOW_COLUMNS

    def test_flow_source_row_count(self, cfg):
        df = SyntheticFlowSource(cfg).load()
        n_months = cfg["synthetic"]["predictor"]["n_months"]
        assert len(df) == n_months * len(cfg["categories"])

    def test_macro_source_one_row_per_period(self, cfg):
        df = SyntheticMacroSource(cfg).load()
        n_months = cfg["synthetic"]["predictor"]["n_months"]
        assert list(df.columns) == MACRO_COLUMNS
        assert len(df) == n_months
        assert not df["period"].duplicated().any()


class TestCsvSources:
    def test_flow_csv_roundtrip(self, cfg, tmp_path):
        src = SyntheticFlowSource(cfg).load()
        path = tmp_path / "flows.csv"
        src.to_csv(path, index=False)
        loaded = CsvFlowSource(str(path)).load()
        assert list(loaded.columns) == FLOW_COLUMNS
        assert len(loaded) == len(src)

    def test_macro_csv_roundtrip(self, cfg, tmp_path):
        src = SyntheticMacroSource(cfg).load()
        path = tmp_path / "macro.csv"
        src.to_csv(path, index=False)
        loaded = CsvMacroSource(str(path)).load()
        assert list(loaded.columns) == MACRO_COLUMNS
        assert len(loaded) == len(src)


class TestRealSourceStubs:
    def test_anbima_stub_raises(self):
        # Anbima flows are still a documented seam (spreadsheet scraping TBD).
        with pytest.raises(NotImplementedError):
            AnbimaFlowSource().load()

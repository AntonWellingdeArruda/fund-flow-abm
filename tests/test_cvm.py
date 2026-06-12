import pandas as pd
import pytest

from fund_flow.data import cvm
from fund_flow.data.cvm import (
    _parse_cadastro_csv,
    _parse_informe_csv,
    normalize_cnpj,
)
from fund_flow.data.sources import CvmFlowSource, FLOW_COLUMNS

INFORME_CSV = (
    "CNPJ_FUNDO_CLASSE;DT_COMPTC;VL_TOTAL;VL_QUOTA;VL_PATRIM_LIQ;CAPTC_DIA;RESG_DIA;NR_COTST\n"
    "00.000.000/0001-00;2024-04-01;1000,5;1,23;900,25;100,0;30,5;10\n"
    "00.000.000/0001-00;2024-04-02;1010,0;1,24;905,00;50,0;20,0;10\n"
    "11.111.111/0001-11;2024-04-01;2000,0;2,00;1800,00;200,0;80,0;5\n"
)

# older schema variant (CNPJ_FUNDO instead of CNPJ_FUNDO_CLASSE)
INFORME_CSV_OLD = (
    "CNPJ_FUNDO;DT_COMPTC;VL_TOTAL;VL_QUOTA;VL_PATRIM_LIQ;CAPTC_DIA;RESG_DIA;NR_COTST\n"
    "22.222.222/0001-22;2020-01-02;5,0;1,0;5,0;3,0;1,0;2\n"
)


class TestNormalizeCnpj:
    def test_strips_formatting(self):
        assert normalize_cnpj("00.000.000/0001-00") == "00000000000100"


class TestParseInforme:
    def test_decimal_comma_and_columns(self):
        df = _parse_informe_csv(INFORME_CSV)
        assert list(df.columns) == ["cnpj", "date", "captacao", "resgate", "pl"]
        assert df["captacao"].iloc[0] == pytest.approx(100.0)
        assert df["resgate"].iloc[0] == pytest.approx(30.5)
        assert df["cnpj"].iloc[0] == "00000000000100"

    def test_old_schema_cnpj_alias(self):
        df = _parse_informe_csv(INFORME_CSV_OLD)
        assert df["cnpj"].iloc[0] == "22222222000122"

    def test_missing_cnpj_column_raises(self):
        with pytest.raises(cvm.CvmFetchError, match="CNPJ"):
            _parse_informe_csv("DT_COMPTC;CAPTC_DIA;RESG_DIA\n2024-01-01;1,0;1,0\n")


class TestParseCadastro:
    def test_extracts_cnpj_classe(self):
        text = ("CNPJ_FUNDO;DENOM_SOCIAL;CLASSE\n"
                "00.000.000/0001-00;FUNDO X;Fundo de Ações\n")
        df = _parse_cadastro_csv(text)
        assert df["classe"].iloc[0] == "Fundo de Ações"
        assert df["cnpj"].iloc[0] == "00000000000100"


class TestCvmFlowSource:
    def test_aggregates_monthly_per_category(self, monkeypatch):
        # both CNPJs map to Ações; one month of data
        def fake_fetch(ym, timeout=60.0):
            return _parse_informe_csv(INFORME_CSV)
        monkeypatch.setattr(cvm, "fetch_informe_diario", fake_fetch)

        class FakeMapper:
            def mapping(self):
                return {"00000000000100": "Ações", "11111111000111": "Ações"}

        out = CvmFlowSource("2024-04", "2024-04", FakeMapper()).load()
        assert list(out.columns) == FLOW_COLUMNS
        assert len(out) == 1
        row = out.iloc[0]
        assert row["category"] == "Ações"
        # captação = 100+50+200 = 350 ; resgate = 30.5+20+80 = 130.5
        assert row["redemption_gross_brl"] == pytest.approx(130.5)
        assert row["net_flow_brl"] == pytest.approx(350 - 130.5)

    def test_unmapped_cnpj_dropped(self, monkeypatch):
        monkeypatch.setattr(cvm, "fetch_informe_diario",
                            lambda ym, timeout=60.0: _parse_informe_csv(INFORME_CSV))

        class OnlyOne:
            def mapping(self):
                return {"11111111000111": "Renda Fixa"}   # drops the other CNPJ

        out = CvmFlowSource("2024-04", "2024-04", OnlyOne()).load()
        assert len(out) == 1
        assert out.iloc[0]["redemption_gross_brl"] == pytest.approx(80.0)

    def test_scale_to_billions(self, monkeypatch):
        monkeypatch.setattr(cvm, "fetch_informe_diario",
                            lambda ym, timeout=60.0: _parse_informe_csv(INFORME_CSV))

        class M:
            def mapping(self):
                return {"00000000000100": "Ações", "11111111000111": "Ações"}

        out = CvmFlowSource("2024-04", "2024-04", M(), scale=1e9).load()
        assert out.iloc[0]["redemption_gross_brl"] == pytest.approx(130.5 / 1e9)

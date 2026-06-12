import pandas as pd
import pytest

from fund_flow.data.categories import (
    AnbimaCategoryMapper,
    CvmCadastroCategoryMapper,
    map_anbima_levels,
)


class TestMapAnbimaLevels:
    @pytest.mark.parametrize("levels,expected", [
        (("Ações",), "Ações"),
        (("Multimercado",), "Multimercado"),
        (("Cambial",), "Cambial"),
        (("Renda Fixa", "Duração Livre"), "Renda Fixa"),
        (("Renda Fixa", "Crédito Privado"), "Crédito Privado"),
        (("Renda Fixa", None, None, "Renda Fixa Crédito Privado"), "Crédito Privado"),
        (("FIDC",), None),
        (("FIP", "Multiestratégia"), None),
    ])
    def test_rules(self, levels, expected):
        assert map_anbima_levels(*levels) == expected

    def test_previdencia_wins_over_asset_class(self):
        # a Previdência-Ações fund must classify as Previdência, not Ações
        assert map_anbima_levels("Ações", None, None, "Previdência Ações") == "Previdência"
        assert map_anbima_levels("Renda Fixa", "Previdência") == "Previdência"


class FakeClient:
    """Returns canned paginated Fundos v2 responses."""
    def __init__(self, pages):
        self._pages = pages

    def get(self, path, params=None):
        page = params["page"] if params else 0
        content = self._pages[page] if page < len(self._pages) else []
        return {"content": content}


class TestAnbimaCategoryMapper:
    def test_builds_cnpj_to_category_across_pages(self):
        pages = [
            [{"tipo_identificador_fundo": "CNPJ", "identificador_fundo": "00.000.000/0001-00",
              "nivel_1_categoria": "Ações"},
             {"tipo_identificador_fundo": "CNPJ", "identificador_fundo": "11111111000111",
              "nivel_1_categoria": "Renda Fixa", "nivel_2_categoria": "Crédito Privado"}],
            [{"tipo_identificador_fundo": "CNPJ", "identificador_fundo": "22222222000122",
              "nivel_1_categoria": "Multimercado"}],
        ]
        mapper = AnbimaCategoryMapper(FakeClient(pages), page_size=2)
        m = mapper.mapping()
        assert m["00000000000100"] == "Ações"
        assert m["11111111000111"] == "Crédito Privado"
        assert m["22222222000122"] == "Multimercado"

    def test_skips_non_cnpj_and_unmapped(self):
        pages = [[
            {"tipo_identificador_fundo": "CPF", "identificador_fundo": "x",
             "nivel_1_categoria": "Ações"},
            {"tipo_identificador_fundo": "CNPJ", "identificador_fundo": "33333333000133",
             "nivel_1_categoria": "FIDC"},
        ]]
        m = AnbimaCategoryMapper(FakeClient(pages), page_size=2).mapping()
        assert m == {}


class TestCvmCadastroCategoryMapper:
    def test_maps_known_classes(self):
        df = pd.DataFrame({
            "cnpj": ["00000000000100", "11111111000111", "22222222000122"],
            "classe": ["Fundo de Ações", "Fundo de Renda Fixa", "Fundo Imobiliário"],
        })
        m = CvmCadastroCategoryMapper(df).mapping()
        assert m["00000000000100"] == "Ações"
        assert m["11111111000111"] == "Renda Fixa"
        assert "22222222000122" not in m        # FII is out of scope

import pandas as pd
import pytest

from fund_flow.data.categories import (
    AnbimaCategoryMapper,
    CvmCadastroCategoryMapper,
    CvmRegistroClasseCategoryMapper,
    map_anbima_classificacao,
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
        # CLASSE strings are the real values from CVM cad_fi.csv (Instrução 555).
        df = pd.DataFrame({
            "cnpj": ["00000000000100", "11111111000111", "22222222000122",
                     "33333333000133", "44444444000144"],
            "classe": ["Ações", "Renda Fixa", "FII",
                       "Referenciado", "FIDC"],
        })
        m = CvmCadastroCategoryMapper(df).mapping()
        assert m["00000000000100"] == "Ações"
        assert m["11111111000111"] == "Renda Fixa"
        assert m["33333333000133"] == "Renda Fixa"   # Referenciado is an RF subtype
        assert "22222222000122" not in m             # FII is out of scope
        assert "44444444000144" not in m             # FIDC is out of scope


class TestMapAnbimaClassificacao:
    # Real Classificacao_Anbima strings from CVM registro_classe.csv.
    @pytest.mark.parametrize("s,expected", [
        ("Ações Livre", "Ações"),
        ("Ações Invest. no Exterior", "Ações"),
        ("Multimercados Livre", "Multimercado"),
        ("Multimercados Macro", "Multimercado"),
        ("Cambial", "Cambial"),
        # Renda Fixa: credit axis decides RF vs Crédito Privado.
        ("Renda Fixa Duração Livre Soberano", "Renda Fixa"),
        ("Renda Fixa Indexados", "Renda Fixa"),
        ("Renda Fixa Simples", "Renda Fixa"),
        ("Renda Fixa Dívida Externa", "Renda Fixa"),
        ("Renda Fixa Duração Livre Crédito Livre", "Crédito Privado"),
        ("Renda Fixa Duração Baixa Grau de Invest.", "Crédito Privado"),
        # Previdência wins over the underlying asset class.
        ("Previdência RF Duração Livre Crédito Liv", "Previdência"),
        ("Previdência Multimercado Livre", "Previdência"),
        ("Previdência Ações Ativo", "Previdência"),
        # Out of scope / missing. Closed-end ("Fechados …") funds have no daily
        # subscriptions/redemptions → outside the open-ended flow universe.
        (None, None),
        ("", None),
        ("Fechados de Ações", None),
    ])
    def test_maps_classificacao(self, s, expected):
        assert map_anbima_classificacao(s) == expected


class TestCvmRegistroClasseCategoryMapper:
    def test_maps_all_six_categories(self):
        df = pd.DataFrame({
            "cnpj": [f"{i:014d}" for i in range(6)],
            "anbima": [
                "Ações Livre",
                "Multimercados Livre",
                "Renda Fixa Duração Livre Soberano",
                "Renda Fixa Duração Livre Crédito Livre",
                "Previdência Multimercado Livre",
                "Cambial",
            ],
        })
        m = CvmRegistroClasseCategoryMapper(df).mapping()
        assert m["00000000000000"] == "Ações"
        assert m["00000000000001"] == "Multimercado"
        assert m["00000000000002"] == "Renda Fixa"
        assert m["00000000000003"] == "Crédito Privado"
        assert m["00000000000004"] == "Previdência"
        assert m["00000000000005"] == "Cambial"

    def test_drops_unmappable_and_missing(self):
        df = pd.DataFrame({
            "cnpj": ["00000000000100", "00000000000200"],
            "anbima": [None, "Fundo Estruturado Qualquer"],
        })
        m = CvmRegistroClasseCategoryMapper(df).mapping()
        assert m == {}

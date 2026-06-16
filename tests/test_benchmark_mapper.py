"""Unit tests for the pure benchmark mapper (Phase 2.6, Step 1)."""
import pytest

from fund_flow.data.benchmark_mapper import (
    CANONICAL_IDS,
    canonical_benchmark,
    canonical_benchmark_traced,
)


class TestDeclaredField:
    """Tier 1 — the declared Indicador_Desempenho string alone."""

    @pytest.mark.parametrize("raw, expected", [
        ("100% CDI", "CDI"),
        ("DI+", "CDI"),
        ("DI de um dia", "CDI"),          # the real CVM value (10,720 classes)
        ("Taxa Anbid", "CDI"),
        ("Taxa Selic", "SELIC"),
        ("IBOVESPA", "IBOV"),
        ("Ibovespa", "IBOV"),
        ("IBrX", "IBRX100"),
        ("IBrX-50", "IBRX100"),
        ("S&P 500 (USD)", "SP500_BRL"),
        ("MSCI World", "MSCI_WORLD_BRL"),
        ("IHFA", "IHFA"),
        ("IMA-B 5+", "IMA-B5"),                                   # spec case
        ("Índice de Mercado Andima NTN-B até 5 anos", "IMA-B5"),  # real value
        ("Índice de Mercado Andima todas NTN-B", "IMA-B"),        # real value
        ("Índice de Mercado Andima NTN-B mais de 5 anos", "IMA-B"),
        ("IDA-IPCA", "IDA-IPCA"),
        ("Índice de Preços ao Consumidor Amplo (IPCA/IBGE)", "IPCA"),
        ("IPCA + 3%", "IPCA"),                                    # spec case
    ])
    def test_declared_maps(self, raw, expected):
        assert canonical_benchmark(raw) == expected
        assert canonical_benchmark_traced(raw)[1] == "declared"

    def test_imab5_precedes_imab(self):
        # the '5 anos' specific rule must win over the generic NTN-B rule
        assert canonical_benchmark("IMA-B 5") == "IMA-B5"
        assert canonical_benchmark("IMA-B") == "IMA-B"

    def test_output_always_canonical(self):
        for raw in ["100% CDI", "Ibovespa", "lorem ipsum", "", None]:
            assert canonical_benchmark(raw) in CANONICAL_IDS


class TestBlankAndUnknown:
    @pytest.mark.parametrize("raw", ["", None, "Não se aplica", "OUTROS"])
    def test_blank_or_na_without_fallback_is_zero(self, raw):
        # no name, no class → nothing to fall back to → ZERO
        assert canonical_benchmark(raw) == "ZERO"
        assert canonical_benchmark_traced(raw)[1] == "none"

    def test_genuinely_unknown_string_is_zero(self):
        assert canonical_benchmark("Cota de PIBB") == "ZERO"
        assert canonical_benchmark("Dólar comercial") == "ZERO"
        assert canonical_benchmark("Índice Geral de Preços-Mercado (IGP-M)") == "ZERO"


class TestNameFallback:
    """Tier 2 — declared blank/NA, recover from the fund name."""

    def test_name_recovers_ibov(self):
        cid, tier = canonical_benchmark_traced(
            "Não se aplica", fund_name="ITAÚ AÇÕES IBOVESPA ATIVO FIA")
        assert cid == "IBOV" and tier == "name"

    def test_name_recovers_sp500(self):
        cid, tier = canonical_benchmark_traced(
            None, fund_name="XP S&P 500 FIM")
        assert cid == "SP500_BRL" and tier == "name"

    def test_name_recovers_imab(self):
        cid, tier = canonical_benchmark_traced(
            "OUTROS", fund_name="FUNDO IMA-B TÍTULOS PÚBLICOS FI RF")
        assert cid == "IMA-B" and tier == "name"


class TestAnbimaDefault:
    """Tier 3 — declared + name yield nothing → ANBIMA-class default."""

    @pytest.mark.parametrize("anbima, expected", [
        # Ações: only index-tracking / exterior sub-types get a default; active
        # equity with no declared benchmark is EXCLUDED (tightened, not assumed IBOV).
        ("Ações Indexados", "IBOV"),
        ("Ações Invest. no Exterior", "SP500_BRL"),
        ("Ações Livre", "ZERO"),          # active, undeclared → excluded
        ("Ações Ativo", "ZERO"),
        # Multimercado / Renda Fixa / Crédito → CDI is industry convention (low risk).
        ("Multimercados Livre", "CDI"),
        ("Renda Fixa Duração Livre Crédito Livre", "CDI"),
        ("Renda Fixa Duração Baixa Soberano", "CDI"),
        ("Previdência Multimercado Livre", "CDI"),
        ("Previdência Ações", "IBOV"),
        ("Cambial", "ZERO"),
    ])
    def test_anbima_default(self, anbima, expected):
        cid, tier = canonical_benchmark_traced(
            "Não se aplica", fund_name="NOME SEM PISTA", anbima_class=anbima)
        assert cid == expected
        assert tier == ("anbima" if expected != "ZERO" else "none")

    def test_declared_beats_anbima_default(self):
        # an explicit declared benchmark must override the class default
        cid, tier = canonical_benchmark_traced(
            "Ibovespa", fund_name="X", anbima_class="Renda Fixa Duração Livre")
        assert cid == "IBOV" and tier == "declared"

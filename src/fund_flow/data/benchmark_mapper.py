"""Map a fund's declared benchmark to a canonical benchmark ID (Phase 2.6).

The CVM RCVM-175 class registry carries a declared performance benchmark in
`Indicador_Desempenho` (e.g. 'DI de um dia', 'Ibovespa', 'Índice de Mercado
Andima todas NTN-B'). It is clean where present, but ~62% of classes are
blank / 'Não se aplica' / 'OUTROS', so we fall back — in order — to the fund
NAME (`Denominacao_Social`, 100% populated) and then to a sensible default per
ANBIMA category. Each tier is recorded so the coverage audit can show how much
of the panel rests on the assumption-heavy ANBIMA-class default.

`canonical_benchmark` is a PURE function (string → ID); no I/O. Returns one of:
  CDI, SELIC, IPCA, IBOV, IBRX100, IMA-B, IMA-B5, IDA-IPCA, SP500_BRL,
  MSCI_WORLD_BRL, IHFA, ZERO
ZERO means "no usable benchmark" — those fund-months are excluded from the
excess-return aggregation, never given a fabricated benchmark.
"""
from __future__ import annotations

import re
import unicodedata

from fund_flow.data.categories import map_anbima_classificacao

CANONICAL_IDS = (
    "CDI", "SELIC", "IPCA", "IBOV", "IBRX100", "IMA-B", "IMA-B5",
    "IDA-IPCA", "SP500_BRL", "MSCI_WORLD_BRL", "IHFA", "ZERO",
)

# Declared values that explicitly mean "no benchmark" → straight to fallback.
_NA_TOKENS = ("nao se aplica", "outros", "n a", "")


def _norm(s: str | None) -> str:
    """Lower, strip accents, reduce to alnum-and-spaces, collapse whitespace.

    'Índice de Mercado Andima NTN-B até 5 anos' → 'indice de mercado andima ntn
    b ate 5 anos'; 'S&P 500 (USD)' → 's p 500 usd'; 'IMA-B 5+' → 'ima b 5'.
    """
    if s is None:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(s))
    ascii_ = nfkd.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", ascii_)).strip()


# Ordered keyword rules (FIRST match wins). Each entry: (canonical_id, predicate).
# Order is load-bearing: the more specific token must precede the general one
# (IMA-B5 before IMA-B; SP500/MSCI before nothing else collides).
def _scan(blob: str) -> str | None:
    has = lambda *ks: any(k in blob for k in ks)  # noqa: E731
    rules: list[tuple[str, bool]] = [
        ("IMA-B5", has("ntn b ate 5", "ntn b 5", "ima b 5", "imab 5", "ima b5")),
        ("IMA-B",  has("ntn b", "ima b", "imab", "andima geral", "ima geral")),
        ("IDA-IPCA", bool(re.search(r"\bida\b", blob))),
        ("IBRX100", has("ibrx", "ibx", "ibr x")),
        ("IBOV",   has("ibovespa", "ibov")),
        ("SP500_BRL", has("s p 500", "sp 500", "sp500", "standard poor", "s e p 500")),
        ("MSCI_WORLD_BRL", has("msci")),
        ("IHFA",   has("ihfa")),
        ("SELIC",  has("selic")),
        # 'DI' (CDI / DI+ / 'DI de um dia') as a standalone token. \bdi\b is safe:
        # it does not fire inside 'dividendos', 'credito', 'predial', etc.
        ("CDI",    has("cdi", "anbid", "deposito interfinanceiro")
                   or bool(re.search(r"\bdi\b", blob))),
        ("IPCA",   has("ipca")),
    ]
    for cid, hit in rules:
        if hit:
            return cid
    return None


# ANBIMA-class default benchmark (tier 3) — used only when the declared field and
# the fund name yield nothing.
#
# CDI is a near-universal convention for Multimercado / Renda Fixa / Crédito
# Privado (the cash / opportunity-cost benchmark), so defaulting them to CDI is
# low-assumption. EQUITY benchmark choice genuinely varies, so for Ações we do NOT
# fabricate one: only ANBIMA sub-types that imply a specific index get a default
# (Indexados → IBOV; '... no Exterior' → SP500-in-BRL). Active equity with no
# declared / name benchmark → ZERO (excluded), to avoid IBOV-excess noise. Cambial
# has no clean canonical series → ZERO.
def _anbima_default(anbima_class: str | None) -> tuple[str, str]:
    blob = _norm(anbima_class)
    cat = map_anbima_classificacao(anbima_class)
    exterior = "exterior" in blob
    indexed = "indexad" in blob or "indice" in blob
    if cat == "Ações":
        if exterior:
            return "SP500_BRL", "anbima"
        if indexed:
            return "IBOV", "anbima"
        return "ZERO", "none"  # active equity, undeclared benchmark → excluded
    if cat in ("Multimercado", "Renda Fixa", "Crédito Privado"):
        return "CDI", "anbima"
    if cat == "Previdência":
        if "acoes" in blob:                  # equity-sleeve previdência
            return "IBOV", "anbima"
        return "CDI", "anbima"               # Previdência RF/Multi → CDI
    return "ZERO", "anbima"                   # Cambial and anything unmapped


def canonical_benchmark_traced(
    raw_benchmark: str | None,
    fund_name: str | None = None,
    anbima_class: str | None = None,
) -> tuple[str, str]:
    """Return (canonical_id, source_tier) where tier ∈ {declared, name, anbima, none}.

    Tiers are tried in order: the declared `Indicador_Desempenho`, then the fund
    name, then the ANBIMA-class default. 'none' means even the default declined
    (→ ZERO).
    """
    decl = _norm(raw_benchmark)
    if decl and decl not in _NA_TOKENS:
        hit = _scan(decl)
        if hit:
            return hit, "declared"
    hit = _scan(_norm(fund_name))
    if hit:
        return hit, "name"
    cid, tier = _anbima_default(anbima_class)
    return (cid, tier if cid != "ZERO" else "none")


def canonical_benchmark(
    raw_benchmark: str | None,
    fund_name: str | None = None,
    anbima_class: str | None = None,
) -> str:
    """Pure string→canonical-benchmark-ID map (see module docstring)."""
    return canonical_benchmark_traced(raw_benchmark, fund_name, anbima_class)[0]

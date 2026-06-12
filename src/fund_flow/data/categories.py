"""Map a fund (by CNPJ) to one of the 6 Anbima categories we forecast.

Two mappers behind a common interface:
  - AnbimaCategoryMapper  — official ANBIMA taxonomy via Fundos v2 API
    (highest fidelity; needs production API access — see project notes).
  - CvmCadastroCategoryMapper — CVM cad_fi.csv CLASSE field (free, no auth;
    covers Ações/Renda Fixa/Multimercado/Cambial well, but cannot by itself
    distinguish Crédito Privado or Previdência).

Both expose `mapping() -> dict[cnpj_digits -> category]`.
"""
from __future__ import annotations

from typing import Protocol

from fund_flow.data.cvm import normalize_cnpj
from fund_flow.schema import VALID_CATEGORIES


class CategoryMapper(Protocol):
    def mapping(self) -> dict[str, str]:
        """Return {normalized_cnpj: anbima_category}."""
        ...


# --- ANBIMA taxonomy → our 6 categories ----------------------------------

def map_anbima_levels(
    nivel_1: str | None,
    nivel_2: str | None = None,
    nivel_3: str | None = None,
    tipo_anbima: str | None = None,
) -> str | None:
    """Collapse ANBIMA's classification levels into one of the 6 categories.

    Rules (keyword-based so they tolerate the exact ANBIMA strings, which must
    be confirmed against live data once production access is enabled):
      - any level mentioning 'Previdência'      → Previdência
      - Renda Fixa + 'Crédito Privado' mention   → Crédito Privado
      - Nível 1 in {Ações, Multimercado, Cambial, Renda Fixa} → itself
      - otherwise None (dropped as out-of-scope)
    """
    blob = " ".join(x for x in (nivel_1, nivel_2, nivel_3, tipo_anbima) if x).lower()
    n1 = (nivel_1 or "").strip().lower()

    if "previd" in blob:
        return "Previdência"
    if n1.startswith("ações") or n1.startswith("acoes"):
        return "Ações"
    if n1.startswith("multimercado"):
        return "Multimercado"
    if n1.startswith("cambial"):
        return "Cambial"
    if n1.startswith("renda fixa"):
        return "Crédito Privado" if "crédito privado" in blob or "credito privado" in blob \
            else "Renda Fixa"
    return None


class AnbimaCategoryMapper:
    """Build CNPJ→category from the ANBIMA Fundos v2 funds feed (paginated)."""

    def __init__(self, client, page_size: int = 1000, max_pages: int = 1000):
        self._client = client
        self._page_size = page_size
        self._max_pages = max_pages

    def mapping(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for page in range(self._max_pages):
            payload = self._client.get(
                "feed/fundos/v2/fundos",
                params={"page": page, "size": self._page_size},
            )
            funds = payload.get("content", payload) if isinstance(payload, dict) else payload
            if not funds:
                break
            for f in funds:
                if (f.get("tipo_identificador_fundo") or "").upper() not in ("", "CNPJ"):
                    continue
                cat = map_anbima_levels(
                    f.get("nivel_1_categoria"), f.get("nivel_2_categoria"),
                    f.get("nivel_3_subcategoria"), f.get("tipo_anbima"),
                )
                if cat in VALID_CATEGORIES:
                    result[normalize_cnpj(f.get("identificador_fundo", ""))] = cat
            if len(funds) < self._page_size:
                break
        return result


# --- CVM cadastro CLASSE → our 6 categories (free fallback) ---------------

_CVM_CLASSE_MAP = {
    "Fundo de Ações": "Ações",
    "Fundo de Renda Fixa": "Renda Fixa",
    "Fundo Multimercado": "Multimercado",
    "Fundo Cambial": "Cambial",
}


class CvmCadastroCategoryMapper:
    """Map via CVM cad_fi.csv CLASSE. No auth; coarser than ANBIMA.

    Cannot isolate Crédito Privado (an ANBIMA sub-class within Renda Fixa) or
    Previdência from CLASSE alone, so those funds fall under their CVM CLASSE
    (mostly Renda Fixa). Use AnbimaCategoryMapper for full 6-way fidelity.
    """

    def __init__(self, cadastro_df, classe_map: dict[str, str] | None = None):
        self._df = cadastro_df
        self._map = classe_map or _CVM_CLASSE_MAP

    def mapping(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for cnpj, classe in zip(self._df["cnpj"], self._df["classe"]):
            cat = self._map.get(str(classe).strip())
            if cat in VALID_CATEGORIES:
                out[normalize_cnpj(cnpj)] = cat
        return out

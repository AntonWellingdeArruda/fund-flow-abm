"""Real Brazilian fund flows from CVM's open-data portal (free, no auth).

Informe Diário — per-fund, per-day inflows/outflows/NAV:
  https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_YYYYMM.zip
  cols: CNPJ_FUNDO_CLASSE, DT_COMPTC, CAPTC_DIA, RESG_DIA, VL_PATRIM_LIQ, ...
        (older files use CNPJ_FUNDO / TP_FUNDO)

Cadastro — fund registry (used by the CVM-native category fallback):
  https://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv
  cols: CNPJ_FUNDO, DENOM_SOCIAL, CLASSE, ...

CSVs are ';'-separated, decimal ',', latin-1. The portal 403s requests without
a browser User-Agent. HTTP and parsing are split so parsing is testable offline.
NOTE: CVM blocks some datacenter IPs — run live fetches from an allowed network.
"""
from __future__ import annotations

import io
import re
import time
import urllib.error
import urllib.request
import zipfile

import pandas as pd

INFORME_URL = (
    "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_{ym}.zip"
)
# Older months live in yearly HIST archives (one zip/year, 12 monthly CSVs
# inside, CNPJ_FUNDO + period-decimals). The monthly endpoint only keeps the
# trailing ~5 years. LAST_HIST_YEAR is the last year served as a yearly archive.
HIST_URL = (
    "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/HIST/inf_diario_fi_{year}.zip"
)
LAST_HIST_YEAR = 2020
CADASTRO_URL = "https://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv"
# RCVM 175 class-level registry. registro_classe.csv keys on CNPJ_Classe (which
# matches the informe's CNPJ_FUNDO_CLASSE) and carries the full ANBIMA taxonomy
# in Classificacao_Anbima — the same data the blocked ANBIMA Fundos API serves.
REGISTRO_URL = "https://dados.cvm.gov.br/dados/FI/CAD/DADOS/registro_fundo_classe.zip"

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124 Safari/537.36"

# Canonical column aliases (informe schema changed with RCVM 175).
_CNPJ_ALIASES = ("CNPJ_FUNDO_CLASSE", "CNPJ_FUNDO")
_INFORME_RENAME = {
    "DT_COMPTC": "date",
    "CAPTC_DIA": "captacao",
    "RESG_DIA": "resgate",
    "VL_PATRIM_LIQ": "pl",
}


class CvmFetchError(RuntimeError):
    """Raised when a CVM download fails (network, HTTP, or bad archive)."""


def normalize_cnpj(value: str) -> str:
    """Strip CNPJ formatting → 14 digit string ('00.000.000/0001-00' → digits)."""
    return re.sub(r"\D", "", str(value))


def _http_get_bytes(url: str, timeout: float, retries: int = 3) -> bytes:
    """GET bytes from CVM, retrying transient failures.

    The large HIST yearly archives intermittently time out mid-download (seen on
    inf_diario_fi_2008.zip); a single failure should not silently drop a whole
    year, so we back off (5s, 10s) and retry before giving up.
    """
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
    raise CvmFetchError(f"CVM request failed after {retries} tries: {url}\n{last}") from last


def _parse_informe_csv(text: str) -> pd.DataFrame:
    """Parse one informe-diário CSV into [cnpj, date, captacao, resgate, pl, quota].

    `quota` (VL_QUOTA, the daily share/quota value) is retained so Phase 2.6 can
    compute fund-level monthly returns; the flows pipeline ignores it. VL_QUOTA is
    present in both the RCVM-175 and the older schema (see test fixtures), but some
    rows lack it, so it is parsed leniently (NaN when absent) and rows are still
    only dropped for missing flow values.
    """
    # All columns as str to preserve CNPJ leading zeros; numerics use decimal
    # comma, so convert ',' → '.' before to_numeric.
    df = pd.read_csv(io.StringIO(text), sep=";", dtype=str)
    cnpj_col = next((c for c in _CNPJ_ALIASES if c in df.columns), None)
    if cnpj_col is None:
        raise CvmFetchError(f"informe CSV missing CNPJ column; has {list(df.columns)[:6]}")

    out = pd.DataFrame({"cnpj": df[cnpj_col].map(normalize_cnpj)})
    out["date"] = df["DT_COMPTC"]
    for src, dst in (("CAPTC_DIA", "captacao"), ("RESG_DIA", "resgate"),
                     ("VL_PATRIM_LIQ", "pl"), ("VL_QUOTA", "quota")):
        col = df[src] if src in df.columns else pd.Series([None] * len(df))
        out[dst] = pd.to_numeric(
            col.str.replace(",", ".", regex=False), errors="coerce"
        )
    return out.dropna(subset=["captacao", "resgate"])


def fetch_informe_diario(year_month: str, timeout: float = 60.0) -> pd.DataFrame:
    """Download + parse one month's informe diário (year_month = 'YYYYMM').

    Reads every CSV member in the zip (RCVM 175 may split by class) and concats.
    """
    raw = _http_get_bytes(INFORME_URL.format(ym=year_month), timeout)
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise CvmFetchError(f"CVM returned a non-zip payload for {year_month}") from exc

    frames = [
        _parse_informe_csv(zf.read(name).decode("latin-1"))
        for name in zf.namelist() if name.lower().endswith(".csv")
    ]
    if not frames:
        raise CvmFetchError(f"no CSV members in informe zip for {year_month}")
    return pd.concat(frames, ignore_index=True)


def fetch_informe_year(year: int, timeout: float = 180.0) -> dict[str, pd.DataFrame]:
    """Download one yearly HIST archive → {YYYYMM: DataFrame} for its 12 months.

    Each member is a distinct month (inf_diario_fi_YYYYMM.csv), so unlike the
    monthly zip (which may split one month across class members) we key by month
    rather than concatenating. Same column schema → same `_parse_informe_csv`.
    """
    raw = _http_get_bytes(HIST_URL.format(year=year), timeout)
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise CvmFetchError(f"CVM returned a non-zip payload for HIST {year}") from exc

    out: dict[str, pd.DataFrame] = {}

    def _add(ym: str, df: "pd.DataFrame") -> None:
        out[ym] = pd.concat([out[ym], df], ignore_index=True) if ym in out else df

    for name in zf.namelist():
        if not name.lower().endswith(".csv"):
            continue
        df = _parse_informe_csv(zf.read(name).decode("latin-1"))
        m = re.search(r"_(\d{6})\.csv$", name)
        if m:                                   # 2005+: one CSV per month
            _add(m.group(1), df)
        else:                                   # 2000-2004: whole year in one CSV;
            # split rows by month using DT_COMPTC (kept as 'date', 'YYYY-MM-DD').
            ym_col = df["date"].str.replace("-", "", regex=False).str.slice(0, 6)
            for ym, grp in df.groupby(ym_col):
                _add(str(ym), grp.drop(columns=[]))
    if not out:
        raise CvmFetchError(f"no CSV members in HIST archive for {year}")
    return out


def _parse_cadastro_csv(text: str) -> pd.DataFrame:
    """Parse cad_fi.csv into [cnpj, classe]."""
    df = pd.read_csv(io.StringIO(text), sep=";", dtype=str, encoding=None)
    cnpj_col = next((c for c in _CNPJ_ALIASES if c in df.columns), None)
    classe_col = "CLASSE" if "CLASSE" in df.columns else None
    if cnpj_col is None or classe_col is None:
        raise CvmFetchError(f"cadastro missing CNPJ/CLASSE; has {list(df.columns)[:8]}")
    return pd.DataFrame({
        "cnpj": df[cnpj_col].map(normalize_cnpj),
        "classe": df[classe_col],
    }).dropna(subset=["classe"])


def fetch_cadastro(timeout: float = 60.0) -> pd.DataFrame:
    """Download + parse the CVM fund registry (cad_fi.csv) → [cnpj, classe]."""
    raw = _http_get_bytes(CADASTRO_URL, timeout)
    return _parse_cadastro_csv(raw.decode("latin-1"))


def _parse_registro_classe_csv(text: str) -> pd.DataFrame:
    """Parse registro_classe.csv into [cnpj, anbima, classificacao, situacao,
    indicador, nome].

    `indicador` is Indicador_Desempenho — the fund's declared performance
    benchmark (e.g. 'DI de um dia', 'Ibovespa', 'Índice de Mercado Andima todas
    NTN-B'); ~62% are blank/'Não se aplica'/'OUTROS', so `nome' (Denominacao_Social,
    100% populated) backs the Phase-2.6 benchmark mapper's name fallback.
    """
    df = pd.read_csv(io.StringIO(text), sep=";", dtype=str)
    if "CNPJ_Classe" not in df.columns:
        raise CvmFetchError(
            f"registro_classe missing CNPJ_Classe; has {list(df.columns)[:8]}"
        )
    return pd.DataFrame({
        "cnpj": df["CNPJ_Classe"].map(normalize_cnpj),
        "anbima": df.get("Classificacao_Anbima"),
        "classificacao": df.get("Classificacao"),
        "situacao": df.get("Situacao"),
        "indicador": df.get("Indicador_Desempenho"),
        "nome": df.get("Denominacao_Social"),
    })


def fetch_registro_classe(timeout: float = 120.0) -> pd.DataFrame:
    """Download + parse the RCVM 175 class registry → [cnpj, anbima, ...].

    `cnpj` is the share-class CNPJ (CNPJ_Classe), which joins directly to the
    informe diário's CNPJ_FUNDO_CLASSE. `anbima` is the ANBIMA classification.
    """
    raw = _http_get_bytes(REGISTRO_URL, timeout)
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise CvmFetchError("CVM returned a non-zip payload for registro") from exc
    member = next(
        (n for n in zf.namelist() if n.lower() == "registro_classe.csv"), None
    )
    if member is None:
        raise CvmFetchError(f"registro_classe.csv not in zip; has {zf.namelist()}")
    return _parse_registro_classe_csv(zf.read(member).decode("latin-1"))

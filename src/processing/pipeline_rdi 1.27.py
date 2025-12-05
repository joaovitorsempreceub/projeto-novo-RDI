#!/usr/bin/env python3
"""
pipeline_rdi.py
Pipeline unificado: limpa State of Data (2021-2024), coleta IBGE (Sidra) com retry,
faz integração por UF (+ sexo/faixa etária quando disponível) e salva CSV/Parquet.

Coloque este arquivo em src/processing/ e execute:
& ".\.venv\Scripts\python.exe" "src\processing\pipeline_rdi.py"
ou
python -m src.processing.pipeline_rdi
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Optional, Tuple
import time

import pandas as pd
import numpy as np
import requests
from slugify import slugify
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from rapidfuzz import fuzz
from rich import print as rprint
from rich.logging import RichHandler

# Optional: sidrapy (wrapper). We'll use where available but fallback to requests.
try:
    import sidrapy as sidra
except Exception:
    sidra = None

# ===== CONFIG =====
ROOT = Path.cwd()
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SOD_INPUTS = [
    RAW_DIR / "sod_2021.csv",
    RAW_DIR / "sod_2022.csv",
    RAW_DIR / "sod_2023.csv",
    RAW_DIR / "sod_2024.csv",
]
SOD_CLEAN_CSV = PROCESSED_DIR / "sod_2021_2024_clean.csv"
SOD_CLEAN_PARQUET = PROCESSED_DIR / "sod_2021_2024_clean.parquet"

LOG_FMT = "%(asctime)s | %(levelname)-8s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, handlers=[RichHandler()])
logger = logging.getLogger("pipeline_rdi")
logger.setLevel(logging.INFO)

# ===== Utility helpers =====

def detectar_separador(path: Path, n_lines: int = 5) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        amostra = "".join([f.readline() for _ in range(n_lines)])
    candidatos = [",", ";", "|", "\t"]
    return max(candidatos, key=lambda sep: amostra.count(sep))

def limpar_nome(col: str) -> str:
    col = str(col).lower()
    col = re.sub(r"[^\w]+", "_", col)
    col = col.strip("_")
    return col

def normaliza_texto_valor(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    s = (s.replace("ã","a").replace("á","a").replace("â","a").replace("à","a")
         .replace("é","e").replace("ê","e")
         .replace("í","i")
         .replace("ó","o").replace("ô","o").replace("õ","o")
         .replace("ú","u").replace("ç","c"))
    return s

def slugify_safe(x: str) -> str:
    if pd.isna(x):
        return ""
    return slugify(str(x), lowercase=True, separator="_", remove=r"[^\w\s-]")

# ===== SOD cleaning (combines your snippets with minor improvements) =====

def carregar_e_limpar_sod(paths: list[SPath := Path]) -> pd.DataFrame:
    """
    Lê os CSVs do State of Data, concatena e aplica limpeza/normalização.
    Se já existir arquivo limpo em processed, carrega e retorna (evita reprocessar).
    """
    if SOD_CLEAN_CSV.exists() and SOD_CLEAN_PARQUET.exists():
        logger.info("Versão limpa já existe. Carregando de processed (não refazendo limpeza).")
        # carregar parquet por performance
        df = pd.read_parquet(SOD_CLEAN_PARQUET)
        return df

    dfs = []
    for p in paths:
        if not p.exists():
            logger.warning(f"Arquivo não encontrado: {p} — pulando.")
            continue
        sep = detectar_separador(p)
        logger.info(f"📄 Lendo: {p}")
        try:
            # leitura robusta; deixe pandas inferir dtypes mas sem low_memory warnings
            df = pd.read_csv(p, sep=sep, encoding="utf-8", low_memory=False)
        except Exception as e:
            logger.warning(f"Falha ao ler {p} com sep='{sep}': {e}. Tentando delim whitespace.")
            df = pd.read_csv(p, sep=None, engine="python", encoding="utf-8")
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("Nenhum CSV SOD encontrado nos caminhos fornecidos.")

    df_raw = pd.concat(dfs, axis=0, ignore_index=True)
    logger.info(f"Dimensão inicial concatenada: {df_raw.shape}")

    # remover colunas 100% vazias
    col_vazias = [c for c in df_raw.columns if df_raw[c].isna().all()]
    if col_vazias:
        logger.info(f"Colunas 100% vazias: {len(col_vazias)}. Removendo.")
        df_raw = df_raw.drop(columns=col_vazias)

    # normalizar nomes de colunas (mantém padrões pX_..)
    cols_original = list(df_raw.columns)
    # cria nomes limpos mas preserva os originais como mapping
    nome_map = {}
    for c in cols_original:
        cleaned = limpar_nome(c)
        # manter original se já existe
        if cleaned in nome_map.values():
            # garantir unicidade
            cleaned = f"{cleaned}_{len(nome_map)}"
        nome_map[c] = cleaned
    df_raw = df_raw.rename(columns=nome_map)

    # Agrupar colunas similares por SequenceMatcher heurística (rápido usando rapidfuzz)
    cols = list(df_raw.columns)
    visitadas = set()
    grupos = []
    for c in cols:
        if c in visitadas:
            continue
        grupo = [c]
        visitadas.add(c)
        for c2 in cols:
            if c2 in visitadas:
                continue
            score = fuzz.ratio(c, c2) / 100.0
            if score >= 0.60:
                grupo.append(c2)
                visitadas.add(c2)
        grupos.append(grupo)

    # montar df_clean com priorização dos grupos
    # Mas ao invés de inserir coluna-a-coluna (fragmenta o frame), vamos construir colunas numa lista e concat
    cols_result = {}
    # garantir coluna "ano" se existir em dataset
    if "ano" in df_raw.columns:
        cols_result["ano"] = df_raw["ano"].astype("Int64")
    for grupo in grupos:
        base = grupo[0]
        cols_pres = [c for c in grupo if c in df_raw.columns]
        if not cols_pres:
            continue
        if len(cols_pres) == 1:
            cols_result[base] = df_raw[cols_pres[0]]
        else:
            # prioriza left->right (bfill) mas usamos combine_first reduce
            combined = df_raw[cols_pres].bfill(axis=1).iloc[:, 0]
            cols_result[base] = combined

    df_clean = pd.concat(cols_result, axis=1)

    # aplicar limpeza textual nas colunas object
    obj_cols = df_clean.select_dtypes(include="object").columns.tolist()
    for col in obj_cols:
        df_clean[col] = df_clean[col].apply(normaliza_texto_valor)

    # conversão segura: detectar candidatas numéricas
    candidate_cols = []
    for c in df_clean.columns:
        if df_clean[c].dtype == "object":
            coerced = pd.to_numeric(df_clean[c], errors="coerce")
            frac_numeric = coerced.notna().mean()
            if frac_numeric >= 0.8 and df_clean[c].nunique() > 5:
                candidate_cols.append(c)
    for c in candidate_cols:
        df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce").astype("Int64")

    # remover colunas com >80% NA
    nulos_frac = df_clean.isna().mean()
    to_drop = nulos_frac[nulos_frac >= 0.80].index.tolist()
    if to_drop:
        logger.info(f"Remover por NA>={0.8}: {len(to_drop)} colunas")
        df_clean = df_clean.drop(columns=to_drop)

    # remover colunas vazias em algum ano (incompletas por ano) — seguir sua heurística
    if "ano" in df_clean.columns:
        anos = df_clean["ano"].dropna().unique().tolist()
        colunas_ruins = []
        for coluna in df_clean.columns:
            if coluna == "ano":
                continue
            bad = False
            for ano in anos:
                if df_clean[df_clean["ano"] == ano][coluna].dropna().shape[0] == 0:
                    bad = True
                    break
            if bad:
                colunas_ruins.append(coluna)
        if colunas_ruins:
            logger.info(f"Remover por vazia em algum ano: {len(colunas_ruins)} colunas")
            df_clean = df_clean.drop(columns=colunas_ruins)

    # padronizar nomes finais: remover prefixos p\d+_
    def limpar_nome_final(col):
        col = re.sub(r"^p\d+_[a-z]_", "", col)
        col = re.sub(r"^p\d+_", "", col)
        col = col.strip().lower()
        return col

    df_clean.columns = [limpar_nome_final(c) for c in df_clean.columns]

    # identificar colunas binarias 0/1 e trocar para Sim/Não
    binarias = []
    for col in df_clean.columns:
        vals = set(df_clean[col].dropna().unique())
        # checar representação de 0,1
        if vals and vals.issubset({0.0, 1.0, 0, 1}):
            binarias.append(col)
    if binarias:
        df_clean[binarias] = df_clean[binarias].replace({1.0: "sim", 0.0: "nao", 1: "sim", 0: "nao"})

    # reset index e id_global
    df_clean = df_clean.reset_index().rename(columns={"index": "id_global"})

    # salvar
    logger.info(f"Salvando CSV: {SOD_CLEAN_CSV}")
    df_clean.to_csv(SOD_CLEAN_CSV, index=False, encoding="utf-8")
    logger.info(f"Salvando Parquet: {SOD_CLEAN_PARQUET}")
    df_clean.to_parquet(SOD_CLEAN_PARQUET, index=False)

    return df_clean

# ===== IBGE fetching helpers =====

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10),
       retry=retry_if_exception_type(Exception))
def sidra_get_table(table: str, territorial_level: str = "all", year: Optional[int] = None) -> pd.DataFrame:
    """
    Tenta usar sidrapy.get_table; se sidrapy ausente, usa requests direto para a API Sidra.
    Retorna DataFrame.
    """
    base_url = "https://apisidra.ibge.gov.br/values/t/{t}/n1/{n1}/v/all/p/{p}"
    # observação: muitas tabelas do Sidra precisam de parâmetros corretos; o script fará tentativas.
    if sidra:
        try:
            logger.info("Usando sidrapy.get_table()")
            # sidrapy.get_table espera alguns args, mas interfaces variam por versão => uso fallback
            df = sidra.get_table(table)
            if isinstance(df, pd.DataFrame):
                return df
        except TypeError as te:
            logger.warning(f"sidrapy.get_table() falhou por TypeError: {te}")
        except Exception as e:
            logger.warning(f"sidrapy erro: {e}")

    # fallback para requests (montar URL simples: tabela t, territorial n1=all, p=2022)
    p = year or 2022
    url = f"https://apisidra.ibge.gov.br/values/t/{table}/n1/all/v/all/p/{p}"
    logger.info(f"Requisição direta Sidra: {url}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    payload = r.json()
    # payload é lista: primeira linha header
    headers = payload[0]
    rows = payload[1:]
    df = pd.DataFrame(rows, columns=headers)
    return df

def coletar_ibge_all_tables() -> dict:
    """
    Tenta coletar várias tabelas usuais: população por UF, por UF×sexo, por UF×sexo×faixa etária.
    Retorna dict com dataframes e salva CSVs parciales.
    """
    results = {}
    # Tabela exemplo de população: t=6579 (exemplo usado antes). Se falhar, o retry cuidará.
    # Você pode ajustar os códigos de tabela conforme necessidade.
    tabelas_tentativas = {
        "pop_uf": ["6579",  "6579"],  # placeholder (ex.: população) - verifique o código real
        # se quiser outras tabelas, adicione aqui com os códigos certos do Sidra
    }

    # Preferimos usar queries explícitas (se souber o código certo, coloque-o).
    # Vou tentar obter:
    # - população por UF (tabela 6579 ou 6578 dependendo do parâmetro)
    # - população por UF × sexo × faixa etária (se existir)
    try_tables = [
        ("6579", "pop_uf"),
        # adicione aqui quaisquer códigos que você sabe que quer
    ]

    for code, name in try_tables:
        try:
            df = sidra_get_table(code, year=2022)
            # salvar bruto
            out_csv = PROCESSED_DIR / f"ibge_raw_{name}_{code}.csv"
            df.to_csv(out_csv, index=False, encoding="utf-8")
            logger.info(f"✅ Dados IBGE coletados: gravado {out_csv}")
            results[name] = df
        except Exception as e:
            logger.warning(f"Erro coletando dados IBGE (t={code}): {e}")
            results[name] = None
    return results

# ===== Integration logic =====

def detectar_coluna_uf(df: pd.DataFrame) -> Optional[str]:
    """
    Busca colunas que pareçam conter UF (sigla ou nome): ex. 'uf', 'uf_onde_mora', 'estado', 'uf_onde_mora'
    Retorna o nome da coluna mapeada para 'uf_sigla' se encontrar, senão None.
    """
    candidatos = [c for c in df.columns if "uf" in c or "estado" in c or "onde_mora" in c or "uf_onde" in c]
    # preferir colunas que contenham 'sigla' ou 'uf'
    for c in candidatos:
        if "sig" in c or c.endswith("uf") or c == "uf":
            return c
    if candidatos:
        return candidatos[0]
    return None

def mapear_uf_sigla(df: pd.DataFrame, col_guess: Optional[str]) -> pd.DataFrame:
    """
    Garante a coluna 'uf_sigla' com siglas (ex: sp, rj, mg).
    Tenta extrair sigla de strings (quando for nome do estado, converte para sigla via heurística).
    """
    if col_guess is None:
        logger.warning("df_state não possui coluna 'uf' detectada automaticamente. Será criada coluna 'uf_sigla' vazia.")
        df["uf_sigla"] = pd.NA
        return df

    col = col_guess
    # normaliza
    vals = df[col].dropna().astype(str).unique()[:20]
    # se os valores já forem siglas de 2 letras
    sample = df[col].dropna().astype(str).head(200).tolist()
    sample_short = all(len(s.strip()) <= 3 and s.strip().isalpha() for s in sample)
    if sample_short:
        df["uf_sigla"] = df[col].str.upper().str.strip()
        return df

    # se tiver nomes completos do estado (ex: sao_paulo), mapeamos com heurística
    # lista de estados BR (sigla->nome)
    uf_map = {
        "AC": "acre","AL": "alagoas","AP": "amapa","AM": "amazonas","BA": "bahia","CE": "ceara",
        "DF": "distrito federal","ES": "espirito santo","GO": "goias","MA": "maranhao","MT": "mato grosso",
        "MS": "mato grosso do sul","MG": "minas gerais","PA": "para","PB": "paraiba","PR": "parana","PE": "pernambuco",
        "PI": "piaui","RJ": "rio de janeiro","RN": "rio grande do norte","RS": "rio grande do sul","RO": "rondonia",
        "RR": "roraima","SC": "santa catarina","SP": "sao paulo","SE": "sergipe","TO": "tocantins"
    }
    # invert map: nome->sigla (lower)
    nome2sig = {v: k for k, v in uf_map.items()}
    def to_sig(x):
        if pd.isna(x):
            return pd.NA
        s = str(x).strip().lower()
        s_n = normaliza_texto_valor(s)
        # exact match
        if s_n in nome2sig:
            return nome2sig[s_n]
        # shoot for substring match
        for nome, sig in nome2sig.items():
            if nome in s_n:
                return sig
        # maybe already sigla
        if len(s_n) <= 3 and s_n.isalpha():
            return s_n.upper()
        return pd.NA

    df["uf_sigla"] = df[col].apply(to_sig)
    missing = df["uf_sigla"].isna().mean()
    if missing > 0.3:
        logger.warning(f"Alta taxa de UFs não mapeadas ({missing:.1%}). Verifique a coluna {col}.")
    return df

def integrar_state_ibge(df_state: pd.DataFrame, ibge_results: dict) -> pd.DataFrame:
    """
    Integra SOD com os dados coletados do IBGE.
    Por padrão faz merge por 'uf_sigla'. Se IBGE retorna tabelas com sexo/faixa, tenta merges adicionais.
    """
    df = df_state.copy()
    guessed = detectar_coluna_uf(df)
    df = mapear_uf_sigla(df, guessed)

    # agrega SOD por UF (contagem e médias relevantes)
    # identificar colunas de interesse: genero / idade / faixa_idade / nivel_de_ensino / faixa_salarial
    genero_cols = [c for c in df.columns if "genero" in c]
    idade_cols = [c for c in df.columns if "idade" in c]
    faixa_cols = [c for c in df.columns if "faixa_idade" in c or "faixa" in c and "idade" in c]
    nivel_ensino_cols = [c for c in df.columns if "nivel_de_ensino" in c or "nivel" in c and "ensino" in c]
    faixa_salarial_cols = [c for c in df.columns if "faixa_salarial" in c or "faixa_salarial" in c]

    # preferências: usar a primeira encontrada
    genero_col = genero_cols[0] if genero_cols else None
    idade_col = idade_cols[0] if idade_cols else None
    faixa_col = faixa_cols[0] if faixa_cols else None
    nivel_col = nivel_ensino_cols[0] if nivel_ensino_cols else None
    salario_col = faixa_salarial_cols[0] if faixa_salarial_cols else None

    logger.info(f"Colunas detectadas - genero: {genero_col}, idade: {idade_col}, faixa: {faixa_col}, nivel: {nivel_col}, faixa_sal: {salario_col}")

    # make sure uf_sigla exists
    if "uf_sigla" not in df.columns:
        df["uf_sigla"] = pd.NA

    # agregações simples por UF
    agg = df.groupby("uf_sigla").agg(
        total_sod=("id_global", "count")
    ).reset_index()

    # se genero_col presente, agregar proporção por genero
    if genero_col:
        tmp = df.groupby(["uf_sigla", genero_col]).size().reset_index(name="count")
        tmp["genero_norm"] = tmp[genero_col].fillna("nao_informado").astype(str).apply(slugify_safe)
        # pivot para wide
        pivot = tmp.pivot(index="uf_sigla", columns="genero_norm", values="count").fillna(0).reset_index()
        agg = agg.merge(pivot, on="uf_sigla", how="left")

    # merge com IBGE populacao por UF, se existir
    ibge_pop = ibge_results.get("pop_uf")
    if ibge_pop is not None:
        # tentar detectar coluna do IBGE com sigla/UF
        # ibge payloads variam: procurar colunas 'UF', 'Territorial', 'Sigla', 'D1N' etc
        ibge_cols = [c.lower() for c in ibge_pop.columns]
        candidate = None
        for c in ibge_pop.columns:
            if "uf" in c.lower() or "territorial" in c.lower() or "sigla" in c.lower():
                candidate = c
                break
        if candidate is None:
            logger.warning("Não detectei coluna de UF no dataframe IBGE bruto — deixando sem merge por UF.")
            merged = agg
        else:
            # normalizar valores do IBGE para sigla
            ibge_pop = ibge_pop.rename(columns={candidate: "uf_raw"})
            ibge_pop["uf_sigla"] = ibge_pop["uf_raw"].apply(lambda x: slugify_safe(x).upper()[:2] if pd.notna(x) else pd.NA)
            # muito dependente do payload — tentar identificar a coluna de valores
            # valores geralmente em coluna com nome 'Valor' ou 'V' ou 'valor'
            val_col = None
            for c in ibge_pop.columns:
                if "valor" in c.lower() or c.lower() == "v" or c.lower().startswith("valor"):
                    val_col = c
                    break
            if val_col is None:
                # procurar a primeira coluna numérica
                numeric_cols = ibge_pop.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    val_col = numeric_cols[0]
            if val_col is None:
                logger.warning("Não achei coluna de valores numéricos no IBGE para 'pop_uf'. Salvando bruto e seguindo sem merge.")
                merged = agg
            else:
                # renomear e converter
                ibge_pop["ibge_pop"] = pd.to_numeric(ibge_pop[val_col], errors="coerce")
                ibge_small = ibge_pop[["uf_sigla", "ibge_pop"]].groupby("uf_sigla").sum().reset_index()
                merged = agg.merge(ibge_small, on="uf_sigla", how="left")
    else:
        merged = agg

    # marca fonte
    merged["fonte"] = "state_of_data"
    return merged

# ===== Main =====

def main():
    logger.info("🚀 Iniciando Pipeline RDI...")
    # 1) carregar + limpar SOD (ou carregar versão já limpa)
    df_state = carregar_e_limpar_sod(SOD_INPUTS)
    logger.info("✅ State of Data carregado e limpo.")

    # 2) coletar IBGE (várias tentativas)
    logger.info("🌐 Coletando dados IBGE (População por UF)...")
    ibge_results = coletar_ibge_all_tables()

    # 3) integrar
    try:
        merged = integrar_state_ibge(df_state, ibge_results)
    except Exception as e:
        logger.exception(f"Erro ao integrar SOD + IBGE: {e}")
        # salve só os agregados sod
        agg_only = df_state.groupby("uf_sigla").agg(total_sod=("id_global","count")).reset_index()
        out = PROCESSED_DIR / "integracao_state_ibge_nacional.csv"
        agg_only.to_csv(out, index=False, encoding="utf-8")
        logger.info(f"IBGE não disponível — gerando apenas agregados SOD nacionais. Arquivo salvo: {out}")
        return

    out_csv = PROCESSED_DIR / "integracao_state_ibge.csv"
    out_parquet = PROCESSED_DIR / "integracao_state_ibge.parquet"
    merged.to_csv(out_csv, index=False, encoding="utf-8")
    merged.to_parquet(out_parquet, index=False)
    logger.info(f"📁 Arquivo final salvo em: {out_csv}")

if __name__ == "__main__":
    main()

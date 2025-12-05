#!/usr/bin/env python3
"""
Pipeline de Engenharia de Dados - Projeto RDI (Versão Final Estável)
--------------------------------------------------------------------
Correções aplicadas:
1. Resolução do SettingWithCopyWarning usando .copy() após filtros.
2. Tratamento de fragmentação de DataFrame.
3. Conversão segura de tipos para compatibilidade com Parquet/PyArrow.

Codigo: João vitor silva sousa com colaboração de Marcello Amorim Romão
"""

import os
import re
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from rich.logging import RichHandler

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

ROOT = Path(r"C:\Users\PC\Desktop\WORKSPACE\projeto novo RDI")
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SOD_INPUTS = [
    RAW_DIR / "sod_2021.csv",
    RAW_DIR / "sod_2022.csv",
    RAW_DIR / "sod_2023.csv",
    RAW_DIR / "sod_2024.csv",
]

FINAL_OUTPUT = PROCESSED_DIR / "resumo_sod_ibge_profissoes.csv"
SOD_CLEAN_PARQUET = PROCESSED_DIR / "sod_clean_full.parquet"

LOG_FMT = "%(asctime)s | %(levelname)-8s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, handlers=[RichHandler(markup=False)])
logger = logging.getLogger("pipeline_rdi")

# ==============================================================================
# UTILITÁRIOS
# ==============================================================================

def normalizar_texto(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    s = (s.replace("ã","a").replace("á","a").replace("â","a").replace("à","a")
         .replace("é","e").replace("ê","e").replace("í","i")
         .replace("ó","o").replace("ô","o").replace("õ","o")
         .replace("ú","u").replace("ç","c"))
    return s

def detectar_separador(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            line = f.readline()
        if ";" in line: return ";"
        if "," in line: return ","
        return ","
    except:
        return ","

def classificar_cargo_macro(texto):
    """
    Agrupa cargos variados em categorias macro.
    """
    if pd.isna(texto) or str(texto).lower() == "nan":
        return "Outros"
    
    t = normalizar_texto(texto)
    
    if "engenh" in t and "dad" in t: return "Engenharia de Dados"
    if "scien" in t or "cient" in t: return "Ciencia de Dados"
    if "analis" in t or "analyst" in t: return "Analise de Dados"
    if "machine" in t or "learning" in t or "ml" in t: return "Engenharia de ML/IA"
    if "arquitet" in t: return "Arquitetura de Dados"
    if "gest" in t or "lead" in t or "head" in t or "gerente" in t: return "Gestao"
    
    return "Outros"

# ==============================================================================
# 1. PROCESSAMENTO DO STATE OF DATA
# ==============================================================================

def processar_state_of_data():
    if SOD_CLEAN_PARQUET.exists():
        logger.info("Carregando State of Data pré-processado...")
        return pd.read_parquet(SOD_CLEAN_PARQUET)

    dfs = []
    for p in SOD_INPUTS:
        if p.exists():
            try:
                # Lê tudo como string (dtype=str) para evitar inferência errada de tipos
                df = pd.read_csv(p, sep=detectar_separador(p), low_memory=False, encoding="utf-8", dtype=str)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Erro ao ler {p}: {e}")
    
    if not dfs:
        raise FileNotFoundError("Nenhum CSV do State of Data encontrado.")

    df_raw = pd.concat(dfs, ignore_index=True)

    # 1. Normaliza nomes das colunas
    df_raw.columns = [normalizar_texto(c).replace(" ", "_") for c in df_raw.columns]

    # 2. Encontra coluna de Cargo e UF
    cols = df_raw.columns
    col_cargo = next((c for c in cols if "cargo" in c or "role" in c or "atuacao" in c), "cargo_atual")
    col_uf = next((c for c in cols if "uf" in c or "estado" in c and "residencia" in c), "uf")

    logger.info(f"Coluna Cargo: {col_cargo} | Coluna UF: {col_uf}")

    # 3. Cria coluna Macro de Cargo
    # .copy() garante que estamos trabalhando num frame novo, evitando fragmentação
    df_raw = df_raw.copy()
    df_raw["cargo_macro"] = df_raw[col_cargo].apply(classificar_cargo_macro)

    # 4. Normaliza UF
    def limpar_uf(x):
        if pd.isna(x): return "ND"
        s = str(x).upper()
        if "(" in s and ")" in s:
            return s.split('(')[-1].split(')')[0].strip()
        return s.strip()

    df_raw["uf_sigla"] = df_raw[col_uf].apply(limpar_uf)
    
    mapa_sigla = {"SAO PAULO": "SP", "RIO DE JANEIRO": "RJ", "MINAS GERAIS": "MG", "ESPIRITO SANTO": "ES"}
    df_raw["uf_sigla"] = df_raw["uf_sigla"].replace(mapa_sigla)
    
    # 5. Filtro e Correção de Cópia (AQUI ESTAVA O PROBLEMA)
    # Ao filtrar e adicionar .copy(), criamos um DataFrame novo e independente.
    df_raw = df_raw[df_raw["uf_sigla"].str.len() == 2].copy()

    # 6. Conversão de Tipos Segura (Evita erro ArrowInvalid)
    # Como já demos .copy() acima, isso agora é seguro.
    for col in df_raw.columns:
        if df_raw[col].dtype == 'object':
            df_raw[col] = df_raw[col].astype(str)

    logger.info("Salvando arquivo Parquet intermediário...")
    df_raw.to_parquet(SOD_CLEAN_PARQUET)
    return df_raw

# ==============================================================================
# 2. COLETA IBGE
# ==============================================================================

def coletar_ibge():
    headers = {"User-Agent": "Mozilla/5.0"}
    res = {}
    
    def baixar(nome, url):
        try:
            logger.info(f"Baixando IBGE: {nome}...")
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                df = pd.DataFrame(r.json())
                if not df.empty:
                    # Tenta achar colunas dinamicamente
                    col_uf = next((c for c in df.columns if c.endswith('N') and df[c].str.len().mean() > 3), None)
                    col_v = next((c for c in df.columns if c.lower() in ['v', 'valor']), None)
                    
                    if col_uf and col_v:
                        df['uf_sigla'] = df[col_uf].apply(lambda x: normalizar_texto(x))
                        df['V'] = df[col_v]
                        return df
            return None
        except Exception as e:
            logger.warning(f"Erro IBGE {nome}: {e}")
            return None

    # População
    df_pop = baixar("populacao", "https://apisidra.ibge.gov.br/values/t/9514/n3/all/v/93/p/2022/c2/all/c287/all")
    if df_pop is not None:
        mapa = {"acre":"AC","alagoas":"AL","amapa":"AP","amazonas":"AM","bahia":"BA","ceara":"CE","distrito federal":"DF","espirito santo":"ES","goias":"GO","maranhao":"MA","mato grosso":"MT","mato grosso do sul":"MS","minas gerais":"MG","para":"PA","paraiba":"PB","parana":"PR","pernambuco":"PE","piaui":"PI","rio de janeiro":"RJ","rio grande do norte":"RN","rio grande do sul":"RS","rondonia":"RO","roraima":"RR","santa catarina":"SC","sao paulo":"SP","sergipe":"SE","tocantins":"TO"}
        df_pop['uf_sigla'] = df_pop['uf_sigla'].map(mapa).str.upper()
        df_pop['populacao'] = pd.to_numeric(df_pop['V'], errors='coerce')
        res['populacao'] = df_pop[['uf_sigla', 'populacao']].groupby('uf_sigla').max().reset_index()

    # PIB
    df_pib = baixar("pib", "https://apisidra.ibge.gov.br/values/t/5938/n3/all/v/37/p/last")
    if df_pib is not None:
        mapa = {"acre":"AC","alagoas":"AL","amapa":"AP","amazonas":"AM","bahia":"BA","ceara":"CE","distrito federal":"DF","espirito santo":"ES","goias":"GO","maranhao":"MA","mato grosso":"MT","mato grosso do sul":"MS","minas gerais":"MG","para":"PA","paraiba":"PB","parana":"PR","pernambuco":"PE","piaui":"PI","rio de janeiro":"RJ","rio grande do norte":"RN","rio grande do sul":"RS","rondonia":"RO","roraima":"RR","santa catarina":"SC","sao paulo":"SP","sergipe":"SE","tocantins":"TO"}
        df_pib['uf_sigla'] = df_pib['uf_sigla'].map(mapa).str.upper()
        df_pib['pib_per_capita'] = pd.to_numeric(df_pib['V'], errors='coerce')
        res['pib'] = df_pib[['uf_sigla', 'pib_per_capita']].groupby('uf_sigla').max().reset_index()

    return res

# ==============================================================================
# 3. UNIFICAÇÃO INTELIGENTE
# ==============================================================================

def gerar_analise_final(df_sod, dados_ibge):
    logger.info("Gerando análise cruzada de profissões...")

    # Pivot Table
    resumo_cargos = pd.crosstab(df_sod["uf_sigla"], df_sod["cargo_macro"])
    resumo_cargos.columns = [f"qtd_{normalizar_texto(c).replace(' ', '_')}" for c in resumo_cargos.columns]
    resumo_cargos = resumo_cargos.reset_index()

    resumo_cargos["total_tech"] = resumo_cargos.sum(axis=1, numeric_only=True)

    # Merge
    if "populacao" in dados_ibge:
        resumo_cargos = resumo_cargos.merge(dados_ibge["populacao"], on="uf_sigla", how="left")
    
    if "pib" in dados_ibge:
        resumo_cargos = resumo_cargos.merge(dados_ibge["pib"], on="uf_sigla", how="left")

    # Métricas Derivadas
    if "populacao" in resumo_cargos.columns:
        cols_qtd = [c for c in resumo_cargos.columns if c.startswith("qtd_")]
        for col in cols_qtd:
            nova_col = col.replace("qtd_", "densidade_") + "_por_milhao"
            resumo_cargos[nova_col] = ((resumo_cargos[col] / resumo_cargos["populacao"]) * 1000000).round(1)

    return resumo_cargos

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    try:
        logger.info("Iniciando pipeline...")
        
        # 1. Processa
        df_sod = processar_state_of_data()
        
        # 2. Baixa IBGE
        dados_ibge = coletar_ibge()
        
        # 3. Cruza
        df_final = gerar_analise_final(df_sod, dados_ibge)
        
        df_final.to_csv(FINAL_OUTPUT, index=False)
        logger.info(f"Sucesso! Arquivo gerado em: {FINAL_OUTPUT}")
        
    except Exception as e:
        logger.exception("Erro fatal no pipeline")

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
# trabalho de Guilherme Mendes Carlos
import pandas as pd
import csv
import re
import numpy as np
from difflib import SequenceMatcher

# =============================
# AJUSTE DOS CAMINHOS LOCAIS
# =============================
base_path = r"C:\Users\PC\Desktop\commit 2\projeto novo RDI\data\raw"

path21 = fr"{base_path}\sod_2021.csv"
path22 = fr"{base_path}\sod_2022.csv"
path23 = fr"{base_path}\sod_2023.csv"
path24 = fr"{base_path}\sod_2024.csv"

# =============================
# Funções auxiliares
# =============================

def detectar_separador(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        amostra = "".join([f.readline() for _ in range(5)])
    candidatos = [",", ";", "|", "\t"]
    return max(candidatos, key=lambda sep: amostra.count(sep))

def limpar_nome(col):
    col = str(col).lower()
    col = re.sub(r"[^\w]+", "_", col)
    col = col.strip("_")
    return col

def carregar_csv_padrao(path):
    sep = detectar_separador(path)
    try:
        df = pd.read_csv(path, sep=sep, encoding="utf-8", engine="python")
    except:
        df = pd.read_csv(path, sep=sep, encoding="latin1", engine="python")

    df.columns = [limpar_nome(c) for c in df.columns]
    return df

# =============================
# Carregar arquivos
# =============================

df21 = carregar_csv_padrao(path21)
df22 = carregar_csv_padrao(path22)
df23 = carregar_csv_padrao(path23)
df24 = carregar_csv_padrao(path24)

df21["ano"] = 2021
df22["ano"] = 2022
df23["ano"] = 2023
df24["ano"] = 2024

print(df21.shape, df22.shape, df23.shape, df24.shape)

# =============================
# Concatenar
# =============================
df_raw = pd.concat([df21, df22, df23, df24], ignore_index=True)
print("Shape df_raw:", df_raw.shape)

# =============================
# Remover colunas vazias
# =============================
colunas_vazias = [c for c in df_raw.columns if df_raw[c].isna().mean() == 1]
df_raw = df_raw.drop(columns=colunas_vazias)
print("Shape após remover vazias:", df_raw.shape)

# =============================
# Agrupar colunas similares
# =============================

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

grupos = []
visitadas = set()

for col in df_raw.columns:
    if col in visitadas:
        continue
    grupo = [col]
    visitadas.add(col)
    for col2 in df_raw.columns:
        if col2 in visitadas:
            continue
        if similar(col, col2) >= 0.60:
            grupo.append(col2)
            visitadas.add(col2)
    grupos.append(grupo)

print("Grupos:", len(grupos))
for g in grupos[:30]:
    print(g, "\n")

# =============================
# Unificar colunas dos grupos
# =============================
df = df_raw.copy()
df_clean = pd.DataFrame()
df_clean["ano"] = df["ano"]

def limpar(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower()
    x = (x.replace("ã","a").replace("á","a").replace("â","a").replace("à","a")
           .replace("é","e").replace("ê","e")
           .replace("í","i")
           .replace("ó","o").replace("ô","o").replace("õ","o")
           .replace("ú","u")
           .replace("ç","c"))
    return x

for grupo in grupos:
    coluna_padrao = grupo[0]
    cols_presentes = [c for c in grupo if c in df.columns]

    if len(cols_presentes) == 1:
        df_clean[coluna_padrao] = df[cols_presentes[0]]
    else:
        df_clean[coluna_padrao] = df[cols_presentes].bfill(axis=1).iloc[:, 0]

for col in df_clean.select_dtypes(include="object").columns:
    df_clean[col] = df_clean[col].apply(limpar)

# =============================
# Converter p1_a_idade
# =============================

print("Valores únicos sample (p1_a_idade):",
      df_clean["p1_a_idade"].dropna().unique()[:20])

df_clean["p1_a_idade_num"] = pd.to_numeric(df_clean["p1_a_idade"], errors="coerce")
df_clean["p1_a_idade"] = df_clean["p1_a_idade_num"].astype("Int64")
df_clean = df_clean.drop(columns=["p1_a_idade_num"])

print("Conversão concluída:", df_clean["p1_a_idade"].dtype)

# =============================
# Detectar possíveis numéricas
# =============================
candidate_cols = []
for c in df_clean.columns:
    if df_clean[c].dtype == "object":
        coerced = pd.to_numeric(df_clean[c], errors="coerce")
        frac_numeric = coerced.notna().mean()
        if frac_numeric >= 0.8 and df_clean[c].nunique() > 5:
            candidate_cols.append(c)

print("Colunas candidatas:", candidate_cols)

for c in candidate_cols:
    df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce").astype("Int64")
    print(f"Convertida {c}")

# =============================
# Remover totalmente vazias novamente
# =============================
cols_vazias = [c for c in df_clean.columns if df_clean[c].isna().mean() == 1]
df_clean = df_clean.drop(columns=cols_vazias)

# Criar ID global
df_clean = df_clean.reset_index().rename(columns={"index": "id_global"})

# Salvar
df_clean.to_csv(
    r"C:\Users\PC\Desktop\commit 2\projeto novo RDI\data\processed\sod_2021_2024_limpo.csv",
    index=False,
    encoding="utf-8"
)
print("Exportado CSV final — shape:", df_clean.shape)

# =============================
# Padronizar nomes das colunas
# =============================

def limpar_nome_final(col):
    col = re.sub(r"^p\d+_[a-z]_", "", col)
    col = re.sub(r"^p\d+_", "", col)
    col = col.strip().lower()
    return col

df_clean.columns = [limpar_nome_final(c) for c in df_clean.columns]

print("Shape:", df_clean.shape)
print(df_clean.head())

# =============================
# Remover colunas incompletas por ano
# =============================
anos = df_clean["ano"].unique()
colunas_ruins = []

for coluna in df_clean.columns:
    if coluna == "ano":
        continue
    for ano in anos:
        subset = df_clean[df_clean["ano"] == ano][coluna]
        if subset.notna().sum() == 0:
            colunas_ruins.append(coluna)
            break

df_filtrado = df_clean.drop(columns=colunas_ruins)

# =============================
# % nulos
# =============================
nulos = df_clean.isna().mean().sort_values(ascending=False)
limite = 0.80
colunas_remover = nulos[nulos > limite].index
df_clean = df_clean.drop(columns=colunas_remover)

existencia = df_clean.groupby("ano").count().T
colunas_incompletas = existencia[(existencia == 0).any(axis=1)].index
df_clean = df_clean.drop(columns=colunas_incompletas)

# =============================
# Converter binárias para Sim/Não
# =============================
binarias = []

for col in df_clean.columns:
    valores = set(df_clean[col].dropna().unique())
    if valores.issubset({0.0, 1.0}):
        binarias.append(col)

df_clean[binarias] = df_clean[binarias].replace({0.0: "Não", 1.0: "Sim"})

print("Colunas convertidas para Sim/Não:", binarias)
"""
unify_sod_ibge.py

Lê e unifica State of Data (2021-2024), limpa, normaliza, e faz merge com IBGE (Censo 2022).
Salva saídas em data/processed como CSV e Parquet.

Coloque este arquivo em: src/processing/unify_sod_ibge.py
Execute (a partir da raiz do projeto):
    python -m src.processing.unify_sod_ibge
ou
    python src/processing/unify_sod_ibge.py
"""

from pathlib import Path
import re
import logging
from difflib import SequenceMatcher
import json

import pandas as pd
import numpy as np

# slugify para normalizar nomes
try:
    from slugify import slugify
except Exception:
    def slugify(x, separator="_"):
        # fallback simples
        x = str(x)
        x = x.lower()
        x = re.sub(r"[^\w\s-]", "", x, flags=re.UNICODE)
        x = re.sub(r"[\s-]+", separator, x).strip(separator)
        return x

# Sidra (prefer sidrapy, fallback requests)
HAS_SIDRAPY = False
try:
    import sidrapy
    HAS_SIDRAPY = True
except Exception:
    import requests

# ---------------- CONFIG ----------------
BASE_RAW = Path(r"C:\Users\PC\Desktop\WORKSPACE\projeto novo RDI\data\raw")
BASE_PROCESSED = Path(r"C:\Users\PC\Desktop\WORKSPACE\projeto novo RDI\data\processed")

SOD_FILES = [
    BASE_RAW / "sod_2021.csv",
    BASE_RAW / "sod_2022.csv",
    BASE_RAW / "sod_2023.csv",
    BASE_RAW / "sod_2024.csv",
]

SOD_OUTPUT_BASENAME = "sod_2021_2024_clean"
FINAL_OUTPUT_BASENAME = "sod_ibge_2021_2024"

COL_SIMILARITY_THRESHOLD = 0.60
REMOVE_COLS_NA_FRAC = 0.80

# Mapa para transformar nomes de UF em siglas (em lower-case)
MAPA_ESTADOS = {
    'acre': 'AC','alagoas':'AL','amapa':'AP','amazonas':'AM','bahia':'BA','ceara':'CE',
    'distrito_federal':'DF','espirito_santo':'ES','goias':'GO','maranhao':'MA',
    'mato_grosso':'MT','mato_grosso_do_sul':'MS','minas_gerais':'MG','para':'PA',
    'paraiba':'PB','parana':'PR','pernambuco':'PE','piaui':'PI','rio_de_janeiro':'RJ',
    'rio_grande_do_norte':'RN','rio_grande_do_sul':'RS','rondonia':'RO','roraima':'RR',
    'santa_catarina':'SC','sao_paulo':'SP','sergipe':'SE','tocantins':'TO'
}

# --------------- logs -------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")


# ---------------- util ------------------
def ensure_dirs():
    BASE_PROCESSED.mkdir(parents=True, exist_ok=True)
    logging.info(f"Pasta de saída garantida: {BASE_PROCESSED}")


def detectar_separador(path: Path, n_lines: int = 5):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        amostra = "".join([f.readline() for _ in range(n_lines)])
    candidatos = [",", ";", "|", "\t"]
    scores = {sep: amostra.count(sep) for sep in candidatos}
    sep = max(scores, key=scores.get)
    logging.debug(f"Separador detectado para {path.name}: {sep} (scores={scores})")
    return sep


def normalizar_nome_coluna(col):
    if pd.isna(col):
        return col
    # transformar para snake_case sem acentos
    s = slugify(str(col), separator="_")
    s = re.sub(r"_+", "_", s)
    s = s.strip("_").lower()
    return s if s else "col"


def carregar_csv_padrao(path: Path):
    logging.info(f"Lendo: {path}")
    if not path.exists():
        logging.warning(f"Arquivo não encontrado: {path}")
        return None
    sep = detectar_separador(path)
    try:
        df = pd.read_csv(path, sep=sep, encoding="utf-8", engine="python")
    except Exception:
        logging.warning("utf-8 falhou, tentando latin1")
        df = pd.read_csv(path, sep=sep, encoding="latin1", engine="python")
    df.columns = [normalizar_nome_coluna(c) for c in df.columns]
    return df


def normalizar_texto(valor):
    if pd.isna(valor):
        return np.nan
    s = str(valor).strip().lower()
    s = slugify(s, separator="_")
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s != "" else np.nan


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def agrupar_colunas_similares(cols, threshold=COL_SIMILARITY_THRESHOLD):
    grupos = []
    visitadas = set()
    for c in cols:
        if c in visitadas:
            continue
        grupo = [c]
        visitadas.add(c)
        for c2 in cols:
            if c2 in visitadas:
                continue
            if similar(c, c2) >= threshold:
                grupo.append(c2)
                visitadas.add(c2)
        grupos.append(grupo)
    return grupos


def combinar_grupos_em_coluna(df, grupos):
    resultado = pd.DataFrame(index=df.index)
    for grupo in grupos:
        nome = grupo[0]
        cols_presentes = [c for c in grupo if c in df.columns]
        if not cols_presentes:
            continue
        if len(cols_presentes) == 1:
            resultado[nome] = df[cols_presentes[0]]
        else:
            resultado[nome] = df[cols_presentes].bfill(axis=1).iloc[:, 0]
    return resultado


def detectar_colunas_numericas_possiveis(df, min_frac_numeric=0.8, min_unique=5):
    cand = []
    for c in df.columns:
        if df[c].dtype == "object":
            coerced = pd.to_numeric(df[c], errors="coerce")
            frac = coerced.notna().mean()
            if frac >= min_frac_numeric and df[c].nunique() >= min_unique:
                cand.append(c)
    return cand


def converter_binarios_para_sim_nao(df):
    bin_cols = []
    for col in df.columns:
        vals = df[col].dropna().unique()
        if len(vals) == 0:
            continue
        uniq = set()
        for v in vals:
            try:
                fv = float(v)
                if fv in (0.0, 1.0):
                    uniq.add(int(fv))
                else:
                    uniq.add(v)
            except Exception:
                uniq.add(v)
        if uniq.issubset({0, 1}):
            df[col] = df[col].replace({1: "Sim", 1.0: "Sim", 0: "Não", 0.0: "Não"})
            bin_cols.append(col)
    return bin_cols


def remover_colunas_por_frac_na(df, frac=REMOVE_COLS_NA_FRAC):
    na_frac = df.isna().mean()
    to_remove = na_frac[na_frac > frac].index.tolist()
    logging.info(f"Remover por NA>={frac}: {len(to_remove)} colunas")
    return df.drop(columns=to_remove), to_remove


def remover_colunas_vazias_por_ano(df, ano_col="ano"):
    if ano_col not in df.columns:
        return df, []
    anos = df[ano_col].unique()
    colunas_ruins = []
    for coluna in df.columns:
        if coluna == ano_col:
            continue
        for ano in anos:
            s = df[df[ano_col] == ano][coluna]
            if s.notna().sum() == 0:
                colunas_ruins.append(coluna)
                break
    colunas_ruins = list(set(colunas_ruins))
    logging.info(f"Remover por vazia em algum ano: {len(colunas_ruins)} colunas")
    return df.drop(columns=colunas_ruins), colunas_ruins


def salvar_df(df, basename, out_folder=BASE_PROCESSED):
    csv_path = out_folder / (basename + ".csv")
    parquet_path = out_folder / (basename + ".parquet")
    logging.info(f"Salvando CSV: {csv_path}")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logging.info(f"Salvando Parquet: {parquet_path}")
    # tenta parquet com pyarrow
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as e:
        logging.warning(f"Parquet falhou: {e}")
    return csv_path, parquet_path


# -------------- IBGE (Sidra) ----------------

def coletar_ibge_populacao_censo_2022():
    logging.info("Coletando dados IBGE (Censo 2022) - população por UF")
    # Tentar sidrapy se disponível
    try:
        if HAS_SIDRAPY:
            df = sidrapy.get_table(table_code="6579", territorial_level="3", variable="93", period="2022")
            if isinstance(df, pd.DataFrame) and not df.empty:
                cols = [c.lower() for c in df.columns]
                uf_col = next((c for c in df.columns if 'unidade' in c.lower() or 'uf' in c.lower()), df.columns[0])
                val_col = next((c for c in df.columns if 'valor' in c.lower() or 'value' in c.lower()), df.columns[-1])
                out = df[[uf_col, val_col]].copy()
                out.columns = ['uf_nome', 'populacao']
                out['populacao'] = pd.to_numeric(out['populacao'], errors='coerce')
                out['uf_sigla'] = out['uf_nome'].astype(str).str.lower().apply(lambda x: MAPA_ESTADOS.get(slugify(str(x)), None))
                out = out[['uf_sigla', 'populacao']].dropna(subset=['uf_sigla'])
                return out.groupby('uf_sigla', as_index=False).agg({'populacao': 'sum'})
    except Exception as e:
        logging.warning(f"sidrapy erro: {e}")

    # Fallback via requests (API Sidra)
    try:
        url = "https://apisidra.ibge.gov.br/values/t/6579/n1/all/v/93/p/2022"
        logging.info(f"Requisição direta Sidra: {url}")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        header = data[0]
        rows = data[1:]
        df = pd.DataFrame(rows, columns=header)
        # achar colunas sensíveis
        possible_uf = next((c for c in df.columns if 'Unidade' in c or 'unidade' in c.lower() or 'territorial' in c.lower()), df.columns[0])
        possible_val = next((c for c in df.columns if 'Valor' in c or 'valor' in c.lower() or 'Value' in c.lower()), df.columns[-1])
        out = df[[possible_uf, possible_val]].copy()
        out.columns = ['uf_nome', 'populacao']
        out['populacao'] = pd.to_numeric(out['populacao'], errors='coerce')
        out['uf_sigla'] = out['uf_nome'].astype(str).str.lower().apply(lambda x: MAPA_ESTADOS.get(slugify(x), None))
        out = out[['uf_sigla', 'populacao']].dropna(subset=['uf_sigla'])
        return out.groupby('uf_sigla', as_index=False).agg({'populacao': 'sum'})
    except Exception as e:
        logging.error(f"Erro coletando dados via API Sidra: {e}")
        return None


# ---------------- main pipelines ----------------

def pipeline_unificar_sod(arquivo_paths):
    ensure_dirs()
    dfs = []
    for path in arquivo_paths:
        if path is None:
            continue
        path = Path(path)
        if not path.exists():
            logging.warning(f"Aviso: arquivo ausente {path}. Pulando.")
            continue
        df = carregar_csv_padrao(path)
        if df is None:
            continue

        # detectar ano pelo nome do arquivo ou coluna 'ano'
        ano_detectado = None
        m = re.search(r"20(21|22|23|24)", str(path))
        if m:
            ano_detectado = int("20" + m.group(1))
        if 'ano' in df.columns:
            try:
                val = df['ano'].dropna().unique()
                if len(val) == 1:
                    ano_detectado = int(val[0])
            except Exception:
                pass
        if ano_detectado is None:
            ano_detectado = 0
        df['ano'] = ano_detectado
        dfs.append(df)

    if not dfs:
        logging.error("Nenhum CSV válido para processar. Verifique os caminhos e nomes.")
        return None

    logging.info("Concatenando datasets SOD (raw)")
    df_raw = pd.concat(dfs, ignore_index=True, sort=False)
    # normalizar nomes de colunas novamente (garantia)
    df_raw.columns = [normalizar_nome_coluna(c) for c in df_raw.columns]
    # remover colunas 100% vazias
    all_empty = [c for c in df_raw.columns if df_raw[c].isna().all()]
    logging.info(f"Colunas 100% vazias: {len(all_empty)}. Removendo.")
    df_raw = df_raw.drop(columns=all_empty)

    logging.info(f"Dimensão inicial concatenada: {df_raw.shape}")

    # Agrupar colunas por similaridade de nome
    grupos = agrupar_colunas_similares(list(df_raw.columns), threshold=COL_SIMILARITY_THRESHOLD)
    logging.info(f"Grupos detectados (heurística): {len(grupos)}")

    df_comb = combinar_grupos_em_coluna(df_raw, grupos)

    # garantir ano presente
    if 'ano' not in df_comb.columns and 'ano' in df_raw.columns:
        df_comb['ano'] = df_raw['ano']

    # normalizar texto nas colunas object
    for col in df_comb.select_dtypes(include="object").columns:
        df_comb[col] = df_comb[col].apply(normalizar_texto)

    # detectar colunas que são majoritariamente numéricas e converter
    cand_num = detectar_colunas_numericas_possiveis(df_comb)
    logging.info(f"Colunas candidatas a numéricas: {cand_num}")
    for c in cand_num:
        try:
            df_comb[c] = pd.to_numeric(df_comb[c], errors="coerce").astype("Float64")
        except Exception:
            df_comb[c] = pd.to_numeric(df_comb[c], errors="coerce")

    # converter possíveis binárias
    bin_cols = converter_binarios_para_sim_nao(df_comb)
    logging.info(f"Colunas convertidas Sim/Não: {bin_cols}")

    # remover colunas com alta proporção de NA
    df_comb, removed_by_na = remover_colunas_por_frac_na(df_comb, REMOVE_COLS_NA_FRAC)
    logging.info(f"Removidas por NA: {removed_by_na[:10]}")

    # remover colunas vazias em algum ano
    df_comb, removed_by_year = remover_colunas_vazias_por_ano(df_comb, ano_col='ano')
    logging.info(f"Removidas por vazio por ano: {removed_by_year[:10]}")

    # reset index e criar id_global
    df_comb = df_comb.reset_index(drop=True).reset_index().rename(columns={"index": "id_global"})
    # garantir tipos mínimos
    # converter string "nan"/"none" para NaN
    df_comb = df_comb.replace({"nan": np.nan, "none": np.nan})

    logging.info(f"Dimensão final SOD combinado: {df_comb.shape}")

    sod_csv, sod_parquet = salvar_df(df_comb, SOD_OUTPUT_BASENAME)
    return df_comb


def pipeline_ibge_merge(df_sod):
    df_ibge = coletar_ibge_populacao_censo_2022()
    if df_ibge is None or df_ibge.empty:
        logging.error("Falha ao obter dados IBGE. Abortando merge.")
        return None

    # tentar usar uf_sigla do SOD
    if 'uf_sigla' in df_sod.columns:
        df_sod['uf_sigla'] = df_sod['uf_sigla'].astype(str).str.upper()
        df_ibge['uf_sigla'] = df_ibge['uf_sigla'].astype(str).str.upper()
        if 'qtd_profissionais' not in df_sod.columns:
            agg = df_sod.groupby(['ano', 'uf_sigla']).size().reset_index(name='qtd_profissionais')
        else:
            agg = df_sod.groupby(['ano', 'uf_sigla'])['qtd_profissionais'].sum().reset_index()
    else:
        # tenta inferir colunas comuns
        possible = [c for c in df_sod.columns if c in ('state','uf','estado','estado_onde_mora','state_sigla','uf_sigla')]
        if possible:
            col = possible[0]
            df_sod['uf_sigla'] = df_sod[col].astype(str).str.upper()
            if 'qtd_profissionais' not in df_sod.columns:
                agg = df_sod.groupby(['ano', 'uf_sigla']).size().reset_index(name='qtd_profissionais')
            else:
                agg = df_sod.groupby(['ano', 'uf_sigla'])['qtd_profissionais'].sum().reset_index()
        else:
            logging.error("Não foi possível gerar agregação por UF no SOD. Abortando merge.")
            return None

    merged = pd.merge(agg, df_ibge, on='uf_sigla', how='left')
    merged['populacao'] = merged['populacao'].fillna(0).astype("Int64")
    # métricas
    merged['respondentes_por_100k'] = (merged['qtd_profissionais'] / merged['populacao'].replace({0: np.nan})) * 100000
    merged['respondentes_por_100k'] = merged['respondentes_por_100k'].round(3)

    salvar_df(merged, FINAL_OUTPUT_BASENAME)
    logging.info("Merge SOD + IBGE salvo com sucesso.")
    return merged


def main():
    logging.info("Iniciando pipeline de unificação SOD (2021-2024) e integração com IBGE (Censo 2022)")
    df_sod = pipeline_unificar_sod(SOD_FILES)
    if df_sod is None:
        logging.error("Pipeline de SOD falhou. Verifique arquivos raw.")
        return
    merged = pipeline_ibge_merge(df_sod)
    if merged is None:
        logging.warning("Merge final não gerado. Verifique mensagens de erro.")
    else:
        # salvar resumo simples
        resumo = merged.groupby('ano')['qtd_profissionais'].sum().reset_index()
        resumo_path = BASE_PROCESSED / "resumo_qtd_por_ano.json"
        resumo.to_json(resumo_path, orient="records", force_ascii=False)
        logging.info(f"Resumo por ano salvo em: {resumo_path}")
        logging.info("Pipeline completo.")


if __name__ == "__main__":
    main()
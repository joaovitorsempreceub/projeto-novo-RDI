import pandas as pd
import requests
import sidrapy
from pathlib import Path
import logging

# ============================
# CONFIGURAÇÃO DE LOG
# ============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ============================
# HEADERS PARA BURLAR RATE LIMIT DO SIDRA
# ============================
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Connection": "keep-alive",
    "Referer": "https://sidra.ibge.gov.br/",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
}


# ============================
# FUNÇÃO GENÉRICA PARA DOWNLOAD
# ============================
def baixar_tabela(nome, url, tabela_sidra):
    logger.info(f"🔍 Baixando {nome}...")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()

        df = pd.DataFrame(resp.json())
        path = PROCESSED_DIR / f"ibge_{nome}.csv"
        df.to_csv(path, index=False)

        logger.info(f"   ✔ salvo em {path}")
        return df

    except Exception as e:
        logger.error(f"❌ Erro direto no SIDRA para {nome}: {e}")
        logger.info("➡ Tentando fallback via sidrapy...")

        try:
            df = sidrapy.get_table(
                table_code=tabela_sidra,
                territorial_level="3",
                ibge_territorial_code="all"
            )
            df = pd.DataFrame(df)
            path = PROCESSED_DIR / f"ibge_{nome}.csv"
            df.to_csv(path, index=False)

            logger.info(f"   ✔ (fallback) salvo em {path}")
            return df

        except Exception as err:
            logger.error(f"❌ Fallback também falhou para {nome}: {err}")
            return None


# ============================
# COLETA DE TODAS AS TABELAS
# ============================
def coletar_ibge():
    resultados = {}

    resultados["populacao"] = baixar_tabela(
        nome="populacao",
        url="https://apisidra.ibge.gov.br/values/t/9514/n3/all/v/93/p/2022/c2/all/c287/all",
        tabela_sidra="9514"
    )

    resultados["renda"] = baixar_tabela(
        nome="renda",
        url="https://apisidra.ibge.gov.br/values/t/7327/n3/all/v/991/p/last",
        tabela_sidra="7327"
    )

    resultados["instrucao"] = baixar_tabela(
        nome="instrucao",
        url="https://apisidra.ibge.gov.br/values/t/6729/n3/all/p/last",
        tabela_sidra="6729"
    )

    logger.info("✅ Coleta IBGE finalizada.")
    return resultados


# ============================
# MAIN
# ============================
if __name__ == "__main__":
    logger.info("🌐 Iniciando coleta IBGE...")
    coletar_ibge()
    logger.info("🏁 Finalizado.")

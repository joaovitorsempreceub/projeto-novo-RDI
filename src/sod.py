import pandas as pd
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
DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ============================
# FUNÇÃO SOD
# ============================
def carregar_sod():
    """Carrega o arquivo ATUAL do State of Data."""
    path = DATA_DIR / "state_of_data_2024.csv"

    logger.info(f"📄 Lendo State of Data: {path}")

    df = pd.read_csv(path)
    logger.info(f"   ✔ Linhas carregadas: {len(df)}")

    # Seleção simples de colunas chave
    colunas_interesse = [
        "region",
        "state",
        "job_role",
        "salary",
        "experience",
        "education"
    ]

    df_sod = df[colunas_interesse].copy()

    output = PROCESSED_DIR / "sod_limpo.csv"
    df_sod.to_csv(output, index=False)

    logger.info(f"   ✔ Arquivo processado salvo em: {output}")

    return df_sod

# ============================
# MAIN
# ============================
if __name__ == "__main__":
    logger.info("🚀 Iniciando pipeline SOD...")
    carregar_sod()
    logger.info("🏁 Finalizado.")

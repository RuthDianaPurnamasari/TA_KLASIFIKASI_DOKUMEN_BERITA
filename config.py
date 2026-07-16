from pathlib import Path


# ============================================================
# ROOT PROJECT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# FOLDER DATA
# ============================================================

DATA_DIR = BASE_DIR / "2_data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SCENARIO_DATA_DIR = DATA_DIR / "scenarios"
DATA_SCRIPTS_DIR = DATA_DIR / "scripts"


# ============================================================
# FOLDER HASIL
# ============================================================

RESULTS_DIR = BASE_DIR / "9_results"

TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
LOGS_DIR = RESULTS_DIR / "logs"


# ============================================================
# FOLDER MODEL
# ============================================================

MODELS_DIR = BASE_DIR / "8_save_models"


# ============================================================
# DATASET MENTAH KOMPAS
# ============================================================

KOMPAS_RAW_FILES = {
    "bola": RAW_DATA_DIR / "kompas_bola_2500.csv",
    "global": RAW_DATA_DIR / "kompas_global_2500.csv",
    "money": RAW_DATA_DIR / "kompas_money_2500.csv",
    "tekno": RAW_DATA_DIR / "kompas_tekno_2500.csv",
}


# ============================================================
# DATASET MENTAH AG NEWS
# ============================================================

AG_NEWS_TRAIN_RAW = (
    RAW_DATA_DIR / "train dataset AG News.csv"
)

AG_NEWS_TEST_RAW = (
    RAW_DATA_DIR / "test dataset AG News.csv"
)


# ============================================================
# OUTPUT DATASET PROCESSED
# ============================================================

KOMPAS_PROCESSED_PATH = (
    PROCESSED_DATA_DIR / "kompas_processed.csv"
)

AG_NEWS_TRAIN_PROCESSED_PATH = (
    PROCESSED_DATA_DIR / "ag_news_train_processed.csv"
)

AG_NEWS_TEST_PROCESSED_PATH = (
    PROCESSED_DATA_DIR / "ag_news_test_processed.csv"
)


# ============================================================
# OUTPUT LAPORAN VALIDASI
# ============================================================

KOMPAS_VALIDATION_REPORT = (
    TABLES_DIR / "kompas_validation_report.csv"
)

AG_NEWS_VALIDATION_REPORT = (
    TABLES_DIR / "ag_news_validation_report.csv"
)


# ============================================================
# KONFIGURASI DATASET KOMPAS
# ============================================================

KOMPAS_EXPECTED_COLUMNS = [
    "title",
    "description",
    "content",
    "date",
    "category",
    "link",
]

KOMPAS_EXPECTED_CATEGORIES = [
    "bola",
    "global",
    "money",
    "tekno",
]


# ============================================================
# KONFIGURASI AG NEWS
# ============================================================

AG_NEWS_LABEL_MAPPING = {
    1: "world",
    2: "sports",
    3: "business",
    4: "sci_tech",
}


# ============================================================
# KONFIGURASI UMUM
# ============================================================

RANDOM_SEED = 42


# ============================================================
# MEMBUAT FOLDER OTOMATIS
# ============================================================

DIRECTORIES = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SCENARIO_DATA_DIR,
    DATA_SCRIPTS_DIR,
    RESULTS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    LOGS_DIR,
    MODELS_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )
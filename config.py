from pathlib import Path


# ============================================================
# ROOT PROJECT
# ============================================================

# Lokasi file config.py sekaligus root proyek
BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# FOLDER DATA
# ============================================================

DATA_DIR = BASE_DIR / "2_data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SCENARIO_DATA_DIR = DATA_DIR / "scenarios"
DATA_SCRIPTS_DIR = DATA_DIR / "scripts"
SPLIT_DATA_DIR = DATA_DIR / "splits"
VECTORIZED_DATA_DIR = DATA_DIR / "vectorized"


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

# File Kaggle:
# 2_data/raw/ag_news_train.csv
# 2_data/raw/ag_news_test.csv

AG_NEWS_TRAIN_RAW = (
    RAW_DATA_DIR
    / "ag_news_train.csv"
)

AG_NEWS_TEST_RAW = (
    RAW_DATA_DIR
    / "ag_news_test.csv"
)


# ============================================================
# OUTPUT DATASET PROCESSED
# ============================================================

KOMPAS_PROCESSED_PATH = (
    PROCESSED_DATA_DIR
    / "kompas_processed.csv"
)

AG_NEWS_TRAIN_PROCESSED_PATH = (
    PROCESSED_DATA_DIR
    / "ag_news_train_processed.csv"
)

AG_NEWS_TEST_PROCESSED_PATH = (
    PROCESSED_DATA_DIR
    / "ag_news_test_processed.csv"
)


# ============================================================
# OUTPUT LAPORAN VALIDASI
# ============================================================

KOMPAS_VALIDATION_REPORT = (
    TABLES_DIR
    / "kompas_validation_report.csv"
)

# Laporan gabungan train dan test.
# Variabel ini digunakan oleh 02_prepare_agnews.py.
AG_NEWS_VALIDATION_REPORT = (
    TABLES_DIR
    / "ag_news_validation_report.csv"
)

# Disediakan untuk script lain apabila laporan train
# dan test ingin dipisahkan.
AG_NEWS_TRAIN_VALIDATION_REPORT = (
    TABLES_DIR
    / "ag_news_train_validation_report.csv"
)

AG_NEWS_TEST_VALIDATION_REPORT = (
    TABLES_DIR
    / "ag_news_test_validation_report.csv"
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


# Label yang digunakan oleh model Kompas
KOMPAS_LABEL_MAPPING = {
    "bola": 0,
    "global": 1,
    "money": 2,
    "tekno": 3,
}


# Mengubah indeks model kembali menjadi nama kategori
KOMPAS_INDEX_TO_LABEL = {
    index: label
    for label, index
    in KOMPAS_LABEL_MAPPING.items()
}


# ============================================================
# KONFIGURASI DATASET AG NEWS
# ============================================================

# Label asli dari dataset AG News Kaggle
AG_NEWS_RAW_LABEL_MAPPING = {
    1: "world",
    2: "sports",
    3: "business",
    4: "sci_tech",
}


# Alias agar kompatibel dengan 02_prepare_agnews.py
AG_NEWS_LABEL_MAPPING = (
    AG_NEWS_RAW_LABEL_MAPPING
)


# Label yang digunakan saat pelatihan model
AG_NEWS_MODEL_LABEL_MAPPING = {
    "world": 0,
    "sports": 1,
    "business": 2,
    "sci_tech": 3,
}


# Mengubah indeks prediksi model menjadi nama kategori
AG_NEWS_INDEX_TO_LABEL = {
    index: label
    for label, index
    in AG_NEWS_MODEL_LABEL_MAPPING.items()
}


# ============================================================
# SKENARIO EKSPERIMEN FINAL
# ============================================================

EXPERIMENT_SCENARIOS = {
    "K1": {
        "dataset": "kompas",
        "representation": "title",
        "use_yake": False,
    },
    "K2": {
        "dataset": "kompas",
        "representation": "title_description",
        "use_yake": False,
    },
    "K3": {
        "dataset": "kompas",
        "representation": "title_description_keyword",
        "use_yake": True,
    },
    "A1": {
        "dataset": "ag_news",
        "representation": "title",
        "use_yake": False,
    },
    "A2": {
        "dataset": "ag_news",
        "representation": "title_description",
        "use_yake": False,
    },
}


# ============================================================
# KONFIGURASI UMUM
# ============================================================

RANDOM_SEED = 42


# ============================================================
# DAFTAR FOLDER YANG DIBUAT OTOMATIS
# ============================================================

DIRECTORIES = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SCENARIO_DATA_DIR,
    DATA_SCRIPTS_DIR,
    SPLIT_DATA_DIR,
    VECTORIZED_DATA_DIR,
    RESULTS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    LOGS_DIR,
    MODELS_DIR,
]


# ============================================================
# MEMBUAT FOLDER OTOMATIS
# ============================================================

def ensure_directories() -> None:
    """
    Membuat seluruh folder proyek apabila belum tersedia.
    """

    for directory in DIRECTORIES:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# Fungsi dijalankan otomatis saat config.py diimpor
ensure_directories()
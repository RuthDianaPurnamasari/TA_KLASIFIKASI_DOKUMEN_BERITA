# =============================================================================
# STREAMLIT DASHBOARD CONFIGURATION
# =============================================================================
# File ini menyimpan seluruh konfigurasi utama dashboard, seperti:
# - lokasi folder project;
# - lokasi model deployment;
# - lokasi hasil penelitian;
# - sequence length;
# - label kategori;
# - ringkasan performa model.
#
# Tujuannya agar path dan konfigurasi tidak ditulis berulang-ulang
# pada setiap halaman Streamlit.
# =============================================================================

from __future__ import annotations

from pathlib import Path


# =============================================================================
# PROJECT DIRECTORY
# =============================================================================

# Lokasi root project:
# C:\TA_KLASIFIKASI_DOKUMEN_BERITA
#
# File config.py berada di:
# C:\TA_KLASIFIKASI_DOKUMEN_BERITA\10_streamlit\config.py
#
# parents[1] berarti naik dua tingkat:
# config.py -> 10_streamlit -> root project
PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


# =============================================================================
# MAIN DIRECTORIES
# =============================================================================

# Folder dashboard Streamlit
STREAMLIT_DIR = (
    PROJECT_ROOT
    / "10_streamlit"
)

# Folder model yang sudah disiapkan untuk deployment
DEPLOYMENT_DIR = (
    PROJECT_ROOT
    / "8_save_models"
    / "deployment"
)

# Folder seluruh hasil penelitian
RESULTS_DIR = (
    PROJECT_ROOT
    / "9_results"
)

# Folder tabel hasil penelitian
TABLES_DIR = (
    RESULTS_DIR
    / "tables"
)

# Folder grafik hasil penelitian
FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)


# =============================================================================
# DEPLOYMENT MODEL PATHS
# =============================================================================

# Model utama penelitian:
# CNN K2 — Title + Description
CNN_MODEL_PATH = (
    DEPLOYMENT_DIR
    / "cnn_k2.keras"
)

# Model pembanding:
# Attention-BiLSTM K2 — Title + Description
ATTENTION_BILSTM_MODEL_PATH = (
    DEPLOYMENT_DIR
    / "attention_bilstm_k2.keras"
)


# =============================================================================
# SUPPORTING DEPLOYMENT FILES
# =============================================================================

# Vocabulary yang sama dengan vocabulary saat training CNN K2
VOCABULARY_PATH = (
    DEPLOYMENT_DIR
    / "vocabulary.txt"
)

# Konfigurasi TextVectorization
VECTORIZER_CONFIG_PATH = (
    DEPLOYMENT_DIR
    / "vectorizer_config.json"
)

# Mapping label angka ke nama kategori
LABEL_MAPPING_PATH = (
    DEPLOYMENT_DIR
    / "label_mapping.json"
)

# Konfigurasi deployment secara umum
DEPLOYMENT_CONFIG_PATH = (
    DEPLOYMENT_DIR
    / "deployment_config.json"
)


# =============================================================================
# MODEL INPUT CONFIGURATION
# =============================================================================

# Panjang maksimal sequence untuk K2.
#
# Berdasarkan eksperimen final:
# K2 = Title + Description
# max sequence length = 60
MAX_SEQUENCE_LENGTH = 60

# Jumlah kelas berita Kompas
NUM_CLASSES = 4


# =============================================================================
# DEFAULT LABEL MAPPING
# =============================================================================

# Mapping ini digunakan sebagai cadangan jika file
# label_mapping.json tidak ditemukan atau tidak dapat dibaca.
DEFAULT_LABEL_MAPPING = {
    0: "bola",
    1: "global",
    2: "money",
    3: "tekno",
}


# =============================================================================
# DISPLAY LABELS
# =============================================================================

# Nama kategori untuk ditampilkan di dashboard.
# Mapping model tetap menggunakan huruf kecil,
# sedangkan dashboard menampilkan huruf kapital.
DISPLAY_LABELS = {
    "bola": "Bola",
    "global": "Global",
    "money": "Money",
    "tekno": "Tekno",
}


# =============================================================================
# RESEARCH INFORMATION
# =============================================================================

RESEARCH_TITLE = (
    "Analisis Perbandingan Kinerja CNN dan Attention-BiLSTM "
    "untuk Klasifikasi Berita Berbahasa Indonesia "
    "Menggunakan Representasi Teks Berbasis Dashboard Streamlit"
)

PRIMARY_DATASET = "Kompas"

BENCHMARK_DATASET = "AG News"

PRIMARY_MODEL = "CNN K2"

COMPARISON_MODEL = "Attention-BiLSTM K2"

PRIMARY_SCENARIO = "K2 — Title + Description"


# =============================================================================
# FINAL MODEL PERFORMANCE
# =============================================================================

# Ringkasan hasil model final pada test set Kompas.
MODEL_PERFORMANCE = {
    "CNN K2": {
        "model_name": "CNN",
        "scenario_code": "K2",
        "scenario_name": "Title + Description",
        "accuracy": 0.9580,
        "f1_macro": 0.958068,
    },

    "Attention-BiLSTM K2": {
        "model_name": "Attention-BiLSTM",
        "scenario_code": "K2",
        "scenario_name": "Title + Description",
        "accuracy": 0.9530,
        "f1_macro": 0.9530,
    },
}


# =============================================================================
# DATASET INFORMATION
# =============================================================================

DATASET_INFORMATION = {
    "Kompas": {
        "jumlah_data_awal": 10000,
        "jumlah_data_setelah_cleaning": 9997,
        "jumlah_kategori": 4,
        "categories": [
            "Bola",
            "Global",
            "Money",
            "Tekno",
        ],
    },

    "AG News": {
        "jumlah_data_train_awal": 120000,
        "jumlah_data_train_setelah_cleaning": 119871,
        "jumlah_data_test_setelah_cleaning": 7464,
        "jumlah_kategori": 4,
        "categories": [
            "Business",
            "Sci/Tech",
            "Sports",
            "World",
        ],
    },
}


# =============================================================================
# SHAP CONFIGURATION
# =============================================================================

SHAP_MODEL_NAME = "CNN K2"

SHAP_SCENARIO = "K2 — Title + Description"

SPECIAL_TOKENS = {
    "[PAD]",
    "[SEP]",
    "[UNK]",
    "[OOV]",
}

if __name__ == "__main__":
    print("=" * 80)
    print("STREAMLIT CONFIGURATION TEST")
    print("=" * 80)

    print(f"Project root          : {PROJECT_ROOT}")
    print(f"Deployment directory  : {DEPLOYMENT_DIR}")
    print(f"CNN model             : {CNN_MODEL_PATH}")
    print(f"Attention-BiLSTM      : {ATTENTION_BILSTM_MODEL_PATH}")
    print(f"Vocabulary            : {VOCABULARY_PATH}")
    print(f"Sequence length       : {MAX_SEQUENCE_LENGTH}")

    print("\nValidasi file:")

    print(
        f"CNN model tersedia           : "
        f"{CNN_MODEL_PATH.exists()}"
    )

    print(
        f"Attention-BiLSTM tersedia    : "
        f"{ATTENTION_BILSTM_MODEL_PATH.exists()}"
    )

    print(
        f"Vocabulary tersedia          : "
        f"{VOCABULARY_PATH.exists()}"
    )

    print(
        f"Label mapping tersedia       : "
        f"{LABEL_MAPPING_PATH.exists()}"
    )
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# =============================================================================
# PROJECT DIRECTORY
# =============================================================================

# Lokasi file:
# C:\TA_KLASIFIKASI_DOKUMEN_BERITA\10_streamlit\config.py
#
# parents[0] = 10_streamlit
# parents[1] = root project
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# MAIN DIRECTORIES
# =============================================================================

STREAMLIT_DIR = (
    PROJECT_ROOT
    / "10_streamlit"
)

ASSETS_DIR = (
    STREAMLIT_DIR
    / "assets"
)

DEPLOYMENT_DIR = (
    PROJECT_ROOT
    / "8_save_models"
    / "deployment"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "9_results"
)

TABLES_DIR = (
    RESULTS_DIR
    / "tables"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

PREDICTIONS_DIR = (
    RESULTS_DIR
    / "predictions"
)

TRAINING_HISTORY_DIR = (
    RESULTS_DIR
    / "training_history"
)


# =============================================================================
# DATASET DIRECTORIES
# =============================================================================

DATA_DIR = (
    PROJECT_ROOT
    / "2_data"
)

RAW_DATA_DIR = (
    DATA_DIR
    / "raw"
)

PROCESSED_DATA_DIR = (
    DATA_DIR
    / "processed"
)


# =============================================================================
# FINAL DATASET PATHS
# =============================================================================

KOMPAS_FINAL_PATH = (
    PROCESSED_DATA_DIR
    / "kompas_clean.csv"
)

AG_NEWS_TRAIN_FINAL_PATH = (
    PROCESSED_DATA_DIR
    / "ag_news_train_clean.csv"
)

AG_NEWS_TEST_FINAL_PATH = (
    PROCESSED_DATA_DIR
    / "ag_news_test_clean.csv"
)


# =============================================================================
# DEPLOYMENT MODEL PATHS
# =============================================================================

CNN_MODEL_PATH = (
    DEPLOYMENT_DIR
    / "cnn_k2.keras"
)

ATTENTION_BILSTM_MODEL_PATH = (
    DEPLOYMENT_DIR
    / "attention_bilstm_k2.keras"
)


# =============================================================================
# SUPPORTING DEPLOYMENT FILES
# =============================================================================

VOCABULARY_PATH = (
    DEPLOYMENT_DIR
    / "vocabulary.txt"
)

VECTORIZER_CONFIG_PATH = (
    DEPLOYMENT_DIR
    / "vectorizer_config.json"
)

LABEL_MAPPING_PATH = (
    DEPLOYMENT_DIR
    / "label_mapping.json"
)

DEPLOYMENT_CONFIG_PATH = (
    DEPLOYMENT_DIR
    / "deployment_config.json"
)

DEPLOYMENT_REPORT_PATH = (
    DEPLOYMENT_DIR
    / "deployment_report.json"
)


# =============================================================================
# EVALUATION RESULT PATHS
# =============================================================================

TEST_EVALUATION_PATH = (
    TABLES_DIR
    / "test_evaluation_summary.csv"
)

MODEL_COMPARISON_PATH = (
    TABLES_DIR
    / "model_comparison.csv"
)

SCENARIO_COMPARISON_PATH = (
    TABLES_DIR
    / "scenario_comparison.csv"
)

DESCRIPTION_CONTRIBUTION_PATH = (
    TABLES_DIR
    / "description_contribution.csv"
)

YAKE_CONTRIBUTION_PATH = (
    TABLES_DIR
    / "yake_contribution.csv"
)

MISCLASSIFICATION_ANALYSIS_PATH = (
    TABLES_DIR
    / "misclassification_analysis.csv"
)


# =============================================================================
# EDA RESULT PATHS
# =============================================================================

EDA_SUMMARY_PATH = (
    TABLES_DIR
    / "eda_summary.csv"
)

EDA_RESEARCH_FINDINGS_PATH = (
    TABLES_DIR
    / "eda_research_findings.csv"
)

EDA_STAGE_VALIDATION_PATH = (
    TABLES_DIR
    / "eda_stage_validation.csv"
)

CLASS_DISTRIBUTION_PATH = (
    TABLES_DIR
    / "class_distribution.csv"
)

DATA_CLEANING_COMPARISON_PATH = (
    TABLES_DIR
    / "data_cleaning_comparison.csv"
)

CLEANING_INTEGRITY_SUMMARY_PATH = (
    TABLES_DIR
    / "cleaning_integrity_summary.csv"
)

TEXT_STATISTICS_PATH = (
    TABLES_DIR
    / "text_statistics.csv"
)

TEXT_STATISTICS_BY_CATEGORY_PATH = (
    TABLES_DIR
    / "text_statistics_by_category.csv"
)

SEQUENCE_LENGTH_COVERAGE_PATH = (
    TABLES_DIR
    / "sequence_length_coverage.csv"
)

WORD_FREQUENCY_OVERALL_PATH = (
    TABLES_DIR
    / "word_frequency_overall.csv"
)

WORD_FREQUENCY_BY_CATEGORY_PATH = (
    TABLES_DIR
    / "word_frequency_by_category.csv"
)

WORDCLOUD_SUMMARY_PATH = (
    TABLES_DIR
    / "wordcloud_summary.csv"
)

KOMPAS_MONTHLY_DISTRIBUTION_PATH = (
    TABLES_DIR
    / "kompas_monthly_distribution.csv"
)

KOMPAS_MONTHLY_CATEGORY_DISTRIBUTION_PATH = (
    TABLES_DIR
    / "kompas_monthly_category_distribution.csv"
)

KOMPAS_DAILY_DISTRIBUTION_PATH = (
    TABLES_DIR
    / "kompas_daily_distribution.csv"
)

KOMPAS_HOURLY_DISTRIBUTION_PATH = (
    TABLES_DIR
    / "kompas_hourly_distribution.csv"
)

KOMPAS_WEEKDAY_DISTRIBUTION_PATH = (
    TABLES_DIR
    / "kompas_weekday_distribution.csv"
)

KOMPAS_TEMPORAL_SUMMARY_PATH = (
    TABLES_DIR
    / "kompas_temporal_summary.csv"
)


# =============================================================================
# WORD CLOUD PATHS
# =============================================================================

WORDCLOUD_DIR = (
    FIGURES_DIR
    / "wordclouds"
)

WORDCLOUD_PATHS = {
    "Kompas": {
        "all":
            WORDCLOUD_DIR
            / "kompas_overall_wordcloud.png",

        "bola":
            WORDCLOUD_DIR
            / "kompas_bola_wordcloud.png",

        "global":
            WORDCLOUD_DIR
            / "kompas_global_wordcloud.png",

        "money":
            WORDCLOUD_DIR
            / "kompas_money_wordcloud.png",

        "tekno":
            WORDCLOUD_DIR
            / "kompas_tekno_wordcloud.png",
    },

    "AG News": {
        "all":
            WORDCLOUD_DIR
            / "agnews_train_overall_wordcloud.png",

        "business":
            WORDCLOUD_DIR
            / "agnews_train_business_wordcloud.png",

        "sci_tech":
            WORDCLOUD_DIR
            / "agnews_train_sci_tech_wordcloud.png",

        "sports":
            WORDCLOUD_DIR
            / "agnews_train_sports_wordcloud.png",

        "world":
            WORDCLOUD_DIR
            / "agnews_train_world_wordcloud.png",
    },
}


# =============================================================================
# SHAP RESULT PATHS
# =============================================================================

SHAP_TABLES_DIR = (
    TABLES_DIR
    / "shap"
)

SHAP_FIGURES_DIR = (
    FIGURES_DIR
    / "shap"
)

SHAP_WATERFALL_DIR = (
    SHAP_FIGURES_DIR
    / "waterfall"
)

GLOBAL_SHAP_PATH = (
    SHAP_TABLES_DIR
    / "cnn_k2_global_shap.csv"
)

GLOBAL_SHAP_BY_CLASS_PATH = (
    SHAP_TABLES_DIR
    / "cnn_k2_global_shap_by_class.csv"
)

LOCAL_SHAP_SUMMARY_PATH = (
    SHAP_TABLES_DIR
    / "cnn_k2_local_shap_summary.csv"
)

LOCAL_TOKEN_CONTRIBUTIONS_PATH = (
    SHAP_TABLES_DIR
    / "cnn_k2_local_token_contributions.csv"
)

WATERFALL_SUMMARY_PATH = (
    SHAP_TABLES_DIR
    / "cnn_k2_waterfall_summary.csv"
)


# =============================================================================
# MODEL INPUT CONFIGURATION
# =============================================================================

MAX_SEQUENCE_LENGTH = 60

TITLE_SEQUENCE_LENGTH = 20

NUM_CLASSES = 4

TEXT_SEPARATOR = "[SEP]"


# =============================================================================
# DEFAULT LABEL MAPPING
# =============================================================================

DEFAULT_LABEL_MAPPING = {
    0: "bola",
    1: "global",
    2: "money",
    3: "tekno",
}


# =============================================================================
# DISPLAY LABELS
# =============================================================================

DISPLAY_LABELS = {
    "bola": "Bola",
    "global": "Global",
    "money": "Money",
    "tekno": "Tekno",
    "business": "Business",
    "sci_tech": "Sci/Tech",
    "sports": "Sports",
    "world": "World",
}


# =============================================================================
# LABELS BY DATASET
# =============================================================================

LABELS_BY_DATASET = {
    "Kompas": [
        "bola",
        "global",
        "money",
        "tekno",
    ],

    "AG News": [
        "business",
        "sci_tech",
        "sports",
        "world",
    ],
}


# =============================================================================
# RESEARCH INFORMATION
# =============================================================================

RESEARCH_TITLE = (
    "Analisis Perbandingan Kinerja CNN dan Attention-BiLSTM "
    "pada Klasifikasi Berita Berbahasa Indonesia "
    "Menggunakan Representasi Teks Berbasis Dashboard Streamlit"
)

PRIMARY_DATASET = "Kompas"

BENCHMARK_DATASET = "AG News"

PRIMARY_MODEL = "CNN K2"

COMPARISON_MODEL = "Attention-BiLSTM K2"

PRIMARY_SCENARIO = "K2 — Title + Description"

TOTAL_EXPERIMENTS = 10


# =============================================================================
# EXPERIMENT SCENARIOS
# =============================================================================

SCENARIO_INFORMATION = {
    "K1": {
        "dataset": "Kompas",
        "text_representation": "Title",
        "uses_yake": False,
        "max_sequence_length": 20,
    },

    "K2": {
        "dataset": "Kompas",
        "text_representation": (
            "Title + [SEP] + Description"
        ),
        "uses_yake": False,
        "max_sequence_length": 60,
    },

    "K3": {
        "dataset": "Kompas",
        "text_representation": (
            "Title + [SEP] + Description "
            "+ [SEP] + Keyword YAKE"
        ),
        "uses_yake": True,
        "max_sequence_length": 60,
    },

    "A1": {
        "dataset": "AG News",
        "text_representation": "Title",
        "uses_yake": False,
        "max_sequence_length": 20,
    },

    "A2": {
        "dataset": "AG News",
        "text_representation": (
            "Title + [SEP] + Description"
        ),
        "uses_yake": False,
        "max_sequence_length": 60,
    },
}


# =============================================================================
# YAKE CONFIGURATION
# =============================================================================

YAKE_CONFIGURATION = {
    "language": "id",
    "source_text": "Title + Description",
    "max_ngram_size": 3,
    "top_keywords": 5,
    "deduplication_threshold": 0.9,
    "uses_content": False,
}


# =============================================================================
# FINAL MODEL PERFORMANCE
# =============================================================================

# Nilai berikut menjadi fallback dashboard.
# Sumber utama dashboard tetap file evaluasi final di 9_results/tables.

MODEL_PERFORMANCE = {
    "CNN K2": {
        "model_name":
            "CNN",

        "scenario_code":
            "K2",

        "scenario_name":
            "Title + Description",

        "accuracy":
            0.9690,

        "f1_macro":
            0.969036,

        "log_loss":
            0.091183,

        "test_size":
            1_000,

        "correct_predictions":
            969,

        "incorrect_predictions":
            31,
    },

    "Attention-BiLSTM K2": {
        "model_name":
            "Attention-BiLSTM",

        "scenario_code":
            "K2",

        "scenario_name":
            "Title + Description",

        "accuracy":
            0.9680,

        "f1_macro":
            0.968043,

        "log_loss":
            0.111530,

        "test_size":
            1_000,

        "correct_predictions":
            968,

        "incorrect_predictions":
            32,
    },
}


# =============================================================================
# DATASET INFORMATION
# =============================================================================

DATASET_INFORMATION = {
    "Kompas": {
        "jumlah_data_awal":
            10_000,

        "jumlah_data_setelah_cleaning":
            9_997,

        "jumlah_data_dihapus":
            3,

        "jumlah_kategori":
            4,

        "categories": [
            "Bola",
            "Global",
            "Money",
            "Tekno",
        ],

        "category_distribution": {
            "Bola": 2_500,
            "Global": 2_500,
            "Money": 2_500,
            "Tekno": 2_497,
        },

        "date_start":
            "2026-01-26 09:01:00",

        "date_end":
            "2026-05-25 19:24:00",
    },

    "AG News": {
        "jumlah_data_train_awal":
            120_000,

        "jumlah_data_train_setelah_cleaning":
            119_817,

        "jumlah_data_train_dihapus":
            183,

        "jumlah_data_test_awal":
            7_600,

        "jumlah_data_test_setelah_cleaning":
            7_600,

        "jumlah_data_test_dihapus":
            0,

        "jumlah_kategori":
            4,

        "categories": [
            "Business",
            "Sci/Tech",
            "Sports",
            "World",
        ],

        "train_category_distribution": {
            "Business": 29_947,
            "Sci/Tech": 29_917,
            "Sports": 29_974,
            "World": 29_979,
        },

        "test_category_distribution": {
            "Business": 1_900,
            "Sci/Tech": 1_900,
            "Sports": 1_900,
            "World": 1_900,
        },
    },
}


# =============================================================================
# SHAP CONFIGURATION
# =============================================================================

SHAP_MODEL_NAME = "CNN K2"

SHAP_SCENARIO = "K2 — Title + Description"

SHAP_EXPLANATION_TYPE = "Precomputed Test-Set Explanation"

SPECIAL_TOKENS = {
    "[PAD]",
    "[SEP]",
    "[UNK]",
    "[OOV]",
    "",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_json_file(
    file_path: Path,
) -> dict[str, Any]:
    """
    Membaca file JSON dengan aman.
    """

    if not file_path.exists():
        return {}

    try:
        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if isinstance(data, dict):
            return data

    except (
        OSError,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ):
        return {}

    return {}


def load_label_mapping() -> dict[int, str]:
    """
    Membaca label mapping deployment.

    Jika file tidak tersedia atau tidak valid,
    menggunakan DEFAULT_LABEL_MAPPING.
    """

    mapping_data = load_json_file(
        LABEL_MAPPING_PATH
    )

    if not mapping_data:
        return DEFAULT_LABEL_MAPPING.copy()

    candidate_mapping = (
        mapping_data.get("index_to_label")
        or mapping_data.get("label_mapping")
        or mapping_data
    )

    if not isinstance(candidate_mapping, dict):
        return DEFAULT_LABEL_MAPPING.copy()

    normalized_mapping: dict[int, str] = {}

    for key, value in candidate_mapping.items():
        try:
            normalized_key = int(key)
        except (
            TypeError,
            ValueError,
        ):
            continue

        normalized_mapping[
            normalized_key
        ] = str(value).strip().lower()

    if set(normalized_mapping.keys()) != {
        0,
        1,
        2,
        3,
    }:
        return DEFAULT_LABEL_MAPPING.copy()

    return normalized_mapping


def validate_required_files() -> dict[str, bool]:
    """
    Memvalidasi file inti dashboard.
    """

    required_files = {
        "cnn_model":
            CNN_MODEL_PATH,

        "attention_bilstm_model":
            ATTENTION_BILSTM_MODEL_PATH,

        "vocabulary":
            VOCABULARY_PATH,

        "vectorizer_config":
            VECTORIZER_CONFIG_PATH,

        "label_mapping":
            LABEL_MAPPING_PATH,

        "deployment_config":
            DEPLOYMENT_CONFIG_PATH,

        "kompas_final":
            KOMPAS_FINAL_PATH,

        "ag_news_train_final":
            AG_NEWS_TRAIN_FINAL_PATH,

        "ag_news_test_final":
            AG_NEWS_TEST_FINAL_PATH,
    }

    return {
        name: (
            path.exists()
            and path.is_file()
            and path.stat().st_size > 0
        )
        for name, path
        in required_files.items()
    }


# =============================================================================
# CONFIGURATION TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("STREAMLIT CONFIGURATION TEST")
    print("=" * 80)

    print(f"Project root          : {PROJECT_ROOT}")
    print(f"Streamlit directory   : {STREAMLIT_DIR}")
    print(f"Deployment directory  : {DEPLOYMENT_DIR}")
    print(f"Results directory     : {RESULTS_DIR}")
    print(f"Tables directory      : {TABLES_DIR}")
    print(f"Figures directory     : {FIGURES_DIR}")

    print("\nModel deployment:")

    print(f"CNN model             : {CNN_MODEL_PATH}")
    print(f"Attention-BiLSTM      : {ATTENTION_BILSTM_MODEL_PATH}")
    print(f"Vocabulary            : {VOCABULARY_PATH}")
    print(f"Vectorizer config     : {VECTORIZER_CONFIG_PATH}")
    print(f"Label mapping         : {LABEL_MAPPING_PATH}")
    print(f"Sequence length       : {MAX_SEQUENCE_LENGTH}")

    print("\nDataset final:")

    print(f"Kompas                : {KOMPAS_FINAL_PATH}")
    print(f"AG News Train         : {AG_NEWS_TRAIN_FINAL_PATH}")
    print(f"AG News Test          : {AG_NEWS_TEST_FINAL_PATH}")

    print("\nLabel mapping:")

    resolved_mapping = load_label_mapping()

    for index, label in sorted(
        resolved_mapping.items()
    ):
        print(
            f"{index} -> {label}"
        )

    print("\nValidasi file inti:")

    validation = validate_required_files()

    for name, status in validation.items():
        print(
            f"{name:<30}: "
            f"{'LULUS' if status else 'TIDAK DITEMUKAN'}"
        )

    passed = sum(
        validation.values()
    )

    total = len(
        validation
    )

    print(
        f"\nRingkasan validasi: "
        f"{passed}/{total} file tersedia."
    )

    if passed == total:
        print(
            "Validasi konfigurasi: LULUS"
        )
    else:
        print(
            "Validasi konfigurasi: PERIKSA FILE YANG BELUM TERSEDIA"
        )
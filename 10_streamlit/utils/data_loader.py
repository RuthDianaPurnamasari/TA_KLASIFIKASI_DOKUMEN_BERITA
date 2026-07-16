# =============================================================================
# STREAMLIT DATA LOADER
# =============================================================================
# File:
# 10_streamlit/utils/data_loader.py
#
# Fungsi:
# 1. Membaca tabel hasil penelitian dari folder 9_results/tables.
# 2. Menyediakan path grafik dari folder 9_results/figures.
# 3. Menghindari penulisan path berulang pada setiap halaman dashboard.
# 4. Mengembalikan DataFrame kosong jika file belum ditemukan,
#    sehingga dashboard tidak langsung error.
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================

CURRENT_FILE = Path(
    __file__
).resolve()

STREAMLIT_DIR = (
    CURRENT_FILE
    .parents[1]
)

PROJECT_ROOT = (
    CURRENT_FILE
    .parents[2]
)


# Menambahkan folder 10_streamlit ke sys.path
# agar config.py khusus dashboard dapat di-import.
if str(STREAMLIT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(STREAMLIT_DIR),
    )


# =============================================================================
# IMPORT DASHBOARD CONFIGURATION
# =============================================================================

from config import (
    FIGURES_DIR,
    RESULTS_DIR,
    TABLES_DIR,
)


# =============================================================================
# GENERAL CSV LOADER
# =============================================================================

@st.cache_data(
    show_spinner=False,
)
def read_csv_safe(
    file_path: Path,
) -> pd.DataFrame:
    """
    Membaca file CSV dengan aman.

    Jika file tidak ditemukan:
    - tidak menyebabkan dashboard crash;
    - mengembalikan DataFrame kosong.

    Parameters
    ----------
    file_path:
        Lokasi file CSV.

    Returns
    -------
    pandas.DataFrame
        Isi CSV atau DataFrame kosong.
    """

    file_path = Path(
        file_path
    )

    if not file_path.exists():
        return pd.DataFrame()

    try:
        dataframe = pd.read_csv(
            file_path,
            encoding="utf-8-sig",
        )

    except UnicodeDecodeError:
        dataframe = pd.read_csv(
            file_path,
            encoding="utf-8",
        )

    except Exception:
        return pd.DataFrame()

    return dataframe


def find_first_existing_file(
    candidates: list[Path],
) -> Path | None:
    """
    Mencari file pertama yang tersedia
    dari beberapa kemungkinan path.

    Fungsi ini berguna karena nama file hasil evaluasi
    dapat berbeda selama pengembangan project.
    """

    for candidate in candidates:
        candidate = Path(
            candidate
        )

        if candidate.exists():
            return candidate

    return None


# =============================================================================
# EVALUATION TABLES
# =============================================================================

@st.cache_data(
    show_spinner=False,
)
def load_test_evaluation() -> pd.DataFrame:
    """
    Membaca ringkasan evaluasi final test set.

    Script evaluasi dapat menghasilkan nama file berbeda.
    Fungsi ini mencoba beberapa kemungkinan nama file.
    """

    candidates = [
        TABLES_DIR
        / "test_evaluation_summary.csv",

        TABLES_DIR
        / "model_evaluation_summary.csv",

        TABLES_DIR
        / "evaluation_summary.csv",

        RESULTS_DIR
        / "metrics"
        / "model_test_metrics.csv",

        RESULTS_DIR
        / "metrics"
        / "test_metrics.csv",

        RESULTS_DIR
        / "metrics"
        / "all_model_metrics.csv",
    ]

    selected_path = (
        find_first_existing_file(
            candidates
        )
    )

    if selected_path is None:
        return pd.DataFrame()

    return read_csv_safe(
        selected_path
    )


@st.cache_data(
    show_spinner=False,
)
def load_model_comparison() -> pd.DataFrame:
    """
    Membaca tabel perbandingan CNN
    dan Attention-BiLSTM.
    """

    file_path = (
        TABLES_DIR
        / "model_comparison.csv"
    )

    return read_csv_safe(
        file_path
    )


@st.cache_data(
    show_spinner=False,
)
def load_scenario_comparison() -> pd.DataFrame:
    """
    Membaca tabel perbandingan skenario:
    K1, K2, K3, A1, dan A2.
    """

    file_path = (
        TABLES_DIR
        / "scenario_comparison.csv"
    )

    return read_csv_safe(
        file_path
    )


@st.cache_data(
    show_spinner=False,
)
def load_description_contribution() -> pd.DataFrame:
    """
    Membaca hasil analisis kontribusi Description.
    """

    file_path = (
        TABLES_DIR
        / "description_contribution_analysis.csv"
    )

    return read_csv_safe(
        file_path
    )


@st.cache_data(
    show_spinner=False,
)
def load_yake_contribution() -> pd.DataFrame:
    """
    Membaca hasil analisis kontribusi keyword YAKE.
    """

    file_path = (
        TABLES_DIR
        / "yake_contribution_analysis.csv"
    )

    return read_csv_safe(
        file_path
    )


@st.cache_data(
    show_spinner=False,
)
def load_best_model_summary() -> pd.DataFrame:
    """
    Membaca ringkasan model terbaik
    pada Kompas dan AG News.
    """

    file_path = (
        TABLES_DIR
        / "best_model_summary.csv"
    )

    return read_csv_safe(
        file_path
    )


@st.cache_data(
    show_spinner=False,
)
def load_inference_efficiency() -> pd.DataFrame:
    """
    Membaca analisis waktu inferensi model.
    """

    file_path = (
        TABLES_DIR
        / "inference_efficiency_analysis.csv"
    )

    return read_csv_safe(
        file_path
    )


@st.cache_data(
    show_spinner=False,
)
def load_final_research_findings() -> pd.DataFrame:
    """
    Membaca ringkasan temuan akhir penelitian.
    """

    file_path = (
        TABLES_DIR
        / "final_research_findings.csv"
    )

    return read_csv_safe(
        file_path
    )


# =============================================================================
# TRAINING AND OVERFITTING TABLES
# =============================================================================

@st.cache_data(
    show_spinner=False,
)
def load_training_curve_report() -> pd.DataFrame:
    """
    Membaca ringkasan training curve
    seluruh eksperimen.
    """

    file_path = (
        TABLES_DIR
        / "training_curve_report.csv"
    )

    return read_csv_safe(
        file_path
    )


@st.cache_data(
    show_spinner=False,
)
def load_overfitting_analysis() -> pd.DataFrame:
    """
    Membaca hasil analisis overfitting.
    """

    file_path = (
        TABLES_DIR
        / "overfitting_analysis.csv"
    )

    return read_csv_safe(
        file_path
    )


# =============================================================================
# CONFUSION MATRIX TABLES
# =============================================================================

@st.cache_data(
    show_spinner=False,
)
def load_confusion_matrix_summary() -> pd.DataFrame:
    """
    Membaca ringkasan performa kelas
    dari confusion matrix.
    """

    file_path = (
        TABLES_DIR
        / "confusion_matrix_summary.csv"
    )

    return read_csv_safe(
        file_path
    )


@st.cache_data(
    show_spinner=False,
)
def load_misclassification_analysis() -> pd.DataFrame:
    """
    Membaca analisis kesalahan klasifikasi.
    """

    file_path = (
        TABLES_DIR
        / "misclassification_analysis.csv"
    )

    return read_csv_safe(
        file_path
    )


# =============================================================================
# SHAP TABLES
# =============================================================================

@st.cache_data(
    show_spinner=False,
)
def load_global_shap() -> pd.DataFrame:
    """
    Membaca global token importance CNN K2.
    """

    file_path = (
        TABLES_DIR
        / "shap"
        / "cnn_k2_global_token_importance.csv"
    )

    return read_csv_safe(
        file_path
    )


@st.cache_data(
    show_spinner=False,
)
def load_global_shap_by_class() -> pd.DataFrame:
    """
    Membaca token importance per kelas.
    """

    file_path = (
        TABLES_DIR
        / "shap"
        / (
            "cnn_k2_global_"
            "token_importance_by_class.csv"
        )
    )

    return read_csv_safe(
        file_path
    )


@st.cache_data(
    show_spinner=False,
)
def load_shap_sample_summary() -> pd.DataFrame:
    """
    Membaca ringkasan 100 sampel
    yang digunakan pada global SHAP.
    """

    file_path = (
        TABLES_DIR
        / "shap"
        / "cnn_k2_shap_sample_summary.csv"
    )

    return read_csv_safe(
        file_path
    )


@st.cache_data(
    show_spinner=False,
)
def load_local_shap_summary() -> pd.DataFrame:
    """
    Membaca ringkasan delapan sampel local SHAP.
    """

    file_path = (
        TABLES_DIR
        / "shap"
        / "local"
        / "cnn_k2_local_shap_summary.csv"
    )

    return read_csv_safe(
        file_path
    )


@st.cache_data(
    show_spinner=False,
)
def load_local_token_contributions() -> pd.DataFrame:
    """
    Membaca kontribusi token
    pada setiap sampel local SHAP.
    """

    file_path = (
        TABLES_DIR
        / "shap"
        / "local"
        / (
            "cnn_k2_local_"
            "token_contributions.csv"
        )
    )

    return read_csv_safe(
        file_path
    )


@st.cache_data(
    show_spinner=False,
)
def load_waterfall_summary() -> pd.DataFrame:
    """
    Membaca ringkasan waterfall-style plot.
    """

    file_path = (
        TABLES_DIR
        / "shap"
        / "cnn_k2_waterfall_summary.csv"
    )

    return read_csv_safe(
        file_path
    )


# =============================================================================
# FIGURE PATH FUNCTIONS
# =============================================================================

def get_training_curve_path(
    experiment_name: str,
) -> Path:
    """
    Menghasilkan path training curve
    berdasarkan nama eksperimen.
    """

    return (
        FIGURES_DIR
        / "training_curves"
        / (
            f"{experiment_name}"
            "_training_curve.png"
        )
    )


def get_confusion_matrix_path(
    experiment_name: str,
    normalized: bool = False,
) -> Path:
    """
    Menghasilkan path confusion matrix.

    Parameters
    ----------
    experiment_name:
        Contoh: cnn_k2.

    normalized:
        False -> confusion matrix jumlah.
        True  -> confusion matrix normalized.
    """

    if normalized:
        filename = (
            f"{experiment_name}"
            "_confusion_matrix_normalized.png"
        )

    else:
        filename = (
            f"{experiment_name}"
            "_confusion_matrix.png"
        )

    return (
        FIGURES_DIR
        / "confusion_matrices"
        / filename
    )


def get_comparative_figure_path(
    figure_name: str,
) -> Path:
    """
    Menghasilkan path grafik comparative analysis.

    Contoh figure_name:
    - accuracy_comparison.png
    - f1_macro_comparison.png
    - description_contribution.png
    - yake_contribution.png
    """

    return (
        FIGURES_DIR
        / "comparative_analysis"
        / figure_name
    )


def get_global_shap_figure_path(
    figure_name: str,
) -> Path:
    """
    Menghasilkan path grafik global SHAP.
    """

    return (
        FIGURES_DIR
        / "shap"
        / "global"
        / figure_name
    )


def get_waterfall_figure_path(
    document_id: str,
    selection_type: str,
    actual_label: str,
    predicted_label: str,
) -> Path:
    """
    Menghasilkan path waterfall plot
    berdasarkan informasi sampel.
    """

    filename = (
        f"{document_id}_"
        f"{selection_type}_"
        f"actual_{actual_label}_"
        f"pred_{predicted_label}_"
        "waterfall.png"
    )

    return (
        FIGURES_DIR
        / "shap"
        / "waterfall"
        / filename
    )


# =============================================================================
# TERMINAL TEST
# =============================================================================

def print_dataframe_status(
    label: str,
    dataframe: pd.DataFrame,
) -> None:
    """
    Menampilkan status DataFrame saat pengujian.
    """

    status = (
        "TERSEDIA"
        if not dataframe.empty
        else "TIDAK DITEMUKAN"
    )

    print(
        f"{label:<35}: "
        f"{status}"
    )

    if not dataframe.empty:
        print(
            f"{'Jumlah baris':<35}: "
            f"{len(dataframe):,}"
        )


def main() -> None:
    """
    Menguji data loader dari terminal.
    """

    print("=" * 80)
    print(
        "STREAMLIT DATA LOADER TEST"
    )
    print("=" * 80)

    test_evaluation = (
        load_test_evaluation()
    )

    model_comparison = (
        load_model_comparison()
    )

    scenario_comparison = (
        load_scenario_comparison()
    )

    description_analysis = (
        load_description_contribution()
    )

    yake_analysis = (
        load_yake_contribution()
    )

    best_model = (
        load_best_model_summary()
    )

    global_shap = (
        load_global_shap()
    )

    local_shap = (
        load_local_shap_summary()
    )

    waterfall = (
        load_waterfall_summary()
    )

    print("\nStatus tabel:")

    print_dataframe_status(
        "Test evaluation",
        test_evaluation,
    )

    print_dataframe_status(
        "Model comparison",
        model_comparison,
    )

    print_dataframe_status(
        "Scenario comparison",
        scenario_comparison,
    )

    print_dataframe_status(
        "Description contribution",
        description_analysis,
    )

    print_dataframe_status(
        "YAKE contribution",
        yake_analysis,
    )

    print_dataframe_status(
        "Best model summary",
        best_model,
    )

    print_dataframe_status(
        "Global SHAP",
        global_shap,
    )

    print_dataframe_status(
        "Local SHAP",
        local_shap,
    )

    print_dataframe_status(
        "Waterfall summary",
        waterfall,
    )

    print("\nStatus grafik:")

    sample_training_curve = (
        get_training_curve_path(
            "cnn_k2"
        )
    )

    sample_confusion_matrix = (
        get_confusion_matrix_path(
            "cnn_k2",
            normalized=False,
        )
    )

    sample_global_shap = (
        get_global_shap_figure_path(
            "cnn_k2_global_"
            "shap_top_tokens.png"
        )
    )

    print(
        f"Training curve cnn_k2         : "
        f"{sample_training_curve.exists()}"
    )

    print(
        f"Confusion matrix cnn_k2       : "
        f"{sample_confusion_matrix.exists()}"
    )

    print(
        f"Global SHAP figure            : "
        f"{sample_global_shap.exists()}"
    )

    print("\n" + "=" * 80)
    print(
        "Data loader test selesai."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
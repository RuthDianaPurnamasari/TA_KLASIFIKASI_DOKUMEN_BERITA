# =============================================================================
# STREAMLIT DATA LOADER
# =============================================================================
# File:
# 10_streamlit/utils/data_loader.py
#
# Fungsi:
# 1. Membaca tabel hasil penelitian dari folder 9_results.
# 2. Menyediakan path grafik hasil penelitian.
# 3. Menghindari penulisan path berulang pada setiap halaman dashboard.
# 4. Mengembalikan DataFrame kosong jika file belum tersedia.
# 5. Memperbarui cache otomatis ketika file berubah.
# 6. Menangani path Windows dan Linux secara lebih aman.
# =============================================================================

from __future__ import annotations

import importlib.util
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st


# =============================================================================
# LOGGING
# =============================================================================

LOGGER = logging.getLogger(
    "streamlit_data_loader"
)

if not LOGGER.handlers:
    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    handler.setFormatter(
        formatter
    )

    LOGGER.addHandler(
        handler
    )

LOGGER.setLevel(
    logging.INFO
)


# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================

CURRENT_FILE = Path(
    __file__
).resolve()

UTILS_DIR = CURRENT_FILE.parent

STREAMLIT_DIR = CURRENT_FILE.parents[1]

PROJECT_ROOT = CURRENT_FILE.parents[2]

CONFIG_PATH = (
    STREAMLIT_DIR
    / "config.py"
)


# =============================================================================
# LOAD DASHBOARD CONFIGURATION
# =============================================================================

def load_dashboard_config_module() -> Any:
    """
    Memuat config.py menggunakan path absolut.

    Pendekatan ini digunakan agar tidak terjadi bentrok
    dengan package lain yang juga bernama config.
    """

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            "File konfigurasi Streamlit tidak ditemukan:\n"
            f"{CONFIG_PATH}"
        )

    module_name = (
        "ta_streamlit_dashboard_config"
    )

    if module_name in sys.modules:
        return sys.modules[
            module_name
        ]

    specification = (
        importlib.util.spec_from_file_location(
            module_name,
            CONFIG_PATH,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise ImportError(
            "Tidak dapat membuat module specification "
            f"untuk {CONFIG_PATH}."
        )

    module = (
        importlib.util.module_from_spec(
            specification
        )
    )

    sys.modules[
        module_name
    ] = module

    specification.loader.exec_module(
        module
    )

    required_attributes = {
        "FIGURES_DIR",
        "RESULTS_DIR",
        "TABLES_DIR",
    }

    missing_attributes = [
        attribute
        for attribute
        in required_attributes
        if not hasattr(
            module,
            attribute,
        )
    ]

    if missing_attributes:
        raise AttributeError(
            "config.py tidak memiliki konfigurasi "
            "yang dibutuhkan.\n"
            f"Atribut hilang: {missing_attributes}"
        )

    return module


DASHBOARD_CONFIG = (
    load_dashboard_config_module()
)

FIGURES_DIR = Path(
    DASHBOARD_CONFIG.FIGURES_DIR
).resolve()

RESULTS_DIR = Path(
    DASHBOARD_CONFIG.RESULTS_DIR
).resolve()

TABLES_DIR = Path(
    DASHBOARD_CONFIG.TABLES_DIR
).resolve()


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def safe_filename(
    text: Any,
) -> str:
    """
    Membersihkan teks agar aman digunakan sebagai nama file.

    Fungsi ini dibuat konsisten dengan nama file
    pada tahap SHAP waterfall.
    """

    value = str(
        text
    ).strip()

    value = re.sub(
        r'[<>:"/\\|?*]+',
        "_",
        value,
    )

    value = re.sub(
        r"\s+",
        "_",
        value,
    )

    value = re.sub(
        r"_+",
        "_",
        value,
    )

    value = value.strip(
        "._"
    )

    return (
        value
        if value
        else "unknown"
    )


def is_valid_file(
    file_path: Path,
) -> bool:
    """
    Memeriksa apakah path merupakan file valid dan tidak kosong.
    """

    path = Path(
        file_path
    )

    return (
        path.exists()
        and path.is_file()
        and path.stat().st_size > 0
    )


def find_first_existing_file(
    candidates: Iterable[Path],
) -> Path | None:
    """
    Mencari file valid pertama dari sejumlah kandidat.
    """

    for candidate in candidates:
        candidate_path = Path(
            candidate
        )

        if is_valid_file(
            candidate_path
        ):
            return candidate_path

    return None


def extract_filename_from_stored_path(
    stored_path: Any,
) -> str:
    """
    Mengambil nama file dari path yang tersimpan di CSV.

    Mendukung:
    - Path Windows dengan backslash.
    - Path Linux dengan slash.
    - Path relatif.
    """

    if stored_path is None:
        return ""

    normalized = (
        str(stored_path)
        .strip()
        .replace(
            "\\",
            "/",
        )
    )

    if not normalized:
        return ""

    return Path(
        normalized
    ).name


def resolve_saved_file_path(
    stored_path: Any,
    fallback_directory: Path,
) -> Path | None:
    """
    Mengubah path lama dari CSV menjadi path lokal proyek saat ini.

    Contoh path lama:
    C:\\TA_KLASIFIKASI_DOKUMEN_BERITA\\9_results\\figures\\...

    Fungsi hanya mengambil nama file, lalu menggabungkannya
    dengan folder hasil proyek yang sedang digunakan.
    """

    filename = extract_filename_from_stored_path(
        stored_path
    )

    if not filename:
        return None

    candidate = (
        Path(
            fallback_directory
        )
        / filename
    )

    return candidate


# =============================================================================
# CSV LOADER
# =============================================================================

@st.cache_data(
    show_spinner=False,
)
def _read_csv_cached(
    file_path_string: str,
    modified_time_ns: int,
    file_size_bytes: int,
) -> pd.DataFrame:
    """
    Fungsi internal pembacaan CSV yang menggunakan cache.

    modified_time_ns dan file_size_bytes menjadi bagian
    dari cache key agar cache diperbarui ketika file berubah.
    """

    del modified_time_ns
    del file_size_bytes

    file_path = Path(
        file_path_string
    )

    encoding_candidates = [
        "utf-8-sig",
        "utf-8",
        "latin-1",
    ]

    last_error: Exception | None = None

    for encoding in encoding_candidates:
        try:
            dataframe = pd.read_csv(
                file_path,
                encoding=encoding,
            )

            return dataframe

        except UnicodeDecodeError as error:
            last_error = error
            continue

        except Exception as error:
            last_error = error
            break

    if last_error is not None:
        raise last_error

    return pd.DataFrame()


def read_csv_safe(
    file_path: Path,
) -> pd.DataFrame:
    """
    Membaca CSV dengan aman.

    Jika file tidak ditemukan atau gagal dibaca,
    fungsi mengembalikan DataFrame kosong dan
    menulis informasi error ke terminal.
    """

    path = Path(
        file_path
    )

    if not is_valid_file(
        path
    ):
        LOGGER.warning(
            "CSV tidak ditemukan atau kosong: %s",
            path,
        )

        return pd.DataFrame()

    try:
        stat = path.stat()

        dataframe = _read_csv_cached(
            file_path_string=str(
                path.resolve()
            ),
            modified_time_ns=int(
                stat.st_mtime_ns
            ),
            file_size_bytes=int(
                stat.st_size
            ),
        )

    except Exception as error:
        LOGGER.exception(
            "Gagal membaca CSV %s: %s",
            path,
            error,
        )

        return pd.DataFrame()

    return dataframe.copy()


# =============================================================================
# JSON LOADER
# =============================================================================

@st.cache_data(
    show_spinner=False,
)
def _read_json_cached(
    file_path_string: str,
    modified_time_ns: int,
    file_size_bytes: int,
) -> dict[str, Any]:
    """
    Fungsi internal pembacaan JSON dengan cache.
    """

    del modified_time_ns
    del file_size_bytes

    path = Path(
        file_path_string
    )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(
            file
        )

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            "Isi JSON harus berupa dictionary."
        )

    return data


def read_json_safe(
    file_path: Path,
) -> dict[str, Any]:
    """
    Membaca JSON dengan aman.

    Jika file tidak tersedia atau rusak,
    fungsi mengembalikan dictionary kosong.
    """

    path = Path(
        file_path
    )

    if not is_valid_file(
        path
    ):
        LOGGER.warning(
            "JSON tidak ditemukan atau kosong: %s",
            path,
        )

        return {}

    try:
        stat = path.stat()

        result = _read_json_cached(
            file_path_string=str(
                path.resolve()
            ),
            modified_time_ns=int(
                stat.st_mtime_ns
            ),
            file_size_bytes=int(
                stat.st_size
            ),
        )

    except Exception as error:
        LOGGER.exception(
            "Gagal membaca JSON %s: %s",
            path,
            error,
        )

        return {}

    return dict(
        result
    )


# =============================================================================
# EVALUATION TABLES
# =============================================================================

def load_test_evaluation() -> pd.DataFrame:
    """
    Membaca metrik evaluasi final test set seluruh model.
    """

    candidates = [
        RESULTS_DIR
        / "metrics"
        / "model_test_metrics.csv",

        TABLES_DIR
        / "test_evaluation_summary.csv",

        TABLES_DIR
        / "model_evaluation_summary.csv",

        TABLES_DIR
        / "evaluation_summary.csv",

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


def load_model_comparison() -> pd.DataFrame:
    """
    Membaca tabel perbandingan CNN dan Attention-BiLSTM.
    """

    candidates = [
        TABLES_DIR
        / "model_comparison.csv",

        TABLES_DIR
        / "comparative_analysis"
        / "model_comparison.csv",
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


def load_scenario_comparison() -> pd.DataFrame:
    """
    Membaca tabel perbandingan skenario K1, K2, K3, A1, dan A2.
    """

    candidates = [
        TABLES_DIR
        / "scenario_comparison.csv",

        TABLES_DIR
        / "comparative_analysis"
        / "scenario_comparison.csv",
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


def load_description_contribution() -> pd.DataFrame:
    """
    Membaca analisis kontribusi Description.
    """

    candidates = [
        TABLES_DIR
        / "description_contribution_analysis.csv",

        TABLES_DIR
        / "comparative_analysis"
        / "description_contribution_analysis.csv",
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


def load_yake_contribution() -> pd.DataFrame:
    """
    Membaca analisis kontribusi keyword YAKE.
    """

    candidates = [
        TABLES_DIR
        / "yake_contribution_analysis.csv",

        TABLES_DIR
        / "comparative_analysis"
        / "yake_contribution_analysis.csv",
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


def load_best_model_summary() -> pd.DataFrame:
    """
    Membaca ringkasan model terbaik pada Kompas dan AG News.
    """

    candidates = [
        TABLES_DIR
        / "best_model_summary.csv",

        TABLES_DIR
        / "comparative_analysis"
        / "best_model_summary.csv",
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


def load_inference_efficiency() -> pd.DataFrame:
    """
    Membaca analisis waktu inference model.
    """

    candidates = [
        TABLES_DIR
        / "inference_efficiency_analysis.csv",

        TABLES_DIR
        / "comparative_analysis"
        / "inference_efficiency_analysis.csv",
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


def load_final_research_findings() -> pd.DataFrame:
    """
    Membaca ringkasan temuan akhir penelitian.
    """

    candidates = [
        TABLES_DIR
        / "final_research_findings.csv",

        TABLES_DIR
        / "comparative_analysis"
        / "final_research_findings.csv",
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


# =============================================================================
# TRAINING AND OVERFITTING TABLES
# =============================================================================

def load_training_curve_report() -> pd.DataFrame:
    """
    Membaca ringkasan training curve seluruh eksperimen.
    """

    candidates = [
        TABLES_DIR
        / "training_curve_report.csv",

        TABLES_DIR
        / "training_curves"
        / "training_curve_report.csv",
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


def load_overfitting_analysis() -> pd.DataFrame:
    """
    Membaca hasil analisis overfitting.
    """

    candidates = [
        TABLES_DIR
        / "overfitting_analysis.csv",

        TABLES_DIR
        / "training_curves"
        / "overfitting_analysis.csv",
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


# =============================================================================
# CONFUSION MATRIX TABLES
# =============================================================================

def load_confusion_matrix_summary() -> pd.DataFrame:
    """
    Membaca ringkasan performa kelas dari confusion matrix.
    """

    candidates = [
        TABLES_DIR
        / "confusion_matrix_summary.csv",

        TABLES_DIR
        / "confusion_matrices"
        / "confusion_matrix_summary.csv",
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


def load_misclassification_analysis() -> pd.DataFrame:
    """
    Membaca analisis kesalahan klasifikasi.
    """

    candidates = [
        TABLES_DIR
        / "misclassification_analysis.csv",

        TABLES_DIR
        / "confusion_matrices"
        / "misclassification_analysis.csv",
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


# =============================================================================
# SHAP TABLES
# =============================================================================

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


def load_global_shap_by_class() -> pd.DataFrame:
    """
    Membaca token importance global per kelas.
    """

    file_path = (
        TABLES_DIR
        / "shap"
        / "cnn_k2_global_token_importance_by_class.csv"
    )

    return read_csv_safe(
        file_path
    )


def load_shap_sample_summary() -> pd.DataFrame:
    """
    Membaca ringkasan sampel global SHAP.
    """

    file_path = (
        TABLES_DIR
        / "shap"
        / "cnn_k2_shap_sample_summary.csv"
    )

    return read_csv_safe(
        file_path
    )


def load_local_shap_summary() -> pd.DataFrame:
    """
    Membaca ringkasan sampel local SHAP
    yang berhasil dipilih pada tahap 7.2.
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


def load_local_token_contributions() -> pd.DataFrame:
    """
    Membaca kontribusi token agregat pada setiap sampel local SHAP.
    """

    file_path = (
        TABLES_DIR
        / "shap"
        / "local"
        / "cnn_k2_local_token_contributions.csv"
    )

    return read_csv_safe(
        file_path
    )


def load_local_token_position_contributions() -> pd.DataFrame:
    """
    Membaca kontribusi SHAP pada setiap posisi token.
    """

    file_path = (
        TABLES_DIR
        / "shap"
        / "local"
        / "cnn_k2_local_token_position_contributions.csv"
    )

    return read_csv_safe(
        file_path
    )


def load_waterfall_summary() -> pd.DataFrame:
    """
    Membaca ringkasan waterfall SHAP CNN K2.
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
# DEPLOYMENT TABLES AND CONFIGURATION
# =============================================================================

def load_inference_pipeline_report() -> dict[str, Any]:
    """
    Membaca laporan pengujian inference pipeline.
    """

    file_path = (
        TABLES_DIR
        / "deployment"
        / "inference_pipeline_test.json"
    )

    return read_json_safe(
        file_path
    )


def load_deployment_config() -> dict[str, Any]:
    """
    Membaca konfigurasi deployment model.
    """

    file_path = (
        PROJECT_ROOT
        / "8_save_models"
        / "deployment"
        / "deployment_config.json"
    )

    return read_json_safe(
        file_path
    )


def load_deployment_report() -> dict[str, Any]:
    """
    Membaca laporan penyiapan artefak deployment.
    """

    file_path = (
        PROJECT_ROOT
        / "8_save_models"
        / "deployment"
        / "deployment_report.json"
    )

    return read_json_safe(
        file_path
    )


# =============================================================================
# FIGURE PATH FUNCTIONS
# =============================================================================

def get_training_curve_path(
    experiment_name: str,
) -> Path:
    """
    Menghasilkan path training curve berdasarkan eksperimen.
    """

    filename = (
        f"{safe_filename(experiment_name)}"
        "_training_curve.png"
    )

    return (
        FIGURES_DIR
        / "training_curves"
        / filename
    )


def get_training_accuracy_path(
    experiment_name: str,
) -> Path:
    """
    Menghasilkan path grafik accuracy training.
    """

    filename = (
        f"{safe_filename(experiment_name)}"
        "_accuracy_curve.png"
    )

    return (
        FIGURES_DIR
        / "training_curves"
        / filename
    )


def get_training_loss_path(
    experiment_name: str,
) -> Path:
    """
    Menghasilkan path grafik loss training.
    """

    filename = (
        f"{safe_filename(experiment_name)}"
        "_loss_curve.png"
    )

    return (
        FIGURES_DIR
        / "training_curves"
        / filename
    )


def get_confusion_matrix_path(
    experiment_name: str,
    normalized: bool = False,
) -> Path:
    """
    Menghasilkan path confusion matrix.

    normalized=False:
        Confusion matrix dalam jumlah data.

    normalized=True:
        Confusion matrix dalam proporsi atau persentase.
    """

    safe_experiment_name = safe_filename(
        experiment_name
    )

    if normalized:
        filename = (
            f"{safe_experiment_name}"
            "_confusion_matrix_normalized.png"
        )

    else:
        filename = (
            f"{safe_experiment_name}"
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
    """

    return (
        FIGURES_DIR
        / "comparative_analysis"
        / Path(
            figure_name
        ).name
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
        / Path(
            figure_name
        ).name
    )


def get_local_shap_figure_path(
    figure_name: str,
) -> Path:
    """
    Menghasilkan path grafik local SHAP.
    """

    return (
        FIGURES_DIR
        / "shap"
        / "local"
        / Path(
            figure_name
        ).name
    )


def get_waterfall_figure_path(
    document_id: str,
    selection_type: str,
    actual_label: str,
    predicted_label: str,
) -> Path:
    """
    Menghasilkan path waterfall plot berdasarkan identitas sampel.
    """

    filename = (
        f"{safe_filename(document_id)}_"
        f"{safe_filename(selection_type)}_"
        f"actual_{safe_filename(actual_label)}_"
        f"pred_{safe_filename(predicted_label)}_"
        "waterfall.png"
    )

    return (
        FIGURES_DIR
        / "shap"
        / "waterfall"
        / filename
    )


def resolve_waterfall_figure_from_row(
    row: pd.Series | dict[str, Any],
) -> Path | None:
    """
    Menentukan path waterfall berdasarkan satu baris tabel summary.

    Prioritas:
    1. Menggunakan nama file dari kolom figure_path.
    2. Membangun nama file dari identitas sampel.
    """

    if isinstance(
        row,
        pd.Series,
    ):
        row_data = row.to_dict()
    else:
        row_data = dict(
            row
        )

    waterfall_directory = (
        FIGURES_DIR
        / "shap"
        / "waterfall"
    )

    stored_figure_path = (
        row_data.get(
            "figure_path"
        )
    )

    if stored_figure_path:
        resolved = resolve_saved_file_path(
            stored_path=stored_figure_path,
            fallback_directory=waterfall_directory,
        )

        if (
            resolved is not None
            and resolved.exists()
        ):
            return resolved

    required_fields = [
        "document_id",
        "selection_type",
        "actual_label",
        "predicted_label",
    ]

    if all(
        row_data.get(
            field
        ) is not None
        for field
        in required_fields
    ):
        fallback = get_waterfall_figure_path(
            document_id=str(
                row_data[
                    "document_id"
                ]
            ),
            selection_type=str(
                row_data[
                    "selection_type"
                ]
            ),
            actual_label=str(
                row_data[
                    "actual_label"
                ]
            ),
            predicted_label=str(
                row_data[
                    "predicted_label"
                ]
            ),
        )

        return fallback

    return None


def find_figure_by_keywords(
    directory: Path,
    keywords: Iterable[str],
    suffix: str = ".png",
) -> Path | None:
    """
    Mencari grafik berdasarkan beberapa keyword pada nama file.
    """

    figure_directory = Path(
        directory
    )

    if not figure_directory.exists():
        return None

    normalized_keywords = [
        str(keyword)
        .strip()
        .lower()

        for keyword
        in keywords
        if str(keyword).strip()
    ]

    candidates = sorted(
        figure_directory.glob(
            f"*{suffix}"
        )
    )

    for candidate in candidates:
        filename_lower = (
            candidate.name.lower()
        )

        if all(
            keyword
            in filename_lower
            for keyword
            in normalized_keywords
        ):
            return candidate

    return None


# =============================================================================
# DASHBOARD DISPLAY HELPERS
# =============================================================================

def dataframe_is_available(
    dataframe: pd.DataFrame | None,
) -> bool:
    """
    Memeriksa apakah DataFrame tersedia dan tidak kosong.
    """

    return (
        dataframe is not None
        and isinstance(
            dataframe,
            pd.DataFrame,
        )
        and not dataframe.empty
    )


def display_missing_data_warning(
    data_name: str,
    expected_path: Path | None = None,
) -> None:
    """
    Menampilkan peringatan pada dashboard jika data tidak tersedia.
    """

    message = (
        f"Data **{data_name}** belum tersedia."
    )

    if expected_path is not None:
        message += (
            "\n\nPath yang diperiksa:\n"
            f"`{expected_path}`"
        )

    st.warning(
        message
    )


def display_image_safe(
    image_path: Path | None,
    caption: str | None = None,
    use_container_width: bool = True,
) -> bool:
    """
    Menampilkan gambar jika file tersedia.

    Returns
    -------
    bool
        True jika gambar berhasil ditampilkan.
    """

    if image_path is None:
        st.warning(
            "Path gambar tidak tersedia."
        )

        return False

    path = Path(
        image_path
    )

    if not is_valid_file(
        path
    ):
        st.warning(
            "Gambar belum tersedia:\n"
            f"`{path}`"
        )

        return False

    st.image(
        str(
            path
        ),
        caption=caption,
        use_container_width=use_container_width,
    )

    return True


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

def clear_data_loader_cache() -> None:
    """
    Menghapus cache data loader.

    Dapat dipanggil dari tombol refresh pada dashboard.
    """

    _read_csv_cached.clear()
    _read_json_cached.clear()


# =============================================================================
# TERMINAL TEST HELPERS
# =============================================================================

def print_dataframe_status(
    label: str,
    dataframe: pd.DataFrame,
) -> None:
    """
    Menampilkan status DataFrame saat pengujian terminal.
    """

    status = (
        "TERSEDIA"
        if dataframe_is_available(
            dataframe
        )
        else "TIDAK DITEMUKAN"
    )

    print(
        f"{label:<35}: {status}"
    )

    if dataframe_is_available(
        dataframe
    ):
        print(
            f"{'Jumlah baris':<35}: "
            f"{len(dataframe):,}"
        )

        print(
            f"{'Jumlah kolom':<35}: "
            f"{len(dataframe.columns):,}"
        )


def print_json_status(
    label: str,
    data: dict[str, Any],
) -> None:
    """
    Menampilkan status JSON saat pengujian terminal.
    """

    status = (
        "TERSEDIA"
        if data
        else "TIDAK DITEMUKAN"
    )

    print(
        f"{label:<35}: {status}"
    )

    if data:
        print(
            f"{'Jumlah key':<35}: "
            f"{len(data):,}"
        )


def print_file_status(
    label: str,
    file_path: Path | None,
) -> None:
    """
    Menampilkan status file saat pengujian terminal.
    """

    exists = (
        file_path is not None
        and is_valid_file(
            Path(
                file_path
            )
        )
    )

    print(
        f"{label:<35}: {exists}"
    )

    if file_path is not None:
        print(
            f"{'Path':<35}: {file_path}"
        )


# =============================================================================
# TERMINAL TEST
# =============================================================================

def main() -> None:
    """
    Menguji data loader melalui terminal.
    """

    print("=" * 80)
    print(
        "STREAMLIT DATA LOADER TEST"
    )
    print("=" * 80)

    print(
        "\nKonfigurasi direktori:"
    )

    print(
        f"Project root                       : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Streamlit directory                : "
        f"{STREAMLIT_DIR}"
    )

    print(
        f"Results directory                  : "
        f"{RESULTS_DIR}"
    )

    print(
        f"Tables directory                   : "
        f"{TABLES_DIR}"
    )

    print(
        f"Figures directory                  : "
        f"{FIGURES_DIR}"
    )

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

    inference_efficiency = (
        load_inference_efficiency()
    )

    final_findings = (
        load_final_research_findings()
    )

    training_curve_report = (
        load_training_curve_report()
    )

    overfitting_analysis = (
        load_overfitting_analysis()
    )

    confusion_summary = (
        load_confusion_matrix_summary()
    )

    misclassification = (
        load_misclassification_analysis()
    )

    global_shap = (
        load_global_shap()
    )

    global_shap_by_class = (
        load_global_shap_by_class()
    )

    local_shap = (
        load_local_shap_summary()
    )

    local_contributions = (
        load_local_token_contributions()
    )

    waterfall = (
        load_waterfall_summary()
    )

    deployment_config = (
        load_deployment_config()
    )

    deployment_report = (
        load_deployment_report()
    )

    inference_report = (
        load_inference_pipeline_report()
    )

    print(
        "\nStatus tabel:"
    )

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
        "Inference efficiency",
        inference_efficiency,
    )

    print_dataframe_status(
        "Final research findings",
        final_findings,
    )

    print_dataframe_status(
        "Training curve report",
        training_curve_report,
    )

    print_dataframe_status(
        "Overfitting analysis",
        overfitting_analysis,
    )

    print_dataframe_status(
        "Confusion matrix summary",
        confusion_summary,
    )

    print_dataframe_status(
        "Misclassification analysis",
        misclassification,
    )

    print_dataframe_status(
        "Global SHAP",
        global_shap,
    )

    print_dataframe_status(
        "Global SHAP by class",
        global_shap_by_class,
    )

    print_dataframe_status(
        "Local SHAP",
        local_shap,
    )

    print_dataframe_status(
        "Local token contributions",
        local_contributions,
    )

    print_dataframe_status(
        "Waterfall summary",
        waterfall,
    )

    print(
        "\nStatus JSON:"
    )

    print_json_status(
        "Deployment config",
        deployment_config,
    )

    print_json_status(
        "Deployment report",
        deployment_report,
    )

    print_json_status(
        "Inference pipeline report",
        inference_report,
    )

    print(
        "\nStatus grafik:"
    )

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

    sample_normalized_confusion = (
        get_confusion_matrix_path(
            "cnn_k2",
            normalized=True,
        )
    )

    sample_global_shap = (
        get_global_shap_figure_path(
            "cnn_k2_global_shap_top_tokens.png"
        )
    )

    print_file_status(
        "Training curve cnn_k2",
        sample_training_curve,
    )

    print_file_status(
        "Confusion matrix cnn_k2",
        sample_confusion_matrix,
    )

    print_file_status(
        "Normalized confusion cnn_k2",
        sample_normalized_confusion,
    )

    print_file_status(
        "Global SHAP figure",
        sample_global_shap,
    )

    if dataframe_is_available(
        waterfall
    ):
        first_waterfall_path = (
            resolve_waterfall_figure_from_row(
                waterfall.iloc[0]
            )
        )

        print_file_status(
            "Waterfall sample",
            first_waterfall_path,
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "Data loader test selesai."
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
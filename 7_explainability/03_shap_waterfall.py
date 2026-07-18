# =============================================================================
# STEP 7.3 - SHAP WATERFALL PLOT
# =============================================================================
# File:
# 7_explainability/03_shap_waterfall.py
#
# Tujuan:
# Membuat visualisasi waterfall kumulatif dari hasil local SHAP CNN K2.
#
# Konsep:
#
# baseline probability
# + kontribusi token pertama
# + kontribusi token kedua
# + ...
# + kontribusi token lainnya
# = probabilitas prediksi model
#
# Nilai SHAP positif:
# - Meningkatkan probabilitas kelas prediksi.
#
# Nilai SHAP negatif:
# - Menurunkan probabilitas kelas prediksi.
#
# Input:
# - 9_results/tables/shap/local/cnn_k2_local_shap_summary.csv
# - 9_results/tables/shap/local/cnn_k2_local_token_contributions.csv
#
# Output:
# - 9_results/figures/shap/waterfall/*.png
# - 9_results/tables/shap/cnn_k2_waterfall_summary.csv
# - 9_results/tables/shap/cnn_k2_waterfall_configuration.json
#
# Catatan:
# Urutan token dalam waterfall ditentukan berdasarkan absolute SHAP
# terbesar. Nilai kumulatif di antara baseline dan hasil akhir merupakan
# alur visualisasi kontribusi, bukan prediksi model setelah token
# ditambahkan satu per satu secara aktual.
# =============================================================================

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# KONFIGURASI
# =============================================================================

MODEL_NAME = "cnn_k2"
DATASET_NAME = "Kompas"
SCENARIO_CODE = "K2"
SCENARIO_NAME = "Title + Description"

# Jumlah token utama yang ditampilkan secara terpisah.
TOP_N_TOKENS = 15

# Kontribusi token yang tidak masuk TOP_N_TOKENS serta kontribusi token
# teknis akan digabungkan menjadi "Token lainnya".
OTHER_TOKEN_LABEL = "Token lainnya"

# Toleransi rekonstruksi:
#
# baseline + total SHAP = prediction confidence
RECONSTRUCTION_TOLERANCE = 2e-2

# Toleransi kecil untuk menentukan nilai nol.
ZERO_TOLERANCE = 1e-12

SPECIAL_TOKENS = {
    "",
    "PAD",
    "SEP",
    "UNK",
    "OOV",
}


# =============================================================================
# PATH INPUT
# =============================================================================

LOCAL_SUMMARY_PATH = (
    PROJECT_ROOT
    / "9_results"
    / "tables"
    / "shap"
    / "local"
    / "cnn_k2_local_shap_summary.csv"
)

LOCAL_CONTRIBUTIONS_PATH = (
    PROJECT_ROOT
    / "9_results"
    / "tables"
    / "shap"
    / "local"
    / "cnn_k2_local_token_contributions.csv"
)


# =============================================================================
# PATH OUTPUT
# =============================================================================

OUTPUT_FIGURES_DIR = (
    PROJECT_ROOT
    / "9_results"
    / "figures"
    / "shap"
    / "waterfall"
)

OUTPUT_TABLES_DIR = (
    PROJECT_ROOT
    / "9_results"
    / "tables"
    / "shap"
)

WATERFALL_SUMMARY_PATH = (
    OUTPUT_TABLES_DIR
    / "cnn_k2_waterfall_summary.csv"
)

WATERFALL_CONFIGURATION_PATH = (
    OUTPUT_TABLES_DIR
    / "cnn_k2_waterfall_configuration.json"
)


# =============================================================================
# UTILITAS UMUM
# =============================================================================

def print_header(
    title: str,
) -> None:
    """
    Menampilkan header pada terminal.
    """

    print("=" * 80)
    print(title)
    print("=" * 80)


def create_output_directories() -> None:
    """
    Membuat seluruh folder output.
    """

    directories = [
        OUTPUT_FIGURES_DIR,
        OUTPUT_TABLES_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def safe_filename(
    text: Any,
) -> str:
    """
    Membersihkan teks agar aman digunakan sebagai nama file.
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

    if not value:
        return "unknown"

    return value


def is_special_token(
    token: Any,
) -> bool:
    """
    Menentukan apakah token merupakan token teknis.
    """

    normalized_token = (
        str(token)
        .strip()
        .replace(
            "[",
            "",
        )
        .replace(
            "]",
            "",
        )
        .upper()
    )

    return (
        normalized_token
        in SPECIAL_TOKENS
    )


def parse_boolean_series(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """
    Mengubah kolom boolean dari CSV menjadi bool secara aman.
    """

    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }

    result = normalized.map(
        mapping
    )

    if result.isna().any():
        invalid_values = (
            series[
                result.isna()
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Kolom {column_name} memiliki nilai boolean "
            "yang tidak dikenali.\n"
            f"Nilai: {invalid_values}"
        )

    return result.astype(
        bool
    )


def convert_numeric_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
    dataframe_name: str,
) -> pd.DataFrame:
    """
    Mengubah kolom menjadi numerik dan menghentikan proses
    apabila ditemukan nilai tidak valid.
    """

    dataframe = dataframe.copy()

    for column in columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    invalid_rows = dataframe[
        columns
    ].isna().any(
        axis=1
    )

    if invalid_rows.any():
        invalid_indices = (
            dataframe.index[
                invalid_rows
            ]
            .tolist()
        )

        raise ValueError(
            f"Ditemukan nilai numerik tidak valid pada "
            f"{dataframe_name}.\n"
            f"Baris: {invalid_indices[:20]}"
        )

    return dataframe


def validate_integer_column(
    dataframe: pd.DataFrame,
    column: str,
    allow_zero: bool = True,
) -> pd.DataFrame:
    """
    Memastikan kolom numerik merupakan bilangan bulat.
    """

    dataframe = dataframe.copy()

    non_integer = (
        dataframe[column]
        % 1
        != 0
    )

    if non_integer.any():
        invalid_values = (
            dataframe.loc[
                non_integer,
                column,
            ]
            .tolist()
        )

        raise ValueError(
            f"Kolom {column} harus berupa bilangan bulat.\n"
            f"Nilai tidak valid: {invalid_values[:20]}"
        )

    dataframe[column] = dataframe[
        column
    ].astype(
        int
    )

    minimum_value = (
        0
        if allow_zero
        else 1
    )

    if (
        dataframe[column]
        < minimum_value
    ).any():
        raise ValueError(
            f"Kolom {column} harus memiliki nilai "
            f">= {minimum_value}."
        )

    return dataframe


# =============================================================================
# MEMUAT DAN MEMVALIDASI DATA
# =============================================================================

def load_local_shap_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Memuat ringkasan dan kontribusi token dari STEP 7.2.
    """

    if not LOCAL_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "File ringkasan local SHAP tidak ditemukan:\n"
            f"{LOCAL_SUMMARY_PATH}\n\n"
            "Jalankan 02_shap_local.py terlebih dahulu."
        )

    if not LOCAL_CONTRIBUTIONS_PATH.exists():
        raise FileNotFoundError(
            "File kontribusi token local SHAP tidak ditemukan:\n"
            f"{LOCAL_CONTRIBUTIONS_PATH}\n\n"
            "Jalankan 02_shap_local.py terlebih dahulu."
        )

    if LOCAL_SUMMARY_PATH.stat().st_size <= 0:
        raise ValueError(
            "File ringkasan local SHAP kosong:\n"
            f"{LOCAL_SUMMARY_PATH}"
        )

    if LOCAL_CONTRIBUTIONS_PATH.stat().st_size <= 0:
        raise ValueError(
            "File kontribusi local SHAP kosong:\n"
            f"{LOCAL_CONTRIBUTIONS_PATH}"
        )

    summary = pd.read_csv(
        LOCAL_SUMMARY_PATH,
        encoding="utf-8-sig",
    )

    contributions = pd.read_csv(
        LOCAL_CONTRIBUTIONS_PATH,
        encoding="utf-8-sig",
    )

    required_summary_columns = {
        "shap_sample_position",
        "original_test_index",
        "document_id",
        "selection_type",
        "actual_label",
        "predicted_label",
        "is_correct",
        "prediction_confidence",
        "baseline_probability_predicted_class",
        "reconstructed_probability_predicted_class",
        "local_additivity_error",
    }

    required_contribution_columns = {
        "shap_sample_position",
        "original_test_index",
        "document_id",
        "selection_type",
        "actual_label",
        "predicted_label",
        "token_id",
        "token",
        "signed_shap",
        "absolute_shap",
        "occurrence_count",
        "direction",
    }

    missing_summary_columns = (
        required_summary_columns
        - set(
            summary.columns
        )
    )

    missing_contribution_columns = (
        required_contribution_columns
        - set(
            contributions.columns
        )
    )

    if missing_summary_columns:
        raise KeyError(
            "Kolom ringkasan local SHAP tidak lengkap.\n"
            f"Kolom hilang: "
            f"{sorted(missing_summary_columns)}"
        )

    if missing_contribution_columns:
        raise KeyError(
            "Kolom kontribusi token tidak lengkap.\n"
            f"Kolom hilang: "
            f"{sorted(missing_contribution_columns)}"
        )

    if summary.empty:
        raise ValueError(
            "Ringkasan local SHAP kosong."
        )

    if contributions.empty:
        raise ValueError(
            "Kontribusi token local SHAP kosong."
        )

    # -------------------------------------------------------------------------
    # NORMALISASI STRING
    # -------------------------------------------------------------------------

    summary_string_columns = [
        "document_id",
        "selection_type",
        "actual_label",
        "predicted_label",
    ]

    contribution_string_columns = [
        "document_id",
        "selection_type",
        "actual_label",
        "predicted_label",
        "token",
        "direction",
    ]

    for column in summary_string_columns:
        summary[column] = (
            summary[column]
            .astype(str)
            .str.strip()
        )

    for column in contribution_string_columns:
        contributions[column] = (
            contributions[column]
            .astype(str)
            .str.strip()
        )

    # -------------------------------------------------------------------------
    # BOOLEAN
    # -------------------------------------------------------------------------

    summary["is_correct"] = (
        parse_boolean_series(
            summary["is_correct"],
            column_name="is_correct",
        )
    )

    # -------------------------------------------------------------------------
    # NUMERIK
    # -------------------------------------------------------------------------

    summary_numeric_columns = [
        "shap_sample_position",
        "original_test_index",
        "prediction_confidence",
        "baseline_probability_predicted_class",
        "reconstructed_probability_predicted_class",
        "local_additivity_error",
    ]

    contribution_numeric_columns = [
        "shap_sample_position",
        "original_test_index",
        "token_id",
        "signed_shap",
        "absolute_shap",
        "occurrence_count",
    ]

    summary = convert_numeric_columns(
        dataframe=summary,
        columns=summary_numeric_columns,
        dataframe_name="ringkasan local SHAP",
    )

    contributions = convert_numeric_columns(
        dataframe=contributions,
        columns=contribution_numeric_columns,
        dataframe_name="kontribusi token local SHAP",
    )

    integer_summary_columns = [
        "shap_sample_position",
        "original_test_index",
    ]

    integer_contribution_columns = [
        "shap_sample_position",
        "original_test_index",
        "token_id",
        "occurrence_count",
    ]

    for column in integer_summary_columns:
        summary = validate_integer_column(
            dataframe=summary,
            column=column,
            allow_zero=True,
        )

    for column in integer_contribution_columns:
        contributions = validate_integer_column(
            dataframe=contributions,
            column=column,
            allow_zero=(
                column
                != "occurrence_count"
            ),
        )

    # -------------------------------------------------------------------------
    # VALIDASI PROBABILITAS
    # -------------------------------------------------------------------------

    probability_columns = [
        "prediction_confidence",
        "baseline_probability_predicted_class",
        "reconstructed_probability_predicted_class",
    ]

    for column in probability_columns:
        invalid_probability = (
            (summary[column] < -1e-7)
            | (summary[column] > 1.0 + 1e-7)
        )

        if invalid_probability.any():
            invalid_documents = (
                summary.loc[
                    invalid_probability,
                    "document_id",
                ]
                .tolist()
            )

            raise ValueError(
                f"Kolom {column} memiliki nilai di luar "
                "rentang 0 sampai 1.\n"
                f"Dokumen: {invalid_documents}"
            )

    if (
        summary[
            "local_additivity_error"
        ]
        < 0.0
    ).any():
        raise ValueError(
            "Local additivity error tidak boleh negatif."
        )

    excessive_additivity_error = (
        summary[
            "local_additivity_error"
        ]
        > RECONSTRUCTION_TOLERANCE
    )

    if excessive_additivity_error.any():
        invalid_documents = (
            summary.loc[
                excessive_additivity_error,
                "document_id",
            ]
            .tolist()
        )

        raise ValueError(
            "Local additivity error melebihi toleransi.\n"
            f"Toleransi: {RECONSTRUCTION_TOLERANCE}\n"
            f"Dokumen  : {invalid_documents}"
        )

    reconstructed_difference = np.abs(
        summary[
            "prediction_confidence"
        ]
        - summary[
            "reconstructed_probability_predicted_class"
        ]
    )

    if (
        reconstructed_difference
        > RECONSTRUCTION_TOLERANCE
    ).any():
        invalid_documents = (
            summary.loc[
                reconstructed_difference
                > RECONSTRUCTION_TOLERANCE,
                "document_id",
            ]
            .tolist()
        )

        raise ValueError(
            "Reconstructed probability tidak konsisten "
            "dengan prediction confidence.\n"
            f"Dokumen: {invalid_documents}"
        )

    # -------------------------------------------------------------------------
    # VALIDASI STATUS BENAR/SALAH
    # -------------------------------------------------------------------------

    calculated_correctness = (
        summary[
            "actual_label"
        ].str.lower()
        == summary[
            "predicted_label"
        ].str.lower()
    )

    if not np.array_equal(
        calculated_correctness.to_numpy(
            dtype=bool
        ),
        summary[
            "is_correct"
        ].to_numpy(
            dtype=bool
        ),
    ):
        invalid_documents = (
            summary.loc[
                calculated_correctness
                != summary[
                    "is_correct"
                ],
                "document_id",
            ]
            .tolist()
        )

        raise ValueError(
            "Nilai is_correct tidak konsisten dengan "
            "label aktual dan label prediksi.\n"
            f"Dokumen: {invalid_documents}"
        )

    # -------------------------------------------------------------------------
    # VALIDASI KONTRIBUSI
    # -------------------------------------------------------------------------

    if (
        contributions[
            "absolute_shap"
        ]
        < 0.0
    ).any():
        raise ValueError(
            "Absolute SHAP tidak boleh negatif."
        )

    invalid_absolute_shap = (
        contributions[
            "absolute_shap"
        ]
        + 1e-10
        < np.abs(
            contributions[
                "signed_shap"
            ]
        )
    )

    if invalid_absolute_shap.any():
        invalid_documents = (
            contributions.loc[
                invalid_absolute_shap,
                "document_id",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Absolute SHAP lebih kecil daripada absolute "
            "signed SHAP.\n"
            f"Dokumen: {invalid_documents}"
        )

    # -------------------------------------------------------------------------
    # VALIDASI DUPLIKASI RINGKASAN
    # -------------------------------------------------------------------------

    duplicated_summary = summary[
        "shap_sample_position"
    ].duplicated(
        keep=False
    )

    if duplicated_summary.any():
        duplicated_positions = (
            summary.loc[
                duplicated_summary,
                "shap_sample_position",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Ditemukan shap_sample_position duplikat "
            "pada ringkasan.\n"
            f"Posisi: {duplicated_positions}"
        )

    # -------------------------------------------------------------------------
    # VALIDASI SETIAP SAMPEL MEMILIKI KONTRIBUSI
    # -------------------------------------------------------------------------

    summary_positions = set(
        summary[
            "shap_sample_position"
        ].tolist()
    )

    contribution_positions = set(
        contributions[
            "shap_sample_position"
        ].tolist()
    )

    missing_contribution_positions = (
        summary_positions
        - contribution_positions
    )

    if missing_contribution_positions:
        raise ValueError(
            "Beberapa sampel tidak memiliki kontribusi token.\n"
            f"SHAP sample position: "
            f"{sorted(missing_contribution_positions)}"
        )

    summary = (
        summary
        .sort_values(
            [
                "selection_type",
                "shap_sample_position",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    contributions = contributions.reset_index(
        drop=True
    )

    return (
        summary,
        contributions,
    )


# =============================================================================
# MENGAMBIL KONTRIBUSI SATU SAMPEL
# =============================================================================

def get_sample_contributions(
    summary_row: pd.Series,
    contributions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mengambil kontribusi token untuk satu sampel menggunakan
    shap_sample_position dan identitas sampel lainnya.
    """

    shap_sample_position = int(
        summary_row[
            "shap_sample_position"
        ]
    )

    original_test_index = int(
        summary_row[
            "original_test_index"
        ]
    )

    document_id = str(
        summary_row[
            "document_id"
        ]
    )

    selection_type = str(
        summary_row[
            "selection_type"
        ]
    )

    sample_contributions = contributions[
        (
            contributions[
                "shap_sample_position"
            ]
            == shap_sample_position
        )
        & (
            contributions[
                "original_test_index"
            ]
            == original_test_index
        )
        & (
            contributions[
                "document_id"
            ].astype(str)
            == document_id
        )
        & (
            contributions[
                "selection_type"
            ].astype(str)
            == selection_type
        )
    ].copy()

    if sample_contributions.empty:
        raise ValueError(
            "Kontribusi token tidak ditemukan untuk sampel:\n"
            f"SHAP position : {shap_sample_position}\n"
            f"Test index    : {original_test_index}\n"
            f"Document ID   : {document_id}"
        )

    actual_labels = set(
        sample_contributions[
            "actual_label"
        ].astype(str)
    )

    predicted_labels = set(
        sample_contributions[
            "predicted_label"
        ].astype(str)
    )

    if actual_labels != {
        str(
            summary_row[
                "actual_label"
            ]
        )
    }:
        raise ValueError(
            "Label aktual pada kontribusi tidak konsisten "
            f"untuk {document_id}."
        )

    if predicted_labels != {
        str(
            summary_row[
                "predicted_label"
            ]
        )
    }:
        raise ValueError(
            "Label prediksi pada kontribusi tidak konsisten "
            f"untuk {document_id}."
        )

    return sample_contributions


# =============================================================================
# MENYIAPKAN KONTRIBUSI WATERFALL
# =============================================================================

def prepare_waterfall_data(
    summary_row: pd.Series,
    sample_contributions: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, float],
]:
    """
    Menyiapkan token utama dan residual "Token lainnya".

    Residual dihitung dengan:

    prediction confidence
    - baseline probability
    - total kontribusi token utama

    Dengan demikian, waterfall selalu berakhir pada probabilitas
    prediksi model.
    """

    semantic_mask = (
        ~sample_contributions[
            "token"
        ]
        .astype(str)
        .map(
            is_special_token
        )
    )

    semantic_contributions = sample_contributions[
        semantic_mask
    ].copy()

    if semantic_contributions.empty:
        raise ValueError(
            "Tidak ditemukan token semantik "
            "untuk sampel ini."
        )

    # Local SHAP tahap sebelumnya sebenarnya sudah mengagregasikan
    # token berulang. Agregasi ulang dilakukan secara aman berdasarkan
    # token_id dan token untuk memastikan satu token hanya memiliki
    # satu baris.
    aggregated = (
        semantic_contributions
        .groupby(
            [
                "token_id",
                "token",
            ],
            as_index=False,
        )
        .agg(
            signed_shap=(
                "signed_shap",
                "sum",
            ),

            absolute_shap=(
                "absolute_shap",
                "sum",
            ),

            occurrence_count=(
                "occurrence_count",
                "sum",
            ),
        )
    )

    aggregated = (
        aggregated
        .sort_values(
            [
                "absolute_shap",
                "signed_shap",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    top_tokens = (
        aggregated
        .head(
            TOP_N_TOKENS
        )
        .copy()
    )

    baseline_probability = float(
        summary_row[
            "baseline_probability_predicted_class"
        ]
    )

    prediction_probability = float(
        summary_row[
            "prediction_confidence"
        ]
    )

    total_prediction_delta = (
        prediction_probability
        - baseline_probability
    )

    displayed_token_sum = float(
        top_tokens[
            "signed_shap"
        ].sum()
    )

    other_token_contribution = (
        total_prediction_delta
        - displayed_token_sum
    )

    waterfall_rows: list[
        dict[str, Any]
    ] = []

    for token_row in top_tokens.itertuples(
        index=False
    ):
        token_label = str(
            token_row.token
        )

        occurrence_count = int(
            token_row.occurrence_count
        )

        if occurrence_count > 1:
            token_label = (
                f"{token_label} "
                f"(×{occurrence_count})"
            )

        waterfall_rows.append(
            {
                "token":
                    str(
                        token_row.token
                    ),

                "plot_label":
                    token_label,

                "signed_shap":
                    float(
                        token_row.signed_shap
                    ),

                "absolute_shap":
                    float(
                        token_row.absolute_shap
                    ),

                "occurrence_count":
                    occurrence_count,

                "is_residual":
                    False,
            }
        )

    # Residual tetap dimasukkan meskipun sangat kecil agar rekonstruksi
    # waterfall transparan.
    waterfall_rows.append(
        {
            "token":
                OTHER_TOKEN_LABEL,

            "plot_label":
                OTHER_TOKEN_LABEL,

            "signed_shap":
                float(
                    other_token_contribution
                ),

            "absolute_shap":
                abs(
                    float(
                        other_token_contribution
                    )
                ),

            "occurrence_count":
                int(
                    max(
                        len(
                            aggregated
                        )
                        - len(
                            top_tokens
                        ),
                        0,
                    )
                ),

            "is_residual":
                True,
        }
    )

    waterfall_data = pd.DataFrame(
        waterfall_rows
    )

    current_value = baseline_probability

    start_values: list[float] = []
    end_values: list[float] = []
    left_values: list[float] = []
    widths: list[float] = []

    for contribution in waterfall_data[
        "signed_shap"
    ].to_numpy(
        dtype=float
    ):
        start_value = current_value
        end_value = (
            start_value
            + contribution
        )

        start_values.append(
            start_value
        )

        end_values.append(
            end_value
        )

        left_values.append(
            min(
                start_value,
                end_value,
            )
        )

        widths.append(
            abs(
                contribution
            )
        )

        current_value = end_value

    waterfall_data[
        "start_value"
    ] = start_values

    waterfall_data[
        "end_value"
    ] = end_values

    waterfall_data[
        "bar_left"
    ] = left_values

    waterfall_data[
        "bar_width"
    ] = widths

    reconstructed_final = float(
        current_value
    )

    reconstruction_error = abs(
        reconstructed_final
        - prediction_probability
    )

    if (
        reconstruction_error
        > RECONSTRUCTION_TOLERANCE
    ):
        raise ValueError(
            "Waterfall gagal merekonstruksi probabilitas prediksi.\n"
            f"Baseline       : {baseline_probability:.8f}\n"
            f"Prediction     : {prediction_probability:.8f}\n"
            f"Reconstructed  : {reconstructed_final:.8f}\n"
            f"Error          : {reconstruction_error:.8f}"
        )

    omitted_semantic_sum = float(
        aggregated.iloc[
            len(
                top_tokens
            ):
        ][
            "signed_shap"
        ].sum()
    )

    residual_unaccounted_component = (
        other_token_contribution
        - omitted_semantic_sum
    )

    diagnostics = {
        "baseline_probability":
            baseline_probability,

        "prediction_probability":
            prediction_probability,

        "total_prediction_delta":
            total_prediction_delta,

        "displayed_token_sum":
            displayed_token_sum,

        "omitted_semantic_token_sum":
            omitted_semantic_sum,

        "other_token_contribution":
            float(
                other_token_contribution
            ),

        "residual_unaccounted_component":
            float(
                residual_unaccounted_component
            ),

        "reconstructed_final":
            reconstructed_final,

        "reconstruction_error":
            reconstruction_error,

        "total_semantic_token_count":
            float(
                len(
                    aggregated
                )
            ),

        "displayed_token_count":
            float(
                len(
                    top_tokens
                )
            ),
    }

    return (
        waterfall_data,
        diagnostics,
    )


# =============================================================================
# MEMBUAT WATERFALL PLOT
# =============================================================================

def create_waterfall_plot(
    summary_row: pd.Series,
    waterfall_data: pd.DataFrame,
    diagnostics: dict[str, float],
) -> Path:
    """
    Membuat grafik waterfall kumulatif.

    Setiap batang dimulai dari nilai kumulatif sebelumnya.
    """

    document_id = str(
        summary_row[
            "document_id"
        ]
    )

    selection_type = str(
        summary_row[
            "selection_type"
        ]
    )

    actual_label = str(
        summary_row[
            "actual_label"
        ]
    )

    predicted_label = str(
        summary_row[
            "predicted_label"
        ]
    )

    is_correct = bool(
        summary_row[
            "is_correct"
        ]
    )

    confidence = float(
        summary_row[
            "prediction_confidence"
        ]
    )

    output_filename = (
        f"{safe_filename(document_id)}_"
        f"{safe_filename(selection_type)}_"
        f"actual_{safe_filename(actual_label)}_"
        f"pred_{safe_filename(predicted_label)}_"
        f"waterfall.png"
    )

    output_path = (
        OUTPUT_FIGURES_DIR
        / output_filename
    )

    number_of_bars = len(
        waterfall_data
    )

    positions = np.arange(
        number_of_bars
    )

    figure_height = max(
        8.0,
        number_of_bars
        * 0.48
        + 2.5,
    )

    figure, axis = plt.subplots(
        figsize=(
            13,
            figure_height,
        )
    )

    positive_color = "tab:red"
    negative_color = "tab:blue"
    residual_color = "tab:gray"

    bar_colors = []

    for row in waterfall_data.itertuples(
        index=False
    ):
        if bool(
            row.is_residual
        ):
            bar_colors.append(
                residual_color
            )

        elif float(
            row.signed_shap
        ) >= 0.0:
            bar_colors.append(
                positive_color
            )

        else:
            bar_colors.append(
                negative_color
            )

    axis.barh(
        positions,
        waterfall_data[
            "bar_width"
        ],
        left=waterfall_data[
            "bar_left"
        ],
        color=bar_colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )

    axis.set_yticks(
        positions
    )

    axis.set_yticklabels(
        waterfall_data[
            "plot_label"
        ].astype(str)
    )

    axis.invert_yaxis()

    baseline_probability = float(
        diagnostics[
            "baseline_probability"
        ]
    )

    prediction_probability = float(
        diagnostics[
            "prediction_probability"
        ]
    )

    axis.axvline(
        baseline_probability,
        color="black",
        linestyle="--",
        linewidth=1.4,
    )

    axis.axvline(
        prediction_probability,
        color="black",
        linestyle="-",
        linewidth=1.6,
    )

    all_values = np.concatenate(
        [
            waterfall_data[
                "start_value"
            ].to_numpy(
                dtype=float
            ),

            waterfall_data[
                "end_value"
            ].to_numpy(
                dtype=float
            ),

            np.asarray(
                [
                    baseline_probability,
                    prediction_probability,
                ],
                dtype=float,
            ),
        ]
    )

    minimum_value = float(
        np.min(
            all_values
        )
    )

    maximum_value = float(
        np.max(
            all_values
        )
    )

    value_range = max(
        maximum_value
        - minimum_value,
        0.05,
    )

    margin = (
        value_range
        * 0.15
    )

    axis.set_xlim(
        minimum_value
        - margin,
        maximum_value
        + margin,
    )

    label_offset = (
        value_range
        * 0.025
    )

    for position, row in enumerate(
        waterfall_data.itertuples(
            index=False
        )
    ):
        contribution = float(
            row.signed_shap
        )

        end_value = float(
            row.end_value
        )

        if contribution >= 0.0:
            text_x = (
                end_value
                + label_offset
            )

            horizontal_alignment = (
                "left"
            )

        else:
            text_x = (
                end_value
                - label_offset
            )

            horizontal_alignment = (
                "right"
            )

        axis.text(
            text_x,
            position,
            f"{contribution:+.4f}",
            va="center",
            ha=horizontal_alignment,
            fontsize=8,
        )

    top_position = -0.75

    axis.text(
        baseline_probability,
        top_position,
        (
            "Baseline\n"
            f"{baseline_probability:.4f}"
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

    axis.text(
        prediction_probability,
        top_position,
        (
            "Prediksi\n"
            f"{prediction_probability:.4f}"
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

    axis.set_xlabel(
        "Probabilitas Kelas Prediksi"
    )

    axis.set_ylabel(
        "Token berdasarkan Absolute SHAP"
    )

    status_text = (
        "BENAR"
        if is_correct
        else "SALAH"
    )

    axis.set_title(
        "SHAP Waterfall Plot — CNN K2\n"
        f"Dokumen: {document_id} | "
        f"Aktual: {actual_label} | "
        f"Prediksi: {predicted_label} | "
        f"Confidence: {confidence:.2%} | "
        f"{status_text}"
    )

    axis.grid(
        axis="x",
        alpha=0.25,
    )

    legend_elements = [
        Patch(
            facecolor=positive_color,
            edgecolor="black",
            label=(
                "Meningkatkan probabilitas"
            ),
        ),

        Patch(
            facecolor=negative_color,
            edgecolor="black",
            label=(
                "Menurunkan probabilitas"
            ),
        ),

        Patch(
            facecolor=residual_color,
            edgecolor="black",
            label=(
                OTHER_TOKEN_LABEL
            ),
        ),

        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            label="Baseline",
        ),

        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            label="Probabilitas prediksi",
        ),
    ]

    axis.legend(
        handles=legend_elements,
        loc="best",
    )

    figure.text(
        0.5,
        0.01,
        (
            "Catatan: urutan token berdasarkan absolute SHAP. "
            "Nilai kumulatif di antara baseline dan prediksi "
            "merupakan alur visualisasi, bukan prediksi model "
            "setelah token ditambahkan satu per satu."
        ),
        ha="center",
        fontsize=8,
    )

    figure.tight_layout(
        rect=[
            0.0,
            0.04,
            1.0,
            1.0,
        ]
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


# =============================================================================
# RINGKASAN WATERFALL
# =============================================================================

def build_waterfall_summary_row(
    summary_row: pd.Series,
    sample_contributions: pd.DataFrame,
    waterfall_data: pd.DataFrame,
    diagnostics: dict[str, float],
    figure_path: Path,
) -> dict[str, Any]:
    """
    Membentuk satu baris ringkasan waterfall.
    """

    semantic_contributions = sample_contributions[
        ~sample_contributions[
            "token"
        ]
        .astype(str)
        .map(
            is_special_token
        )
    ].copy()

    aggregated = (
        semantic_contributions
        .groupby(
            [
                "token_id",
                "token",
            ],
            as_index=False,
        )
        .agg(
            signed_shap=(
                "signed_shap",
                "sum",
            ),

            absolute_shap=(
                "absolute_shap",
                "sum",
            ),

            occurrence_count=(
                "occurrence_count",
                "sum",
            ),
        )
    )

    supporting_tokens = (
        aggregated[
            aggregated[
                "signed_shap"
            ]
            > 0.0
        ]
        .sort_values(
            "signed_shap",
            ascending=False,
        )
        .head(
            5
        )
    )

    opposing_tokens = (
        aggregated[
            aggregated[
                "signed_shap"
            ]
            < 0.0
        ]
        .sort_values(
            "signed_shap",
            ascending=True,
        )
        .head(
            5
        )
    )

    supporting_text = "; ".join(
        (
            f"{row.token} "
            f"({row.signed_shap:+.4f})"
        )
        for row in supporting_tokens.itertuples(
            index=False
        )
    )

    opposing_text = "; ".join(
        (
            f"{row.token} "
            f"({row.signed_shap:+.4f})"
        )
        for row in opposing_tokens.itertuples(
            index=False
        )
    )

    positive_displayed_sum = float(
        waterfall_data.loc[
            waterfall_data[
                "signed_shap"
            ]
            > 0.0,
            "signed_shap",
        ].sum()
    )

    negative_displayed_sum = float(
        waterfall_data.loc[
            waterfall_data[
                "signed_shap"
            ]
            < 0.0,
            "signed_shap",
        ].sum()
    )

    return {
        "shap_sample_position":
            int(
                summary_row[
                    "shap_sample_position"
                ]
            ),

        "original_test_index":
            int(
                summary_row[
                    "original_test_index"
                ]
            ),

        "document_id":
            str(
                summary_row[
                    "document_id"
                ]
            ),

        "selection_type":
            str(
                summary_row[
                    "selection_type"
                ]
            ),

        "actual_label":
            str(
                summary_row[
                    "actual_label"
                ]
            ),

        "predicted_label":
            str(
                summary_row[
                    "predicted_label"
                ]
            ),

        "is_correct":
            bool(
                summary_row[
                    "is_correct"
                ]
            ),

        "baseline_probability":
            float(
                diagnostics[
                    "baseline_probability"
                ]
            ),

        "prediction_confidence":
            float(
                diagnostics[
                    "prediction_probability"
                ]
            ),

        "total_prediction_delta":
            float(
                diagnostics[
                    "total_prediction_delta"
                ]
            ),

        "displayed_token_count":
            int(
                diagnostics[
                    "displayed_token_count"
                ]
            ),

        "total_semantic_token_count":
            int(
                diagnostics[
                    "total_semantic_token_count"
                ]
            ),

        "displayed_token_contribution_sum":
            float(
                diagnostics[
                    "displayed_token_sum"
                ]
            ),

        "other_token_contribution":
            float(
                diagnostics[
                    "other_token_contribution"
                ]
            ),

        "omitted_semantic_token_sum":
            float(
                diagnostics[
                    "omitted_semantic_token_sum"
                ]
            ),

        "residual_unaccounted_component":
            float(
                diagnostics[
                    "residual_unaccounted_component"
                ]
            ),

        "total_positive_displayed_contribution":
            positive_displayed_sum,

        "total_negative_displayed_contribution":
            negative_displayed_sum,

        "waterfall_final_value":
            float(
                diagnostics[
                    "reconstructed_final"
                ]
            ),

        "waterfall_reconstruction_error":
            float(
                diagnostics[
                    "reconstruction_error"
                ]
            ),

        "top_supporting_tokens":
            supporting_text,

        "top_opposing_tokens":
            opposing_text,

        "figure_path":
            str(
                figure_path
            ),
    }


# =============================================================================
# MENYIMPAN KONFIGURASI
# =============================================================================

def save_configuration(
    number_of_samples: int,
    success_count: int,
    failure_count: int,
    waterfall_summary: pd.DataFrame,
) -> None:
    """
    Menyimpan konfigurasi waterfall.
    """

    maximum_reconstruction_error = (
        float(
            waterfall_summary[
                "waterfall_reconstruction_error"
            ].max()
        )
        if not waterfall_summary.empty
        else None
    )

    configuration = {
        "generated_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "model":
            MODEL_NAME,

        "dataset":
            DATASET_NAME,

        "scenario_code":
            SCENARIO_CODE,

        "scenario_name":
            SCENARIO_NAME,

        "source_summary":
            str(
                LOCAL_SUMMARY_PATH
            ),

        "source_token_contributions":
            str(
                LOCAL_CONTRIBUTIONS_PATH
            ),

        "visualization_type":
            "Cumulative SHAP waterfall plot",

        "top_n_tokens_per_plot":
            TOP_N_TOKENS,

        "residual_label":
            OTHER_TOKEN_LABEL,

        "number_of_samples":
            number_of_samples,

        "successful_plots":
            success_count,

        "failed_plots":
            failure_count,

        "reconstruction_tolerance":
            RECONSTRUCTION_TOLERANCE,

        "maximum_reconstruction_error":
            maximum_reconstruction_error,

        "special_tokens_excluded":
            sorted(
                SPECIAL_TOKENS
            ),

        "token_ordering":
            (
                "Descending absolute SHAP contribution."
            ),

        "residual_method":
            (
                "Prediction confidence minus baseline probability "
                "minus the sum of the displayed token contributions."
            ),

        "waterfall_equation":
            (
                "baseline probability + displayed token contributions "
                "+ other token contribution = prediction probability"
            ),

        "interpretation_note":
            (
                "Positive SHAP values increase the predicted class "
                "probability, while negative SHAP values decrease it. "
                "Intermediate cumulative values depend on the selected "
                "display order and are not sequential model predictions."
            ),
    }

    with open(
        WATERFALL_CONFIGURATION_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            ensure_ascii=False,
            indent=4,
        )


# =============================================================================
# PROGRAM UTAMA
# =============================================================================

def main() -> None:
    """
    Membuat waterfall plot untuk seluruh sampel
    yang tersedia pada hasil local SHAP.
    """

    print_header(
        "STEP 7.3 - SHAP WATERFALL PLOT"
    )

    create_output_directories()

    print("\nKonfigurasi:")

    print(
        f"Model                   : "
        f"{MODEL_NAME}"
    )

    print(
        f"Dataset                 : "
        f"{DATASET_NAME}"
    )

    print(
        f"Skenario                : "
        f"{SCENARIO_CODE}"
    )

    print(
        f"Representasi            : "
        f"{SCENARIO_NAME}"
    )

    print(
        f"Top token per plot      : "
        f"{TOP_N_TOKENS}"
    )

    print(
        f"Residual                : "
        f"{OTHER_TOKEN_LABEL}"
    )

    print(
        f"Reconstruction tolerance: "
        f"{RECONSTRUCTION_TOLERANCE}"
    )

    print(
        "\nMemuat hasil local SHAP..."
    )

    (
        summary,
        contributions,
    ) = load_local_shap_data()

    print(
        f"File ringkasan          : "
        f"{LOCAL_SUMMARY_PATH}"
    )

    print(
        f"File kontribusi         : "
        f"{LOCAL_CONTRIBUTIONS_PATH}"
    )

    print(
        f"Jumlah sampel           : "
        f"{len(summary)}"
    )

    print(
        f"Jumlah baris token      : "
        f"{len(contributions):,}"
    )

    success_count = 0
    failure_count = 0

    output_rows: list[
        dict[str, Any]
    ] = []

    print(
        "\nMembuat waterfall plot..."
    )

    for row_number, summary_row in (
        summary.iterrows()
    ):
        document_id = str(
            summary_row[
                "document_id"
            ]
        )

        shap_sample_position = int(
            summary_row[
                "shap_sample_position"
            ]
        )

        print(
            "\n" + "-" * 80
        )

        print(
            f"{row_number + 1}/{len(summary)} "
            f"- {document_id}"
        )

        print(
            f"SHAP sample position : "
            f"{shap_sample_position}"
        )

        print(
            f"Jenis sampel         : "
            f"{summary_row['selection_type']}"
        )

        print(
            f"Label aktual         : "
            f"{summary_row['actual_label']}"
        )

        print(
            f"Label prediksi       : "
            f"{summary_row['predicted_label']}"
        )

        print(
            f"Baseline probability : "
            f"{float(summary_row['baseline_probability_predicted_class']):.6f}"
        )

        print(
            f"Prediction confidence: "
            f"{float(summary_row['prediction_confidence']):.6f}"
        )

        try:
            sample_contributions = (
                get_sample_contributions(
                    summary_row=summary_row,
                    contributions=contributions,
                )
            )

            (
                waterfall_data,
                diagnostics,
            ) = prepare_waterfall_data(
                summary_row=summary_row,
                sample_contributions=(
                    sample_contributions
                ),
            )

            figure_path = (
                create_waterfall_plot(
                    summary_row=summary_row,
                    waterfall_data=(
                        waterfall_data
                    ),
                    diagnostics=diagnostics,
                )
            )

            output_row = (
                build_waterfall_summary_row(
                    summary_row=summary_row,
                    sample_contributions=(
                        sample_contributions
                    ),
                    waterfall_data=(
                        waterfall_data
                    ),
                    diagnostics=diagnostics,
                    figure_path=figure_path,
                )
            )

            output_rows.append(
                output_row
            )

            success_count += 1

            print(
                f"Token ditampilkan    : "
                f"{int(diagnostics['displayed_token_count'])}"
            )

            print(
                f"Kontribusi lainnya   : "
                f"{diagnostics['other_token_contribution']:+.6f}"
            )

            print(
                f"Nilai akhir waterfall: "
                f"{diagnostics['reconstructed_final']:.6f}"
            )

            print(
                f"Reconstruction error : "
                f"{diagnostics['reconstruction_error']:.10f}"
            )

            print(
                f"Grafik tersimpan     : "
                f"{figure_path}"
            )

        except Exception as error:
            failure_count += 1

            print(
                f"Gagal                : "
                f"{error}"
            )

    if not output_rows:
        raise RuntimeError(
            "Seluruh waterfall plot gagal dibuat."
        )

    waterfall_summary = pd.DataFrame(
        output_rows
    )

    waterfall_summary = (
        waterfall_summary
        .sort_values(
            [
                "selection_type",
                "shap_sample_position",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    waterfall_summary.to_csv(
        WATERFALL_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    save_configuration(
        number_of_samples=len(
            summary
        ),
        success_count=success_count,
        failure_count=failure_count,
        waterfall_summary=(
            waterfall_summary
        ),
    )

    print("\n")

    print_header(
        "HASIL SHAP WATERFALL PLOT"
    )

    print(
        f"\nJumlah sampel          : "
        f"{len(summary)}"
    )

    print(
        f"Plot berhasil          : "
        f"{success_count}"
    )

    print(
        f"Plot gagal             : "
        f"{failure_count}"
    )

    print(
        f"Max reconstruction err : "
        f"{waterfall_summary['waterfall_reconstruction_error'].max():.10f}"
    )

    display_columns = [
        "document_id",
        "actual_label",
        "predicted_label",
        "is_correct",
        "baseline_probability",
        "prediction_confidence",
        "other_token_contribution",
        "waterfall_reconstruction_error",
    ]

    print(
        "\n"
        + waterfall_summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nFolder waterfall:"
    )

    print(
        OUTPUT_FIGURES_DIR
    )

    print(
        "\nRingkasan waterfall:"
    )

    print(
        WATERFALL_SUMMARY_PATH
    )

    print(
        "\nKonfigurasi:"
    )

    print(
        WATERFALL_CONFIGURATION_PATH
    )

    print("\n")

    print_header(
        "Tahap SHAP waterfall selesai."
    )


if __name__ == "__main__":
    main()
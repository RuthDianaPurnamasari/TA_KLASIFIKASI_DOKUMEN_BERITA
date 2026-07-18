# =============================================================================
# STEP 6.3 - CONFUSION MATRICES
# =============================================================================
# File:
# 6_evaluation/03_confusion_matrices.py
#
# Tujuan:
# Membuat confusion matrix untuk 10 eksperimen final:
# - Kompas: K1, K2, K3
# - AG News: A1, A2
# - Model: CNN dan Attention-BiLSTM
#
# Input:
# 9_results/metrics/model_test_confusion_matrix_data.csv
#
# Output:
# - Confusion matrix jumlah data
# - Confusion matrix normalized berdasarkan kelas aktual
# - Ringkasan performa setiap kelas
# - Ringkasan kesalahan klasifikasi
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# PATH INPUT
# =============================================================================

CONFUSION_DATA_PATH = (
    PROJECT_ROOT
    / "9_results"
    / "metrics"
    / "model_test_confusion_matrix_data.csv"
)


# =============================================================================
# PATH OUTPUT
# =============================================================================

FIGURES_DIR = (
    PROJECT_ROOT
    / "9_results"
    / "figures"
    / "confusion_matrices"
)

TABLES_DIR = (
    PROJECT_ROOT
    / "9_results"
    / "tables"
)

CONFUSION_SUMMARY_PATH = (
    TABLES_DIR
    / "confusion_matrix_summary.csv"
)

MISCLASSIFICATION_PATH = (
    TABLES_DIR
    / "misclassification_analysis.csv"
)


# =============================================================================
# KONFIGURASI EKSPERIMEN
# =============================================================================

EXPERIMENT_ORDER = [
    "cnn_k1",
    "cnn_k2",
    "cnn_k3",
    "cnn_a1",
    "cnn_a2",
    "attention_bilstm_k1",
    "attention_bilstm_k2",
    "attention_bilstm_k3",
    "attention_bilstm_a1",
    "attention_bilstm_a2",
]

EXPECTED_NUM_CLASSES = 4


# =============================================================================
# MEMBUAT FOLDER OUTPUT
# =============================================================================

def create_output_directories() -> None:
    """
    Membuat seluruh folder output jika belum tersedia.
    """

    directories = [
        FIGURES_DIR,
        TABLES_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# =============================================================================
# MEMUAT DATA CONFUSION MATRIX
# =============================================================================

def load_confusion_data() -> pd.DataFrame:
    """
    Membaca dan memvalidasi data confusion matrix
    dari hasil evaluasi test set.
    """

    if not CONFUSION_DATA_PATH.exists():
        raise FileNotFoundError(
            "File confusion matrix tidak ditemukan:\n"
            f"{CONFUSION_DATA_PATH}"
        )

    if CONFUSION_DATA_PATH.stat().st_size == 0:
        raise ValueError(
            "File confusion matrix ditemukan, tetapi kosong:\n"
            f"{CONFUSION_DATA_PATH}"
        )

    dataframe = pd.read_csv(
        CONFUSION_DATA_PATH,
        encoding="utf-8-sig",
    )

    required_columns = {
        "experiment_name",
        "model",
        "dataset",
        "scenario_code",
        "actual_index",
        "actual_class",
        "predicted_index",
        "predicted_class",
        "count",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Kolom confusion matrix tidak lengkap.\n"
            f"Kolom hilang: {sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    string_columns = [
        "experiment_name",
        "model",
        "dataset",
        "scenario_code",
        "actual_class",
        "predicted_class",
    ]

    for column in string_columns:
        dataframe[column] = (
            dataframe[column]
            .astype(str)
            .str.strip()
        )

    numeric_columns = [
        "actual_index",
        "predicted_index",
        "count",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    if dataframe[numeric_columns].isna().any().any():
        invalid_rows = dataframe[
            dataframe[numeric_columns]
            .isna()
            .any(axis=1)
        ]

        raise ValueError(
            "Ditemukan nilai numerik yang tidak valid "
            "pada data confusion matrix.\n"
            f"Jumlah baris tidak valid: {len(invalid_rows)}"
        )

    dataframe[
        "actual_index"
    ] = dataframe[
        "actual_index"
    ].astype(int)

    dataframe[
        "predicted_index"
    ] = dataframe[
        "predicted_index"
    ].astype(int)

    dataframe[
        "count"
    ] = dataframe[
        "count"
    ].astype(int)

    if (
        dataframe["count"]
        < 0
    ).any():
        raise ValueError(
            "Ditemukan nilai count negatif "
            "pada data confusion matrix."
        )

    if (
        dataframe["actual_index"]
        < 0
    ).any():
        raise ValueError(
            "Ditemukan actual_index negatif."
        )

    if (
        dataframe["predicted_index"]
        < 0
    ).any():
        raise ValueError(
            "Ditemukan predicted_index negatif."
        )

    if dataframe.empty:
        raise ValueError(
            "Data confusion matrix kosong."
        )

    return dataframe


# =============================================================================
# VALIDASI EKSPERIMEN
# =============================================================================

def validate_available_experiments(
    confusion_data: pd.DataFrame,
) -> None:
    """
    Memastikan data confusion matrix mencakup seluruh
    10 eksperimen final.
    """

    available_experiments = set(
        confusion_data[
            "experiment_name"
        ].unique()
    )

    expected_experiments = set(
        EXPERIMENT_ORDER
    )

    missing_experiments = (
        expected_experiments
        - available_experiments
    )

    unexpected_experiments = (
        available_experiments
        - expected_experiments
    )

    if missing_experiments:
        raise ValueError(
            "Data confusion matrix belum mencakup "
            "seluruh eksperimen final.\n"
            f"Eksperimen hilang: "
            f"{sorted(missing_experiments)}"
        )

    if unexpected_experiments:
        print(
            "\nPeringatan: ditemukan eksperimen "
            "tambahan yang tidak diproses:"
        )

        for experiment_name in sorted(
            unexpected_experiments
        ):
            print(
                f"- {experiment_name}"
            )


# =============================================================================
# VALIDASI DATA SATU EKSPERIMEN
# =============================================================================

def validate_experiment_data(
    experiment_data: pd.DataFrame,
    experiment_name: str,
) -> None:
    """
    Memvalidasi konsistensi data untuk satu eksperimen.
    """

    if experiment_data.empty:
        raise ValueError(
            f"{experiment_name}: data eksperimen kosong."
        )

    metadata_columns = [
        "model",
        "dataset",
        "scenario_code",
    ]

    for column in metadata_columns:
        unique_values = (
            experiment_data[
                column
            ]
            .dropna()
            .unique()
        )

        if len(unique_values) != 1:
            raise ValueError(
                f"{experiment_name}: kolom {column} "
                "tidak konsisten.\n"
                f"Nilai ditemukan: "
                f"{unique_values.tolist()}"
            )

    actual_indices = set(
        experiment_data[
            "actual_index"
        ].astype(int)
    )

    predicted_indices = set(
        experiment_data[
            "predicted_index"
        ].astype(int)
    )

    expected_indices = set(
        range(
            EXPECTED_NUM_CLASSES
        )
    )

    if actual_indices != expected_indices:
        raise ValueError(
            f"{experiment_name}: indeks kelas aktual "
            "tidak lengkap.\n"
            f"Expected: {sorted(expected_indices)}\n"
            f"Actual  : {sorted(actual_indices)}"
        )

    if predicted_indices != expected_indices:
        raise ValueError(
            f"{experiment_name}: indeks kelas prediksi "
            "tidak lengkap.\n"
            f"Expected: {sorted(expected_indices)}\n"
            f"Actual  : {sorted(predicted_indices)}"
        )

    actual_class_mapping = (
        experiment_data[
            [
                "actual_index",
                "actual_class",
            ]
        ]
        .drop_duplicates()
    )

    actual_mapping_counts = (
        actual_class_mapping
        .groupby(
            "actual_index"
        )[
            "actual_class"
        ]
        .nunique()
    )

    if (
        actual_mapping_counts
        != 1
    ).any():
        raise ValueError(
            f"{experiment_name}: terdapat actual_index "
            "yang memiliki lebih dari satu nama kelas."
        )

    predicted_class_mapping = (
        experiment_data[
            [
                "predicted_index",
                "predicted_class",
            ]
        ]
        .drop_duplicates()
    )

    predicted_mapping_counts = (
        predicted_class_mapping
        .groupby(
            "predicted_index"
        )[
            "predicted_class"
        ]
        .nunique()
    )

    if (
        predicted_mapping_counts
        != 1
    ).any():
        raise ValueError(
            f"{experiment_name}: terdapat predicted_index "
            "yang memiliki lebih dari satu nama kelas."
        )


# =============================================================================
# MEMBENTUK CONFUSION MATRIX
# =============================================================================

def build_confusion_matrix(
    experiment_data: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """
    Membentuk array confusion matrix dan urutan nama kelas.

    Apabila terdapat baris pasangan kelas yang duplikat,
    nilai count akan dijumlahkan menggunakan +=.
    """

    class_information = (
        experiment_data[
            [
                "actual_index",
                "actual_class",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "actual_index"
        )
        .reset_index(
            drop=True
        )
    )

    class_names = (
        class_information[
            "actual_class"
        ]
        .astype(str)
        .tolist()
    )

    class_indices = (
        class_information[
            "actual_index"
        ]
        .astype(int)
        .tolist()
    )

    matrix_size = len(
        class_indices
    )

    if matrix_size != EXPECTED_NUM_CLASSES:
        raise ValueError(
            "Jumlah kelas pada confusion matrix "
            "tidak sesuai.\n"
            f"Expected: {EXPECTED_NUM_CLASSES}\n"
            f"Actual  : {matrix_size}"
        )

    matrix = np.zeros(
        (
            matrix_size,
            matrix_size,
        ),
        dtype=np.int64,
    )

    index_position = {
        class_index: position
        for position, class_index
        in enumerate(class_indices)
    }

    for row in experiment_data.itertuples(
        index=False
    ):
        actual_index = int(
            row.actual_index
        )

        predicted_index = int(
            row.predicted_index
        )

        if actual_index not in index_position:
            raise KeyError(
                "Indeks kelas aktual tidak ditemukan "
                f"dalam mapping: {actual_index}"
            )

        if predicted_index not in index_position:
            raise KeyError(
                "Indeks kelas prediksi tidak ditemukan "
                f"dalam mapping: {predicted_index}"
            )

        actual_position = index_position[
            actual_index
        ]

        predicted_position = index_position[
            predicted_index
        ]

        matrix[
            actual_position,
            predicted_position,
        ] += int(
            row.count
        )

    return (
        matrix,
        class_names,
    )


# =============================================================================
# NORMALISASI CONFUSION MATRIX
# =============================================================================

def normalize_confusion_matrix(
    matrix: np.ndarray,
) -> np.ndarray:
    """
    Menormalisasi confusion matrix berdasarkan jumlah
    data pada setiap kelas aktual.

    Setiap baris memiliki total 1, kecuali kelas
    yang tidak mempunyai data.
    """

    row_totals = matrix.sum(
        axis=1,
        keepdims=True,
    )

    normalized_matrix = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(
            matrix,
            dtype=np.float64,
        ),
        where=row_totals != 0,
    )

    return normalized_matrix


# =============================================================================
# MEMBUAT HEATMAP CONFUSION MATRIX
# =============================================================================

def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    title: str,
    output_path: Path,
    normalized: bool = False,
) -> None:
    """
    Membuat visualisasi confusion matrix menggunakan matplotlib.
    """

    figure, axis = plt.subplots(
        figsize=(8, 7)
    )

    image = axis.imshow(
        matrix,
        interpolation="nearest",
        aspect="auto",
    )

    figure.colorbar(
        image,
        ax=axis,
    )

    tick_positions = np.arange(
        len(class_names)
    )

    axis.set_xticks(
        tick_positions
    )

    axis.set_yticks(
        tick_positions
    )

    axis.set_xticklabels(
        class_names,
        rotation=35,
        ha="right",
    )

    axis.set_yticklabels(
        class_names
    )

    axis.set_xlabel(
        "Predicted Class"
    )

    axis.set_ylabel(
        "Actual Class"
    )

    axis.set_title(
        title
    )

    threshold = (
        float(matrix.max()) / 2.0
        if matrix.size > 0
        else 0.0
    )

    for row_index in range(
        matrix.shape[0]
    ):
        for column_index in range(
            matrix.shape[1]
        ):
            value = matrix[
                row_index,
                column_index,
            ]

            if normalized:
                display_text = (
                    f"{value * 100:.1f}%"
                )
            else:
                display_text = (
                    f"{int(value):,}"
                )

            text_color = (
                "white"
                if value > threshold
                else "black"
            )

            axis.text(
                column_index,
                row_index,
                display_text,
                horizontalalignment="center",
                verticalalignment="center",
                color=text_color,
                fontsize=10,
            )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# =============================================================================
# ANALISIS PERFORMA PER KELAS
# =============================================================================

def analyze_class_performance(
    matrix: np.ndarray,
    class_names: list[str],
    experiment_name: str,
    model_name: str,
    dataset_name: str,
    scenario_code: str,
) -> list[dict[str, Any]]:
    """
    Menghitung jumlah prediksi benar, salah,
    recall, dan kelas yang paling sering tertukar
    untuk setiap kelas aktual.
    """

    rows: list[
        dict[str, Any]
    ] = []

    for class_index, class_name in enumerate(
        class_names
    ):
        total_actual = int(
            matrix[
                class_index,
                :
            ].sum()
        )

        correct = int(
            matrix[
                class_index,
                class_index,
            ]
        )

        incorrect = (
            total_actual
            - correct
        )

        recall = (
            correct
            / total_actual
            if total_actual > 0
            else 0.0
        )

        predicted_counts = (
            matrix[
                class_index,
                :
            ].copy()
        )

        predicted_counts[
            class_index
        ] = 0

        most_confused_index = int(
            np.argmax(
                predicted_counts
            )
        )

        most_confused_count = int(
            predicted_counts[
                most_confused_index
            ]
        )

        if most_confused_count > 0:
            most_confused_class = (
                class_names[
                    most_confused_index
                ]
            )
        else:
            most_confused_class = ""

        rows.append(
            {
                "experiment_name":
                    experiment_name,

                "model":
                    model_name,

                "dataset":
                    dataset_name,

                "scenario_code":
                    scenario_code,

                "class_index":
                    class_index,

                "actual_class":
                    class_name,

                "total_actual":
                    total_actual,

                "correct_predictions":
                    correct,

                "incorrect_predictions":
                    incorrect,

                "class_recall":
                    recall,

                "most_confused_with":
                    most_confused_class,

                "most_confused_count":
                    most_confused_count,
            }
        )

    return rows


# =============================================================================
# ANALISIS PASANGAN KELAS YANG TERTUKAR
# =============================================================================

def analyze_misclassifications(
    matrix: np.ndarray,
    class_names: list[str],
    experiment_name: str,
    model_name: str,
    dataset_name: str,
    scenario_code: str,
) -> list[dict[str, Any]]:
    """
    Menyimpan seluruh pasangan kelas yang mengalami
    salah klasifikasi.
    """

    rows: list[
        dict[str, Any]
    ] = []

    for actual_index, actual_class in enumerate(
        class_names
    ):
        total_actual = int(
            matrix[
                actual_index,
                :
            ].sum()
        )

        for predicted_index, predicted_class in enumerate(
            class_names
        ):
            if actual_index == predicted_index:
                continue

            count = int(
                matrix[
                    actual_index,
                    predicted_index,
                ]
            )

            if count == 0:
                continue

            percentage = (
                count
                / total_actual
                * 100.0
                if total_actual > 0
                else 0.0
            )

            rows.append(
                {
                    "experiment_name":
                        experiment_name,

                    "model":
                        model_name,

                    "dataset":
                        dataset_name,

                    "scenario_code":
                        scenario_code,

                    "actual_index":
                        actual_index,

                    "actual_class":
                        actual_class,

                    "predicted_index":
                        predicted_index,

                    "predicted_class":
                        predicted_class,

                    "misclassification_count":
                        count,

                    "percentage_of_actual_class":
                        percentage,
                }
            )

    return rows


# =============================================================================
# PROGRAM UTAMA
# =============================================================================

def main() -> None:
    """
    Membuat confusion matrix untuk seluruh
    10 eksperimen final.
    """

    print("=" * 80)

    print(
        "STEP 6.3 - CONFUSION MATRICES"
    )

    print("=" * 80)

    create_output_directories()

    confusion_data = load_confusion_data()

    validate_available_experiments(
        confusion_data
    )

    experiments_to_process = (
        EXPERIMENT_ORDER.copy()
    )

    print(
        f"\nJumlah eksperimen: "
        f"{len(experiments_to_process)}"
    )

    print("\nDaftar eksperimen:")

    for number, experiment_name in enumerate(
        experiments_to_process,
        start=1,
    ):
        print(
            f"{number:02d}. "
            f"{experiment_name}"
        )

    summary_rows: list[
        dict[str, Any]
    ] = []

    misclassification_rows: list[
        dict[str, Any]
    ] = []

    status_rows: list[
        dict[str, Any]
    ] = []

    success_count = 0
    failed_count = 0

    for number, experiment_name in enumerate(
        experiments_to_process,
        start=1,
    ):
        print(
            "\n" + "-" * 80
        )

        print(
            f"{number}/{len(experiments_to_process)} "
            f"- {experiment_name}"
        )

        try:
            experiment_data = (
                confusion_data[
                    confusion_data[
                        "experiment_name"
                    ]
                    == experiment_name
                ]
                .copy()
            )

            validate_experiment_data(
                experiment_data=experiment_data,
                experiment_name=experiment_name,
            )

            model_name = str(
                experiment_data[
                    "model"
                ].iloc[0]
            )

            dataset_name = str(
                experiment_data[
                    "dataset"
                ].iloc[0]
            )

            scenario_code = str(
                experiment_data[
                    "scenario_code"
                ].iloc[0]
            )

            matrix, class_names = (
                build_confusion_matrix(
                    experiment_data
                )
            )

            normalized_matrix = (
                normalize_confusion_matrix(
                    matrix
                )
            )

            count_output_path = (
                FIGURES_DIR
                / (
                    f"{experiment_name}"
                    "_confusion_matrix.png"
                )
            )

            normalized_output_path = (
                FIGURES_DIR
                / (
                    f"{experiment_name}"
                    "_confusion_matrix_normalized.png"
                )
            )

            plot_confusion_matrix(
                matrix=matrix,
                class_names=class_names,
                title=(
                    "Confusion Matrix - "
                    f"{experiment_name.upper()}"
                ),
                output_path=count_output_path,
                normalized=False,
            )

            plot_confusion_matrix(
                matrix=normalized_matrix,
                class_names=class_names,
                title=(
                    "Normalized Confusion Matrix - "
                    f"{experiment_name.upper()}"
                ),
                output_path=normalized_output_path,
                normalized=True,
            )

            class_rows = (
                analyze_class_performance(
                    matrix=matrix,
                    class_names=class_names,
                    experiment_name=experiment_name,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    scenario_code=scenario_code,
                )
            )

            summary_rows.extend(
                class_rows
            )

            error_rows = (
                analyze_misclassifications(
                    matrix=matrix,
                    class_names=class_names,
                    experiment_name=experiment_name,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    scenario_code=scenario_code,
                )
            )

            misclassification_rows.extend(
                error_rows
            )

            total_data = int(
                matrix.sum()
            )

            correct_data = int(
                np.trace(
                    matrix
                )
            )

            incorrect_data = (
                total_data
                - correct_data
            )

            accuracy = (
                correct_data
                / total_data
                if total_data > 0
                else 0.0
            )

            status_rows.append(
                {
                    "experiment_name":
                        experiment_name,

                    "model":
                        model_name,

                    "dataset":
                        dataset_name,

                    "scenario_code":
                        scenario_code,

                    "status":
                        "success",

                    "total_data":
                        total_data,

                    "correct_predictions":
                        correct_data,

                    "incorrect_predictions":
                        incorrect_data,

                    "accuracy":
                        accuracy,

                    "error_message":
                        "",
                }
            )

            print(
                f"Dataset            : "
                f"{dataset_name}"
            )

            print(
                f"Model              : "
                f"{model_name}"
            )

            print(
                f"Skenario           : "
                f"{scenario_code}"
            )

            print(
                f"Jumlah data        : "
                f"{total_data:,}"
            )

            print(
                f"Prediksi benar     : "
                f"{correct_data:,}"
            )

            print(
                f"Prediksi salah     : "
                f"{incorrect_data:,}"
            )

            print(
                f"Accuracy           : "
                f"{accuracy:.4f}"
            )

            print(
                f"Matrix jumlah      : "
                f"{count_output_path}"
            )

            print(
                f"Matrix normalized  : "
                f"{normalized_output_path}"
            )

            success_count += 1

        except Exception as error:
            failed_count += 1

            status_rows.append(
                {
                    "experiment_name":
                        experiment_name,

                    "model":
                        "",

                    "dataset":
                        "",

                    "scenario_code":
                        "",

                    "status":
                        "failed",

                    "total_data":
                        None,

                    "correct_predictions":
                        None,

                    "incorrect_predictions":
                        None,

                    "accuracy":
                        None,

                    "error_message":
                        str(error),
                }
            )

            print(
                "Gagal membuat confusion matrix:"
            )

            print(
                str(error)
            )

    # =========================================================================
    # MEMBENTUK DATAFRAME
    # =========================================================================

    summary_dataframe = pd.DataFrame(
        summary_rows
    )

    misclassification_dataframe = pd.DataFrame(
        misclassification_rows
    )

    status_dataframe = pd.DataFrame(
        status_rows
    )

    if not summary_dataframe.empty:
        summary_dataframe = (
            summary_dataframe
            .sort_values(
                [
                    "dataset",
                    "scenario_code",
                    "model",
                    "class_index",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    if not misclassification_dataframe.empty:
        misclassification_dataframe = (
            misclassification_dataframe
            .sort_values(
                [
                    "experiment_name",
                    "misclassification_count",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
        )

    if not status_dataframe.empty:
        status_dataframe = (
            status_dataframe
            .sort_values(
                "experiment_name"
            )
            .reset_index(
                drop=True
            )
        )

    # =========================================================================
    # MENYIMPAN LAPORAN
    # =========================================================================

    summary_dataframe.to_csv(
        CONFUSION_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    misclassification_dataframe.to_csv(
        MISCLASSIFICATION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # MENAMPILKAN RINGKASAN
    # =========================================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "RINGKASAN CONFUSION MATRIX"
    )

    print("=" * 80)

    print(
        f"\nEksperimen berhasil : "
        f"{success_count}"
    )

    print(
        f"Eksperimen gagal    : "
        f"{failed_count}"
    )

    print(
        "\nFolder grafik:"
    )

    print(
        FIGURES_DIR
    )

    print(
        "\nRingkasan performa kelas:"
    )

    print(
        CONFUSION_SUMMARY_PATH
    )

    print(
        "\nAnalisis salah klasifikasi:"
    )

    print(
        MISCLASSIFICATION_PATH
    )

    if not misclassification_dataframe.empty:
        print(
            "\n10 kesalahan klasifikasi terbanyak:"
        )

        top_errors = (
            misclassification_dataframe
            .sort_values(
                "misclassification_count",
                ascending=False,
            )
            .head(10)
            .copy()
        )

        display_columns = [
            "experiment_name",
            "actual_class",
            "predicted_class",
            "misclassification_count",
            "percentage_of_actual_class",
        ]

        top_errors[
            "percentage_of_actual_class"
        ] = (
            top_errors[
                "percentage_of_actual_class"
            ]
            .map(
                lambda value:
                f"{value:.2f}%"
            )
        )

        print(
            "\n"
            + top_errors[
                display_columns
            ].to_string(
                index=False
            )
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "Tahap confusion matrices selesai."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
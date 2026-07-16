# =============================================================================
# STEP 6.3 - CONFUSION MATRICES
# =============================================================================
# File:
# 6_evaluation/03_confusion_matrices.py
#
# Tujuan:
# Membuat confusion matrix untuk 10 eksperimen utama:
# - Kompas: K1, K2, K3
# - AG News: A1, A2
# - Model: CNN dan Attention-BiLSTM
#
# Input:
# 9_results/metrics/model_test_confusion_matrix_data.csv
#
# Output:
# - Confusion matrix jumlah data
# - Confusion matrix normalized per kelas aktual
# - Ringkasan kesalahan klasifikasi
# =============================================================================

from __future__ import annotations

from pathlib import Path

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


# =============================================================================
# MEMBUAT FOLDER OUTPUT
# =============================================================================

def create_output_directories() -> None:
    """
    Membuat folder output jika belum tersedia.
    """

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# MEMUAT DATA CONFUSION MATRIX
# =============================================================================

def load_confusion_data() -> pd.DataFrame:
    """
    Membaca data confusion matrix dari hasil evaluasi test set.
    """

    if not CONFUSION_DATA_PATH.exists():
        raise FileNotFoundError(
            "File confusion matrix tidak ditemukan:\n"
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
            f"Kolom hilang: {missing_columns}"
        )

    dataframe["count"] = pd.to_numeric(
        dataframe["count"],
        errors="coerce",
    ).fillna(0).astype(int)

    return dataframe


# =============================================================================
# MEMBENTUK MATRIX
# =============================================================================

def build_confusion_matrix(
    experiment_data: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """
    Membentuk array confusion matrix dan urutan nama kelas.
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

    matrix = np.zeros(
        (
            matrix_size,
            matrix_size,
        ),
        dtype=int,
    )

    index_position = {
        class_index: position
        for position, class_index
        in enumerate(class_indices)
    }

    for row in experiment_data.itertuples(
        index=False
    ):
        actual_position = index_position[
            int(row.actual_index)
        ]

        predicted_position = index_position[
            int(row.predicted_index)
        ]

        matrix[
            actual_position,
            predicted_position,
        ] = int(row.count)

    return matrix, class_names


# =============================================================================
# NORMALISASI MATRIX
# =============================================================================

def normalize_confusion_matrix(
    matrix: np.ndarray,
) -> np.ndarray:
    """
    Menormalisasi confusion matrix berdasarkan jumlah kelas aktual.

    Setiap baris akan berjumlah 1 atau 100%.
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
            dtype=float,
        ),
        where=row_totals != 0,
    )

    return normalized_matrix


# =============================================================================
# MEMBUAT HEATMAP MANUAL
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

    plt.figure(
        figsize=(8, 7)
    )

    image = plt.imshow(
        matrix,
        interpolation="nearest",
        aspect="auto",
    )

    plt.colorbar(
        image
    )

    tick_positions = np.arange(
        len(class_names)
    )

    plt.xticks(
        tick_positions,
        class_names,
        rotation=35,
        ha="right",
    )

    plt.yticks(
        tick_positions,
        class_names,
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "Actual Class"
    )

    plt.title(
        title
    )

    threshold = (
        matrix.max() / 2
        if matrix.size > 0
        else 0
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

            plt.text(
                column_index,
                row_index,
                display_text,
                horizontalalignment="center",
                verticalalignment="center",
                color=text_color,
                fontsize=10,
            )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# =============================================================================
# ANALISIS KESALAHAN PER KELAS
# =============================================================================

def analyze_class_performance(
    matrix: np.ndarray,
    class_names: list[str],
    experiment_name: str,
    model_name: str,
    dataset_name: str,
    scenario_code: str,
) -> list[dict]:
    """
    Menghitung jumlah benar, salah, dan recall setiap kelas.
    """

    rows = []

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
            correct / total_actual
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
) -> list[dict]:
    """
    Menyimpan seluruh pasangan salah klasifikasi.
    """

    rows = []

    for actual_index, actual_class in enumerate(
        class_names
    ):
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

            total_actual = int(
                matrix[
                    actual_index,
                    :
                ].sum()
            )

            percentage = (
                count / total_actual * 100
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

                    "actual_class":
                        actual_class,

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
    Membuat confusion matrix seluruh eksperimen utama.
    """

    print("=" * 80)
    print(
        "STEP 6.3 - CONFUSION MATRICES"
    )
    print("=" * 80)

    create_output_directories()

    confusion_data = load_confusion_data()

    available_experiments = set(
        confusion_data[
            "experiment_name"
        ].unique()
    )

    experiments_to_process = [
        experiment_name
        for experiment_name
        in EXPERIMENT_ORDER
        if experiment_name
        in available_experiments
    ]

    print(
        f"\nJumlah eksperimen: "
        f"{len(experiments_to_process)}"
    )

    summary_rows = []
    misclassification_rows = []

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

            if experiment_data.empty:
                raise ValueError(
                    "Data eksperimen kosong."
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
                    f"Confusion Matrix - "
                    f"{experiment_name.upper()}"
                ),
                output_path=count_output_path,
                normalized=False,
            )

            plot_confusion_matrix(
                matrix=normalized_matrix,
                class_names=class_names,
                title=(
                    f"Normalized Confusion Matrix - "
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
                correct_data / total_data
                if total_data > 0
                else 0.0
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

            print(
                "Gagal membuat confusion matrix:"
            )

            print(
                str(error)
            )

    summary_dataframe = pd.DataFrame(
        summary_rows
    )

    misclassification_dataframe = pd.DataFrame(
        misclassification_rows
    )

    if not summary_dataframe.empty:
        summary_dataframe = (
            summary_dataframe
            .sort_values(
                [
                    "dataset",
                    "scenario_code",
                    "model",
                    "actual_class",
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

    print("\n" + "=" * 80)
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
        )

        display_columns = [
            "experiment_name",
            "actual_class",
            "predicted_class",
            "misclassification_count",
            "percentage_of_actual_class",
        ]

        print(
            "\n"
            + top_errors[
                display_columns
            ].to_string(
                index=False
            )
        )

    print("\n" + "=" * 80)
    print(
        "Tahap confusion matrices selesai."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
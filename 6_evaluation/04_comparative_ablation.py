# =============================================================================
# STEP 6.4 - COMPARATIVE AND ABLATION ANALYSIS
# =============================================================================
# File:
# 6_evaluation/04_comparative_ablation.py
#
# Tujuan:
# 1. Membandingkan CNN dan Attention-BiLSTM pada skenario yang sama.
# 2. Membandingkan skenario representasi teks.
# 3. Menganalisis perubahan K1 -> K2 dan A1 -> A2.
# 4. Menganalisis pengaruh keyword YAKE melalui K2 -> K3.
# 5. Membandingkan efisiensi waktu inferensi.
#
# Eksperimen final:
# Kompas:
# - K1 = Title, sequence length 20
# - K2 = Title + Description, sequence length 60
# - K3 = Title + Description + Keyword YAKE, sequence length 60
#
# AG News:
# - A1 = Title, sequence length 60
# - A2 = Title + Description, sequence length 60
#
# Model:
# - CNN
# - Attention-BiLSTM
#
# Input:
# 9_results/metrics/model_test_metrics.csv
#
# Output:
# - Tabel perbandingan model
# - Tabel perbandingan skenario
# - Tabel analisis description
# - Tabel analisis YAKE
# - Tabel model terbaik per dataset
# - Tabel efisiensi inferensi
# - Ringkasan temuan penelitian
# - Grafik accuracy, macro F1, description, YAKE, dan inferensi
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

METRICS_PATH = (
    PROJECT_ROOT
    / "9_results"
    / "metrics"
    / "model_test_metrics.csv"
)


# =============================================================================
# PATH OUTPUT
# =============================================================================

TABLES_DIR = (
    PROJECT_ROOT
    / "9_results"
    / "tables"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "9_results"
    / "figures"
    / "comparative_analysis"
)

MODEL_COMPARISON_PATH = (
    TABLES_DIR
    / "model_comparison.csv"
)

SCENARIO_COMPARISON_PATH = (
    TABLES_DIR
    / "scenario_comparison.csv"
)

DESCRIPTION_ANALYSIS_PATH = (
    TABLES_DIR
    / "description_contribution_analysis.csv"
)

YAKE_ANALYSIS_PATH = (
    TABLES_DIR
    / "yake_contribution_analysis.csv"
)

BEST_MODEL_PATH = (
    TABLES_DIR
    / "best_model_summary.csv"
)

EFFICIENCY_ANALYSIS_PATH = (
    TABLES_DIR
    / "inference_efficiency_analysis.csv"
)

FINAL_RESEARCH_SUMMARY_PATH = (
    TABLES_DIR
    / "final_research_findings.csv"
)


# =============================================================================
# KONFIGURASI
# =============================================================================

MODEL_DISPLAY_NAMES = {
    "cnn": "CNN",
    "attention_bilstm": "Attention-BiLSTM",
}

SCENARIO_DISPLAY_NAMES = {
    "K1": "Title",
    "K2": "Title + Description",
    "K3": "Title + Description + Keyword YAKE",
    "A1": "Title",
    "A2": "Title + Description",
}

SCENARIO_ORDER = {
    "K1": 1,
    "K2": 2,
    "K3": 3,
    "A1": 1,
    "A2": 2,
}

SEQUENCE_LENGTHS = {
    "K1": 20,
    "K2": 60,
    "K3": 60,
    "A1": 60,
    "A2": 60,
}

MAIN_EXPERIMENTS = [
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

MODEL_COMPARISON_TOLERANCE = 1e-12

INFERENCE_COMPARISON_TOLERANCE = 1e-6


# =============================================================================
# MEMBUAT FOLDER OUTPUT
# =============================================================================

def create_output_directories() -> None:
    """
    Membuat seluruh folder output yang dibutuhkan.
    """

    directories = [
        TABLES_DIR,
        FIGURES_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# =============================================================================
# MEMUAT DAN MEMVALIDASI METRIK
# =============================================================================

def load_metrics() -> pd.DataFrame:
    """
    Membaca dan memvalidasi hasil evaluasi test set final.
    """

    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            "File metrik evaluasi tidak ditemukan:\n"
            f"{METRICS_PATH}"
        )

    if METRICS_PATH.stat().st_size == 0:
        raise ValueError(
            "File metrik evaluasi ditemukan, tetapi kosong:\n"
            f"{METRICS_PATH}"
        )

    dataframe = pd.read_csv(
        METRICS_PATH,
        encoding="utf-8-sig",
    )

    required_columns = {
        "experiment_name",
        "model",
        "dataset",
        "scenario_code",
        "scenario_name",
        "jumlah_test",
        "correct_predictions",
        "incorrect_predictions",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "log_loss",
        "inference_time_seconds",
        "average_inference_ms_per_sample",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Kolom metrik belum lengkap.\n"
            f"Kolom hilang: {sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    string_columns = [
        "experiment_name",
        "model",
        "dataset",
        "scenario_code",
        "scenario_name",
    ]

    for column in string_columns:
        dataframe[column] = (
            dataframe[column]
            .astype(str)
            .str.strip()
        )

    expected_experiments = set(
        MAIN_EXPERIMENTS
    )

    available_experiments = set(
        dataframe[
            "experiment_name"
        ].tolist()
    )

    missing_experiments = (
        expected_experiments
        - available_experiments
    )

    unexpected_experiments = (
        available_experiments
        - expected_experiments
    )

    duplicate_experiments = (
        dataframe.loc[
            dataframe[
                "experiment_name"
            ].duplicated(
                keep=False
            ),
            "experiment_name",
        ]
        .unique()
        .tolist()
    )

    if missing_experiments:
        raise ValueError(
            "Metrik belum mencakup seluruh eksperimen final.\n"
            f"Eksperimen hilang: "
            f"{sorted(missing_experiments)}"
        )

    if unexpected_experiments:
        raise ValueError(
            "Ditemukan eksperimen di luar konfigurasi final.\n"
            f"Eksperimen tambahan: "
            f"{sorted(unexpected_experiments)}"
        )

    if duplicate_experiments:
        raise ValueError(
            "Ditemukan eksperimen duplikat.\n"
            f"Eksperimen: "
            f"{sorted(duplicate_experiments)}"
        )

    if len(dataframe) != len(MAIN_EXPERIMENTS):
        raise ValueError(
            "Jumlah baris eksperimen tidak sesuai.\n"
            f"Expected: {len(MAIN_EXPERIMENTS)}\n"
            f"Actual  : {len(dataframe)}"
        )

    numeric_columns = [
        "jumlah_test",
        "correct_predictions",
        "incorrect_predictions",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "log_loss",
        "inference_time_seconds",
        "average_inference_ms_per_sample",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    invalid_numeric_rows = dataframe[
        numeric_columns
    ].isna().any(
        axis=1
    )

    if invalid_numeric_rows.any():
        invalid_experiments = (
            dataframe.loc[
                invalid_numeric_rows,
                "experiment_name",
            ]
            .tolist()
        )

        raise ValueError(
            "Ditemukan nilai numerik kosong atau tidak valid.\n"
            f"Eksperimen: {invalid_experiments}"
        )

    integer_columns = [
        "jumlah_test",
        "correct_predictions",
        "incorrect_predictions",
    ]

    for column in integer_columns:
        non_integer = (
            dataframe[column]
            % 1
            != 0
        )

        if non_integer.any():
            invalid_experiments = (
                dataframe.loc[
                    non_integer,
                    "experiment_name",
                ]
                .tolist()
            )

            raise ValueError(
                f"Kolom {column} harus berupa bilangan bulat.\n"
                f"Eksperimen: {invalid_experiments}"
            )

        dataframe[column] = (
            dataframe[column]
            .astype(int)
        )

    if (
        dataframe[
            integer_columns
        ]
        < 0
    ).any().any():
        raise ValueError(
            "Jumlah data dan jumlah prediksi tidak boleh negatif."
        )

    invalid_total_rows = (
        dataframe[
            "correct_predictions"
        ]
        + dataframe[
            "incorrect_predictions"
        ]
        != dataframe[
            "jumlah_test"
        ]
    )

    if invalid_total_rows.any():
        invalid_experiments = (
            dataframe.loc[
                invalid_total_rows,
                "experiment_name",
            ]
            .tolist()
        )

        raise ValueError(
            "Jumlah prediksi benar dan salah tidak sama "
            "dengan jumlah test.\n"
            f"Eksperimen: {invalid_experiments}"
        )

    bounded_metric_columns = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    ]

    for column in bounded_metric_columns:
        invalid_values = (
            (dataframe[column] < 0.0)
            | (dataframe[column] > 1.0)
        )

        if invalid_values.any():
            invalid_experiments = (
                dataframe.loc[
                    invalid_values,
                    "experiment_name",
                ]
                .tolist()
            )

            raise ValueError(
                f"Ditemukan nilai {column} di luar rentang 0 sampai 1.\n"
                f"Eksperimen: {invalid_experiments}"
            )

    if (
        dataframe[
            "log_loss"
        ]
        < 0.0
    ).any():
        raise ValueError(
            "Ditemukan nilai log loss negatif."
        )

    if (
        dataframe[
            "inference_time_seconds"
        ]
        <= 0.0
    ).any():
        raise ValueError(
            "Waktu inferensi harus lebih besar dari nol."
        )

    if (
        dataframe[
            "average_inference_ms_per_sample"
        ]
        <= 0.0
    ).any():
        raise ValueError(
            "Rata-rata waktu inferensi per sampel "
            "harus lebih besar dari nol."
        )

    calculated_accuracy = np.divide(
        dataframe[
            "correct_predictions"
        ].to_numpy(
            dtype=float
        ),
        dataframe[
            "jumlah_test"
        ].to_numpy(
            dtype=float
        ),
        out=np.zeros(
            len(dataframe),
            dtype=float,
        ),
        where=(
            dataframe[
                "jumlah_test"
            ].to_numpy(
                dtype=float
            )
            != 0
        ),
    )

    accuracy_mismatch = ~np.isclose(
        dataframe[
            "accuracy"
        ].to_numpy(
            dtype=float
        ),
        calculated_accuracy,
        atol=1e-12,
    )

    if accuracy_mismatch.any():
        invalid_experiments = (
            dataframe.loc[
                accuracy_mismatch,
                "experiment_name",
            ]
            .tolist()
        )

        raise ValueError(
            "Accuracy tidak konsisten dengan jumlah prediksi benar.\n"
            f"Eksperimen: {invalid_experiments}"
        )

    expected_models = set(
        MODEL_DISPLAY_NAMES.keys()
    )

    actual_models = set(
        dataframe[
            "model"
        ].unique()
    )

    if actual_models != expected_models:
        raise ValueError(
            "Daftar model pada metrik tidak sesuai.\n"
            f"Expected: {sorted(expected_models)}\n"
            f"Actual  : {sorted(actual_models)}"
        )

    expected_scenarios = set(
        SCENARIO_DISPLAY_NAMES.keys()
    )

    actual_scenarios = set(
        dataframe[
            "scenario_code"
        ].unique()
    )

    if actual_scenarios != expected_scenarios:
        raise ValueError(
            "Daftar skenario pada metrik tidak sesuai.\n"
            f"Expected: {sorted(expected_scenarios)}\n"
            f"Actual  : {sorted(actual_scenarios)}"
        )

    dataframe["model_display"] = (
        dataframe["model"]
        .map(MODEL_DISPLAY_NAMES)
    )

    dataframe["scenario_display"] = (
        dataframe["scenario_code"]
        .map(SCENARIO_DISPLAY_NAMES)
    )

    dataframe["scenario_order"] = (
        dataframe["scenario_code"]
        .map(SCENARIO_ORDER)
    )

    dataframe["sequence_length"] = (
        dataframe["scenario_code"]
        .map(SEQUENCE_LENGTHS)
        .astype(int)
    )

    dataframe = (
        dataframe
        .sort_values(
            [
                "dataset",
                "scenario_order",
                "model_display",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return dataframe


# =============================================================================
# PERBANDINGAN MODEL
# =============================================================================

def build_model_comparison(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Membandingkan CNN dan Attention-BiLSTM
    pada skenario yang sama.
    """

    rows: list[
        dict[str, Any]
    ] = []

    groups = metrics.groupby(
        [
            "dataset",
            "scenario_code",
            "scenario_display",
        ],
        sort=False,
    )

    for (
        dataset_name,
        scenario_code,
        scenario_name,
    ), group in groups:

        cnn_rows = group[
            group["model"]
            == "cnn"
        ]

        bilstm_rows = group[
            group["model"]
            == "attention_bilstm"
        ]

        if (
            len(cnn_rows) != 1
            or len(bilstm_rows) != 1
        ):
            raise ValueError(
                "Perbandingan model tidak lengkap untuk "
                f"{dataset_name} {scenario_code}."
            )

        cnn_row = cnn_rows.iloc[0]
        bilstm_row = bilstm_rows.iloc[0]

        accuracy_difference = (
            float(
                cnn_row[
                    "accuracy"
                ]
            )
            - float(
                bilstm_row[
                    "accuracy"
                ]
            )
        )

        f1_difference = (
            float(
                cnn_row[
                    "f1_macro"
                ]
            )
            - float(
                bilstm_row[
                    "f1_macro"
                ]
            )
        )

        inference_difference_ms = (
            float(
                cnn_row[
                    "average_inference_ms_per_sample"
                ]
            )
            - float(
                bilstm_row[
                    "average_inference_ms_per_sample"
                ]
            )
        )

        if np.isclose(
            accuracy_difference,
            0.0,
            atol=MODEL_COMPARISON_TOLERANCE,
        ):
            accuracy_winner = "Sama"

        elif accuracy_difference > 0.0:
            accuracy_winner = "CNN"

        else:
            accuracy_winner = "Attention-BiLSTM"

        if np.isclose(
            f1_difference,
            0.0,
            atol=MODEL_COMPARISON_TOLERANCE,
        ):
            f1_winner = "Sama"

        elif f1_difference > 0.0:
            f1_winner = "CNN"

        else:
            f1_winner = "Attention-BiLSTM"

        if np.isclose(
            inference_difference_ms,
            0.0,
            atol=INFERENCE_COMPARISON_TOLERANCE,
        ):
            faster_model = "Sama"

        elif inference_difference_ms < 0.0:
            faster_model = "CNN"

        else:
            faster_model = "Attention-BiLSTM"

        rows.append(
            {
                "dataset":
                    dataset_name,

                "scenario_code":
                    scenario_code,

                "scenario_name":
                    scenario_name,

                "sequence_length":
                    int(
                        cnn_row[
                            "sequence_length"
                        ]
                    ),

                "cnn_accuracy":
                    float(
                        cnn_row[
                            "accuracy"
                        ]
                    ),

                "attention_bilstm_accuracy":
                    float(
                        bilstm_row[
                            "accuracy"
                        ]
                    ),

                "accuracy_difference_cnn_minus_bilstm":
                    accuracy_difference,

                "accuracy_difference_percentage_point":
                    accuracy_difference
                    * 100.0,

                "accuracy_winner":
                    accuracy_winner,

                "cnn_f1_macro":
                    float(
                        cnn_row[
                            "f1_macro"
                        ]
                    ),

                "attention_bilstm_f1_macro":
                    float(
                        bilstm_row[
                            "f1_macro"
                        ]
                    ),

                "f1_difference_cnn_minus_bilstm":
                    f1_difference,

                "f1_difference_percentage_point":
                    f1_difference
                    * 100.0,

                "f1_winner":
                    f1_winner,

                "cnn_inference_ms_per_sample":
                    float(
                        cnn_row[
                            "average_inference_ms_per_sample"
                        ]
                    ),

                "attention_bilstm_inference_ms_per_sample":
                    float(
                        bilstm_row[
                            "average_inference_ms_per_sample"
                        ]
                    ),

                "inference_difference_ms_cnn_minus_bilstm":
                    inference_difference_ms,

                "faster_model":
                    faster_model,
            }
        )

    result = pd.DataFrame(
        rows
    )

    if not result.empty:
        result["scenario_order"] = (
            result["scenario_code"]
            .map(SCENARIO_ORDER)
        )

        result = (
            result
            .sort_values(
                [
                    "dataset",
                    "scenario_order",
                ]
            )
            .drop(
                columns=[
                    "scenario_order"
                ]
            )
            .reset_index(
                drop=True
            )
        )

    return result


# =============================================================================
# PERBANDINGAN SKENARIO
# =============================================================================

def build_scenario_comparison(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Menyusun tabel performa setiap model dan skenario.
    """

    columns = [
        "experiment_name",
        "model_display",
        "dataset",
        "scenario_code",
        "scenario_display",
        "sequence_length",
        "jumlah_test",
        "correct_predictions",
        "incorrect_predictions",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "log_loss",
        "inference_time_seconds",
        "average_inference_ms_per_sample",
        "scenario_order",
    ]

    result = metrics[
        columns
    ].copy()

    result = result.rename(
        columns={
            "model_display":
                "model",

            "scenario_display":
                "scenario_name",
        }
    )

    result = (
        result
        .sort_values(
            [
                "dataset",
                "model",
                "scenario_order",
            ]
        )
        .drop(
            columns=[
                "scenario_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return result


# =============================================================================
# ANALISIS DESCRIPTION
# =============================================================================

def build_description_analysis(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Menganalisis perubahan K1 -> K2 dan A1 -> A2.

    Catatan metodologis:
    - Kompas K1 dan K2 memiliki sequence length berbeda,
      sehingga kontribusi description belum sepenuhnya terisolasi.
    - AG News A1 dan A2 memiliki sequence length sama,
      sehingga perbandingannya lebih terkontrol.
    """

    comparison_pairs = [
        {
            "dataset": "Kompas",
            "baseline": "K1",
            "treatment": "K2",
            "comparison": (
                "Title vs Title + Description"
            ),
            "baseline_sequence_length": 20,
            "treatment_sequence_length": 60,
            "controlled_sequence_length": False,
        },
        {
            "dataset": "AG News",
            "baseline": "A1",
            "treatment": "A2",
            "comparison": (
                "Title vs Title + Description"
            ),
            "baseline_sequence_length": 60,
            "treatment_sequence_length": 60,
            "controlled_sequence_length": True,
        },
    ]

    rows: list[
        dict[str, Any]
    ] = []

    for pair in comparison_pairs:
        dataset_name = pair[
            "dataset"
        ]

        for model_name in [
            "cnn",
            "attention_bilstm",
        ]:
            baseline_rows = metrics[
                (
                    metrics["dataset"]
                    == dataset_name
                )
                & (
                    metrics["scenario_code"]
                    == pair["baseline"]
                )
                & (
                    metrics["model"]
                    == model_name
                )
            ]

            treatment_rows = metrics[
                (
                    metrics["dataset"]
                    == dataset_name
                )
                & (
                    metrics["scenario_code"]
                    == pair["treatment"]
                )
                & (
                    metrics["model"]
                    == model_name
                )
            ]

            if (
                len(baseline_rows) != 1
                or len(treatment_rows) != 1
            ):
                raise ValueError(
                    "Data analisis description tidak lengkap untuk "
                    f"{dataset_name} {model_name}."
                )

            baseline_row = (
                baseline_rows.iloc[0]
            )

            treatment_row = (
                treatment_rows.iloc[0]
            )

            accuracy_change = (
                float(
                    treatment_row[
                        "accuracy"
                    ]
                )
                - float(
                    baseline_row[
                        "accuracy"
                    ]
                )
            )

            precision_change = (
                float(
                    treatment_row[
                        "precision_macro"
                    ]
                )
                - float(
                    baseline_row[
                        "precision_macro"
                    ]
                )
            )

            recall_change = (
                float(
                    treatment_row[
                        "recall_macro"
                    ]
                )
                - float(
                    baseline_row[
                        "recall_macro"
                    ]
                )
            )

            f1_change = (
                float(
                    treatment_row[
                        "f1_macro"
                    ]
                )
                - float(
                    baseline_row[
                        "f1_macro"
                    ]
                )
            )

            log_loss_change = (
                float(
                    treatment_row[
                        "log_loss"
                    ]
                )
                - float(
                    baseline_row[
                        "log_loss"
                    ]
                )
            )

            error_before = int(
                baseline_row[
                    "incorrect_predictions"
                ]
            )

            error_after = int(
                treatment_row[
                    "incorrect_predictions"
                ]
            )

            error_reduction = (
                error_before
                - error_after
            )

            relative_error_reduction_percent = (
                error_reduction
                / error_before
                * 100.0
                if error_before > 0
                else 0.0
            )

            if accuracy_change > 0.0:
                if pair[
                    "controlled_sequence_length"
                ]:
                    interpretation = (
                        "Penambahan description meningkatkan performa "
                        "pada sequence length yang sama."
                    )

                else:
                    interpretation = (
                        "Representasi Title + Description menghasilkan "
                        "performa lebih tinggi, tetapi perubahan juga "
                        "disertai peningkatan sequence length sehingga "
                        "kontribusi description belum sepenuhnya terisolasi."
                    )

            elif accuracy_change < 0.0:
                if pair[
                    "controlled_sequence_length"
                ]:
                    interpretation = (
                        "Penambahan description menurunkan performa "
                        "pada sequence length yang sama."
                    )

                else:
                    interpretation = (
                        "Representasi treatment menghasilkan performa "
                        "lebih rendah, tetapi sequence length juga berubah."
                    )

            else:
                interpretation = (
                    "Tidak ditemukan perubahan accuracy."
                )

            rows.append(
                {
                    "dataset":
                        dataset_name,

                    "model":
                        MODEL_DISPLAY_NAMES[
                            model_name
                        ],

                    "comparison":
                        pair[
                            "comparison"
                        ],

                    "baseline_scenario":
                        pair[
                            "baseline"
                        ],

                    "treatment_scenario":
                        pair[
                            "treatment"
                        ],

                    "baseline_sequence_length":
                        pair[
                            "baseline_sequence_length"
                        ],

                    "treatment_sequence_length":
                        pair[
                            "treatment_sequence_length"
                        ],

                    "controlled_sequence_length":
                        pair[
                            "controlled_sequence_length"
                        ],

                    "baseline_accuracy":
                        float(
                            baseline_row[
                                "accuracy"
                            ]
                        ),

                    "treatment_accuracy":
                        float(
                            treatment_row[
                                "accuracy"
                            ]
                        ),

                    "accuracy_change":
                        accuracy_change,

                    "accuracy_change_percentage_point":
                        accuracy_change
                        * 100.0,

                    "precision_macro_change":
                        precision_change,

                    "recall_macro_change":
                        recall_change,

                    "baseline_f1_macro":
                        float(
                            baseline_row[
                                "f1_macro"
                            ]
                        ),

                    "treatment_f1_macro":
                        float(
                            treatment_row[
                                "f1_macro"
                            ]
                        ),

                    "f1_change":
                        f1_change,

                    "log_loss_change":
                        log_loss_change,

                    "incorrect_before":
                        error_before,

                    "incorrect_after":
                        error_after,

                    "error_reduction":
                        error_reduction,

                    "relative_error_reduction_percent":
                        relative_error_reduction_percent,

                    "interpretation":
                        interpretation,
                }
            )

    result = pd.DataFrame(
        rows
    )

    if not result.empty:
        result = (
            result
            .sort_values(
                [
                    "dataset",
                    "model",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    return result


# =============================================================================
# ANALISIS YAKE
# =============================================================================

def build_yake_analysis(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Membandingkan Kompas K2 dan K3.

    K2 dan K3 menggunakan split, vocabulary,
    dan sequence length yang sama sehingga
    perbandingan lebih terkontrol.
    """

    rows: list[
        dict[str, Any]
    ] = []

    for model_name in [
        "cnn",
        "attention_bilstm",
    ]:
        k2_rows = metrics[
            (
                metrics["dataset"]
                == "Kompas"
            )
            & (
                metrics["scenario_code"]
                == "K2"
            )
            & (
                metrics["model"]
                == model_name
            )
        ]

        k3_rows = metrics[
            (
                metrics["dataset"]
                == "Kompas"
            )
            & (
                metrics["scenario_code"]
                == "K3"
            )
            & (
                metrics["model"]
                == model_name
            )
        ]

        if (
            len(k2_rows) != 1
            or len(k3_rows) != 1
        ):
            raise ValueError(
                "Data analisis YAKE tidak lengkap untuk "
                f"{model_name}."
            )

        k2_row = k2_rows.iloc[0]

        k3_row = k3_rows.iloc[0]

        accuracy_change = (
            float(
                k3_row[
                    "accuracy"
                ]
            )
            - float(
                k2_row[
                    "accuracy"
                ]
            )
        )

        precision_change = (
            float(
                k3_row[
                    "precision_macro"
                ]
            )
            - float(
                k2_row[
                    "precision_macro"
                ]
            )
        )

        recall_change = (
            float(
                k3_row[
                    "recall_macro"
                ]
            )
            - float(
                k2_row[
                    "recall_macro"
                ]
            )
        )

        f1_change = (
            float(
                k3_row[
                    "f1_macro"
                ]
            )
            - float(
                k2_row[
                    "f1_macro"
                ]
            )
        )

        log_loss_change = (
            float(
                k3_row[
                    "log_loss"
                ]
            )
            - float(
                k2_row[
                    "log_loss"
                ]
            )
        )

        incorrect_k2 = int(
            k2_row[
                "incorrect_predictions"
            ]
        )

        incorrect_k3 = int(
            k3_row[
                "incorrect_predictions"
            ]
        )

        additional_errors = (
            incorrect_k3
            - incorrect_k2
        )

        if accuracy_change > 0.0:
            conclusion = (
                "Penambahan keyword YAKE meningkatkan accuracy."
            )

        elif accuracy_change < 0.0:
            conclusion = (
                "Penambahan keyword YAKE belum meningkatkan accuracy."
            )

        else:
            conclusion = (
                "Penambahan keyword YAKE tidak mengubah accuracy."
            )

        rows.append(
            {
                "dataset":
                    "Kompas",

                "model":
                    MODEL_DISPLAY_NAMES[
                        model_name
                    ],

                "baseline_scenario":
                    "K2",

                "baseline_representation":
                    "Title + Description",

                "yake_scenario":
                    "K3",

                "yake_representation":
                    (
                        "Title + Description "
                        "+ Keyword YAKE"
                    ),

                "controlled_sequence_length":
                    True,

                "sequence_length":
                    60,

                "k2_accuracy":
                    float(
                        k2_row[
                            "accuracy"
                        ]
                    ),

                "k3_accuracy":
                    float(
                        k3_row[
                            "accuracy"
                        ]
                    ),

                "accuracy_change":
                    accuracy_change,

                "accuracy_change_percentage_point":
                    accuracy_change
                    * 100.0,

                "precision_macro_change":
                    precision_change,

                "recall_macro_change":
                    recall_change,

                "f1_macro_change":
                    f1_change,

                "log_loss_change":
                    log_loss_change,

                "k2_incorrect_predictions":
                    incorrect_k2,

                "k3_incorrect_predictions":
                    incorrect_k3,

                "additional_errors_after_yake":
                    additional_errors,

                "conclusion":
                    conclusion,
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# MODEL TERBAIK PER DATASET
# =============================================================================

def build_best_model_summary(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mencari model terbaik pada setiap dataset.
    """

    rows: list[
        dict[str, Any]
    ] = []

    for dataset_name in sorted(
        metrics[
            "dataset"
        ].unique()
    ):
        dataset_metrics = metrics[
            metrics["dataset"]
            == dataset_name
        ].copy()

        if dataset_metrics.empty:
            continue

        best_accuracy_row = (
            dataset_metrics
            .sort_values(
                [
                    "accuracy",
                    "f1_macro",
                    "log_loss",
                    "average_inference_ms_per_sample",
                ],
                ascending=[
                    False,
                    False,
                    True,
                    True,
                ],
            )
            .iloc[0]
        )

        best_f1_row = (
            dataset_metrics
            .sort_values(
                [
                    "f1_macro",
                    "accuracy",
                    "log_loss",
                    "average_inference_ms_per_sample",
                ],
                ascending=[
                    False,
                    False,
                    True,
                    True,
                ],
            )
            .iloc[0]
        )

        lowest_log_loss_row = (
            dataset_metrics
            .sort_values(
                [
                    "log_loss",
                    "accuracy",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .iloc[0]
        )

        fastest_row = (
            dataset_metrics
            .sort_values(
                [
                    "average_inference_ms_per_sample",
                    "accuracy",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .iloc[0]
        )

        rows.append(
            {
                "dataset":
                    dataset_name,

                "best_accuracy_experiment":
                    best_accuracy_row[
                        "experiment_name"
                    ],

                "best_accuracy_model":
                    best_accuracy_row[
                        "model_display"
                    ],

                "best_accuracy_scenario":
                    best_accuracy_row[
                        "scenario_code"
                    ],

                "best_accuracy_representation":
                    best_accuracy_row[
                        "scenario_display"
                    ],

                "best_accuracy":
                    float(
                        best_accuracy_row[
                            "accuracy"
                        ]
                    ),

                "best_accuracy_f1_macro":
                    float(
                        best_accuracy_row[
                            "f1_macro"
                        ]
                    ),

                "best_f1_experiment":
                    best_f1_row[
                        "experiment_name"
                    ],

                "best_f1_model":
                    best_f1_row[
                        "model_display"
                    ],

                "best_f1_scenario":
                    best_f1_row[
                        "scenario_code"
                    ],

                "best_f1_macro":
                    float(
                        best_f1_row[
                            "f1_macro"
                        ]
                    ),

                "lowest_log_loss_experiment":
                    lowest_log_loss_row[
                        "experiment_name"
                    ],

                "lowest_log_loss":
                    float(
                        lowest_log_loss_row[
                            "log_loss"
                        ]
                    ),

                "fastest_experiment":
                    fastest_row[
                        "experiment_name"
                    ],

                "fastest_model":
                    fastest_row[
                        "model_display"
                    ],

                "fastest_scenario":
                    fastest_row[
                        "scenario_code"
                    ],

                "fastest_inference_ms_per_sample":
                    float(
                        fastest_row[
                            "average_inference_ms_per_sample"
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# ANALISIS EFISIENSI
# =============================================================================

def build_efficiency_analysis(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Membandingkan efisiensi inferensi seluruh eksperimen.
    """

    result = metrics[
        [
            "experiment_name",
            "model_display",
            "dataset",
            "scenario_code",
            "scenario_display",
            "sequence_length",
            "jumlah_test",
            "accuracy",
            "f1_macro",
            "inference_time_seconds",
            "average_inference_ms_per_sample",
        ]
    ].copy()

    result = result.rename(
        columns={
            "model_display":
                "model",

            "scenario_display":
                "scenario_name",
        }
    )

    result["samples_per_second"] = (
        result[
            "jumlah_test"
        ]
        / result[
            "inference_time_seconds"
        ]
    )

    result = (
        result
        .sort_values(
            [
                "average_inference_ms_per_sample",
                "accuracy",
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

    result["efficiency_rank"] = np.arange(
        1,
        len(result) + 1,
    )

    return result


# =============================================================================
# TEMUAN PENELITIAN
# =============================================================================

def build_final_findings(
    description_analysis: pd.DataFrame,
    yake_analysis: pd.DataFrame,
    best_model_summary: pd.DataFrame,
    model_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """
    Membentuk ringkasan temuan utama penelitian
    secara otomatis.
    """

    findings: list[
        dict[str, Any]
    ] = []

    for row in description_analysis.itertuples(
        index=False
    ):
        findings.append(
            {
                "finding_category":
                    "Description Contribution",

                "dataset":
                    row.dataset,

                "model":
                    row.model,

                "finding":
                    (
                        "Perbandingan "
                        f"{row.baseline_scenario} ke "
                        f"{row.treatment_scenario} mengubah accuracy "
                        f"dari {row.baseline_accuracy:.4f} menjadi "
                        f"{row.treatment_accuracy:.4f}."
                    ),

                "change_percentage_point":
                    row.accuracy_change_percentage_point,

                "controlled_comparison":
                    row.controlled_sequence_length,

                "interpretation":
                    row.interpretation,
            }
        )

    for row in yake_analysis.itertuples(
        index=False
    ):
        findings.append(
            {
                "finding_category":
                    "YAKE Contribution",

                "dataset":
                    "Kompas",

                "model":
                    row.model,

                "finding":
                    (
                        "Penambahan keyword YAKE mengubah accuracy "
                        f"dari {row.k2_accuracy:.4f} pada K2 menjadi "
                        f"{row.k3_accuracy:.4f} pada K3."
                    ),

                "change_percentage_point":
                    row.accuracy_change_percentage_point,

                "controlled_comparison":
                    row.controlled_sequence_length,

                "interpretation":
                    row.conclusion,
            }
        )

    for row in best_model_summary.itertuples(
        index=False
    ):
        findings.append(
            {
                "finding_category":
                    "Best Model",

                "dataset":
                    row.dataset,

                "model":
                    row.best_accuracy_model,

                "finding":
                    (
                        "Model terbaik berdasarkan accuracy adalah "
                        f"{row.best_accuracy_experiment} dengan accuracy "
                        f"{row.best_accuracy:.4f}."
                    ),

                "change_percentage_point":
                    np.nan,

                "controlled_comparison":
                    np.nan,

                "interpretation":
                    (
                        "Model terbaik ditentukan secara terpisah "
                        "untuk setiap dataset berdasarkan test set."
                    ),
            }
        )

    winner_counts = {
        "CNN": 0,
        "Attention-BiLSTM": 0,
        "Sama": 0,
    }

    for winner in model_comparison[
        "accuracy_winner"
    ]:
        winner_counts[winner] = (
            winner_counts.get(
                winner,
                0,
            )
            + 1
        )

    findings.append(
        {
            "finding_category":
                "Model Comparison",

            "dataset":
                "All",

            "model":
                "CNN vs Attention-BiLSTM",

            "finding":
                (
                    f"CNN unggul pada {winner_counts['CNN']} skenario, "
                    "Attention-BiLSTM unggul pada "
                    f"{winner_counts['Attention-BiLSTM']} skenario, "
                    f"dan seri pada {winner_counts['Sama']} skenario."
                ),

            "change_percentage_point":
                np.nan,

            "controlled_comparison":
                True,

            "interpretation":
                (
                    "Perbandingan arsitektur dilakukan pada skenario "
                    "representasi dan test set yang sama."
                ),
        }
    )

    return pd.DataFrame(
        findings
    )


# =============================================================================
# UTILITAS GRAFIK
# =============================================================================

def add_bar_value_labels(
    axis: plt.Axes,
    decimals: int = 2,
    suffix: str = "",
) -> None:
    """
    Menambahkan label nilai pada setiap batang grafik.
    """

    for patch in axis.patches:
        height = patch.get_height()

        axis.annotate(
            f"{height:.{decimals}f}{suffix}",
            xy=(
                patch.get_x()
                + patch.get_width() / 2.0,
                height,
            ),
            xytext=(
                0,
                4,
            ),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


# =============================================================================
# GRAFIK PERBANDINGAN ACCURACY
# =============================================================================

def plot_accuracy_comparison(
    metrics: pd.DataFrame,
) -> Path:
    """
    Membuat grafik accuracy seluruh eksperimen.
    """

    output_path = (
        FIGURES_DIR
        / "accuracy_comparison.png"
    )

    plot_data = metrics.copy()

    plot_data["label"] = (
        plot_data["dataset"]
        + "\n"
        + plot_data["model_display"]
        + " "
        + plot_data["scenario_code"]
    )

    plot_data = (
        plot_data
        .sort_values(
            [
                "dataset",
                "scenario_order",
                "model_display",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    positions = np.arange(
        len(plot_data)
    )

    figure, axis = plt.subplots(
        figsize=(13, 6)
    )

    axis.bar(
        positions,
        plot_data["accuracy"]
        * 100.0,
    )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        plot_data["label"],
        rotation=40,
        ha="right",
    )

    axis.set_ylabel(
        "Accuracy (%)"
    )

    axis.set_xlabel(
        "Eksperimen"
    )

    axis.set_title(
        "Perbandingan Accuracy pada Test Set"
    )

    minimum_value = float(
        (
            plot_data["accuracy"]
            * 100.0
        ).min()
    )

    axis.set_ylim(
        bottom=max(
            0.0,
            minimum_value - 5.0,
        ),
        top=100.5,
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    add_bar_value_labels(
        axis,
        decimals=2,
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

    return output_path


# =============================================================================
# GRAFIK PERBANDINGAN MACRO F1
# =============================================================================

def plot_f1_comparison(
    metrics: pd.DataFrame,
) -> Path:
    """
    Membuat grafik macro F1 seluruh eksperimen.
    """

    output_path = (
        FIGURES_DIR
        / "f1_macro_comparison.png"
    )

    plot_data = metrics.copy()

    plot_data["label"] = (
        plot_data["dataset"]
        + "\n"
        + plot_data["model_display"]
        + " "
        + plot_data["scenario_code"]
    )

    plot_data = (
        plot_data
        .sort_values(
            [
                "dataset",
                "scenario_order",
                "model_display",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    positions = np.arange(
        len(plot_data)
    )

    figure, axis = plt.subplots(
        figsize=(13, 6)
    )

    axis.bar(
        positions,
        plot_data["f1_macro"]
        * 100.0,
    )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        plot_data["label"],
        rotation=40,
        ha="right",
    )

    axis.set_ylabel(
        "Macro F1-score (%)"
    )

    axis.set_xlabel(
        "Eksperimen"
    )

    axis.set_title(
        "Perbandingan Macro F1-score pada Test Set"
    )

    minimum_value = float(
        (
            plot_data["f1_macro"]
            * 100.0
        ).min()
    )

    axis.set_ylim(
        bottom=max(
            0.0,
            minimum_value - 5.0,
        ),
        top=100.5,
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    add_bar_value_labels(
        axis,
        decimals=2,
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

    return output_path


# =============================================================================
# GRAFIK PENGARUH DESCRIPTION
# =============================================================================

def plot_description_contribution(
    description_analysis: pd.DataFrame,
) -> Path:
    """
    Membuat grafik perubahan accuracy dari
    baseline ke treatment.
    """

    if description_analysis.empty:
        raise ValueError(
            "Data analisis description kosong."
        )

    output_path = (
        FIGURES_DIR
        / "description_contribution.png"
    )

    plot_data = (
        description_analysis
        .copy()
        .sort_values(
            [
                "dataset",
                "model",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    plot_data["label"] = (
        plot_data["dataset"]
        + "\n"
        + plot_data["model"]
    )

    positions = np.arange(
        len(plot_data)
    )

    bar_width = 0.35

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    axis.bar(
        positions
        - bar_width / 2.0,
        plot_data[
            "baseline_accuracy"
        ]
        * 100.0,
        width=bar_width,
        label="Title",
    )

    axis.bar(
        positions
        + bar_width / 2.0,
        plot_data[
            "treatment_accuracy"
        ]
        * 100.0,
        width=bar_width,
        label="Title + Description",
    )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        plot_data["label"]
    )

    axis.set_ylabel(
        "Accuracy (%)"
    )

    axis.set_xlabel(
        "Dataset dan Model"
    )

    axis.set_title(
        "Perbandingan Title dan Title + Description"
    )

    minimum_value = float(
        min(
            (
                plot_data[
                    "baseline_accuracy"
                ]
                * 100.0
            ).min(),
            (
                plot_data[
                    "treatment_accuracy"
                ]
                * 100.0
            ).min(),
        )
    )

    axis.set_ylim(
        bottom=max(
            0.0,
            minimum_value - 5.0,
        ),
        top=100.5,
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    axis.legend()

    add_bar_value_labels(
        axis,
        decimals=2,
    )

    figure.text(
        0.5,
        -0.02,
        (
            "Catatan: Kompas K1 dan K2 memiliki sequence length berbeda; "
            "AG News A1 dan A2 memiliki sequence length sama."
        ),
        ha="center",
        fontsize=9,
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

    return output_path


# =============================================================================
# GRAFIK PENGARUH YAKE
# =============================================================================

def plot_yake_contribution(
    yake_analysis: pd.DataFrame,
) -> Path:
    """
    Membuat grafik K2 dibandingkan dengan K3.
    """

    if yake_analysis.empty:
        raise ValueError(
            "Data analisis YAKE kosong."
        )

    output_path = (
        FIGURES_DIR
        / "yake_contribution.png"
    )

    plot_data = (
        yake_analysis
        .copy()
    )

    positions = np.arange(
        len(plot_data)
    )

    bar_width = 0.35

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.bar(
        positions
        - bar_width / 2.0,
        plot_data[
            "k2_accuracy"
        ]
        * 100.0,
        width=bar_width,
        label=(
            "K2: Title + Description"
        ),
    )

    axis.bar(
        positions
        + bar_width / 2.0,
        plot_data[
            "k3_accuracy"
        ]
        * 100.0,
        width=bar_width,
        label=(
            "K3: + Keyword YAKE"
        ),
    )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        plot_data["model"]
    )

    axis.set_ylabel(
        "Accuracy (%)"
    )

    axis.set_xlabel(
        "Model"
    )

    axis.set_title(
        "Analisis Pengaruh Keyword YAKE\n"
        "K2 dan K3 Menggunakan Sequence Length 60"
    )

    minimum_value = float(
        min(
            (
                plot_data[
                    "k2_accuracy"
                ]
                * 100.0
            ).min(),
            (
                plot_data[
                    "k3_accuracy"
                ]
                * 100.0
            ).min(),
        )
    )

    axis.set_ylim(
        bottom=max(
            0.0,
            minimum_value - 3.0,
        ),
        top=100.5,
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    axis.legend()

    add_bar_value_labels(
        axis,
        decimals=2,
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

    return output_path


# =============================================================================
# GRAFIK WAKTU INFERENSI
# =============================================================================

def plot_inference_comparison(
    metrics: pd.DataFrame,
) -> Path:
    """
    Membuat grafik rata-rata waktu inferensi
    per artikel.
    """

    output_path = (
        FIGURES_DIR
        / "inference_time_comparison.png"
    )

    plot_data = metrics.copy()

    plot_data["label"] = (
        plot_data["dataset"]
        + "\n"
        + plot_data["model_display"]
        + " "
        + plot_data["scenario_code"]
    )

    plot_data = (
        plot_data
        .sort_values(
            "average_inference_ms_per_sample",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    positions = np.arange(
        len(plot_data)
    )

    figure, axis = plt.subplots(
        figsize=(13, 6)
    )

    axis.bar(
        positions,
        plot_data[
            "average_inference_ms_per_sample"
        ],
    )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        plot_data["label"],
        rotation=40,
        ha="right",
    )

    axis.set_ylabel(
        "Rata-rata Waktu Inferensi (ms/artikel)"
    )

    axis.set_xlabel(
        "Eksperimen"
    )

    axis.set_title(
        "Perbandingan Efisiensi Inferensi"
    )

    axis.set_ylim(
        bottom=0.0
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    add_bar_value_labels(
        axis,
        decimals=3,
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

    return output_path


# =============================================================================
# MENAMPILKAN RINGKASAN
# =============================================================================

def print_summary(
    model_comparison: pd.DataFrame,
    description_analysis: pd.DataFrame,
    yake_analysis: pd.DataFrame,
    best_model_summary: pd.DataFrame,
) -> None:
    """
    Menampilkan ringkasan hasil pada terminal.
    """

    print(
        "\n" + "=" * 80
    )

    print(
        "RINGKASAN COMPARATIVE AND ABLATION ANALYSIS"
    )

    print("=" * 80)

    print(
        "\nPERBANDINGAN CNN VS ATTENTION-BILSTM"
    )

    if not model_comparison.empty:
        display_columns = [
            "dataset",
            "scenario_code",
            "cnn_accuracy",
            "attention_bilstm_accuracy",
            "accuracy_winner",
            "faster_model",
        ]

        print(
            "\n"
            + model_comparison[
                display_columns
            ].to_string(
                index=False
            )
        )

    print(
        "\nPENGARUH TITLE + DESCRIPTION"
    )

    if not description_analysis.empty:
        display_columns = [
            "dataset",
            "model",
            "controlled_sequence_length",
            "baseline_accuracy",
            "treatment_accuracy",
            "accuracy_change_percentage_point",
            "error_reduction",
        ]

        print(
            "\n"
            + description_analysis[
                display_columns
            ].to_string(
                index=False
            )
        )

    print(
        "\nPENGARUH KEYWORD YAKE"
    )

    if not yake_analysis.empty:
        display_columns = [
            "model",
            "k2_accuracy",
            "k3_accuracy",
            "accuracy_change_percentage_point",
            "additional_errors_after_yake",
            "conclusion",
        ]

        print(
            "\n"
            + yake_analysis[
                display_columns
            ].to_string(
                index=False
            )
        )

    print(
        "\nMODEL TERBAIK PER DATASET"
    )

    if not best_model_summary.empty:
        display_columns = [
            "dataset",
            "best_accuracy_experiment",
            "best_accuracy_model",
            "best_accuracy_scenario",
            "best_accuracy",
            "best_accuracy_f1_macro",
        ]

        print(
            "\n"
            + best_model_summary[
                display_columns
            ].to_string(
                index=False
            )
        )


# =============================================================================
# PROGRAM UTAMA
# =============================================================================

def main() -> None:
    """
    Menjalankan comparative dan ablation analysis final.
    """

    print("=" * 80)

    print(
        "STEP 6.4 - COMPARATIVE AND ABLATION ANALYSIS"
    )

    print("=" * 80)

    create_output_directories()

    print(
        "\nMemuat hasil evaluasi test set..."
    )

    metrics = load_metrics()

    print(
        f"Jumlah eksperimen final: "
        f"{len(metrics)}"
    )

    print(
        "\nMembentuk perbandingan model..."
    )

    model_comparison = (
        build_model_comparison(
            metrics
        )
    )

    print(
        "Membentuk perbandingan skenario..."
    )

    scenario_comparison = (
        build_scenario_comparison(
            metrics
        )
    )

    print(
        "Menganalisis perubahan Title ke "
        "Title + Description..."
    )

    description_analysis = (
        build_description_analysis(
            metrics
        )
    )

    print(
        "Menganalisis kontribusi YAKE..."
    )

    yake_analysis = (
        build_yake_analysis(
            metrics
        )
    )

    print(
        "Menentukan model terbaik per dataset..."
    )

    best_model_summary = (
        build_best_model_summary(
            metrics
        )
    )

    print(
        "Menganalisis efisiensi inferensi..."
    )

    efficiency_analysis = (
        build_efficiency_analysis(
            metrics
        )
    )

    print(
        "Membentuk ringkasan temuan penelitian..."
    )

    final_findings = (
        build_final_findings(
            description_analysis=(
                description_analysis
            ),
            yake_analysis=(
                yake_analysis
            ),
            best_model_summary=(
                best_model_summary
            ),
            model_comparison=(
                model_comparison
            ),
        )
    )

    # =========================================================================
    # MENYIMPAN TABEL
    # =========================================================================

    model_comparison.to_csv(
        MODEL_COMPARISON_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    scenario_comparison.to_csv(
        SCENARIO_COMPARISON_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    description_analysis.to_csv(
        DESCRIPTION_ANALYSIS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    yake_analysis.to_csv(
        YAKE_ANALYSIS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    best_model_summary.to_csv(
        BEST_MODEL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    efficiency_analysis.to_csv(
        EFFICIENCY_ANALYSIS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    final_findings.to_csv(
        FINAL_RESEARCH_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # MEMBUAT GRAFIK
    # =========================================================================

    print(
        "\nMembuat grafik perbandingan..."
    )

    accuracy_plot = (
        plot_accuracy_comparison(
            metrics
        )
    )

    f1_plot = (
        plot_f1_comparison(
            metrics
        )
    )

    description_plot = (
        plot_description_contribution(
            description_analysis
        )
    )

    yake_plot = (
        plot_yake_contribution(
            yake_analysis
        )
    )

    inference_plot = (
        plot_inference_comparison(
            metrics
        )
    )

    print_summary(
        model_comparison=(
            model_comparison
        ),
        description_analysis=(
            description_analysis
        ),
        yake_analysis=(
            yake_analysis
        ),
        best_model_summary=(
            best_model_summary
        ),
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "OUTPUT COMPARATIVE ANALYSIS"
    )

    print("=" * 80)

    print(
        "\nTabel perbandingan model:"
    )

    print(
        MODEL_COMPARISON_PATH
    )

    print(
        "\nTabel perbandingan skenario:"
    )

    print(
        SCENARIO_COMPARISON_PATH
    )

    print(
        "\nAnalisis Title + Description:"
    )

    print(
        DESCRIPTION_ANALYSIS_PATH
    )

    print(
        "\nAnalisis YAKE:"
    )

    print(
        YAKE_ANALYSIS_PATH
    )

    print(
        "\nRingkasan model terbaik:"
    )

    print(
        BEST_MODEL_PATH
    )

    print(
        "\nAnalisis efisiensi:"
    )

    print(
        EFFICIENCY_ANALYSIS_PATH
    )

    print(
        "\nTemuan akhir penelitian:"
    )

    print(
        FINAL_RESEARCH_SUMMARY_PATH
    )

    print(
        "\nGrafik accuracy:"
    )

    print(
        accuracy_plot
    )

    print(
        "\nGrafik macro F1:"
    )

    print(
        f1_plot
    )

    print(
        "\nGrafik Title + Description:"
    )

    print(
        description_plot
    )

    print(
        "\nGrafik kontribusi YAKE:"
    )

    print(
        yake_plot
    )

    print(
        "\nGrafik waktu inferensi:"
    )

    print(
        inference_plot
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "Tahap comparative dan ablation analysis selesai."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
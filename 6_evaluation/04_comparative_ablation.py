# =============================================================================
# STEP 6.4 - COMPARATIVE AND ABLATION ANALYSIS
# =============================================================================
# File:
# 6_evaluation/04_comparative_ablation.py
#
# Tujuan:
# 1. Membandingkan CNN dan Attention-BiLSTM.
# 2. Membandingkan skenario representasi teks.
# 3. Menganalisis pengaruh penambahan description.
# 4. Menganalisis pengaruh keyword YAKE.
# 5. Membandingkan efisiensi waktu inferensi.
#
# Eksperimen utama:
# Kompas:
# - K1 = Title
# - K2 = Title + Description
# - K3 = Title + Description + Keyword YAKE
#
# AG News:
# - A1 = Title
# - A2 = Title + Description
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
# - Tabel analisis skenario
# - Tabel pengaruh description
# - Tabel pengaruh YAKE
# - Tabel model terbaik
# - Grafik accuracy, macro F1, dan inference time
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


# =============================================================================
# MEMBUAT FOLDER OUTPUT
# =============================================================================

def create_output_directories() -> None:
    """
    Membuat folder output jika belum tersedia.
    """

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# MEMUAT METRIK
# =============================================================================

def load_metrics() -> pd.DataFrame:
    """
    Membaca hasil evaluasi test set.
    """

    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            "File metrik evaluasi tidak ditemukan:\n"
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
            f"Kolom hilang: {missing_columns}"
        )

    dataframe = dataframe[
        dataframe["experiment_name"].isin(
            MAIN_EXPERIMENTS
        )
    ].copy()

    if len(dataframe) != 10:
        print(
            "Peringatan: jumlah eksperimen utama "
            f"yang ditemukan adalah {len(dataframe)}, "
            "seharusnya 10."
        )

    numeric_columns = [
        "jumlah_test",
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

    dataframe = dataframe.dropna(
        subset=[
            "accuracy",
            "f1_macro",
            "inference_time_seconds",
        ]
    )

    dataframe["model_display"] = (
        dataframe["model"]
        .map(MODEL_DISPLAY_NAMES)
        .fillna(dataframe["model"])
    )

    dataframe["scenario_display"] = (
        dataframe["scenario_code"]
        .map(SCENARIO_DISPLAY_NAMES)
        .fillna(dataframe["scenario_name"])
    )

    dataframe["scenario_order"] = (
        dataframe["scenario_code"]
        .map(SCENARIO_ORDER)
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

    rows = []

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

        cnn_row = group[
            group["model"] == "cnn"
        ]

        bilstm_row = group[
            group["model"] == "attention_bilstm"
        ]

        if cnn_row.empty or bilstm_row.empty:
            continue

        cnn_row = cnn_row.iloc[0]
        bilstm_row = bilstm_row.iloc[0]

        accuracy_difference = (
            float(cnn_row["accuracy"])
            - float(bilstm_row["accuracy"])
        )

        f1_difference = (
            float(cnn_row["f1_macro"])
            - float(bilstm_row["f1_macro"])
        )

        inference_difference = (
            float(cnn_row["inference_time_seconds"])
            - float(bilstm_row["inference_time_seconds"])
        )

        if accuracy_difference > 0:
            accuracy_winner = "CNN"
        elif accuracy_difference < 0:
            accuracy_winner = "Attention-BiLSTM"
        else:
            accuracy_winner = "Sama"

        if f1_difference > 0:
            f1_winner = "CNN"
        elif f1_difference < 0:
            f1_winner = "Attention-BiLSTM"
        else:
            f1_winner = "Sama"

        if inference_difference < 0:
            faster_model = "CNN"
        elif inference_difference > 0:
            faster_model = "Attention-BiLSTM"
        else:
            faster_model = "Sama"

        rows.append(
            {
                "dataset":
                    dataset_name,

                "scenario_code":
                    scenario_code,

                "scenario_name":
                    scenario_name,

                "cnn_accuracy":
                    float(cnn_row["accuracy"]),

                "attention_bilstm_accuracy":
                    float(bilstm_row["accuracy"]),

                "accuracy_difference_cnn_minus_bilstm":
                    accuracy_difference,

                "accuracy_winner":
                    accuracy_winner,

                "cnn_f1_macro":
                    float(cnn_row["f1_macro"]),

                "attention_bilstm_f1_macro":
                    float(bilstm_row["f1_macro"]),

                "f1_difference_cnn_minus_bilstm":
                    f1_difference,

                "f1_winner":
                    f1_winner,

                "cnn_inference_seconds":
                    float(
                        cnn_row[
                            "inference_time_seconds"
                        ]
                    ),

                "attention_bilstm_inference_seconds":
                    float(
                        bilstm_row[
                            "inference_time_seconds"
                        ]
                    ),

                "faster_model":
                    faster_model,
            }
        )

    result = pd.DataFrame(rows)

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
                columns=["scenario_order"]
            )
            .reset_index(drop=True)
        )

    return result


# =============================================================================
# PERBANDINGAN SKENARIO
# =============================================================================

def build_scenario_comparison(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Membandingkan performa skenario dalam setiap model.
    """

    columns = [
        "experiment_name",
        "model_display",
        "dataset",
        "scenario_code",
        "scenario_display",
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
            "model_display": "model",
            "scenario_display": "scenario_name",
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
            columns=["scenario_order"]
        )
        .reset_index(drop=True)
    )

    return result


# =============================================================================
# ANALISIS DESCRIPTION
# =============================================================================

def build_description_analysis(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Menganalisis perubahan dari:
    Kompas K1 -> K2
    AG News A1 -> A2
    """

    comparison_pairs = [
        {
            "dataset": "Kompas",
            "baseline": "K1",
            "treatment": "K2",
            "comparison": (
                "Title vs Title + Description"
            ),
        },
        {
            "dataset": "AG News",
            "baseline": "A1",
            "treatment": "A2",
            "comparison": (
                "Title vs Title + Description"
            ),
        },
    ]

    rows = []

    for pair in comparison_pairs:
        dataset_name = pair["dataset"]

        for model_name in [
            "cnn",
            "attention_bilstm",
        ]:
            baseline_row = metrics[
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

            treatment_row = metrics[
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
                baseline_row.empty
                or treatment_row.empty
            ):
                continue

            baseline_row = baseline_row.iloc[0]
            treatment_row = treatment_row.iloc[0]

            accuracy_change = (
                float(treatment_row["accuracy"])
                - float(baseline_row["accuracy"])
            )

            f1_change = (
                float(treatment_row["f1_macro"])
                - float(baseline_row["f1_macro"])
            )

            error_before = (
                int(baseline_row["jumlah_test"])
                - int(
                    round(
                        baseline_row["accuracy"]
                        * baseline_row["jumlah_test"]
                    )
                )
            )

            error_after = (
                int(treatment_row["jumlah_test"])
                - int(
                    round(
                        treatment_row["accuracy"]
                        * treatment_row["jumlah_test"]
                    )
                )
            )

            error_reduction = (
                error_before
                - error_after
            )

            if accuracy_change > 0:
                interpretation = (
                    "Description meningkatkan performa."
                )
            elif accuracy_change < 0:
                interpretation = (
                    "Description menurunkan performa."
                )
            else:
                interpretation = (
                    "Description tidak mengubah accuracy."
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
                        pair["comparison"],

                    "baseline_scenario":
                        pair["baseline"],

                    "treatment_scenario":
                        pair["treatment"],

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
                        accuracy_change * 100,

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

                    "incorrect_before":
                        error_before,

                    "incorrect_after":
                        error_after,

                    "error_reduction":
                        error_reduction,

                    "interpretation":
                        interpretation,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# ANALISIS YAKE
# =============================================================================

def build_yake_analysis(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Membandingkan K2 dan K3.

    K2 dan K3 menggunakan sequence length yang sama,
    yaitu 60, sehingga analisis lebih terkontrol.
    """

    rows = []

    for model_name in [
        "cnn",
        "attention_bilstm",
    ]:
        k2_row = metrics[
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

        k3_row = metrics[
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

        if k2_row.empty or k3_row.empty:
            continue

        k2_row = k2_row.iloc[0]
        k3_row = k3_row.iloc[0]

        accuracy_change = (
            float(k3_row["accuracy"])
            - float(k2_row["accuracy"])
        )

        precision_change = (
            float(k3_row["precision_macro"])
            - float(k2_row["precision_macro"])
        )

        recall_change = (
            float(k3_row["recall_macro"])
            - float(k2_row["recall_macro"])
        )

        f1_change = (
            float(k3_row["f1_macro"])
            - float(k2_row["f1_macro"])
        )

        log_loss_change = (
            float(k3_row["log_loss"])
            - float(k2_row["log_loss"])
        )

        incorrect_k2 = (
            int(k2_row["jumlah_test"])
            - int(
                round(
                    k2_row["accuracy"]
                    * k2_row["jumlah_test"]
                )
            )
        )

        incorrect_k3 = (
            int(k3_row["jumlah_test"])
            - int(
                round(
                    k3_row["accuracy"]
                    * k3_row["jumlah_test"]
                )
            )
        )

        additional_errors = (
            incorrect_k3
            - incorrect_k2
        )

        if accuracy_change > 0:
            conclusion = (
                "YAKE meningkatkan accuracy."
            )
        elif accuracy_change < 0:
            conclusion = (
                "YAKE belum meningkatkan accuracy."
            )
        else:
            conclusion = (
                "YAKE tidak mengubah accuracy."
            )

        rows.append(
            {
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
                    60,

                "k2_accuracy":
                    float(k2_row["accuracy"]),

                "k3_accuracy":
                    float(k3_row["accuracy"]),

                "accuracy_change":
                    accuracy_change,

                "accuracy_change_percentage_point":
                    accuracy_change * 100,

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

    return pd.DataFrame(rows)


# =============================================================================
# MODEL TERBAIK
# =============================================================================

def build_best_model_summary(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mencari model terbaik pada setiap dataset.
    """

    rows = []

    for dataset_name in metrics[
        "dataset"
    ].unique():

        dataset_metrics = metrics[
            metrics["dataset"]
            == dataset_name
        ].copy()

        best_accuracy_row = (
            dataset_metrics
            .sort_values(
                [
                    "accuracy",
                    "f1_macro",
                ],
                ascending=False,
            )
            .iloc[0]
        )

        best_f1_row = (
            dataset_metrics
            .sort_values(
                [
                    "f1_macro",
                    "accuracy",
                ],
                ascending=False,
            )
            .iloc[0]
        )

        fastest_row = (
            dataset_metrics
            .sort_values(
                "average_inference_ms_per_sample",
                ascending=True,
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

                "best_accuracy":
                    float(
                        best_accuracy_row[
                            "accuracy"
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

    return pd.DataFrame(rows)


# =============================================================================
# ANALISIS EFISIENSI
# =============================================================================

def build_efficiency_analysis(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Membandingkan kecepatan inferensi seluruh eksperimen.
    """

    result = metrics[
        [
            "experiment_name",
            "model_display",
            "dataset",
            "scenario_code",
            "scenario_display",
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

    result["samples_per_second"] = np.divide(
        result["jumlah_test"],
        result["inference_time_seconds"],
        out=np.zeros(
            len(result),
            dtype=float,
        ),
        where=(
            result["inference_time_seconds"]
            != 0
        ),
    )

    result = result.sort_values(
        "average_inference_ms_per_sample",
        ascending=True,
    ).reset_index(drop=True)

    result["efficiency_rank"] = (
        np.arange(
            1,
            len(result) + 1,
        )
    )

    return result


# =============================================================================
# TEMUAN PENELITIAN
# =============================================================================

def build_final_findings(
    metrics: pd.DataFrame,
    description_analysis: pd.DataFrame,
    yake_analysis: pd.DataFrame,
    best_model_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Membentuk temuan utama penelitian secara otomatis.
    """

    findings = []

    # Temuan penambahan description.
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
                        f"Penambahan description "
                        f"mengubah accuracy dari "
                        f"{row.baseline_accuracy:.4f} "
                        f"menjadi "
                        f"{row.treatment_accuracy:.4f}."
                    ),

                "change_percentage_point":
                    row.accuracy_change_percentage_point,

                "interpretation":
                    row.interpretation,
            }
        )

    # Temuan YAKE.
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
                        f"Penambahan keyword YAKE "
                        f"mengubah accuracy K2 "
                        f"{row.k2_accuracy:.4f} "
                        f"menjadi K3 "
                        f"{row.k3_accuracy:.4f}."
                    ),

                "change_percentage_point":
                    row.accuracy_change_percentage_point,

                "interpretation":
                    row.conclusion,
            }
        )

    # Temuan model terbaik.
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
                        f"Model terbaik berdasarkan "
                        f"accuracy adalah "
                        f"{row.best_accuracy_experiment} "
                        f"dengan accuracy "
                        f"{row.best_accuracy:.4f}."
                    ),

                "change_percentage_point":
                    np.nan,

                "interpretation":
                    (
                        "Model dipilih berdasarkan "
                        "hasil test set."
                    ),
            }
        )

    # Temuan model yang lebih sering unggul.
    winner_counts = {
        "CNN": 0,
        "Attention-BiLSTM": 0,
        "Sama": 0,
    }

    model_comparison = build_model_comparison(
        metrics
    )

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
                    f"CNN unggul pada "
                    f"{winner_counts['CNN']} skenario, "
                    f"Attention-BiLSTM unggul pada "
                    f"{winner_counts['Attention-BiLSTM']} "
                    f"skenario, dan seri pada "
                    f"{winner_counts['Sama']} skenario."
                ),

            "change_percentage_point":
                np.nan,

            "interpretation":
                (
                    "Tidak ada satu arsitektur yang "
                    "mutlak unggul pada seluruh skenario."
                ),
        }
    )

    return pd.DataFrame(findings)


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
        plot_data["model_display"]
        + " "
        + plot_data["scenario_code"]
    )

    plot_data = plot_data.sort_values(
        [
            "dataset",
            "scenario_order",
            "model_display",
        ]
    )

    plt.figure(
        figsize=(12, 6)
    )

    positions = np.arange(
        len(plot_data)
    )

    plt.bar(
        positions,
        plot_data["accuracy"] * 100,
    )

    plt.xticks(
        positions,
        plot_data["label"],
        rotation=45,
        ha="right",
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.xlabel(
        "Eksperimen"
    )

    plt.title(
        "Perbandingan Accuracy pada Test Set"
    )

    plt.ylim(
        0,
        100,
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    for position, value in zip(
        positions,
        plot_data["accuracy"] * 100,
    ):
        plt.text(
            position,
            value + 0.5,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

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
        plot_data["model_display"]
        + " "
        + plot_data["scenario_code"]
    )

    plot_data = plot_data.sort_values(
        [
            "dataset",
            "scenario_order",
            "model_display",
        ]
    )

    plt.figure(
        figsize=(12, 6)
    )

    positions = np.arange(
        len(plot_data)
    )

    plt.bar(
        positions,
        plot_data["f1_macro"] * 100,
    )

    plt.xticks(
        positions,
        plot_data["label"],
        rotation=45,
        ha="right",
    )

    plt.ylabel(
        "Macro F1-score (%)"
    )

    plt.xlabel(
        "Eksperimen"
    )

    plt.title(
        "Perbandingan Macro F1-score pada Test Set"
    )

    plt.ylim(
        0,
        100,
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    for position, value in zip(
        positions,
        plot_data["f1_macro"] * 100,
    ):
        plt.text(
            position,
            value + 0.5,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


# =============================================================================
# GRAFIK PENGARUH DESCRIPTION
# =============================================================================

def plot_description_contribution(
    description_analysis: pd.DataFrame,
) -> Path:
    """
    Membuat grafik perubahan accuracy akibat description.
    """

    output_path = (
        FIGURES_DIR
        / "description_contribution.png"
    )

    plot_data = (
        description_analysis
        .copy()
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

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        positions - bar_width / 2,
        plot_data["baseline_accuracy"] * 100,
        width=bar_width,
        label="Title",
    )

    plt.bar(
        positions + bar_width / 2,
        plot_data["treatment_accuracy"] * 100,
        width=bar_width,
        label="Title + Description",
    )

    plt.xticks(
        positions,
        plot_data["label"],
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.xlabel(
        "Dataset dan Model"
    )

    plt.title(
        "Pengaruh Penambahan Description"
    )

    plt.ylim(
        0,
        100,
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


# =============================================================================
# GRAFIK PENGARUH YAKE
# =============================================================================

def plot_yake_contribution(
    yake_analysis: pd.DataFrame,
) -> Path:
    """
    Membuat grafik K2 vs K3.
    """

    output_path = (
        FIGURES_DIR
        / "yake_contribution.png"
    )

    plot_data = yake_analysis.copy()

    positions = np.arange(
        len(plot_data)
    )

    bar_width = 0.35

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        positions - bar_width / 2,
        plot_data["k2_accuracy"] * 100,
        width=bar_width,
        label="K2: Title + Description",
    )

    plt.bar(
        positions + bar_width / 2,
        plot_data["k3_accuracy"] * 100,
        width=bar_width,
        label="K3: + Keyword YAKE",
    )

    plt.xticks(
        positions,
        plot_data["model"],
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.xlabel(
        "Model"
    )

    plt.title(
        "Analisis Pengaruh Keyword YAKE\n"
        "Sequence Length K2 dan K3 = 60"
    )

    plt.ylim(
        0,
        100,
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


# =============================================================================
# GRAFIK WAKTU INFERENSI
# =============================================================================

def plot_inference_comparison(
    metrics: pd.DataFrame,
) -> Path:
    """
    Membuat grafik waktu inferensi rata-rata per sampel.
    """

    output_path = (
        FIGURES_DIR
        / "inference_time_comparison.png"
    )

    plot_data = metrics.copy()

    plot_data["label"] = (
        plot_data["model_display"]
        + " "
        + plot_data["scenario_code"]
    )

    plot_data = plot_data.sort_values(
        "average_inference_ms_per_sample",
        ascending=True,
    )

    positions = np.arange(
        len(plot_data)
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        positions,
        plot_data[
            "average_inference_ms_per_sample"
        ],
    )

    plt.xticks(
        positions,
        plot_data["label"],
        rotation=45,
        ha="right",
    )

    plt.ylabel(
        "Rata-rata Waktu Inferensi (ms/artikel)"
    )

    plt.xlabel(
        "Eksperimen"
    )

    plt.title(
        "Perbandingan Efisiensi Inferensi"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

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

    print("\n" + "=" * 80)
    print("RINGKASAN COMPARATIVE AND ABLATION ANALYSIS")
    print("=" * 80)

    print("\nPERBANDINGAN CNN VS ATTENTION-BILSTM")

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

    print("\nPENGARUH DESCRIPTION")

    if not description_analysis.empty:
        display_columns = [
            "dataset",
            "model",
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

    print("\nPENGARUH KEYWORD YAKE")

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

    print("\nMODEL TERBAIK")

    if not best_model_summary.empty:
        display_columns = [
            "dataset",
            "best_accuracy_experiment",
            "best_accuracy_model",
            "best_accuracy_scenario",
            "best_accuracy",
            "best_f1_macro",
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
    Menjalankan comparative dan ablation analysis.
    """

    print("=" * 80)
    print(
        "STEP 6.4 - COMPARATIVE AND ABLATION ANALYSIS"
    )
    print("=" * 80)

    create_output_directories()

    print("\nMemuat hasil evaluasi test set...")

    metrics = load_metrics()

    print(
        f"Jumlah eksperimen utama: "
        f"{len(metrics)}"
    )

    print("\nMembentuk perbandingan model...")

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
        "Menganalisis kontribusi description..."
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
        "Menentukan model terbaik..."
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
            metrics=metrics,
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
    )

    # Simpan tabel.
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

    # Buat grafik.
    print("\nMembuat grafik perbandingan...")

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

    print("\n" + "=" * 80)
    print("OUTPUT COMPARATIVE ANALYSIS")
    print("=" * 80)

    print("\nTabel perbandingan model:")
    print(MODEL_COMPARISON_PATH)

    print("\nTabel perbandingan skenario:")
    print(SCENARIO_COMPARISON_PATH)

    print("\nAnalisis description:")
    print(DESCRIPTION_ANALYSIS_PATH)

    print("\nAnalisis YAKE:")
    print(YAKE_ANALYSIS_PATH)

    print("\nRingkasan model terbaik:")
    print(BEST_MODEL_PATH)

    print("\nAnalisis efisiensi:")
    print(EFFICIENCY_ANALYSIS_PATH)

    print("\nTemuan akhir penelitian:")
    print(FINAL_RESEARCH_SUMMARY_PATH)

    print("\nGrafik accuracy:")
    print(accuracy_plot)

    print("\nGrafik macro F1:")
    print(f1_plot)

    print("\nGrafik kontribusi description:")
    print(description_plot)

    print("\nGrafik kontribusi YAKE:")
    print(yake_plot)

    print("\nGrafik waktu inferensi:")
    print(inference_plot)

    print("\n" + "=" * 80)
    print(
        "Tahap comparative dan ablation analysis selesai."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
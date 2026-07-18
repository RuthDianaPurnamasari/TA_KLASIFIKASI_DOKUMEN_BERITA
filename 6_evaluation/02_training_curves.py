# =============================================================================
# STEP 6.2 - TRAINING CURVES
# =============================================================================
# File:
# 6_evaluation/02_training_curves.py
#
# Tujuan:
# Membuat grafik training accuracy, validation accuracy,
# training loss, dan validation loss untuk 10 eksperimen final.
#
# Eksperimen:
# - Kompas: K1, K2, K3
# - AG News: A1, A2
# - Model: CNN dan Attention-BiLSTM
#
# Output:
# - Grafik accuracy setiap eksperimen
# - Grafik loss setiap eksperimen
# - Grafik gabungan accuracy dan loss
# - Ringkasan epoch terbaik
# - Ringkasan indikasi overfitting
#
# Catatan:
# Epoch terbaik ditentukan berdasarkan validation loss terendah.
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# PATH INPUT
# =============================================================================

TRAINING_HISTORY_DIR = (
    PROJECT_ROOT
    / "9_results"
    / "training_history"
)

TRAINING_SUMMARY_DIR = (
    PROJECT_ROOT
    / "9_results"
    / "training_summary"
)


# =============================================================================
# PATH OUTPUT
# =============================================================================

FIGURES_DIR = (
    PROJECT_ROOT
    / "9_results"
    / "figures"
    / "training_curves"
)

TABLES_DIR = (
    PROJECT_ROOT
    / "9_results"
    / "tables"
)

TRAINING_CURVE_REPORT_PATH = (
    TABLES_DIR
    / "training_curve_report.csv"
)

OVERFITTING_REPORT_PATH = (
    TABLES_DIR
    / "overfitting_analysis.csv"
)


# =============================================================================
# KONFIGURASI EKSPERIMEN
# =============================================================================

EXPERIMENTS = [
    {
        "experiment_name": "cnn_k1",
        "model": "CNN",
        "dataset": "Kompas",
        "scenario_code": "K1",
        "scenario_name": "Title",
    },
    {
        "experiment_name": "cnn_k2",
        "model": "CNN",
        "dataset": "Kompas",
        "scenario_code": "K2",
        "scenario_name": "Title + Description",
    },
    {
        "experiment_name": "cnn_k3",
        "model": "CNN",
        "dataset": "Kompas",
        "scenario_code": "K3",
        "scenario_name": (
            "Title + Description + Keyword YAKE"
        ),
    },
    {
        "experiment_name": "cnn_a1",
        "model": "CNN",
        "dataset": "AG News",
        "scenario_code": "A1",
        "scenario_name": "Title",
    },
    {
        "experiment_name": "cnn_a2",
        "model": "CNN",
        "dataset": "AG News",
        "scenario_code": "A2",
        "scenario_name": "Title + Description",
    },
    {
        "experiment_name": "attention_bilstm_k1",
        "model": "Attention-BiLSTM",
        "dataset": "Kompas",
        "scenario_code": "K1",
        "scenario_name": "Title",
    },
    {
        "experiment_name": "attention_bilstm_k2",
        "model": "Attention-BiLSTM",
        "dataset": "Kompas",
        "scenario_code": "K2",
        "scenario_name": "Title + Description",
    },
    {
        "experiment_name": "attention_bilstm_k3",
        "model": "Attention-BiLSTM",
        "dataset": "Kompas",
        "scenario_code": "K3",
        "scenario_name": (
            "Title + Description + Keyword YAKE"
        ),
    },
    {
        "experiment_name": "attention_bilstm_a1",
        "model": "Attention-BiLSTM",
        "dataset": "AG News",
        "scenario_code": "A1",
        "scenario_name": "Title",
    },
    {
        "experiment_name": "attention_bilstm_a2",
        "model": "Attention-BiLSTM",
        "dataset": "AG News",
        "scenario_code": "A2",
        "scenario_name": "Title + Description",
    },
]


# =============================================================================
# MEMBUAT FOLDER OUTPUT
# =============================================================================

def create_output_directories() -> None:
    """
    Membuat seluruh folder output yang dibutuhkan.
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
# PATH FILE HISTORY DAN SUMMARY
# =============================================================================

def get_history_path(
    experiment_name: str,
) -> Path:
    """
    Menghasilkan lokasi file training history.
    """

    return (
        TRAINING_HISTORY_DIR
        / f"{experiment_name}_history.csv"
    )


def get_summary_path(
    experiment_name: str,
) -> Path:
    """
    Menghasilkan lokasi file training summary.
    """

    return (
        TRAINING_SUMMARY_DIR
        / f"{experiment_name}_summary.json"
    )


# =============================================================================
# MEMBACA TRAINING HISTORY
# =============================================================================

def load_training_history(
    experiment_name: str,
) -> pd.DataFrame:
    """
    Membaca dan memvalidasi file training history.

    Kolom wajib:
    - epoch
    - loss
    - accuracy
    - val_loss
    - val_accuracy
    """

    history_path = get_history_path(
        experiment_name
    )

    if not history_path.exists():
        raise FileNotFoundError(
            "File training history tidak ditemukan:\n"
            f"{history_path}"
        )

    if history_path.stat().st_size == 0:
        raise ValueError(
            "File training history ditemukan, "
            "tetapi file kosong:\n"
            f"{history_path}"
        )

    history = pd.read_csv(
        history_path,
        encoding="utf-8-sig",
    )

    required_columns = {
        "epoch",
        "loss",
        "accuracy",
        "val_loss",
        "val_accuracy",
    }

    missing_columns = (
        required_columns
        - set(history.columns)
    )

    if missing_columns:
        raise KeyError(
            f"{experiment_name}: kolom history "
            "tidak lengkap.\n"
            f"Kolom hilang: "
            f"{sorted(missing_columns)}"
        )

    history = history.copy()

    numeric_columns = [
        "epoch",
        "loss",
        "accuracy",
        "val_loss",
        "val_accuracy",
    ]

    for column in numeric_columns:
        history[column] = pd.to_numeric(
            history[column],
            errors="coerce",
        )

    history = history.dropna(
        subset=numeric_columns
    )

    if history.empty:
        raise ValueError(
            f"{experiment_name}: training history "
            "kosong setelah validasi."
        )

    # Memastikan nilai epoch berupa bilangan bulat.
    non_integer_epoch = (
        history["epoch"]
        % 1
        != 0
    )

    if non_integer_epoch.any():
        invalid_epochs = (
            history.loc[
                non_integer_epoch,
                "epoch",
            ]
            .tolist()
        )

        raise ValueError(
            f"{experiment_name}: ditemukan nilai "
            "epoch bukan integer.\n"
            f"Nilai: {invalid_epochs}"
        )

    # Mengurutkan epoch dan menghapus duplikasi.
    history = (
        history
        .sort_values(
            "epoch"
        )
        .drop_duplicates(
            subset=["epoch"],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    # CSVLogger biasanya menyimpan epoch mulai dari 0.
    # Untuk grafik dan laporan, epoch ditampilkan mulai dari 1.
    if int(history["epoch"].min()) == 0:
        history["epoch_display"] = (
            history["epoch"]
            + 1
        )
    else:
        history["epoch_display"] = (
            history["epoch"]
        )

    history["epoch"] = (
        history["epoch"]
        .astype(int)
    )

    history["epoch_display"] = (
        history["epoch_display"]
        .astype(int)
    )

    # Validasi rentang metrik accuracy.
    for accuracy_column in [
        "accuracy",
        "val_accuracy",
    ]:
        invalid_accuracy = (
            (
                history[accuracy_column]
                < 0.0
            )
            |
            (
                history[accuracy_column]
                > 1.0
            )
        )

        if invalid_accuracy.any():
            raise ValueError(
                f"{experiment_name}: ditemukan nilai "
                f"{accuracy_column} di luar rentang "
                "0 sampai 1."
            )

    # Loss tidak boleh negatif.
    for loss_column in [
        "loss",
        "val_loss",
    ]:
        if (
            history[loss_column]
            < 0.0
        ).any():
            raise ValueError(
                f"{experiment_name}: ditemukan nilai "
                f"{loss_column} negatif."
            )

    return history


# =============================================================================
# MEMBACA TRAINING SUMMARY
# =============================================================================

def load_training_summary(
    experiment_name: str,
) -> dict[str, Any]:
    """
    Membaca JSON ringkasan training.

    Mengembalikan dictionary kosong apabila file summary
    tidak ditemukan atau tidak dapat dibaca.
    """

    summary_path = get_summary_path(
        experiment_name
    )

    if not summary_path.exists():
        return {}

    if summary_path.stat().st_size == 0:
        return {}

    try:
        with open(
            summary_path,
            "r",
            encoding="utf-8",
        ) as file:
            summary = json.load(
                file
            )

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return {}

    if not isinstance(
        summary,
        dict,
    ):
        return {}

    return summary


# =============================================================================
# MENCARI EPOCH TERBAIK
# =============================================================================

def get_best_epoch_information(
    history: pd.DataFrame,
) -> dict[str, float | int]:
    """
    Menentukan epoch terbaik berdasarkan validation loss minimum.
    """

    best_index = history[
        "val_loss"
    ].idxmin()

    best_row = history.loc[
        best_index
    ]

    return {
        "best_epoch": int(
            best_row[
                "epoch_display"
            ]
        ),

        "best_train_accuracy": float(
            best_row[
                "accuracy"
            ]
        ),

        "best_validation_accuracy": float(
            best_row[
                "val_accuracy"
            ]
        ),

        "best_train_loss": float(
            best_row[
                "loss"
            ]
        ),

        "best_validation_loss": float(
            best_row[
                "val_loss"
            ]
        ),
    }


# =============================================================================
# VALIDASI SUMMARY DAN HISTORY
# =============================================================================

def validate_summary_consistency(
    experiment_name: str,
    summary: dict[str, Any],
    best_information: dict[str, float | int],
) -> int | None:
    """
    Memastikan best epoch pada summary konsisten dengan history.

    Apabila summary tidak memiliki best_epoch, fungsi
    mengembalikan None.
    """

    summary_best_epoch = summary.get(
        "best_epoch"
    )

    if summary_best_epoch is None:
        return None

    try:
        summary_best_epoch = int(
            summary_best_epoch
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"{experiment_name}: best_epoch pada "
            "summary tidak valid.\n"
            f"Nilai: {summary_best_epoch}"
        ) from error

    history_best_epoch = int(
        best_information[
            "best_epoch"
        ]
    )

    if (
        summary_best_epoch
        != history_best_epoch
    ):
        raise ValueError(
            f"{experiment_name}: best epoch pada "
            "history dan summary berbeda.\n"
            f"History : {history_best_epoch}\n"
            f"Summary : {summary_best_epoch}"
        )

    return summary_best_epoch


# =============================================================================
# ANALISIS OVERFITTING
# =============================================================================

def analyze_overfitting(
    history: pd.DataFrame,
    best_epoch: int,
) -> dict[str, Any]:
    """
    Memberikan indikasi sederhana mengenai overfitting.

    Indikasi overfitting ditentukan ketika:
    - gap final train-validation accuracy minimal 0,03;
    - validation loss akhir lebih tinggi daripada
      validation loss pada epoch terbaik.

    Analisis ini merupakan indikator berbasis kurva training,
    bukan pengujian statistik.
    """

    last_row = history.iloc[-1]

    best_rows = history[
        history["epoch_display"]
        == best_epoch
    ]

    if best_rows.empty:
        best_index = history[
            "val_loss"
        ].idxmin()

        best_row = history.loc[
            best_index
        ]

    else:
        best_row = best_rows.iloc[0]

    final_epoch = int(
        last_row[
            "epoch_display"
        ]
    )

    final_train_accuracy = float(
        last_row[
            "accuracy"
        ]
    )

    final_validation_accuracy = float(
        last_row[
            "val_accuracy"
        ]
    )

    final_train_loss = float(
        last_row[
            "loss"
        ]
    )

    final_validation_loss = float(
        last_row[
            "val_loss"
        ]
    )

    best_validation_accuracy = float(
        best_row[
            "val_accuracy"
        ]
    )

    best_validation_loss = float(
        best_row[
            "val_loss"
        ]
    )

    accuracy_gap = (
        final_train_accuracy
        - final_validation_accuracy
    )

    validation_loss_change = (
        final_validation_loss
        - best_validation_loss
    )

    validation_accuracy_change = (
        final_validation_accuracy
        - best_validation_accuracy
    )

    epochs_after_best = (
        final_epoch
        - best_epoch
    )

    overfitting_detected = bool(
        accuracy_gap >= 0.03
        and validation_loss_change > 0.0
        and epochs_after_best > 0
    )

    if overfitting_detected:
        interpretation = (
            "Terdapat indikasi overfitting setelah "
            "epoch terbaik. Train accuracy meningkat "
            "dan validation loss memburuk."
        )

    elif (
        validation_loss_change > 0.0
        and epochs_after_best > 0
    ):
        interpretation = (
            "Validation loss meningkat setelah epoch "
            "terbaik, tetapi gap accuracy belum mencapai "
            "ambang indikasi overfitting."
        )

    else:
        interpretation = (
            "Tidak ditemukan indikasi overfitting "
            "yang kuat berdasarkan training history."
        )

    return {
        "final_epoch":
            final_epoch,

        "epochs_after_best":
            epochs_after_best,

        "final_train_accuracy":
            final_train_accuracy,

        "final_validation_accuracy":
            final_validation_accuracy,

        "final_train_loss":
            final_train_loss,

        "final_validation_loss":
            final_validation_loss,

        "final_accuracy_gap":
            accuracy_gap,

        "validation_loss_change_after_best":
            validation_loss_change,

        "validation_accuracy_change_after_best":
            validation_accuracy_change,

        "overfitting_detected":
            overfitting_detected,

        "interpretation":
            interpretation,
    }


# =============================================================================
# MEMBUAT GRAFIK ACCURACY
# =============================================================================

def plot_accuracy_curve(
    history: pd.DataFrame,
    experiment: dict[str, str],
    best_epoch: int,
) -> Path:
    """
    Membuat grafik train dan validation accuracy.
    """

    experiment_name = experiment[
        "experiment_name"
    ]

    output_path = (
        FIGURES_DIR
        / f"{experiment_name}_accuracy_curve.png"
    )

    figure = plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        history[
            "epoch_display"
        ],
        history[
            "accuracy"
        ],
        marker="o",
        label="Train Accuracy",
    )

    plt.plot(
        history[
            "epoch_display"
        ],
        history[
            "val_accuracy"
        ],
        marker="o",
        label="Validation Accuracy",
    )

    plt.axvline(
        best_epoch,
        linestyle="--",
        label=(
            f"Best Epoch = "
            f"{best_epoch}"
        ),
    )

    plt.title(
        f"Accuracy Curve - "
        f"{experiment['model']} "
        f"{experiment['scenario_code']}\n"
        f"{experiment['scenario_name']}"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.xticks(
        history[
            "epoch_display"
        ]
    )

    minimum_accuracy = min(
        history[
            "accuracy"
        ].min(),
        history[
            "val_accuracy"
        ].min(),
    )

    plt.ylim(
        bottom=max(
            0.0,
            minimum_accuracy
            - 0.05,
        ),
        top=1.02,
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

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
# MEMBUAT GRAFIK LOSS
# =============================================================================

def plot_loss_curve(
    history: pd.DataFrame,
    experiment: dict[str, str],
    best_epoch: int,
) -> Path:
    """
    Membuat grafik train dan validation loss.
    """

    experiment_name = experiment[
        "experiment_name"
    ]

    output_path = (
        FIGURES_DIR
        / f"{experiment_name}_loss_curve.png"
    )

    figure = plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        history[
            "epoch_display"
        ],
        history[
            "loss"
        ],
        marker="o",
        label="Train Loss",
    )

    plt.plot(
        history[
            "epoch_display"
        ],
        history[
            "val_loss"
        ],
        marker="o",
        label="Validation Loss",
    )

    plt.axvline(
        best_epoch,
        linestyle="--",
        label=(
            f"Best Epoch = "
            f"{best_epoch}"
        ),
    )

    plt.title(
        f"Loss Curve - "
        f"{experiment['model']} "
        f"{experiment['scenario_code']}\n"
        f"{experiment['scenario_name']}"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.xticks(
        history[
            "epoch_display"
        ]
    )

    plt.ylim(
        bottom=0.0
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

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
# MEMBUAT GRAFIK GABUNGAN
# =============================================================================

def plot_combined_curve(
    history: pd.DataFrame,
    experiment: dict[str, str],
    best_epoch: int,
) -> Path:
    """
    Membuat satu gambar yang berisi dua panel:
    - accuracy;
    - loss.
    """

    experiment_name = experiment[
        "experiment_name"
    ]

    output_path = (
        FIGURES_DIR
        / f"{experiment_name}_training_curve.png"
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14, 5),
    )

    # -------------------------------------------------------------------------
    # PANEL ACCURACY
    # -------------------------------------------------------------------------

    axes[0].plot(
        history[
            "epoch_display"
        ],
        history[
            "accuracy"
        ],
        marker="o",
        label="Train Accuracy",
    )

    axes[0].plot(
        history[
            "epoch_display"
        ],
        history[
            "val_accuracy"
        ],
        marker="o",
        label="Validation Accuracy",
    )

    axes[0].axvline(
        best_epoch,
        linestyle="--",
        label=(
            f"Best Epoch = "
            f"{best_epoch}"
        ),
    )

    axes[0].set_title(
        "Accuracy"
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Accuracy"
    )

    axes[0].set_xticks(
        history[
            "epoch_display"
        ]
    )

    minimum_accuracy = min(
        history[
            "accuracy"
        ].min(),
        history[
            "val_accuracy"
        ].min(),
    )

    axes[0].set_ylim(
        bottom=max(
            0.0,
            minimum_accuracy
            - 0.05,
        ),
        top=1.02,
    )

    axes[0].grid(
        alpha=0.3
    )

    axes[0].legend()

    # -------------------------------------------------------------------------
    # PANEL LOSS
    # -------------------------------------------------------------------------

    axes[1].plot(
        history[
            "epoch_display"
        ],
        history[
            "loss"
        ],
        marker="o",
        label="Train Loss",
    )

    axes[1].plot(
        history[
            "epoch_display"
        ],
        history[
            "val_loss"
        ],
        marker="o",
        label="Validation Loss",
    )

    axes[1].axvline(
        best_epoch,
        linestyle="--",
        label=(
            f"Best Epoch = "
            f"{best_epoch}"
        ),
    )

    axes[1].set_title(
        "Loss"
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "Loss"
    )

    axes[1].set_xticks(
        history[
            "epoch_display"
        ]
    )

    axes[1].set_ylim(
        bottom=0.0
    )

    axes[1].grid(
        alpha=0.3
    )

    axes[1].legend()

    # -------------------------------------------------------------------------
    # JUDUL UTAMA
    # -------------------------------------------------------------------------

    figure.suptitle(
        f"Training Curve - "
        f"{experiment['model']} "
        f"{experiment['scenario_code']}\n"
        f"{experiment['scenario_name']}"
    )

    figure.tight_layout(
        rect=[
            0.0,
            0.0,
            1.0,
            0.90,
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
# PROGRAM UTAMA
# =============================================================================

def main() -> None:
    """
    Membuat seluruh grafik training curve untuk
    10 eksperimen final.
    """

    print("=" * 80)

    print(
        "STEP 6.2 - TRAINING CURVES"
    )

    print("=" * 80)

    create_output_directories()

    training_report_rows: list[
        dict[str, Any]
    ] = []

    overfitting_report_rows: list[
        dict[str, Any]
    ] = []

    success_count = 0
    failed_count = 0

    for number, experiment in enumerate(
        EXPERIMENTS,
        start=1,
    ):
        experiment_name = experiment[
            "experiment_name"
        ]

        print(
            "\n" + "-" * 80
        )

        print(
            f"{number}/{len(EXPERIMENTS)} "
            f"- {experiment_name}"
        )

        try:
            # -----------------------------------------------------------------
            # MEMUAT HISTORY DAN SUMMARY
            # -----------------------------------------------------------------

            history = load_training_history(
                experiment_name
            )

            summary = load_training_summary(
                experiment_name
            )

            # -----------------------------------------------------------------
            # MENENTUKAN EPOCH TERBAIK
            # -----------------------------------------------------------------

            best_information = (
                get_best_epoch_information(
                    history
                )
            )

            best_epoch = int(
                best_information[
                    "best_epoch"
                ]
            )

            summary_best_epoch = (
                validate_summary_consistency(
                    experiment_name=(
                        experiment_name
                    ),
                    summary=summary,
                    best_information=(
                        best_information
                    ),
                )
            )

            # -----------------------------------------------------------------
            # ANALISIS OVERFITTING
            # -----------------------------------------------------------------

            overfitting_information = (
                analyze_overfitting(
                    history=history,
                    best_epoch=best_epoch,
                )
            )

            # -----------------------------------------------------------------
            # MEMBUAT GRAFIK
            # -----------------------------------------------------------------

            accuracy_path = (
                plot_accuracy_curve(
                    history=history,
                    experiment=experiment,
                    best_epoch=best_epoch,
                )
            )

            loss_path = (
                plot_loss_curve(
                    history=history,
                    experiment=experiment,
                    best_epoch=best_epoch,
                )
            )

            combined_path = (
                plot_combined_curve(
                    history=history,
                    experiment=experiment,
                    best_epoch=best_epoch,
                )
            )

            # -----------------------------------------------------------------
            # MEMBENTUK LAPORAN TRAINING CURVE
            # -----------------------------------------------------------------

            training_report_rows.append(
                {
                    "experiment_name":
                        experiment_name,

                    "model":
                        experiment[
                            "model"
                        ],

                    "dataset":
                        experiment[
                            "dataset"
                        ],

                    "scenario_code":
                        experiment[
                            "scenario_code"
                        ],

                    "scenario_name":
                        experiment[
                            "scenario_name"
                        ],

                    "epochs_completed":
                        int(
                            len(history)
                        ),

                    **best_information,

                    "summary_best_epoch":
                        summary_best_epoch,

                    "history_summary_consistent":
                        (
                            summary_best_epoch
                            is None
                            or summary_best_epoch
                            == best_epoch
                        ),

                    "accuracy_curve_path":
                        str(
                            accuracy_path
                        ),

                    "loss_curve_path":
                        str(
                            loss_path
                        ),

                    "combined_curve_path":
                        str(
                            combined_path
                        ),
                }
            )

            # -----------------------------------------------------------------
            # MEMBENTUK LAPORAN OVERFITTING
            # -----------------------------------------------------------------

            overfitting_report_rows.append(
                {
                    "experiment_name":
                        experiment_name,

                    "model":
                        experiment[
                            "model"
                        ],

                    "dataset":
                        experiment[
                            "dataset"
                        ],

                    "scenario_code":
                        experiment[
                            "scenario_code"
                        ],

                    "scenario_name":
                        experiment[
                            "scenario_name"
                        ],

                    "best_epoch":
                        best_epoch,

                    **overfitting_information,
                }
            )

            success_count += 1

            print(
                f"Epoch dijalankan    : "
                f"{len(history)}"
            )

            print(
                f"Best epoch         : "
                f"{best_epoch}"
            )

            print(
                f"Best val accuracy  : "
                f"{best_information['best_validation_accuracy']:.4f}"
            )

            print(
                f"Best val loss      : "
                f"{best_information['best_validation_loss']:.6f}"
            )

            print(
                f"Overfitting        : "
                f"{overfitting_information['overfitting_detected']}"
            )

            print(
                f"Grafik tersimpan   : "
                f"{combined_path}"
            )

        except Exception as error:
            failed_count += 1

            print(
                "Gagal memproses eksperimen:"
            )

            print(
                str(error)
            )

    # =========================================================================
    # MEMBENTUK DATAFRAME LAPORAN
    # =========================================================================

    training_report = pd.DataFrame(
        training_report_rows
    )

    overfitting_report = pd.DataFrame(
        overfitting_report_rows
    )

    if not training_report.empty:
        training_report = (
            training_report
            .sort_values(
                [
                    "dataset",
                    "scenario_code",
                    "model",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    if not overfitting_report.empty:
        overfitting_report = (
            overfitting_report
            .sort_values(
                [
                    "dataset",
                    "scenario_code",
                    "model",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    # =========================================================================
    # MENYIMPAN LAPORAN
    # =========================================================================

    training_report.to_csv(
        TRAINING_CURVE_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    overfitting_report.to_csv(
        OVERFITTING_REPORT_PATH,
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
        "RINGKASAN TRAINING CURVES"
    )

    print("=" * 80)

    if not training_report.empty:
        display_columns = [
            "experiment_name",
            "epochs_completed",
            "best_epoch",
            "best_validation_accuracy",
            "best_validation_loss",
        ]

        display_dataframe = (
            training_report[
                display_columns
            ]
            .copy()
        )

        display_dataframe[
            "best_validation_accuracy"
        ] = (
            display_dataframe[
                "best_validation_accuracy"
            ]
            .map(
                lambda value:
                f"{value:.6f}"
            )
        )

        display_dataframe[
            "best_validation_loss"
        ] = (
            display_dataframe[
                "best_validation_loss"
            ]
            .map(
                lambda value:
                f"{value:.6f}"
            )
        )

        print(
            "\n"
            + display_dataframe.to_string(
                index=False
            )
        )

    else:
        print(
            "\nTidak ada eksperimen yang "
            "berhasil diproses."
        )

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
        "\nLaporan training curve:"
    )

    print(
        TRAINING_CURVE_REPORT_PATH
    )

    print(
        "\nLaporan overfitting:"
    )

    print(
        OVERFITTING_REPORT_PATH
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "Tahap training curves selesai."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
# =============================================================================
# STEP 5.6 - RUN ALL EXPERIMENTS
# =============================================================================
# File:
# 5_modeling/run_all_experiments.py
#
# Tujuan:
# Menjalankan seluruh eksperimen CNN dan Attention-BiLSTM secara
# berurutan, mencatat status, durasi, dan memungkinkan resume.
#
# Contoh:
# python 5_modeling/run_all_experiments.py
#
# Menjalankan model tertentu:
# python 5_modeling/run_all_experiments.py --model cnn
#
# Menjalankan skenario tertentu:
# python 5_modeling/run_all_experiments.py --scenario K2
#
# Menjalankan ulang walaupun hasil sudah tersedia:
# python 5_modeling/run_all_experiments.py --force
# =============================================================================

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELING_DIR = PROJECT_ROOT / "5_modeling"

CHECKPOINTS_DIR = (
    PROJECT_ROOT
    / "8_save_models"
    / "checkpoints"
)

TRAINING_SUMMARY_DIR = (
    PROJECT_ROOT
    / "9_results"
    / "training_summary"
)

EXPERIMENT_LOG_DIR = (
    PROJECT_ROOT
    / "9_results"
    / "experiment_logs"
)

EXPERIMENT_STATUS_PATH = (
    EXPERIMENT_LOG_DIR
    / "all_experiments_status.csv"
)

EXPERIMENT_SUMMARY_PATH = (
    EXPERIMENT_LOG_DIR
    / "all_experiments_summary.json"
)


# =============================================================================
# KONFIGURASI EKSPERIMEN
# =============================================================================

ALL_MODELS = [
    "cnn",
    "attention_bilstm",
]

ALL_SCENARIOS = [
    "K1",
    "K2",
    "K3",
    "K4",
    "A1",
    "A2",
]

TRAINING_SCRIPTS = {
    "cnn": (
        MODELING_DIR
        / "train_cnn.py"
    ),
    "attention_bilstm": (
        MODELING_DIR
        / "train_attention_bilstm.py"
    ),
}


# =============================================================================
# NAMA EKSPERIMEN
# =============================================================================

def get_experiment_name(
    model_name: str,
    scenario_code: str,
) -> str:
    """
    Membentuk nama eksperimen.

    Contoh:
    cnn_k2
    attention_bilstm_a1
    """

    return (
        f"{model_name}_"
        f"{scenario_code.lower()}"
    )


# =============================================================================
# PATH OUTPUT EKSPERIMEN
# =============================================================================

def get_checkpoint_path(
    model_name: str,
    scenario_code: str,
) -> Path:
    """
    Menghasilkan path checkpoint terbaik.
    """

    experiment_name = (
        get_experiment_name(
            model_name,
            scenario_code,
        )
    )

    return (
        CHECKPOINTS_DIR
        / f"{experiment_name}_best.keras"
    )


def get_training_summary_path(
    model_name: str,
    scenario_code: str,
) -> Path:
    """
    Menghasilkan path ringkasan training.
    """

    experiment_name = (
        get_experiment_name(
            model_name,
            scenario_code,
        )
    )

    return (
        TRAINING_SUMMARY_DIR
        / f"{experiment_name}_summary.json"
    )


# =============================================================================
# PEMERIKSAAN HASIL YANG SUDAH ADA
# =============================================================================

def experiment_is_complete(
    model_name: str,
    scenario_code: str,
) -> bool:
    """
    Eksperimen dianggap selesai jika:
    - checkpoint terbaik tersedia;
    - ringkasan training tersedia.
    """

    checkpoint_path = (
        get_checkpoint_path(
            model_name,
            scenario_code,
        )
    )

    summary_path = (
        get_training_summary_path(
            model_name,
            scenario_code,
        )
    )

    return (
        checkpoint_path.exists()
        and summary_path.exists()
    )


# =============================================================================
# MEMBACA RINGKASAN TRAINING
# =============================================================================

def read_training_summary(
    model_name: str,
    scenario_code: str,
) -> dict:
    """
    Membaca training summary jika tersedia.
    """

    summary_path = (
        get_training_summary_path(
            model_name,
            scenario_code,
        )
    )

    if not summary_path.exists():
        return {}

    try:
        with open(
            summary_path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return {}


# =============================================================================
# MENJALANKAN SATU EKSPERIMEN
# =============================================================================

def run_single_experiment(
    model_name: str,
    scenario_code: str,
) -> dict:
    """
    Menjalankan satu script training melalui subprocess.

    Subprocess digunakan agar setiap eksperimen berjalan
    pada proses Python terpisah. Hal ini membantu membersihkan
    memori TensorFlow setelah eksperimen selesai.
    """

    experiment_name = (
        get_experiment_name(
            model_name,
            scenario_code,
        )
    )

    training_script = (
        TRAINING_SCRIPTS[
            model_name
        ]
    )

    if not training_script.exists():
        raise FileNotFoundError(
            "Script training tidak ditemukan:\n"
            f"{training_script}"
        )

    command = [
        sys.executable,
        str(training_script),
        scenario_code,
    ]

    print("\n" + "=" * 80)
    print(
        f"MEMULAI EKSPERIMEN: "
        f"{experiment_name}"
    )
    print("=" * 80)

    print(
        "Perintah:"
    )

    print(
        " ".join(command)
    )

    start_datetime = datetime.now()

    start_time = time.perf_counter()

    try:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
        )

        return_code = (
            process.returncode
        )

        status = (
            "success"
            if return_code == 0
            else "failed"
        )

        error_message = (
            ""
            if return_code == 0
            else (
                "Script training berhenti "
                f"dengan return code {return_code}."
            )
        )

    except KeyboardInterrupt:
        status = "interrupted"
        return_code = -1
        error_message = (
            "Eksperimen dihentikan oleh pengguna."
        )

    except Exception as error:
        status = "failed"
        return_code = -1
        error_message = str(error)

    end_time = time.perf_counter()

    end_datetime = datetime.now()

    duration_seconds = (
        end_time
        - start_time
    )

    training_summary = (
        read_training_summary(
            model_name,
            scenario_code,
        )
    )

    result = {
        "experiment_name":
            experiment_name,

        "model":
            model_name,

        "scenario_code":
            scenario_code,

        "status":
            status,

        "return_code":
            return_code,

        "start_time":
            start_datetime.isoformat(
                timespec="seconds"
            ),

        "end_time":
            end_datetime.isoformat(
                timespec="seconds"
            ),

        "duration_seconds":
            round(
                duration_seconds,
                4,
            ),

        "duration_minutes":
            round(
                duration_seconds / 60,
                4,
            ),

        "best_epoch":
            training_summary.get(
                "best_epoch"
            ),

        "best_validation_loss":
            training_summary.get(
                "best_validation_loss"
            ),

        "best_validation_accuracy":
            training_summary.get(
                "best_validation_accuracy"
            ),

        "epochs_completed":
            training_summary.get(
                "epochs_completed"
            ),

        "error_message":
            error_message,

        "checkpoint_path":
            str(
                get_checkpoint_path(
                    model_name,
                    scenario_code,
                )
            ),

        "summary_path":
            str(
                get_training_summary_path(
                    model_name,
                    scenario_code,
                )
            ),
    }

    print("\n" + "-" * 80)

    print(
        f"Status eksperimen : "
        f"{status}"
    )

    print(
        f"Durasi            : "
        f"{duration_seconds / 60:.2f} menit"
    )

    if error_message:
        print(
            f"Pesan             : "
            f"{error_message}"
        )

    print("-" * 80)

    return result


# =============================================================================
# MEMBACA STATUS SEBELUMNYA
# =============================================================================

def load_existing_status() -> pd.DataFrame:
    """
    Membaca laporan eksperimen yang sudah ada.
    """

    if not EXPERIMENT_STATUS_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            EXPERIMENT_STATUS_PATH,
            encoding="utf-8-sig",
        )

    except Exception:
        return pd.DataFrame()


# =============================================================================
# MENYIMPAN STATUS
# =============================================================================

def save_status_report(
    results: list[dict],
) -> pd.DataFrame:
    """
    Menyimpan seluruh status eksperimen ke CSV.

    Jika eksperimen yang sama dijalankan kembali,
    hasil terbaru menggantikan status sebelumnya.
    """

    existing_status = (
        load_existing_status()
    )

    new_status = pd.DataFrame(
        results
    )

    if existing_status.empty:
        combined_status = (
            new_status.copy()
        )

    else:
        combined_status = pd.concat(
            [
                existing_status,
                new_status,
            ],
            ignore_index=True,
        )

    if not combined_status.empty:
        combined_status = (
            combined_status
            .drop_duplicates(
                subset=[
                    "experiment_name",
                ],
                keep="last",
            )
            .sort_values(
                [
                    "model",
                    "scenario_code",
                ]
            )
            .reset_index(drop=True)
        )

    EXPERIMENT_LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_status.to_csv(
        EXPERIMENT_STATUS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    return combined_status


# =============================================================================
# MENYIMPAN SUMMARY GLOBAL
# =============================================================================

def save_global_summary(
    status_dataframe: pd.DataFrame,
) -> None:
    """
    Menyimpan ringkasan seluruh eksperimen dalam JSON.
    """

    if status_dataframe.empty:
        summary = {
            "total_experiments": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "interrupted": 0,
        }

    else:
        status_counts = (
            status_dataframe[
                "status"
            ]
            .value_counts()
            .to_dict()
        )

        summary = {
            "generated_at":
                datetime.now().isoformat(
                    timespec="seconds"
                ),

            "total_experiments":
                int(
                    len(
                        status_dataframe
                    )
                ),

            "success":
                int(
                    status_counts.get(
                        "success",
                        0,
                    )
                ),

            "failed":
                int(
                    status_counts.get(
                        "failed",
                        0,
                    )
                ),

            "skipped":
                int(
                    status_counts.get(
                        "skipped",
                        0,
                    )
                ),

            "interrupted":
                int(
                    status_counts.get(
                        "interrupted",
                        0,
                    )
                ),

            "total_duration_minutes":
                round(
                    float(
                        status_dataframe[
                            "duration_minutes"
                        ]
                        .fillna(0)
                        .sum()
                    ),
                    4,
                ),

            "experiments":
                status_dataframe
                .fillna("")
                .to_dict(
                    orient="records"
                ),
        }

    with open(
        EXPERIMENT_SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=4,
        )


# =============================================================================
# MEMBUAT DAFTAR EKSPERIMEN
# =============================================================================

def build_experiment_queue(
    selected_model: str | None,
    selected_scenario: str | None,
) -> list[tuple[str, str]]:
    """
    Membuat daftar eksperimen berdasarkan argumen pengguna.
    """

    models = (
        [selected_model]
        if selected_model
        else ALL_MODELS
    )

    scenarios = (
        [selected_scenario]
        if selected_scenario
        else ALL_SCENARIOS
    )

    queue = []

    for model_name in models:
        for scenario_code in scenarios:
            queue.append(
                (
                    model_name,
                    scenario_code,
                )
            )

    return queue


# =============================================================================
# VALIDASI ARGUMEN
# =============================================================================

def validate_arguments(
    model_name: str | None,
    scenario_code: str | None,
) -> None:
    """
    Memastikan argumen valid.
    """

    if (
        model_name is not None
        and model_name not in ALL_MODELS
    ):
        raise ValueError(
            f"Model tidak valid: {model_name}"
        )

    if (
        scenario_code is not None
        and scenario_code not in ALL_SCENARIOS
    ):
        raise ValueError(
            f"Skenario tidak valid: "
            f"{scenario_code}"
        )


# =============================================================================
# PROGRAM UTAMA
# =============================================================================

def main() -> None:
    """
    Menjalankan seluruh eksperimen yang dipilih.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Menjalankan eksperimen CNN dan "
            "Attention-BiLSTM."
        )
    )

    parser.add_argument(
        "--model",
        choices=ALL_MODELS,
        default=None,
        help=(
            "Batasi eksperimen pada satu model."
        ),
    )

    parser.add_argument(
        "--scenario",
        choices=ALL_SCENARIOS,
        default=None,
        help=(
            "Batasi eksperimen pada satu skenario."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Jalankan ulang eksperimen walaupun "
            "checkpoint dan summary sudah tersedia."
        ),
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help=(
            "Hentikan seluruh proses jika satu "
            "eksperimen gagal."
        ),
    )

    arguments = parser.parse_args()

    validate_arguments(
        arguments.model,
        arguments.scenario,
    )

    EXPERIMENT_LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    queue = build_experiment_queue(
        selected_model=arguments.model,
        selected_scenario=(
            arguments.scenario
        ),
    )

    print("=" * 80)
    print("STEP 5.6 - RUN ALL EXPERIMENTS")
    print("=" * 80)

    print(
        f"\nJumlah eksperimen dalam antrean: "
        f"{len(queue)}"
    )

    print(
        f"Force rerun                 : "
        f"{arguments.force}"
    )

    print(
        f"Stop on error               : "
        f"{arguments.stop_on_error}"
    )

    print("\nDaftar eksperimen:")

    for number, (
        model_name,
        scenario_code,
    ) in enumerate(
        queue,
        start=1,
    ):
        experiment_name = (
            get_experiment_name(
                model_name,
                scenario_code,
            )
        )

        completed = (
            experiment_is_complete(
                model_name,
                scenario_code,
            )
        )

        status_text = (
            "sudah tersedia"
            if completed
            else "belum tersedia"
        )

        print(
            f"{number:02d}. "
            f"{experiment_name:<25} "
            f"{status_text}"
        )

    results: list[dict] = []

    overall_start_time = (
        time.perf_counter()
    )

    for index, (
        model_name,
        scenario_code,
    ) in enumerate(
        queue,
        start=1,
    ):
        experiment_name = (
            get_experiment_name(
                model_name,
                scenario_code,
            )
        )

        print(
            "\n" + "#" * 80
        )

        print(
            f"EKSPERIMEN {index}/{len(queue)}"
        )

        print(
            f"Nama: {experiment_name}"
        )

        print(
            "#" * 80
        )

        if (
            experiment_is_complete(
                model_name,
                scenario_code,
            )
            and not arguments.force
        ):
            print(
                "Eksperimen dilewati karena "
                "checkpoint dan summary sudah tersedia."
            )

            existing_summary = (
                read_training_summary(
                    model_name,
                    scenario_code,
                )
            )

            result = {
                "experiment_name":
                    experiment_name,

                "model":
                    model_name,

                "scenario_code":
                    scenario_code,

                "status":
                    "skipped",

                "return_code":
                    0,

                "start_time":
                    datetime.now().isoformat(
                        timespec="seconds"
                    ),

                "end_time":
                    datetime.now().isoformat(
                        timespec="seconds"
                    ),

                "duration_seconds":
                    0.0,

                "duration_minutes":
                    0.0,

                "best_epoch":
                    existing_summary.get(
                        "best_epoch"
                    ),

                "best_validation_loss":
                    existing_summary.get(
                        "best_validation_loss"
                    ),

                "best_validation_accuracy":
                    existing_summary.get(
                        "best_validation_accuracy"
                    ),

                "epochs_completed":
                    existing_summary.get(
                        "epochs_completed"
                    ),

                "error_message":
                    "",

                "checkpoint_path":
                    str(
                        get_checkpoint_path(
                            model_name,
                            scenario_code,
                        )
                    ),

                "summary_path":
                    str(
                        get_training_summary_path(
                            model_name,
                            scenario_code,
                        )
                    ),
            }

        else:
            result = run_single_experiment(
                model_name=model_name,
                scenario_code=scenario_code,
            )

        results.append(
            result
        )

        status_dataframe = (
            save_status_report(
                results
            )
        )

        save_global_summary(
            status_dataframe
        )

        if (
            result["status"]
            in {
                "failed",
                "interrupted",
            }
            and arguments.stop_on_error
        ):
            print(
                "\nProses dihentikan karena "
                "terjadi kegagalan."
            )
            break

        if result["status"] == "interrupted":
            print(
                "\nProses dihentikan oleh pengguna."
            )
            break

    overall_duration_seconds = (
        time.perf_counter()
        - overall_start_time
    )

    final_status = (
        save_status_report(
            results
        )
    )

    save_global_summary(
        final_status
    )

    print("\n" + "=" * 80)
    print("RINGKASAN SELURUH EKSPERIMEN")
    print("=" * 80)

    if final_status.empty:
        print(
            "\nBelum ada hasil eksperimen."
        )

    else:
        display_columns = [
            "experiment_name",
            "status",
            "best_epoch",
            "best_validation_accuracy",
            "best_validation_loss",
            "duration_minutes",
        ]

        available_columns = [
            column
            for column in display_columns
            if column in final_status.columns
        ]

        print(
            "\n"
            + final_status[
                available_columns
            ].to_string(
                index=False
            )
        )

    print(
        f"\nTotal waktu proses: "
        f"{overall_duration_seconds / 60:.2f} menit"
    )

    print(
        "\nLaporan status:"
    )

    print(
        EXPERIMENT_STATUS_PATH
    )

    print(
        "\nRingkasan JSON:"
    )

    print(
        EXPERIMENT_SUMMARY_PATH
    )

    print("\n" + "=" * 80)
    print("Proses eksperimen selesai.")
    print("=" * 80)


if __name__ == "__main__":
    main()
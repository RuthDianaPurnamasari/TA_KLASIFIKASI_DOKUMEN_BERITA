# =============================================================================
# STEP 5.6 - RUN ALL EXPERIMENTS
# =============================================================================
# File:
# 5_modeling/run_all_experiments.py
#
# Tujuan:
# Menjalankan seluruh eksperimen CNN dan Attention-BiLSTM secara
# berurutan, mencatat status, durasi, log terminal, dan mendukung
# proses resume.
#
# Eksperimen final:
#
# CNN:
# - K1, K2, K3
# - A1, A2
#
# Attention-BiLSTM:
# - K1, K2, K3
# - A1, A2
#
# Total: 10 eksperimen
#
# Contoh:
# python 5_modeling/run_all_experiments.py
#
# Model tertentu:
# python 5_modeling/run_all_experiments.py --model cnn
#
# Skenario tertentu:
# python 5_modeling/run_all_experiments.py --scenario K2
#
# Satu kombinasi:
# python 5_modeling/run_all_experiments.py --model cnn --scenario K2
#
# Menjalankan ulang eksperimen:
# python 5_modeling/run_all_experiments.py --force
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
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

MODELING_DIR = (
    PROJECT_ROOT
    / "5_modeling"
)

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

# K4 tidak digunakan dalam eksperimen final.
ALL_SCENARIOS = [
    "K1",
    "K2",
    "K3",
    "A1",
    "A2",
]

SCENARIO_DATASETS = {
    "K1": "Kompas",
    "K2": "Kompas",
    "K3": "Kompas",
    "A1": "AG News",
    "A2": "AG News",
}

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

    experiment_name = get_experiment_name(
        model_name=model_name,
        scenario_code=scenario_code,
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

    experiment_name = get_experiment_name(
        model_name=model_name,
        scenario_code=scenario_code,
    )

    return (
        TRAINING_SUMMARY_DIR
        / f"{experiment_name}_summary.json"
    )


def get_experiment_log_path(
    model_name: str,
    scenario_code: str,
) -> Path:
    """
    Menghasilkan path log terminal eksperimen.
    """

    experiment_name = get_experiment_name(
        model_name=model_name,
        scenario_code=scenario_code,
    )

    return (
        EXPERIMENT_LOG_DIR
        / f"{experiment_name}.log"
    )


# =============================================================================
# MEMBACA RINGKASAN TRAINING
# =============================================================================

def read_training_summary(
    model_name: str,
    scenario_code: str,
) -> dict:
    """
    Membaca training summary jika tersedia dan valid.
    """

    summary_path = get_training_summary_path(
        model_name=model_name,
        scenario_code=scenario_code,
    )

    if not summary_path.exists():
        return {}

    if not summary_path.is_file():
        return {}

    try:
        with open(
            summary_path,
            "r",
            encoding="utf-8",
        ) as file:
            content = json.load(file)

        if not isinstance(
            content,
            dict,
        ):
            return {}

        return content

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return {}


# =============================================================================
# MENGAMBIL NILAI SUMMARY
# =============================================================================

def get_summary_value(
    summary: dict,
    candidate_keys: list[str],
):
    """
    Mengambil nilai berdasarkan beberapa kemungkinan nama key.
    """

    for key in candidate_keys:
        if (
            key in summary
            and summary[key] is not None
        ):
            return summary[key]

    return None


# =============================================================================
# PEMERIKSAAN KELENGKAPAN ARTEFAK
# =============================================================================

def experiment_is_complete(
    model_name: str,
    scenario_code: str,
) -> bool:
    """
    Eksperimen dianggap selesai jika:

    - checkpoint terbaik tersedia dan tidak kosong;
    - training summary tersedia dan valid;
    - summary tidak berstatus gagal;
    - metrik utama tersedia.
    """

    checkpoint_path = get_checkpoint_path(
        model_name=model_name,
        scenario_code=scenario_code,
    )

    if not checkpoint_path.exists():
        return False

    if not checkpoint_path.is_file():
        return False

    if checkpoint_path.stat().st_size <= 0:
        return False

    summary = read_training_summary(
        model_name=model_name,
        scenario_code=scenario_code,
    )

    if not summary:
        return False

    summary_status = str(
        summary.get(
            "status",
            "success",
        )
    ).strip().lower()

    if summary_status in {
        "failed",
        "error",
        "interrupted",
    }:
        return False

    required_values = [
        get_summary_value(
            summary,
            [
                "best_epoch",
            ],
        ),
        get_summary_value(
            summary,
            [
                "best_validation_loss",
                "best_val_loss",
            ],
        ),
        get_summary_value(
            summary,
            [
                "best_validation_accuracy",
                "best_val_accuracy",
            ],
        ),
        get_summary_value(
            summary,
            [
                "epochs_completed",
                "completed_epochs",
            ],
        ),
    ]

    return all(
        value is not None
        for value in required_values
    )


# =============================================================================
# VALIDASI FILE PROJECT
# =============================================================================

def validate_project_files() -> None:
    """
    Memastikan seluruh script training tersedia.
    """

    missing_scripts = [
        str(script_path)
        for script_path
        in TRAINING_SCRIPTS.values()
        if not script_path.exists()
    ]

    if missing_scripts:
        raise FileNotFoundError(
            "Script training berikut tidak ditemukan:\n"
            + "\n".join(missing_scripts)
        )


# =============================================================================
# MEMBUAT RESULT BERDASARKAN SUMMARY
# =============================================================================

def build_result_record(
    model_name: str,
    scenario_code: str,
    status: str,
    run_action: str,
    return_code: int,
    start_datetime: datetime,
    end_datetime: datetime,
    duration_seconds: float,
    error_message: str = "",
) -> dict:
    """
    Membuat satu record hasil eksperimen.
    """

    experiment_name = get_experiment_name(
        model_name=model_name,
        scenario_code=scenario_code,
    )

    training_summary = read_training_summary(
        model_name=model_name,
        scenario_code=scenario_code,
    )

    checkpoint_path = get_checkpoint_path(
        model_name=model_name,
        scenario_code=scenario_code,
    )

    summary_path = get_training_summary_path(
        model_name=model_name,
        scenario_code=scenario_code,
    )

    log_path = get_experiment_log_path(
        model_name=model_name,
        scenario_code=scenario_code,
    )

    return {
        "experiment_name":
            experiment_name,

        "dataset":
            SCENARIO_DATASETS[
                scenario_code
            ],

        "model":
            model_name,

        "scenario_code":
            scenario_code,

        "status":
            status,

        "run_action":
            run_action,

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
            get_summary_value(
                training_summary,
                [
                    "best_epoch",
                ],
            ),

        "best_validation_loss":
            get_summary_value(
                training_summary,
                [
                    "best_validation_loss",
                    "best_val_loss",
                ],
            ),

        "best_validation_accuracy":
            get_summary_value(
                training_summary,
                [
                    "best_validation_accuracy",
                    "best_val_accuracy",
                ],
            ),

        "epochs_completed":
            get_summary_value(
                training_summary,
                [
                    "epochs_completed",
                    "completed_epochs",
                ],
            ),

        "artifact_complete":
            experiment_is_complete(
                model_name=model_name,
                scenario_code=scenario_code,
            ),

        "checkpoint_exists":
            checkpoint_path.exists(),

        "summary_exists":
            summary_path.exists(),

        "error_message":
            error_message,

        "checkpoint_path":
            str(checkpoint_path),

        "summary_path":
            str(summary_path),

        "log_path":
            str(log_path),
    }


# =============================================================================
# MENJALANKAN SATU EKSPERIMEN
# =============================================================================

def run_single_experiment(
    model_name: str,
    scenario_code: str,
) -> dict:
    """
    Menjalankan satu script training melalui subprocess.

    Setiap eksperimen berjalan pada proses Python terpisah
    sehingga memori TensorFlow dapat dibersihkan setelah
    eksperimen selesai.

    Output terminal tetap ditampilkan dan disimpan ke file log.
    """

    experiment_name = get_experiment_name(
        model_name=model_name,
        scenario_code=scenario_code,
    )

    training_script = TRAINING_SCRIPTS[
        model_name
    ]

    if not training_script.exists():
        raise FileNotFoundError(
            "Script training tidak ditemukan:\n"
            f"{training_script}"
        )

    log_path = get_experiment_log_path(
        model_name=model_name,
        scenario_code=scenario_code,
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

    print("\nPerintah:")

    print(
        " ".join(command)
    )

    print("\nLog eksperimen:")

    print(log_path)

    start_datetime = datetime.now()
    start_time = time.perf_counter()

    status = "failed"
    return_code = -1
    error_message = ""

    process: subprocess.Popen | None = None

    environment = os.environ.copy()

    # Menampilkan log Python subprocess secara langsung.
    environment[
        "PYTHONUNBUFFERED"
    ] = "1"

    try:
        with open(
            log_path,
            "w",
            encoding="utf-8",
        ) as log_file:

            log_file.write(
                "=" * 80
                + "\n"
            )

            log_file.write(
                f"EXPERIMENT: "
                f"{experiment_name}\n"
            )

            log_file.write(
                f"START TIME: "
                f"{start_datetime.isoformat()}\n"
            )

            log_file.write(
                f"COMMAND: "
                f"{' '.join(command)}\n"
            )

            log_file.write(
                "=" * 80
                + "\n\n"
            )

            log_file.flush()

            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
            )

            if process.stdout is None:
                raise RuntimeError(
                    "Output subprocess tidak tersedia."
                )

            for output_line in process.stdout:
                print(
                    output_line,
                    end="",
                )

                log_file.write(
                    output_line
                )

                log_file.flush()

            return_code = process.wait()

        if return_code != 0:
            status = "failed"

            error_message = (
                "Script training berhenti dengan "
                f"return code {return_code}."
            )

        elif not experiment_is_complete(
            model_name=model_name,
            scenario_code=scenario_code,
        ):
            status = "failed"

            error_message = (
                "Script training selesai tanpa error, "
                "tetapi checkpoint atau training summary "
                "belum lengkap."
            )

        else:
            status = "success"
            error_message = ""

    except KeyboardInterrupt:
        status = "interrupted"
        return_code = -1

        error_message = (
            "Eksperimen dihentikan oleh pengguna."
        )

        if (
            process is not None
            and process.poll() is None
        ):
            process.terminate()

            try:
                process.wait(
                    timeout=10
                )

            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    except Exception as error:
        status = "failed"
        return_code = -1
        error_message = (
            f"{type(error).__name__}: "
            f"{error}"
        )

        if (
            process is not None
            and process.poll() is None
        ):
            process.terminate()

    end_time = time.perf_counter()
    end_datetime = datetime.now()

    duration_seconds = (
        end_time
        - start_time
    )

    result = build_result_record(
        model_name=model_name,
        scenario_code=scenario_code,
        status=status,
        run_action="executed",
        return_code=return_code,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        duration_seconds=duration_seconds,
        error_message=error_message,
    )

    print("\n" + "-" * 80)

    print(
        f"Status eksperimen : "
        f"{status}"
    )

    print(
        f"Artefak lengkap   : "
        f"{result['artifact_complete']}"
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
# MEMBUAT RESULT UNTUK EKSPERIMEN YANG DILEWATI
# =============================================================================

def create_skipped_result(
    model_name: str,
    scenario_code: str,
) -> dict:
    """
    Membuat record untuk eksperimen yang tidak dijalankan ulang
    karena artefaknya sudah lengkap.

    Status tetap success karena model sudah berhasil tersedia.
    """

    current_datetime = datetime.now()

    return build_result_record(
        model_name=model_name,
        scenario_code=scenario_code,
        status="success",
        run_action="skipped_existing",
        return_code=0,
        start_datetime=current_datetime,
        end_datetime=current_datetime,
        duration_seconds=0.0,
        error_message="",
    )


# =============================================================================
# MEMBACA STATUS SEBELUMNYA
# =============================================================================

def load_existing_status() -> pd.DataFrame:
    """
    Membaca laporan eksperimen sebelumnya.

    Record K4 atau skenario lain yang sudah tidak digunakan
    otomatis dikeluarkan dari laporan aktif.
    """

    if not EXPERIMENT_STATUS_PATH.exists():
        return pd.DataFrame()

    try:
        dataframe = pd.read_csv(
            EXPERIMENT_STATUS_PATH,
            encoding="utf-8-sig",
            keep_default_na=False,
        )

    except (
        OSError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ):
        return pd.DataFrame()

    required_columns = {
        "model",
        "scenario_code",
        "experiment_name",
    }

    if not required_columns.issubset(
        dataframe.columns
    ):
        return pd.DataFrame()

    dataframe = dataframe[
        dataframe["model"].isin(
            ALL_MODELS
        )
        & dataframe[
            "scenario_code"
        ].isin(
            ALL_SCENARIOS
        )
    ].copy()

    return dataframe.reset_index(
        drop=True
    )


# =============================================================================
# MENYIMPAN STATUS
# =============================================================================

def save_status_report(
    new_results: list[dict],
) -> pd.DataFrame:
    """
    Menyimpan status eksperimen ke CSV.

    Hasil terbaru menggantikan record sebelumnya dengan nama
    eksperimen yang sama.
    """

    existing_status = load_existing_status()

    new_status = pd.DataFrame(
        new_results
    )

    if existing_status.empty:
        combined_status = (
            new_status.copy()
        )

    elif new_status.empty:
        combined_status = (
            existing_status.copy()
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
        combined_status = combined_status[
            combined_status["model"].isin(
                ALL_MODELS
            )
            & combined_status[
                "scenario_code"
            ].isin(
                ALL_SCENARIOS
            )
        ]

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
                ],
                kind="stable",
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

    EXPERIMENT_LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if status_dataframe.empty:
        summary = {
            "generated_at":
                datetime.now().isoformat(
                    timespec="seconds"
                ),
            "expected_total_experiments":
                len(ALL_MODELS)
                * len(ALL_SCENARIOS),
            "recorded_experiments": 0,
            "success": 0,
            "failed": 0,
            "interrupted": 0,
            "skipped_existing": 0,
            "total_duration_minutes": 0.0,
            "experiments": [],
        }

    else:
        status_counts = (
            status_dataframe["status"]
            .value_counts()
            .to_dict()
        )

        action_counts = (
            status_dataframe["run_action"]
            .value_counts()
            .to_dict()
            if "run_action"
            in status_dataframe.columns
            else {}
        )

        duration_series = pd.to_numeric(
            status_dataframe[
                "duration_minutes"
            ],
            errors="coerce",
        ).fillna(0)

        summary = {
            "generated_at":
                datetime.now().isoformat(
                    timespec="seconds"
                ),

            "expected_total_experiments":
                len(ALL_MODELS)
                * len(ALL_SCENARIOS),

            "recorded_experiments":
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

            "interrupted":
                int(
                    status_counts.get(
                        "interrupted",
                        0,
                    )
                ),

            "skipped_existing":
                int(
                    action_counts.get(
                        "skipped_existing",
                        0,
                    )
                ),

            "total_duration_minutes":
                round(
                    float(
                        duration_series.sum()
                    ),
                    4,
                ),

            "k4_used":
                False,

            "models":
                ALL_MODELS,

            "scenarios":
                ALL_SCENARIOS,

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
        if selected_model is not None
        else ALL_MODELS
    )

    scenarios = (
        [selected_scenario]
        if selected_scenario is not None
        else ALL_SCENARIOS
    )

    return [
        (
            model_name,
            scenario_code,
        )
        for model_name in models
        for scenario_code in scenarios
    ]


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
        type=str.lower,
        choices=ALL_MODELS,
        default=None,
        help=(
            "Batasi eksperimen pada satu model."
        ),
    )

    parser.add_argument(
        "--scenario",
        type=str.upper,
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
            "Hentikan seluruh proses ketika satu "
            "eksperimen gagal."
        ),
    )

    arguments = parser.parse_args()

    validate_project_files()

    CHECKPOINTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TRAINING_SUMMARY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXPERIMENT_LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    queue = build_experiment_queue(
        selected_model=arguments.model,
        selected_scenario=arguments.scenario,
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
        experiment_name = get_experiment_name(
            model_name=model_name,
            scenario_code=scenario_code,
        )

        completed = experiment_is_complete(
            model_name=model_name,
            scenario_code=scenario_code,
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

    overall_start_time = (
        time.perf_counter()
    )

    final_status = load_existing_status()

    for index, (
        model_name,
        scenario_code,
    ) in enumerate(
        queue,
        start=1,
    ):
        experiment_name = get_experiment_name(
            model_name=model_name,
            scenario_code=scenario_code,
        )

        print(
            "\n"
            + "#"
            * 80
        )

        print(
            f"EKSPERIMEN "
            f"{index}/{len(queue)}"
        )

        print(
            f"Nama: {experiment_name}"
        )

        print(
            "#"
            * 80
        )

        if (
            experiment_is_complete(
                model_name=model_name,
                scenario_code=scenario_code,
            )
            and not arguments.force
        ):
            print(
                "Eksperimen dilewati karena "
                "checkpoint dan summary sudah lengkap."
            )

            result = create_skipped_result(
                model_name=model_name,
                scenario_code=scenario_code,
            )

        else:
            result = run_single_experiment(
                model_name=model_name,
                scenario_code=scenario_code,
            )

        final_status = save_status_report(
            [
                result
            ]
        )

        save_global_summary(
            final_status
        )

        if result["status"] == "interrupted":
            print(
                "\nProses dihentikan oleh pengguna."
            )
            break

        if (
            result["status"] == "failed"
            and arguments.stop_on_error
        ):
            print(
                "\nProses dihentikan karena "
                "eksperimen gagal."
            )
            break

    overall_duration_seconds = (
        time.perf_counter()
        - overall_start_time
    )

    final_status = load_existing_status()

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
            "dataset",
            "status",
            "run_action",
            "best_epoch",
            "best_validation_accuracy",
            "best_validation_loss",
            "duration_minutes",
        ]

        available_columns = [
            column
            for column in display_columns
            if column
            in final_status.columns
        ]

        print(
            "\n"
            + final_status[
                available_columns
            ].to_string(
                index=False
            )
        )

    success_count = 0
    failed_count = 0
    interrupted_count = 0

    if not final_status.empty:
        success_count = int(
            (
                final_status["status"]
                == "success"
            ).sum()
        )

        failed_count = int(
            (
                final_status["status"]
                == "failed"
            ).sum()
        )

        interrupted_count = int(
            (
                final_status["status"]
                == "interrupted"
            ).sum()
        )

    print("\nRingkasan status:")

    print(
        f"Berhasil     : "
        f"{success_count}"
    )

    print(
        f"Gagal        : "
        f"{failed_count}"
    )

    print(
        f"Terinterupsi : "
        f"{interrupted_count}"
    )

    print(
        f"\nTotal waktu proses: "
        f"{overall_duration_seconds / 60:.2f} menit"
    )

    print("\nLaporan status:")

    print(
        EXPERIMENT_STATUS_PATH
    )

    print("\nRingkasan JSON:")

    print(
        EXPERIMENT_SUMMARY_PATH
    )

    print("\n" + "=" * 80)
    print("Proses eksperimen selesai.")
    print("=" * 80)


if __name__ == "__main__":
    main()
# =============================================================================
# STEP 5.4 - TRAIN CNN
# =============================================================================
# File:
# 5_modeling/train_cnn.py
#
# Tujuan:
# Melatih model CNN pada satu skenario representasi teks.
#
# Contoh:
# python 5_modeling/train_cnn.py K1
#
# Alur:
# 1. Membaca konfigurasi skenario
# 2. Memuat train dan validation dari file NPZ
# 3. Memvalidasi bentuk data, label, token, dan document_id
# 4. Membuat tf.data.Dataset
# 5. Membangun model CNN
# 6. Melatih model
# 7. Menyimpan checkpoint berdasarkan val_loss terbaik
# 8. Memuat kembali checkpoint terbaik
# 9. Menyimpan checkpoint terbaik sebagai model final
# 10. Menyimpan history dan training summary
#
# Test set tidak digunakan dalam proses training.
# =============================================================================

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf


# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# =============================================================================
# IMPORT MODULE PROJECT
# =============================================================================

from cnn_model import build_cnn_model  # noqa: E402

from training_config import (  # noqa: E402
    RANDOM_SEED,
    NUM_CLASSES,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    VERBOSE,
    EMBEDDING_DIM,
    DENSE_UNITS,
    SPATIAL_DROPOUT_RATE,
    DROPOUT_RATE,
    CNN_NUM_FILTERS,
    CNN_KERNEL_SIZE,
    set_global_seed,
    create_training_directories,
    get_scenario_config,
    get_vocabulary_size,
    get_split_path,
    get_experiment_name,
    get_checkpoint_path,
    get_final_model_path,
    get_history_path,
    create_training_callbacks,
    validate_scenario_code,
)


# =============================================================================
# KONFIGURASI EKSPERIMEN FINAL
# =============================================================================

VALID_SCENARIOS = [
    "K1",
    "K2",
    "K3",
    "A1",
    "A2",
]

MODEL_CODE = "cnn"

MODEL_DISPLAY_NAME = "CNN"


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

TRAINING_SUMMARY_DIR = (
    PROJECT_ROOT
    / "9_results"
    / "training_summary"
)


# =============================================================================
# MEMUAT FILE NPZ
# =============================================================================

def load_npz_dataset(
    file_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Memuat data hasil text vectorization dari file NPZ.

    File NPZ wajib memiliki:
    - X
    - y
    - document_id
    - category

    Returns
    -------
    tuple
        X, y, document_id, category.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            "File dataset tidak ditemukan:\n"
            f"{file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            "Path dataset bukan file:\n"
            f"{file_path}"
        )

    if file_path.stat().st_size <= 0:
        raise ValueError(
            "File dataset kosong:\n"
            f"{file_path}"
        )

    with np.load(
        file_path,
        allow_pickle=False,
    ) as data:

        available_keys = set(
            data.files
        )

        required_keys = {
            "X",
            "y",
            "document_id",
            "category",
        }

        missing_keys = (
            required_keys
            - available_keys
        )

        if missing_keys:
            raise KeyError(
                "File NPZ tidak memiliki seluruh key "
                "yang diperlukan.\n"
                f"Key wajib      : {sorted(required_keys)}\n"
                f"Key ditemukan : {sorted(available_keys)}\n"
                f"Key tidak ada : {sorted(missing_keys)}"
            )

        X = np.asarray(
            data["X"]
        )

        y = np.asarray(
            data["y"]
        )

        document_ids = np.asarray(
            data["document_id"]
        )

        categories = np.asarray(
            data["category"]
        )

    return (
        X,
        y,
        document_ids,
        categories,
    )


# =============================================================================
# VALIDASI DATASET
# =============================================================================

def validate_dataset(
    X: np.ndarray,
    y: np.ndarray,
    document_ids: np.ndarray,
    categories: np.ndarray,
    split_name: str,
    expected_sequence_length: int,
    vocabulary_size: int,
) -> None:
    """
    Memastikan array hasil vectorization sesuai kebutuhan CNN.
    """

    # -------------------------------------------------------------------------
    # VALIDASI DIMENSI
    # -------------------------------------------------------------------------

    if X.ndim != 2:
        raise ValueError(
            f"{split_name}: X harus memiliki dua dimensi.\n"
            f"Shape ditemukan: {X.shape}"
        )

    if y.ndim != 1:
        raise ValueError(
            f"{split_name}: y harus memiliki satu dimensi.\n"
            f"Shape ditemukan: {y.shape}"
        )

    if document_ids.ndim != 1:
        raise ValueError(
            f"{split_name}: document_id harus "
            "memiliki satu dimensi.\n"
            f"Shape ditemukan: {document_ids.shape}"
        )

    if categories.ndim != 1:
        raise ValueError(
            f"{split_name}: category harus "
            "memiliki satu dimensi.\n"
            f"Shape ditemukan: {categories.shape}"
        )

    # -------------------------------------------------------------------------
    # VALIDASI JUMLAH DATA
    # -------------------------------------------------------------------------

    row_counts = {
        "X": len(X),
        "y": len(y),
        "document_id": len(document_ids),
        "category": len(categories),
    }

    if len(
        set(
            row_counts.values()
        )
    ) != 1:
        raise ValueError(
            f"{split_name}: jumlah baris array tidak sama.\n"
            f"{row_counts}"
        )

    if len(X) == 0:
        raise ValueError(
            f"{split_name}: dataset kosong."
        )

    # -------------------------------------------------------------------------
    # VALIDASI SEQUENCE LENGTH
    # -------------------------------------------------------------------------

    if X.shape[1] != expected_sequence_length:
        raise ValueError(
            f"{split_name}: sequence length tidak sesuai.\n"
            f"Expected : {expected_sequence_length}\n"
            f"Actual   : {X.shape[1]}"
        )

    # -------------------------------------------------------------------------
    # VALIDASI TIPE DATA
    # -------------------------------------------------------------------------

    if not np.issubdtype(
        X.dtype,
        np.integer,
    ):
        raise TypeError(
            f"{split_name}: X harus bertipe integer.\n"
            f"Ditemukan: {X.dtype}"
        )

    if not np.issubdtype(
        y.dtype,
        np.integer,
    ):
        raise TypeError(
            f"{split_name}: y harus bertipe integer.\n"
            f"Ditemukan: {y.dtype}"
        )

    # -------------------------------------------------------------------------
    # VALIDASI TOKEN ID
    # -------------------------------------------------------------------------

    minimum_token_id = int(
        X.min()
    )

    maximum_token_id = int(
        X.max()
    )

    if minimum_token_id < 0:
        raise ValueError(
            f"{split_name}: terdapat token ID negatif.\n"
            f"Token ID minimum: {minimum_token_id}"
        )

    if maximum_token_id >= vocabulary_size:
        raise ValueError(
            f"{split_name}: token ID melebihi "
            "ukuran vocabulary.\n"
            f"Token ID maksimum : {maximum_token_id:,}\n"
            f"Vocabulary size   : {vocabulary_size:,}\n"
            f"Indeks maksimum yang diperbolehkan: "
            f"{vocabulary_size - 1:,}"
        )

    all_padding_rows = int(
        np.all(
            X == 0,
            axis=1,
        ).sum()
    )

    if all_padding_rows > 0:
        raise ValueError(
            f"{split_name}: terdapat "
            f"{all_padding_rows:,} sequence yang "
            "seluruhnya berisi padding."
        )

    # -------------------------------------------------------------------------
    # VALIDASI LABEL
    # -------------------------------------------------------------------------

    unique_labels = np.unique(
        y
    )

    if unique_labels.size == 0:
        raise ValueError(
            f"{split_name}: label tidak ditemukan."
        )

    minimum_label = int(
        unique_labels.min()
    )

    maximum_label = int(
        unique_labels.max()
    )

    if minimum_label < 0:
        raise ValueError(
            f"{split_name}: terdapat label negatif."
        )

    if maximum_label >= NUM_CLASSES:
        raise ValueError(
            f"{split_name}: label melebihi jumlah kelas.\n"
            f"Label maksimum : {maximum_label}\n"
            f"Jumlah kelas   : {NUM_CLASSES}"
        )

    expected_labels = set(
        range(NUM_CLASSES)
    )

    actual_labels = {
        int(label)
        for label in unique_labels
    }

    if actual_labels != expected_labels:
        raise ValueError(
            f"{split_name}: label tidak lengkap.\n"
            f"Seharusnya: {sorted(expected_labels)}\n"
            f"Ditemukan : {sorted(actual_labels)}"
        )

    # -------------------------------------------------------------------------
    # VALIDASI DOCUMENT ID
    # -------------------------------------------------------------------------

    normalized_document_ids = (
        document_ids
        .astype(str)
    )

    empty_document_ids = int(
        np.sum(
            np.char.strip(
                normalized_document_ids
            )
            == ""
        )
    )

    if empty_document_ids > 0:
        raise ValueError(
            f"{split_name}: ditemukan "
            f"{empty_document_ids:,} document_id kosong."
        )

    unique_document_count = len(
        np.unique(
            normalized_document_ids
        )
    )

    duplicate_document_count = (
        len(normalized_document_ids)
        - unique_document_count
    )

    if duplicate_document_count > 0:
        raise ValueError(
            f"{split_name}: terdapat "
            f"{duplicate_document_count:,} document_id duplikat."
        )

    # -------------------------------------------------------------------------
    # VALIDASI CATEGORY
    # -------------------------------------------------------------------------

    normalized_categories = (
        categories
        .astype(str)
    )

    empty_categories = int(
        np.sum(
            np.char.strip(
                normalized_categories
            )
            == ""
        )
    )

    if empty_categories > 0:
        raise ValueError(
            f"{split_name}: ditemukan "
            f"{empty_categories:,} category kosong."
        )


# =============================================================================
# VALIDASI TRAIN DAN VALIDATION TERPISAH
# =============================================================================

def validate_split_disjointness(
    train_document_ids: np.ndarray,
    validation_document_ids: np.ndarray,
) -> None:
    """
    Memastikan tidak ada document_id yang muncul pada train
    dan validation secara bersamaan.
    """

    train_ids = set(
        train_document_ids
        .astype(str)
        .tolist()
    )

    validation_ids = set(
        validation_document_ids
        .astype(str)
        .tolist()
    )

    overlap = train_ids.intersection(
        validation_ids
    )

    if overlap:
        overlap_examples = sorted(
            overlap
        )[:10]

        raise ValueError(
            "Ditemukan document_id yang sama pada train "
            "dan validation.\n"
            f"Jumlah overlap : {len(overlap):,}\n"
            f"Contoh         : {overlap_examples}"
        )


# =============================================================================
# DISTRIBUSI LABEL
# =============================================================================

def get_label_distribution(
    y: np.ndarray,
) -> dict[
    str,
    dict[str, float | int],
]:
    """
    Menghasilkan distribusi label untuk training summary.
    """

    labels, counts = np.unique(
        y,
        return_counts=True,
    )

    distribution: dict[
        str,
        dict[str, float | int],
    ] = {}

    for label, count in zip(
        labels,
        counts,
    ):
        distribution[
            str(
                int(label)
            )
        ] = {
            "count": int(count),
            "percentage": round(
                float(
                    count
                    / len(y)
                    * 100
                ),
                4,
            ),
        }

    return distribution


def print_label_distribution(
    y: np.ndarray,
    split_name: str,
) -> None:
    """
    Menampilkan distribusi label pada terminal.
    """

    distribution = get_label_distribution(
        y
    )

    print(
        f"\nDistribusi label {split_name}:"
    )

    for label in sorted(
        distribution,
        key=int,
    ):
        count = distribution[
            label
        ]["count"]

        percentage = distribution[
            label
        ]["percentage"]

        print(
            f"Label {label} : "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )


# =============================================================================
# MEMBUAT TF.DATA DATASET
# =============================================================================

def create_tf_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    training: bool,
) -> tf.data.Dataset:
    """
    Mengubah array NumPy menjadi tf.data.Dataset.

    Train:
    - shuffle;
    - batch;
    - prefetch.

    Validation:
    - batch;
    - prefetch.
    """

    dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                X,
                y,
            )
        )
    )

    options = tf.data.Options()

    options.experimental_deterministic = True

    dataset = dataset.with_options(
        options
    )

    if training:
        shuffle_buffer = min(
            len(X),
            10_000,
        )

        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer,
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(
        batch_size,
        drop_remainder=False,
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# =============================================================================
# MENYIMPAN TRAINING HISTORY
# =============================================================================

def save_training_history(
    history: tf.keras.callbacks.History,
    output_path: Path,
) -> pd.DataFrame:
    """
    Menyimpan riwayat training dalam format CSV.
    """

    history_dataframe = pd.DataFrame(
        history.history
    )

    if history_dataframe.empty:
        raise ValueError(
            "Training history kosong."
        )

    history_dataframe.insert(
        0,
        "epoch",
        np.arange(
            1,
            len(history_dataframe) + 1,
            dtype=np.int32,
        ),
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    if not output_path.exists():
        raise FileNotFoundError(
            "File training history gagal dibuat:\n"
            f"{output_path}"
        )

    if output_path.stat().st_size <= 0:
        raise ValueError(
            "File training history kosong:\n"
            f"{output_path}"
        )

    return history_dataframe


# =============================================================================
# MENGAMBIL INFORMASI EPOCH TERBAIK
# =============================================================================

def get_best_epoch_information(
    history_dataframe: pd.DataFrame,
) -> dict[str, float | int]:
    """
    Memilih epoch terbaik berdasarkan val_loss terkecil.
    """

    required_columns = {
        "epoch",
        "loss",
        "accuracy",
        "val_loss",
        "val_accuracy",
    }

    missing_columns = (
        required_columns
        - set(
            history_dataframe.columns
        )
    )

    if missing_columns:
        raise KeyError(
            "Training history tidak memiliki seluruh "
            "kolom yang diperlukan.\n"
            f"Kolom tidak ada: {sorted(missing_columns)}"
        )

    validation_losses = pd.to_numeric(
        history_dataframe[
            "val_loss"
        ],
        errors="coerce",
    )

    if validation_losses.isna().all():
        raise ValueError(
            "Seluruh nilai val_loss tidak valid."
        )

    best_index = validation_losses.idxmin()

    best_row = history_dataframe.loc[
        best_index
    ]

    result: dict[str, float | int] = {
        "best_epoch": int(
            best_row["epoch"]
        ),

        "best_train_loss": float(
            best_row["loss"]
        ),

        "best_validation_loss": float(
            best_row["val_loss"]
        ),

        "best_train_accuracy": float(
            best_row["accuracy"]
        ),

        "best_validation_accuracy": float(
            best_row["val_accuracy"]
        ),
    }

    learning_rate_column = None

    if (
        "learning_rate"
        in history_dataframe.columns
    ):
        learning_rate_column = (
            "learning_rate"
        )

    elif "lr" in history_dataframe.columns:
        learning_rate_column = "lr"

    if learning_rate_column is not None:
        learning_rate_value = best_row[
            learning_rate_column
        ]

        if pd.notna(
            learning_rate_value
        ):
            result[
                "learning_rate_at_best_epoch"
            ] = float(
                learning_rate_value
            )

    return result


# =============================================================================
# JSON SAFE CONVERSION
# =============================================================================

def convert_to_json_safe(
    value: Any,
) -> Any:
    """
    Mengubah tipe NumPy menjadi tipe Python agar dapat
    disimpan ke JSON.
    """

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):
        return float(value)

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    return value


# =============================================================================
# MENYIMPAN TRAINING SUMMARY
# =============================================================================

def save_training_summary(
    summary: dict,
    output_path: Path,
) -> None:
    """
    Menyimpan training summary dalam JSON secara atomik.

    File sementara ditulis terlebih dahulu agar summary tidak
    rusak apabila proses penulisan terhenti.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        f"{output_path.stem}.tmp"
        f"{output_path.suffix}"
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=4,
            default=convert_to_json_safe,
        )

    temporary_path.replace(
        output_path
    )


# =============================================================================
# MEMBERSIHKAN ARTEFAK EKSPERIMEN LAMA
# =============================================================================

def remove_previous_artifacts(
    paths: list[Path],
) -> None:
    """
    Menghapus artefak lama agar hasil sebelumnya tidak dianggap
    sebagai hasil eksperimen yang sedang dijalankan.
    """

    for artifact_path in paths:
        if artifact_path.exists():
            if artifact_path.is_file():
                artifact_path.unlink()

                print(
                    "Artefak lama dihapus:"
                )

                print(
                    artifact_path
                )

            else:
                raise ValueError(
                    "Path artefak seharusnya berupa file:\n"
                    f"{artifact_path}"
                )


# =============================================================================
# VALIDASI FILE OUTPUT
# =============================================================================

def validate_output_file(
    file_path: Path,
    description: str,
) -> None:
    """
    Memastikan file output tersedia dan tidak kosong.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"{description} tidak terbentuk:\n"
            f"{file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"{description} bukan file:\n"
            f"{file_path}"
        )

    if file_path.stat().st_size <= 0:
        raise ValueError(
            f"{description} kosong:\n"
            f"{file_path}"
        )


# =============================================================================
# INFORMASI PERANGKAT
# =============================================================================

def get_device_information() -> dict:
    """
    Mengambil informasi GPU dan CPU yang tersedia.
    """

    gpu_devices = tf.config.list_physical_devices(
        "GPU"
    )

    cpu_devices = tf.config.list_physical_devices(
        "CPU"
    )

    return {
        "tensorflow_version": tf.__version__,
        "gpu_available": bool(
            gpu_devices
        ),
        "gpu_count": len(
            gpu_devices
        ),
        "gpu_devices": [
            device.name
            for device in gpu_devices
        ],
        "cpu_count_detected": len(
            cpu_devices
        ),
    }


# =============================================================================
# TRAINING CNN
# =============================================================================

def train_cnn(
    scenario_code: str,
) -> None:
    """
    Menjalankan training CNN untuk satu skenario.
    """

    # -------------------------------------------------------------------------
    # 1. VALIDASI SKENARIO
    # -------------------------------------------------------------------------

    scenario_code = (
        str(
            scenario_code
        )
        .strip()
        .upper()
    )

    if scenario_code not in VALID_SCENARIOS:
        raise ValueError(
            f"Skenario tidak valid: {scenario_code}\n"
            f"Skenario valid: {VALID_SCENARIOS}"
        )

    scenario_code = validate_scenario_code(
        scenario_code
    )

    if scenario_code not in VALID_SCENARIOS:
        raise ValueError(
            "training_config mengembalikan skenario "
            "yang tidak termasuk eksperimen final:\n"
            f"{scenario_code}"
        )

    # -------------------------------------------------------------------------
    # 2. REPRODUCIBILITY DAN DIRECTORY
    # -------------------------------------------------------------------------

    set_global_seed()

    create_training_directories()

    TRAINING_SUMMARY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tf.keras.backend.clear_session()

    gc.collect()

    # -------------------------------------------------------------------------
    # 3. INFORMASI SKENARIO
    # -------------------------------------------------------------------------

    scenario_config = get_scenario_config(
        scenario_code
    )

    vocabulary_size = int(
        get_vocabulary_size(
            scenario_code
        )
    )

    max_sequence_length = int(
        scenario_config[
            "max_sequence_length"
        ]
    )

    experiment_name = get_experiment_name(
        model_name=MODEL_CODE,
        scenario_code=scenario_code,
    )

    # -------------------------------------------------------------------------
    # 4. PATH DATA
    # -------------------------------------------------------------------------

    train_path = get_split_path(
        scenario_code,
        "train",
    )

    validation_path = get_split_path(
        scenario_code,
        "validation",
    )

    # -------------------------------------------------------------------------
    # 5. PATH OUTPUT
    # -------------------------------------------------------------------------

    checkpoint_path = get_checkpoint_path(
        model_name=MODEL_CODE,
        scenario_code=scenario_code,
    )

    final_model_path = get_final_model_path(
        model_name=MODEL_CODE,
        scenario_code=scenario_code,
    )

    history_path = get_history_path(
        model_name=MODEL_CODE,
        scenario_code=scenario_code,
    )

    summary_path = (
        TRAINING_SUMMARY_DIR
        / (
            f"{experiment_name}_"
            f"summary.json"
        )
    )

    # -------------------------------------------------------------------------
    # 6. MEMBERSIHKAN ARTEFAK LAMA
    # -------------------------------------------------------------------------

    remove_previous_artifacts(
        [
            checkpoint_path,
            final_model_path,
            history_path,
            summary_path,
        ]
    )

    # -------------------------------------------------------------------------
    # 7. MENYIMPAN STATUS RUNNING
    # -------------------------------------------------------------------------

    started_at = datetime.now()

    running_summary = {
        "status": "running",
        "experiment_name": experiment_name,
        "model": MODEL_DISPLAY_NAME,
        "model_code": MODEL_CODE,
        "dataset": scenario_config[
            "dataset"
        ],
        "scenario_code": scenario_code,
        "scenario_name": scenario_config[
            "scenario_name"
        ],
        "started_at": started_at.isoformat(
            timespec="seconds"
        ),
        "random_seed": RANDOM_SEED,
        "tensorflow_version": tf.__version__,
    }

    save_training_summary(
        summary=running_summary,
        output_path=summary_path,
    )

    # -------------------------------------------------------------------------
    # 8. INFORMASI TERMINAL
    # -------------------------------------------------------------------------

    print("=" * 72)
    print("STEP 5.4 - TRAIN CNN")
    print("=" * 72)

    print(
        f"\nExperiment         : "
        f"{experiment_name}"
    )

    print(
        f"Dataset            : "
        f"{scenario_config['dataset']}"
    )

    print(
        f"Scenario           : "
        f"{scenario_code}"
    )

    print(
        f"Scenario name      : "
        f"{scenario_config['scenario_name']}"
    )

    print(
        f"Vocabulary size    : "
        f"{vocabulary_size:,}"
    )

    print(
        f"Sequence length    : "
        f"{max_sequence_length}"
    )

    print(
        f"Epoch maksimum     : "
        f"{EPOCHS}"
    )

    print(
        f"Batch size         : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Learning rate      : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Embedding dimension: "
        f"{EMBEDDING_DIM}"
    )

    print(
        f"Jumlah filter CNN  : "
        f"{CNN_NUM_FILTERS}"
    )

    print(
        f"Kernel size        : "
        f"{CNN_KERNEL_SIZE}"
    )

    print(
        f"Dense units        : "
        f"{DENSE_UNITS}"
    )

    gpu_devices = tf.config.list_physical_devices(
        "GPU"
    )

    print(
        f"GPU terdeteksi     : "
        f"{len(gpu_devices)}"
    )

    if not gpu_devices:
        print(
            "Training menggunakan CPU."
        )

    # -------------------------------------------------------------------------
    # 9. LOAD DATA TRAIN
    # -------------------------------------------------------------------------

    print(
        "\nMemuat data train..."
    )

    (
        X_train,
        y_train,
        train_document_ids,
        train_categories,
    ) = load_npz_dataset(
        train_path
    )

    # -------------------------------------------------------------------------
    # 10. LOAD DATA VALIDATION
    # -------------------------------------------------------------------------

    print(
        "Memuat data validation..."
    )

    (
        X_validation,
        y_validation,
        validation_document_ids,
        validation_categories,
    ) = load_npz_dataset(
        validation_path
    )

    # -------------------------------------------------------------------------
    # 11. VALIDASI DATA TRAIN
    # -------------------------------------------------------------------------

    validate_dataset(
        X=X_train,
        y=y_train,
        document_ids=train_document_ids,
        categories=train_categories,
        split_name="Train",
        expected_sequence_length=(
            max_sequence_length
        ),
        vocabulary_size=vocabulary_size,
    )

    # -------------------------------------------------------------------------
    # 12. VALIDASI DATA VALIDATION
    # -------------------------------------------------------------------------

    validate_dataset(
        X=X_validation,
        y=y_validation,
        document_ids=(
            validation_document_ids
        ),
        categories=(
            validation_categories
        ),
        split_name="Validation",
        expected_sequence_length=(
            max_sequence_length
        ),
        vocabulary_size=vocabulary_size,
    )

    # -------------------------------------------------------------------------
    # 13. VALIDASI TRAIN DAN VALIDATION DISJOINT
    # -------------------------------------------------------------------------

    validate_split_disjointness(
        train_document_ids=(
            train_document_ids
        ),
        validation_document_ids=(
            validation_document_ids
        ),
    )

    # -------------------------------------------------------------------------
    # 14. INFORMASI DATA
    # -------------------------------------------------------------------------

    print(
        "\nShape data:"
    )

    print(
        f"X_train            : "
        f"{X_train.shape}"
    )

    print(
        f"y_train            : "
        f"{y_train.shape}"
    )

    print(
        f"X_validation       : "
        f"{X_validation.shape}"
    )

    print(
        f"y_validation       : "
        f"{y_validation.shape}"
    )

    print(
        f"Token maksimum train      : "
        f"{int(X_train.max()):,}"
    )

    print(
        f"Token maksimum validation : "
        f"{int(X_validation.max()):,}"
    )

    print(
        "\nValidasi overlap train-validation:"
    )

    print(
        "Tidak ditemukan document_id yang sama."
    )

    print_label_distribution(
        y_train,
        "Train",
    )

    print_label_distribution(
        y_validation,
        "Validation",
    )

    # -------------------------------------------------------------------------
    # 15. MEMBUAT TF.DATA
    # -------------------------------------------------------------------------

    train_dataset = create_tf_dataset(
        X=X_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        training=True,
    )

    validation_dataset = create_tf_dataset(
        X=X_validation,
        y=y_validation,
        batch_size=BATCH_SIZE,
        training=False,
    )

    # -------------------------------------------------------------------------
    # 16. MEMBANGUN MODEL
    # -------------------------------------------------------------------------

    print(
        "\nMembangun model CNN..."
    )

    model = build_cnn_model(
        vocabulary_size=vocabulary_size,
        max_sequence_length=(
            max_sequence_length
        ),
        num_classes=NUM_CLASSES,
        embedding_dim=EMBEDDING_DIM,
        num_filters=CNN_NUM_FILTERS,
        kernel_size=CNN_KERNEL_SIZE,
        dense_units=DENSE_UNITS,
        spatial_dropout_rate=(
            SPATIAL_DROPOUT_RATE
        ),
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE,
        model_name=(
            f"CNN_{scenario_code}"
        ),
    )

    print(
        "\nModel Summary:"
    )

    model.summary()

    # -------------------------------------------------------------------------
    # 17. CALLBACK
    # -------------------------------------------------------------------------

    callbacks = create_training_callbacks(
        model_name=MODEL_CODE,
        scenario_code=scenario_code,
    )

    if not callbacks:
        raise ValueError(
            "Daftar callback training kosong."
        )

    # -------------------------------------------------------------------------
    # 18. TRAINING
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "MEMULAI TRAINING"
    )

    print(
        "=" * 72
    )

    training_start_time = (
        time.perf_counter()
    )

    history = model.fit(
        train_dataset,
        validation_data=(
            validation_dataset
        ),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=VERBOSE,
    )

    training_end_time = (
        time.perf_counter()
    )

    training_time_seconds = (
        training_end_time
        - training_start_time
    )

    # -------------------------------------------------------------------------
    # 19. SIMPAN HISTORY
    # -------------------------------------------------------------------------

    history_dataframe = save_training_history(
        history=history,
        output_path=history_path,
    )

    best_information = (
        get_best_epoch_information(
            history_dataframe
        )
    )

    # -------------------------------------------------------------------------
    # 20. VALIDASI CHECKPOINT TERBAIK
    # -------------------------------------------------------------------------

    validate_output_file(
        file_path=checkpoint_path,
        description="Checkpoint terbaik",
    )

    # -------------------------------------------------------------------------
    # 21. MEMUAT CHECKPOINT TERBAIK
    # -------------------------------------------------------------------------

    print(
        "\nMemuat checkpoint terbaik..."
    )

    best_model = tf.keras.models.load_model(
        checkpoint_path
    )

    # -------------------------------------------------------------------------
    # 22. EVALUASI CHECKPOINT PADA VALIDATION
    # -------------------------------------------------------------------------

    print(
        "Memvalidasi checkpoint terbaik "
        "pada data validation..."
    )

    best_validation_result = (
        best_model.evaluate(
            validation_dataset,
            verbose=0,
            return_dict=True,
        )
    )

    if "loss" not in best_validation_result:
        raise KeyError(
            "Hasil evaluasi checkpoint tidak "
            "memiliki key loss."
        )

    if "accuracy" not in best_validation_result:
        raise KeyError(
            "Hasil evaluasi checkpoint tidak "
            "memiliki key accuracy."
        )

    best_checkpoint_validation_loss = float(
        best_validation_result[
            "loss"
        ]
    )

    best_checkpoint_validation_accuracy = float(
        best_validation_result[
            "accuracy"
        ]
    )

    if not np.isfinite(
        best_checkpoint_validation_loss
    ):
        raise ValueError(
            "Validation loss checkpoint tidak valid."
        )

    if not np.isfinite(
        best_checkpoint_validation_accuracy
    ):
        raise ValueError(
            "Validation accuracy checkpoint tidak valid."
        )

    # -------------------------------------------------------------------------
    # 23. SIMPAN MODEL FINAL DARI CHECKPOINT TERBAIK
    # -------------------------------------------------------------------------

    final_model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model.save(
        final_model_path
    )

    validate_output_file(
        file_path=final_model_path,
        description="Model final",
    )

    # -------------------------------------------------------------------------
    # 24. TRAINING SUMMARY
    # -------------------------------------------------------------------------

    finished_at = datetime.now()

    training_summary = {
        "status":
            "success",

        "experiment_name":
            experiment_name,

        "model":
            MODEL_DISPLAY_NAME,

        "model_code":
            MODEL_CODE,

        "dataset":
            scenario_config[
                "dataset"
            ],

        "scenario_code":
            scenario_code,

        "scenario_name":
            scenario_config[
                "scenario_name"
            ],

        "started_at":
            started_at.isoformat(
                timespec="seconds"
            ),

        "finished_at":
            finished_at.isoformat(
                timespec="seconds"
            ),

        "jumlah_train":
            int(
                len(X_train)
            ),

        "jumlah_validation":
            int(
                len(X_validation)
            ),

        "train_label_distribution":
            get_label_distribution(
                y_train
            ),

        "validation_label_distribution":
            get_label_distribution(
                y_validation
            ),

        "train_validation_document_overlap":
            0,

        "vocabulary_size":
            int(
                vocabulary_size
            ),

        "max_token_id_train":
            int(
                X_train.max()
            ),

        "max_token_id_validation":
            int(
                X_validation.max()
            ),

        "max_sequence_length":
            int(
                max_sequence_length
            ),

        "num_classes":
            int(
                NUM_CLASSES
            ),

        "embedding_dim":
            int(
                EMBEDDING_DIM
            ),

        "num_filters":
            int(
                CNN_NUM_FILTERS
            ),

        "kernel_size":
            int(
                CNN_KERNEL_SIZE
            ),

        "dense_units":
            int(
                DENSE_UNITS
            ),

        "spatial_dropout_rate":
            float(
                SPATIAL_DROPOUT_RATE
            ),

        "dropout_rate":
            float(
                DROPOUT_RATE
            ),

        "epochs_maximum":
            int(
                EPOCHS
            ),

        "epochs_completed":
            int(
                len(
                    history_dataframe
                )
            ),

        "batch_size":
            int(
                BATCH_SIZE
            ),

        "learning_rate_initial":
            float(
                LEARNING_RATE
            ),

        "random_seed":
            int(
                RANDOM_SEED
            ),

        "monitor":
            "val_loss",

        "selection_mode":
            "min",

        "training_time_seconds":
            round(
                training_time_seconds,
                4,
            ),

        "training_time_minutes":
            round(
                training_time_seconds
                / 60,
                4,
            ),

        **best_information,

        "best_checkpoint_validation_loss":
            best_checkpoint_validation_loss,

        "best_checkpoint_validation_accuracy":
            best_checkpoint_validation_accuracy,

        "final_model_source":
            "best_validation_loss_checkpoint",

        "tensorflow_environment":
            get_device_information(),

        "checkpoint_path":
            str(
                checkpoint_path
            ),

        "final_model_path":
            str(
                final_model_path
            ),

        "history_path":
            str(
                history_path
            ),

        "summary_path":
            str(
                summary_path
            ),

        "test_set_used_during_training":
            False,

        "k4_used":
            False,
    }

    save_training_summary(
        summary=training_summary,
        output_path=summary_path,
    )

    validate_output_file(
        file_path=summary_path,
        description="Training summary",
    )

    # -------------------------------------------------------------------------
    # 25. MENAMPILKAN HASIL
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "HASIL TRAINING CNN"
    )

    print(
        "=" * 72
    )

    print(
        f"\nExperiment          : "
        f"{experiment_name}"
    )

    print(
        f"Epoch dijalankan    : "
        f"{len(history_dataframe)}"
    )

    print(
        f"Best epoch          : "
        f"{best_information['best_epoch']}"
    )

    print(
        f"Best train loss     : "
        f"{best_information['best_train_loss']:.6f}"
    )

    print(
        f"Best validation loss: "
        f"{best_information['best_validation_loss']:.6f}"
    )

    print(
        f"Best train accuracy : "
        f"{best_information['best_train_accuracy']:.4f}"
    )

    print(
        f"Best val accuracy   : "
        f"{best_information['best_validation_accuracy']:.4f}"
    )

    print(
        "\nValidasi checkpoint terbaik:"
    )

    print(
        f"Validation loss     : "
        f"{best_checkpoint_validation_loss:.6f}"
    )

    print(
        f"Validation accuracy : "
        f"{best_checkpoint_validation_accuracy:.4f}"
    )

    print(
        f"\nWaktu training      : "
        f"{training_time_seconds / 60:.2f} menit"
    )

    print(
        "\nCheckpoint terbaik:"
    )

    print(
        checkpoint_path
    )

    print(
        "\nModel final dari checkpoint terbaik:"
    )

    print(
        final_model_path
    )

    print(
        "\nTraining history:"
    )

    print(
        history_path
    )

    print(
        "\nTraining summary:"
    )

    print(
        summary_path
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "Training CNN selesai."
    )

    print(
        "=" * 72
    )

    # -------------------------------------------------------------------------
    # 26. MEMBERSIHKAN MEMORI
    # -------------------------------------------------------------------------

    del model
    del best_model
    del train_dataset
    del validation_dataset
    del history

    del X_train
    del y_train
    del train_document_ids
    del train_categories

    del X_validation
    del y_validation
    del validation_document_ids
    del validation_categories

    tf.keras.backend.clear_session()

    gc.collect()


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Membaca kode skenario dari command line.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Melatih model CNN untuk satu skenario."
        )
    )

    parser.add_argument(
        "scenario",
        type=str.upper,
        choices=VALID_SCENARIOS,
        help=(
            "Kode skenario: K1, K2, K3, A1, atau A2."
        ),
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Menjalankan training berdasarkan argumen command line.
    """

    arguments = parse_arguments()

    train_cnn(
        scenario_code=arguments.scenario
    )


if __name__ == "__main__":
    main()
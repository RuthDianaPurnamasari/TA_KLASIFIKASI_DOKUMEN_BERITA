# =============================================================================
# STEP 5.5 - TRAIN ATTENTION-BILSTM
# =============================================================================
# File:
# 5_modeling/train_attention_bilstm.py
#
# Tujuan:
# Melatih model Attention-BiLSTM pada satu skenario representasi teks.
#
# Contoh menjalankan:
# python 5_modeling/train_attention_bilstm.py K1
#
# Alur:
# 1. Membaca konfigurasi skenario
# 2. Memuat train dan validation dari file NPZ
# 3. Memvalidasi bentuk data
# 4. Membuat model Attention-BiLSTM
# 5. Melatih model
# 6. Menyimpan checkpoint terbaik
# 7. Menyimpan model final
# 8. Menyimpan training history
# 9. Menyimpan ringkasan hasil training
#
# Test set tidak digunakan pada tahap training.
# =============================================================================

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

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

from attention_bilstm_model import (
    build_attention_bilstm_model,
)

from training_config import (
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
    BILSTM_UNITS,
    ATTENTION_UNITS,
    RECURRENT_DROPOUT_RATE,
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Memuat X dan y dari file NPZ.

    File NPZ harus memiliki:
    - X: sequence integer
    - y: label integer
    """

    if not file_path.exists():
        raise FileNotFoundError(
            "File dataset tidak ditemukan:\n"
            f"{file_path}"
        )

    with np.load(
        file_path,
        allow_pickle=False,
    ) as data:

        available_keys = list(
            data.files
        )

        if (
            "X" not in available_keys
            or "y" not in available_keys
        ):
            raise KeyError(
                "File NPZ harus memiliki key "
                "'X' dan 'y'.\n"
                f"Key ditemukan: {available_keys}"
            )

        X = data["X"]
        y = data["y"]

    return X, y


# =============================================================================
# VALIDASI DATASET
# =============================================================================

def validate_dataset(
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
    expected_sequence_length: int,
) -> None:
    """
    Memastikan bentuk data sesuai kebutuhan model.
    """

    if X.ndim != 2:
        raise ValueError(
            f"{split_name}: X harus dua dimensi. "
            f"Shape ditemukan: {X.shape}"
        )

    if y.ndim != 1:
        raise ValueError(
            f"{split_name}: y harus satu dimensi. "
            f"Shape ditemukan: {y.shape}"
        )

    if len(X) != len(y):
        raise ValueError(
            f"{split_name}: jumlah X dan y tidak sama."
        )

    if X.shape[1] != expected_sequence_length:
        raise ValueError(
            f"{split_name}: sequence length tidak sesuai.\n"
            f"Expected : {expected_sequence_length}\n"
            f"Actual   : {X.shape[1]}"
        )

    if len(X) == 0:
        raise ValueError(
            f"{split_name}: dataset kosong."
        )

    unique_labels = np.unique(
        y
    )

    if np.min(unique_labels) < 0:
        raise ValueError(
            f"{split_name}: ditemukan label negatif."
        )

    if np.max(unique_labels) >= NUM_CLASSES:
        raise ValueError(
            f"{split_name}: label melebihi jumlah kelas."
        )


# =============================================================================
# DISTRIBUSI LABEL
# =============================================================================

def print_label_distribution(
    y: np.ndarray,
    split_name: str,
) -> None:
    """
    Menampilkan distribusi label pada terminal.
    """

    labels, counts = np.unique(
        y,
        return_counts=True,
    )

    print(
        f"\nDistribusi label {split_name}:"
    )

    for label, count in zip(
        labels,
        counts,
    ):
        percentage = (
            count
            / len(y)
            * 100
        )

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
    Mengubah NumPy array menjadi tf.data.Dataset.

    Train:
    - shuffle
    - batch
    - prefetch

    Validation:
    - batch
    - prefetch
    """

    dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (X, y)
        )
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
    Menyimpan riwayat training dalam bentuk CSV.
    """

    history_dataframe = pd.DataFrame(
        history.history
    )

    history_dataframe.insert(
        0,
        "epoch",
        np.arange(
            1,
            len(history_dataframe) + 1,
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

    return history_dataframe


# =============================================================================
# MENGAMBIL INFORMASI EPOCH TERBAIK
# =============================================================================

def get_best_epoch_information(
    history_dataframe: pd.DataFrame,
) -> dict:
    """
    Memilih epoch terbaik berdasarkan val_loss terkecil.
    """

    if (
        "val_loss"
        not in history_dataframe.columns
    ):
        raise KeyError(
            "Kolom val_loss tidak ditemukan "
            "pada training history."
        )

    best_index = (
        history_dataframe[
            "val_loss"
        ]
        .idxmin()
    )

    best_row = (
        history_dataframe
        .loc[best_index]
    )

    result = {
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

    if (
        "learning_rate"
        in history_dataframe.columns
    ):
        result[
            "learning_rate_at_best_epoch"
        ] = float(
            best_row["learning_rate"]
        )

    return result


# =============================================================================
# MENYIMPAN TRAINING SUMMARY
# =============================================================================

def save_training_summary(
    summary: dict,
    output_path: Path,
) -> None:
    """
    Menyimpan ringkasan training dalam JSON.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
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
# TRAINING ATTENTION-BILSTM
# =============================================================================

def train_attention_bilstm(
    scenario_code: str,
) -> None:
    """
    Menjalankan training Attention-BiLSTM
    untuk satu skenario.
    """

    # -------------------------------------------------------------------------
    # 1. PERSIAPAN
    # -------------------------------------------------------------------------

    scenario_code = (
        validate_scenario_code(
            scenario_code
        )
    )

    set_global_seed()

    create_training_directories()

    TRAINING_SUMMARY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tf.keras.backend.clear_session()

    gc.collect()

    # -------------------------------------------------------------------------
    # 2. INFORMASI SKENARIO
    # -------------------------------------------------------------------------

    scenario_config = (
        get_scenario_config(
            scenario_code
        )
    )

    vocabulary_size = (
        get_vocabulary_size(
            scenario_code
        )
    )

    max_sequence_length = (
        scenario_config[
            "max_sequence_length"
        ]
    )

    experiment_name = (
        get_experiment_name(
            model_name=(
                "attention_bilstm"
            ),
            scenario_code=(
                scenario_code
            ),
        )
    )

    # -------------------------------------------------------------------------
    # 3. PATH DATA DAN OUTPUT
    # -------------------------------------------------------------------------

    train_path = (
        get_split_path(
            scenario_code,
            "train",
        )
    )

    validation_path = (
        get_split_path(
            scenario_code,
            "validation",
        )
    )

    checkpoint_path = (
        get_checkpoint_path(
            model_name=(
                "attention_bilstm"
            ),
            scenario_code=(
                scenario_code
            ),
        )
    )

    final_model_path = (
        get_final_model_path(
            model_name=(
                "attention_bilstm"
            ),
            scenario_code=(
                scenario_code
            ),
        )
    )

    history_path = (
        get_history_path(
            model_name=(
                "attention_bilstm"
            ),
            scenario_code=(
                scenario_code
            ),
        )
    )

    summary_path = (
        TRAINING_SUMMARY_DIR
        / (
            f"{experiment_name}_"
            f"summary.json"
        )
    )

    # -------------------------------------------------------------------------
    # 4. INFORMASI TERMINAL
    # -------------------------------------------------------------------------

    print("=" * 72)
    print(
        "STEP 5.5 - TRAIN ATTENTION-BILSTM"
    )
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
        f"LSTM units/arah    : "
        f"{BILSTM_UNITS}"
    )

    print(
        f"Attention units    : "
        f"{ATTENTION_UNITS}"
    )

    # -------------------------------------------------------------------------
    # 5. LOAD DATA
    # -------------------------------------------------------------------------

    print(
        "\nMemuat data train..."
    )

    X_train, y_train = (
        load_npz_dataset(
            train_path
        )
    )

    print(
        "Memuat data validation..."
    )

    X_validation, y_validation = (
        load_npz_dataset(
            validation_path
        )
    )

    # -------------------------------------------------------------------------
    # 6. VALIDASI DATA
    # -------------------------------------------------------------------------

    validate_dataset(
        X=X_train,
        y=y_train,
        split_name="Train",
        expected_sequence_length=(
            max_sequence_length
        ),
    )

    validate_dataset(
        X=X_validation,
        y=y_validation,
        split_name="Validation",
        expected_sequence_length=(
            max_sequence_length
        ),
    )

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

    print_label_distribution(
        y_train,
        "Train",
    )

    print_label_distribution(
        y_validation,
        "Validation",
    )

    # -------------------------------------------------------------------------
    # 7. MEMBUAT TF.DATA
    # -------------------------------------------------------------------------

    train_dataset = (
        create_tf_dataset(
            X=X_train,
            y=y_train,
            batch_size=BATCH_SIZE,
            training=True,
        )
    )

    validation_dataset = (
        create_tf_dataset(
            X=X_validation,
            y=y_validation,
            batch_size=BATCH_SIZE,
            training=False,
        )
    )

    # -------------------------------------------------------------------------
    # 8. MEMBANGUN MODEL
    # -------------------------------------------------------------------------

    print(
        "\nMembangun model "
        "Attention-BiLSTM..."
    )

    model = (
        build_attention_bilstm_model(
            vocabulary_size=(
                vocabulary_size
            ),
            max_sequence_length=(
                max_sequence_length
            ),
            num_classes=NUM_CLASSES,
            embedding_dim=EMBEDDING_DIM,
            lstm_units=BILSTM_UNITS,
            attention_units=(
                ATTENTION_UNITS
            ),
            dense_units=DENSE_UNITS,
            spatial_dropout_rate=(
                SPATIAL_DROPOUT_RATE
            ),
            recurrent_dropout_rate=(
                RECURRENT_DROPOUT_RATE
            ),
            dropout_rate=(
                DROPOUT_RATE
            ),
            learning_rate=(
                LEARNING_RATE
            ),
            model_name=(
                f"Attention_BiLSTM_"
                f"{scenario_code}"
            ),
        )
    )

    print(
        "\nModel Summary:"
    )

    model.summary()

    # -------------------------------------------------------------------------
    # 9. CALLBACK
    # -------------------------------------------------------------------------

    callbacks = (
        create_training_callbacks(
            model_name=(
                "attention_bilstm"
            ),
            scenario_code=(
                scenario_code
            ),
        )
    )

    # -------------------------------------------------------------------------
    # 10. TRAINING
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

    start_time = (
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

    end_time = (
        time.perf_counter()
    )

    training_time_seconds = (
        end_time
        - start_time
    )

    # -------------------------------------------------------------------------
    # 11. SIMPAN HISTORY
    # -------------------------------------------------------------------------

    history_dataframe = (
        save_training_history(
            history=history,
            output_path=history_path,
        )
    )

    best_information = (
        get_best_epoch_information(
            history_dataframe
        )
    )

    # -------------------------------------------------------------------------
    # 12. SIMPAN MODEL FINAL
    # -------------------------------------------------------------------------

    final_model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(
        final_model_path
    )

    # -------------------------------------------------------------------------
    # 13. SIMPAN RINGKASAN TRAINING
    # -------------------------------------------------------------------------

    training_summary = {
        "experiment_name":
            experiment_name,

        "model":
            "Attention-BiLSTM",

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

        "jumlah_train":
            int(
                len(X_train)
            ),

        "jumlah_validation":
            int(
                len(X_validation)
            ),

        "vocabulary_size":
            int(
                vocabulary_size
            ),

        "max_sequence_length":
            int(
                max_sequence_length
            ),

        "num_classes":
            NUM_CLASSES,

        "embedding_dim":
            EMBEDDING_DIM,

        "lstm_units_per_direction":
            BILSTM_UNITS,

        "attention_units":
            ATTENTION_UNITS,

        "epochs_maximum":
            EPOCHS,

        "epochs_completed":
            int(
                len(history_dataframe)
            ),

        "batch_size":
            BATCH_SIZE,

        "learning_rate_initial":
            LEARNING_RATE,

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
    }

    save_training_summary(
        summary=training_summary,
        output_path=summary_path,
    )

    # -------------------------------------------------------------------------
    # 14. TAMPILKAN HASIL
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "HASIL TRAINING ATTENTION-BILSTM"
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
        f"Waktu training      : "
        f"{training_time_seconds / 60:.2f} menit"
    )

    print(
        "\nCheckpoint terbaik:"
    )

    print(
        checkpoint_path
    )

    print(
        "\nModel final:"
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
        "Training Attention-BiLSTM selesai."
    )

    print(
        "=" * 72
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        selected_scenario = (
            sys.argv[1]
        )

    else:
        selected_scenario = "K1"

        print(
            "\nTidak ada kode skenario "
            "yang diberikan."
        )

        print(
            "Menggunakan K1 sebagai "
            "default pipeline validation.\n"
        )

    train_attention_bilstm(
        scenario_code=(
            selected_scenario
        )
    )
# =============================================================================
# STEP 5.3 - SHARED TRAINING CONFIGURATION
# =============================================================================
# File:
# 5_modeling/training_config.py
#
# Tujuan:
# Menyimpan konfigurasi pelatihan bersama untuk CNN dan
# Attention-BiLSTM agar eksperimen konsisten dan reproducible.
# =============================================================================

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras


# =============================================================================
# ROOT PROJECT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# FOLDER INPUT
# =============================================================================

VECTORIZED_DIR = (
    PROJECT_ROOT
    / "2_data"
    / "vectorized"
)


# =============================================================================
# FOLDER OUTPUT MODELING
# =============================================================================

SAVED_MODELS_DIR = (
    PROJECT_ROOT
    / "8_save_models"
)

CHECKPOINTS_DIR = (
    SAVED_MODELS_DIR
    / "checkpoints"
)

FINAL_MODELS_DIR = (
    SAVED_MODELS_DIR
    / "final_models"
)

TRAINING_HISTORY_DIR = (
    PROJECT_ROOT
    / "9_results"
    / "training_history"
)

TRAINING_LOGS_DIR = (
    PROJECT_ROOT
    / "9_results"
    / "logs"
)

TRAINING_CONFIG_OUTPUT_PATH = (
    PROJECT_ROOT
    / "9_results"
    / "tables"
    / "training_configuration.json"
)


# =============================================================================
# KONFIGURASI UMUM
# =============================================================================

RANDOM_SEED = 42

NUM_CLASSES = 4

EPOCHS = 20

BATCH_SIZE = 64

LEARNING_RATE = 0.001

VERBOSE = 1


# =============================================================================
# KONFIGURASI CALLBACK
# =============================================================================

EARLY_STOPPING_MONITOR = "val_loss"

EARLY_STOPPING_PATIENCE = 3

EARLY_STOPPING_MIN_DELTA = 0.0001

REDUCE_LR_MONITOR = "val_loss"

REDUCE_LR_PATIENCE = 2

REDUCE_LR_FACTOR = 0.5

MIN_LEARNING_RATE = 0.000001

CHECKPOINT_MONITOR = "val_loss"


# =============================================================================
# KONFIGURASI ARSITEKTUR BERSAMA
# =============================================================================

EMBEDDING_DIM = 128

DENSE_UNITS = 128

SPATIAL_DROPOUT_RATE = 0.2

DROPOUT_RATE = 0.5


# =============================================================================
# KONFIGURASI CNN
# =============================================================================

CNN_NUM_FILTERS = 128

CNN_KERNEL_SIZE = 5


# =============================================================================
# KONFIGURASI ATTENTION-BILSTM
# =============================================================================

BILSTM_UNITS = 64

ATTENTION_UNITS = 64

RECURRENT_DROPOUT_RATE = 0.0


# =============================================================================
# INFORMASI SETIAP SKENARIO
# =============================================================================
#
# Folder harus sama dengan hasil:
# 4_preprocessing/07_text_vectorization.py
#
# Masing-masing folder berisi:
# train.npz
# validation.npz
# test.npz
# vocabulary.txt
# vectorizer_config.json
# =============================================================================

SCENARIO_CONFIGS: dict[str, dict[str, Any]] = {
    "K1": {
        "dataset": "Kompas",
        "scenario_name": "Title",
        "vectorized_directory": (
            VECTORIZED_DIR
            / "kompas_k1"
        ),
        "max_sequence_length": 20,
        "expected_vocabulary_size": 10_234,
    },

    "K2": {
        "dataset": "Kompas",
        "scenario_name": (
            "Title + Description"
        ),
        "vectorized_directory": (
            VECTORIZED_DIR
            / "kompas_k2"
        ),
        "max_sequence_length": 60,
        "expected_vocabulary_size": 15_446,
    },

    "K3": {
        "dataset": "Kompas",
        "scenario_name": (
            "Title + Description + Keyword"
        ),
        "vectorized_directory": (
            VECTORIZED_DIR
            / "kompas_k3"
        ),
        "max_sequence_length": 60,
        "expected_vocabulary_size": 15_447,
    },

    "K4": {
        "dataset": "Kompas",
        "scenario_name": (
            "Title + Description + "
            "Keyword + Content"
        ),
        "vectorized_directory": (
            VECTORIZED_DIR
            / "kompas_k4"
        ),
        "max_sequence_length": 525,
        "expected_vocabulary_size": 49_333,
    },

    "A1": {
        "dataset": "AG News",
        "scenario_name": "Title",
        "vectorized_directory": (
            VECTORIZED_DIR
            / "agnews_a1"
        ),
        "max_sequence_length": 20,
        "expected_vocabulary_size": 33_320,
    },

    "A2": {
        "dataset": "AG News",
        "scenario_name": (
            "Title + Description"
        ),
        "vectorized_directory": (
            VECTORIZED_DIR
            / "agnews_a2"
        ),
        "max_sequence_length": 60,
        "expected_vocabulary_size": 64_868,
    },
}


# =============================================================================
# RANDOM SEED
# =============================================================================

def set_global_seed(
    seed: int = RANDOM_SEED,
) -> None:
    """
    Mengatur random seed agar hasil eksperimen lebih konsisten.

    Random seed diterapkan pada:
    - Python
    - NumPy
    - TensorFlow
    """

    os.environ[
        "PYTHONHASHSEED"
    ] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()

    except (
        AttributeError,
        RuntimeError,
    ):
        # Beberapa perangkat atau versi TensorFlow
        # mungkin tidak mendukung determinism penuh.
        pass


# =============================================================================
# MEMBUAT FOLDER OUTPUT
# =============================================================================

def create_training_directories() -> None:
    """
    Membuat semua folder output yang dibutuhkan.
    """

    directories = [
        SAVED_MODELS_DIR,
        CHECKPOINTS_DIR,
        FINAL_MODELS_DIR,
        TRAINING_HISTORY_DIR,
        TRAINING_LOGS_DIR,
        TRAINING_CONFIG_OUTPUT_PATH.parent,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# =============================================================================
# VALIDASI MODEL DAN SKENARIO
# =============================================================================

def validate_model_name(
    model_name: str,
) -> str:
    """
    Memastikan nama model valid.
    """

    normalized_name = (
        model_name
        .strip()
        .lower()
        .replace("-", "_")
    )

    allowed_models = {
        "cnn",
        "attention_bilstm",
    }

    if normalized_name not in allowed_models:
        raise ValueError(
            "Nama model tidak valid. "
            "Gunakan 'cnn' atau "
            "'attention_bilstm'."
        )

    return normalized_name


def validate_scenario_code(
    scenario_code: str,
) -> str:
    """
    Memastikan kode skenario tersedia.
    """

    normalized_code = (
        scenario_code
        .strip()
        .upper()
    )

    if (
        normalized_code
        not in SCENARIO_CONFIGS
    ):
        raise ValueError(
            f"Skenario '{scenario_code}' "
            f"tidak ditemukan.\n"
            f"Skenario tersedia: "
            f"{list(SCENARIO_CONFIGS.keys())}"
        )

    return normalized_code


# =============================================================================
# MENGAMBIL KONFIGURASI SKENARIO
# =============================================================================

def get_scenario_config(
    scenario_code: str,
) -> dict[str, Any]:
    """
    Mengambil konfigurasi berdasarkan kode skenario.
    """

    normalized_code = (
        validate_scenario_code(
            scenario_code
        )
    )

    configuration = (
        SCENARIO_CONFIGS[
            normalized_code
        ].copy()
    )

    configuration[
        "scenario_code"
    ] = normalized_code

    return configuration


# =============================================================================
# MEMBACA UKURAN VOCABULARY AKTUAL
# =============================================================================

def get_vocabulary_size(
    scenario_code: str,
) -> int:
    """
    Menghitung vocabulary aktual dari vocabulary.txt.

    Cara ini lebih aman daripada hanya menggunakan nilai
    vocabulary yang ditulis manual.
    """

    configuration = (
        get_scenario_config(
            scenario_code
        )
    )

    vocabulary_path = (
        configuration[
            "vectorized_directory"
        ]
        / "vocabulary.txt"
    )

    if not vocabulary_path.exists():
        raise FileNotFoundError(
            "Vocabulary tidak ditemukan:\n"
            f"{vocabulary_path}"
        )

    with open(
        vocabulary_path,
        "r",
        encoding="utf-8",
    ) as file:
        vocabulary_size = sum(
            1
            for _ in file
        )

    if vocabulary_size <= 2:
        raise ValueError(
            f"Vocabulary {scenario_code} "
            f"tidak valid: {vocabulary_size}"
        )

    return vocabulary_size


# =============================================================================
# PATH DATASET NPZ
# =============================================================================

def get_split_path(
    scenario_code: str,
    split_name: str,
) -> Path:
    """
    Mendapatkan path file NPZ berdasarkan skenario dan split.
    """

    normalized_code = (
        validate_scenario_code(
            scenario_code
        )
    )

    normalized_split = (
        split_name
        .strip()
        .lower()
    )

    allowed_splits = {
        "train",
        "validation",
        "test",
    }

    if normalized_split not in allowed_splits:
        raise ValueError(
            f"Split '{split_name}' tidak valid."
        )

    scenario_directory = (
        SCENARIO_CONFIGS[
            normalized_code
        ][
            "vectorized_directory"
        ]
    )

    split_path = (
        scenario_directory
        / f"{normalized_split}.npz"
    )

    if not split_path.exists():
        raise FileNotFoundError(
            f"File split tidak ditemukan:\n"
            f"{split_path}"
        )

    return split_path


# =============================================================================
# PATH OUTPUT EKSPERIMEN
# =============================================================================

def get_experiment_name(
    model_name: str,
    scenario_code: str,
) -> str:
    """
    Membentuk nama eksperimen.

    Contoh:
    cnn_k1
    attention_bilstm_k3
    """

    normalized_model = (
        validate_model_name(
            model_name
        )
    )

    normalized_scenario = (
        validate_scenario_code(
            scenario_code
        )
    )

    return (
        f"{normalized_model}_"
        f"{normalized_scenario.lower()}"
    )


def get_checkpoint_path(
    model_name: str,
    scenario_code: str,
) -> Path:
    """
    Path penyimpanan model terbaik berdasarkan validation loss.
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


def get_final_model_path(
    model_name: str,
    scenario_code: str,
) -> Path:
    """
    Path penyimpanan model akhir.
    """

    experiment_name = (
        get_experiment_name(
            model_name,
            scenario_code,
        )
    )

    return (
        FINAL_MODELS_DIR
        / f"{experiment_name}_final.keras"
    )


def get_history_path(
    model_name: str,
    scenario_code: str,
) -> Path:
    """
    Path CSV riwayat training.
    """

    experiment_name = (
        get_experiment_name(
            model_name,
            scenario_code,
        )
    )

    return (
        TRAINING_HISTORY_DIR
        / f"{experiment_name}_history.csv"
    )


def get_log_path(
    model_name: str,
    scenario_code: str,
) -> Path:
    """
    Path log training dari CSVLogger.
    """

    experiment_name = (
        get_experiment_name(
            model_name,
            scenario_code,
        )
    )

    return (
        TRAINING_LOGS_DIR
        / f"{experiment_name}_training_log.csv"
    )


# =============================================================================
# CALLBACK TRAINING
# =============================================================================

def create_training_callbacks(
    model_name: str,
    scenario_code: str,
) -> list[keras.callbacks.Callback]:
    """
    Membuat callback yang sama untuk CNN dan Attention-BiLSTM.

    Callback:
    1. EarlyStopping
    2. ReduceLROnPlateau
    3. ModelCheckpoint
    4. CSVLogger
    """

    create_training_directories()

    checkpoint_path = (
        get_checkpoint_path(
            model_name=model_name,
            scenario_code=scenario_code,
        )
    )

    log_path = (
        get_log_path(
            model_name=model_name,
            scenario_code=scenario_code,
        )
    )

    early_stopping = (
        keras.callbacks.EarlyStopping(
            monitor=(
                EARLY_STOPPING_MONITOR
            ),
            patience=(
                EARLY_STOPPING_PATIENCE
            ),
            min_delta=(
                EARLY_STOPPING_MIN_DELTA
            ),
            mode="min",
            restore_best_weights=True,
            verbose=1,
        )
    )

    reduce_learning_rate = (
        keras.callbacks.ReduceLROnPlateau(
            monitor=(
                REDUCE_LR_MONITOR
            ),
            factor=(
                REDUCE_LR_FACTOR
            ),
            patience=(
                REDUCE_LR_PATIENCE
            ),
            mode="min",
            min_lr=(
                MIN_LEARNING_RATE
            ),
            verbose=1,
        )
    )

    model_checkpoint = (
        keras.callbacks.ModelCheckpoint(
            filepath=str(
                checkpoint_path
            ),
            monitor=(
                CHECKPOINT_MONITOR
            ),
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )
    )

    csv_logger = (
        keras.callbacks.CSVLogger(
            filename=str(
                log_path
            ),
            separator=",",
            append=False,
        )
    )

    return [
        early_stopping,
        reduce_learning_rate,
        model_checkpoint,
        csv_logger,
    ]


# =============================================================================
# MENYIMPAN KONFIGURASI KE JSON
# =============================================================================

def save_training_configuration() -> None:
    """
    Menyimpan konfigurasi training agar dapat dilaporkan dan
    direproduksi.
    """

    create_training_directories()

    scenario_output = {}

    for (
        scenario_code,
        scenario_configuration,
    ) in SCENARIO_CONFIGS.items():

        vocabulary_path = (
            scenario_configuration[
                "vectorized_directory"
            ]
            / "vocabulary.txt"
        )

        vocabulary_size = None

        if vocabulary_path.exists():
            vocabulary_size = (
                get_vocabulary_size(
                    scenario_code
                )
            )

        scenario_output[
            scenario_code
        ] = {
            "dataset": (
                scenario_configuration[
                    "dataset"
                ]
            ),
            "scenario_name": (
                scenario_configuration[
                    "scenario_name"
                ]
            ),
            "max_sequence_length": (
                scenario_configuration[
                    "max_sequence_length"
                ]
            ),
            "actual_vocabulary_size":
                vocabulary_size,
            "vectorized_directory": str(
                scenario_configuration[
                    "vectorized_directory"
                ]
            ),
        }

    configuration = {
        "framework": (
            "TensorFlow/Keras"
        ),
        "tensorflow_version": (
            tf.__version__
        ),
        "random_seed": RANDOM_SEED,
        "num_classes": NUM_CLASSES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "loss": (
            "sparse_categorical_crossentropy"
        ),
        "training_metric": "accuracy",

        "shared_architecture": {
            "embedding_dim":
                EMBEDDING_DIM,
            "dense_units":
                DENSE_UNITS,
            "spatial_dropout_rate":
                SPATIAL_DROPOUT_RATE,
            "dropout_rate":
                DROPOUT_RATE,
        },

        "cnn": {
            "num_filters":
                CNN_NUM_FILTERS,
            "kernel_size":
                CNN_KERNEL_SIZE,
        },

        "attention_bilstm": {
            "lstm_units_per_direction":
                BILSTM_UNITS,
            "attention_units":
                ATTENTION_UNITS,
            "recurrent_dropout_rate":
                RECURRENT_DROPOUT_RATE,
        },

        "callbacks": {
            "early_stopping": {
                "monitor":
                    EARLY_STOPPING_MONITOR,
                "patience":
                    EARLY_STOPPING_PATIENCE,
                "min_delta":
                    EARLY_STOPPING_MIN_DELTA,
                "restore_best_weights":
                    True,
            },

            "reduce_lr_on_plateau": {
                "monitor":
                    REDUCE_LR_MONITOR,
                "factor":
                    REDUCE_LR_FACTOR,
                "patience":
                    REDUCE_LR_PATIENCE,
                "minimum_learning_rate":
                    MIN_LEARNING_RATE,
            },

            "model_checkpoint": {
                "monitor":
                    CHECKPOINT_MONITOR,
                "save_best_only":
                    True,
            },
        },

        "scenarios":
            scenario_output,

        "fair_comparison_note": (
            "CNN dan Attention-BiLSTM menggunakan "
            "data split, label, batch size, epoch maksimum, "
            "learning rate, optimizer, loss, dan callback "
            "yang sama. Perbedaan utama berada pada "
            "arsitektur model."
        ),
    }

    with open(
        TRAINING_CONFIG_OUTPUT_PATH,
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
# MENAMPILKAN RINGKASAN KONFIGURASI
# =============================================================================

def print_training_configuration() -> None:
    """
    Menampilkan konfigurasi utama di terminal.
    """

    print("=" * 72)
    print("STEP 5.3 - SHARED TRAINING CONFIGURATION")
    print("=" * 72)

    print("\nKonfigurasi umum:")
    print(
        f"TensorFlow version : "
        f"{tf.__version__}"
    )
    print(
        f"Random seed        : "
        f"{RANDOM_SEED}"
    )
    print(
        f"Jumlah kelas       : "
        f"{NUM_CLASSES}"
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
        "Optimizer          : Adam"
    )
    print(
        "Loss               : "
        "sparse_categorical_crossentropy"
    )

    print("\nKonfigurasi callback:")
    print(
        "Early stopping     : "
        f"patience "
        f"{EARLY_STOPPING_PATIENCE}"
    )
    print(
        "Reduce LR          : "
        f"factor "
        f"{REDUCE_LR_FACTOR}"
    )
    print(
        "Checkpoint         : "
        "berdasarkan val_loss terbaik"
    )

    print("\nKonfigurasi skenario:")

    for scenario_code in SCENARIO_CONFIGS:

        config = get_scenario_config(
            scenario_code
        )

        vocabulary_size = (
            get_vocabulary_size(
                scenario_code
            )
        )

        print(
            f"\n{scenario_code} "
            f"- {config['dataset']}"
        )
        print(
            f"Nama skenario      : "
            f"{config['scenario_name']}"
        )
        print(
            f"Sequence length    : "
            f"{config['max_sequence_length']}"
        )
        print(
            f"Vocabulary size    : "
            f"{vocabulary_size:,}"
        )
        print(
            f"Folder vectorized  : "
            f"{config['vectorized_directory']}"
        )


# =============================================================================
# TEST KONFIGURASI
# =============================================================================

if __name__ == "__main__":

    set_global_seed()

    create_training_directories()

    print_training_configuration()

    save_training_configuration()

    print("\n" + "=" * 72)
    print("VALIDASI CALLBACK")
    print("=" * 72)

    test_callbacks = (
        create_training_callbacks(
            model_name="cnn",
            scenario_code="K1",
        )
    )

    print("\nCallback yang digunakan:")

    for callback in test_callbacks:
        print(
            f"- {callback.__class__.__name__}"
        )

    print("\nFolder output berhasil dibuat:")
    print(
        f"Checkpoint       : "
        f"{CHECKPOINTS_DIR}"
    )
    print(
        f"Final models     : "
        f"{FINAL_MODELS_DIR}"
    )
    print(
        f"Training history : "
        f"{TRAINING_HISTORY_DIR}"
    )
    print(
        f"Training logs    : "
        f"{TRAINING_LOGS_DIR}"
    )

    print("\nKonfigurasi tersimpan di:")
    print(
        TRAINING_CONFIG_OUTPUT_PATH
    )

    print("\n" + "=" * 72)
    print(
        "Konfigurasi training berhasil dibuat."
    )
    print("=" * 72)
# =============================================================================
# STEP 5.3 - SHARED TRAINING CONFIGURATION
# =============================================================================
# File:
# 5_modeling/training_config.py
#
# Tujuan:
# Menyimpan konfigurasi pelatihan bersama untuk CNN dan
# Attention-BiLSTM agar seluruh eksperimen konsisten,
# terkontrol, dan reproducible.
#
# Eksperimen final:
#
# Kompas:
# - K1: Title
# - K2: Title + Description
# - K3: Title + Description + Keyword YAKE
#
# AG News:
# - A1: Title
# - A2: Title + Description
#
# K4 tidak digunakan.
# =============================================================================

from __future__ import annotations

import hashlib
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
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


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

# Patience dibuat lebih besar daripada ReduceLROnPlateau agar
# model memperoleh kesempatan berlatih setelah learning rate turun.
EARLY_STOPPING_PATIENCE = 5

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
# MODEL DAN SKENARIO FINAL
# =============================================================================

VALID_MODELS = [
    "cnn",
    "attention_bilstm",
]

VALID_SCENARIOS = [
    "K1",
    "K2",
    "K3",
    "A1",
    "A2",
]


# =============================================================================
# INFORMASI SETIAP SKENARIO
# =============================================================================
#
# Folder berikut berasal dari hasil:
# 4_preprocessing/07_text_vectorization.py
#
# Setiap folder harus memiliki:
# - train.npz
# - validation.npz
# - test.npz
# - vocabulary.txt
# - vectorizer_config.json
#
# Vocabulary bersama:
# - K1, K2, K3 menggunakan kompas_shared
# - A1, A2 menggunakan agnews_shared
# =============================================================================

SCENARIO_CONFIGS: dict[
    str,
    dict[str, Any],
] = {
    "K1": {
        "dataset": "Kompas",
        "scenario_name": "Title",
        "uses_yake": False,
        "comparison_group": (
            "kompas_text_representation"
        ),
        "shared_vocabulary_group": (
            "kompas_shared"
        ),
        "vectorized_directory": (
            VECTORIZED_DIR
            / "kompas_k1"
        ),
        "max_sequence_length": 20,
        "expected_vocabulary_size": 15_492,
        "expected_split_rows": {
            "train": 7_997,
            "validation": 1_000,
            "test": 1_000,
        },
    },

    "K2": {
        "dataset": "Kompas",
        "scenario_name": (
            "Title + Description"
        ),
        "uses_yake": False,
        "comparison_group": (
            "kompas_k2_k3_yake_ablation"
        ),
        "shared_vocabulary_group": (
            "kompas_shared"
        ),
        "vectorized_directory": (
            VECTORIZED_DIR
            / "kompas_k2"
        ),
        "max_sequence_length": 60,
        "expected_vocabulary_size": 15_492,
        "expected_split_rows": {
            "train": 7_997,
            "validation": 1_000,
            "test": 1_000,
        },
    },

    "K3": {
        "dataset": "Kompas",
        "scenario_name": (
            "Title + Description + Keyword YAKE"
        ),
        "uses_yake": True,
        "comparison_group": (
            "kompas_k2_k3_yake_ablation"
        ),
        "shared_vocabulary_group": (
            "kompas_shared"
        ),
        "vectorized_directory": (
            VECTORIZED_DIR
            / "kompas_k3"
        ),
        "max_sequence_length": 60,
        "expected_vocabulary_size": 15_492,
        "expected_split_rows": {
            "train": 7_997,
            "validation": 1_000,
            "test": 1_000,
        },
    },

    "A1": {
        "dataset": "AG News",
        "scenario_name": "Title",
        "uses_yake": False,
        "comparison_group": (
            "agnews_a1_a2_text_ablation"
        ),
        "shared_vocabulary_group": (
            "agnews_shared"
        ),
        "vectorized_directory": (
            VECTORIZED_DIR
            / "agnews_a1"
        ),
        "max_sequence_length": 60,
        "expected_vocabulary_size": 65_049,
        "expected_split_rows": {
            "train": 107_835,
            "validation": 11_982,
            "test": 7_600,
        },
    },

    "A2": {
        "dataset": "AG News",
        "scenario_name": (
            "Title + Description"
        ),
        "uses_yake": False,
        "comparison_group": (
            "agnews_a1_a2_text_ablation"
        ),
        "shared_vocabulary_group": (
            "agnews_shared"
        ),
        "vectorized_directory": (
            VECTORIZED_DIR
            / "agnews_a2"
        ),
        "max_sequence_length": 60,
        "expected_vocabulary_size": 65_049,
        "expected_split_rows": {
            "train": 107_835,
            "validation": 11_982,
            "test": 7_600,
        },
    },
}


# =============================================================================
# RANDOM SEED
# =============================================================================

def set_global_seed(
    seed: int = RANDOM_SEED,
) -> None:
    """
    Mengatur random seed untuk Python, NumPy, dan TensorFlow.

    Deterministic operations juga diaktifkan apabila tersedia.
    """

    if not isinstance(
        seed,
        int,
    ):
        raise TypeError(
            "seed harus bertipe integer."
        )

    if seed < 0:
        raise ValueError(
            "seed tidak boleh negatif."
        )

    os.environ[
        "PYTHONHASHSEED"
    ] = str(seed)

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    tf.random.set_seed(
        seed
    )

    keras.utils.set_random_seed(
        seed
    )

    try:
        tf.config.experimental.enable_op_determinism()

    except (
        AttributeError,
        RuntimeError,
    ):
        # Tidak semua versi atau perangkat TensorFlow
        # mendukung deterministic operations penuh.
        pass


# =============================================================================
# MEMBUAT FOLDER OUTPUT
# =============================================================================

def create_training_directories() -> None:
    """
    Membuat seluruh folder output training.
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
# VALIDASI MODEL
# =============================================================================

def validate_model_name(
    model_name: str,
) -> str:
    """
    Memastikan nama model valid.
    """

    if not isinstance(
        model_name,
        str,
    ):
        raise TypeError(
            "model_name harus bertipe string."
        )

    normalized_name = (
        model_name
        .strip()
        .lower()
        .replace("-", "_")
    )

    if normalized_name not in VALID_MODELS:
        raise ValueError(
            f"Nama model tidak valid: {model_name}\n"
            f"Model tersedia: {VALID_MODELS}"
        )

    return normalized_name


# =============================================================================
# VALIDASI SKENARIO
# =============================================================================

def validate_scenario_code(
    scenario_code: str,
) -> str:
    """
    Memastikan kode skenario termasuk eksperimen final.
    """

    if not isinstance(
        scenario_code,
        str,
    ):
        raise TypeError(
            "scenario_code harus bertipe string."
        )

    normalized_code = (
        scenario_code
        .strip()
        .upper()
    )

    if normalized_code not in SCENARIO_CONFIGS:
        raise ValueError(
            f"Skenario '{scenario_code}' tidak ditemukan.\n"
            f"Skenario tersedia: {VALID_SCENARIOS}"
        )

    return normalized_code


# =============================================================================
# MENGAMBIL KONFIGURASI SKENARIO
# =============================================================================

def get_scenario_config(
    scenario_code: str,
) -> dict[str, Any]:
    """
    Mengambil salinan konfigurasi berdasarkan kode skenario.
    """

    normalized_code = validate_scenario_code(
        scenario_code
    )

    configuration = SCENARIO_CONFIGS[
        normalized_code
    ].copy()

    configuration[
        "scenario_code"
    ] = normalized_code

    return configuration


# =============================================================================
# PATH ARTEFAK VECTORIZATION
# =============================================================================

def get_vocabulary_path(
    scenario_code: str,
) -> Path:
    """
    Mengambil path vocabulary.txt.
    """

    configuration = get_scenario_config(
        scenario_code
    )

    return (
        configuration[
            "vectorized_directory"
        ]
        / "vocabulary.txt"
    )


def get_vectorizer_config_path(
    scenario_code: str,
) -> Path:
    """
    Mengambil path vectorizer_config.json.
    """

    configuration = get_scenario_config(
        scenario_code
    )

    return (
        configuration[
            "vectorized_directory"
        ]
        / "vectorizer_config.json"
    )


# =============================================================================
# MEMBACA KONFIGURASI VECTORIZER
# =============================================================================

def load_vectorizer_config(
    scenario_code: str,
) -> dict[str, Any]:
    """
    Membaca vectorizer_config.json satu skenario.
    """

    normalized_code = validate_scenario_code(
        scenario_code
    )

    config_path = get_vectorizer_config_path(
        normalized_code
    )

    if not config_path.exists():
        raise FileNotFoundError(
            "Vectorizer config tidak ditemukan:\n"
            f"{config_path}"
        )

    if not config_path.is_file():
        raise ValueError(
            "Vectorizer config bukan file:\n"
            f"{config_path}"
        )

    if config_path.stat().st_size <= 0:
        raise ValueError(
            "Vectorizer config kosong:\n"
            f"{config_path}"
        )

    try:
        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as file:
            configuration = json.load(
                file
            )

    except json.JSONDecodeError as error:
        raise ValueError(
            "Vectorizer config tidak valid:\n"
            f"{config_path}"
        ) from error

    if not isinstance(
        configuration,
        dict,
    ):
        raise ValueError(
            "Isi vectorizer config harus berupa object JSON."
        )

    return configuration


# =============================================================================
# MEMBACA UKURAN VOCABULARY AKTUAL
# =============================================================================

def get_vocabulary_size(
    scenario_code: str,
) -> int:
    """
    Menghitung vocabulary aktual dari vocabulary.txt.

    Nilai kemudian dibandingkan dengan:
    - expected_vocabulary_size;
    - actual_vocabulary_size pada vectorizer_config.json.
    """

    normalized_code = validate_scenario_code(
        scenario_code
    )

    configuration = get_scenario_config(
        normalized_code
    )

    vocabulary_path = get_vocabulary_path(
        normalized_code
    )

    if not vocabulary_path.exists():
        raise FileNotFoundError(
            "Vocabulary tidak ditemukan:\n"
            f"{vocabulary_path}"
        )

    if not vocabulary_path.is_file():
        raise ValueError(
            "Vocabulary path bukan file:\n"
            f"{vocabulary_path}"
        )

    if vocabulary_path.stat().st_size <= 0:
        raise ValueError(
            "Vocabulary file kosong:\n"
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
            f"Vocabulary {normalized_code} "
            f"tidak valid: {vocabulary_size}"
        )

    expected_vocabulary_size = int(
        configuration[
            "expected_vocabulary_size"
        ]
    )

    if vocabulary_size != expected_vocabulary_size:
        raise ValueError(
            f"Ukuran vocabulary {normalized_code} "
            "tidak sesuai konfigurasi.\n"
            f"Expected : {expected_vocabulary_size:,}\n"
            f"Actual   : {vocabulary_size:,}"
        )

    vectorizer_config = load_vectorizer_config(
        normalized_code
    )

    config_vocabulary_size = (
        vectorizer_config.get(
            "actual_vocabulary_size"
        )
    )

    if config_vocabulary_size is not None:
        config_vocabulary_size = int(
            config_vocabulary_size
        )

        if (
            config_vocabulary_size
            != vocabulary_size
        ):
            raise ValueError(
                f"Ukuran vocabulary {normalized_code} "
                "berbeda dengan vectorizer config.\n"
                f"Vocabulary.txt : {vocabulary_size:,}\n"
                f"Config         : "
                f"{config_vocabulary_size:,}"
            )

    return vocabulary_size


# =============================================================================
# HASH VOCABULARY
# =============================================================================

def calculate_file_sha256(
    file_path: Path,
) -> str:
    """
    Menghitung SHA-256 suatu file.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan:\n{file_path}"
        )

    sha256 = hashlib.sha256()

    with open(
        file_path,
        "rb",
    ) as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            sha256.update(
                chunk
            )

    return sha256.hexdigest()


# =============================================================================
# PATH DATASET NPZ
# =============================================================================

def get_split_path(
    scenario_code: str,
    split_name: str,
) -> Path:
    """
    Mendapatkan path NPZ berdasarkan skenario dan split.
    """

    normalized_code = validate_scenario_code(
        scenario_code
    )

    if not isinstance(
        split_name,
        str,
    ):
        raise TypeError(
            "split_name harus bertipe string."
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
            f"Split '{split_name}' tidak valid.\n"
            f"Split tersedia: {sorted(allowed_splits)}"
        )

    scenario_directory = SCENARIO_CONFIGS[
        normalized_code
    ][
        "vectorized_directory"
    ]

    split_path = (
        scenario_directory
        / f"{normalized_split}.npz"
    )

    if not split_path.exists():
        raise FileNotFoundError(
            "File split tidak ditemukan:\n"
            f"{split_path}"
        )

    if not split_path.is_file():
        raise ValueError(
            "Path split bukan file:\n"
            f"{split_path}"
        )

    if split_path.stat().st_size <= 0:
        raise ValueError(
            "File split kosong:\n"
            f"{split_path}"
        )

    return split_path


# =============================================================================
# VALIDASI SATU FILE NPZ
# =============================================================================

def validate_npz_artifact(
    scenario_code: str,
    split_name: str,
) -> dict[str, Any]:
    """
    Memvalidasi key dan shape dasar file NPZ.
    """

    normalized_code = validate_scenario_code(
        scenario_code
    )

    configuration = get_scenario_config(
        normalized_code
    )

    split_path = get_split_path(
        normalized_code,
        split_name,
    )

    required_keys = {
        "X",
        "y",
        "document_id",
        "category",
    }

    with np.load(
        split_path,
        allow_pickle=False,
    ) as data:

        available_keys = set(
            data.files
        )

        missing_keys = (
            required_keys
            - available_keys
        )

        if missing_keys:
            raise KeyError(
                f"NPZ {normalized_code} {split_name} "
                "tidak memiliki seluruh key.\n"
                f"Key tidak ada: {sorted(missing_keys)}"
            )

        X = data["X"]
        y = data["y"]
        document_ids = data["document_id"]
        categories = data["category"]

        row_counts = {
            "X": len(X),
            "y": len(y),
            "document_id": len(
                document_ids
            ),
            "category": len(
                categories
            ),
        }

        if len(
            set(
                row_counts.values()
            )
        ) != 1:
            raise ValueError(
                f"Jumlah baris NPZ "
                f"{normalized_code} {split_name} "
                f"tidak sama:\n{row_counts}"
            )

        expected_rows = int(
            configuration[
                "expected_split_rows"
            ][
                split_name
            ]
        )

        actual_rows = int(
            len(X)
        )

        if actual_rows != expected_rows:
            raise ValueError(
                f"Jumlah data {normalized_code} "
                f"{split_name} tidak sesuai.\n"
                f"Expected : {expected_rows:,}\n"
                f"Actual   : {actual_rows:,}"
            )

        expected_sequence_length = int(
            configuration[
                "max_sequence_length"
            ]
        )

        if X.ndim != 2:
            raise ValueError(
                f"X {normalized_code} {split_name} "
                f"harus dua dimensi: {X.shape}"
            )

        if (
            X.shape[1]
            != expected_sequence_length
        ):
            raise ValueError(
                f"Sequence length {normalized_code} "
                f"{split_name} tidak sesuai.\n"
                f"Expected : {expected_sequence_length}\n"
                f"Actual   : {X.shape[1]}"
            )

        if y.ndim != 1:
            raise ValueError(
                f"y {normalized_code} {split_name} "
                f"harus satu dimensi: {y.shape}"
            )

    return {
        "scenario_code":
            normalized_code,
        "split":
            split_name,
        "path":
            str(split_path),
        "rows":
            actual_rows,
        "sequence_length":
            expected_sequence_length,
    }


# =============================================================================
# VALIDASI ARTEFAK SATU SKENARIO
# =============================================================================

def validate_scenario_artifacts(
    scenario_code: str,
) -> dict[str, Any]:
    """
    Memvalidasi vocabulary, vectorizer config, dan seluruh NPZ.
    """

    normalized_code = validate_scenario_code(
        scenario_code
    )

    configuration = get_scenario_config(
        normalized_code
    )

    vocabulary_size = get_vocabulary_size(
        normalized_code
    )

    vectorizer_config = load_vectorizer_config(
        normalized_code
    )

    config_scenario_code = (
        vectorizer_config.get(
            "scenario_code"
        )
    )

    if (
        config_scenario_code is not None
        and str(
            config_scenario_code
        ).upper()
        != normalized_code
    ):
        raise ValueError(
            f"Scenario code pada vectorizer config "
            f"{normalized_code} tidak sesuai.\n"
            f"Ditemukan: {config_scenario_code}"
        )

    config_sequence_length = (
        vectorizer_config.get(
            "output_sequence_length"
        )
    )

    if config_sequence_length is not None:
        config_sequence_length = int(
            config_sequence_length
        )

        expected_sequence_length = int(
            configuration[
                "max_sequence_length"
            ]
        )

        if (
            config_sequence_length
            != expected_sequence_length
        ):
            raise ValueError(
                f"Sequence length vectorizer "
                f"{normalized_code} tidak sesuai.\n"
                f"Expected : {expected_sequence_length}\n"
                f"Config   : {config_sequence_length}"
            )

    split_results = {}

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        split_results[
            split_name
        ] = validate_npz_artifact(
            normalized_code,
            split_name,
        )

    return {
        "scenario_code":
            normalized_code,
        "dataset":
            configuration[
                "dataset"
            ],
        "scenario_name":
            configuration[
                "scenario_name"
            ],
        "vocabulary_size":
            vocabulary_size,
        "max_sequence_length":
            configuration[
                "max_sequence_length"
            ],
        "vocabulary_sha256":
            calculate_file_sha256(
                get_vocabulary_path(
                    normalized_code
                )
            ),
        "splits":
            split_results,
    }


# =============================================================================
# VALIDASI VOCABULARY BERSAMA
# =============================================================================

def validate_shared_vocabularies() -> dict[str, Any]:
    """
    Memastikan skenario dalam kelompok yang sama benar-benar
    menggunakan vocabulary dan token ID yang sama.
    """

    groups = {
        "kompas_shared": [
            "K1",
            "K2",
            "K3",
        ],
        "agnews_shared": [
            "A1",
            "A2",
        ],
    }

    result = {}

    for group_name, scenario_codes in (
        groups.items()
    ):
        hashes = {
            scenario_code:
                calculate_file_sha256(
                    get_vocabulary_path(
                        scenario_code
                    )
                )
            for scenario_code
            in scenario_codes
        }

        unique_hashes = set(
            hashes.values()
        )

        if len(unique_hashes) != 1:
            raise ValueError(
                f"Vocabulary group {group_name} "
                "tidak identik.\n"
                f"Hash: {hashes}"
            )

        result[
            group_name
        ] = {
            "scenarios":
                scenario_codes,
            "vocabulary_sha256":
                next(
                    iter(
                        unique_hashes
                    )
                ),
            "vocabulary_size":
                get_vocabulary_size(
                    scenario_codes[0]
                ),
        }

    return result


# =============================================================================
# VALIDASI KONFIGURASI TRAINING
# =============================================================================

def validate_training_configuration() -> None:
    """
    Memvalidasi seluruh konstanta dan skenario eksperimen.
    """

    if set(
        SCENARIO_CONFIGS.keys()
    ) != set(
        VALID_SCENARIOS
    ):
        raise ValueError(
            "SCENARIO_CONFIGS harus tepat berisi "
            "K1, K2, K3, A1, dan A2."
        )

    if NUM_CLASSES <= 1:
        raise ValueError(
            "NUM_CLASSES harus lebih besar dari 1."
        )

    if EPOCHS <= 0:
        raise ValueError(
            "EPOCHS harus lebih besar dari 0."
        )

    if BATCH_SIZE <= 0:
        raise ValueError(
            "BATCH_SIZE harus lebih besar dari 0."
        )

    if LEARNING_RATE <= 0:
        raise ValueError(
            "LEARNING_RATE harus lebih besar dari 0."
        )

    if EMBEDDING_DIM <= 0:
        raise ValueError(
            "EMBEDDING_DIM harus lebih besar dari 0."
        )

    if DENSE_UNITS <= 0:
        raise ValueError(
            "DENSE_UNITS harus lebih besar dari 0."
        )

    if CNN_NUM_FILTERS <= 0:
        raise ValueError(
            "CNN_NUM_FILTERS harus lebih besar dari 0."
        )

    if CNN_KERNEL_SIZE <= 0:
        raise ValueError(
            "CNN_KERNEL_SIZE harus lebih besar dari 0."
        )

    if BILSTM_UNITS <= 0:
        raise ValueError(
            "BILSTM_UNITS harus lebih besar dari 0."
        )

    if ATTENTION_UNITS <= 0:
        raise ValueError(
            "ATTENTION_UNITS harus lebih besar dari 0."
        )

    dropout_values = {
        "SPATIAL_DROPOUT_RATE":
            SPATIAL_DROPOUT_RATE,
        "DROPOUT_RATE":
            DROPOUT_RATE,
        "RECURRENT_DROPOUT_RATE":
            RECURRENT_DROPOUT_RATE,
    }

    for name, value in (
        dropout_values.items()
    ):
        if not 0.0 <= value < 1.0:
            raise ValueError(
                f"{name} harus berada pada "
                "rentang 0 <= rate < 1."
            )

    if not 0.0 < REDUCE_LR_FACTOR < 1.0:
        raise ValueError(
            "REDUCE_LR_FACTOR harus berada "
            "pada rentang 0 < factor < 1."
        )

    if MIN_LEARNING_RATE <= 0:
        raise ValueError(
            "MIN_LEARNING_RATE harus lebih besar dari 0."
        )

    if (
        MIN_LEARNING_RATE
        >= LEARNING_RATE
    ):
        raise ValueError(
            "MIN_LEARNING_RATE harus lebih kecil "
            "dari LEARNING_RATE."
        )

    if (
        EARLY_STOPPING_PATIENCE
        <= REDUCE_LR_PATIENCE
    ):
        raise ValueError(
            "EARLY_STOPPING_PATIENCE harus lebih besar "
            "daripada REDUCE_LR_PATIENCE."
        )

    for scenario_code in VALID_SCENARIOS:
        scenario_config = get_scenario_config(
            scenario_code
        )

        if (
            CNN_KERNEL_SIZE
            > scenario_config[
                "max_sequence_length"
            ]
        ):
            raise ValueError(
                f"CNN_KERNEL_SIZE lebih besar dari "
                f"sequence length {scenario_code}."
            )


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
    - cnn_k1
    - attention_bilstm_k3
    """

    normalized_model = validate_model_name(
        model_name
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
    Path checkpoint berdasarkan val_loss terbaik.
    """

    experiment_name = get_experiment_name(
        model_name,
        scenario_code,
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
    Path model final dari checkpoint terbaik.
    """

    experiment_name = get_experiment_name(
        model_name,
        scenario_code,
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

    experiment_name = get_experiment_name(
        model_name,
        scenario_code,
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

    experiment_name = get_experiment_name(
        model_name,
        scenario_code,
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
    1. TerminateOnNaN
    2. ModelCheckpoint
    3. ReduceLROnPlateau
    4. EarlyStopping
    5. CSVLogger
    """

    create_training_directories()

    checkpoint_path = get_checkpoint_path(
        model_name=model_name,
        scenario_code=scenario_code,
    )

    log_path = get_log_path(
        model_name=model_name,
        scenario_code=scenario_code,
    )

    terminate_on_nan = (
        keras.callbacks.TerminateOnNaN()
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
            min_delta=(
                EARLY_STOPPING_MIN_DELTA
            ),
            min_lr=(
                MIN_LEARNING_RATE
            ),
            verbose=1,
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
        terminate_on_nan,
        model_checkpoint,
        reduce_learning_rate,
        early_stopping,
        csv_logger,
    ]


# =============================================================================
# MENYIMPAN KONFIGURASI KE JSON
# =============================================================================

def save_training_configuration() -> None:
    """
    Menyimpan konfigurasi training ke JSON secara atomik.
    """

    create_training_directories()

    scenario_output = {}

    for scenario_code in VALID_SCENARIOS:
        scenario_configuration = (
            get_scenario_config(
                scenario_code
            )
        )

        vocabulary_size = (
            get_vocabulary_size(
                scenario_code
            )
        )

        scenario_output[
            scenario_code
        ] = {
            "dataset":
                scenario_configuration[
                    "dataset"
                ],

            "scenario_name":
                scenario_configuration[
                    "scenario_name"
                ],

            "uses_yake":
                scenario_configuration[
                    "uses_yake"
                ],

            "comparison_group":
                scenario_configuration[
                    "comparison_group"
                ],

            "shared_vocabulary_group":
                scenario_configuration[
                    "shared_vocabulary_group"
                ],

            "max_sequence_length":
                scenario_configuration[
                    "max_sequence_length"
                ],

            "actual_vocabulary_size":
                vocabulary_size,

            "expected_split_rows":
                scenario_configuration[
                    "expected_split_rows"
                ],

            "vectorized_directory":
                str(
                    scenario_configuration[
                        "vectorized_directory"
                    ]
                ),

            "vocabulary_sha256":
                calculate_file_sha256(
                    get_vocabulary_path(
                        scenario_code
                    )
                ),
        }

    shared_vocabulary_validation = (
        validate_shared_vocabularies()
    )

    configuration = {
        "framework":
            "TensorFlow/Keras",

        "tensorflow_version":
            tf.__version__,

        "random_seed":
            RANDOM_SEED,

        "num_classes":
            NUM_CLASSES,

        "epochs_maximum":
            EPOCHS,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "optimizer":
            "Adam",

        "loss":
            "sparse_categorical_crossentropy",

        "training_metric":
            "accuracy",

        "model_selection_metric":
            "val_loss",

        "model_selection_mode":
            "min",

        "test_set_used_during_training":
            False,

        "k4_used":
            False,

        "total_models":
            len(
                VALID_MODELS
            ),

        "total_scenarios":
            len(
                VALID_SCENARIOS
            ),

        "total_experiments":
            len(
                VALID_MODELS
            )
            * len(
                VALID_SCENARIOS
            ),

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

            "padding_handling":
                (
                    "Padding embedding dibuat nol dan "
                    "full-padding convolution windows "
                    "diabaikan dalam global max pooling."
                ),
        },

        "attention_bilstm": {
            "lstm_units_per_direction":
                BILSTM_UNITS,

            "bilstm_output_units":
                BILSTM_UNITS * 2,

            "attention_units":
                ATTENTION_UNITS,

            "recurrent_dropout_rate":
                RECURRENT_DROPOUT_RATE,

            "padding_handling":
                (
                    "Embedding mask_zero=True dan mask "
                    "diteruskan ke BiLSTM serta attention."
                ),
        },

        "callbacks": {
            "terminate_on_nan":
                True,

            "early_stopping": {
                "monitor":
                    EARLY_STOPPING_MONITOR,

                "patience":
                    EARLY_STOPPING_PATIENCE,

                "min_delta":
                    EARLY_STOPPING_MIN_DELTA,

                "mode":
                    "min",

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

                "min_delta":
                    EARLY_STOPPING_MIN_DELTA,

                "minimum_learning_rate":
                    MIN_LEARNING_RATE,

                "mode":
                    "min",
            },

            "model_checkpoint": {
                "monitor":
                    CHECKPOINT_MONITOR,

                "mode":
                    "min",

                "save_best_only":
                    True,

                "save_weights_only":
                    False,
            },

            "csv_logger": {
                "append":
                    False,
            },
        },

        "shared_vocabulary_validation":
            shared_vocabulary_validation,

        "scenarios":
            scenario_output,

        "fair_comparison_controls": {
            "same_document_assignment":
                True,

            "same_label_mapping":
                True,

            "same_optimizer":
                True,

            "same_learning_rate":
                True,

            "same_batch_size":
                True,

            "same_epoch_maximum":
                True,

            "same_callbacks":
                True,

            "same_embedding_dimension":
                True,

            "k2_k3_same_max_length":
                True,

            "k2_k3_same_vocabulary":
                True,

            "a1_a2_same_max_length":
                True,

            "a1_a2_same_vocabulary":
                True,
        },

        "fair_comparison_note": (
            "CNN dan Attention-BiLSTM menggunakan data split, "
            "label mapping, batch size, epoch maksimum, "
            "learning rate, optimizer, loss, dimensi embedding, "
            "dan callback yang sama. Perbedaan utama berada "
            "pada mekanisme ekstraksi fitur arsitektur model."
        ),
    }

    temporary_path = (
        TRAINING_CONFIG_OUTPUT_PATH
        .with_name(
            f"{TRAINING_CONFIG_OUTPUT_PATH.stem}"
            ".tmp"
            f"{TRAINING_CONFIG_OUTPUT_PATH.suffix}"
        )
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            ensure_ascii=False,
            indent=4,
        )

    temporary_path.replace(
        TRAINING_CONFIG_OUTPUT_PATH
    )


# =============================================================================
# MENAMPILKAN RINGKASAN KONFIGURASI
# =============================================================================

def print_training_configuration() -> None:
    """
    Menampilkan konfigurasi utama dan memvalidasi artefak.
    """

    print("=" * 72)
    print(
        "STEP 5.3 - SHARED TRAINING CONFIGURATION"
    )
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

    print(
        f"Total skenario     : "
        f"{len(VALID_SCENARIOS)}"
    )

    print(
        f"Total model        : "
        f"{len(VALID_MODELS)}"
    )

    print(
        f"Total eksperimen   : "
        f"{len(VALID_SCENARIOS) * len(VALID_MODELS)}"
    )

    print(
        "K4 digunakan       : Tidak"
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
        f"{REDUCE_LR_FACTOR}, "
        f"patience "
        f"{REDUCE_LR_PATIENCE}"
    )

    print(
        "Checkpoint         : "
        "val_loss terbaik"
    )

    print("\nValidasi skenario:")

    for scenario_code in VALID_SCENARIOS:
        validation_result = (
            validate_scenario_artifacts(
                scenario_code
            )
        )

        config = get_scenario_config(
            scenario_code
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
            f"{validation_result['max_sequence_length']}"
        )

        print(
            f"Vocabulary size    : "
            f"{validation_result['vocabulary_size']:,}"
        )

        print(
            f"Vocabulary group   : "
            f"{config['shared_vocabulary_group']}"
        )

        print(
            f"Train              : "
            f"{validation_result['splits']['train']['rows']:,}"
        )

        print(
            f"Validation         : "
            f"{validation_result['splits']['validation']['rows']:,}"
        )

        print(
            f"Test               : "
            f"{validation_result['splits']['test']['rows']:,}"
        )

        print(
            f"Folder vectorized  : "
            f"{config['vectorized_directory']}"
        )

    print("\nValidasi shared vocabulary:")

    shared_groups = (
        validate_shared_vocabularies()
    )

    for group_name, group_data in (
        shared_groups.items()
    ):
        print(
            f"\n{group_name}"
        )

        print(
            f"Skenario           : "
            f"{group_data['scenarios']}"
        )

        print(
            f"Vocabulary size    : "
            f"{group_data['vocabulary_size']:,}"
        )

        print(
            "Vocabulary identik : Ya"
        )


# =============================================================================
# TEST KONFIGURASI
# =============================================================================

def main() -> None:
    """
    Memvalidasi dan menyimpan konfigurasi training.
    """

    set_global_seed()

    create_training_directories()

    validate_training_configuration()

    print_training_configuration()

    save_training_configuration()

    print("\n" + "=" * 72)
    print("VALIDASI CALLBACK")
    print("=" * 72)

    test_callbacks = create_training_callbacks(
        model_name="cnn",
        scenario_code="K1",
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
        "Konfigurasi training berhasil "
        "dibuat dan divalidasi."
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
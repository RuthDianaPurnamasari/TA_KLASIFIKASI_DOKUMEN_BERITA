"""
STEP 8.1 - PREPARE DEPLOYMENT MODELS

Menyiapkan seluruh artefak deployment untuk dashboard Streamlit.

Model deployment:
1. CNN K2
2. Attention-BiLSTM K2

Representasi teks:
Title + Description
"""

from __future__ import annotations

import gc
import hashlib
import json
import platform
import shutil
import sys
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

MODELING_DIR = (
    PROJECT_ROOT
    / "5_modeling"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

if str(MODELING_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(MODELING_DIR),
    )


# =============================================================================
# IMPORT CUSTOM MODEL COMPONENTS
# =============================================================================

from attention_bilstm_model import (  # noqa: E402
    AttentionPooling1D,
)

from cnn_model import (  # noqa: E402
    MaskedGlobalMaxPooling1D,
    ZeroPaddingEmbeddingOutput,
)


# =============================================================================
# KONFIGURASI PENELITIAN
# =============================================================================

DATASET_NAME = "Kompas"
SCENARIO_CODE = "K2"
SCENARIO_NAME = "Title + Description"

SEQUENCE_LENGTH = 60
NUM_CLASSES = 4

PRIMARY_EXPERIMENT = "cnn_k2"
COMPARISON_EXPERIMENT = "attention_bilstm_k2"

PREDICTION_ATOL = 1e-6
PROBABILITY_ATOL = 1e-4

EXPECTED_INDEX_TO_LABEL = {
    0: "bola",
    1: "global",
    2: "money",
    3: "tekno",
}


# =============================================================================
# MODEL SPECIFICATION
# =============================================================================

MODEL_SPECS = {
    "cnn_k2": {
        "display_name": "CNN",
        "model_family": "cnn",
        "deployment_filename": "cnn_k2.keras",
        "required_custom_layers": {
            "ZeroPaddingEmbeddingOutput",
            "MaskedGlobalMaxPooling1D",
        },
    },

    "attention_bilstm_k2": {
        "display_name": "Attention-BiLSTM",
        "model_family": "attention_bilstm",
        "deployment_filename": "attention_bilstm_k2.keras",
        "required_custom_layers": {
            "AttentionPooling1D",
        },
    },
}


# =============================================================================
# CUSTOM OBJECTS
# =============================================================================

CUSTOM_OBJECTS = {
    "AttentionPooling1D":
        AttentionPooling1D,

    "TAKlasifikasiBerita>AttentionPooling1D":
        AttentionPooling1D,

    "ZeroPaddingEmbeddingOutput":
        ZeroPaddingEmbeddingOutput,

    "TAKlasifikasiBerita>ZeroPaddingEmbeddingOutput":
        ZeroPaddingEmbeddingOutput,

    "MaskedGlobalMaxPooling1D":
        MaskedGlobalMaxPooling1D,

    "TAKlasifikasiBerita>MaskedGlobalMaxPooling1D":
        MaskedGlobalMaxPooling1D,
}


# =============================================================================
# PATH INPUT
# =============================================================================

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "8_save_models"
    / "checkpoints"
)

FINAL_MODEL_DIR = (
    PROJECT_ROOT
    / "8_save_models"
    / "final_models"
)

VECTORIZED_DIR = (
    PROJECT_ROOT
    / "2_data"
    / "vectorized"
    / "kompas_k2"
)

TEST_DATA_PATH = (
    VECTORIZED_DIR
    / "test.npz"
)

VOCABULARY_SOURCE_PATH = (
    VECTORIZED_DIR
    / "vocabulary.txt"
)

VECTORIZER_CONFIG_SOURCE_PATH = (
    VECTORIZED_DIR
    / "vectorizer_config.json"
)

LABEL_MAPPING_SOURCE_PATH = (
    PROJECT_ROOT
    / "9_results"
    / "tables"
    / "label_mapping.json"
)

# Path yang benar untuk hasil evaluasi seluruh model.
MODEL_METRICS_PATH = (
    PROJECT_ROOT
    / "9_results"
    / "metrics"
    / "model_test_metrics.csv"
)


# =============================================================================
# PATH OUTPUT
# =============================================================================

DEPLOYMENT_DIR = (
    PROJECT_ROOT
    / "8_save_models"
    / "deployment"
)

VOCABULARY_OUTPUT_PATH = (
    DEPLOYMENT_DIR
    / "vocabulary.txt"
)

VECTORIZER_CONFIG_OUTPUT_PATH = (
    DEPLOYMENT_DIR
    / "vectorizer_config.json"
)

LABEL_MAPPING_OUTPUT_PATH = (
    DEPLOYMENT_DIR
    / "label_mapping.json"
)

DEPLOYMENT_CONFIG_PATH = (
    DEPLOYMENT_DIR
    / "deployment_config.json"
)

DEPLOYMENT_REPORT_PATH = (
    DEPLOYMENT_DIR
    / "deployment_report.json"
)


# =============================================================================
# UTILITAS UMUM
# =============================================================================

def print_header(
    title: str,
) -> None:
    """Menampilkan header terminal."""

    print("=" * 80)
    print(title)
    print("=" * 80)


def json_default(
    value: Any,
) -> Any:
    """Mengubah tipe khusus menjadi format yang dapat disimpan ke JSON."""

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, set):
        return sorted(value)

    if isinstance(value, tuple):
        return list(value)

    raise TypeError(
        f"Tipe data tidak dapat disimpan ke JSON: {type(value)}"
    )


def write_json(
    output_path: Path,
    data: dict[str, Any],
) -> None:
    """Menyimpan dictionary sebagai JSON."""

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
            data,
            file,
            ensure_ascii=False,
            indent=4,
            default=json_default,
        )


def sha256_file(
    file_path: Path,
) -> str:
    """Menghitung SHA-256 file."""

    digest = hashlib.sha256()

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

            digest.update(
                chunk
            )

    return digest.hexdigest()


def get_file_metadata(
    file_path: Path,
) -> dict[str, Any]:
    """Membuat metadata audit sebuah file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Path bukan file: {file_path}"
        )

    return {
        "path":
            str(file_path),

        "filename":
            file_path.name,

        "size_bytes":
            int(
                file_path.stat().st_size
            ),

        "sha256":
            sha256_file(
                file_path
            ),
    }


def copy_verified(
    source_path: Path,
    destination_path: Path,
) -> dict[str, Any]:
    """Menyalin file dan memeriksa hash sumber dan tujuan."""

    if not source_path.exists():
        raise FileNotFoundError(
            "File sumber tidak ditemukan:\n"
            f"{source_path}"
        )

    if not source_path.is_file():
        raise ValueError(
            "Path sumber bukan file:\n"
            f"{source_path}"
        )

    if source_path.stat().st_size <= 0:
        raise ValueError(
            "File sumber kosong:\n"
            f"{source_path}"
        )

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source_path,
        destination_path,
    )

    source_hash = sha256_file(
        source_path
    )

    destination_hash = sha256_file(
        destination_path
    )

    if source_hash != destination_hash:
        raise IOError(
            "Hash file sumber dan hasil penyalinan tidak sama."
        )

    return {
        "status":
            "success",

        "source":
            get_file_metadata(
                source_path
            ),

        "deployment":
            get_file_metadata(
                destination_path
            ),
    }


def clean_old_outputs() -> None:
    """Menghapus artefak deployment lama."""

    DEPLOYMENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_paths = [
        VOCABULARY_OUTPUT_PATH,
        VECTORIZER_CONFIG_OUTPUT_PATH,
        LABEL_MAPPING_OUTPUT_PATH,
        DEPLOYMENT_CONFIG_PATH,
        DEPLOYMENT_REPORT_PATH,
    ]

    for specification in MODEL_SPECS.values():
        output_paths.append(
            DEPLOYMENT_DIR
            / specification[
                "deployment_filename"
            ]
        )

    for output_path in output_paths:
        if output_path.exists():
            if output_path.is_file():
                output_path.unlink()
            else:
                shutil.rmtree(
                    output_path
                )


def scalar_to_json(
    value: Any,
) -> Any:
    """Mengubah satu nilai Pandas atau NumPy menjadi tipe JSON."""

    if value is None:
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if np.isnan(value):
            return None

        return float(value)

    if isinstance(value, float):
        if np.isnan(value):
            return None

        return value

    if pd.isna(value):
        return None

    return value


# =============================================================================
# TEST DATA
# =============================================================================

def load_test_data() -> dict[str, np.ndarray]:
    """Memuat test set Kompas K2."""

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            "Test set Kompas K2 tidak ditemukan:\n"
            f"{TEST_DATA_PATH}"
        )

    with np.load(
        TEST_DATA_PATH,
        allow_pickle=False,
    ) as data:
        required_keys = {
            "X",
            "y",
            "document_id",
            "category",
        }

        missing_keys = (
            required_keys
            - set(data.files)
        )

        if missing_keys:
            raise KeyError(
                "Komponen test set tidak lengkap.\n"
                f"Key hilang: {sorted(missing_keys)}"
            )

        test_data = {
            "X":
                np.asarray(
                    data["X"],
                    dtype=np.int32,
                ),

            "y":
                np.asarray(
                    data["y"],
                    dtype=np.int32,
                ),

            "document_id":
                np.asarray(
                    data["document_id"],
                    dtype=str,
                ),

            "category":
                np.asarray(
                    data["category"],
                    dtype=str,
                ),
        }

    X_test = test_data["X"]
    y_test = test_data["y"]

    if X_test.ndim != 2:
        raise ValueError(
            f"X test harus dua dimensi. Shape: {X_test.shape}"
        )

    if X_test.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(
            "Sequence length test set tidak sesuai.\n"
            f"Expected : {SEQUENCE_LENGTH}\n"
            f"Actual   : {X_test.shape[1]}"
        )

    lengths = {
        key: len(value)
        for key, value
        in test_data.items()
    }

    if len(set(lengths.values())) != 1:
        raise ValueError(
            "Jumlah data pada setiap komponen test set tidak sama.\n"
            f"{lengths}"
        )

    if np.any(X_test < 0):
        raise ValueError(
            "Ditemukan token ID negatif pada test set."
        )

    actual_classes = set(
        np.unique(
            y_test
        ).tolist()
    )

    expected_classes = set(
        range(
            NUM_CLASSES
        )
    )

    if actual_classes != expected_classes:
        raise ValueError(
            "Kelas test set tidak sesuai.\n"
            f"Expected : {sorted(expected_classes)}\n"
            f"Actual   : {sorted(actual_classes)}"
        )

    return test_data


def select_smoke_test_samples(
    test_data: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Memilih satu sampel test dari setiap kelas."""

    selected_indices: list[int] = []

    for class_index in range(
        NUM_CLASSES
    ):
        matching_indices = np.where(
            test_data["y"]
            == class_index
        )[0]

        if len(matching_indices) == 0:
            raise ValueError(
                "Tidak ditemukan sampel test untuk kelas "
                f"{class_index}."
            )

        selected_indices.append(
            int(
                matching_indices[0]
            )
        )

    selected_indices_array = np.asarray(
        selected_indices,
        dtype=np.int32,
    )

    return {
        "indices":
            selected_indices_array,

        "X":
            test_data["X"][
                selected_indices_array
            ],

        "y":
            test_data["y"][
                selected_indices_array
            ],

        "document_id":
            test_data["document_id"][
                selected_indices_array
            ],

        "category":
            test_data["category"][
                selected_indices_array
            ],
    }


# =============================================================================
# VOCABULARY
# =============================================================================

def load_vocabulary() -> list[str]:
    """Memuat vocabulary Kompas K2."""

    if not VOCABULARY_SOURCE_PATH.exists():
        raise FileNotFoundError(
            "Vocabulary Kompas K2 tidak ditemukan:\n"
            f"{VOCABULARY_SOURCE_PATH}"
        )

    vocabulary = (
        VOCABULARY_SOURCE_PATH
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

    if len(vocabulary) <= 2:
        raise ValueError(
            "Vocabulary tidak valid.\n"
            f"Vocabulary size: {len(vocabulary)}"
        )

    return vocabulary


# =============================================================================
# VECTORIZER CONFIG
# =============================================================================

def find_recursive_values(
    data: Any,
    target_keys: set[str],
) -> list[Any]:
    """Mencari nilai berdasarkan key secara rekursif."""

    results: list[Any] = []

    if isinstance(data, dict):
        for key, value in data.items():
            normalized_key = (
                str(key)
                .strip()
                .lower()
            )

            if normalized_key in target_keys:
                results.append(
                    value
                )

            results.extend(
                find_recursive_values(
                    value,
                    target_keys,
                )
            )

    elif isinstance(data, list):
        for item in data:
            results.extend(
                find_recursive_values(
                    item,
                    target_keys,
                )
            )

    return results


def load_vectorizer_config(
    vocabulary_size: int,
) -> dict[str, Any]:
    """Memuat dan memvalidasi vectorizer configuration."""

    if not VECTORIZER_CONFIG_SOURCE_PATH.exists():
        raise FileNotFoundError(
            "Vectorizer configuration tidak ditemukan:\n"
            f"{VECTORIZER_CONFIG_SOURCE_PATH}"
        )

    with open(
        VECTORIZER_CONFIG_SOURCE_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        configuration = json.load(
            file
        )

    sequence_values = find_recursive_values(
        configuration,
        {
            "sequence_length",
            "output_sequence_length",
            "max_sequence_length",
            "max_len",
        },
    )

    parsed_sequence_values: list[int] = []

    for value in sequence_values:
        try:
            parsed_sequence_values.append(
                int(value)
            )
        except (TypeError, ValueError):
            continue

    if not parsed_sequence_values:
        raise KeyError(
            "Sequence length tidak ditemukan pada vectorizer configuration."
        )

    if SEQUENCE_LENGTH not in parsed_sequence_values:
        raise ValueError(
            "Sequence length vectorizer tidak sesuai.\n"
            f"Expected   : {SEQUENCE_LENGTH}\n"
            f"Ditemukan  : {parsed_sequence_values}"
        )

    vocabulary_values = find_recursive_values(
        configuration,
        {
            "vocabulary_size",
            "vocab_size",
            "max_tokens",
        },
    )

    parsed_vocabulary_values: list[int] = []

    for value in vocabulary_values:
        if value is None:
            continue

        try:
            parsed_vocabulary_values.append(
                int(value)
            )
        except (TypeError, ValueError):
            continue

    for config_vocabulary_size in parsed_vocabulary_values:
        if config_vocabulary_size < vocabulary_size:
            raise ValueError(
                "Vocabulary size pada vectorizer configuration "
                "lebih kecil dari vocabulary aktual.\n"
                f"Vocabulary aktual : {vocabulary_size}\n"
                f"Nilai config      : {config_vocabulary_size}"
            )

    return configuration


# =============================================================================
# LABEL MAPPING
# =============================================================================

def extract_index_to_label(
    mapping_data: dict[str, Any],
) -> dict[int, str]:
    """Mengambil mapping label Kompas dari beberapa format JSON."""

    mapping: Any = mapping_data

    for key in [
        "Kompas",
        "kompas",
        "KOMPAS",
    ]:
        if (
            isinstance(mapping_data, dict)
            and key in mapping_data
        ):
            mapping = mapping_data[key]
            break

    if not isinstance(mapping, dict):
        raise ValueError(
            "Format label mapping tidak valid."
        )

    if "index_to_label" in mapping:
        raw_mapping = mapping[
            "index_to_label"
        ]

        result = {
            int(index):
                str(label)
                .strip()
                .lower()

            for index, label
            in raw_mapping.items()
        }

    elif "label_to_index" in mapping:
        raw_mapping = mapping[
            "label_to_index"
        ]

        result = {
            int(index):
                str(label)
                .strip()
                .lower()

            for label, index
            in raw_mapping.items()
        }

    elif all(
        str(key).isdigit()
        for key in mapping.keys()
    ):
        result = {
            int(index):
                str(label)
                .strip()
                .lower()

            for index, label
            in mapping.items()
        }

    elif all(
        isinstance(value, (int, np.integer))
        for value in mapping.values()
    ):
        result = {
            int(index):
                str(label)
                .strip()
                .lower()

            for label, index
            in mapping.items()
        }

    else:
        raise ValueError(
            "Struktur label mapping tidak dikenali."
        )

    return dict(
        sorted(
            result.items()
        )
    )


def load_label_mapping() -> dict[int, str]:
    """Memuat label mapping Kompas."""

    if not LABEL_MAPPING_SOURCE_PATH.exists():
        raise FileNotFoundError(
            "Label mapping tidak ditemukan:\n"
            f"{LABEL_MAPPING_SOURCE_PATH}"
        )

    with open(
        LABEL_MAPPING_SOURCE_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        mapping_data = json.load(
            file
        )

    index_to_label = extract_index_to_label(
        mapping_data
    )

    if index_to_label != EXPECTED_INDEX_TO_LABEL:
        raise ValueError(
            "Label mapping tidak sesuai dengan penelitian.\n"
            f"Expected : {EXPECTED_INDEX_TO_LABEL}\n"
            f"Actual   : {index_to_label}"
        )

    return index_to_label


def save_deployment_label_mapping(
    index_to_label: dict[int, str],
) -> dict[str, Any]:
    """Menyimpan label mapping khusus deployment."""

    mapping = {
        "dataset":
            DATASET_NAME,

        "num_classes":
            NUM_CLASSES,

        "index_to_label": {
            str(index): label
            for index, label
            in index_to_label.items()
        },

        "label_to_index": {
            label: index
            for index, label
            in index_to_label.items()
        },
    }

    write_json(
        LABEL_MAPPING_OUTPUT_PATH,
        mapping,
    )

    return {
        "status":
            "success",

        "source_path":
            str(
                LABEL_MAPPING_SOURCE_PATH
            ),

        "deployment":
            get_file_metadata(
                LABEL_MAPPING_OUTPUT_PATH
            ),

        "mapping":
            mapping,
    }


# =============================================================================
# MODEL METRICS
# =============================================================================

def load_model_metrics() -> dict[str, dict[str, Any]]:
    """Memuat hasil evaluasi test set CNN K2 dan Attention-BiLSTM K2."""

    if not MODEL_METRICS_PATH.exists():
        raise FileNotFoundError(
            "Metrik test tidak ditemukan:\n"
            f"{MODEL_METRICS_PATH}"
        )

    dataframe = pd.read_csv(
        MODEL_METRICS_PATH,
        encoding="utf-8-sig",
    )

    if dataframe.empty:
        raise ValueError(
            "File model_test_metrics.csv kosong."
        )

    required_columns = {
        "experiment_name",
        "dataset",
        "scenario_code",
        "accuracy",
        "f1_macro",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Kolom metrik test tidak lengkap.\n"
            f"Kolom hilang: {sorted(missing_columns)}"
        )

    string_columns = [
        "experiment_name",
        "dataset",
        "scenario_code",
    ]

    for column in string_columns:
        dataframe[column] = (
            dataframe[column]
            .astype(str)
            .str.strip()
        )

    numeric_candidates = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "log_loss",
        "inference_time_seconds",
        "average_inference_ms_per_sample",
        "jumlah_test",
        "correct_predictions",
        "incorrect_predictions",
    ]

    numeric_columns = [
        column
        for column in numeric_candidates
        if column in dataframe.columns
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    if dataframe[numeric_columns].isna().any().any():
        invalid_columns = (
            dataframe[
                numeric_columns
            ]
            .columns[
                dataframe[
                    numeric_columns
                ]
                .isna()
                .any()
            ]
            .tolist()
        )

        raise ValueError(
            "Terdapat nilai numerik tidak valid pada kolom:\n"
            f"{invalid_columns}"
        )

    result: dict[str, dict[str, Any]] = {}

    for experiment_name in MODEL_SPECS:
        selected = dataframe[
            (
                dataframe[
                    "experiment_name"
                ].str.lower()
                == experiment_name.lower()
            )
            & (
                dataframe[
                    "dataset"
                ].str.lower()
                == DATASET_NAME.lower()
            )
            & (
                dataframe[
                    "scenario_code"
                ].str.upper()
                == SCENARIO_CODE.upper()
            )
        ]

        if len(selected) != 1:
            raise ValueError(
                "Metrik eksperimen harus ditemukan tepat satu baris.\n"
                f"Experiment   : {experiment_name}\n"
                f"Jumlah baris : {len(selected)}"
            )

        row = selected.iloc[0]

        experiment_metrics = {
            column:
                scalar_to_json(
                    row[column]
                )

            for column
            in dataframe.columns
        }

        accuracy = float(
            experiment_metrics[
                "accuracy"
            ]
        )

        f1_macro = float(
            experiment_metrics[
                "f1_macro"
            ]
        )

        if not 0.0 <= accuracy <= 1.0:
            raise ValueError(
                f"Accuracy {experiment_name} di luar rentang 0–1."
            )

        if not 0.0 <= f1_macro <= 1.0:
            raise ValueError(
                f"Macro F1 {experiment_name} di luar rentang 0–1."
            )

        result[
            experiment_name
        ] = experiment_metrics

    ranked_models = sorted(
        result.keys(),
        key=lambda name: (
            -float(
                result[name][
                    "accuracy"
                ]
            ),
            -float(
                result[name][
                    "f1_macro"
                ]
            ),
            float(
                result[name].get(
                    "log_loss",
                    np.inf,
                )
                if result[name].get(
                    "log_loss"
                ) is not None
                else np.inf
            ),
        ),
    )

    if ranked_models[0] != PRIMARY_EXPERIMENT:
        raise ValueError(
            "Model terbaik berdasarkan metrik test bukan CNN K2.\n"
            f"Model terbaik : {ranked_models[0]}\n"
            f"Model utama   : {PRIMARY_EXPERIMENT}"
        )

    return result


# =============================================================================
# PEMILIHAN CHECKPOINT MODEL
# =============================================================================

def get_candidate_model_paths(
    experiment_name: str,
) -> list[Path]:
    """Membuat daftar kandidat checkpoint berdasarkan prioritas."""

    candidate_names = [
        f"{experiment_name}_best.keras",
        f"best_{experiment_name}.keras",
        f"{experiment_name}_final.keras",
        f"{experiment_name}.keras",

        f"{experiment_name}_best.h5",
        f"best_{experiment_name}.h5",
        f"{experiment_name}_final.h5",
        f"{experiment_name}.h5",
    ]

    candidate_paths: list[Path] = []

    for directory in [
        CHECKPOINT_DIR,
        FINAL_MODEL_DIR,
    ]:
        for candidate_name in candidate_names:
            candidate_paths.append(
                directory
                / candidate_name
            )

    return candidate_paths


def find_model_file(
    experiment_name: str,
) -> Path:
    """Mencari checkpoint model secara deterministik."""

    candidate_paths = get_candidate_model_paths(
        experiment_name
    )

    for candidate_path in candidate_paths:
        if (
            candidate_path.exists()
            and candidate_path.is_file()
            and candidate_path.stat().st_size > 0
        ):
            return candidate_path

    # Flexible fallback apabila nama file memiliki suffix tambahan.
    flexible_matches: list[Path] = []

    for directory in [
        CHECKPOINT_DIR,
        FINAL_MODEL_DIR,
    ]:
        if not directory.exists():
            continue

        flexible_matches.extend(
            directory.glob(
                f"*{experiment_name}*.keras"
            )
        )

        flexible_matches.extend(
            directory.glob(
                f"*{experiment_name}*.h5"
            )
        )

    flexible_matches = sorted(
        {
            path.resolve()
            for path in flexible_matches
            if path.is_file()
            and path.stat().st_size > 0
        },
        key=lambda path: (
            0
            if "best" in path.name.lower()
            else 1,

            0
            if path.suffix.lower() == ".keras"
            else 1,

            path.name.lower(),
        ),
    )

    if flexible_matches:
        return Path(
            flexible_matches[0]
        )

    checked_paths = "\n".join(
        f"- {path}"
        for path in candidate_paths
    )

    raise FileNotFoundError(
        f"Model {experiment_name} tidak ditemukan.\n"
        f"Path yang diperiksa:\n{checked_paths}"
    )


# =============================================================================
# VALIDASI MODEL
# =============================================================================

def normalize_model_shape(
    shape: Any,
    shape_name: str,
) -> tuple[Any, ...]:
    """Menormalisasi input atau output shape model."""

    if isinstance(shape, list):
        if len(shape) != 1:
            raise ValueError(
                f"Model harus memiliki satu {shape_name}.\n"
                f"Shape: {shape}"
            )

        shape = shape[0]

    return tuple(
        shape
    )


def find_embedding_layer(
    model: tf.keras.Model,
) -> tf.keras.layers.Embedding:
    """Mencari layer Embedding pada model."""

    for layer in model.layers:
        if isinstance(
            layer,
            tf.keras.layers.Embedding,
        ):
            return layer

    raise ValueError(
        "Layer Embedding tidak ditemukan pada model."
    )


def validate_model_architecture(
    model: tf.keras.Model,
    experiment_name: str,
    vocabulary_size: int,
) -> dict[str, Any]:
    """Memvalidasi arsitektur model."""

    input_shape = normalize_model_shape(
        model.input_shape,
        "input",
    )

    output_shape = normalize_model_shape(
        model.output_shape,
        "output",
    )

    expected_input_shape = (
        None,
        SEQUENCE_LENGTH,
    )

    expected_output_shape = (
        None,
        NUM_CLASSES,
    )

    if input_shape != expected_input_shape:
        raise ValueError(
            f"Input shape {experiment_name} tidak sesuai.\n"
            f"Expected : {expected_input_shape}\n"
            f"Actual   : {input_shape}"
        )

    if output_shape != expected_output_shape:
        raise ValueError(
            f"Output shape {experiment_name} tidak sesuai.\n"
            f"Expected : {expected_output_shape}\n"
            f"Actual   : {output_shape}"
        )

    embedding_layer = find_embedding_layer(
        model
    )

    embedding_input_dim = int(
        embedding_layer.input_dim
    )

    embedding_output_dim = int(
        embedding_layer.output_dim
    )

    if embedding_input_dim != vocabulary_size:
        raise ValueError(
            f"Embedding input_dim {experiment_name} "
            "tidak sama dengan vocabulary size.\n"
            f"Embedding input_dim : {embedding_input_dim}\n"
            f"Vocabulary size     : {vocabulary_size}"
        )

    layer_types = {
        type(layer).__name__
        for layer in model.layers
    }

    required_custom_layers = MODEL_SPECS[
        experiment_name
    ][
        "required_custom_layers"
    ]

    missing_custom_layers = (
        required_custom_layers
        - layer_types
    )

    if missing_custom_layers:
        raise ValueError(
            f"Custom layer {experiment_name} tidak lengkap.\n"
            f"Layer hilang: {sorted(missing_custom_layers)}"
        )

    return {
        "input_shape":
            list(
                input_shape
            ),

        "output_shape":
            list(
                output_shape
            ),

        "embedding_layer_name":
            embedding_layer.name,

        "embedding_input_dim":
            embedding_input_dim,

        "embedding_output_dim":
            embedding_output_dim,

        "parameter_count":
            int(
                model.count_params()
            ),

        "layer_count":
            int(
                len(
                    model.layers
                )
            ),

        "layer_types":
            [
                type(layer).__name__
                for layer
                in model.layers
            ],

        "required_custom_layers":
            sorted(
                required_custom_layers
            ),
    }


def predict_and_validate(
    model: tf.keras.Model,
    X: np.ndarray,
    context: str,
) -> np.ndarray:
    """Menjalankan prediksi dan memvalidasi probabilitas."""

    probabilities = model.predict(
        X,
        verbose=0,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    expected_shape = (
        len(X),
        NUM_CLASSES,
    )

    if probabilities.shape != expected_shape:
        raise ValueError(
            f"Shape prediksi {context} tidak sesuai.\n"
            f"Expected : {expected_shape}\n"
            f"Actual   : {probabilities.shape}"
        )

    if not np.all(
        np.isfinite(
            probabilities
        )
    ):
        raise ValueError(
            f"Prediksi {context} mengandung NaN atau infinity."
        )

    if (
        np.any(probabilities < -1e-7)
        or np.any(probabilities > 1.0 + 1e-7)
    ):
        raise ValueError(
            f"Probabilitas {context} berada di luar rentang 0–1."
        )

    probability_sums = np.sum(
        probabilities,
        axis=1,
    )

    if not np.allclose(
        probability_sums,
        1.0,
        atol=PROBABILITY_ATOL,
    ):
        raise ValueError(
            f"Jumlah probabilitas {context} tidak mendekati 1.\n"
            f"Nilai: {probability_sums.tolist()}"
        )

    return probabilities


# =============================================================================
# PREPARE MODEL
# =============================================================================

def prepare_single_model(
    experiment_name: str,
    vocabulary_size: int,
    smoke_batch: dict[str, Any],
    index_to_label: dict[int, str],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Menyiapkan satu model deployment."""

    specification = MODEL_SPECS[
        experiment_name
    ]

    source_path = find_model_file(
        experiment_name
    )

    deployment_path = (
        DEPLOYMENT_DIR
        / specification[
            "deployment_filename"
        ]
    )

    source_model: tf.keras.Model | None = None
    deployed_model: tf.keras.Model | None = None

    try:
        source_model = tf.keras.models.load_model(
            source_path,
            compile=False,
            custom_objects=CUSTOM_OBJECTS,
        )

        source_architecture = validate_model_architecture(
            model=source_model,
            experiment_name=experiment_name,
            vocabulary_size=vocabulary_size,
        )

        source_probabilities = predict_and_validate(
            model=source_model,
            X=smoke_batch["X"],
            context=f"source {experiment_name}",
        )

        if deployment_path.exists():
            deployment_path.unlink()

        # Menyimpan ulang model agar format deployment selalu .keras.
        source_model.save(
            deployment_path
        )

        deployed_model = tf.keras.models.load_model(
            deployment_path,
            compile=False,
            custom_objects=CUSTOM_OBJECTS,
        )

        deployed_architecture = validate_model_architecture(
            model=deployed_model,
            experiment_name=experiment_name,
            vocabulary_size=vocabulary_size,
        )

        if deployed_architecture != source_architecture:
            raise ValueError(
                "Arsitektur model berubah setelah disimpan ulang."
            )

        deployed_probabilities = predict_and_validate(
            model=deployed_model,
            X=smoke_batch["X"],
            context=f"deployment {experiment_name}",
        )

        maximum_prediction_difference = float(
            np.max(
                np.abs(
                    source_probabilities
                    - deployed_probabilities
                )
            )
        )

        if not np.allclose(
            source_probabilities,
            deployed_probabilities,
            atol=PREDICTION_ATOL,
            rtol=0.0,
        ):
            raise ValueError(
                "Prediksi model deployment berbeda dengan model sumber.\n"
                f"Maximum difference: "
                f"{maximum_prediction_difference:.10f}"
            )

        predicted_classes = np.argmax(
            deployed_probabilities,
            axis=1,
        )

        smoke_test_results: list[dict[str, Any]] = []

        for position, original_test_index in enumerate(
            smoke_batch["indices"]
        ):
            actual_index = int(
                smoke_batch[
                    "y"
                ][position]
            )

            predicted_index = int(
                predicted_classes[
                    position
                ]
            )

            smoke_test_results.append(
                {
                    "original_test_index":
                        int(
                            original_test_index
                        ),

                    "document_id":
                        str(
                            smoke_batch[
                                "document_id"
                            ][position]
                        ),

                    "actual_index":
                        actual_index,

                    "actual_label":
                        index_to_label[
                            actual_index
                        ],

                    "predicted_index":
                        predicted_index,

                    "predicted_label":
                        index_to_label[
                            predicted_index
                        ],

                    "is_correct":
                        actual_index
                        == predicted_index,

                    "prediction_confidence":
                        float(
                            deployed_probabilities[
                                position,
                                predicted_index,
                            ]
                        ),

                    "probabilities":
                        deployed_probabilities[
                            position
                        ].tolist(),
                }
            )

        result = {
            "experiment_name":
                experiment_name,

            "display_name":
                specification[
                    "display_name"
                ],

            "model_family":
                specification[
                    "model_family"
                ],

            "status":
                "success",

            "source_model":
                get_file_metadata(
                    source_path
                ),

            "deployment_model":
                get_file_metadata(
                    deployment_path
                ),

            "architecture":
                source_architecture,

            "metrics":
                metrics,

            "smoke_test": {
                "status":
                    "success",

                "sample_count":
                    int(
                        len(
                            smoke_batch[
                                "X"
                            ]
                        )
                    ),

                "prediction_atol":
                    PREDICTION_ATOL,

                "maximum_source_deployment_difference":
                    maximum_prediction_difference,

                "samples":
                    smoke_test_results,
            },
        }

        return result

    except Exception:
        if deployment_path.exists():
            deployment_path.unlink()

        raise

    finally:
        if source_model is not None:
            del source_model

        if deployed_model is not None:
            del deployed_model

        tf.keras.backend.clear_session()
        gc.collect()


def prepare_models(
    vocabulary_size: int,
    smoke_batch: dict[str, Any],
    index_to_label: dict[int, str],
    metrics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Menyiapkan seluruh model deployment."""

    print(
        "\nMenyiapkan model deployment..."
    )

    results: list[dict[str, Any]] = []

    for experiment_name in MODEL_SPECS:
        print(
            "\n" + "-" * 80
        )

        print(
            f"Model                  : {experiment_name}"
        )

        try:
            result = prepare_single_model(
                experiment_name=experiment_name,
                vocabulary_size=vocabulary_size,
                smoke_batch=smoke_batch,
                index_to_label=index_to_label,
                metrics=metrics[
                    experiment_name
                ],
            )

            results.append(
                result
            )

            print(
                f"Sumber                 : "
                f"{result['source_model']['path']}"
            )

            print(
                f"Input                  : "
                f"{result['architecture']['input_shape']}"
            )

            print(
                f"Output                 : "
                f"{result['architecture']['output_shape']}"
            )

            print(
                f"Parameter              : "
                f"{result['architecture']['parameter_count']:,}"
            )

            print(
                f"Accuracy test          : "
                f"{float(result['metrics']['accuracy']):.6f}"
            )

            print(
                f"Macro F1 test          : "
                f"{float(result['metrics']['f1_macro']):.6f}"
            )

            print(
                f"Max prediction diff    : "
                f"{result['smoke_test']['maximum_source_deployment_difference']:.10f}"
            )

            print(
                f"Output                 : "
                f"{result['deployment_model']['path']}"
            )

            print(
                "Status                 : BERHASIL"
            )

        except Exception as error:
            results.append(
                {
                    "experiment_name":
                        experiment_name,

                    "display_name":
                        MODEL_SPECS[
                            experiment_name
                        ][
                            "display_name"
                        ],

                    "status":
                        "failed",

                    "error":
                        str(error),
                }
            )

            print(
                "Status                 : GAGAL"
            )

            print(
                f"Error                  : {error}"
            )

    return results


# =============================================================================
# DEPLOYMENT CONFIG
# =============================================================================

def build_deployment_config(
    index_to_label: dict[int, str],
    model_results: list[dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
    vocabulary_size: int,
    vectorizer_config: dict[str, Any],
) -> dict[str, Any]:
    """Membentuk deployment_config.json."""

    successful_models = {
        result[
            "experiment_name"
        ]: result

        for result
        in model_results

        if result.get(
            "status"
        ) == "success"
    }

    if PRIMARY_EXPERIMENT not in successful_models:
        raise RuntimeError(
            "Model utama CNN K2 tidak berhasil disiapkan."
        )

    model_configuration: dict[str, Any] = {}

    for experiment_name, specification in MODEL_SPECS.items():
        result = successful_models.get(
            experiment_name
        )

        experiment_metrics = metrics[
            experiment_name
        ]

        model_configuration[
            specification[
                "model_family"
            ]
        ] = {
            "experiment_name":
                experiment_name,

            "display_name":
                specification[
                    "display_name"
                ],

            "filename":
                specification[
                    "deployment_filename"
                ],

            "available":
                result is not None,

            "accuracy":
                float(
                    experiment_metrics[
                        "accuracy"
                    ]
                ),

            "precision_macro":
                (
                    float(
                        experiment_metrics[
                            "precision_macro"
                        ]
                    )
                    if experiment_metrics.get(
                        "precision_macro"
                    ) is not None
                    else None
                ),

            "recall_macro":
                (
                    float(
                        experiment_metrics[
                            "recall_macro"
                        ]
                    )
                    if experiment_metrics.get(
                        "recall_macro"
                    ) is not None
                    else None
                ),

            "f1_macro":
                float(
                    experiment_metrics[
                        "f1_macro"
                    ]
                ),

            "log_loss":
                (
                    float(
                        experiment_metrics[
                            "log_loss"
                        ]
                    )
                    if experiment_metrics.get(
                        "log_loss"
                    ) is not None
                    else None
                ),

            "inference_time_seconds":
                (
                    float(
                        experiment_metrics[
                            "inference_time_seconds"
                        ]
                    )
                    if experiment_metrics.get(
                        "inference_time_seconds"
                    ) is not None
                    else None
                ),

            "average_inference_ms_per_sample":
                (
                    float(
                        experiment_metrics[
                            "average_inference_ms_per_sample"
                        ]
                    )
                    if experiment_metrics.get(
                        "average_inference_ms_per_sample"
                    ) is not None
                    else None
                ),

            "sha256":
                (
                    result[
                        "deployment_model"
                    ][
                        "sha256"
                    ]
                    if result is not None
                    else None
                ),
        }

    primary_metrics = metrics[
        PRIMARY_EXPERIMENT
    ]

    return {
        "generated_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "project":
            "TA Klasifikasi Dokumen Berita",

        "dataset":
            DATASET_NAME,

        "scenario_code":
            SCENARIO_CODE,

        "scenario_name":
            SCENARIO_NAME,

        "sequence_length":
            SEQUENCE_LENGTH,

        "num_classes":
            NUM_CLASSES,

        "vocabulary_size":
            vocabulary_size,

        "models":
            model_configuration,

        "best_research_model": {
            "experiment_name":
                PRIMARY_EXPERIMENT,

            "model_name":
                MODEL_SPECS[
                    PRIMARY_EXPERIMENT
                ][
                    "display_name"
                ],

            "accuracy":
                float(
                    primary_metrics[
                        "accuracy"
                    ]
                ),

            "f1_macro":
                float(
                    primary_metrics[
                        "f1_macro"
                    ]
                ),

            "log_loss":
                (
                    float(
                        primary_metrics[
                            "log_loss"
                        ]
                    )
                    if primary_metrics.get(
                        "log_loss"
                    ) is not None
                    else None
                ),
        },

        "input": {
            "required_fields": [
                "title",
                "description",
            ],

            "separator_token":
                "[SEP]",

            "text_combination":
                "title + [SEP] + description",

            "validation":
                "Minimal salah satu field harus berisi teks.",
        },

        "labels": {
            "index_to_label": {
                str(index): label
                for index, label
                in index_to_label.items()
            },

            "label_to_index": {
                label: index
                for index, label
                in index_to_label.items()
            },
        },

        "artifacts": {
            "vocabulary":
                VOCABULARY_OUTPUT_PATH.name,

            "vectorizer_config":
                VECTORIZER_CONFIG_OUTPUT_PATH.name,

            "label_mapping":
                LABEL_MAPPING_OUTPUT_PATH.name,
        },

        "vectorizer_configuration_source":
            vectorizer_config,
    }


# =============================================================================
# DEPLOYMENT STATUS DAN REPORT
# =============================================================================

def determine_deployment_status(
    model_results: list[dict[str, Any]],
    supporting_artifacts: dict[str, dict[str, Any]],
) -> str:
    """Menentukan status akhir deployment."""

    model_statuses = {
        result[
            "experiment_name"
        ]: result.get(
            "status"
        )

        for result
        in model_results
    }

    primary_success = (
        model_statuses.get(
            PRIMARY_EXPERIMENT
        )
        == "success"
    )

    comparison_success = (
        model_statuses.get(
            COMPARISON_EXPERIMENT
        )
        == "success"
    )

    supporting_success = all(
        artifact.get(
            "status"
        )
        == "success"

        for artifact
        in supporting_artifacts.values()
    )

    if (
        primary_success
        and comparison_success
        and supporting_success
    ):
        return "success"

    if (
        primary_success
        and supporting_success
    ):
        return "partial"

    return "failed"


def save_deployment_report(
    deployment_status: str,
    model_results: list[dict[str, Any]],
    supporting_artifacts: dict[str, dict[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    """Menyimpan deployment_report.json."""

    report = {
        "generated_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "deployment_status":
            deployment_status,

        "primary_model":
            PRIMARY_EXPERIMENT,

        "comparison_model":
            COMPARISON_EXPERIMENT,

        "dataset":
            DATASET_NAME,

        "scenario":
            SCENARIO_CODE,

        "representation":
            SCENARIO_NAME,

        "environment": {
            "python_version":
                platform.python_version(),

            "platform":
                platform.platform(),

            "tensorflow_version":
                tf.__version__,

            "keras_version":
                getattr(
                    tf.keras,
                    "__version__",
                    None,
                ),

            "numpy_version":
                np.__version__,

            "pandas_version":
                pd.__version__,
        },

        "models":
            model_results,

        "supporting_artifacts":
            supporting_artifacts,

        "errors":
            errors,
    }

    write_json(
        DEPLOYMENT_REPORT_PATH,
        report,
    )

    return report


# =============================================================================
# PROGRAM UTAMA
# =============================================================================

def main() -> None:
    """Menyiapkan seluruh artefak deployment."""

    print_header(
        "STEP 8.1 - PREPARE DEPLOYMENT MODELS"
    )

    clean_old_outputs()

    print(
        "\nKonfigurasi deployment:"
    )

    print(
        f"Dataset                : {DATASET_NAME}"
    )

    print(
        f"Skenario               : {SCENARIO_CODE}"
    )

    print(
        f"Representasi           : {SCENARIO_NAME}"
    )

    print(
        f"Sequence length        : {SEQUENCE_LENGTH}"
    )

    print(
        f"Jumlah kelas           : {NUM_CLASSES}"
    )

    print(
        f"Model utama            : {PRIMARY_EXPERIMENT}"
    )

    print(
        f"Model pembanding       : {COMPARISON_EXPERIMENT}"
    )

    supporting_artifacts: dict[
        str,
        dict[str, Any]
    ] = {}

    model_results: list[
        dict[str, Any]
    ] = []

    errors: list[str] = []

    test_data: dict[str, np.ndarray] | None = None
    smoke_batch: dict[str, Any] | None = None
    vocabulary: list[str] | None = None
    vectorizer_config: dict[str, Any] | None = None
    index_to_label: dict[int, str] | None = None
    metrics: dict[str, dict[str, Any]] | None = None

    # =========================================================================
    # TEST SET
    # =========================================================================

    print(
        "\n" + "-" * 80
    )

    print(
        "Memuat test set untuk smoke test..."
    )

    try:
        test_data = load_test_data()

        smoke_batch = select_smoke_test_samples(
            test_data
        )

        supporting_artifacts[
            "test_data"
        ] = {
            "status":
                "success",

            "path":
                str(
                    TEST_DATA_PATH
                ),

            "shape":
                list(
                    test_data[
                        "X"
                    ].shape
                ),

            "maximum_token_id":
                int(
                    np.max(
                        test_data[
                            "X"
                        ]
                    )
                ),

            "smoke_test_indices":
                smoke_batch[
                    "indices"
                ].tolist(),

            "smoke_test_document_ids":
                smoke_batch[
                    "document_id"
                ].tolist(),
        }

        print(
            f"Test shape             : {test_data['X'].shape}"
        )

        print(
            f"Smoke test samples     : {len(smoke_batch['X'])}"
        )

        print(
            "Status                 : BERHASIL"
        )

    except Exception as error:
        errors.append(
            f"test_data: {error}"
        )

        supporting_artifacts[
            "test_data"
        ] = {
            "status":
                "failed",

            "error":
                str(error),
        }

        print(
            "Status                 : GAGAL"
        )

        print(
            f"Error                  : {error}"
        )

    # =========================================================================
    # VOCABULARY
    # =========================================================================

    print(
        "\n" + "-" * 80
    )

    print(
        "Menyiapkan vocabulary..."
    )

    try:
        vocabulary = load_vocabulary()

        if test_data is not None:
            maximum_token_id = int(
                np.max(
                    test_data[
                        "X"
                    ]
                )
            )

            if maximum_token_id >= len(vocabulary):
                raise ValueError(
                    "Maximum token ID melebihi vocabulary size.\n"
                    f"Maximum token ID : {maximum_token_id}\n"
                    f"Vocabulary size  : {len(vocabulary)}"
                )

        vocabulary_result = copy_verified(
            VOCABULARY_SOURCE_PATH,
            VOCABULARY_OUTPUT_PATH,
        )

        vocabulary_result[
            "vocabulary_size"
        ] = len(
            vocabulary
        )

        supporting_artifacts[
            "vocabulary"
        ] = vocabulary_result

        print(
            f"Vocabulary size        : {len(vocabulary):,}"
        )

        print(
            f"Output                 : {VOCABULARY_OUTPUT_PATH}"
        )

        print(
            "Status                 : BERHASIL"
        )

    except Exception as error:
        errors.append(
            f"vocabulary: {error}"
        )

        supporting_artifacts[
            "vocabulary"
        ] = {
            "status":
                "failed",

            "error":
                str(error),
        }

        print(
            "Status                 : GAGAL"
        )

        print(
            f"Error                  : {error}"
        )

    # =========================================================================
    # VECTORIZER CONFIG
    # =========================================================================

    print(
        "\n" + "-" * 80
    )

    print(
        "Menyiapkan vectorizer config..."
    )

    try:
        if vocabulary is None:
            raise RuntimeError(
                "Vocabulary belum tersedia."
            )

        vectorizer_config = load_vectorizer_config(
            vocabulary_size=len(
                vocabulary
            ),
        )

        vectorizer_result = copy_verified(
            VECTORIZER_CONFIG_SOURCE_PATH,
            VECTORIZER_CONFIG_OUTPUT_PATH,
        )

        vectorizer_result[
            "validated_sequence_length"
        ] = SEQUENCE_LENGTH

        supporting_artifacts[
            "vectorizer_config"
        ] = vectorizer_result

        print(
            f"Validated sequence     : {SEQUENCE_LENGTH}"
        )

        print(
            f"Output                 : {VECTORIZER_CONFIG_OUTPUT_PATH}"
        )

        print(
            "Status                 : BERHASIL"
        )

    except Exception as error:
        errors.append(
            f"vectorizer_config: {error}"
        )

        supporting_artifacts[
            "vectorizer_config"
        ] = {
            "status":
                "failed",

            "error":
                str(error),
        }

        print(
            "Status                 : GAGAL"
        )

        print(
            f"Error                  : {error}"
        )

    # =========================================================================
    # LABEL MAPPING
    # =========================================================================

    print(
        "\n" + "-" * 80
    )

    print(
        "Menyiapkan label mapping..."
    )

    try:
        index_to_label = load_label_mapping()

        supporting_artifacts[
            "label_mapping"
        ] = save_deployment_label_mapping(
            index_to_label
        )

        print(
            f"Label mapping          : {index_to_label}"
        )

        print(
            f"Output                 : {LABEL_MAPPING_OUTPUT_PATH}"
        )

        print(
            "Status                 : BERHASIL"
        )

    except Exception as error:
        errors.append(
            f"label_mapping: {error}"
        )

        supporting_artifacts[
            "label_mapping"
        ] = {
            "status":
                "failed",

            "error":
                str(error),
        }

        print(
            "Status                 : GAGAL"
        )

        print(
            f"Error                  : {error}"
        )

    # =========================================================================
    # MODEL METRICS
    # =========================================================================

    print(
        "\n" + "-" * 80
    )

    print(
        "Memuat metrik test set..."
    )

    try:
        metrics = load_model_metrics()

        supporting_artifacts[
            "model_metrics"
        ] = {
            "status":
                "success",

            "source":
                get_file_metadata(
                    MODEL_METRICS_PATH
                ),

            "metrics":
                metrics,
        }

        for experiment_name, values in metrics.items():
            print(
                f"{experiment_name:<24}: "
                f"accuracy={float(values['accuracy']):.6f}, "
                f"f1_macro={float(values['f1_macro']):.6f}"
            )

        print(
            "Status                 : BERHASIL"
        )

    except Exception as error:
        errors.append(
            f"model_metrics: {error}"
        )

        supporting_artifacts[
            "model_metrics"
        ] = {
            "status":
                "failed",

            "error":
                str(error),
        }

        print(
            "Status                 : GAGAL"
        )

        print(
            f"Error                  : {error}"
        )

    # =========================================================================
    # MODEL DEPLOYMENT
    # =========================================================================

    model_dependencies_ready = all(
        value is not None
        for value in [
            vocabulary,
            smoke_batch,
            index_to_label,
            metrics,
        ]
    )

    if model_dependencies_ready:
        assert vocabulary is not None
        assert smoke_batch is not None
        assert index_to_label is not None
        assert metrics is not None

        model_results = prepare_models(
            vocabulary_size=len(
                vocabulary
            ),
            smoke_batch=smoke_batch,
            index_to_label=index_to_label,
            metrics=metrics,
        )

    else:
        dependency_error = (
            "Model tidak diproses karena dependency "
            "deployment belum lengkap."
        )

        errors.append(
            dependency_error
        )

        model_results = [
            {
                "experiment_name":
                    experiment_name,

                "display_name":
                    specification[
                        "display_name"
                    ],

                "status":
                    "not_processed",

                "error":
                    dependency_error,
            }

            for experiment_name, specification
            in MODEL_SPECS.items()
        ]

    for result in model_results:
        if result.get("status") != "success":
            errors.append(
                f"model {result['experiment_name']}: "
                f"{result.get('error', 'unknown error')}"
            )

    # =========================================================================
    # DEPLOYMENT CONFIG
    # =========================================================================

    print(
        "\n" + "-" * 80
    )

    print(
        "Menyimpan deployment config..."
    )

    try:
        config_dependencies_ready = all(
            value is not None
            for value in [
                index_to_label,
                metrics,
                vocabulary,
                vectorizer_config,
            ]
        )

        if not config_dependencies_ready:
            raise RuntimeError(
                "Dependency deployment config belum lengkap."
            )

        assert index_to_label is not None
        assert metrics is not None
        assert vocabulary is not None
        assert vectorizer_config is not None

        deployment_config = build_deployment_config(
            index_to_label=index_to_label,
            model_results=model_results,
            metrics=metrics,
            vocabulary_size=len(
                vocabulary
            ),
            vectorizer_config=vectorizer_config,
        )

        write_json(
            DEPLOYMENT_CONFIG_PATH,
            deployment_config,
        )

        supporting_artifacts[
            "deployment_config"
        ] = {
            "status":
                "success",

            "deployment":
                get_file_metadata(
                    DEPLOYMENT_CONFIG_PATH
                ),
        }

        print(
            f"Output                 : {DEPLOYMENT_CONFIG_PATH}"
        )

        print(
            "Status                 : BERHASIL"
        )

    except Exception as error:
        errors.append(
            f"deployment_config: {error}"
        )

        supporting_artifacts[
            "deployment_config"
        ] = {
            "status":
                "failed",

            "error":
                str(error),
        }

        print(
            "Status                 : GAGAL"
        )

        print(
            f"Error                  : {error}"
        )

    # =========================================================================
    # DEPLOYMENT REPORT
    # =========================================================================

    deployment_status = determine_deployment_status(
        model_results=model_results,
        supporting_artifacts=supporting_artifacts,
    )

    deployment_report = save_deployment_report(
        deployment_status=deployment_status,
        model_results=model_results,
        supporting_artifacts=supporting_artifacts,
        errors=errors,
    )

    # =========================================================================
    # RINGKASAN
    # =========================================================================

    successful_model_count = sum(
        result.get(
            "status"
        )
        == "success"

        for result
        in model_results
    )

    print("\n")

    print_header(
        "HASIL PREPARE DEPLOYMENT"
    )

    print(
        f"\nDeployment status      : {deployment_status.upper()}"
    )

    print(
        f"Model berhasil         : "
        f"{successful_model_count}/{len(MODEL_SPECS)}"
    )

    print(
        f"Jumlah error           : {len(errors)}"
    )

    print(
        "\nFolder deployment:"
    )

    print(
        DEPLOYMENT_DIR
    )

    print(
        "\nDeployment config:"
    )

    print(
        DEPLOYMENT_CONFIG_PATH
    )

    print(
        "\nDeployment report:"
    )

    print(
        DEPLOYMENT_REPORT_PATH
    )

    if errors:
        print(
            "\nDaftar error:"
        )

        for error in errors:
            print(
                f"- {error}"
            )

    print("\n")

    print_header(
        "Tahap prepare deployment selesai."
    )

    if (
        deployment_report[
            "deployment_status"
        ]
        != "success"
    ):
        raise RuntimeError(
            "Deployment belum siap sepenuhnya. "
            "Periksa deployment_report.json."
        )


if __name__ == "__main__":
    main()
# =============================================================================
# STEP 7.1 - GLOBAL SHAP EXPLANATION
# =============================================================================
# File:
# 7_explainability/01_shap_global.py
#
# Tujuan:
# Menjelaskan model terbaik CNN K2 secara global menggunakan SHAP.
#
# Model:
# - CNN
# - Dataset Kompas
# - Skenario K2: Title + Description
#
# Metode:
# - SHAP KernelExplainer
# - Fitur yang dijelaskan adalah posisi token.
# - Token dipertahankan ketika mask = 1.
# - Token dihapus atau diubah menjadi padding ketika mask = 0.
# - Prediksi selalu dilakukan menggunakan model CNN K2 asli.
#
# Alasan:
# Model CNN final menggunakan:
# - ZeroPaddingEmbeddingOutput
# - MaskedGlobalMaxPooling1D
#
# Karena itu, model tidak dipisahkan menjadi embedding model dan tail model.
# Pendekatan token masking mempertahankan arsitektur serta mekanisme padding
# yang digunakan oleh model asli.
#
# Global importance:
# - Nilai absolut SHAP dihitung untuk setiap output kelas.
# - Nilai setiap posisi token diagregasi pada seluruh kelas.
# - Token yang sama kemudian digabungkan pada seluruh sampel.
#
# Importance per kelas:
# - Nilai SHAP dihitung terhadap setiap output kelas.
# - Tabel per kelas tidak hanya menggunakan predicted class.
#
# Baseline:
# - Seluruh token yang dijelaskan diubah menjadi padding.
#
# Output:
# - Global token importance
# - Token importance per output class
# - Ringkasan sampel
# - Grafik token global
# - Grafik token per kelas
# - Array SHAP
# - Konfigurasi dan diagnostik SHAP
# =============================================================================

from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# Mengurangi pesan informasi TensorFlow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
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
# IMPORT CUSTOM LAYER
# =============================================================================

from cnn_model import (  # noqa: E402
    MaskedGlobalMaxPooling1D,
    ZeroPaddingEmbeddingOutput,
)


# =============================================================================
# KONFIGURASI ANALISIS
# =============================================================================

RANDOM_SEED = 42

EXPERIMENT_NAME = "cnn_k2"
DATASET_NAME = "Kompas"
SCENARIO_CODE = "K2"
SCENARIO_NAME = "Title + Description"

MAX_SEQUENCE_LENGTH = 60
NUM_CLASSES = 4

# Jumlah sampel test yang dijelaskan.
EXPLAIN_SIZE = 100

# Jumlah evaluasi pendekatan Kernel SHAP.
#
# Untuk sequence dengan maksimal 60 token, nilai 100 merupakan
# konfigurasi awal yang cukup ringan. Nilai yang lebih besar
# meningkatkan stabilitas tetapi memperpanjang waktu proses.
SHAP_NSAMPLES = 100

# Batch prediksi ketika KernelExplainer mengirim banyak kombinasi mask.
PREDICTION_BATCH_SIZE = 256

TOP_N_TOKENS = 20
TOP_N_TOKENS_PER_CLASS = 10

# Toleransi pemeriksaan jumlah probabilitas.
PROBABILITY_SUM_TOLERANCE = 1e-4

# Toleransi pemeriksaan local accuracy/additivity SHAP.
ADDITIVITY_TOLERANCE = 2e-2

# Token teknis tetap tersedia pada tabel mentah,
# tetapi tidak ditampilkan pada visualisasi semantik.
#
# "SEP" ditambahkan karena proses standardisasi terkadang
# mengubah [SEP] menjadi sep.
SPECIAL_TOKENS = {
    "",
    "[PAD]",
    "PAD",
    "[SEP]",
    "SEP",
    "[UNK]",
    "UNK",
    "[OOV]",
    "OOV",
}


# =============================================================================
# PATH INPUT
# =============================================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "8_save_models"
    / "checkpoints"
    / "cnn_k2_best.keras"
)

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "2_data"
    / "vectorized"
    / "kompas_k2"
    / "test.npz"
)

VOCABULARY_PATH = (
    PROJECT_ROOT
    / "2_data"
    / "vectorized"
    / "kompas_k2"
    / "vocabulary.txt"
)

LABEL_MAPPING_PATH = (
    PROJECT_ROOT
    / "9_results"
    / "tables"
    / "label_mapping.json"
)


# =============================================================================
# PATH OUTPUT
# =============================================================================

RESULTS_DIR = (
    PROJECT_ROOT
    / "9_results"
)

SHAP_TABLES_DIR = (
    RESULTS_DIR
    / "tables"
    / "shap"
)

SHAP_FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
    / "shap"
    / "global"
)

SHAP_ARRAYS_DIR = (
    RESULTS_DIR
    / "shap_values"
)

GLOBAL_IMPORTANCE_PATH = (
    SHAP_TABLES_DIR
    / "cnn_k2_global_token_importance.csv"
)

CLASS_IMPORTANCE_PATH = (
    SHAP_TABLES_DIR
    / "cnn_k2_global_token_importance_by_class.csv"
)

SAMPLE_SUMMARY_PATH = (
    SHAP_TABLES_DIR
    / "cnn_k2_shap_sample_summary.csv"
)

TOKEN_DETAIL_PATH = (
    SHAP_TABLES_DIR
    / "cnn_k2_shap_token_detail.csv"
)

GLOBAL_FIGURE_PATH = (
    SHAP_FIGURES_DIR
    / "cnn_k2_global_shap_top_tokens.png"
)

CLASS_FIGURE_PATH = (
    SHAP_FIGURES_DIR
    / "cnn_k2_global_shap_by_class.png"
)

SHAP_VALUES_PATH = (
    SHAP_ARRAYS_DIR
    / "cnn_k2_global_shap_values.npz"
)

CONFIGURATION_PATH = (
    SHAP_TABLES_DIR
    / "cnn_k2_global_shap_configuration.json"
)

DIAGNOSTIC_PATH = (
    SHAP_TABLES_DIR
    / "cnn_k2_global_shap_diagnostics.csv"
)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_random_seed(
    seed: int = RANDOM_SEED,
) -> None:
    """
    Mengatur random seed agar pemilihan sampel dapat direproduksi.
    """

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    tf.random.set_seed(
        seed
    )


# =============================================================================
# MEMBUAT FOLDER OUTPUT
# =============================================================================

def create_output_directories() -> None:
    """
    Membuat seluruh folder output analisis SHAP.
    """

    directories = [
        SHAP_TABLES_DIR,
        SHAP_FIGURES_DIR,
        SHAP_ARRAYS_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# =============================================================================
# MEMUAT DATA NPZ
# =============================================================================

def load_npz_dataset(
    file_path: Path,
) -> dict[str, np.ndarray]:
    """
    Memuat X, y, document_id, dan category dari file NPZ.
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
            "File dataset ditemukan, tetapi kosong:\n"
            f"{file_path}"
        )

    with np.load(
        file_path,
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
                "Komponen dataset NPZ tidak lengkap.\n"
                f"Key hilang: {sorted(missing_keys)}"
            )

        result = {
            "X": np.asarray(
                data["X"],
                dtype=np.int32,
            ),

            "y": np.asarray(
                data["y"],
                dtype=np.int32,
            ),

            "document_id": np.asarray(
                data["document_id"],
                dtype=str,
            ),

            "category": np.asarray(
                data["category"],
                dtype=str,
            ),
        }

    if result["X"].ndim != 2:
        raise ValueError(
            "X harus dua dimensi.\n"
            f"Shape ditemukan: {result['X'].shape}"
        )

    if result["y"].ndim != 1:
        raise ValueError(
            "y harus satu dimensi.\n"
            f"Shape ditemukan: {result['y'].shape}"
        )

    component_lengths = {
        "X": len(
            result["X"]
        ),
        "y": len(
            result["y"]
        ),
        "document_id": len(
            result["document_id"]
        ),
        "category": len(
            result["category"]
        ),
    }

    if len(
        set(
            component_lengths.values()
        )
    ) != 1:
        raise ValueError(
            "Jumlah data setiap komponen NPZ tidak sama.\n"
            f"{component_lengths}"
        )

    if (
        result["X"].shape[1]
        != MAX_SEQUENCE_LENGTH
    ):
        raise ValueError(
            "Panjang sequence tidak sesuai.\n"
            f"Expected: {MAX_SEQUENCE_LENGTH}\n"
            f"Actual  : {result['X'].shape[1]}"
        )

    if (
        result["X"]
        < 0
    ).any():
        raise ValueError(
            "Ditemukan token ID negatif."
        )

    unique_labels = set(
        np.unique(
            result["y"]
        ).tolist()
    )

    expected_labels = set(
        range(
            NUM_CLASSES
        )
    )

    if unique_labels != expected_labels:
        raise ValueError(
            "Label test set tidak lengkap atau tidak sesuai.\n"
            f"Expected: {sorted(expected_labels)}\n"
            f"Actual  : {sorted(unique_labels)}"
        )

    return result


# =============================================================================
# MEMUAT VOCABULARY
# =============================================================================

def load_vocabulary(
    vocabulary_path: Path,
) -> list[str]:
    """
    Membaca vocabulary berdasarkan urutan indeks TextVectorization.

    Baris kosong pertama tetap dipertahankan karena indeks 0
    digunakan sebagai padding.
    """

    if not vocabulary_path.exists():
        raise FileNotFoundError(
            "Vocabulary tidak ditemukan:\n"
            f"{vocabulary_path}"
        )

    if not vocabulary_path.is_file():
        raise ValueError(
            "Path vocabulary bukan file:\n"
            f"{vocabulary_path}"
        )

    with open(
        vocabulary_path,
        "r",
        encoding="utf-8",
    ) as file:
        vocabulary = (
            file.read()
            .splitlines()
        )

    if not vocabulary:
        raise ValueError(
            "File vocabulary kosong."
        )

    if len(
        vocabulary
    ) <= 2:
        raise ValueError(
            "Ukuran vocabulary tidak valid.\n"
            f"Vocabulary size: {len(vocabulary)}"
        )

    return vocabulary


def token_id_to_word(
    token_id: int,
    vocabulary: list[str],
) -> str:
    """
    Mengubah indeks token menjadi teks.
    """

    if token_id == 0:
        return "[PAD]"

    if (
        0
        <= token_id
        < len(vocabulary)
    ):
        token = (
            vocabulary[
                token_id
            ]
            .strip()
        )

        if not token:
            return "[PAD]"

        return token

    return "[OOV]"


def is_special_token(
    token: str,
) -> bool:
    """
    Menentukan apakah token merupakan token teknis.
    """

    normalized_token = (
        str(token)
        .strip()
        .upper()
    )

    return (
        normalized_token
        in SPECIAL_TOKENS
    )


# =============================================================================
# MEMUAT LABEL MAPPING
# =============================================================================

def load_index_to_label() -> dict[int, str]:
    """
    Membaca mapping indeks kelas Kompas.
    """

    if not LABEL_MAPPING_PATH.exists():
        raise FileNotFoundError(
            "Label mapping tidak ditemukan:\n"
            f"{LABEL_MAPPING_PATH}"
        )

    with open(
        LABEL_MAPPING_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        mapping_data = json.load(
            file
        )

    kompas_mapping = mapping_data.get(
        "Kompas",
        mapping_data.get(
            "kompas",
            {},
        ),
    )

    if not kompas_mapping:
        raise KeyError(
            "Mapping label Kompas tidak ditemukan."
        )

    index_to_label: dict[int, str]

    if (
        "index_to_label"
        in kompas_mapping
    ):
        raw_mapping = kompas_mapping[
            "index_to_label"
        ]

        index_to_label = {
            int(index): str(label)
            for index, label
            in raw_mapping.items()
        }

    elif (
        "label_to_index"
        in kompas_mapping
    ):
        raw_mapping = kompas_mapping[
            "label_to_index"
        ]

        index_to_label = {
            int(index): str(label)
            for label, index
            in raw_mapping.items()
        }

    elif all(
        isinstance(
            value,
            int,
        )
        for value in kompas_mapping.values()
    ):
        index_to_label = {
            int(index): str(label)
            for label, index
            in kompas_mapping.items()
        }

    else:
        raise KeyError(
            "Format mapping label Kompas tidak dikenali."
        )

    expected_indices = set(
        range(
            NUM_CLASSES
        )
    )

    actual_indices = set(
        index_to_label.keys()
    )

    if actual_indices != expected_indices:
        raise ValueError(
            "Indeks label Kompas tidak sesuai.\n"
            f"Expected: {sorted(expected_indices)}\n"
            f"Actual  : {sorted(actual_indices)}"
        )

    return dict(
        sorted(
            index_to_label.items()
        )
    )


# =============================================================================
# MEMUAT MODEL CNN
# =============================================================================

def load_cnn_model() -> tf.keras.Model:
    """
    Memuat checkpoint terbaik CNN K2 beserta custom layer.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Checkpoint CNN K2 tidak ditemukan:\n"
            f"{MODEL_PATH}"
        )

    if not MODEL_PATH.is_file():
        raise ValueError(
            "Path checkpoint bukan file:\n"
            f"{MODEL_PATH}"
        )

    if MODEL_PATH.stat().st_size <= 0:
        raise ValueError(
            "Checkpoint CNN K2 ditemukan, tetapi kosong:\n"
            f"{MODEL_PATH}"
        )

    custom_objects = {
        "ZeroPaddingEmbeddingOutput":
            ZeroPaddingEmbeddingOutput,

        "MaskedGlobalMaxPooling1D":
            MaskedGlobalMaxPooling1D,
    }

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects=custom_objects,
        compile=False,
    )

    if isinstance(
        model.input_shape,
        list,
    ):
        raise ValueError(
            "Model CNN diharapkan memiliki satu input."
        )

    if (
        len(
            model.input_shape
        )
        != 2
    ):
        raise ValueError(
            "Shape input model tidak sesuai.\n"
            f"Model input: {model.input_shape}"
        )

    if (
        model.input_shape[1]
        != MAX_SEQUENCE_LENGTH
    ):
        raise ValueError(
            "Input model tidak sesuai.\n"
            f"Expected sequence: {MAX_SEQUENCE_LENGTH}\n"
            f"Model input      : {model.input_shape}"
        )

    if isinstance(
        model.output_shape,
        list,
    ):
        raise ValueError(
            "Model CNN diharapkan memiliki satu output."
        )

    if (
        model.output_shape[-1]
        != NUM_CLASSES
    ):
        raise ValueError(
            "Jumlah output kelas model tidak sesuai.\n"
            f"Expected: {NUM_CLASSES}\n"
            f"Actual  : {model.output_shape}"
        )

    has_zero_padding_layer = any(
        isinstance(
            layer,
            ZeroPaddingEmbeddingOutput,
        )
        for layer in model.layers
    )

    has_masked_pooling_layer = any(
        isinstance(
            layer,
            MaskedGlobalMaxPooling1D,
        )
        for layer in model.layers
    )

    if not has_zero_padding_layer:
        raise ValueError(
            "ZeroPaddingEmbeddingOutput tidak ditemukan "
            "pada model CNN K2."
        )

    if not has_masked_pooling_layer:
        raise ValueError(
            "MaskedGlobalMaxPooling1D tidak ditemukan "
            "pada model CNN K2."
        )

    return model


# =============================================================================
# PREDIKSI MODEL
# =============================================================================

def validate_probabilities(
    probabilities: np.ndarray,
    expected_rows: int,
    context_name: str,
) -> None:
    """
    Memastikan output model berupa probabilitas yang valid.
    """

    expected_shape = (
        expected_rows,
        NUM_CLASSES,
    )

    if probabilities.shape != expected_shape:
        raise ValueError(
            f"{context_name}: shape probabilitas tidak sesuai.\n"
            f"Expected: {expected_shape}\n"
            f"Actual  : {probabilities.shape}"
        )

    if not np.all(
        np.isfinite(
            probabilities
        )
    ):
        raise ValueError(
            f"{context_name}: ditemukan probabilitas "
            "NaN atau infinite."
        )

    if (
        probabilities
        < -1e-7
    ).any():
        raise ValueError(
            f"{context_name}: ditemukan probabilitas negatif."
        )

    if (
        probabilities
        > 1.0 + 1e-7
    ).any():
        raise ValueError(
            f"{context_name}: ditemukan probabilitas melebihi 1."
        )

    probability_sums = probabilities.sum(
        axis=1
    )

    if not np.allclose(
        probability_sums,
        1.0,
        atol=PROBABILITY_SUM_TOLERANCE,
    ):
        raise ValueError(
            f"{context_name}: jumlah probabilitas "
            "tidak mendekati 1."
        )


def predict_probabilities(
    model: tf.keras.Model,
    sequences: np.ndarray,
    batch_size: int = PREDICTION_BATCH_SIZE,
    context_name: str = "Prediction",
) -> np.ndarray:
    """
    Menjalankan prediksi CNN dan memvalidasi probabilitas.
    """

    sequences = np.asarray(
        sequences,
        dtype=np.int32,
    )

    if sequences.ndim == 1:
        sequences = sequences.reshape(
            1,
            -1,
        )

    if sequences.ndim != 2:
        raise ValueError(
            f"{context_name}: input prediksi harus 2 dimensi.\n"
            f"Shape: {sequences.shape}"
        )

    if (
        sequences.shape[1]
        != MAX_SEQUENCE_LENGTH
    ):
        raise ValueError(
            f"{context_name}: sequence length tidak sesuai.\n"
            f"Expected: {MAX_SEQUENCE_LENGTH}\n"
            f"Actual  : {sequences.shape[1]}"
        )

    if (
        sequences
        < 0
    ).any():
        raise ValueError(
            f"{context_name}: ditemukan token ID negatif."
        )

    probabilities = model.predict(
        sequences,
        batch_size=batch_size,
        verbose=0,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    validate_probabilities(
        probabilities=probabilities,
        expected_rows=len(
            sequences
        ),
        context_name=context_name,
    )

    return probabilities


# =============================================================================
# PEMILIHAN SAMPEL
# =============================================================================

def stratified_sample_indices(
    labels: np.ndarray,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    """
    Memilih sampel secara seimbang dari setiap kelas.
    """

    labels = np.asarray(
        labels,
        dtype=np.int32,
    )

    if labels.ndim != 1:
        raise ValueError(
            "Labels harus satu dimensi."
        )

    if sample_size <= 0:
        raise ValueError(
            "sample_size harus lebih besar dari nol."
        )

    if sample_size > len(
        labels
    ):
        raise ValueError(
            "sample_size melebihi jumlah data.\n"
            f"Sample size : {sample_size}\n"
            f"Jumlah data : {len(labels)}"
        )

    rng = np.random.default_rng(
        seed
    )

    unique_labels = np.unique(
        labels
    )

    base_per_class = (
        sample_size
        // len(
            unique_labels
        )
    )

    remainder = (
        sample_size
        % len(
            unique_labels
        )
    )

    selected_indices: list[int] = []

    for position, label in enumerate(
        unique_labels
    ):
        class_indices = np.where(
            labels == label
        )[0]

        class_sample_size = (
            base_per_class
            + (
                1
                if position < remainder
                else 0
            )
        )

        if (
            class_sample_size
            > len(
                class_indices
            )
        ):
            raise ValueError(
                "Jumlah data kelas tidak mencukupi "
                "untuk stratified sampling.\n"
                f"Label             : {label}\n"
                f"Diminta           : {class_sample_size}\n"
                f"Data tersedia     : {len(class_indices)}"
            )

        sampled_indices = rng.choice(
            class_indices,
            size=class_sample_size,
            replace=False,
        )

        selected_indices.extend(
            sampled_indices.tolist()
        )

    selected_array = np.asarray(
        selected_indices,
        dtype=np.int32,
    )

    rng.shuffle(
        selected_array
    )

    if (
        len(
            selected_array
        )
        != sample_size
    ):
        raise ValueError(
            "Jumlah hasil stratified sampling tidak sesuai.\n"
            f"Expected: {sample_size}\n"
            f"Actual  : {len(selected_array)}"
        )

    if (
        len(
            np.unique(
                selected_array
            )
        )
        != len(
            selected_array
        )
    ):
        raise ValueError(
            "Stratified sampling menghasilkan indeks duplikat."
        )

    return selected_array


# =============================================================================
# NORMALISASI OUTPUT KERNEL SHAP
# =============================================================================

def normalize_kernel_shap_output(
    shap_values: Any,
    num_features: int,
    num_classes: int,
) -> np.ndarray:
    """
    Mengubah berbagai kemungkinan output KernelExplainer menjadi:

    (num_features, num_classes)
    """

    if isinstance(
        shap_values,
        list,
    ):
        if (
            len(
                shap_values
            )
            != num_classes
        ):
            raise ValueError(
                "Jumlah output SHAP tidak sesuai jumlah kelas.\n"
                f"Expected: {num_classes}\n"
                f"Actual  : {len(shap_values)}"
            )

        class_arrays = []

        for class_index, value in enumerate(
            shap_values
        ):
            array = np.asarray(
                value,
                dtype=np.float64,
            )

            array = np.squeeze(
                array
            )

            if array.ndim != 1:
                raise ValueError(
                    "Output SHAP list tidak dapat dinormalisasi.\n"
                    f"Kelas: {class_index}\n"
                    f"Shape: {array.shape}"
                )

            if (
                len(
                    array
                )
                != num_features
            ):
                raise ValueError(
                    "Jumlah fitur SHAP tidak sesuai.\n"
                    f"Expected: {num_features}\n"
                    f"Actual  : {len(array)}"
                )

            class_arrays.append(
                array
            )

        result = np.stack(
            class_arrays,
            axis=1,
        )

    else:
        array = np.asarray(
            shap_values,
            dtype=np.float64,
        )

        # Format:
        # (1, num_features, num_classes)
        if (
            array.ndim == 3
            and array.shape[0] == 1
            and array.shape[1] == num_features
            and array.shape[2] == num_classes
        ):
            result = array[0]

        # Format:
        # (num_classes, 1, num_features)
        elif (
            array.ndim == 3
            and array.shape[0] == num_classes
            and array.shape[1] == 1
            and array.shape[2] == num_features
        ):
            result = (
                array[
                    :,
                    0,
                    :,
                ]
                .T
            )

        # Format:
        # (num_features, num_classes)
        elif (
            array.ndim == 2
            and array.shape
            == (
                num_features,
                num_classes,
            )
        ):
            result = array

        # Format:
        # (num_classes, num_features)
        elif (
            array.ndim == 2
            and array.shape
            == (
                num_classes,
                num_features,
            )
        ):
            result = array.T

        else:
            raise ValueError(
                "Format output Kernel SHAP tidak dikenali.\n"
                f"Shape ditemukan: {array.shape}"
            )

    expected_shape = (
        num_features,
        num_classes,
    )

    if (
        result.shape
        != expected_shape
    ):
        raise ValueError(
            "Shape SHAP setelah normalisasi tidak sesuai.\n"
            f"Expected: {expected_shape}\n"
            f"Actual  : {result.shape}"
        )

    if not np.all(
        np.isfinite(
            result
        )
    ):
        raise ValueError(
            "Ditemukan nilai SHAP NaN atau infinite."
        )

    return result


def normalize_expected_value(
    expected_value: Any,
    num_classes: int,
) -> np.ndarray:
    """
    Menyeragamkan expected value menjadi shape (num_classes,).
    """

    expected_array = np.asarray(
        expected_value,
        dtype=np.float64,
    )

    expected_array = np.squeeze(
        expected_array
    )

    if expected_array.ndim == 0:
        expected_array = expected_array.reshape(
            1,
        )

    if expected_array.ndim != 1:
        raise ValueError(
            "Expected value SHAP tidak satu dimensi.\n"
            f"Shape: {expected_array.shape}"
        )

    if (
        len(
            expected_array
        )
        != num_classes
    ):
        raise ValueError(
            "Jumlah expected value tidak sesuai.\n"
            f"Expected: {num_classes}\n"
            f"Actual  : {len(expected_array)}"
        )

    if not np.all(
        np.isfinite(
            expected_array
        )
    ):
        raise ValueError(
            "Expected value SHAP mengandung "
            "NaN atau infinite."
        )

    return expected_array


# =============================================================================
# PREDICTION FUNCTION UNTUK MASK TOKEN
# =============================================================================

def build_token_mask_prediction_function(
    model: tf.keras.Model,
    original_sequence: np.ndarray,
    valid_positions: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Membentuk fungsi prediksi untuk KernelExplainer.

    Input fungsi:
    - matrix mask dengan shape:
      (jumlah_kombinasi, jumlah_token_nonpadding)

    Nilai mask:
    - 1: token dipertahankan;
    - 0: token diganti menjadi padding.
    """

    original_sequence = np.asarray(
        original_sequence,
        dtype=np.int32,
    )

    valid_positions = np.asarray(
        valid_positions,
        dtype=np.int32,
    )

    if original_sequence.shape != (
        MAX_SEQUENCE_LENGTH,
    ):
        raise ValueError(
            "Shape original sequence tidak sesuai.\n"
            f"Expected: {(MAX_SEQUENCE_LENGTH,)}\n"
            f"Actual  : {original_sequence.shape}"
        )

    if valid_positions.ndim != 1:
        raise ValueError(
            "valid_positions harus satu dimensi."
        )

    if len(
        valid_positions
    ) == 0:
        raise ValueError(
            "Sequence tidak memiliki token non-padding."
        )

    def predict_from_masks(
        masks: np.ndarray,
    ) -> np.ndarray:
        """
        Mengubah mask token menjadi sequence lalu menjalankan model.
        """

        masks_array = np.asarray(
            masks,
            dtype=np.float64,
        )

        if masks_array.ndim == 1:
            masks_array = masks_array.reshape(
                1,
                -1,
            )

        if masks_array.ndim != 2:
            raise ValueError(
                "Mask SHAP harus dua dimensi.\n"
                f"Shape: {masks_array.shape}"
            )

        if (
            masks_array.shape[1]
            != len(
                valid_positions
            )
        ):
            raise ValueError(
                "Jumlah fitur mask tidak sesuai.\n"
                f"Expected: {len(valid_positions)}\n"
                f"Actual  : {masks_array.shape[1]}"
            )

        # Kernel SHAP biasanya menghasilkan nilai 0 dan 1.
        # Threshold digunakan sebagai pengamanan kompatibilitas.
        keep_mask = (
            masks_array
            >= 0.5
        )

        masked_sequences = np.repeat(
            original_sequence[
                np.newaxis,
                :,
            ],
            repeats=len(
                masks_array
            ),
            axis=0,
        )

        for row_index in range(
            len(
                masked_sequences
            )
        ):
            positions_to_remove = valid_positions[
                ~keep_mask[
                    row_index
                ]
            ]

            masked_sequences[
                row_index,
                positions_to_remove,
            ] = 0

        return predict_probabilities(
            model=model,
            sequences=masked_sequences,
            batch_size=PREDICTION_BATCH_SIZE,
            context_name="Kernel SHAP prediction",
        )

    return predict_from_masks


# =============================================================================
# SHAP SATU SAMPEL
# =============================================================================

def explain_single_sequence(
    model: tf.keras.Model,
    sequence: np.ndarray,
    nsamples: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """
    Menghitung SHAP untuk satu sequence.

    Returns
    -------
    shap_full:
        SHAP shape:
        (MAX_SEQUENCE_LENGTH, NUM_CLASSES)

    expected_value:
        Expected probability setiap kelas.

    full_prediction:
        Probabilitas model ketika seluruh token dipertahankan.

    diagnostics:
        Informasi local accuracy/additivity.
    """

    sequence = np.asarray(
        sequence,
        dtype=np.int32,
    )

    if sequence.shape != (
        MAX_SEQUENCE_LENGTH,
    ):
        raise ValueError(
            "Shape sequence tidak sesuai.\n"
            f"Expected: {(MAX_SEQUENCE_LENGTH,)}\n"
            f"Actual  : {sequence.shape}"
        )

    valid_positions = np.flatnonzero(
        sequence
        != 0
    ).astype(
        np.int32
    )

    num_features = len(
        valid_positions
    )

    if num_features == 0:
        raise ValueError(
            "Sequence tidak memiliki token non-padding."
        )

    prediction_function = (
        build_token_mask_prediction_function(
            model=model,
            original_sequence=sequence,
            valid_positions=valid_positions,
        )
    )

    # Baseline:
    # seluruh token non-padding dihapus.
    background_mask = np.zeros(
        (
            1,
            num_features,
        ),
        dtype=np.float64,
    )

    # Instance:
    # seluruh token asli dipertahankan.
    full_mask = np.ones(
        (
            1,
            num_features,
        ),
        dtype=np.float64,
    )

    baseline_prediction = (
        prediction_function(
            background_mask
        )[0]
    )

    full_prediction = (
        prediction_function(
            full_mask
        )[0]
    )

    explainer = shap.KernelExplainer(
        prediction_function,
        background_mask,
        link="identity",
    )

    # Minimal evaluasi dibuat lebih besar dari jumlah fitur.
    # Hal ini membantu regresi lokal Kernel SHAP.
    effective_nsamples = max(
        int(
            nsamples
        ),
        num_features + 2,
    )

    l1_regularization = (
        f"num_features({num_features})"
    )

    # Beberapa versi SHAP mendukung parameter silent,
    # sedangkan versi lama tidak.
    try:
        raw_shap_values = (
            explainer.shap_values(
                full_mask,
                nsamples=effective_nsamples,
                l1_reg=l1_regularization,
                silent=True,
            )
        )

    except TypeError:
        raw_shap_values = (
            explainer.shap_values(
                full_mask,
                nsamples=effective_nsamples,
                l1_reg=l1_regularization,
            )
        )

    shap_valid = (
        normalize_kernel_shap_output(
            shap_values=raw_shap_values,
            num_features=num_features,
            num_classes=NUM_CLASSES,
        )
    )

    expected_value = (
        normalize_expected_value(
            expected_value=(
                explainer.expected_value
            ),
            num_classes=NUM_CLASSES,
        )
    )

    shap_full = np.zeros(
        (
            MAX_SEQUENCE_LENGTH,
            NUM_CLASSES,
        ),
        dtype=np.float64,
    )

    shap_full[
        valid_positions,
        :,
    ] = shap_valid

    reconstructed_prediction = (
        expected_value
        + shap_valid.sum(
            axis=0
        )
    )

    additivity_errors = np.abs(
        reconstructed_prediction
        - full_prediction
    )

    max_additivity_error = float(
        additivity_errors.max()
    )

    baseline_expected_error = float(
        np.max(
            np.abs(
                baseline_prediction
                - expected_value
            )
        )
    )

    if (
        max_additivity_error
        > ADDITIVITY_TOLERANCE
    ):
        raise ValueError(
            "Local accuracy SHAP melebihi toleransi.\n"
            f"Maximum error : {max_additivity_error:.8f}\n"
            f"Tolerance     : {ADDITIVITY_TOLERANCE:.8f}"
        )

    diagnostics = {
        "num_explained_tokens":
            num_features,

        "effective_nsamples":
            effective_nsamples,

        "max_additivity_error":
            max_additivity_error,

        "mean_additivity_error":
            float(
                additivity_errors.mean()
            ),

        "baseline_expected_error":
            baseline_expected_error,

        "baseline_probability_sum":
            float(
                baseline_prediction.sum()
            ),

        "full_probability_sum":
            float(
                full_prediction.sum()
            ),
    }

    del explainer

    return (
        shap_full,
        expected_value,
        full_prediction,
        diagnostics,
    )


# =============================================================================
# MENGHITUNG TOKEN IMPORTANCE
# =============================================================================

def calculate_token_importance(
    token_sequences: np.ndarray,
    shap_values: np.ndarray,
    predicted_classes: np.ndarray,
    vocabulary: list[str],
    index_to_label: dict[int, str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Menghitung:
    1. global token importance;
    2. token importance per output class;
    3. detail kontribusi setiap kemunculan token.

    Global importance satu posisi token:
    mean absolute SHAP pada seluruh output kelas.

    Importance per kelas:
    absolute SHAP terhadap output kelas tertentu.
    """

    expected_shap_shape = (
        len(
            token_sequences
        ),
        MAX_SEQUENCE_LENGTH,
        NUM_CLASSES,
    )

    if (
        shap_values.shape
        != expected_shap_shape
    ):
        raise ValueError(
            "Shape SHAP tidak sesuai untuk agregasi.\n"
            f"Expected: {expected_shap_shape}\n"
            f"Actual  : {shap_values.shape}"
        )

    global_records: list[
        dict[str, Any]
    ] = []

    class_records: list[
        dict[str, Any]
    ] = []

    detail_records: list[
        dict[str, Any]
    ] = []

    for sample_index in range(
        len(
            token_sequences
        )
    ):
        predicted_class = int(
            predicted_classes[
                sample_index
            ]
        )

        sequence = token_sequences[
            sample_index
        ]

        sample_shap = shap_values[
            sample_index
        ]

        for position, token_id_value in enumerate(
            sequence
        ):
            token_id = int(
                token_id_value
            )

            if token_id == 0:
                continue

            token = token_id_to_word(
                token_id=token_id,
                vocabulary=vocabulary,
            )

            class_shap_vector = sample_shap[
                position
            ]

            mean_abs_across_classes = float(
                np.mean(
                    np.abs(
                        class_shap_vector
                    )
                )
            )

            predicted_class_shap = float(
                class_shap_vector[
                    predicted_class
                ]
            )

            global_records.append(
                {
                    "token_id":
                        token_id,

                    "token":
                        token,

                    "importance":
                        mean_abs_across_classes,

                    "predicted_class_abs_shap":
                        abs(
                            predicted_class_shap
                        ),

                    "predicted_class_signed_shap":
                        predicted_class_shap,
                }
            )

            for class_index in range(
                NUM_CLASSES
            ):
                signed_value = float(
                    class_shap_vector[
                        class_index
                    ]
                )

                class_records.append(
                    {
                        "class_index":
                            class_index,

                        "class_name":
                            index_to_label[
                                class_index
                            ],

                        "token_id":
                            token_id,

                        "token":
                            token,

                        "signed_shap":
                            signed_value,

                        "abs_shap":
                            abs(
                                signed_value
                            ),
                    }
                )

                detail_records.append(
                    {
                        "sample_number":
                            sample_index + 1,

                        "position":
                            position,

                        "token_id":
                            token_id,

                        "token":
                            token,

                        "predicted_class_index":
                            predicted_class,

                        "predicted_class_name":
                            index_to_label[
                                predicted_class
                            ],

                        "output_class_index":
                            class_index,

                        "output_class_name":
                            index_to_label[
                                class_index
                            ],

                        "signed_shap":
                            signed_value,

                        "abs_shap":
                            abs(
                                signed_value
                            ),

                        "is_predicted_class":
                            (
                                class_index
                                == predicted_class
                            ),
                    }
                )

    global_dataframe = pd.DataFrame(
        global_records
    )

    class_dataframe = pd.DataFrame(
        class_records
    )

    detail_dataframe = pd.DataFrame(
        detail_records
    )

    if global_dataframe.empty:
        raise ValueError(
            "Tidak ada global token importance "
            "yang berhasil dihitung."
        )

    if class_dataframe.empty:
        raise ValueError(
            "Tidak ada class token importance "
            "yang berhasil dihitung."
        )

    global_summary = (
        global_dataframe
        .groupby(
            [
                "token_id",
                "token",
            ],
            as_index=False,
        )
        .agg(
            total_abs_shap=(
                "importance",
                "sum",
            ),

            mean_abs_shap=(
                "importance",
                "mean",
            ),

            total_abs_predicted_class_shap=(
                "predicted_class_abs_shap",
                "sum",
            ),

            mean_abs_predicted_class_shap=(
                "predicted_class_abs_shap",
                "mean",
            ),

            mean_signed_predicted_class_shap=(
                "predicted_class_signed_shap",
                "mean",
            ),

            occurrence_count=(
                "importance",
                "size",
            ),
        )
    )

    total_global_importance = float(
        global_summary[
            "total_abs_shap"
        ].sum()
    )

    if total_global_importance <= 0.0:
        raise ValueError(
            "Total global importance tidak lebih besar dari nol."
        )

    global_summary[
        "normalized_global_importance"
    ] = (
        global_summary[
            "total_abs_shap"
        ]
        / total_global_importance
    )

    global_summary = (
        global_summary
        .sort_values(
            [
                "total_abs_shap",
                "mean_abs_shap",
                "occurrence_count",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    global_summary.insert(
        0,
        "rank",
        np.arange(
            1,
            len(
                global_summary
            ) + 1,
        ),
    )

    class_summary = (
        class_dataframe
        .groupby(
            [
                "class_index",
                "class_name",
                "token_id",
                "token",
            ],
            as_index=False,
        )
        .agg(
            total_abs_shap=(
                "abs_shap",
                "sum",
            ),

            mean_abs_shap=(
                "abs_shap",
                "mean",
            ),

            mean_signed_shap=(
                "signed_shap",
                "mean",
            ),

            positive_contribution_count=(
                "signed_shap",
                lambda values:
                int(
                    (
                        values
                        > 0.0
                    ).sum()
                ),
            ),

            negative_contribution_count=(
                "signed_shap",
                lambda values:
                int(
                    (
                        values
                        < 0.0
                    ).sum()
                ),
            ),

            occurrence_count=(
                "signed_shap",
                "size",
            ),
        )
    )

    class_totals = (
        class_summary
        .groupby(
            "class_index"
        )[
            "total_abs_shap"
        ]
        .transform(
            "sum"
        )
    )

    class_summary[
        "normalized_importance_in_class"
    ] = np.divide(
        class_summary[
            "total_abs_shap"
        ],
        class_totals,
        out=np.zeros(
            len(
                class_summary
            ),
            dtype=np.float64,
        ),
        where=(
            class_totals
            != 0
        ),
    )

    class_summary[
        "rank_in_class"
    ] = (
        class_summary
        .groupby(
            "class_index"
        )[
            "total_abs_shap"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(
            int
        )
    )

    class_summary = (
        class_summary
        .sort_values(
            [
                "class_index",
                "rank_in_class",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    detail_dataframe = (
        detail_dataframe
        .sort_values(
            [
                "sample_number",
                "position",
                "output_class_index",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        global_summary,
        class_summary,
        detail_dataframe,
    )


# =============================================================================
# RINGKASAN SAMPEL
# =============================================================================

def build_sample_summary(
    test_data: dict[str, np.ndarray],
    selected_indices: np.ndarray,
    probabilities: np.ndarray,
    index_to_label: dict[int, str],
    diagnostics_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Menyimpan daftar sampel yang digunakan pada global SHAP.
    """

    y_true = test_data[
        "y"
    ][
        selected_indices
    ]

    y_pred = np.argmax(
        probabilities,
        axis=1,
    )

    confidence = np.max(
        probabilities,
        axis=1,
    )

    summary = pd.DataFrame(
        {
            "sample_number":
                np.arange(
                    1,
                    len(
                        selected_indices
                    ) + 1,
                ),

            "original_test_index":
                selected_indices,

            "document_id":
                test_data[
                    "document_id"
                ][
                    selected_indices
                ],

            "category_from_npz":
                test_data[
                    "category"
                ][
                    selected_indices
                ],

            "actual_index":
                y_true,

            "actual_label":
                [
                    index_to_label[
                        int(
                            value
                        )
                    ]
                    for value in y_true
                ],

            "predicted_index":
                y_pred,

            "predicted_label":
                [
                    index_to_label[
                        int(
                            value
                        )
                    ]
                    for value in y_pred
                ],

            "is_correct":
                (
                    y_true
                    == y_pred
                ),

            "prediction_confidence":
                confidence,
        }
    )

    diagnostic_columns = [
        "sample_number",
        "num_explained_tokens",
        "effective_nsamples",
        "max_additivity_error",
        "mean_additivity_error",
        "baseline_expected_error",
    ]

    summary = summary.merge(
        diagnostics_dataframe[
            diagnostic_columns
        ],
        on="sample_number",
        how="left",
        validate="one_to_one",
    )

    return summary


# =============================================================================
# GRAFIK GLOBAL TOKEN IMPORTANCE
# =============================================================================

def plot_global_importance(
    global_importance: pd.DataFrame,
) -> None:
    """
    Membuat grafik token global paling berpengaruh.
    """

    special_mask = (
        global_importance[
            "token"
        ]
        .astype(
            str
        )
        .map(
            is_special_token
        )
    )

    filtered_importance = global_importance[
        ~special_mask
    ].copy()

    plot_data = (
        filtered_importance
        .head(
            TOP_N_TOKENS
        )
        .sort_values(
            "total_abs_shap",
            ascending=True,
        )
    )

    if plot_data.empty:
        raise ValueError(
            "Tidak ada token semantik "
            "yang dapat divisualisasikan."
        )

    figure, axis = plt.subplots(
        figsize=(11, 8)
    )

    axis.barh(
        plot_data[
            "token"
        ],
        plot_data[
            "total_abs_shap"
        ],
    )

    axis.set_xlabel(
        "Total Absolute SHAP Value"
    )

    axis.set_ylabel(
        "Token"
    )

    axis.set_title(
        "20 Token Paling Berpengaruh secara Global\n"
        "CNN K2 — Title + Description"
    )

    axis.grid(
        axis="x",
        alpha=0.3,
    )

    figure.tight_layout()

    figure.savefig(
        GLOBAL_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# =============================================================================
# GRAFIK IMPORTANCE PER KELAS
# =============================================================================

def plot_importance_by_class(
    class_importance: pd.DataFrame,
    index_to_label: dict[int, str],
) -> None:
    """
    Membuat grafik token utama untuk setiap output kelas.
    """

    class_top_tokens: list[
        pd.DataFrame
    ] = []

    for class_index in sorted(
        index_to_label
    ):
        class_data = class_importance[
            class_importance[
                "class_index"
            ]
            == class_index
        ].copy()

        special_mask = (
            class_data[
                "token"
            ]
            .astype(
                str
            )
            .map(
                is_special_token
            )
        )

        class_data = class_data[
            ~special_mask
        ]

        class_data = (
            class_data
            .sort_values(
                "total_abs_shap",
                ascending=False,
            )
            .head(
                TOP_N_TOKENS_PER_CLASS
            )
            .copy()
        )

        if class_data.empty:
            print(
                "Peringatan: token semantik tidak ditemukan "
                f"untuk kelas {index_to_label[class_index]}."
            )

            continue

        class_data[
            "plot_label"
        ] = (
            class_data[
                "token"
            ]
            + " ("
            + index_to_label[
                class_index
            ]
            + ")"
        )

        class_top_tokens.append(
            class_data
        )

    if not class_top_tokens:
        raise ValueError(
            "Tidak ada token per kelas "
            "yang dapat divisualisasikan."
        )

    plot_data = pd.concat(
        class_top_tokens,
        ignore_index=True,
    )

    plot_data = plot_data.sort_values(
        "total_abs_shap",
        ascending=True,
    )

    figure, axis = plt.subplots(
        figsize=(12, 13)
    )

    axis.barh(
        plot_data[
            "plot_label"
        ],
        plot_data[
            "total_abs_shap"
        ],
    )

    axis.set_xlabel(
        "Total Absolute SHAP Value"
    )

    axis.set_ylabel(
        "Token dan Output Kelas"
    )

    axis.set_title(
        "Token Paling Berpengaruh pada Setiap Output Kelas\n"
        "CNN K2 — Title + Description"
    )

    axis.grid(
        axis="x",
        alpha=0.3,
    )

    figure.tight_layout()

    figure.savefig(
        CLASS_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# =============================================================================
# MENYIMPAN KONFIGURASI
# =============================================================================

def save_configuration(
    processing_seconds: float,
    selected_indices: np.ndarray,
    diagnostics_dataframe: pd.DataFrame,
    vocabulary_size: int,
) -> None:
    """
    Menyimpan konfigurasi analisis global SHAP.
    """

    configuration = {
        "generated_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "experiment_name":
            EXPERIMENT_NAME,

        "dataset":
            DATASET_NAME,

        "scenario_code":
            SCENARIO_CODE,

        "scenario_name":
            SCENARIO_NAME,

        "model_path":
            str(
                MODEL_PATH
            ),

        "explainer":
            "shap.KernelExplainer",

        "explained_representation":
            (
                "Binary token-presence masks "
                "applied to integer token sequences"
            ),

        "explained_model":
            (
                "Original CNN K2 checkpoint including "
                "ZeroPaddingEmbeddingOutput and "
                "MaskedGlobalMaxPooling1D"
            ),

        "baseline":
            (
                "All non-padding tokens in the explained "
                "document are replaced with token ID 0."
            ),

        "global_aggregation_method":
            (
                "Mean absolute SHAP across four output classes "
                "for each token occurrence, followed by sum "
                "across occurrences."
            ),

        "class_aggregation_method":
            (
                "Absolute SHAP for each output class, "
                "aggregated across all explained documents."
            ),

        "max_sequence_length":
            MAX_SEQUENCE_LENGTH,

        "num_classes":
            NUM_CLASSES,

        "vocabulary_size":
            vocabulary_size,

        "explain_size_requested":
            EXPLAIN_SIZE,

        "explain_size_actual":
            len(
                selected_indices
            ),

        "shap_nsamples_requested":
            SHAP_NSAMPLES,

        "effective_nsamples_min":
            int(
                diagnostics_dataframe[
                    "effective_nsamples"
                ].min()
            ),

        "effective_nsamples_max":
            int(
                diagnostics_dataframe[
                    "effective_nsamples"
                ].max()
            ),

        "random_seed":
            RANDOM_SEED,

        "prediction_batch_size":
            PREDICTION_BATCH_SIZE,

        "processing_seconds":
            round(
                processing_seconds,
                6,
            ),

        "maximum_additivity_error":
            float(
                diagnostics_dataframe[
                    "max_additivity_error"
                ].max()
            ),

        "mean_additivity_error":
            float(
                diagnostics_dataframe[
                    "mean_additivity_error"
                ].mean()
            ),

        "additivity_tolerance":
            ADDITIVITY_TOLERANCE,

        "selected_test_indices":
            [
                int(
                    value
                )
                for value in selected_indices
            ],

        "important_note":
            (
                "Global SHAP importance is an approximate, "
                "model-agnostic explanation computed on a "
                "stratified subset of the Kompas test set. "
                "Token removal is represented by padding."
            ),
    }

    with open(
        CONFIGURATION_PATH,
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
# PROGRAM UTAMA
# =============================================================================

def main() -> None:
    """
    Menjalankan global SHAP untuk CNN K2.
    """

    print("=" * 80)

    print(
        "STEP 7.1 - GLOBAL SHAP EXPLANATION"
    )

    print("=" * 80)

    set_random_seed()
    create_output_directories()

    print("\nKonfigurasi:")

    print(
        f"Model                : "
        f"{EXPERIMENT_NAME}"
    )

    print(
        f"Dataset              : "
        f"{DATASET_NAME}"
    )

    print(
        f"Skenario             : "
        f"{SCENARIO_CODE}"
    )

    print(
        f"Representasi         : "
        f"{SCENARIO_NAME}"
    )

    print(
        f"Metode SHAP          : "
        "KernelExplainer"
    )

    print(
        f"Explain size         : "
        f"{EXPLAIN_SIZE}"
    )

    print(
        f"SHAP nsamples        : "
        f"{SHAP_NSAMPLES}"
    )

    print(
        "Baseline             : "
        "seluruh token menjadi padding"
    )

    # =========================================================================
    # MEMUAT TEST SET
    # =========================================================================

    print(
        "\nMemuat dataset test..."
    )

    test_data = load_npz_dataset(
        TEST_DATA_PATH
    )

    print(
        f"Test shape           : "
        f"{test_data['X'].shape}"
    )

    # =========================================================================
    # MEMUAT VOCABULARY
    # =========================================================================

    print(
        "\nMemuat vocabulary..."
    )

    vocabulary = load_vocabulary(
        VOCABULARY_PATH
    )

    max_token_id = int(
        test_data[
            "X"
        ].max()
    )

    if (
        max_token_id
        >= len(
            vocabulary
        )
    ):
        raise ValueError(
            "Vocabulary tidak sesuai dengan indeks token.\n"
            f"Maximum token ID : {max_token_id}\n"
            f"Vocabulary size  : {len(vocabulary)}"
        )

    print(
        f"Vocabulary size      : "
        f"{len(vocabulary):,}"
    )

    print(
        f"Maximum token ID     : "
        f"{max_token_id:,}"
    )

    # =========================================================================
    # MEMUAT LABEL MAPPING
    # =========================================================================

    print(
        "\nMemuat label mapping..."
    )

    index_to_label = (
        load_index_to_label()
    )

    print(
        f"Label mapping        : "
        f"{index_to_label}"
    )

    # =========================================================================
    # MEMUAT MODEL
    # =========================================================================

    print(
        "\nMemuat model CNN K2..."
    )

    model = load_cnn_model()

    print(
        f"Model input          : "
        f"{model.input_shape}"
    )

    print(
        f"Model output         : "
        f"{model.output_shape}"
    )

    print(
        "Custom padding layer : tersedia"
    )

    print(
        "Custom pooling layer : tersedia"
    )

    # Warm-up model.
    _ = predict_probabilities(
        model=model,
        sequences=test_data[
            "X"
        ][
            :1
        ],
        batch_size=1,
        context_name="Model warm-up",
    )

    # =========================================================================
    # MEMILIH SAMPEL TEST
    # =========================================================================

    print(
        "\nMemilih sampel test secara stratified..."
    )

    explain_indices = (
        stratified_sample_indices(
            labels=test_data[
                "y"
            ],
            sample_size=EXPLAIN_SIZE,
            seed=RANDOM_SEED,
        )
    )

    if (
        len(
            explain_indices
        )
        != EXPLAIN_SIZE
    ):
        raise ValueError(
            "Jumlah sampel SHAP tidak sesuai."
        )

    explain_sequences = test_data[
        "X"
    ][
        explain_indices
    ]

    explain_labels = test_data[
        "y"
    ][
        explain_indices
    ]

    selected_distribution = pd.Series(
        explain_labels
    ).value_counts().sort_index()

    print(
        f"Jumlah sampel        : "
        f"{len(explain_indices)}"
    )

    print(
        "Distribusi sampel    :"
    )

    for class_index, count in (
        selected_distribution.items()
    ):
        print(
            f"  {class_index} - "
            f"{index_to_label[int(class_index)]}: "
            f"{int(count)}"
        )

    # =========================================================================
    # MENGHITUNG SHAP
    # =========================================================================

    print(
        "\nMenghitung SHAP values..."
    )

    print(
        "Proses dilakukan per dokumen dan dapat "
        "memerlukan beberapa menit pada CPU."
    )

    processing_start = time.perf_counter()

    shap_values_collection: list[
        np.ndarray
    ] = []

    expected_values_collection: list[
        np.ndarray
    ] = []

    probabilities_collection: list[
        np.ndarray
    ] = []

    diagnostic_rows: list[
        dict[str, Any]
    ] = []

    total_samples = len(
        explain_sequences
    )

    for sample_index, sequence in enumerate(
        explain_sequences,
        start=1,
    ):
        sample_start = time.perf_counter()

        (
            sample_shap_values,
            sample_expected_value,
            sample_probabilities,
            sample_diagnostics,
        ) = explain_single_sequence(
            model=model,
            sequence=sequence,
            nsamples=SHAP_NSAMPLES,
        )

        sample_seconds = (
            time.perf_counter()
            - sample_start
        )

        shap_values_collection.append(
            sample_shap_values
        )

        expected_values_collection.append(
            sample_expected_value
        )

        probabilities_collection.append(
            sample_probabilities
        )

        predicted_class = int(
            np.argmax(
                sample_probabilities
            )
        )

        diagnostic_rows.append(
            {
                "sample_number":
                    sample_index,

                "original_test_index":
                    int(
                        explain_indices[
                            sample_index - 1
                        ]
                    ),

                "document_id":
                    str(
                        test_data[
                            "document_id"
                        ][
                            explain_indices[
                                sample_index - 1
                            ]
                        ]
                    ),

                "actual_class_index":
                    int(
                        explain_labels[
                            sample_index - 1
                        ]
                    ),

                "predicted_class_index":
                    predicted_class,

                "predicted_class_name":
                    index_to_label[
                        predicted_class
                    ],

                "processing_seconds":
                    sample_seconds,

                **sample_diagnostics,
            }
        )

        print(
            f"[{sample_index:03d}/{total_samples:03d}] "
            f"token={sample_diagnostics['num_explained_tokens']:02d} | "
            f"prediksi={index_to_label[predicted_class]} | "
            f"confidence={sample_probabilities[predicted_class]:.4f} | "
            f"additivity_error="
            f"{sample_diagnostics['max_additivity_error']:.8f} | "
            f"waktu={sample_seconds:.2f}s"
        )

    processing_seconds = (
        time.perf_counter()
        - processing_start
    )

    shap_values_array = np.stack(
        shap_values_collection,
        axis=0,
    ).astype(
        np.float32
    )

    expected_values_array = np.stack(
        expected_values_collection,
        axis=0,
    ).astype(
        np.float32
    )

    probabilities = np.stack(
        probabilities_collection,
        axis=0,
    ).astype(
        np.float32
    )

    diagnostics_dataframe = pd.DataFrame(
        diagnostic_rows
    )

    expected_shap_shape = (
        EXPLAIN_SIZE,
        MAX_SEQUENCE_LENGTH,
        NUM_CLASSES,
    )

    if (
        shap_values_array.shape
        != expected_shap_shape
    ):
        raise ValueError(
            "Shape akhir SHAP tidak sesuai.\n"
            f"Expected: {expected_shap_shape}\n"
            f"Actual  : {shap_values_array.shape}"
        )

    if probabilities.shape != (
        EXPLAIN_SIZE,
        NUM_CLASSES,
    ):
        raise ValueError(
            "Shape probabilitas akhir tidak sesuai.\n"
            f"Shape: {probabilities.shape}"
        )

    predicted_classes = np.argmax(
        probabilities,
        axis=1,
    ).astype(
        np.int32
    )

    print(
        f"\nSHAP values shape    : "
        f"{shap_values_array.shape}"
    )

    print(
        f"Expected values      : "
        f"{expected_values_array.shape}"
    )

    print(
        f"Probabilities shape  : "
        f"{probabilities.shape}"
    )

    print(
        f"Waktu SHAP           : "
        f"{processing_seconds:.2f} detik"
    )

    # =========================================================================
    # AGREGASI IMPORTANCE
    # =========================================================================

    print(
        "\nMengagregasi token importance..."
    )

    (
        global_importance,
        class_importance,
        token_detail,
    ) = calculate_token_importance(
        token_sequences=explain_sequences,
        shap_values=shap_values_array,
        predicted_classes=predicted_classes,
        vocabulary=vocabulary,
        index_to_label=index_to_label,
    )

    sample_summary = build_sample_summary(
        test_data=test_data,
        selected_indices=explain_indices,
        probabilities=probabilities,
        index_to_label=index_to_label,
        diagnostics_dataframe=(
            diagnostics_dataframe
        ),
    )

    # =========================================================================
    # MENYIMPAN TABEL
    # =========================================================================

    print(
        "Menyimpan tabel hasil..."
    )

    global_importance.to_csv(
        GLOBAL_IMPORTANCE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    class_importance.to_csv(
        CLASS_IMPORTANCE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    sample_summary.to_csv(
        SAMPLE_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    token_detail.to_csv(
        TOKEN_DETAIL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    diagnostics_dataframe.to_csv(
        DIAGNOSTIC_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # MENYIMPAN ARRAY
    # =========================================================================

    np.savez_compressed(
        SHAP_VALUES_PATH,

        shap_values=(
            shap_values_array
        ),

        expected_values=(
            expected_values_array
        ),

        token_sequences=(
            explain_sequences
        ),

        explain_indices=(
            explain_indices
        ),

        probabilities=(
            probabilities
        ),

        predicted_classes=(
            predicted_classes
        ),

        actual_classes=(
            explain_labels
        ),

        max_additivity_errors=(
            diagnostics_dataframe[
                "max_additivity_error"
            ].to_numpy(
                dtype=np.float32
            )
        ),

        num_explained_tokens=(
            diagnostics_dataframe[
                "num_explained_tokens"
            ].to_numpy(
                dtype=np.int32
            )
        ),
    )

    # =========================================================================
    # MEMBUAT GRAFIK
    # =========================================================================

    print(
        "Membuat grafik global SHAP..."
    )

    plot_global_importance(
        global_importance=(
            global_importance
        )
    )

    plot_importance_by_class(
        class_importance=(
            class_importance
        ),
        index_to_label=(
            index_to_label
        ),
    )

    # =========================================================================
    # MENYIMPAN KONFIGURASI
    # =========================================================================

    save_configuration(
        processing_seconds=(
            processing_seconds
        ),
        selected_indices=(
            explain_indices
        ),
        diagnostics_dataframe=(
            diagnostics_dataframe
        ),
        vocabulary_size=len(
            vocabulary
        ),
    )

    # =========================================================================
    # MENAMPILKAN RINGKASAN
    # =========================================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "HASIL GLOBAL SHAP"
    )

    print("=" * 80)

    print(
        f"\nJumlah sampel dijelaskan : "
        f"{len(explain_indices)}"
    )

    print(
        f"Jumlah token unik        : "
        f"{len(global_importance):,}"
    )

    print(
        f"Jumlah detail kontribusi : "
        f"{len(token_detail):,}"
    )

    print(
        f"Waktu pemrosesan         : "
        f"{processing_seconds:.2f} detik"
    )

    print(
        f"Max additivity error     : "
        f"{diagnostics_dataframe['max_additivity_error'].max():.8f}"
    )

    print(
        f"Mean additivity error    : "
        f"{diagnostics_dataframe['mean_additivity_error'].mean():.8f}"
    )

    print(
        "\n20 token semantik paling berpengaruh:"
    )

    semantic_mask = (
        global_importance[
            "token"
        ]
        .astype(
            str
        )
        .map(
            is_special_token
        )
    )

    semantic_importance = global_importance[
        ~semantic_mask
    ].copy()

    semantic_importance.insert(
        0,
        "semantic_rank",
        np.arange(
            1,
            len(
                semantic_importance
            ) + 1,
        ),
    )

    display_columns = [
        "semantic_rank",
        "token",
        "total_abs_shap",
        "mean_abs_shap",
        "occurrence_count",
        "normalized_global_importance",
    ]

    print(
        "\n"
        + semantic_importance[
            display_columns
        ]
        .head(
            TOP_N_TOKENS
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nTabel global importance:"
    )

    print(
        GLOBAL_IMPORTANCE_PATH
    )

    print(
        "\nTabel importance per output kelas:"
    )

    print(
        CLASS_IMPORTANCE_PATH
    )

    print(
        "\nDetail kontribusi token:"
    )

    print(
        TOKEN_DETAIL_PATH
    )

    print(
        "\nRingkasan sampel:"
    )

    print(
        SAMPLE_SUMMARY_PATH
    )

    print(
        "\nDiagnostik SHAP:"
    )

    print(
        DIAGNOSTIC_PATH
    )

    print(
        "\nArray SHAP:"
    )

    print(
        SHAP_VALUES_PATH
    )

    print(
        "\nGrafik global:"
    )

    print(
        GLOBAL_FIGURE_PATH
    )

    print(
        "\nGrafik per kelas:"
    )

    print(
        CLASS_FIGURE_PATH
    )

    print(
        "\nKonfigurasi:"
    )

    print(
        CONFIGURATION_PATH
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "Tahap global SHAP selesai."
    )

    print("=" * 80)

    # =========================================================================
    # PEMBERSIHAN MEMORY
    # =========================================================================

    del model
    del shap_values_collection
    del expected_values_collection
    del probabilities_collection

    tf.keras.backend.clear_session()

    gc.collect()


if __name__ == "__main__":
    main()
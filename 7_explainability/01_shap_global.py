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
# Mekanisme:
# 1. Memuat input test yang sudah berbentuk integer sequence.
# 2. Mengubah sequence menjadi embedding menggunakan embedding layer model.
# 3. Menghitung SHAP pada representasi embedding.
# 4. Menjumlahkan nilai absolut SHAP pada seluruh dimensi embedding.
# 5. Menghasilkan importance setiap token/kata.
#
# Output:
# - Tabel global token importance
# - Tabel global importance per kelas
# - Grafik 20 token paling berpengaruh
# - Konfigurasi analisis SHAP
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
from typing import Any

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
# KONFIGURASI ANALISIS
# =============================================================================

RANDOM_SEED = 42

EXPERIMENT_NAME = "cnn_k2"
DATASET_NAME = "Kompas"
SCENARIO_CODE = "K2"
SCENARIO_NAME = "Title + Description"

MAX_SEQUENCE_LENGTH = 60
NUM_CLASSES = 4

# Jumlah data background yang menjadi referensi SHAP.
BACKGROUND_SIZE = 50

# Jumlah sampel test yang dijelaskan.
EXPLAIN_SIZE = 100

# Jumlah estimasi expected gradients.
# Semakin besar nilainya, estimasi lebih stabil tetapi semakin lambat.
SHAP_NSAMPLES = 100

TOP_N_TOKENS = 20

# Token teknis yang tetap disimpan di tabel mentah,
# tetapi tidak ditampilkan pada visualisasi semantik.
SPECIAL_TOKENS = {
    "[PAD]",
    "[SEP]",
    "[UNK]",
    "[OOV]",
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

TRAIN_DATA_PATH = (
    PROJECT_ROOT
    / "2_data"
    / "vectorized"
    / "kompas_k2"
    / "train.npz"
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


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_random_seed(
    seed: int = RANDOM_SEED,
) -> None:
    """
    Mengatur random seed agar pemilihan sampel dapat direproduksi.
    """

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# =============================================================================
# MEMBUAT FOLDER OUTPUT
# =============================================================================

def create_output_directories() -> None:
    """
    Membuat seluruh folder output analisis SHAP.
    """

    for directory in [
        SHAP_TABLES_DIR,
        SHAP_FIGURES_DIR,
        SHAP_ARRAYS_DIR,
    ]:
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
                f"Key hilang: {missing_keys}"
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
            f"X harus dua dimensi. Shape: {result['X'].shape}"
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

    return result


# =============================================================================
# MEMUAT VOCABULARY
# =============================================================================

def load_vocabulary(
    vocabulary_path: Path,
) -> list[str]:
    """
    Membaca vocabulary sesuai urutan indeks TextVectorization.

    Baris kosong pertama tetap dipertahankan karena indeks 0 biasanya
    digunakan sebagai padding.
    """

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
        vocabulary = file.read().splitlines()

    if not vocabulary:
        raise ValueError(
            "File vocabulary kosong."
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

    if 0 <= token_id < len(vocabulary):
        token = vocabulary[token_id].strip()

        if not token:
            return "[PAD]"

        return token

    return "[OOV]"


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
        mapping_data = json.load(file)

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

    if "index_to_label" in kompas_mapping:
        raw_mapping = kompas_mapping[
            "index_to_label"
        ]

        return {
            int(index): str(label)
            for index, label
            in raw_mapping.items()
        }

    if "label_to_index" in kompas_mapping:
        raw_mapping = kompas_mapping[
            "label_to_index"
        ]

        return {
            int(index): str(label)
            for label, index
            in raw_mapping.items()
        }

    # Fallback jika JSON langsung berisi label -> indeks.
    if all(
        isinstance(value, int)
        for value in kompas_mapping.values()
    ):
        return {
            int(index): str(label)
            for label, index
            in kompas_mapping.items()
        }

    raise KeyError(
        "Format mapping label Kompas tidak dikenali."
    )


# =============================================================================
# MEMUAT MODEL CNN
# =============================================================================

def load_cnn_model() -> tf.keras.Model:
    """
    Memuat checkpoint terbaik CNN K2.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Checkpoint CNN K2 tidak ditemukan:\n"
            f"{MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
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

    return model


# =============================================================================
# MODEL EMBEDDING DAN MODEL BAGIAN AKHIR
# =============================================================================

def build_embedding_model(
    original_model: tf.keras.Model,
) -> tf.keras.Model:
    """
    Membuat model yang menghasilkan embedding dari integer sequence.
    """

    embedding_layer = original_model.get_layer(
        "embedding"
    )

    embedding_model = tf.keras.Model(
        inputs=original_model.input,
        outputs=embedding_layer.output,
        name="CNN_K2_Embedding_Model",
    )

    return embedding_model


def build_embedding_tail_model(
    original_model: tf.keras.Model,
) -> tf.keras.Model:
    """
    Membuat model yang menerima embedding sebagai input,
    kemudian meneruskan embedding ke bagian CNN hingga
    output softmax.

    Model ini digunakan oleh SHAP GradientExplainer.
    """

    embedding_layer = original_model.get_layer(
        "embedding"
    )

    # Dimensi embedding diambil langsung dari konfigurasi layer.
    embedding_dimension = int(
        embedding_layer.output_dim
    )

    embedding_input = tf.keras.Input(
        shape=(
            MAX_SEQUENCE_LENGTH,
            embedding_dimension,
        ),
        dtype=tf.float32,
        name="embedding_input",
    )

    x = embedding_input

    downstream_layer_names = [
        "spatial_dropout",
        "conv1d",
        "global_max_pooling",
        "dense",
        "dropout",
        "output",
    ]

    for layer_name in downstream_layer_names:
        layer = original_model.get_layer(
            layer_name
        )

        if isinstance(
            layer,
            (
                tf.keras.layers.Dropout,
                tf.keras.layers.SpatialDropout1D,
            ),
        ):
            x = layer(
                x,
                training=False,
            )
        else:
            x = layer(x)

    tail_model = tf.keras.Model(
        inputs=embedding_input,
        outputs=x,
        name="CNN_K2_Embedding_Tail",
    )

    return tail_model


# =============================================================================
# PEMILIHAN SAMPEL
# =============================================================================

def stratified_sample_indices(
    labels: np.ndarray,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    """
    Memilih sampel yang relatif seimbang dari setiap kelas.
    """

    rng = np.random.default_rng(
        seed
    )

    unique_labels = np.unique(
        labels
    )

    base_per_class = (
        sample_size
        // len(unique_labels)
    )

    remainder = (
        sample_size
        % len(unique_labels)
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

        class_sample_size = min(
            class_sample_size,
            len(class_indices),
        )

        sampled = rng.choice(
            class_indices,
            size=class_sample_size,
            replace=False,
        )

        selected_indices.extend(
            sampled.tolist()
        )

    selected_array = np.asarray(
        selected_indices,
        dtype=np.int32,
    )

    rng.shuffle(
        selected_array
    )

    return selected_array


# =============================================================================
# NORMALISASI FORMAT OUTPUT SHAP
# =============================================================================

def normalize_shap_output(
    shap_values: Any,
    expected_samples: int,
    sequence_length: int,
    embedding_dimension: int,
    num_classes: int,
) -> np.ndarray:
    """
    Mengubah berbagai kemungkinan output GradientExplainer menjadi:

    (num_classes, samples, sequence_length, embedding_dimension)
    """

    if isinstance(
        shap_values,
        list,
    ):
        class_arrays = [
            np.asarray(
                value,
                dtype=np.float32,
            )
            for value in shap_values
        ]

        result = np.stack(
            class_arrays,
            axis=0,
        )

    else:
        array = np.asarray(
            shap_values,
            dtype=np.float32,
        )

        # Bentuk:
        # samples x sequence x embedding x classes
        if (
            array.ndim == 4
            and array.shape[0] == expected_samples
            and array.shape[1] == sequence_length
            and array.shape[2] == embedding_dimension
            and array.shape[3] == num_classes
        ):
            result = np.moveaxis(
                array,
                -1,
                0,
            )

        # Bentuk:
        # classes x samples x sequence x embedding
        elif (
            array.ndim == 4
            and array.shape[0] == num_classes
            and array.shape[1] == expected_samples
        ):
            result = array

        else:
            raise ValueError(
                "Format output SHAP tidak dikenali.\n"
                f"Shape ditemukan: {array.shape}"
            )

    expected_shape = (
        num_classes,
        expected_samples,
        sequence_length,
        embedding_dimension,
    )

    if result.shape != expected_shape:
        raise ValueError(
            "Shape SHAP setelah normalisasi tidak sesuai.\n"
            f"Expected: {expected_shape}\n"
            f"Actual  : {result.shape}"
        )

    return result


# =============================================================================
# MENGHITUNG TOKEN IMPORTANCE
# =============================================================================

def calculate_token_importance(
    token_sequences: np.ndarray,
    normalized_shap_values: np.ndarray,
    predicted_classes: np.ndarray,
    vocabulary: list[str],
    index_to_label: dict[int, str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Menghitung global importance dan importance per kelas.

    Importance satu token:
    jumlah absolut nilai SHAP pada seluruh dimensi embedding.
    """

    global_records: list[
        dict[str, Any]
    ] = []

    class_records: list[
        dict[str, Any]
    ] = []

    num_samples = token_sequences.shape[0]

    for sample_index in range(
        num_samples
    ):
        predicted_class = int(
            predicted_classes[
                sample_index
            ]
        )

        predicted_class_name = (
            index_to_label[
                predicted_class
            ]
        )

        # Gunakan SHAP untuk kelas yang diprediksi model.
        sample_shap = (
            normalized_shap_values[
                predicted_class,
                sample_index,
            ]
        )

        # Agregasi dimensi embedding menjadi importance per posisi token.
        position_importance = np.sum(
            np.abs(sample_shap),
            axis=-1,
        )

        for position, token_id in enumerate(
            token_sequences[
                sample_index
            ]
        ):
            token_id = int(
                token_id
            )

            # Padding tidak dimasukkan sebagai fitur penting.
            if token_id == 0:
                continue

            token = token_id_to_word(
                token_id,
                vocabulary,
            )

            importance = float(
                position_importance[
                    position
                ]
            )

            record = {
                "token_id":
                    token_id,

                "token":
                    token,

                "importance":
                    importance,
            }

            global_records.append(
                record
            )

            class_records.append(
                {
                    **record,

                    "class_index":
                        predicted_class,

                    "class_name":
                        predicted_class_name,
                }
            )

    global_dataframe = pd.DataFrame(
        global_records
    )

    class_dataframe = pd.DataFrame(
        class_records
    )

    if global_dataframe.empty:
        raise ValueError(
            "Tidak ada token importance yang berhasil dihitung."
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
            occurrence_count=(
                "importance",
                "size",
            ),
        )
    )

    global_summary[
        "normalized_global_importance"
    ] = (
        global_summary[
            "total_abs_shap"
        ]
        / global_summary[
            "total_abs_shap"
        ].sum()
    )

    global_summary = (
        global_summary
        .sort_values(
            "total_abs_shap",
            ascending=False,
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
            len(global_summary) + 1,
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
                "importance",
                "sum",
            ),
            mean_abs_shap=(
                "importance",
                "mean",
            ),
            occurrence_count=(
                "importance",
                "size",
            ),
        )
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
        .astype(int)
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

    return (
        global_summary,
        class_summary,
    )


# =============================================================================
# MEMBUAT RINGKASAN SAMPEL
# =============================================================================

def build_sample_summary(
    test_data: dict[str, np.ndarray],
    selected_indices: np.ndarray,
    probabilities: np.ndarray,
    index_to_label: dict[int, str],
) -> pd.DataFrame:
    """
    Menyimpan daftar sampel yang digunakan pada global SHAP.
    """

    y_true = test_data[
        "y"
    ][selected_indices]

    y_pred = np.argmax(
        probabilities,
        axis=1,
    )

    confidence = probabilities.max(
        axis=1
    )

    return pd.DataFrame(
        {
            "sample_number":
                np.arange(
                    1,
                    len(selected_indices) + 1,
                ),

            "original_test_index":
                selected_indices,

            "document_id":
                test_data[
                    "document_id"
                ][selected_indices],

            "actual_index":
                y_true,

            "actual_label":
                [
                    index_to_label[
                        int(value)
                    ]
                    for value in y_true
                ],

            "predicted_index":
                y_pred,

            "predicted_label":
                [
                    index_to_label[
                        int(value)
                    ]
                    for value in y_pred
                ],

            "is_correct":
                y_true == y_pred,

            "prediction_confidence":
                confidence,
        }
    )


# =============================================================================
# GRAFIK GLOBAL TOKEN IMPORTANCE
# =============================================================================

def plot_global_importance(
    global_importance: pd.DataFrame,
) -> None:
    """
    Membuat grafik token global paling berpengaruh.

    Token khusus seperti padding, separator, unknown,
    dan out-of-vocabulary tidak ditampilkan karena
    tidak memiliki makna semantik sebagai kata berita.
    """

    filtered_importance = global_importance[
        ~global_importance[
            "token"
        ].isin(
            SPECIAL_TOKENS
        )
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
            "Tidak ada token semantik yang dapat divisualisasikan."
        )

    plt.figure(
        figsize=(11, 8)
    )

    plt.barh(
        plot_data["token"],
        plot_data["total_abs_shap"],
    )

    plt.xlabel(
        "Total Absolute SHAP Value"
    )

    plt.ylabel(
        "Token"
    )

    plt.title(
        "20 Token Paling Berpengaruh secara Global\n"
        "Model CNN K2 — Title + Description"
    )

    plt.grid(
        axis="x",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        GLOBAL_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# =============================================================================
# GRAFIK IMPORTANCE PER KELAS
# =============================================================================

def plot_importance_by_class(
    class_importance: pd.DataFrame,
    index_to_label: dict[int, str],
) -> None:
    """
    Membuat grafik token utama pada setiap kelas.

    Token khusus seperti [PAD], [SEP], [UNK],
    dan [OOV] tidak ditampilkan.
    """

    class_top_tokens = []

    for class_index in sorted(
        index_to_label
    ):
        class_data = (
            class_importance[
                (
                    class_importance[
                        "class_index"
                    ]
                    == class_index
                )
                & (
                    ~class_importance[
                        "token"
                    ].isin(
                        SPECIAL_TOKENS
                    )
                )
            ]
            .head(10)
            .copy()
        )

        if class_data.empty:
            print(
                "Peringatan: tidak ditemukan token semantik "
                f"untuk kelas {index_to_label[class_index]}."
            )
            continue

        class_data[
            "label"
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
            "Tidak ada token per kelas yang dapat divisualisasikan."
        )

    plot_data = pd.concat(
        class_top_tokens,
        ignore_index=True,
    )

    plot_data = plot_data.sort_values(
        "total_abs_shap",
        ascending=True,
    )

    plt.figure(
        figsize=(12, 12)
    )

    plt.barh(
        plot_data["label"],
        plot_data["total_abs_shap"],
    )

    plt.xlabel(
        "Total Absolute SHAP Value"
    )

    plt.ylabel(
        "Token dan Kelas Prediksi"
    )

    plt.title(
        "Token Paling Berpengaruh pada Setiap Kelas\n"
        "Model CNN K2"
    )

    plt.grid(
        axis="x",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        CLASS_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# =============================================================================
# MENYIMPAN KONFIGURASI
# =============================================================================

def save_configuration(
    embedding_dimension: int,
    processing_seconds: float,
    selected_indices: np.ndarray,
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

        "explainer":
            "shap.GradientExplainer",

        "explained_representation":
            "token embedding",

        "aggregation_method":
            (
                "Sum absolute SHAP values across "
                "embedding dimensions"
            ),

        "max_sequence_length":
            MAX_SEQUENCE_LENGTH,

        "embedding_dimension":
            embedding_dimension,

        "background_size":
            BACKGROUND_SIZE,

        "explain_size":
            len(
                selected_indices
            ),

        "shap_nsamples":
            SHAP_NSAMPLES,

        "random_seed":
            RANDOM_SEED,

        "processing_seconds":
            round(
                processing_seconds,
                6,
            ),

        "important_note":
            (
                "Global importance is an approximate explanation "
                "computed on a stratified subset of the test set. "
                "Padding tokens are excluded."
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
    print(f"Model                : {EXPERIMENT_NAME}")
    print(f"Dataset              : {DATASET_NAME}")
    print(f"Skenario             : {SCENARIO_CODE}")
    print(f"Representasi         : {SCENARIO_NAME}")
    print(f"Background size      : {BACKGROUND_SIZE}")
    print(f"Explain size         : {EXPLAIN_SIZE}")
    print(f"SHAP nsamples        : {SHAP_NSAMPLES}")

    print("\nMemuat dataset train...")
    train_data = load_npz_dataset(
        TRAIN_DATA_PATH
    )

    print("Memuat dataset test...")
    test_data = load_npz_dataset(
        TEST_DATA_PATH
    )

    print(
        f"Train shape          : {train_data['X'].shape}"
    )

    print(
        f"Test shape           : {test_data['X'].shape}"
    )

    print("\nMemuat vocabulary...")
    vocabulary = load_vocabulary(
        VOCABULARY_PATH
    )

    max_token_id = int(
        max(
            train_data["X"].max(),
            test_data["X"].max(),
        )
    )

    if max_token_id >= len(
        vocabulary
    ):
        raise ValueError(
            "Jumlah vocabulary tidak sesuai dengan indeks token.\n"
            f"Maximum token ID : {max_token_id}\n"
            f"Vocabulary size  : {len(vocabulary)}"
        )

    print(
        f"Vocabulary size      : {len(vocabulary):,}"
    )

    print("\nMemuat label mapping...")
    index_to_label = (
        load_index_to_label()
    )

    print(
        f"Label mapping        : {index_to_label}"
    )

    print("\nMemuat model CNN K2...")
    original_model = load_cnn_model()

    embedding_model = build_embedding_model(
        original_model
    )

    tail_model = build_embedding_tail_model(
        original_model
    )

    embedding_dimension = int(
        original_model.get_layer(
            "embedding"
        ).output_dim
    )

    print(
        f"Embedding dimension  : {embedding_dimension}"
    )

    print(
        f"Tail model input     : {tail_model.input_shape}"
    )

    print(
        f"Tail model output    : {tail_model.output_shape}"
    )

    print("\nMemilih background data...")
    background_indices = (
        stratified_sample_indices(
            labels=train_data["y"],
            sample_size=BACKGROUND_SIZE,
            seed=RANDOM_SEED,
        )
    )

    print("Memilih data test yang dijelaskan...")
    explain_indices = (
        stratified_sample_indices(
            labels=test_data["y"],
            sample_size=EXPLAIN_SIZE,
            seed=RANDOM_SEED + 1,
        )
    )

    background_sequences = (
        train_data["X"][
            background_indices
        ]
    )

    explain_sequences = (
        test_data["X"][
            explain_indices
        ]
    )

    print("\nMembentuk embedding background...")
    background_embeddings = (
        embedding_model.predict(
            background_sequences,
            batch_size=64,
            verbose=0,
        )
    )

    print("Membentuk embedding data penjelasan...")
    explain_embeddings = (
        embedding_model.predict(
            explain_sequences,
            batch_size=64,
            verbose=0,
        )
    )

    background_embeddings = np.asarray(
        background_embeddings,
        dtype=np.float32,
    )

    explain_embeddings = np.asarray(
        explain_embeddings,
        dtype=np.float32,
    )

    print(
        f"Background embedding : {background_embeddings.shape}"
    )

    print(
        f"Explain embedding    : {explain_embeddings.shape}"
    )

    print("\nMembentuk GradientExplainer...")

    explainer = shap.GradientExplainer(
        tail_model,
        background_embeddings,
    )

    print(
        "Menghitung SHAP values...\n"
        "Proses ini dapat memerlukan beberapa menit pada CPU."
    )

    processing_start = time.perf_counter()

    raw_shap_values = explainer.shap_values(
        explain_embeddings,
        nsamples=SHAP_NSAMPLES,
    )

    processing_seconds = (
        time.perf_counter()
        - processing_start
    )

    normalized_shap_values = (
        normalize_shap_output(
            shap_values=raw_shap_values,
            expected_samples=len(
                explain_sequences
            ),
            sequence_length=(
                MAX_SEQUENCE_LENGTH
            ),
            embedding_dimension=(
                embedding_dimension
            ),
            num_classes=NUM_CLASSES,
        )
    )

    print(
        f"SHAP values shape    : "
        f"{normalized_shap_values.shape}"
    )

    print(
        f"Waktu SHAP           : "
        f"{processing_seconds:.2f} detik"
    )

    print("\nMenghitung prediksi sampel...")
    probabilities = tail_model.predict(
        explain_embeddings,
        batch_size=64,
        verbose=0,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32,
    )

    predicted_classes = np.argmax(
        probabilities,
        axis=1,
    )

    print("Mengagregasi importance setiap token...")

    (
        global_importance,
        class_importance,
    ) = calculate_token_importance(
        token_sequences=explain_sequences,
        normalized_shap_values=(
            normalized_shap_values
        ),
        predicted_classes=(
            predicted_classes
        ),
        vocabulary=vocabulary,
        index_to_label=index_to_label,
    )

    sample_summary = build_sample_summary(
        test_data=test_data,
        selected_indices=explain_indices,
        probabilities=probabilities,
        index_to_label=index_to_label,
    )

    print("Menyimpan hasil tabel...")

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

    np.savez_compressed(
        SHAP_VALUES_PATH,
        shap_values=(
            normalized_shap_values
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
    )

    print("Membuat grafik global SHAP...")

    plot_global_importance(
        global_importance
    )

    plot_importance_by_class(
        class_importance=class_importance,
        index_to_label=index_to_label,
    )

    save_configuration(
        embedding_dimension=(
            embedding_dimension
        ),
        processing_seconds=(
            processing_seconds
        ),
        selected_indices=(
            explain_indices
        ),
    )

    print("\n" + "=" * 80)
    print("HASIL GLOBAL SHAP")
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
        f"Waktu pemrosesan         : "
        f"{processing_seconds:.2f} detik"
    )

    print("\n20 token semantik paling berpengaruh:")

    semantic_importance = global_importance[
        ~global_importance[
            "token"
        ].isin(
            SPECIAL_TOKENS
        )
    ].copy()

    semantic_importance.insert(
        0,
        "semantic_rank",
        np.arange(
            1,
            len(semantic_importance) + 1,
        ),
    )

    display_columns = [
        "semantic_rank",
        "token",
        "total_abs_shap",
        "mean_abs_shap",
        "occurrence_count",
    ]

    print(
        "\n"
        + semantic_importance[
            display_columns
        ]
        .head(TOP_N_TOKENS)
        .to_string(
            index=False
        )
    )

    print("\nTabel global importance:")
    print(GLOBAL_IMPORTANCE_PATH)

    print("\nTabel importance per kelas:")
    print(CLASS_IMPORTANCE_PATH)

    print("\nRingkasan sampel:")
    print(SAMPLE_SUMMARY_PATH)

    print("\nSHAP values:")
    print(SHAP_VALUES_PATH)

    print("\nGrafik global:")
    print(GLOBAL_FIGURE_PATH)

    print("\nGrafik per kelas:")
    print(CLASS_FIGURE_PATH)

    print("\nKonfigurasi:")
    print(CONFIGURATION_PATH)

    print("\n" + "=" * 80)
    print(
        "Tahap global SHAP selesai."
    )
    print("=" * 80)

    del original_model
    del embedding_model
    del tail_model
    del explainer

    tf.keras.backend.clear_session()
    gc.collect()


if __name__ == "__main__":
    main()
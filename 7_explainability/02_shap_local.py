# =============================================================================
# STEP 7.2 - LOCAL SHAP EXPLANATION
# =============================================================================
# File:
# 7_explainability/02_shap_local.py
#
# Tujuan:
# Menjelaskan prediksi model CNN K2 pada tingkat artikel menggunakan
# SHAP values yang telah dihitung pada tahap global SHAP.
#
# Model:
# - CNN
# - Dataset Kompas
# - Skenario K2: Title + Description
#
# Input:
# - 9_results/shap_values/cnn_k2_global_shap_values.npz
# - 2_data/vectorized/kompas_k2/test.npz
# - 2_data/vectorized/kompas_k2/vocabulary.txt
# - 9_results/tables/label_mapping.json
#
# Format SHAP:
# - shap_values memiliki shape:
#   (jumlah_sampel, panjang_sequence, jumlah_kelas)
#
# - Setiap nilai SHAP menunjukkan kontribusi satu posisi token
#   terhadap probabilitas suatu output kelas.
#
# Interpretasi local SHAP:
# - SHAP positif:
#   token meningkatkan probabilitas kelas prediksi.
#
# - SHAP negatif:
#   token menurunkan probabilitas kelas prediksi.
#
# - Absolute SHAP:
#   menunjukkan besar pengaruh token tanpa memperhatikan arah.
#
# Pemilihan sampel:
# - 1 prediksi benar dengan confidence tertinggi dari setiap kelas.
# - 1 prediksi salah dengan confidence tertinggi berdasarkan
#   kelas aktual, jika tersedia.
#
# Output:
# - Ringkasan sampel local SHAP.
# - Kontribusi token yang sudah diagregasi.
# - Kontribusi setiap posisi token.
# - Grafik local SHAP setiap sampel.
# - Konfigurasi local SHAP.
# =============================================================================

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# KONFIGURASI
# =============================================================================

EXPERIMENT_NAME = "cnn_k2"
DATASET_NAME = "Kompas"
SCENARIO_CODE = "K2"
SCENARIO_NAME = "Title + Description"

NUM_CLASSES = 4
MAX_SEQUENCE_LENGTH = 60

# Jumlah maksimum sampel local explanation.
#
# Pemilihan dilakukan dengan urutan:
# 1. Satu prediksi benar per kelas.
# 2. Satu prediksi salah per kelas aktual jika tersedia.
MAX_LOCAL_SAMPLES = 8

# Jumlah token teratas yang ditampilkan pada grafik.
TOP_N_TOKENS = 15

# Toleransi pemeriksaan rekonstruksi probabilitas.
ADDITIVITY_TOLERANCE = 2e-2

# Token teknis yang tidak ditampilkan dalam local explanation semantik.
SPECIAL_TOKENS = {
    "",
    "PAD",
    "SEP",
    "UNK",
    "OOV",
}


# =============================================================================
# PATH INPUT
# =============================================================================

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

GLOBAL_SHAP_VALUES_PATH = (
    PROJECT_ROOT
    / "9_results"
    / "shap_values"
    / "cnn_k2_global_shap_values.npz"
)


# =============================================================================
# PATH OUTPUT
# =============================================================================

LOCAL_TABLES_DIR = (
    PROJECT_ROOT
    / "9_results"
    / "tables"
    / "shap"
    / "local"
)

LOCAL_FIGURES_DIR = (
    PROJECT_ROOT
    / "9_results"
    / "figures"
    / "shap"
    / "local"
)

LOCAL_SUMMARY_PATH = (
    LOCAL_TABLES_DIR
    / "cnn_k2_local_shap_summary.csv"
)

LOCAL_TOKEN_CONTRIBUTIONS_PATH = (
    LOCAL_TABLES_DIR
    / "cnn_k2_local_token_contributions.csv"
)

LOCAL_TOKEN_POSITION_PATH = (
    LOCAL_TABLES_DIR
    / "cnn_k2_local_token_position_contributions.csv"
)

LOCAL_CONFIGURATION_PATH = (
    LOCAL_TABLES_DIR
    / "cnn_k2_local_shap_configuration.json"
)


# =============================================================================
# MEMBUAT FOLDER OUTPUT
# =============================================================================

def create_output_directories() -> None:
    """
    Membuat seluruh folder output local SHAP.
    """

    directories = [
        LOCAL_TABLES_DIR,
        LOCAL_FIGURES_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# =============================================================================
# MEMUAT DATA TEST
# =============================================================================

def load_test_data() -> dict[str, np.ndarray]:
    """
    Memuat dan memvalidasi test set Kompas K2.
    """

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            "Test set tidak ditemukan:\n"
            f"{TEST_DATA_PATH}"
        )

    if not TEST_DATA_PATH.is_file():
        raise ValueError(
            "Path test set bukan file:\n"
            f"{TEST_DATA_PATH}"
        )

    if TEST_DATA_PATH.stat().st_size <= 0:
        raise ValueError(
            "File test set ditemukan, tetapi kosong:\n"
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
            "X test harus dua dimensi.\n"
            f"Shape ditemukan: {result['X'].shape}"
        )

    if result["y"].ndim != 1:
        raise ValueError(
            "y test harus satu dimensi.\n"
            f"Shape ditemukan: {result['y'].shape}"
        )

    if (
        result["X"].shape[1]
        != MAX_SEQUENCE_LENGTH
    ):
        raise ValueError(
            "Panjang sequence test tidak sesuai.\n"
            f"Expected: {MAX_SEQUENCE_LENGTH}\n"
            f"Actual  : {result['X'].shape[1]}"
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
            "Jumlah data setiap komponen test set tidak sama.\n"
            f"{component_lengths}"
        )

    if (
        result["X"]
        < 0
    ).any():
        raise ValueError(
            "Ditemukan token ID negatif pada test set."
        )

    expected_labels = set(
        range(
            NUM_CLASSES
        )
    )

    actual_labels = set(
        np.unique(
            result["y"]
        ).tolist()
    )

    if actual_labels != expected_labels:
        raise ValueError(
            "Label test set tidak lengkap atau tidak sesuai.\n"
            f"Expected: {sorted(expected_labels)}\n"
            f"Actual  : {sorted(actual_labels)}"
        )

    return result


# =============================================================================
# MEMUAT VOCABULARY
# =============================================================================

def load_vocabulary() -> list[str]:
    """
    Membaca vocabulary berdasarkan indeks token.
    """

    if not VOCABULARY_PATH.exists():
        raise FileNotFoundError(
            "Vocabulary tidak ditemukan:\n"
            f"{VOCABULARY_PATH}"
        )

    if not VOCABULARY_PATH.is_file():
        raise ValueError(
            "Path vocabulary bukan file:\n"
            f"{VOCABULARY_PATH}"
        )

    with open(
        VOCABULARY_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        vocabulary = (
            file.read()
            .splitlines()
        )

    if not vocabulary:
        raise ValueError(
            "Vocabulary kosong."
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
    Mengubah token ID menjadi teks.
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
        .replace(
            "[",
            "",
        )
        .replace(
            "]",
            "",
        )
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

    if not LABEL_MAPPING_PATH.is_file():
        raise ValueError(
            "Path label mapping bukan file:\n"
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
        index_to_label = {
            int(index): str(label)
            for index, label
            in kompas_mapping[
                "index_to_label"
            ].items()
        }

    elif (
        "label_to_index"
        in kompas_mapping
    ):
        index_to_label = {
            int(index): str(label)
            for label, index
            in kompas_mapping[
                "label_to_index"
            ].items()
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
# MEMUAT SHAP VALUES HASIL GLOBAL
# =============================================================================

def load_global_shap_values() -> dict[str, np.ndarray]:
    """
    Memuat SHAP values yang disimpan oleh 01_shap_global.py.

    Format utama:
    - shap_values:
      (samples, sequence_length, classes)

    - expected_values:
      (samples, classes)

    - token_sequences:
      (samples, sequence_length)

    - probabilities:
      (samples, classes)
    """

    if not GLOBAL_SHAP_VALUES_PATH.exists():
        raise FileNotFoundError(
            "File global SHAP values tidak ditemukan:\n"
            f"{GLOBAL_SHAP_VALUES_PATH}\n\n"
            "Jalankan 01_shap_global.py terlebih dahulu."
        )

    if not GLOBAL_SHAP_VALUES_PATH.is_file():
        raise ValueError(
            "Path global SHAP bukan file:\n"
            f"{GLOBAL_SHAP_VALUES_PATH}"
        )

    if GLOBAL_SHAP_VALUES_PATH.stat().st_size <= 0:
        raise ValueError(
            "File global SHAP ditemukan, tetapi kosong:\n"
            f"{GLOBAL_SHAP_VALUES_PATH}"
        )

    with np.load(
        GLOBAL_SHAP_VALUES_PATH,
        allow_pickle=False,
    ) as data:

        required_keys = {
            "shap_values",
            "expected_values",
            "token_sequences",
            "explain_indices",
            "probabilities",
            "predicted_classes",
            "actual_classes",
        }

        missing_keys = (
            required_keys
            - set(data.files)
        )

        if missing_keys:
            raise KeyError(
                "Komponen file SHAP tidak lengkap.\n"
                f"Key hilang: {sorted(missing_keys)}"
            )

        result = {
            "shap_values": np.asarray(
                data["shap_values"],
                dtype=np.float64,
            ),

            "expected_values": np.asarray(
                data["expected_values"],
                dtype=np.float64,
            ),

            "token_sequences": np.asarray(
                data["token_sequences"],
                dtype=np.int32,
            ),

            "explain_indices": np.asarray(
                data["explain_indices"],
                dtype=np.int32,
            ),

            "probabilities": np.asarray(
                data["probabilities"],
                dtype=np.float64,
            ),

            "predicted_classes": np.asarray(
                data["predicted_classes"],
                dtype=np.int32,
            ),

            "actual_classes": np.asarray(
                data["actual_classes"],
                dtype=np.int32,
            ),
        }

        if (
            "max_additivity_errors"
            in data.files
        ):
            result[
                "max_additivity_errors"
            ] = np.asarray(
                data[
                    "max_additivity_errors"
                ],
                dtype=np.float64,
            )

        if (
            "num_explained_tokens"
            in data.files
        ):
            result[
                "num_explained_tokens"
            ] = np.asarray(
                data[
                    "num_explained_tokens"
                ],
                dtype=np.int32,
            )

    sample_count = len(
        result[
            "explain_indices"
        ]
    )

    expected_shapes = {
        "shap_values": (
            sample_count,
            MAX_SEQUENCE_LENGTH,
            NUM_CLASSES,
        ),

        "expected_values": (
            sample_count,
            NUM_CLASSES,
        ),

        "token_sequences": (
            sample_count,
            MAX_SEQUENCE_LENGTH,
        ),

        "probabilities": (
            sample_count,
            NUM_CLASSES,
        ),

        "predicted_classes": (
            sample_count,
        ),

        "actual_classes": (
            sample_count,
        ),
    }

    for key, expected_shape in (
        expected_shapes.items()
    ):
        actual_shape = result[
            key
        ].shape

        if actual_shape != expected_shape:
            raise ValueError(
                f"Shape {key} tidak sesuai.\n"
                f"Expected: {expected_shape}\n"
                f"Actual  : {actual_shape}"
            )

    if sample_count <= 0:
        raise ValueError(
            "File SHAP tidak memiliki sampel."
        )

    if not np.all(
        np.isfinite(
            result[
                "shap_values"
            ]
        )
    ):
        raise ValueError(
            "SHAP values mengandung NaN atau infinite."
        )

    if not np.all(
        np.isfinite(
            result[
                "expected_values"
            ]
        )
    ):
        raise ValueError(
            "Expected values mengandung NaN atau infinite."
        )

    if not np.all(
        np.isfinite(
            result[
                "probabilities"
            ]
        )
    ):
        raise ValueError(
            "Probabilities mengandung NaN atau infinite."
        )

    if (
        result[
            "token_sequences"
        ]
        < 0
    ).any():
        raise ValueError(
            "Token sequences SHAP mengandung token ID negatif."
        )

    if (
        result[
            "probabilities"
        ]
        < -1e-7
    ).any():
        raise ValueError(
            "Probabilities mengandung nilai negatif."
        )

    if (
        result[
            "probabilities"
        ]
        > 1.0 + 1e-7
    ).any():
        raise ValueError(
            "Probabilities mengandung nilai melebihi 1."
        )

    probability_sums = result[
        "probabilities"
    ].sum(
        axis=1
    )

    if not np.allclose(
        probability_sums,
        1.0,
        atol=1e-4,
    ):
        raise ValueError(
            "Jumlah probabilitas setiap sampel "
            "tidak mendekati 1."
        )

    calculated_predictions = np.argmax(
        result[
            "probabilities"
        ],
        axis=1,
    ).astype(
        np.int32
    )

    if not np.array_equal(
        calculated_predictions,
        result[
            "predicted_classes"
        ],
    ):
        raise ValueError(
            "Predicted classes tidak konsisten "
            "dengan probabilitas yang tersimpan."
        )

    reconstructed_probabilities = (
        result[
            "expected_values"
        ]
        + result[
            "shap_values"
        ].sum(
            axis=1
        )
    )

    additivity_errors = np.abs(
        reconstructed_probabilities
        - result[
            "probabilities"
        ]
    )

    maximum_additivity_error = float(
        additivity_errors.max()
    )

    if (
        maximum_additivity_error
        > ADDITIVITY_TOLERANCE
    ):
        raise ValueError(
            "Additivity error global SHAP melebihi toleransi.\n"
            f"Maximum error : {maximum_additivity_error:.8f}\n"
            f"Tolerance     : {ADDITIVITY_TOLERANCE:.8f}"
        )

    result[
        "calculated_additivity_errors"
    ] = additivity_errors.max(
        axis=1
    )

    return result


# =============================================================================
# VALIDASI KONSISTENSI TEST DAN SHAP
# =============================================================================

def validate_shap_consistency(
    test_data: dict[str, np.ndarray],
    shap_data: dict[str, np.ndarray],
    vocabulary: list[str],
) -> None:
    """
    Memastikan data dalam file SHAP konsisten dengan test set.
    """

    explain_indices = shap_data[
        "explain_indices"
    ]

    if (
        explain_indices
        < 0
    ).any():
        raise ValueError(
            "Ditemukan explain index negatif."
        )

    if (
        explain_indices
        >= len(
            test_data[
                "X"
            ]
        )
    ).any():
        raise ValueError(
            "Ditemukan explain index melebihi jumlah test set."
        )

    if (
        len(
            np.unique(
                explain_indices
            )
        )
        != len(
            explain_indices
        )
    ):
        raise ValueError(
            "Ditemukan explain index duplikat."
        )

    expected_sequences = test_data[
        "X"
    ][
        explain_indices
    ]

    if not np.array_equal(
        expected_sequences,
        shap_data[
            "token_sequences"
        ],
    ):
        raise ValueError(
            "Token sequence pada file SHAP tidak sama "
            "dengan sequence pada test set."
        )

    expected_actual_classes = test_data[
        "y"
    ][
        explain_indices
    ]

    if not np.array_equal(
        expected_actual_classes,
        shap_data[
            "actual_classes"
        ],
    ):
        raise ValueError(
            "Actual classes pada file SHAP tidak sama "
            "dengan label test set."
        )

    max_token_id = int(
        shap_data[
            "token_sequences"
        ].max()
    )

    if max_token_id >= len(
        vocabulary
    ):
        raise ValueError(
            "Vocabulary tidak sesuai dengan token ID SHAP.\n"
            f"Maximum token ID : {max_token_id}\n"
            f"Vocabulary size  : {len(vocabulary)}"
        )


# =============================================================================
# PEMILIHAN SAMPEL LOCAL
# =============================================================================

def select_local_samples(
    shap_data: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """
    Memilih sampel local explanation.

    Prioritas:
    1. Satu prediksi benar dengan confidence tertinggi per kelas.
    2. Satu prediksi salah dengan confidence tertinggi berdasarkan
       kelas aktual, jika tersedia.
    """

    actual_classes = shap_data[
        "actual_classes"
    ]

    predicted_classes = shap_data[
        "predicted_classes"
    ]

    probabilities = shap_data[
        "probabilities"
    ]

    confidences = np.max(
        probabilities,
        axis=1,
    )

    selected: list[
        dict[str, Any]
    ] = []

    # -------------------------------------------------------------------------
    # SATU PREDIKSI BENAR PER KELAS
    # -------------------------------------------------------------------------

    for class_index in range(
        NUM_CLASSES
    ):
        candidates = np.where(
            (
                actual_classes
                == class_index
            )
            & (
                predicted_classes
                == class_index
            )
        )[0]

        if len(
            candidates
        ) == 0:
            continue

        selected_position = int(
            candidates[
                np.argmax(
                    confidences[
                        candidates
                    ]
                )
            ]
        )

        selected.append(
            {
                "shap_sample_position":
                    selected_position,

                "selection_type":
                    "correct_high_confidence",

                "selection_class_index":
                    class_index,

                "selection_confidence":
                    float(
                        confidences[
                            selected_position
                        ]
                    ),
            }
        )

    # -------------------------------------------------------------------------
    # SATU PREDIKSI SALAH PER KELAS AKTUAL
    # -------------------------------------------------------------------------

    for class_index in range(
        NUM_CLASSES
    ):
        candidates = np.where(
            (
                actual_classes
                == class_index
            )
            & (
                predicted_classes
                != class_index
            )
        )[0]

        if len(
            candidates
        ) == 0:
            continue

        selected_position = int(
            candidates[
                np.argmax(
                    confidences[
                        candidates
                    ]
                )
            ]
        )

        selected.append(
            {
                "shap_sample_position":
                    selected_position,

                "selection_type":
                    "incorrect_high_confidence",

                "selection_class_index":
                    class_index,

                "selection_confidence":
                    float(
                        confidences[
                            selected_position
                        ]
                    ),
            }
        )

    # -------------------------------------------------------------------------
    # MENGHILANGKAN DUPLIKASI
    # -------------------------------------------------------------------------

    unique_selected: list[
        dict[str, Any]
    ] = []

    used_positions: set[int] = set()

    for item in selected:
        position = int(
            item[
                "shap_sample_position"
            ]
        )

        if position in used_positions:
            continue

        used_positions.add(
            position
        )

        unique_selected.append(
            item
        )

    return unique_selected[
        :MAX_LOCAL_SAMPLES
    ]


# =============================================================================
# KONTRIBUSI TOKEN LOCAL
# =============================================================================

def calculate_local_token_contributions(
    shap_sample_position: int,
    selection_type: str,
    test_data: dict[str, np.ndarray],
    shap_data: dict[str, np.ndarray],
    vocabulary: list[str],
    index_to_label: dict[int, str],
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Menghitung kontribusi token terhadap kelas prediksi.

    SHAP values terbaru memiliki shape:
    (samples, positions, output_classes)

    Untuk kelas prediksi:
    sample_shap = shap_values[
        shap_sample_position,
        :,
        predicted_class
    ]

    Nilai positif:
    token mendukung kelas prediksi.

    Nilai negatif:
    token menurunkan probabilitas kelas prediksi.
    """

    explain_indices = shap_data[
        "explain_indices"
    ]

    original_test_index = int(
        explain_indices[
            shap_sample_position
        ]
    )

    token_sequence = shap_data[
        "token_sequences"
    ][
        shap_sample_position
    ]

    probabilities = shap_data[
        "probabilities"
    ][
        shap_sample_position
    ]

    predicted_class = int(
        shap_data[
            "predicted_classes"
        ][
            shap_sample_position
        ]
    )

    actual_class = int(
        shap_data[
            "actual_classes"
        ][
            shap_sample_position
        ]
    )

    expected_values = shap_data[
        "expected_values"
    ][
        shap_sample_position
    ]

    # SHAP terhadap output kelas yang diprediksi.
    signed_contributions = shap_data[
        "shap_values"
    ][
        shap_sample_position,
        :,
        predicted_class,
    ]

    signed_contributions = np.asarray(
        signed_contributions,
        dtype=np.float64,
    )

    absolute_contributions = np.abs(
        signed_contributions
    )

    predicted_probability = float(
        probabilities[
            predicted_class
        ]
    )

    baseline_probability = float(
        expected_values[
            predicted_class
        ]
    )

    reconstructed_probability = float(
        baseline_probability
        + signed_contributions.sum()
    )

    local_additivity_error = abs(
        reconstructed_probability
        - predicted_probability
    )

    if (
        local_additivity_error
        > ADDITIVITY_TOLERANCE
    ):
        raise ValueError(
            "Local additivity error melebihi toleransi.\n"
            f"Sample position : {shap_sample_position}\n"
            f"Error           : {local_additivity_error:.8f}\n"
            f"Tolerance       : {ADDITIVITY_TOLERANCE:.8f}"
        )

    position_rows: list[
        dict[str, Any]
    ] = []

    for token_position, token_id_value in enumerate(
        token_sequence
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

        if is_special_token(
            token
        ):
            continue

        signed_value = float(
            signed_contributions[
                token_position
            ]
        )

        absolute_value = float(
            absolute_contributions[
                token_position
            ]
        )

        if signed_value > 0.0:
            direction = (
                "mendukung_prediksi"
            )

        elif signed_value < 0.0:
            direction = (
                "menahan_prediksi"
            )

        else:
            direction = "netral"

        position_rows.append(
            {
                "shap_sample_position":
                    shap_sample_position,

                "original_test_index":
                    original_test_index,

                "document_id":
                    str(
                        test_data[
                            "document_id"
                        ][
                            original_test_index
                        ]
                    ),

                "category_from_npz":
                    str(
                        test_data[
                            "category"
                        ][
                            original_test_index
                        ]
                    ),

                "selection_type":
                    selection_type,

                "actual_index":
                    actual_class,

                "actual_label":
                    index_to_label[
                        actual_class
                    ],

                "predicted_index":
                    predicted_class,

                "predicted_label":
                    index_to_label[
                        predicted_class
                    ],

                "token_position":
                    token_position,

                "token_id":
                    token_id,

                "token":
                    token,

                "signed_shap":
                    signed_value,

                "absolute_shap":
                    absolute_value,

                "direction":
                    direction,
            }
        )

    position_dataframe = pd.DataFrame(
        position_rows
    )

    if position_dataframe.empty:
        raise ValueError(
            "Tidak ada token semantik untuk sampel "
            f"posisi {shap_sample_position}."
        )

    position_dataframe = (
        position_dataframe
        .sort_values(
            "absolute_shap",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    position_dataframe.insert(
        0,
        "position_importance_rank",
        np.arange(
            1,
            len(
                position_dataframe
            ) + 1,
        ),
    )

    # -------------------------------------------------------------------------
    # AGREGASI TOKEN YANG MUNCUL BERULANG
    # -------------------------------------------------------------------------

    aggregated_dataframe = (
        position_dataframe
        .groupby(
            [
                "shap_sample_position",
                "original_test_index",
                "document_id",
                "category_from_npz",
                "selection_type",
                "actual_index",
                "actual_label",
                "predicted_index",
                "predicted_label",
                "token_id",
                "token",
            ],
            as_index=False,
        )
        .agg(
            signed_shap=(
                "signed_shap",
                "sum",
            ),

            absolute_shap=(
                "absolute_shap",
                "sum",
            ),

            mean_signed_shap=(
                "signed_shap",
                "mean",
            ),

            mean_absolute_shap=(
                "absolute_shap",
                "mean",
            ),

            occurrence_count=(
                "token_position",
                "size",
            ),

            token_positions=(
                "token_position",
                lambda values:
                ",".join(
                    str(
                        int(value)
                    )
                    for value in sorted(
                        values
                    )
                ),
            ),
        )
    )

    aggregated_dataframe[
        "direction"
    ] = np.where(
        aggregated_dataframe[
            "signed_shap"
        ]
        > 0.0,
        "mendukung_prediksi",
        np.where(
            aggregated_dataframe[
                "signed_shap"
            ]
            < 0.0,
            "menahan_prediksi",
            "netral",
        ),
    )

    aggregated_dataframe = (
        aggregated_dataframe
        .sort_values(
            [
                "absolute_shap",
                "signed_shap",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    aggregated_dataframe.insert(
        0,
        "importance_rank",
        np.arange(
            1,
            len(
                aggregated_dataframe
            ) + 1,
        ),
    )

    positive_tokens = aggregated_dataframe[
        aggregated_dataframe[
            "signed_shap"
        ]
        > 0.0
    ]

    negative_tokens = aggregated_dataframe[
        aggregated_dataframe[
            "signed_shap"
        ]
        < 0.0
    ]

    if not positive_tokens.empty:
        top_supporting_token = str(
            positive_tokens.iloc[0][
                "token"
            ]
        )

        top_supporting_shap = float(
            positive_tokens.iloc[0][
                "signed_shap"
            ]
        )

    else:
        top_supporting_token = ""
        top_supporting_shap = 0.0

    if not negative_tokens.empty:
        strongest_negative_index = (
            negative_tokens[
                "signed_shap"
            ].idxmin()
        )

        top_opposing_token = str(
            negative_tokens.loc[
                strongest_negative_index,
                "token",
            ]
        )

        top_opposing_shap = float(
            negative_tokens.loc[
                strongest_negative_index,
                "signed_shap",
            ]
        )

    else:
        top_opposing_token = ""
        top_opposing_shap = 0.0

    summary: dict[str, Any] = {
        "shap_sample_position":
            shap_sample_position,

        "original_test_index":
            original_test_index,

        "document_id":
            str(
                test_data[
                    "document_id"
                ][
                    original_test_index
                ]
            ),

        "category_from_npz":
            str(
                test_data[
                    "category"
                ][
                    original_test_index
                ]
            ),

        "selection_type":
            selection_type,

        "actual_index":
            actual_class,

        "actual_label":
            index_to_label[
                actual_class
            ],

        "predicted_index":
            predicted_class,

        "predicted_label":
            index_to_label[
                predicted_class
            ],

        "is_correct":
            actual_class
            == predicted_class,

        "prediction_confidence":
            predicted_probability,

        "baseline_probability_predicted_class":
            baseline_probability,

        "reconstructed_probability_predicted_class":
            reconstructed_probability,

        "local_additivity_error":
            local_additivity_error,

        "probability_bola":
            float(
                probabilities[0]
            ),

        "probability_global":
            float(
                probabilities[1]
            ),

        "probability_money":
            float(
                probabilities[2]
            ),

        "probability_tekno":
            float(
                probabilities[3]
            ),

        "number_of_semantic_token_occurrences":
            int(
                len(
                    position_dataframe
                )
            ),

        "number_of_unique_semantic_tokens":
            int(
                len(
                    aggregated_dataframe
                )
            ),

        "total_positive_shap":
            float(
                aggregated_dataframe.loc[
                    aggregated_dataframe[
                        "signed_shap"
                    ]
                    > 0.0,
                    "signed_shap",
                ].sum()
            ),

        "total_negative_shap":
            float(
                aggregated_dataframe.loc[
                    aggregated_dataframe[
                        "signed_shap"
                    ]
                    < 0.0,
                    "signed_shap",
                ].sum()
            ),

        "top_supporting_token":
            top_supporting_token,

        "top_supporting_shap":
            top_supporting_shap,

        "top_opposing_token":
            top_opposing_token,

        "top_opposing_shap":
            top_opposing_shap,
    }

    return (
        summary,
        aggregated_dataframe,
        position_dataframe,
    )


# =============================================================================
# UTILITAS NAMA FILE
# =============================================================================

def sanitize_filename_component(
    value: Any,
) -> str:
    """
    Membersihkan teks agar aman digunakan sebagai nama file.
    """

    text = str(
        value
    ).strip()

    text = re.sub(
        r'[<>:"/\\|?*]+',
        "_",
        text,
    )

    text = re.sub(
        r"\s+",
        "_",
        text,
    )

    text = re.sub(
        r"_+",
        "_",
        text,
    )

    text = text.strip(
        "._"
    )

    if not text:
        return "unknown"

    return text


# =============================================================================
# MEMBUAT GRAFIK LOCAL SHAP
# =============================================================================

def plot_local_explanation(
    summary: dict[str, Any],
    token_contributions: pd.DataFrame,
) -> Path:
    """
    Membuat grafik kontribusi token terhadap kelas prediksi.

    Nilai positif:
    token mendukung probabilitas kelas prediksi.

    Nilai negatif:
    token menurunkan probabilitas kelas prediksi.
    """

    if token_contributions.empty:
        raise ValueError(
            "Data kontribusi token kosong."
        )

    plot_data = (
        token_contributions
        .head(
            TOP_N_TOKENS
        )
        .copy()
        .sort_values(
            "signed_shap",
            ascending=True,
        )
    )

    output_name = (
        f"{sanitize_filename_component(summary['document_id'])}_"
        f"{sanitize_filename_component(summary['selection_type'])}_"
        f"actual_{sanitize_filename_component(summary['actual_label'])}_"
        f"pred_{sanitize_filename_component(summary['predicted_label'])}.png"
    )

    output_path = (
        LOCAL_FIGURES_DIR
        / output_name
    )

    figure, axis = plt.subplots(
        figsize=(11, 8)
    )

    axis.barh(
        plot_data[
            "token"
        ],
        plot_data[
            "signed_shap"
        ],
    )

    axis.axvline(
        0.0,
        linewidth=1,
    )

    axis.set_xlabel(
        "Signed SHAP Contribution terhadap Kelas Prediksi"
    )

    axis.set_ylabel(
        "Token"
    )

    status_text = (
        "BENAR"
        if summary[
            "is_correct"
        ]
        else "SALAH"
    )

    axis.set_title(
        "Local SHAP Explanation — CNN K2\n"
        f"Actual: {summary['actual_label']} | "
        f"Prediksi: {summary['predicted_label']} | "
        f"Confidence: {summary['prediction_confidence']:.2%} | "
        f"{status_text}"
    )

    axis.grid(
        axis="x",
        alpha=0.3,
    )

    for patch in axis.patches:
        width = patch.get_width()

        if width >= 0.0:
            horizontal_alignment = "left"
            x_position = width

        else:
            horizontal_alignment = "right"
            x_position = width

        axis.annotate(
            f"{width:.4f}",
            xy=(
                x_position,
                patch.get_y()
                + patch.get_height()
                / 2.0,
            ),
            xytext=(
                4
                if width >= 0.0
                else -4,
                0,
            ),
            textcoords="offset points",
            ha=horizontal_alignment,
            va="center",
            fontsize=8,
        )

    figure.tight_layout()

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
# MENYIMPAN KONFIGURASI
# =============================================================================

def save_configuration(
    selected_samples: list[dict[str, Any]],
    summary_dataframe: pd.DataFrame,
) -> None:
    """
    Menyimpan konfigurasi dan metodologi local SHAP.
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

        "source_shap_values":
            str(
                GLOBAL_SHAP_VALUES_PATH
            ),

        "shap_method":
            "shap.KernelExplainer",

        "explained_representation":
            (
                "Binary token-presence masks applied "
                "to integer token sequences."
            ),

        "baseline":
            (
                "All non-padding tokens are replaced "
                "with token ID 0."
            ),

        "source_shap_shape":
            (
                "samples x sequence positions x output classes"
            ),

        "selection_strategy":
            (
                "One highest-confidence correct prediction per class "
                "and one highest-confidence incorrect prediction per "
                "actual class when available."
            ),

        "maximum_local_samples":
            MAX_LOCAL_SAMPLES,

        "selected_sample_count":
            len(
                selected_samples
            ),

        "correct_sample_count":
            int(
                summary_dataframe[
                    "is_correct"
                ].sum()
            ),

        "incorrect_sample_count":
            int(
                (
                    ~summary_dataframe[
                        "is_correct"
                    ]
                ).sum()
            ),

        "top_n_tokens_per_plot":
            TOP_N_TOKENS,

        "token_contribution_method":
            (
                "The signed SHAP value for the predicted output class "
                "is used directly as the token contribution."
            ),

        "repeated_token_aggregation":
            (
                "Signed SHAP values and absolute SHAP values are summed "
                "for repeated occurrences of the same token."
            ),

        "importance_ranking":
            (
                "Tokens are ranked by the sum of absolute SHAP values."
            ),

        "special_tokens_excluded":
            sorted(
                SPECIAL_TOKENS
            ),

        "additivity_tolerance":
            ADDITIVITY_TOLERANCE,

        "maximum_local_additivity_error":
            float(
                summary_dataframe[
                    "local_additivity_error"
                ].max()
            ),

        "interpretation_note":
            (
                "Positive SHAP values increase the probability of the "
                "predicted class. Negative SHAP values decrease the "
                "probability of the predicted class. Absolute SHAP "
                "shows contribution magnitude without direction."
            ),

        "selected_samples":
            [
                {
                    "shap_sample_position":
                        int(
                            item[
                                "shap_sample_position"
                            ]
                        ),

                    "selection_type":
                        str(
                            item[
                                "selection_type"
                            ]
                        ),
                }
                for item in selected_samples
            ],
    }

    with open(
        LOCAL_CONFIGURATION_PATH,
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
    Menjalankan local SHAP explanation.
    """

    print("=" * 80)

    print(
        "STEP 7.2 - LOCAL SHAP EXPLANATION"
    )

    print("=" * 80)

    create_output_directories()

    # =========================================================================
    # MEMUAT DATA
    # =========================================================================

    print(
        "\nMemuat test set..."
    )

    test_data = load_test_data()

    print(
        f"Test shape               : "
        f"{test_data['X'].shape}"
    )

    print(
        "\nMemuat vocabulary..."
    )

    vocabulary = load_vocabulary()

    print(
        f"Vocabulary size          : "
        f"{len(vocabulary):,}"
    )

    print(
        "\nMemuat label mapping..."
    )

    index_to_label = (
        load_index_to_label()
    )

    print(
        f"Label mapping            : "
        f"{index_to_label}"
    )

    print(
        "\nMemuat global SHAP values..."
    )

    shap_data = (
        load_global_shap_values()
    )

    print(
        f"SHAP values shape        : "
        f"{shap_data['shap_values'].shape}"
    )

    print(
        f"Expected values shape    : "
        f"{shap_data['expected_values'].shape}"
    )

    print(
        f"Jumlah sampel tersedia   : "
        f"{len(shap_data['explain_indices'])}"
    )

    print(
        "Memvalidasi konsistensi test dan SHAP..."
    )

    validate_shap_consistency(
        test_data=test_data,
        shap_data=shap_data,
        vocabulary=vocabulary,
    )

    print(
        "Konsistensi data         : valid"
    )

    print(
        f"Max additivity error     : "
        f"{shap_data['calculated_additivity_errors'].max():.8f}"
    )

    # =========================================================================
    # PEMILIHAN SAMPEL
    # =========================================================================

    print(
        "\nMemilih sampel local explanation..."
    )

    selected_samples = (
        select_local_samples(
            shap_data=shap_data
        )
    )

    if not selected_samples:
        raise ValueError(
            "Tidak ada sampel local explanation "
            "yang dapat dipilih."
        )

    print(
        f"Jumlah sampel dipilih    : "
        f"{len(selected_samples)}"
    )

    # =========================================================================
    # MENGHITUNG LOCAL EXPLANATION
    # =========================================================================

    summary_rows: list[
        dict[str, Any]
    ] = []

    contribution_dataframes: list[
        pd.DataFrame
    ] = []

    position_dataframes: list[
        pd.DataFrame
    ] = []

    success_count = 0
    failed_count = 0

    for number, selected in enumerate(
        selected_samples,
        start=1,
    ):
        shap_sample_position = int(
            selected[
                "shap_sample_position"
            ]
        )

        selection_type = str(
            selected[
                "selection_type"
            ]
        )

        print(
            "\n" + "-" * 80
        )

        print(
            f"Sampel "
            f"{number}/{len(selected_samples)}"
        )

        try:
            (
                summary,
                token_contributions,
                position_contributions,
            ) = calculate_local_token_contributions(
                shap_sample_position=(
                    shap_sample_position
                ),
                selection_type=(
                    selection_type
                ),
                test_data=test_data,
                shap_data=shap_data,
                vocabulary=vocabulary,
                index_to_label=index_to_label,
            )

            figure_path = (
                plot_local_explanation(
                    summary=summary,
                    token_contributions=(
                        token_contributions
                    ),
                )
            )

            summary[
                "figure_path"
            ] = str(
                figure_path
            )

            summary_rows.append(
                summary
            )

            contribution_dataframes.append(
                token_contributions
            )

            position_dataframes.append(
                position_contributions
            )

            success_count += 1

            print(
                f"SHAP sample position    : "
                f"{shap_sample_position}"
            )

            print(
                f"Original test index     : "
                f"{summary['original_test_index']}"
            )

            print(
                f"Document ID             : "
                f"{summary['document_id']}"
            )

            print(
                f"Jenis sampel            : "
                f"{selection_type}"
            )

            print(
                f"Label aktual            : "
                f"{summary['actual_label']}"
            )

            print(
                f"Label prediksi          : "
                f"{summary['predicted_label']}"
            )

            print(
                f"Prediksi benar          : "
                f"{summary['is_correct']}"
            )

            print(
                f"Confidence              : "
                f"{summary['prediction_confidence']:.2%}"
            )

            print(
                f"Baseline probability    : "
                f"{summary['baseline_probability_predicted_class']:.6f}"
            )

            print(
                f"Additivity error        : "
                f"{summary['local_additivity_error']:.8f}"
            )

            print(
                "Token paling berpengaruh:"
            )

            display_columns = [
                "importance_rank",
                "token",
                "signed_shap",
                "absolute_shap",
                "occurrence_count",
                "direction",
            ]

            print(
                "\n"
                + token_contributions[
                    display_columns
                ]
                .head(
                    8
                )
                .to_string(
                    index=False
                )
            )

            print(
                f"Grafik                  : "
                f"{figure_path}"
            )

        except Exception as error:
            failed_count += 1

            print(
                "Gagal memproses sampel:"
            )

            print(
                str(
                    error
                )
            )

    if not summary_rows:
        raise RuntimeError(
            "Seluruh sampel local SHAP gagal diproses."
        )

    # =========================================================================
    # MEMBENTUK DATAFRAME
    # =========================================================================

    summary_dataframe = pd.DataFrame(
        summary_rows
    )

    all_contributions = pd.concat(
        contribution_dataframes,
        ignore_index=True,
    )

    all_position_contributions = pd.concat(
        position_dataframes,
        ignore_index=True,
    )

    summary_dataframe = (
        summary_dataframe
        .sort_values(
            [
                "selection_type",
                "actual_index",
                "predicted_index",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    all_contributions = (
        all_contributions
        .sort_values(
            [
                "shap_sample_position",
                "importance_rank",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    all_position_contributions = (
        all_position_contributions
        .sort_values(
            [
                "shap_sample_position",
                "position_importance_rank",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # =========================================================================
    # MENYIMPAN OUTPUT
    # =========================================================================

    summary_dataframe.to_csv(
        LOCAL_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    all_contributions.to_csv(
        LOCAL_TOKEN_CONTRIBUTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    all_position_contributions.to_csv(
        LOCAL_TOKEN_POSITION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    save_configuration(
        selected_samples=selected_samples,
        summary_dataframe=summary_dataframe,
    )

    # =========================================================================
    # MENAMPILKAN RINGKASAN
    # =========================================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "RINGKASAN LOCAL SHAP"
    )

    print("=" * 80)

    display_summary_columns = [
        "document_id",
        "selection_type",
        "actual_label",
        "predicted_label",
        "is_correct",
        "prediction_confidence",
        "top_supporting_token",
        "top_opposing_token",
        "local_additivity_error",
    ]

    print(
        "\n"
        + summary_dataframe[
            display_summary_columns
        ]
        .to_string(
            index=False
        )
    )

    print(
        f"\nSampel berhasil          : "
        f"{success_count}"
    )

    print(
        f"Sampel gagal             : "
        f"{failed_count}"
    )

    print(
        "\nTabel ringkasan local SHAP:"
    )

    print(
        LOCAL_SUMMARY_PATH
    )

    print(
        "\nTabel kontribusi token agregat:"
    )

    print(
        LOCAL_TOKEN_CONTRIBUTIONS_PATH
    )

    print(
        "\nTabel kontribusi setiap posisi token:"
    )

    print(
        LOCAL_TOKEN_POSITION_PATH
    )

    print(
        "\nFolder grafik local SHAP:"
    )

    print(
        LOCAL_FIGURES_DIR
    )

    print(
        "\nKonfigurasi local SHAP:"
    )

    print(
        LOCAL_CONFIGURATION_PATH
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "Tahap local SHAP selesai."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
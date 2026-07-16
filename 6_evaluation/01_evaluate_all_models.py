# =============================================================================
# STEP 6.1 - EVALUATE ALL MODELS ON TEST SET
# =============================================================================
# File:
# 6_evaluation/01_evaluate_all_models.py
#
# Tujuan:
# Mengevaluasi seluruh checkpoint terbaik CNN dan Attention-BiLSTM
# menggunakan test set.
#
# Model yang dievaluasi:
# - CNN: K1, K2, K3, K4, A1, A2
# - Attention-BiLSTM: K1, K2, K3, K4, A1, A2
#
# Output utama:
# - Metrik keseluruhan setiap model
# - Metrik per kelas
# - Hasil prediksi setiap artikel
# - Ringkasan evaluasi JSON
#
# Catatan:
# Test set tidak digunakan untuk training, early stopping,
# atau pemilihan epoch terbaik.
# =============================================================================

from __future__ import annotations

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

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


# =============================================================================
# PROJECT ROOT DAN PYTHON PATH
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
# IMPORT MODULE PROJECT
# =============================================================================
#
# AttentionPooling1D harus diimpor agar TensorFlow dapat mengenali
# custom layer saat memuat model Attention-BiLSTM.

from attention_bilstm_model import (  # noqa: E402
    AttentionPooling1D,
)

from training_config import (  # noqa: E402
    BATCH_SIZE,
    NUM_CLASSES,
    RANDOM_SEED,
    get_checkpoint_path,
    get_scenario_config,
    get_split_path,
)


# =============================================================================
# KONFIGURASI MODEL DAN SKENARIO
# =============================================================================

MODEL_NAMES = [
    "cnn",
    "attention_bilstm",
]

SCENARIO_CODES = [
    "K1",
    "K2",
    "K3",
    "A1",
    "A2",
]

EVALUATION_BATCH_SIZE = BATCH_SIZE

PREDICTION_VERBOSE = 0


# =============================================================================
# PATH OUTPUT
# =============================================================================

RESULTS_DIR = (
    PROJECT_ROOT
    / "9_results"
)

METRICS_DIR = (
    RESULTS_DIR
    / "metrics"
)

PREDICTIONS_DIR = (
    RESULTS_DIR
    / "predictions"
)

EVALUATION_LOGS_DIR = (
    RESULTS_DIR
    / "evaluation_logs"
)

TABLES_DIR = (
    RESULTS_DIR
    / "tables"
)

OVERALL_METRICS_PATH = (
    METRICS_DIR
    / "model_test_metrics.csv"
)

PER_CLASS_METRICS_PATH = (
    METRICS_DIR
    / "model_test_per_class_metrics.csv"
)

CONFUSION_MATRIX_DATA_PATH = (
    METRICS_DIR
    / "model_test_confusion_matrix_data.csv"
)

EVALUATION_STATUS_PATH = (
    EVALUATION_LOGS_DIR
    / "model_evaluation_status.csv"
)

EVALUATION_SUMMARY_PATH = (
    EVALUATION_LOGS_DIR
    / "model_evaluation_summary.json"
)

LABEL_MAPPING_PATH = (
    TABLES_DIR
    / "label_mapping.json"
)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_global_seed(
    seed: int = RANDOM_SEED,
) -> None:
    """
    Mengatur random seed untuk NumPy dan TensorFlow.
    """

    np.random.seed(seed)

    tf.random.set_seed(seed)


# =============================================================================
# MEMBUAT FOLDER OUTPUT
# =============================================================================

def create_output_directories() -> None:
    """
    Membuat seluruh folder output evaluasi.
    """

    directories = [
        METRICS_DIR,
        PREDICTIONS_DIR,
        EVALUATION_LOGS_DIR,
        TABLES_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# =============================================================================
# MEMBENTUK NAMA EKSPERIMEN
# =============================================================================

def get_experiment_name(
    model_name: str,
    scenario_code: str,
) -> str:
    """
    Membentuk nama eksperimen.

    Contoh:
    cnn_k1
    attention_bilstm_a2
    """

    return (
        f"{model_name}_"
        f"{scenario_code.lower()}"
    )


# =============================================================================
# MEMUAT LABEL MAPPING
# =============================================================================

def load_all_label_mappings() -> dict[str, Any]:
    """
    Membaca label_mapping.json hasil text vectorization.
    """

    if not LABEL_MAPPING_PATH.exists():
        raise FileNotFoundError(
            "File label mapping tidak ditemukan:\n"
            f"{LABEL_MAPPING_PATH}"
        )

    with open(
        LABEL_MAPPING_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        mappings = json.load(file)

    required_datasets = [
        "Kompas",
        "AG News",
    ]

    for dataset_name in required_datasets:
        if dataset_name not in mappings:
            raise KeyError(
                f"Mapping untuk {dataset_name} "
                "tidak ditemukan."
            )

    return mappings


def get_index_to_label(
    dataset_name: str,
    all_mappings: dict[str, Any],
) -> dict[int, str]:
    """
    Mengambil mapping indeks integer ke nama kategori.
    """

    dataset_mapping = all_mappings[
        dataset_name
    ]

    raw_mapping = dataset_mapping[
        "index_to_label"
    ]

    index_to_label = {
        int(index): str(label)
        for index, label
        in raw_mapping.items()
    }

    expected_indices = set(
        range(NUM_CLASSES)
    )

    actual_indices = set(
        index_to_label.keys()
    )

    if actual_indices != expected_indices:
        raise ValueError(
            f"Indeks label {dataset_name} "
            f"tidak sesuai.\n"
            f"Expected: {expected_indices}\n"
            f"Actual  : {actual_indices}"
        )

    return index_to_label


# =============================================================================
# MEMUAT TEST SET
# =============================================================================
def load_test_dataset(
    test_path: Path,
) -> dict[str, np.ndarray]:
    """
    Memuat seluruh komponen test set dari file NPZ.

    Komponen yang dibaca:
    - X
    - y
    - document_id
    - category
    """

    if not test_path.exists():
        raise FileNotFoundError(
            "Test set tidak ditemukan:\n"
            f"{test_path}"
        )

    with np.load(
        test_path,
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
                "Komponen test set tidak lengkap.\n"
                f"Key hilang: {missing_keys}"
            )

        test_data = {
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

    return test_data

# def load_test_dataset(
#     test_path: Path,
# ) -> dict[str, np.ndarray]:
#     """
#     Memuat seluruh komponen test set dari file NPZ.

#     Komponen yang dibaca:
#     - X
#     - y
#     - document_id
#     - category
#     """

#     if not test_path.exists():
#         raise FileNotFoundError(
#             "Test set tidak ditemukan:\n"
#             f"{test_path}"
#         )

#     with np.load(
#         test_path,
#         allow_pickle=False,
#     ) as data:

#         available_keys = set(
#             data.files
#         )

#         required_keys = {
#             "X",
#             "y",
#             "document_id",
#             "category",
#         }

#         missing_keys = (
#             required_keys
#             - available_keys
#         )

#         if missing_keys:
#             raise KeyError(
#                 "Komponen test set tidak lengkap.\n"
#                 f"Key hilang: {missing_keys}"
#             )

#         test_data = {
#             "X": data["X"],
#             "y": data["y"],
#             "document_id": data[
#                 "document_id"
#             ],
#             "category": data[
#                 "category"
#             ],
#         }

#     return test_data


# =============================================================================
# VALIDASI TEST SET
# =============================================================================

def validate_test_dataset(
    test_data: dict[str, np.ndarray],
    expected_sequence_length: int,
    scenario_code: str,
) -> None:
    """
    Memastikan test set sesuai dengan konfigurasi model.
    """

    X_test = test_data["X"]
    y_test = test_data["y"]
    document_ids = test_data[
        "document_id"
    ]
    categories = test_data[
        "category"
    ]

    if X_test.ndim != 2:
        raise ValueError(
            f"{scenario_code}: X_test harus "
            f"dua dimensi. Shape: {X_test.shape}"
        )

    if y_test.ndim != 1:
        raise ValueError(
            f"{scenario_code}: y_test harus "
            f"satu dimensi. Shape: {y_test.shape}"
        )

    jumlah_data = len(
        X_test
    )

    if jumlah_data == 0:
        raise ValueError(
            f"{scenario_code}: test set kosong."
        )

    component_lengths = {
        "X_test": len(X_test),
        "y_test": len(y_test),
        "document_id": len(document_ids),
        "category": len(categories),
    }

    if len(
        set(
            component_lengths.values()
        )
    ) != 1:
        raise ValueError(
            f"{scenario_code}: jumlah data "
            "setiap komponen tidak sama.\n"
            f"{component_lengths}"
        )

    if (
        X_test.shape[1]
        != expected_sequence_length
    ):
        raise ValueError(
            f"{scenario_code}: panjang sequence "
            "test tidak sesuai.\n"
            f"Expected: {expected_sequence_length}\n"
            f"Actual  : {X_test.shape[1]}"
        )

    unique_labels = np.unique(
        y_test
    )

    if np.min(unique_labels) < 0:
        raise ValueError(
            f"{scenario_code}: ditemukan "
            "label negatif."
        )

    if np.max(unique_labels) >= NUM_CLASSES:
        raise ValueError(
            f"{scenario_code}: label melebihi "
            "jumlah kelas."
        )


# =============================================================================
# MEMUAT MODEL
# =============================================================================

def load_best_model(
    model_name: str,
    scenario_code: str,
) -> tf.keras.Model:
    """
    Memuat checkpoint terbaik.

    compile=False digunakan karena model hanya dipakai
    untuk inferensi. Metrik dihitung menggunakan sklearn.
    """

    checkpoint_path = (
        get_checkpoint_path(
            model_name=model_name,
            scenario_code=scenario_code,
        )
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint model tidak ditemukan:\n"
            f"{checkpoint_path}"
        )

    custom_objects = {
        "AttentionPooling1D":
            AttentionPooling1D,
    }

    model = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects=custom_objects,
        compile=False,
    )

    return model


# =============================================================================
# VALIDASI OUTPUT MODEL
# =============================================================================

def validate_prediction_probabilities(
    probabilities: np.ndarray,
    expected_rows: int,
    experiment_name: str,
) -> None:
    """
    Memastikan output softmax valid.
    """

    expected_shape = (
        expected_rows,
        NUM_CLASSES,
    )

    if probabilities.shape != expected_shape:
        raise ValueError(
            f"{experiment_name}: shape probabilitas "
            f"tidak sesuai.\n"
            f"Expected: {expected_shape}\n"
            f"Actual  : {probabilities.shape}"
        )

    if not np.all(
        np.isfinite(probabilities)
    ):
        raise ValueError(
            f"{experiment_name}: ditemukan nilai "
            "probabilitas tidak valid."
        )

    probability_sums = probabilities.sum(
        axis=1
    )

    if not np.allclose(
        probability_sums,
        1.0,
        atol=1e-4,
    ):
        raise ValueError(
            f"{experiment_name}: jumlah "
            "probabilitas tidak mendekati 1."
        )


# =============================================================================
# MENGHITUNG METRIK KESELURUHAN
# =============================================================================

def calculate_overall_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:

    probabilities = np.clip(
        probabilities,
        1e-7,
        1.0 - 1e-7,
    )

    probabilities = (
        probabilities
        / probabilities.sum(
            axis=1,
            keepdims=True,
        )
    )
    
    """
    Menghitung metrik klasifikasi keseluruhan.
    """

    metrics = {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "precision_weighted": float(
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),

        "recall_weighted": float(
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),

        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),

        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=list(
                    range(NUM_CLASSES)
                ),
            )
        ),
    }

    return metrics


# =============================================================================
# MENGHITUNG METRIK PER KELAS
# =============================================================================

def calculate_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    index_to_label: dict[int, str],
    model_name: str,
    scenario_code: str,
    dataset_name: str,
    scenario_name: str,
) -> list[dict[str, Any]]:
    """
    Membentuk precision, recall, F1-score,
    dan support setiap kelas.
    """

    target_names = [
        index_to_label[index]
        for index in range(
            NUM_CLASSES
        )
    ]

    report = classification_report(
        y_true,
        y_pred,
        labels=list(
            range(NUM_CLASSES)
        ),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    rows: list[
        dict[str, Any]
    ] = []

    for class_index in range(
        NUM_CLASSES
    ):
        class_name = (
            index_to_label[
                class_index
            ]
        )

        class_metrics = report[
            class_name
        ]

        rows.append(
            {
                "experiment_name":
                    get_experiment_name(
                        model_name,
                        scenario_code,
                    ),

                "model":
                    model_name,

                "dataset":
                    dataset_name,

                "scenario_code":
                    scenario_code,

                "scenario_name":
                    scenario_name,

                "class_index":
                    class_index,

                "class_name":
                    class_name,

                "precision":
                    float(
                        class_metrics[
                            "precision"
                        ]
                    ),

                "recall":
                    float(
                        class_metrics[
                            "recall"
                        ]
                    ),

                "f1_score":
                    float(
                        class_metrics[
                            "f1-score"
                        ]
                    ),

                "support":
                    int(
                        class_metrics[
                            "support"
                        ]
                    ),
            }
        )

    return rows


# =============================================================================
# MEMBENTUK DATA CONFUSION MATRIX
# =============================================================================

def create_confusion_matrix_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    index_to_label: dict[int, str],
    model_name: str,
    scenario_code: str,
    dataset_name: str,
) -> list[dict[str, Any]]:
    """
    Menyimpan confusion matrix dalam format tabel panjang.

    Visualisasi gambarnya dibuat pada tahap berikutnya.
    """

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(NUM_CLASSES)
        ),
    )

    rows: list[
        dict[str, Any]
    ] = []

    for actual_index in range(
        NUM_CLASSES
    ):
        for predicted_index in range(
            NUM_CLASSES
        ):
            rows.append(
                {
                    "experiment_name":
                        get_experiment_name(
                            model_name,
                            scenario_code,
                        ),

                    "model":
                        model_name,

                    "dataset":
                        dataset_name,

                    "scenario_code":
                        scenario_code,

                    "actual_index":
                        actual_index,

                    "actual_class":
                        index_to_label[
                            actual_index
                        ],

                    "predicted_index":
                        predicted_index,

                    "predicted_class":
                        index_to_label[
                            predicted_index
                        ],

                    "count":
                        int(
                            matrix[
                                actual_index,
                                predicted_index,
                            ]
                        ),
                }
            )

    return rows


# =============================================================================
# MENYIMPAN HASIL PREDIKSI
# =============================================================================

def save_predictions(
    document_ids: np.ndarray,
    original_categories: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    index_to_label: dict[int, str],
    model_name: str,
    scenario_code: str,
    dataset_name: str,
    scenario_name: str,
) -> Path:
    """
    Menyimpan hasil prediksi setiap artikel.
    """

    experiment_name = (
        get_experiment_name(
            model_name,
            scenario_code,
        )
    )

    true_labels = [
        index_to_label[
            int(index)
        ]
        for index in y_true
    ]

    predicted_labels = [
        index_to_label[
            int(index)
        ]
        for index in y_pred
    ]

    prediction_dataframe = (
        pd.DataFrame(
            {
                "document_id":
                    document_ids.astype(str),

                "dataset":
                    dataset_name,

                "scenario_code":
                    scenario_code,

                "scenario_name":
                    scenario_name,

                "model":
                    model_name,

                "actual_index":
                    y_true.astype(int),

                "actual_label":
                    true_labels,

                "original_category":
                    original_categories.astype(str),

                "predicted_index":
                    y_pred.astype(int),

                "predicted_label":
                    predicted_labels,

                "is_correct":
                    (
                        y_true
                        == y_pred
                    ),

                "prediction_confidence":
                    probabilities.max(
                        axis=1
                    ),
            }
        )
    )

    for class_index in range(
        NUM_CLASSES
    ):
        class_name = (
            index_to_label[
                class_index
            ]
        )

        safe_class_name = (
            class_name
            .lower()
            .replace(
                " ",
                "_",
            )
            .replace(
                "/",
                "_",
            )
        )

        prediction_dataframe[
            f"probability_{safe_class_name}"
        ] = probabilities[
            :,
            class_index,
        ]

    output_path = (
        PREDICTIONS_DIR
        / f"{experiment_name}_predictions.csv"
    )

    prediction_dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    return output_path


# =============================================================================
# MENGEVALUASI SATU MODEL
# =============================================================================

def evaluate_single_model(
    model_name: str,
    scenario_code: str,
    all_label_mappings: dict[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """
    Mengevaluasi satu model pada satu skenario.
    """

    experiment_name = (
        get_experiment_name(
            model_name,
            scenario_code,
        )
    )

    scenario_config = (
        get_scenario_config(
            scenario_code
        )
    )

    dataset_name = scenario_config[
        "dataset"
    ]

    scenario_name = scenario_config[
        "scenario_name"
    ]

    max_sequence_length = (
        scenario_config[
            "max_sequence_length"
        ]
    )

    test_path = get_split_path(
        scenario_code,
        "test",
    )

    checkpoint_path = (
        get_checkpoint_path(
            model_name=model_name,
            scenario_code=scenario_code,
        )
    )

    index_to_label = (
        get_index_to_label(
            dataset_name,
            all_label_mappings,
        )
    )

    print("\n" + "=" * 80)
    print(
        f"EVALUASI: {experiment_name}"
    )
    print("=" * 80)

    print(
        f"Model              : "
        f"{model_name}"
    )

    print(
        f"Dataset            : "
        f"{dataset_name}"
    )

    print(
        f"Skenario           : "
        f"{scenario_code}"
    )

    print(
        f"Representasi       : "
        f"{scenario_name}"
    )

    print(
        f"Sequence length    : "
        f"{max_sequence_length}"
    )

    print(
        f"Checkpoint         : "
        f"{checkpoint_path}"
    )

    print(
        "\nMemuat test set..."
    )

    test_data = load_test_dataset(
        test_path
    )

    validate_test_dataset(
        test_data=test_data,
        expected_sequence_length=(
            max_sequence_length
        ),
        scenario_code=scenario_code,
    )

    X_test = test_data["X"]
    y_test = test_data["y"].astype(
        np.int32
    )

    print(
        f"X_test             : "
        f"{X_test.shape}"
    )

    print(
        f"y_test             : "
        f"{y_test.shape}"
    )

    print(
        "\nMemuat model terbaik..."
    )

    tf.keras.backend.clear_session()

    gc.collect()

    model = load_best_model(
        model_name=model_name,
        scenario_code=scenario_code,
    )

    if (
        model.input_shape[1]
        != max_sequence_length
    ):
        raise ValueError(
            f"{experiment_name}: input model "
            "tidak sesuai dengan sequence length.\n"
            f"Model input: {model.input_shape}\n"
            f"Expected   : "
            f"{max_sequence_length}"
        )

    print(
        "Melakukan prediksi test set..."
    )

    prediction_start = (
        time.perf_counter()
    )

    probabilities = model.predict(
        X_test,
        batch_size=(
            EVALUATION_BATCH_SIZE
        ),
        verbose=PREDICTION_VERBOSE,
    )

    prediction_end = (
        time.perf_counter()
    )

    inference_seconds = (
        prediction_end
        - prediction_start
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    validate_prediction_probabilities(
        probabilities=probabilities,
        expected_rows=len(
            X_test
        ),
        experiment_name=experiment_name,
    )

    y_pred = np.argmax(
        probabilities,
        axis=1,
    ).astype(
        np.int32
    )

    overall_metrics = (
        calculate_overall_metrics(
            y_true=y_test,
            y_pred=y_pred,
            probabilities=probabilities,
        )
    )

    correct_predictions = int(
        np.sum(
            y_test == y_pred
        )
    )

    incorrect_predictions = int(
        np.sum(
            y_test != y_pred
        )
    )

    average_inference_ms = (
        inference_seconds
        / len(X_test)
        * 1000
    )

    prediction_path = (
        save_predictions(
            document_ids=test_data[
                "document_id"
            ],
            original_categories=test_data[
                "category"
            ],
            y_true=y_test,
            y_pred=y_pred,
            probabilities=probabilities,
            index_to_label=index_to_label,
            model_name=model_name,
            scenario_code=scenario_code,
            dataset_name=dataset_name,
            scenario_name=scenario_name,
        )
    )

    overall_row = {
        "experiment_name":
            experiment_name,

        "model":
            model_name,

        "dataset":
            dataset_name,

        "scenario_code":
            scenario_code,

        "scenario_name":
            scenario_name,

        "jumlah_test":
            int(
                len(X_test)
            ),

        "correct_predictions":
            correct_predictions,

        "incorrect_predictions":
            incorrect_predictions,

        **overall_metrics,

        "inference_time_seconds":
            round(
                inference_seconds,
                6,
            ),

        "average_inference_ms_per_sample":
            round(
                average_inference_ms,
                6,
            ),

        "checkpoint_path":
            str(
                checkpoint_path
            ),

        "test_path":
            str(
                test_path
            ),

        "prediction_path":
            str(
                prediction_path
            ),
    }

    per_class_rows = (
        calculate_per_class_metrics(
            y_true=y_test,
            y_pred=y_pred,
            index_to_label=index_to_label,
            model_name=model_name,
            scenario_code=scenario_code,
            dataset_name=dataset_name,
            scenario_name=scenario_name,
        )
    )

    confusion_rows = (
        create_confusion_matrix_rows(
            y_true=y_test,
            y_pred=y_pred,
            index_to_label=index_to_label,
            model_name=model_name,
            scenario_code=scenario_code,
            dataset_name=dataset_name,
        )
    )

    status_row = {
        "experiment_name":
            experiment_name,

        "model":
            model_name,

        "scenario_code":
            scenario_code,

        "dataset":
            dataset_name,

        "status":
            "success",

        "accuracy":
            overall_metrics[
                "accuracy"
            ],

        "f1_macro":
            overall_metrics[
                "f1_macro"
            ],

        "evaluation_time_seconds":
            round(
                inference_seconds,
                6,
            ),

        "error_message":
            "",
    }

    print("\nHasil test:")

    print(
        f"Accuracy           : "
        f"{overall_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision Macro    : "
        f"{overall_metrics['precision_macro']:.4f}"
    )

    print(
        f"Recall Macro       : "
        f"{overall_metrics['recall_macro']:.4f}"
    )

    print(
        f"F1-score Macro     : "
        f"{overall_metrics['f1_macro']:.4f}"
    )

    print(
        f"Log Loss           : "
        f"{overall_metrics['log_loss']:.6f}"
    )

    print(
        f"Prediksi benar     : "
        f"{correct_predictions:,}"
    )

    print(
        f"Prediksi salah     : "
        f"{incorrect_predictions:,}"
    )

    print(
        f"Waktu inferensi    : "
        f"{inference_seconds:.2f} detik"
    )

    print(
        f"Rata-rata/artikel  : "
        f"{average_inference_ms:.4f} ms"
    )

    del model
    del probabilities
    del y_pred

    tf.keras.backend.clear_session()

    gc.collect()

    return (
        overall_row,
        per_class_rows,
        confusion_rows,
        status_row,
    )


# =============================================================================
# MENYIMPAN LAPORAN
# =============================================================================

def save_evaluation_outputs(
    overall_rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
    confusion_rows: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Menyimpan seluruh hasil evaluasi.
    """

    overall_dataframe = (
        pd.DataFrame(
            overall_rows
        )
    )

    per_class_dataframe = (
        pd.DataFrame(
            per_class_rows
        )
    )

    confusion_dataframe = (
        pd.DataFrame(
            confusion_rows
        )
    )

    status_dataframe = (
        pd.DataFrame(
            status_rows
        )
    )

    if not overall_dataframe.empty:
        overall_dataframe = (
            overall_dataframe
            .sort_values(
                [
                    "dataset",
                    "scenario_code",
                    "model",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    if not per_class_dataframe.empty:
        per_class_dataframe = (
            per_class_dataframe
            .sort_values(
                [
                    "dataset",
                    "scenario_code",
                    "model",
                    "class_index",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    if not confusion_dataframe.empty:
        confusion_dataframe = (
            confusion_dataframe
            .sort_values(
                [
                    "dataset",
                    "scenario_code",
                    "model",
                    "actual_index",
                    "predicted_index",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    overall_dataframe.to_csv(
        OVERALL_METRICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    per_class_dataframe.to_csv(
        PER_CLASS_METRICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    confusion_dataframe.to_csv(
        CONFUSION_MATRIX_DATA_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    status_dataframe.to_csv(
        EVALUATION_STATUS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    return (
        overall_dataframe,
        status_dataframe,
    )


# =============================================================================
# MENYIMPAN SUMMARY JSON
# =============================================================================

def save_evaluation_summary(
    overall_dataframe: pd.DataFrame,
    status_dataframe: pd.DataFrame,
    total_duration_seconds: float,
) -> None:
    """
    Menyimpan ringkasan evaluasi dalam format JSON.
    """

    success_count = int(
        (
            status_dataframe[
                "status"
            ]
            == "success"
        ).sum()
    )

    failed_count = int(
        (
            status_dataframe[
                "status"
            ]
            == "failed"
        ).sum()
    )

    best_experiment = None

    if not overall_dataframe.empty:
        best_row = (
            overall_dataframe
            .sort_values(
                [
                    "f1_macro",
                    "accuracy",
                ],
                ascending=False,
            )
            .iloc[0]
        )

        best_experiment = {
            "experiment_name":
                best_row[
                    "experiment_name"
                ],

            "model":
                best_row[
                    "model"
                ],

            "dataset":
                best_row[
                    "dataset"
                ],

            "scenario_code":
                best_row[
                    "scenario_code"
                ],

            "accuracy":
                float(
                    best_row[
                        "accuracy"
                    ]
                ),

            "f1_macro":
                float(
                    best_row[
                        "f1_macro"
                    ]
                ),
        }

    summary = {
        "generated_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "evaluation_split":
            "test",

        "total_experiments":
            int(
                len(
                    status_dataframe
                )
            ),

        "success":
            success_count,

        "failed":
            failed_count,

        "total_duration_seconds":
            round(
                total_duration_seconds,
                6,
            ),

        "total_duration_minutes":
            round(
                total_duration_seconds
                / 60,
                6,
            ),

        "best_experiment_by_macro_f1":
            best_experiment,

        "important_note": (
            "Model terbaik dimuat dari checkpoint "
            "dengan validation loss terendah. "
            "Test set hanya digunakan untuk "
            "evaluasi performa final."
        ),
    }

    with open(
        EVALUATION_SUMMARY_PATH,
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
# PROGRAM UTAMA
# =============================================================================

def main() -> None:
    """
    Mengevaluasi seluruh 12 eksperimen.
    """

    print("=" * 80)
    print(
        "STEP 6.1 - EVALUATE ALL MODELS "
        "ON TEST SET"
    )
    print("=" * 80)

    set_global_seed()

    create_output_directories()

    all_label_mappings = (
        load_all_label_mappings()
    )

    experiment_queue = [
        (
            model_name,
            scenario_code,
        )
        for model_name in MODEL_NAMES
        for scenario_code in SCENARIO_CODES
    ]

    print(
        f"\nJumlah eksperimen: "
        f"{len(experiment_queue)}"
    )

    print(
        f"Batch size evaluasi: "
        f"{EVALUATION_BATCH_SIZE}"
    )

    print("\nDaftar eksperimen:")

    for number, (
        model_name,
        scenario_code,
    ) in enumerate(
        experiment_queue,
        start=1,
    ):
        print(
            f"{number:02d}. "
            f"{get_experiment_name(model_name, scenario_code)}"
        )

    overall_rows: list[
        dict[str, Any]
    ] = []

    per_class_rows: list[
        dict[str, Any]
    ] = []

    confusion_rows: list[
        dict[str, Any]
    ] = []

    status_rows: list[
        dict[str, Any]
    ] = []

    overall_start = (
        time.perf_counter()
    )

    for experiment_number, (
        model_name,
        scenario_code,
    ) in enumerate(
        experiment_queue,
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
            f"EXPERIMENT "
            f"{experiment_number}/"
            f"{len(experiment_queue)}"
        )

        print(
            f"Nama: {experiment_name}"
        )

        print(
            "#" * 80
        )

        try:
            (
                overall_row,
                experiment_per_class,
                experiment_confusion,
                status_row,
            ) = evaluate_single_model(
                model_name=model_name,
                scenario_code=scenario_code,
                all_label_mappings=(
                    all_label_mappings
                ),
            )

            overall_rows.append(
                overall_row
            )

            per_class_rows.extend(
                experiment_per_class
            )

            confusion_rows.extend(
                experiment_confusion
            )

            status_rows.append(
                status_row
            )

        except Exception as error:
            print(
                "\nEvaluasi gagal:"
            )

            print(
                str(error)
            )

            status_rows.append(
                {
                    "experiment_name":
                        experiment_name,

                    "model":
                        model_name,

                    "scenario_code":
                        scenario_code,

                    "dataset":
                        "",

                    "status":
                        "failed",

                    "accuracy":
                        None,

                    "f1_macro":
                        None,

                    "evaluation_time_seconds":
                        None,

                    "error_message":
                        str(error),
                }
            )

        (
            interim_overall,
            interim_status,
        ) = save_evaluation_outputs(
            overall_rows=overall_rows,
            per_class_rows=per_class_rows,
            confusion_rows=confusion_rows,
            status_rows=status_rows,
        )

    overall_duration_seconds = (
        time.perf_counter()
        - overall_start
    )

    (
        overall_dataframe,
        status_dataframe,
    ) = save_evaluation_outputs(
        overall_rows=overall_rows,
        per_class_rows=per_class_rows,
        confusion_rows=confusion_rows,
        status_rows=status_rows,
    )

    save_evaluation_summary(
        overall_dataframe=(
            overall_dataframe
        ),
        status_dataframe=(
            status_dataframe
        ),
        total_duration_seconds=(
            overall_duration_seconds
        ),
    )

    print("\n" + "=" * 80)
    print("RINGKASAN EVALUASI TEST SET")
    print("=" * 80)

    if overall_dataframe.empty:
        print(
            "\nBelum ada model yang "
            "berhasil dievaluasi."
        )

    else:
        display_columns = [
            "experiment_name",
            "dataset",
            "scenario_code",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "log_loss",
            "inference_time_seconds",
        ]

        display_dataframe = (
            overall_dataframe[
                display_columns
            ]
            .copy()
        )

        numeric_columns = [
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "log_loss",
            "inference_time_seconds",
        ]

        for column in numeric_columns:
            display_dataframe[
                column
            ] = (
                display_dataframe[
                    column
                ]
                .map(
                    lambda value:
                    f"{value:.6f}"
                )
            )

        print(
            "\n"
            + display_dataframe.to_string(
                index=False,
            )
        )

    success_count = int(
        (
            status_dataframe[
                "status"
            ]
            == "success"
        ).sum()
    )

    failed_count = int(
        (
            status_dataframe[
                "status"
            ]
            == "failed"
        ).sum()
    )

    print(
        f"\nEksperimen berhasil : "
        f"{success_count}"
    )

    print(
        f"Eksperimen gagal    : "
        f"{failed_count}"
    )

    print(
        f"Total waktu         : "
        f"{overall_duration_seconds / 60:.2f} menit"
    )

    print("\nOutput metrik keseluruhan:")
    print(
        OVERALL_METRICS_PATH
    )

    print("\nOutput metrik per kelas:")
    print(
        PER_CLASS_METRICS_PATH
    )

    print("\nData confusion matrix:")
    print(
        CONFUSION_MATRIX_DATA_PATH
    )

    print("\nFolder hasil prediksi:")
    print(
        PREDICTIONS_DIR
    )

    print("\nStatus evaluasi:")
    print(
        EVALUATION_STATUS_PATH
    )

    print("\nRingkasan evaluasi:")
    print(
        EVALUATION_SUMMARY_PATH
    )

    print("\n" + "=" * 80)
    print(
        "Evaluasi seluruh model selesai."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
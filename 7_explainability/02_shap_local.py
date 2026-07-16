# =============================================================================
# STEP 7.2 - LOCAL SHAP EXPLANATION
# =============================================================================
# File:
# 7_explainability/02_shap_local.py
#
# Tujuan:
# Menjelaskan prediksi model CNN K2 pada tingkat artikel menggunakan
# SHAP values yang sudah dihitung pada tahap global SHAP.
#
# Input utama:
# - 9_results/shap_values/cnn_k2_global_shap_values.npz
# - 2_data/vectorized/kompas_k2/test.npz
# - 2_data/vectorized/kompas_k2/vocabulary.txt
# - 9_results/tables/label_mapping.json
#
# Output:
# - Tabel ringkasan sampel local SHAP
# - Tabel kontribusi token setiap sampel
# - Grafik kontribusi token setiap sampel
#
# Catatan metodologis:
# Nilai SHAP dihitung pada embedding token. Untuk local explanation,
# nilai SHAP pada 128 dimensi embedding dijumlahkan secara bertanda
# (signed sum) agar dapat menunjukkan token yang mendukung atau
# menahan kelas prediksi.
# =============================================================================

from __future__ import annotations

import json
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

# Jumlah sampel local explanation yang dibuat otomatis.
# Script akan mencoba memilih:
# - 1 prediksi benar per kelas
# - 1 prediksi salah per kelas jika tersedia
MAX_LOCAL_SAMPLES = 8

# Jumlah token teratas yang ditampilkan pada setiap grafik.
TOP_N_TOKENS = 15

SPECIAL_TOKENS = {
    "[PAD]",
    "[SEP]",
    "[UNK]",
    "[OOV]",
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

LOCAL_CONFIGURATION_PATH = (
    LOCAL_TABLES_DIR
    / "cnn_k2_local_shap_configuration.json"
)


# =============================================================================
# MEMBUAT FOLDER OUTPUT
# =============================================================================

def create_output_directories() -> None:
    """
    Membuat folder output local SHAP.
    """

    LOCAL_TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOCAL_FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# MEMUAT DATA TEST
# =============================================================================

def load_test_data() -> dict[str, np.ndarray]:
    """
    Memuat test set K2.
    """

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            "Test set tidak ditemukan:\n"
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

    if result["X"].shape[1] != MAX_SEQUENCE_LENGTH:
        raise ValueError(
            "Panjang sequence test tidak sesuai.\n"
            f"Expected: {MAX_SEQUENCE_LENGTH}\n"
            f"Actual  : {result['X'].shape[1]}"
        )

    return result


# =============================================================================
# MEMUAT VOCABULARY
# =============================================================================

def load_vocabulary() -> list[str]:
    """
    Membaca vocabulary sesuai indeks token.
    """

    if not VOCABULARY_PATH.exists():
        raise FileNotFoundError(
            "Vocabulary tidak ditemukan:\n"
            f"{VOCABULARY_PATH}"
        )

    with open(
        VOCABULARY_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        vocabulary = file.read().splitlines()

    if not vocabulary:
        raise ValueError(
            "Vocabulary kosong."
        )

    return vocabulary


def token_id_to_word(
    token_id: int,
    vocabulary: list[str],
) -> str:
    """
    Mengubah token id menjadi token teks.
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
            "Mapping Kompas tidak ditemukan."
        )

    if "index_to_label" in kompas_mapping:
        return {
            int(index): str(label)
            for index, label
            in kompas_mapping[
                "index_to_label"
            ].items()
        }

    if "label_to_index" in kompas_mapping:
        return {
            int(index): str(label)
            for label, index
            in kompas_mapping[
                "label_to_index"
            ].items()
        }

    raise KeyError(
        "Format mapping label tidak dikenali."
    )


# =============================================================================
# MEMUAT SHAP VALUES HASIL GLOBAL
# =============================================================================

def load_global_shap_values() -> dict[str, np.ndarray]:
    """
    Memuat SHAP values yang disimpan oleh 01_shap_global.py.
    """

    if not GLOBAL_SHAP_VALUES_PATH.exists():
        raise FileNotFoundError(
            "File global SHAP values tidak ditemukan:\n"
            f"{GLOBAL_SHAP_VALUES_PATH}\n\n"
            "Jalankan 01_shap_global.py terlebih dahulu."
        )

    with np.load(
        GLOBAL_SHAP_VALUES_PATH,
        allow_pickle=False,
    ) as data:
        required_keys = {
            "shap_values",
            "token_sequences",
            "explain_indices",
            "probabilities",
            "predicted_classes",
        }

        missing_keys = (
            required_keys
            - set(data.files)
        )

        if missing_keys:
            raise KeyError(
                "Komponen file SHAP tidak lengkap.\n"
                f"Key hilang: {missing_keys}"
            )

        result = {
            "shap_values": np.asarray(
                data["shap_values"],
                dtype=np.float32,
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
                dtype=np.float32,
            ),
            "predicted_classes": np.asarray(
                data["predicted_classes"],
                dtype=np.int32,
            ),
        }

    expected_shape_prefix = (
        NUM_CLASSES,
        len(result["explain_indices"]),
        MAX_SEQUENCE_LENGTH,
    )

    if (
        result["shap_values"].shape[:3]
        != expected_shape_prefix
    ):
        raise ValueError(
            "Shape SHAP values tidak sesuai.\n"
            f"Expected prefix: {expected_shape_prefix}\n"
            f"Actual shape   : {result['shap_values'].shape}"
        )

    return result


# =============================================================================
# PEMILIHAN SAMPEL LOCAL
# =============================================================================

def select_local_samples(
    test_data: dict[str, np.ndarray],
    shap_data: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """
    Memilih sampel local explanation.

    Prioritas:
    1. Satu prediksi benar dengan confidence tertinggi per kelas.
    2. Satu prediksi salah dengan confidence tertinggi per kelas,
       jika tersedia.
    """

    explain_indices = shap_data[
        "explain_indices"
    ]

    probabilities = shap_data[
        "probabilities"
    ]

    predicted_classes = shap_data[
        "predicted_classes"
    ]

    actual_classes = test_data[
        "y"
    ][explain_indices]

    confidences = probabilities.max(
        axis=1
    )

    selected: list[dict[str, Any]] = []

    # Satu prediksi benar per kelas.
    for class_index in range(
        NUM_CLASSES
    ):
        candidates = np.where(
            (
                actual_classes == class_index
            )
            & (
                predicted_classes == class_index
            )
        )[0]

        if len(candidates) == 0:
            continue

        candidate_confidences = confidences[
            candidates
        ]

        selected_position = int(
            candidates[
                np.argmax(
                    candidate_confidences
                )
            ]
        )

        selected.append(
            {
                "shap_sample_position":
                    selected_position,

                "selection_type":
                    "correct_high_confidence",
            }
        )

    # Satu prediksi salah per kelas aktual, jika tersedia.
    for class_index in range(
        NUM_CLASSES
    ):
        candidates = np.where(
            (
                actual_classes == class_index
            )
            & (
                predicted_classes != class_index
            )
        )[0]

        if len(candidates) == 0:
            continue

        candidate_confidences = confidences[
            candidates
        ]

        selected_position = int(
            candidates[
                np.argmax(
                    candidate_confidences
                )
            ]
        )

        selected.append(
            {
                "shap_sample_position":
                    selected_position,

                "selection_type":
                    "incorrect_high_confidence",
            }
        )

    # Hindari posisi yang terduplikasi.
    unique_selected = []
    used_positions = set()

    for item in selected:
        position = item[
            "shap_sample_position"
        ]

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
# MENGHITUNG KONTRIBUSI TOKEN LOCAL
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
]:
    """
    Menghitung kontribusi token terhadap kelas prediksi.

    Signed contribution:
    jumlah nilai SHAP pada seluruh dimensi embedding.

    Positive:
    token mendukung kelas prediksi.

    Negative:
    token menahan atau mengurangi skor kelas prediksi.
    """

    original_test_index = int(
        shap_data[
            "explain_indices"
        ][shap_sample_position]
    )

    token_sequence = shap_data[
        "token_sequences"
    ][shap_sample_position]

    probabilities = shap_data[
        "probabilities"
    ][shap_sample_position]

    predicted_class = int(
        shap_data[
            "predicted_classes"
        ][shap_sample_position]
    )

    actual_class = int(
        test_data[
            "y"
        ][original_test_index]
    )

    # SHAP untuk kelas yang diprediksi.
    sample_shap = shap_data[
        "shap_values"
    ][
        predicted_class,
        shap_sample_position,
    ]

    signed_contributions = np.sum(
        sample_shap,
        axis=-1,
    )

    absolute_contributions = np.sum(
        np.abs(
            sample_shap
        ),
        axis=-1,
    )

    rows = []

    for position, token_id in enumerate(
        token_sequence
    ):
        token_id = int(
            token_id
        )

        if token_id == 0:
            continue

        token = token_id_to_word(
            token_id,
            vocabulary,
        )

        if token in SPECIAL_TOKENS:
            continue

        signed_value = float(
            signed_contributions[
                position
            ]
        )

        absolute_value = float(
            absolute_contributions[
                position
            ]
        )

        if signed_value > 0:
            direction = "mendukung_prediksi"
        elif signed_value < 0:
            direction = "menahan_prediksi"
        else:
            direction = "netral"

        rows.append(
            {
                "shap_sample_position":
                    shap_sample_position,

                "original_test_index":
                    original_test_index,

                "document_id":
                    str(
                        test_data[
                            "document_id"
                        ][original_test_index]
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
                    position,

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

    token_dataframe = pd.DataFrame(
        rows
    )

    if token_dataframe.empty:
        raise ValueError(
            "Tidak ada token semantik untuk sampel "
            f"posisi {shap_sample_position}."
        )

    # Jika token yang sama muncul lebih dari sekali,
    # gabungkan kontribusinya agar grafik lebih mudah dibaca.
    aggregated_dataframe = (
        token_dataframe
        .groupby(
            [
                "shap_sample_position",
                "original_test_index",
                "document_id",
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
            occurrence_count=(
                "token_position",
                "size",
            ),
        )
    )

    aggregated_dataframe[
        "direction"
    ] = np.where(
        aggregated_dataframe[
            "signed_shap"
        ] > 0,
        "mendukung_prediksi",
        np.where(
            aggregated_dataframe[
                "signed_shap"
            ] < 0,
            "menahan_prediksi",
            "netral",
        ),
    )

    aggregated_dataframe = (
        aggregated_dataframe
        .sort_values(
            "absolute_shap",
            ascending=False,
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

    confidence = float(
        probabilities[
            predicted_class
        ]
    )

    summary = {
        "shap_sample_position":
            shap_sample_position,

        "original_test_index":
            original_test_index,

        "document_id":
            str(
                test_data[
                    "document_id"
                ][original_test_index]
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
            confidence,

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

        "number_of_semantic_tokens":
            int(
                len(
                    aggregated_dataframe
                )
            ),
    }

    return (
        summary,
        aggregated_dataframe,
    )


# =============================================================================
# MEMBUAT GRAFIK LOCAL SHAP
# =============================================================================

def plot_local_explanation(
    summary: dict[str, Any],
    token_contributions: pd.DataFrame,
) -> Path:
    """
    Membuat grafik token yang mendukung dan menahan prediksi.

    Nilai positif:
    mendukung kelas prediksi.

    Nilai negatif:
    menahan kelas prediksi.
    """

    plot_data = (
        token_contributions
        .head(
            TOP_N_TOKENS
        )
        .sort_values(
            "signed_shap",
            ascending=True,
        )
    )

    output_name = (
        f"{summary['document_id']}_"
        f"{summary['selection_type']}_"
        f"actual_{summary['actual_label']}_"
        f"pred_{summary['predicted_label']}.png"
    )

    # Bersihkan karakter yang berpotensi tidak aman untuk nama file.
    output_name = (
        output_name
        .replace(
            "/",
            "_",
        )
        .replace(
            "\\",
            "_",
        )
        .replace(
            ":",
            "_",
        )
        .replace(
            " ",
            "_",
        )
    )

    output_path = (
        LOCAL_FIGURES_DIR
        / output_name
    )

    plt.figure(
        figsize=(11, 8)
    )

    plt.barh(
        plot_data["token"],
        plot_data["signed_shap"],
    )

    plt.axvline(
        0,
        linewidth=1,
    )

    plt.xlabel(
        "Signed SHAP Contribution"
    )

    plt.ylabel(
        "Token"
    )

    status_text = (
        "BENAR"
        if summary["is_correct"]
        else "SALAH"
    )

    plt.title(
        "Local SHAP Explanation — CNN K2\n"
        f"Actual: {summary['actual_label']} | "
        f"Prediksi: {summary['predicted_label']} | "
        f"Confidence: {summary['prediction_confidence']:.2%} | "
        f"{status_text}"
    )

    plt.grid(
        axis="x",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


# =============================================================================
# MENYIMPAN KONFIGURASI
# =============================================================================

def save_configuration(
    selected_samples: list[dict[str, Any]],
) -> None:
    """
    Menyimpan konfigurasi local SHAP.
    """

    configuration = {
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

        "selection_strategy": (
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

        "top_n_tokens_per_plot":
            TOP_N_TOKENS,

        "aggregation_method": (
            "Signed sum across embedding dimensions for direction, "
            "and sum of absolute SHAP values across embedding "
            "dimensions for importance."
        ),

        "special_tokens_excluded":
            sorted(
                SPECIAL_TOKENS
            ),

        "interpretation_note": (
            "Positive signed SHAP values support the predicted class. "
            "Negative values reduce the predicted class score. "
            "Because SHAP is calculated at the embedding level, "
            "token contributions are approximate aggregated "
            "attributions."
        ),
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

    print("\nMemuat test set...")
    test_data = load_test_data()

    print(
        f"Test shape               : "
        f"{test_data['X'].shape}"
    )

    print("\nMemuat vocabulary...")
    vocabulary = load_vocabulary()

    print(
        f"Vocabulary size          : "
        f"{len(vocabulary):,}"
    )

    print("\nMemuat label mapping...")
    index_to_label = load_index_to_label()

    print(
        f"Label mapping            : "
        f"{index_to_label}"
    )

    print("\nMemuat global SHAP values...")
    shap_data = load_global_shap_values()

    print(
        f"SHAP values shape        : "
        f"{shap_data['shap_values'].shape}"
    )

    print(
        f"Jumlah sampel tersedia   : "
        f"{len(shap_data['explain_indices'])}"
    )

    print("\nMemilih sampel local explanation...")
    selected_samples = select_local_samples(
        test_data=test_data,
        shap_data=shap_data,
    )

    if not selected_samples:
        raise ValueError(
            "Tidak ada sampel local explanation yang dapat dipilih."
        )

    print(
        f"Jumlah sampel dipilih    : "
        f"{len(selected_samples)}"
    )

    summary_rows = []
    contribution_dataframes = []

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
            f"Sampel {number}/{len(selected_samples)}"
        )

        (
            summary,
            token_contributions,
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

        figure_path = plot_local_explanation(
            summary=summary,
            token_contributions=(
                token_contributions
            ),
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

        print(
            f"Document ID              : "
            f"{summary['document_id']}"
        )

        print(
            f"Jenis sampel             : "
            f"{selection_type}"
        )

        print(
            f"Label aktual             : "
            f"{summary['actual_label']}"
        )

        print(
            f"Label prediksi           : "
            f"{summary['predicted_label']}"
        )

        print(
            f"Prediksi benar           : "
            f"{summary['is_correct']}"
        )

        print(
            f"Confidence               : "
            f"{summary['prediction_confidence']:.2%}"
        )

        print(
            "Token paling berpengaruh :"
        )

        display_columns = [
            "importance_rank",
            "token",
            "signed_shap",
            "absolute_shap",
            "direction",
        ]

        print(
            "\n"
            + token_contributions[
                display_columns
            ]
            .head(8)
            .to_string(
                index=False
            )
        )

        print(
            f"Grafik                   : "
            f"{figure_path}"
        )

    summary_dataframe = pd.DataFrame(
        summary_rows
    )

    all_contributions = pd.concat(
        contribution_dataframes,
        ignore_index=True,
    )

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

    save_configuration(
        selected_samples
    )

    print("\n" + "=" * 80)
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
        "\nTabel ringkasan local SHAP:"
    )

    print(
        LOCAL_SUMMARY_PATH
    )

    print(
        "\nTabel kontribusi token:"
    )

    print(
        LOCAL_TOKEN_CONTRIBUTIONS_PATH
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

    print("\n" + "=" * 80)
    print(
        "Tahap local SHAP selesai."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
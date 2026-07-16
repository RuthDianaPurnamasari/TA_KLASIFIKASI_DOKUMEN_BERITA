# =============================================================================
# STEP 8.2 - INFERENCE PIPELINE
# =============================================================================
# Tujuan:
# Menguji pipeline prediksi deployment untuk:
# 1. CNN K2
# 2. Attention-BiLSTM K2
#
# Input:
# - Title
# - Description
#
# Alur:
# Input mentah
# -> preprocessing
# -> gabungkan Title + [SEP] + Description
# -> TextVectorization dengan vocabulary training K2
# -> sequence length 60
# -> prediksi dua model
# -> probabilitas 4 kelas
# =============================================================================

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf


# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELING_DIR = (
    PROJECT_ROOT
    / "5_modeling"
)

if str(MODELING_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(MODELING_DIR),
    )

from attention_bilstm_model import AttentionPooling1D


# =============================================================================
# PATH
# =============================================================================

DEPLOYMENT_DIR = (
    PROJECT_ROOT
    / "8_save_models"
    / "deployment"
)

CNN_MODEL_PATH = (
    DEPLOYMENT_DIR
    / "cnn_k2.keras"
)

ATTENTION_BILSTM_MODEL_PATH = (
    DEPLOYMENT_DIR
    / "attention_bilstm_k2.keras"
)

VOCABULARY_PATH = (
    DEPLOYMENT_DIR
    / "vocabulary.txt"
)

VECTORIZER_CONFIG_PATH = (
    DEPLOYMENT_DIR
    / "vectorizer_config.json"
)

LABEL_MAPPING_PATH = (
    DEPLOYMENT_DIR
    / "label_mapping.json"
)

DEPLOYMENT_CONFIG_PATH = (
    DEPLOYMENT_DIR
    / "deployment_config.json"
)


# =============================================================================
# KONFIGURASI
# =============================================================================

MAX_SEQUENCE_LENGTH = 60

DEFAULT_LABEL_MAPPING = {
    0: "bola",
    1: "global",
    2: "money",
    3: "tekno",
}


# =============================================================================
# TEXT CLEANING
# =============================================================================

def clean_text(
    text: str,
) -> str:
    """
    Membersihkan teks untuk inference.

    Tahap:
    - ubah ke string
    - case folding
    - normalisasi whitespace
    - hapus URL
    - hapus karakter nonalfanumerik
    - pertahankan angka
    """

    if text is None:
        return ""

    text = str(
        text
    )

    text = text.lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
    )

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def build_input_text(
    title: str,
    description: str,
) -> str:
    """
    Membentuk representasi K2:
    Title + [SEP] + Description.
    """

    clean_title = clean_text(
        title
    )

    clean_description = clean_text(
        description
    )

    if not clean_title:
        raise ValueError(
            "Title tidak boleh kosong."
        )

    if not clean_description:
        raise ValueError(
            "Description tidak boleh kosong."
        )

    combined_text = (
        f"{clean_title} "
        f"[SEP] "
        f"{clean_description}"
    )

    return combined_text


# =============================================================================
# LOAD CONFIGURATION
# =============================================================================

def load_json(
    file_path: Path,
) -> dict:
    """
    Membaca JSON.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan:\n{file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def load_vocabulary() -> list[str]:
    """
    Membaca vocabulary K2.
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


def load_label_mapping() -> dict[int, str]:
    """
    Membaca label mapping Kompas.
    """

    if not LABEL_MAPPING_PATH.exists():
        return DEFAULT_LABEL_MAPPING

    with open(
        LABEL_MAPPING_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(
            file
        )

    kompas_mapping = data.get(
        "Kompas",
        data.get(
            "kompas",
            {},
        ),
    )

    if (
        isinstance(
            kompas_mapping,
            dict,
        )
        and "index_to_label"
        in kompas_mapping
    ):
        return {
            int(index): str(label)
            for index, label
            in kompas_mapping[
                "index_to_label"
            ].items()
        }

    if (
        isinstance(
            kompas_mapping,
            dict,
        )
        and "label_to_index"
        in kompas_mapping
    ):
        return {
            int(index): str(label)
            for label, index
            in kompas_mapping[
                "label_to_index"
            ].items()
        }

    if all(
        str(key).isdigit()
        for key in data.keys()
    ):
        return {
            int(index): str(label)
            for index, label
            in data.items()
        }

    return DEFAULT_LABEL_MAPPING


# =============================================================================
# VECTORIZE
# =============================================================================

def build_text_vectorizer(
    vocabulary: list[str],
) -> tf.keras.layers.TextVectorization:
    """
    Membuat TextVectorization menggunakan vocabulary training.

    standardize=None karena teks sudah dibersihkan manual.
    """

    vectorizer = tf.keras.layers.TextVectorization(
        standardize=None,
        split="whitespace",
        output_mode="int",
        output_sequence_length=(
            MAX_SEQUENCE_LENGTH
        ),
        vocabulary=vocabulary,
        name="deployment_text_vectorizer",
    )

    return vectorizer


def vectorize_text(
    text: str,
    vectorizer: tf.keras.layers.TextVectorization,
) -> np.ndarray:
    """
    Mengubah teks menjadi integer sequence.
    """

    sequence = vectorizer(
        tf.constant(
            [text]
        )
    )

    sequence = np.asarray(
        sequence,
        dtype=np.int32,
    )

    if sequence.shape != (
        1,
        MAX_SEQUENCE_LENGTH,
    ):
        raise ValueError(
            "Shape sequence tidak sesuai.\n"
            f"Expected: {(1, MAX_SEQUENCE_LENGTH)}\n"
            f"Actual  : {sequence.shape}"
        )

    return sequence


# =============================================================================
# LOAD MODELS
# =============================================================================

def load_models() -> dict[str, tf.keras.Model]:
    """
    Memuat CNN dan Attention-BiLSTM.
    """

    if not CNN_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model CNN tidak ditemukan:\n"
            f"{CNN_MODEL_PATH}"
        )

    if not ATTENTION_BILSTM_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model Attention-BiLSTM tidak ditemukan:\n"
            f"{ATTENTION_BILSTM_MODEL_PATH}"
        )

    cnn_model = tf.keras.models.load_model(
        CNN_MODEL_PATH,
        compile=False,
    )

    attention_bilstm_model = (
        tf.keras.models.load_model(
            ATTENTION_BILSTM_MODEL_PATH,
            compile=False,
            custom_objects={
                "AttentionPooling1D":
                    AttentionPooling1D,

                "TAKlasifikasiBerita>AttentionPooling1D":
                    AttentionPooling1D,
            },
        )
    )

    return {
        "cnn":
            cnn_model,

        "attention_bilstm":
            attention_bilstm_model,
    }


# =============================================================================
# PREDICTION
# =============================================================================

def predict_single_model(
    model: tf.keras.Model,
    sequence: np.ndarray,
    label_mapping: dict[int, str],
) -> dict:
    """
    Prediksi satu model.
    """

    start_time = time.perf_counter()

    probabilities = model.predict(
        sequence,
        verbose=0,
    )[0]

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predicted_index = int(
        np.argmax(
            probabilities
        )
    )

    predicted_label = label_mapping[
        predicted_index
    ]

    class_probabilities = {
        label_mapping[index]:
            float(
                probabilities[index]
            )
        for index
        in range(
            len(probabilities)
        )
    }

    return {
        "predicted_index":
            predicted_index,

        "predicted_label":
            predicted_label,

        "confidence":
            float(
                probabilities[
                    predicted_index
                ]
            ),

        "probabilities":
            class_probabilities,

        "inference_time_seconds":
            elapsed_seconds,

        "inference_time_ms":
            elapsed_seconds
            * 1000,
    }


def predict_news(
    title: str,
    description: str,
    models: dict[str, tf.keras.Model],
    vectorizer: tf.keras.layers.TextVectorization,
    label_mapping: dict[int, str],
) -> dict:
    """
    Menjalankan inference lengkap.
    """

    combined_text = build_input_text(
        title=title,
        description=description,
    )

    sequence = vectorize_text(
        text=combined_text,
        vectorizer=vectorizer,
    )

    cnn_result = predict_single_model(
        model=models["cnn"],
        sequence=sequence,
        label_mapping=label_mapping,
    )

    attention_result = predict_single_model(
        model=models[
            "attention_bilstm"
        ],
        sequence=sequence,
        label_mapping=label_mapping,
    )

    agreement = (
        cnn_result[
            "predicted_label"
        ]
        == attention_result[
            "predicted_label"
        ]
    )

    return {
        "input": {
            "title":
                title,

            "description":
                description,

            "combined_text":
                combined_text,

            "sequence_shape":
                list(
                    sequence.shape
                ),

            "non_padding_tokens":
                int(
                    np.count_nonzero(
                        sequence
                    )
                ),
        },

        "cnn":
            cnn_result,

        "attention_bilstm":
            attention_result,

        "model_agreement":
            agreement,

        "recommended_prediction": {
            "source_model":
                "CNN",

            "predicted_label":
                cnn_result[
                    "predicted_label"
                ],

            "confidence":
                cnn_result[
                    "confidence"
                ],
        },
    }


# =============================================================================
# DISPLAY
# =============================================================================

def print_prediction_result(
    result: dict,
) -> None:
    """
    Menampilkan hasil prediksi di terminal.
    """

    print(
        "\n" + "=" * 80
    )
    print(
        "HASIL INFERENCE"
    )
    print(
        "=" * 80
    )

    print(
        f"\nJumlah token non-padding : "
        f"{result['input']['non_padding_tokens']}"
    )

    for model_key, model_name in [
        (
            "cnn",
            "CNN",
        ),
        (
            "attention_bilstm",
            "Attention-BiLSTM",
        ),
    ]:
        model_result = result[
            model_key
        ]

        print(
            "\n" + "-" * 80
        )
        print(
            f"Model                    : "
            f"{model_name}"
        )
        print(
            f"Prediksi                 : "
            f"{model_result['predicted_label']}"
        )
        print(
            f"Confidence               : "
            f"{model_result['confidence']:.2%}"
        )
        print(
            f"Waktu inferensi          : "
            f"{model_result['inference_time_ms']:.2f} ms"
        )

        print(
            "Probabilitas:"
        )

        sorted_probabilities = sorted(
            model_result[
                "probabilities"
            ].items(),
            key=lambda item:
                item[1],
            reverse=True,
        )

        for label, probability in sorted_probabilities:
            print(
                f"  {label:<8}: "
                f"{probability:.2%}"
            )

    print(
        "\n" + "-" * 80
    )
    print(
        f"Kedua model sepakat      : "
        f"{result['model_agreement']}"
    )
    print(
        f"Prediksi rekomendasi     : "
        f"{result['recommended_prediction']['predicted_label']}"
    )
    print(
        f"Model rekomendasi        : "
        f"{result['recommended_prediction']['source_model']}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Menguji inference pipeline menggunakan beberapa contoh.
    """

    print(
        "=" * 80
    )
    print(
        "STEP 8.2 - INFERENCE PIPELINE"
    )
    print(
        "=" * 80
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
        "Membangun TextVectorization..."
    )

    vectorizer = build_text_vectorizer(
        vocabulary
    )

    print(
        "Memuat label mapping..."
    )

    label_mapping = load_label_mapping()

    print(
        f"Label mapping            : "
        f"{label_mapping}"
    )

    print(
        "Memuat model deployment..."
    )

    models = load_models()

    print(
        f"CNN input                : "
        f"{models['cnn'].input_shape}"
    )

    print(
        f"Attention-BiLSTM input   : "
        f"{models['attention_bilstm'].input_shape}"
    )

    sample_inputs = [
        {
            "title":
                "Persib Menang di Liga Indonesia",

            "description":
                (
                    "Persib meraih kemenangan dalam "
                    "pertandingan liga setelah mencetak "
                    "dua gol pada babak kedua."
                ),
        },
        {
            "title":
                "Rupiah Menguat terhadap Dolar AS",

            "description":
                (
                    "Nilai tukar rupiah menguat setelah "
                    "Bank Indonesia mengumumkan kebijakan "
                    "stabilisasi pasar keuangan."
                ),
        },
        {
            "title":
                "Samsung Meluncurkan Ponsel Galaxy Terbaru",

            "description":
                (
                    "Samsung memperkenalkan perangkat Galaxy "
                    "dengan teknologi kecerdasan buatan dan "
                    "peningkatan kapasitas baterai."
                ),
        },
        {
            "title":
                "Iran dan Amerika Serikat Membahas Gencatan Senjata",

            "description":
                (
                    "Perwakilan Iran dan Amerika Serikat "
                    "membahas perkembangan konflik dan "
                    "rencana gencatan senjata."
                ),
        },
    ]

    for number, sample in enumerate(
        sample_inputs,
        start=1,
    ):
        print(
            "\n\n" + "#" * 80
        )
        print(
            f"CONTOH {number}"
        )
        print(
            "#" * 80
        )

        print(
            f"\nTitle       : "
            f"{sample['title']}"
        )
        print(
            f"Description : "
            f"{sample['description']}"
        )

        result = predict_news(
            title=sample[
                "title"
            ],
            description=sample[
                "description"
            ],
            models=models,
            vectorizer=vectorizer,
            label_mapping=label_mapping,
        )

        print_prediction_result(
            result
        )

    print(
        "\n" + "=" * 80
    )
    print(
        "Inference pipeline berhasil dijalankan."
    )
    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
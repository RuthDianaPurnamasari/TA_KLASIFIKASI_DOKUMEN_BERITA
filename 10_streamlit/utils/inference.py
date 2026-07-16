# =============================================================================
# STREAMLIT INFERENCE UTILITY
# =============================================================================
# File:
# 10_streamlit/utils/inference.py
#
# Fungsi:
# 1. Membersihkan input Title dan Description.
# 2. Menggabungkan input menggunakan representasi K2.
# 3. Mengubah teks menjadi sequence menggunakan vocabulary training.
# 4. Memuat model CNN K2 dan Attention-BiLSTM K2.
# 5. Menghasilkan prediksi, confidence, dan probabilitas setiap kelas.
#
# File ini tidak mengatur tampilan Streamlit.
# File ini hanya menangani proses prediksi di belakang layar.
# =============================================================================

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import tensorflow as tf


# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================

# Lokasi file inference.py:
# project_root/10_streamlit/utils/inference.py
CURRENT_FILE = Path(__file__).resolve()

# Folder dashboard:
# project_root/10_streamlit
STREAMLIT_DIR = CURRENT_FILE.parents[1]

# Root project:
# project_root
PROJECT_ROOT = CURRENT_FILE.parents[2]

# Folder modeling yang berisi custom layer AttentionPooling1D
MODELING_DIR = (
    PROJECT_ROOT
    / "5_modeling"
)


# Menambahkan folder dashboard agar config.py dapat di-import
if str(STREAMLIT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(STREAMLIT_DIR),
    )

# Menambahkan folder modeling agar custom layer dapat di-import
if str(MODELING_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(MODELING_DIR),
    )


# =============================================================================
# IMPORT PROJECT CONFIGURATION
# =============================================================================

from config import (
    ATTENTION_BILSTM_MODEL_PATH,
    CNN_MODEL_PATH,
    DEFAULT_LABEL_MAPPING,
    LABEL_MAPPING_PATH,
    MAX_SEQUENCE_LENGTH,
    VOCABULARY_PATH,
)

from attention_bilstm_model import (
    AttentionPooling1D,
)


# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

def clean_text(
    text: str | None,
) -> str:
    """
    Membersihkan teks untuk proses inference.

    Tahapan:
    1. Mengubah input menjadi string.
    2. Case folding menjadi huruf kecil.
    3. Menghapus URL.
    4. Menghapus tag HTML.
    5. Menghapus simbol selain huruf, angka, dan spasi.
    6. Menormalkan spasi.

    Parameters
    ----------
    text:
        Teks Title atau Description.

    Returns
    -------
    str
        Teks yang sudah dibersihkan.
    """

    if text is None:
        return ""

    cleaned_text = str(
        text
    )

    # Case folding
    cleaned_text = (
        cleaned_text
        .lower()
    )

    # Menghapus URL
    cleaned_text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        cleaned_text,
    )

    # Menghapus tag HTML
    cleaned_text = re.sub(
        r"<[^>]+>",
        " ",
        cleaned_text,
    )

    # Menyisakan huruf a-z, angka, dan whitespace
    cleaned_text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        cleaned_text,
    )

    # Menormalkan whitespace
    cleaned_text = re.sub(
        r"\s+",
        " ",
        cleaned_text,
    ).strip()

    return cleaned_text


def build_input_text(
    title: str,
    description: str,
) -> str:
    """
    Membentuk representasi teks K2:

    Title + [SEP] + Description

    Token [SEP] digunakan untuk menandai batas antara
    Title dan Description.
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
# LOAD VOCABULARY
# =============================================================================

@st.cache_data(
    show_spinner=False,
)
def load_vocabulary() -> list[str]:
    """
    Membaca vocabulary K2 yang digunakan saat training.

    Vocabulary yang sama wajib digunakan agar indeks token
    pada dashboard identik dengan indeks token saat training.
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
        vocabulary = (
            file
            .read()
            .splitlines()
        )

    if not vocabulary:
        raise ValueError(
            "Vocabulary deployment kosong."
        )

    return vocabulary


# =============================================================================
# LOAD LABEL MAPPING
# =============================================================================

@st.cache_data(
    show_spinner=False,
)
def load_label_mapping() -> dict[int, str]:
    """
    Membaca mapping indeks kelas ke nama kategori.

    Contoh:
    0 -> bola
    1 -> global
    2 -> money
    3 -> tekno
    """

    if not LABEL_MAPPING_PATH.exists():
        return DEFAULT_LABEL_MAPPING

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

    # Format:
    # {"index_to_label": {"0": "bola", ...}}
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

    # Format:
    # {"label_to_index": {"bola": 0, ...}}
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

    # Format langsung:
    # {"0": "bola", "1": "global", ...}
    if (
        isinstance(
            mapping_data,
            dict,
        )
        and all(
            str(key).isdigit()
            for key
            in mapping_data.keys()
        )
    ):
        return {
            int(index): str(label)
            for index, label
            in mapping_data.items()
        }

    return DEFAULT_LABEL_MAPPING


# =============================================================================
# TEXT VECTORIZATION
# =============================================================================

@st.cache_resource(
    show_spinner=False,
)
def load_text_vectorizer(
) -> tf.keras.layers.TextVectorization:
    """
    Membuat TextVectorization menggunakan vocabulary K2.

    Vocabulary tidak di-fit ulang karena vocabulary harus sama
    dengan vocabulary yang digunakan saat training.
    """

    vocabulary = load_vocabulary()

    vectorizer = (
        tf.keras.layers.TextVectorization(
            standardize=None,
            split="whitespace",
            output_mode="int",
            output_sequence_length=(
                MAX_SEQUENCE_LENGTH
            ),
            vocabulary=vocabulary,
            name=(
                "streamlit_text_vectorizer"
            ),
        )
    )

    return vectorizer


def vectorize_text(
    text: str,
    vectorizer: (
        tf.keras.layers.TextVectorization
    ),
) -> np.ndarray:
    """
    Mengubah satu teks menjadi sequence integer.

    Output:
    shape = (1, 60)
    """

    sequence_tensor = vectorizer(
        tf.constant(
            [text]
        )
    )

    sequence = np.asarray(
        sequence_tensor,
        dtype=np.int32,
    )

    expected_shape = (
        1,
        MAX_SEQUENCE_LENGTH,
    )

    if sequence.shape != expected_shape:
        raise ValueError(
            "Shape sequence tidak sesuai.\n"
            f"Expected: {expected_shape}\n"
            f"Actual  : {sequence.shape}"
        )

    return sequence


# =============================================================================
# LOAD DEPLOYMENT MODELS
# =============================================================================

@st.cache_resource(
    show_spinner=(
        "Memuat model CNN dan "
        "Attention-BiLSTM..."
    ),
)
def load_models(
) -> dict[str, tf.keras.Model]:
    """
    Memuat dua model deployment:

    1. CNN K2 sebagai model utama.
    2. Attention-BiLSTM K2 sebagai model pembanding.

    Model disimpan dalam cache agar tidak dimuat ulang setiap
    kali pengguna berinteraksi dengan dashboard.
    """

    if not CNN_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model CNN K2 tidak ditemukan:\n"
            f"{CNN_MODEL_PATH}"
        )

    if not ATTENTION_BILSTM_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model Attention-BiLSTM K2 "
            "tidak ditemukan:\n"
            f"{ATTENTION_BILSTM_MODEL_PATH}"
        )

    # Memuat CNN
    cnn_model = (
        tf.keras.models.load_model(
            CNN_MODEL_PATH,
            compile=False,
        )
    )

    # Memuat Attention-BiLSTM dengan custom layer
    attention_bilstm_model = (
        tf.keras.models.load_model(
            ATTENTION_BILSTM_MODEL_PATH,
            compile=False,
            custom_objects={
                "AttentionPooling1D":
                    AttentionPooling1D,

                (
                    "TAKlasifikasiBerita>"
                    "AttentionPooling1D"
                ):
                    AttentionPooling1D,
            },
        )
    )

    # Warm-up prediction agar prediksi pertama dashboard
    # tidak terlalu lama.
    dummy_sequence = np.zeros(
        (
            1,
            MAX_SEQUENCE_LENGTH,
        ),
        dtype=np.int32,
    )

    cnn_model.predict(
        dummy_sequence,
        verbose=0,
    )

    attention_bilstm_model.predict(
        dummy_sequence,
        verbose=0,
    )

    return {
        "cnn":
            cnn_model,

        "attention_bilstm":
            attention_bilstm_model,
    }


# =============================================================================
# SINGLE MODEL PREDICTION
# =============================================================================

def predict_single_model(
    model: tf.keras.Model,
    sequence: np.ndarray,
    label_mapping: dict[int, str],
) -> dict[str, Any]:
    """
    Menjalankan prediksi menggunakan satu model.

    Output:
    - predicted_index
    - predicted_label
    - confidence
    - probabilitas seluruh kelas
    - waktu inference
    """

    start_time = time.perf_counter()

    probabilities = model.predict(
        sequence,
        verbose=0,
    )[0]

    inference_time = (
        time.perf_counter()
        - start_time
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if len(probabilities) != len(
        label_mapping
    ):
        raise ValueError(
            "Jumlah output model tidak sesuai "
            "dengan jumlah label."
        )

    predicted_index = int(
        np.argmax(
            probabilities
        )
    )

    predicted_label = (
        label_mapping[
            predicted_index
        ]
    )

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
            inference_time,

        "inference_time_ms":
            inference_time
            * 1000,
    }


# =============================================================================
# COMPLETE NEWS PREDICTION
# =============================================================================

def predict_news(
    title: str,
    description: str,
) -> dict[str, Any]:
    """
    Menjalankan seluruh proses prediksi dashboard.

    Alur:
    Title + Description
    -> cleaning
    -> K2 representation
    -> vectorization
    -> CNN prediction
    -> Attention-BiLSTM prediction
    -> model comparison
    """

    models = load_models()

    vectorizer = (
        load_text_vectorizer()
    )

    label_mapping = (
        load_label_mapping()
    )

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

    attention_result = (
        predict_single_model(
            model=models[
                "attention_bilstm"
            ],
            sequence=sequence,
            label_mapping=label_mapping,
        )
    )

    model_agreement = (
        cnn_result[
            "predicted_label"
        ]
        == attention_result[
            "predicted_label"
        ]
    )

    # CNN digunakan sebagai hasil rekomendasi utama
    # karena CNN K2 merupakan model terbaik penelitian.
    recommended_prediction = {
        "source_model":
            "CNN K2",

        "predicted_label":
            cnn_result[
                "predicted_label"
            ],

        "confidence":
            cnn_result[
                "confidence"
            ],
    }

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
            model_agreement,

        "recommended_prediction":
            recommended_prediction,
    }


# =============================================================================
# TERMINAL TEST
# =============================================================================

def main() -> None:
    """
    Menguji inference utility sebelum digunakan oleh Streamlit.
    """

    print("=" * 80)
    print(
        "STREAMLIT INFERENCE UTILITY TEST"
    )
    print("=" * 80)

    print(
        "\nMemuat vocabulary..."
    )

    vocabulary = load_vocabulary()

    print(
        f"Vocabulary size        : "
        f"{len(vocabulary):,}"
    )

    print(
        "Memuat label mapping..."
    )

    label_mapping = (
        load_label_mapping()
    )

    print(
        f"Label mapping          : "
        f"{label_mapping}"
    )

    print(
        "Membangun vectorizer..."
    )

    vectorizer = (
        load_text_vectorizer()
    )

    print(
        f"Sequence length        : "
        f"{MAX_SEQUENCE_LENGTH}"
    )

    print(
        "Memuat kedua model..."
    )

    models = load_models()

    print(
        f"CNN input              : "
        f"{models['cnn'].input_shape}"
    )

    print(
        f"Attention-BiLSTM input : "
        f"{models['attention_bilstm'].input_shape}"
    )

    sample_title = (
        "Rupiah Menguat terhadap Dolar AS"
    )

    sample_description = (
        "Nilai tukar rupiah menguat setelah "
        "Bank Indonesia mengumumkan kebijakan "
        "stabilisasi pasar keuangan."
    )

    print("\nContoh input:")
    print(
        f"Title       : {sample_title}"
    )
    print(
        f"Description : {sample_description}"
    )

    result = predict_news(
        title=sample_title,
        description=sample_description,
    )

    print("\nHasil prediksi:")

    print(
        f"CNN                 : "
        f"{result['cnn']['predicted_label']} "
        f"({result['cnn']['confidence']:.2%})"
    )

    print(
        f"Attention-BiLSTM    : "
        f"{result['attention_bilstm']['predicted_label']} "
        f"({result['attention_bilstm']['confidence']:.2%})"
    )

    print(
        f"Kedua model sepakat : "
        f"{result['model_agreement']}"
    )

    print(
        f"Rekomendasi sistem  : "
        f"{result['recommended_prediction']['predicted_label']}"
    )

    print(
        f"Token non-padding   : "
        f"{result['input']['non_padding_tokens']}"
    )

    print("\n" + "=" * 80)
    print(
        "Inference utility berhasil dijalankan."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
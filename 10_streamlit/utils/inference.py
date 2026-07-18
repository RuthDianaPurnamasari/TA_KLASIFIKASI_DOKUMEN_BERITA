# =============================================================================
# STREAMLIT INFERENCE UTILITY
# =============================================================================
# File:
# 10_streamlit/utils/inference.py
#
# Fungsi:
# 1. Membaca seluruh artefak deployment.
# 2. Membersihkan Title dan Description.
# 3. Membentuk representasi K2:
#       Title + [SEP] + Description
# 4. Membentuk TextVectorization yang identik dengan training.
# 5. Memuat CNN K2 dan Attention-BiLSTM K2.
# 6. Memvalidasi arsitektur model dan probabilitas.
# 7. Menghasilkan prediksi untuk halaman Streamlit.
#
# File ini tidak mengatur tampilan dashboard.
# =============================================================================

from __future__ import annotations

import hashlib
import html
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import tensorflow as tf


# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================

CURRENT_FILE = Path(
    __file__
).resolve()

UTILS_DIR = CURRENT_FILE.parent

STREAMLIT_DIR = CURRENT_FILE.parents[1]

PROJECT_ROOT = CURRENT_FILE.parents[2]

MODELING_DIR = (
    PROJECT_ROOT
    / "5_modeling"
)

DEPLOYMENT_DIR = (
    PROJECT_ROOT
    / "8_save_models"
    / "deployment"
)


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

if str(STREAMLIT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(STREAMLIT_DIR),
    )

if str(MODELING_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(MODELING_DIR),
    )


# =============================================================================
# CUSTOM MODEL COMPONENTS
# =============================================================================

from attention_bilstm_model import (  # noqa: E402
    AttentionPooling1D,
)

from cnn_model import (  # noqa: E402
    MaskedGlobalMaxPooling1D,
    ZeroPaddingEmbeddingOutput,
)


# =============================================================================
# PATH ARTEFAK DEPLOYMENT
# =============================================================================

DEPLOYMENT_CONFIG_PATH = (
    DEPLOYMENT_DIR
    / "deployment_config.json"
)

VECTORIZER_CONFIG_PATH = (
    DEPLOYMENT_DIR
    / "vectorizer_config.json"
)

VOCABULARY_PATH = (
    DEPLOYMENT_DIR
    / "vocabulary.txt"
)

LABEL_MAPPING_PATH = (
    DEPLOYMENT_DIR
    / "label_mapping.json"
)


# =============================================================================
# KONFIGURASI PENELITIAN
# =============================================================================

EXPECTED_DATASET = "Kompas"
EXPECTED_SCENARIO_CODE = "K2"

EXPECTED_SEQUENCE_LENGTH = 60
EXPECTED_NUM_CLASSES = 4

EXPECTED_INDEX_TO_LABEL = {
    0: "bola",
    1: "global",
    2: "money",
    3: "tekno",
}

DEFAULT_SEPARATOR_TOKEN = "[SEP]"

DEFAULT_MODEL_FILENAMES = {
    "cnn":
        "cnn_k2.keras",

    "attention_bilstm":
        "attention_bilstm_k2.keras",
}

DEFAULT_DISPLAY_NAMES = {
    "cnn":
        "CNN",

    "attention_bilstm":
        "Attention-BiLSTM",
}

DEFAULT_EXPERIMENT_NAMES = {
    "cnn":
        "cnn_k2",

    "attention_bilstm":
        "attention_bilstm_k2",
}

PROBABILITY_ATOL = 1e-4


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
# RUNTIME CONTAINER
# =============================================================================

@dataclass
class InferenceRuntime:
    """
    Menyimpan seluruh komponen yang dibutuhkan untuk inference.
    """

    deployment_config: dict[str, Any]

    vectorizer_config: dict[str, Any]

    vocabulary: list[str]

    label_mapping: dict[int, str]

    vectorizer: tf.keras.layers.TextVectorization

    models: dict[str, tf.keras.Model]

    model_display_names: dict[str, str]

    model_experiment_names: dict[str, str]

    model_metadata: dict[str, dict[str, Any]]

    sequence_length: int

    num_classes: int

    separator_token: str

    required_fields: list[str]

    primary_model_key: str


# =============================================================================
# UTILITAS FILE
# =============================================================================

def validate_file(
    file_path: Path,
    description: str,
) -> Path:
    """
    Memastikan file tersedia, merupakan file, dan tidak kosong.
    """

    path = Path(
        file_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"{description} tidak ditemukan:\n"
            f"{path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Path {description} bukan file:\n"
            f"{path}"
        )

    if path.stat().st_size <= 0:
        raise ValueError(
            f"File {description} kosong:\n"
            f"{path}"
        )

    return path


def get_file_signature(
    file_path: Path,
) -> tuple[str, int, int]:
    """
    Membentuk cache key berdasarkan path, waktu modifikasi,
    dan ukuran file.
    """

    path = validate_file(
        file_path,
        "artefak",
    )

    stat = path.stat()

    return (
        str(
            path.resolve()
        ),
        int(
            stat.st_mtime_ns
        ),
        int(
            stat.st_size
        ),
    )


def sha256_file(
    file_path: Path,
) -> str:
    """
    Menghitung SHA-256 sebuah file.
    """

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


# =============================================================================
# JSON LOADER
# =============================================================================

@st.cache_data(
    show_spinner=False,
)
def _read_json_cached(
    file_path_string: str,
    modified_time_ns: int,
    file_size_bytes: int,
) -> dict[str, Any]:
    """
    Membaca JSON menggunakan cache.

    Waktu modifikasi dan ukuran file menjadi cache key.
    """

    del modified_time_ns
    del file_size_bytes

    file_path = Path(
        file_path_string
    )

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(
            file
        )

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            "Isi file JSON harus berupa dictionary:\n"
            f"{file_path}"
        )

    return data


def read_json(
    file_path: Path,
    description: str,
) -> dict[str, Any]:
    """
    Membaca file JSON yang sudah divalidasi.
    """

    path = validate_file(
        file_path,
        description,
    )

    (
        path_string,
        modified_time_ns,
        file_size_bytes,
    ) = get_file_signature(
        path
    )

    result = _read_json_cached(
        file_path_string=path_string,
        modified_time_ns=modified_time_ns,
        file_size_bytes=file_size_bytes,
    )

    return dict(
        result
    )


# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE,
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+"
    r"@[A-Za-z0-9.-]+"
    r"\.[A-Za-z]{2,}\b"
)

HTML_TAG_PATTERN = re.compile(
    r"<[^>]+>"
)

WHITESPACE_PATTERN = re.compile(
    r"\s+"
)

APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "‛": "'",
        "`": "'",
        "´": "'",
    }
)


def clean_text(
    text: Any,
) -> str:
    """
    Membersihkan teks dengan preprocessing yang sama
    seperti pipeline deployment.

    Tahapan:
    1. Konversi ke string.
    2. HTML entity decoding.
    3. Unicode normalization NFKC.
    4. Normalisasi apostrof.
    5. Menghapus tag HTML.
    6. Menghapus URL.
    7. Menghapus alamat email.
    8. Case folding.
    9. Menghapus karakter kontrol.
    10. Mempertahankan huruf Unicode, angka,
        apostrof, dan whitespace.
    11. Normalisasi whitespace.
    """

    if text is None:
        return ""

    value = str(
        text
    )

    value = html.unescape(
        value
    )

    value = unicodedata.normalize(
        "NFKC",
        value,
    )

    value = value.translate(
        APOSTROPHE_TRANSLATION
    )

    value = HTML_TAG_PATTERN.sub(
        " ",
        value,
    )

    value = URL_PATTERN.sub(
        " ",
        value,
    )

    value = EMAIL_PATTERN.sub(
        " ",
        value,
    )

    value = value.casefold()

    cleaned_characters: list[str] = []

    for character in value:
        category = unicodedata.category(
            character
        )

        if category.startswith(
            "C"
        ):
            cleaned_characters.append(
                " "
            )

            continue

        if (
            character.isalnum()
            or character.isspace()
            or character == "'"
        ):
            cleaned_characters.append(
                character
            )

        else:
            cleaned_characters.append(
                " "
            )

    value = "".join(
        cleaned_characters
    )

    value = WHITESPACE_PATTERN.sub(
        " ",
        value,
    ).strip()

    return value


def build_input_text(
    title: Any,
    description: Any,
    separator_token: str = DEFAULT_SEPARATOR_TOKEN,
    required_fields: list[str] | None = None,
) -> dict[str, str]:
    """
    Membentuk representasi K2:

    Title + [SEP] + Description
    """

    if required_fields is None:
        required_fields = [
            "title",
            "description",
        ]

    normalized_required_fields = {
        str(field)
        .strip()
        .lower()

        for field
        in required_fields
    }

    clean_title = clean_text(
        title
    )

    clean_description = clean_text(
        description
    )

    if (
        "title" in normalized_required_fields
        and not clean_title
    ):
        raise ValueError(
            "Title wajib diisi dan tidak boleh kosong."
        )

    if (
        "description" in normalized_required_fields
        and not clean_description
    ):
        raise ValueError(
            "Description wajib diisi dan tidak boleh kosong."
        )

    components = [
        component
        for component
        in [
            clean_title,
            clean_description,
        ]
        if component
    ]

    if not components:
        raise ValueError(
            "Title dan Description tidak boleh keduanya kosong."
        )

    combined_text = (
        f" {separator_token} "
    ).join(
        components
    )

    return {
        "raw_title":
            ""
            if title is None
            else str(title),

        "raw_description":
            ""
            if description is None
            else str(description),

        "clean_title":
            clean_title,

        "clean_description":
            clean_description,

        "combined_text":
            combined_text,
    }


# =============================================================================
# DEPLOYMENT CONFIGURATION
# =============================================================================

def load_deployment_config() -> dict[str, Any]:
    """
    Membaca dan memvalidasi deployment_config.json.
    """

    configuration = read_json(
        DEPLOYMENT_CONFIG_PATH,
        "deployment config",
    )

    required_keys = {
        "dataset",
        "scenario_code",
        "sequence_length",
        "num_classes",
        "vocabulary_size",
        "models",
        "input",
        "labels",
    }

    missing_keys = (
        required_keys
        - set(
            configuration.keys()
        )
    )

    if missing_keys:
        raise KeyError(
            "Deployment config tidak lengkap.\n"
            f"Key hilang: {sorted(missing_keys)}"
        )

    dataset = str(
        configuration[
            "dataset"
        ]
    ).strip()

    scenario_code = str(
        configuration[
            "scenario_code"
        ]
    ).strip()

    sequence_length = int(
        configuration[
            "sequence_length"
        ]
    )

    num_classes = int(
        configuration[
            "num_classes"
        ]
    )

    vocabulary_size = int(
        configuration[
            "vocabulary_size"
        ]
    )

    if (
        dataset.lower()
        != EXPECTED_DATASET.lower()
    ):
        raise ValueError(
            "Dataset deployment tidak sesuai.\n"
            f"Expected : {EXPECTED_DATASET}\n"
            f"Actual   : {dataset}"
        )

    if (
        scenario_code.upper()
        != EXPECTED_SCENARIO_CODE.upper()
    ):
        raise ValueError(
            "Skenario deployment tidak sesuai.\n"
            f"Expected : {EXPECTED_SCENARIO_CODE}\n"
            f"Actual   : {scenario_code}"
        )

    if (
        sequence_length
        != EXPECTED_SEQUENCE_LENGTH
    ):
        raise ValueError(
            "Sequence length deployment tidak sesuai.\n"
            f"Expected : {EXPECTED_SEQUENCE_LENGTH}\n"
            f"Actual   : {sequence_length}"
        )

    if (
        num_classes
        != EXPECTED_NUM_CLASSES
    ):
        raise ValueError(
            "Jumlah kelas deployment tidak sesuai.\n"
            f"Expected : {EXPECTED_NUM_CLASSES}\n"
            f"Actual   : {num_classes}"
        )

    if vocabulary_size <= 2:
        raise ValueError(
            "Vocabulary size pada deployment config tidak valid."
        )

    return configuration


def load_vectorizer_config() -> dict[str, Any]:
    """
    Membaca vectorizer_config.json.
    """

    return read_json(
        VECTORIZER_CONFIG_PATH,
        "vectorizer config",
    )


def find_recursive_values(
    data: Any,
    target_keys: set[str],
) -> list[Any]:
    """
    Mencari nilai berdasarkan key secara rekursif.
    """

    results: list[Any] = []

    if isinstance(
        data,
        dict,
    ):
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

    elif isinstance(
        data,
        list,
    ):
        for item in data:
            results.extend(
                find_recursive_values(
                    item,
                    target_keys,
                )
            )

    return results


def validate_vectorizer_config(
    vectorizer_config: dict[str, Any],
    sequence_length: int,
    vocabulary_size: int,
) -> None:
    """
    Memvalidasi sequence length dan vocabulary size.
    """

    sequence_values = find_recursive_values(
        vectorizer_config,
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

        except (
            TypeError,
            ValueError,
        ):
            continue

    if not parsed_sequence_values:
        raise KeyError(
            "Sequence length tidak ditemukan "
            "pada vectorizer config."
        )

    if (
        sequence_length
        not in parsed_sequence_values
    ):
        raise ValueError(
            "Sequence length vectorizer config tidak sesuai.\n"
            f"Expected  : {sequence_length}\n"
            f"Ditemukan : {parsed_sequence_values}"
        )

    vocabulary_values = find_recursive_values(
        vectorizer_config,
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

        except (
            TypeError,
            ValueError,
        ):
            continue

    invalid_values = [
        value
        for value
        in parsed_vocabulary_values
        if value < vocabulary_size
    ]

    if invalid_values:
        raise ValueError(
            "Vocabulary size pada vectorizer config "
            "lebih kecil dari vocabulary deployment.\n"
            f"Vocabulary deployment : {vocabulary_size}\n"
            f"Nilai config          : {parsed_vocabulary_values}"
        )


# =============================================================================
# VOCABULARY
# =============================================================================

@st.cache_data(
    show_spinner=False,
)
def _load_vocabulary_cached(
    file_path_string: str,
    modified_time_ns: int,
    file_size_bytes: int,
) -> tuple[str, ...]:
    """
    Membaca vocabulary menggunakan cache.
    """

    del modified_time_ns
    del file_size_bytes

    file_path = Path(
        file_path_string
    )

    vocabulary = (
        file_path
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

    return tuple(
        vocabulary
    )


def load_vocabulary() -> list[str]:
    """
    Membaca dan memvalidasi vocabulary K2.
    """

    path = validate_file(
        VOCABULARY_PATH,
        "vocabulary",
    )

    (
        path_string,
        modified_time_ns,
        file_size_bytes,
    ) = get_file_signature(
        path
    )

    vocabulary = list(
        _load_vocabulary_cached(
            file_path_string=path_string,
            modified_time_ns=modified_time_ns,
            file_size_bytes=file_size_bytes,
        )
    )

    if len(vocabulary) <= 2:
        raise ValueError(
            "Vocabulary deployment tidak valid.\n"
            f"Vocabulary size: {len(vocabulary)}"
        )

    if vocabulary[0] != "":
        raise ValueError(
            "Vocabulary index 0 harus berupa token padding kosong.\n"
            f"Token index 0: {repr(vocabulary[0])}"
        )

    if vocabulary[1] != "[UNK]":
        raise ValueError(
            "Vocabulary index 1 harus berupa token [UNK].\n"
            f"Token index 1: {repr(vocabulary[1])}"
        )

    return vocabulary


# =============================================================================
# LABEL MAPPING
# =============================================================================

def extract_index_to_label(
    data: dict[str, Any],
) -> dict[int, str]:
    """
    Membaca index_to_label dari beberapa struktur JSON.
    """

    mapping: Any = data

    for dataset_key in [
        "Kompas",
        "kompas",
        "KOMPAS",
    ]:
        if (
            isinstance(
                data,
                dict,
            )
            and dataset_key in data
        ):
            mapping = data[
                dataset_key
            ]

            break

    if not isinstance(
        mapping,
        dict,
    ):
        raise ValueError(
            "Struktur label mapping tidak valid."
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
        for key
        in mapping.keys()
    ):
        result = {
            int(index):
                str(label)
                .strip()
                .lower()

            for index, label
            in mapping.items()
        }

    else:
        raise ValueError(
            "index_to_label tidak ditemukan "
            "pada label mapping."
        )

    return dict(
        sorted(
            result.items()
        )
    )


def load_label_mapping(
    deployment_config: dict[str, Any] | None = None,
) -> dict[int, str]:
    """
    Membaca label mapping file dan membandingkannya
    dengan deployment config.
    """

    if deployment_config is None:
        deployment_config = (
            load_deployment_config()
        )

    mapping_data = read_json(
        LABEL_MAPPING_PATH,
        "label mapping",
    )

    file_mapping = extract_index_to_label(
        mapping_data
    )

    config_mapping = extract_index_to_label(
        deployment_config[
            "labels"
        ]
    )

    if file_mapping != config_mapping:
        raise ValueError(
            "Label mapping file dan deployment config berbeda.\n"
            f"File   : {file_mapping}\n"
            f"Config : {config_mapping}"
        )

    if (
        file_mapping
        != EXPECTED_INDEX_TO_LABEL
    ):
        raise ValueError(
            "Label mapping tidak sesuai dengan penelitian.\n"
            f"Expected : {EXPECTED_INDEX_TO_LABEL}\n"
            f"Actual   : {file_mapping}"
        )

    return file_mapping


# =============================================================================
# TEXT VECTORIZATION
# =============================================================================

@st.cache_resource(
    show_spinner=False,
)
def _build_text_vectorizer_cached(
    vocabulary_path_string: str,
    modified_time_ns: int,
    file_size_bytes: int,
    sequence_length: int,
) -> tf.keras.layers.TextVectorization:
    """
    Membangun TextVectorization menggunakan vocabulary training.
    """

    vocabulary = list(
        _load_vocabulary_cached(
            file_path_string=vocabulary_path_string,
            modified_time_ns=modified_time_ns,
            file_size_bytes=file_size_bytes,
        )
    )

    # Token kosong dan [UNK] tidak diberikan kepada constructor
    # karena TextVectorization menambahkannya secara otomatis.
    vectorizer = (
        tf.keras.layers.TextVectorization(
            standardize=None,
            split="whitespace",
            output_mode="int",
            output_sequence_length=sequence_length,
            vocabulary=vocabulary[2:],
            name="streamlit_text_vectorizer",
        )
    )

    reconstructed_vocabulary = (
        vectorizer.get_vocabulary()
    )

    if (
        reconstructed_vocabulary
        != vocabulary
    ):
        mismatch_indices: list[int] = []

        maximum_comparison = min(
            len(
                reconstructed_vocabulary
            ),
            len(
                vocabulary
            ),
        )

        for index in range(
            maximum_comparison
        ):
            if (
                reconstructed_vocabulary[
                    index
                ]
                != vocabulary[
                    index
                ]
            ):
                mismatch_indices.append(
                    index
                )

                if (
                    len(
                        mismatch_indices
                    )
                    >= 10
                ):
                    break

        raise ValueError(
            "Vocabulary vectorizer Streamlit tidak identik "
            "dengan vocabulary training.\n"
            f"Vocabulary training   : {len(vocabulary)}\n"
            f"Vocabulary vectorizer : "
            f"{len(reconstructed_vocabulary)}\n"
            f"Mismatch index        : {mismatch_indices}"
        )

    return vectorizer


def load_text_vectorizer(
) -> tf.keras.layers.TextVectorization:
    """
    Membuat atau mengambil TextVectorization dari cache.
    """

    deployment_config = (
        load_deployment_config()
    )

    sequence_length = int(
        deployment_config[
            "sequence_length"
        ]
    )

    path = validate_file(
        VOCABULARY_PATH,
        "vocabulary",
    )

    (
        path_string,
        modified_time_ns,
        file_size_bytes,
    ) = get_file_signature(
        path
    )

    vectorizer = (
        _build_text_vectorizer_cached(
            vocabulary_path_string=path_string,
            modified_time_ns=modified_time_ns,
            file_size_bytes=file_size_bytes,
            sequence_length=sequence_length,
        )
    )

    return vectorizer


def vectorize_text(
    text: str,
    vectorizer: (
        tf.keras.layers.TextVectorization
    ),
    sequence_length: int | None = None,
    vocabulary_size: int | None = None,
) -> np.ndarray:
    """
    Mengubah satu teks menjadi sequence integer.
    """

    if sequence_length is None:
        deployment_config = (
            load_deployment_config()
        )

        sequence_length = int(
            deployment_config[
                "sequence_length"
            ]
        )

    if vocabulary_size is None:
        vocabulary_size = len(
            load_vocabulary()
        )

    sequence_tensor = vectorizer(
        tf.constant(
            [
                text
            ],
            dtype=tf.string,
        )
    )

    sequence = np.asarray(
        sequence_tensor,
        dtype=np.int32,
    )

    expected_shape = (
        1,
        sequence_length,
    )

    if sequence.shape != expected_shape:
        raise ValueError(
            "Shape sequence tidak sesuai.\n"
            f"Expected : {expected_shape}\n"
            f"Actual   : {sequence.shape}"
        )

    if np.any(
        sequence < 0
    ):
        raise ValueError(
            "Sequence mengandung token ID negatif."
        )

    maximum_token_id = int(
        np.max(
            sequence
        )
    )

    if (
        maximum_token_id
        >= vocabulary_size
    ):
        raise ValueError(
            "Token ID melebihi vocabulary size.\n"
            f"Maximum token ID : {maximum_token_id}\n"
            f"Vocabulary size  : {vocabulary_size}"
        )

    return sequence


def get_sequence_information(
    sequence: np.ndarray,
    sequence_length: int,
) -> dict[str, Any]:
    """
    Membentuk informasi diagnostik sequence.
    """

    non_padding_tokens = int(
        np.count_nonzero(
            sequence
        )
    )

    oov_tokens = int(
        np.count_nonzero(
            sequence == 1
        )
    )

    return {
        "sequence_shape":
            list(
                sequence.shape
            ),

        "non_padding_tokens":
            non_padding_tokens,

        "padding_tokens":
            int(
                sequence_length
                - non_padding_tokens
            ),

        "oov_tokens":
            oov_tokens,

        "maximum_token_id":
            int(
                np.max(
                    sequence
                )
            ),

        "possibly_truncated":
            bool(
                non_padding_tokens
                == sequence_length
            ),
    }


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

def resolve_model_information(
    deployment_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Membaca nama file dan metadata model dari deployment config.
    """

    models_config = deployment_config.get(
        "models",
        {},
    )

    if not isinstance(
        models_config,
        dict,
    ):
        raise ValueError(
            "Bagian models pada deployment config tidak valid."
        )

    result: dict[
        str,
        dict[str, Any]
    ] = {}

    for model_key in [
        "cnn",
        "attention_bilstm",
    ]:
        model_config = models_config.get(
            model_key,
            {},
        )

        if not isinstance(
            model_config,
            dict,
        ):
            model_config = {}

        available = bool(
            model_config.get(
                "available",
                True,
            )
        )

        if not available:
            raise RuntimeError(
                f"Model {model_key} ditandai tidak tersedia "
                "pada deployment config."
            )

        filename = str(
            model_config.get(
                "filename",
                DEFAULT_MODEL_FILENAMES[
                    model_key
                ],
            )
        ).strip()

        if not filename:
            filename = (
                DEFAULT_MODEL_FILENAMES[
                    model_key
                ]
            )

        result[
            model_key
        ] = {
            "path":
                DEPLOYMENT_DIR
                / filename,

            "filename":
                filename,

            "display_name":
                str(
                    model_config.get(
                        "display_name",
                        DEFAULT_DISPLAY_NAMES[
                            model_key
                        ],
                    )
                ).strip(),

            "experiment_name":
                str(
                    model_config.get(
                        "experiment_name",
                        DEFAULT_EXPERIMENT_NAMES[
                            model_key
                        ],
                    )
                ).strip(),

            "expected_sha256":
                model_config.get(
                    "sha256"
                ),
        }

    return result


# =============================================================================
# MODEL VALIDATION
# =============================================================================

def normalize_shape(
    shape: Any,
    shape_name: str,
) -> tuple[Any, ...]:
    """
    Menormalisasi input atau output shape.
    """

    if isinstance(
        shape,
        list,
    ):
        if len(shape) != 1:
            raise ValueError(
                f"Model harus mempunyai tepat satu {shape_name}.\n"
                f"Shape: {shape}"
            )

        shape = shape[0]

    return tuple(
        shape
    )


def find_embedding_layer(
    model: tf.keras.Model,
) -> tf.keras.layers.Embedding:
    """
    Mencari layer Embedding pada model.
    """

    for layer in model.layers:
        if isinstance(
            layer,
            tf.keras.layers.Embedding,
        ):
            return layer

    raise ValueError(
        "Layer Embedding tidak ditemukan pada model."
    )


def validate_model(
    model: tf.keras.Model,
    model_key: str,
    sequence_length: int,
    num_classes: int,
    vocabulary_size: int,
) -> dict[str, Any]:
    """
    Memvalidasi arsitektur model.
    """

    input_shape = normalize_shape(
        model.input_shape,
        "input",
    )

    output_shape = normalize_shape(
        model.output_shape,
        "output",
    )

    expected_input_shape = (
        None,
        sequence_length,
    )

    expected_output_shape = (
        None,
        num_classes,
    )

    if (
        input_shape
        != expected_input_shape
    ):
        raise ValueError(
            f"Input shape {model_key} tidak sesuai.\n"
            f"Expected : {expected_input_shape}\n"
            f"Actual   : {input_shape}"
        )

    if (
        output_shape
        != expected_output_shape
    ):
        raise ValueError(
            f"Output shape {model_key} tidak sesuai.\n"
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

    if (
        embedding_input_dim
        != vocabulary_size
    ):
        raise ValueError(
            f"Embedding input_dim {model_key} "
            "tidak sama dengan vocabulary size.\n"
            f"Embedding input_dim : {embedding_input_dim}\n"
            f"Vocabulary size     : {vocabulary_size}"
        )

    layer_types = {
        type(layer).__name__
        for layer
        in model.layers
    }

    if model_key == "cnn":
        required_custom_layers = {
            "ZeroPaddingEmbeddingOutput",
            "MaskedGlobalMaxPooling1D",
        }

    elif model_key == "attention_bilstm":
        required_custom_layers = {
            "AttentionPooling1D",
        }

    else:
        required_custom_layers = set()

    missing_custom_layers = (
        required_custom_layers
        - layer_types
    )

    if missing_custom_layers:
        raise ValueError(
            f"Custom layer {model_key} tidak lengkap.\n"
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

        "embedding_layer_name":
            embedding_layer.name,

        "embedding_input_dim":
            embedding_input_dim,

        "embedding_output_dim":
            embedding_output_dim,

        "custom_layers":
            sorted(
                required_custom_layers
            ),
    }


# =============================================================================
# MODEL LOADER
# =============================================================================

@st.cache_resource(
    show_spinner=False,
)
def _load_model_cached(
    model_path_string: str,
    modified_time_ns: int,
    file_size_bytes: int,
    expected_sha256: str,
    model_key: str,
    sequence_length: int,
    num_classes: int,
    vocabulary_size: int,
) -> tf.keras.Model:
    """
    Memuat satu model menggunakan cache.

    Cache berubah ketika model, ukuran file, atau waktu
    modifikasi berubah.
    """

    del modified_time_ns
    del file_size_bytes

    model_path = Path(
        model_path_string
    )

    if expected_sha256:
        actual_sha256 = sha256_file(
            model_path
        )

        if (
            actual_sha256
            != expected_sha256
        ):
            raise ValueError(
                f"SHA-256 model {model_key} tidak sesuai.\n"
                f"Expected : {expected_sha256}\n"
                f"Actual   : {actual_sha256}"
            )

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects=CUSTOM_OBJECTS,
    )

    validate_model(
        model=model,
        model_key=model_key,
        sequence_length=sequence_length,
        num_classes=num_classes,
        vocabulary_size=vocabulary_size,
    )

    # Warm-up agar prediksi pertama pengguna
    # tidak memasukkan waktu inisialisasi model.
    dummy_sequence = np.zeros(
        (
            1,
            sequence_length,
        ),
        dtype=np.int32,
    )

    output = model(
        dummy_sequence,
        training=False,
    )

    _ = np.asarray(
        output
    )

    return model


def load_models(
    deployment_config: dict[str, Any] | None = None,
    vocabulary_size: int | None = None,
) -> dict[str, tf.keras.Model]:
    """
    Memuat CNN K2 dan Attention-BiLSTM K2.
    """

    if deployment_config is None:
        deployment_config = (
            load_deployment_config()
        )

    if vocabulary_size is None:
        vocabulary_size = len(
            load_vocabulary()
        )

    sequence_length = int(
        deployment_config[
            "sequence_length"
        ]
    )

    num_classes = int(
        deployment_config[
            "num_classes"
        ]
    )

    model_information = (
        resolve_model_information(
            deployment_config
        )
    )

    models: dict[
        str,
        tf.keras.Model
    ] = {}

    for model_key, information in (
        model_information.items()
    ):
        model_path = validate_file(
            information[
                "path"
            ],
            f"model {model_key}",
        )

        (
            model_path_string,
            modified_time_ns,
            file_size_bytes,
        ) = get_file_signature(
            model_path
        )

        expected_sha256 = str(
            information.get(
                "expected_sha256"
            )
            or ""
        ).strip()

        models[
            model_key
        ] = _load_model_cached(
            model_path_string=model_path_string,
            modified_time_ns=modified_time_ns,
            file_size_bytes=file_size_bytes,
            expected_sha256=expected_sha256,
            model_key=model_key,
            sequence_length=sequence_length,
            num_classes=num_classes,
            vocabulary_size=vocabulary_size,
        )

    return models


# =============================================================================
# RUNTIME LOADER
# =============================================================================

def determine_primary_model_key(
    deployment_config: dict[str, Any],
    model_information: dict[str, dict[str, Any]],
) -> str:
    """
    Menentukan model rekomendasi utama berdasarkan
    best_research_model.
    """

    best_model_config = deployment_config.get(
        "best_research_model",
        {},
    )

    primary_experiment_name = str(
        best_model_config.get(
            "experiment_name",
            "cnn_k2",
        )
    ).strip()

    for model_key, information in (
        model_information.items()
    ):
        if (
            information[
                "experiment_name"
            ]
            == primary_experiment_name
        ):
            return model_key

    return "cnn"


def load_inference_runtime() -> InferenceRuntime:
    """
    Memuat seluruh artefak inference Streamlit.
    """

    deployment_config = (
        load_deployment_config()
    )

    vectorizer_config = (
        load_vectorizer_config()
    )

    sequence_length = int(
        deployment_config[
            "sequence_length"
        ]
    )

    num_classes = int(
        deployment_config[
            "num_classes"
        ]
    )

    configured_vocabulary_size = int(
        deployment_config[
            "vocabulary_size"
        ]
    )

    vocabulary = load_vocabulary()

    if (
        len(
            vocabulary
        )
        != configured_vocabulary_size
    ):
        raise ValueError(
            "Vocabulary size deployment config "
            "berbeda dengan file vocabulary.\n"
            f"Config : {configured_vocabulary_size}\n"
            f"File   : {len(vocabulary)}"
        )

    validate_vectorizer_config(
        vectorizer_config=vectorizer_config,
        sequence_length=sequence_length,
        vocabulary_size=len(
            vocabulary
        ),
    )

    label_mapping = load_label_mapping(
        deployment_config
    )

    vectorizer = load_text_vectorizer()

    input_config = deployment_config.get(
        "input",
        {},
    )

    separator_token = str(
        input_config.get(
            "separator_token",
            DEFAULT_SEPARATOR_TOKEN,
        )
    ).strip()

    if not separator_token:
        raise ValueError(
            "Separator token tidak boleh kosong."
        )

    if separator_token not in vocabulary:
        raise ValueError(
            "Separator token tidak ditemukan "
            "pada vocabulary training.\n"
            f"Separator token: {separator_token}"
        )

    required_fields = input_config.get(
        "required_fields",
        [
            "title",
            "description",
        ],
    )

    if not isinstance(
        required_fields,
        list,
    ):
        raise ValueError(
            "required_fields pada deployment config "
            "harus berupa list."
        )

    required_fields = [
        str(field)
        .strip()
        .lower()

        for field
        in required_fields
    ]

    models = load_models(
        deployment_config=deployment_config,
        vocabulary_size=len(
            vocabulary
        ),
    )

    model_information = (
        resolve_model_information(
            deployment_config
        )
    )

    model_display_names = {
        model_key:
            information[
                "display_name"
            ]

        for model_key, information
        in model_information.items()
    }

    model_experiment_names = {
        model_key:
            information[
                "experiment_name"
            ]

        for model_key, information
        in model_information.items()
    }

    model_metadata: dict[
        str,
        dict[str, Any]
    ] = {}

    for model_key, model in models.items():
        information = model_information[
            model_key
        ]

        model_path = Path(
            information[
                "path"
            ]
        )

        model_metadata[
            model_key
        ] = {
            "path":
                str(
                    model_path
                ),

            "filename":
                model_path.name,

            "display_name":
                information[
                    "display_name"
                ],

            "experiment_name":
                information[
                    "experiment_name"
                ],

            "size_bytes":
                int(
                    model_path.stat().st_size
                ),

            "sha256":
                sha256_file(
                    model_path
                ),

            "architecture":
                validate_model(
                    model=model,
                    model_key=model_key,
                    sequence_length=sequence_length,
                    num_classes=num_classes,
                    vocabulary_size=len(
                        vocabulary
                    ),
                ),
        }

    primary_model_key = (
        determine_primary_model_key(
            deployment_config=deployment_config,
            model_information=model_information,
        )
    )

    return InferenceRuntime(
        deployment_config=deployment_config,
        vectorizer_config=vectorizer_config,
        vocabulary=vocabulary,
        label_mapping=label_mapping,
        vectorizer=vectorizer,
        models=models,
        model_display_names=model_display_names,
        model_experiment_names=model_experiment_names,
        model_metadata=model_metadata,
        sequence_length=sequence_length,
        num_classes=num_classes,
        separator_token=separator_token,
        required_fields=required_fields,
        primary_model_key=primary_model_key,
    )


# =============================================================================
# PROBABILITY VALIDATION
# =============================================================================

def validate_probabilities(
    probabilities: np.ndarray,
    num_classes: int,
    context: str,
) -> np.ndarray:
    """
    Memvalidasi probabilitas keluaran model.
    """

    values = np.asarray(
        probabilities,
        dtype=np.float64,
    ).reshape(
        -1
    )

    expected_shape = (
        num_classes,
    )

    if values.shape != expected_shape:
        raise ValueError(
            f"Shape probabilitas {context} tidak sesuai.\n"
            f"Expected : {expected_shape}\n"
            f"Actual   : {values.shape}"
        )

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise ValueError(
            f"Probabilitas {context} mengandung "
            "NaN atau infinity."
        )

    if (
        np.any(
            values < -1e-7
        )
        or np.any(
            values > 1.0 + 1e-7
        )
    ):
        raise ValueError(
            f"Probabilitas {context} berada "
            "di luar rentang 0 sampai 1."
        )

    probability_sum = float(
        np.sum(
            values
        )
    )

    if not np.isclose(
        probability_sum,
        1.0,
        atol=PROBABILITY_ATOL,
    ):
        raise ValueError(
            f"Jumlah probabilitas {context} "
            "tidak mendekati 1.\n"
            f"Jumlah: {probability_sum}"
        )

    return values


# =============================================================================
# SINGLE MODEL PREDICTION
# =============================================================================

def predict_single_model(
    model: tf.keras.Model,
    sequence: np.ndarray,
    label_mapping: dict[int, str],
    num_classes: int | None = None,
    model_name: str = "model",
) -> dict[str, Any]:
    """
    Menjalankan prediksi menggunakan satu model.
    """

    if num_classes is None:
        num_classes = len(
            label_mapping
        )

    start_time = time.perf_counter()

    output = model(
        sequence,
        training=False,
    )

    inference_time_seconds = (
        time.perf_counter()
        - start_time
    )

    output_array = np.asarray(
        output,
        dtype=np.float64,
    )

    if output_array.shape != (
        1,
        num_classes,
    ):
        raise ValueError(
            f"Shape output {model_name} tidak sesuai.\n"
            f"Expected : {(1, num_classes)}\n"
            f"Actual   : {output_array.shape}"
        )

    probabilities = validate_probabilities(
        probabilities=output_array[
            0
        ],
        num_classes=num_classes,
        context=model_name,
    )

    predicted_index = int(
        np.argmax(
            probabilities
        )
    )

    if (
        predicted_index
        not in label_mapping
    ):
        raise KeyError(
            f"Index {predicted_index} tidak ditemukan "
            "pada label mapping."
        )

    predicted_label = (
        label_mapping[
            predicted_index
        ]
    )

    class_probabilities = {
        label_mapping[
            index
        ]:
            float(
                probabilities[
                    index
                ]
            )

        for index
        in range(
            num_classes
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
            float(
                inference_time_seconds
            ),

        "inference_time_ms":
            float(
                inference_time_seconds
                * 1000
            ),
    }


# =============================================================================
# COMPLETE NEWS PREDICTION
# =============================================================================

def predict_news(
    title: Any,
    description: Any,
) -> dict[str, Any]:
    """
    Menjalankan seluruh proses prediksi dashboard.

    Alur:
    Title dan Description
    -> preprocessing
    -> K2 representation
    -> vectorization
    -> CNN prediction
    -> Attention-BiLSTM prediction
    -> model comparison
    """

    runtime = load_inference_runtime()

    prepared_input = build_input_text(
        title=title,
        description=description,
        separator_token=(
            runtime.separator_token
        ),
        required_fields=(
            runtime.required_fields
        ),
    )

    sequence = vectorize_text(
        text=prepared_input[
            "combined_text"
        ],
        vectorizer=runtime.vectorizer,
        sequence_length=(
            runtime.sequence_length
        ),
        vocabulary_size=len(
            runtime.vocabulary
        ),
    )

    sequence_information = (
        get_sequence_information(
            sequence=sequence,
            sequence_length=(
                runtime.sequence_length
            ),
        )
    )

    model_results: dict[
        str,
        dict[str, Any]
    ] = {}

    for model_key in [
        "cnn",
        "attention_bilstm",
    ]:
        model_results[
            model_key
        ] = predict_single_model(
            model=runtime.models[
                model_key
            ],
            sequence=sequence,
            label_mapping=(
                runtime.label_mapping
            ),
            num_classes=(
                runtime.num_classes
            ),
            model_name=(
                runtime.model_display_names[
                    model_key
                ]
            ),
        )

    model_agreement = (
        model_results[
            "cnn"
        ][
            "predicted_label"
        ]
        == model_results[
            "attention_bilstm"
        ][
            "predicted_label"
        ]
    )

    primary_model_key = (
        runtime.primary_model_key
    )

    primary_result = (
        model_results[
            primary_model_key
        ]
    )

    return {
        "input": {
            **prepared_input,
            **sequence_information,

            "sequence":
                sequence[
                    0
                ].tolist(),
        },

        "cnn":
            model_results[
                "cnn"
            ],

        "attention_bilstm":
            model_results[
                "attention_bilstm"
            ],

        "model_agreement":
            bool(
                model_agreement
            ),

        "recommended_prediction": {
            "source_model_key":
                primary_model_key,

            "source_model":
                runtime.model_display_names[
                    primary_model_key
                ],

            "experiment_name":
                runtime.model_experiment_names[
                    primary_model_key
                ],

            "predicted_index":
                primary_result[
                    "predicted_index"
                ],

            "predicted_label":
                primary_result[
                    "predicted_label"
                ],

            "confidence":
                primary_result[
                    "confidence"
                ],
        },

        "runtime": {
            "dataset":
                runtime.deployment_config[
                    "dataset"
                ],

            "scenario_code":
                runtime.deployment_config[
                    "scenario_code"
                ],

            "scenario_name":
                runtime.deployment_config.get(
                    "scenario_name",
                    "Title + Description",
                ),

            "sequence_length":
                runtime.sequence_length,

            "vocabulary_size":
                len(
                    runtime.vocabulary
                ),

            "num_classes":
                runtime.num_classes,

            "separator_token":
                runtime.separator_token,
        },
    }


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

def clear_inference_cache() -> None:
    """
    Menghapus cache inference Streamlit.

    Fungsi dapat dipanggil melalui tombol refresh dashboard.
    """

    _read_json_cached.clear()
    _load_vocabulary_cached.clear()
    _build_text_vectorizer_cached.clear()
    _load_model_cached.clear()


# =============================================================================
# TERMINAL TEST
# =============================================================================

def main() -> None:
    """
    Menguji utility inference sebelum digunakan
    oleh halaman Streamlit.
    """

    print("=" * 80)
    print(
        "STREAMLIT INFERENCE UTILITY TEST"
    )
    print("=" * 80)

    print(
        "\nMemuat runtime inference..."
    )

    runtime = load_inference_runtime()

    print(
        f"Dataset                 : "
        f"{runtime.deployment_config['dataset']}"
    )

    print(
        f"Skenario                : "
        f"{runtime.deployment_config['scenario_code']}"
    )

    print(
        f"Representasi            : "
        f"{runtime.deployment_config.get('scenario_name')}"
    )

    print(
        f"Sequence length         : "
        f"{runtime.sequence_length}"
    )

    print(
        f"Vocabulary size         : "
        f"{len(runtime.vocabulary):,}"
    )

    print(
        f"Separator token         : "
        f"{runtime.separator_token}"
    )

    print(
        f"Label mapping           : "
        f"{runtime.label_mapping}"
    )

    print(
        f"Model utama             : "
        f"{runtime.model_display_names[runtime.primary_model_key]}"
    )

    print(
        "\nModel deployment:"
    )

    for model_key in [
        "cnn",
        "attention_bilstm",
    ]:
        metadata = runtime.model_metadata[
            model_key
        ]

        architecture = metadata[
            "architecture"
        ]

        print(
            f"\n- {metadata['display_name']}"
        )

        print(
            f"  File                  : "
            f"{metadata['filename']}"
        )

        print(
            f"  Input                 : "
            f"{architecture['input_shape']}"
        )

        print(
            f"  Output                : "
            f"{architecture['output_shape']}"
        )

        print(
            f"  Parameter             : "
            f"{architecture['parameter_count']:,}"
        )

    sample_title = (
        "Rupiah Menguat terhadap Dolar AS"
    )

    sample_description = (
        "Nilai tukar rupiah menguat setelah "
        "Bank Indonesia mengumumkan kebijakan "
        "stabilisasi pasar keuangan."
    )

    print(
        "\nContoh input:"
    )

    print(
        f"Title                   : "
        f"{sample_title}"
    )

    print(
        f"Description             : "
        f"{sample_description}"
    )

    result = predict_news(
        title=sample_title,
        description=sample_description,
    )

    print(
        "\nHasil prediksi:"
    )

    print(
        f"CNN                     : "
        f"{result['cnn']['predicted_label']} "
        f"({result['cnn']['confidence']:.2%})"
    )

    print(
        f"Attention-BiLSTM        : "
        f"{result['attention_bilstm']['predicted_label']} "
        f"({result['attention_bilstm']['confidence']:.2%})"
    )

    print(
        f"Kedua model sepakat     : "
        f"{result['model_agreement']}"
    )

    print(
        f"Rekomendasi sistem      : "
        f"{result['recommended_prediction']['predicted_label']}"
    )

    print(
        f"Model rekomendasi       : "
        f"{result['recommended_prediction']['source_model']}"
    )

    print(
        f"Confidence              : "
        f"{result['recommended_prediction']['confidence']:.2%}"
    )

    print(
        f"Token non-padding       : "
        f"{result['input']['non_padding_tokens']}"
    )

    print(
        f"Token OOV               : "
        f"{result['input']['oov_tokens']}"
    )

    print(
        f"Terindikasi terpotong   : "
        f"{result['input']['possibly_truncated']}"
    )

    print(
        "\nProbabilitas CNN:"
    )

    for label, probability in sorted(
        result[
            "cnn"
        ][
            "probabilities"
        ].items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(
            f"  {label:<8}: "
            f"{probability:.2%}"
        )

    print(
        "\nProbabilitas Attention-BiLSTM:"
    )

    for label, probability in sorted(
        result[
            "attention_bilstm"
        ][
            "probabilities"
        ].items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(
            f"  {label:<8}: "
            f"{probability:.2%}"
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "Inference utility berhasil dijalankan."
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
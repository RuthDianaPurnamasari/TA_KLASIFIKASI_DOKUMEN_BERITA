# =============================================================================
# STEP 8.2 - INFERENCE PIPELINE
# =============================================================================
# File:
# 8_save_models/02_inference_pipeline.py
#
# Tujuan:
# Menguji pipeline inference deployment untuk:
# 1. CNN K2
# 2. Attention-BiLSTM K2
#
# Representasi:
# Title + [SEP] + Description
#
# Alur:
# Input mentah
# -> preprocessing
# -> penggabungan Title + [SEP] + Description
# -> TextVectorization dengan vocabulary training K2
# -> sequence length 60
# -> prediksi CNN dan Attention-BiLSTM
# -> probabilitas empat kelas
#
# Output:
# 9_results/tables/deployment/inference_pipeline_test.json
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
from datetime import datetime
from pathlib import Path
from typing import Any

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
# PATH INPUT
# =============================================================================

DEPLOYMENT_DIR = (
    PROJECT_ROOT
    / "8_save_models"
    / "deployment"
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
# PATH OUTPUT
# =============================================================================

OUTPUT_TABLES_DIR = (
    PROJECT_ROOT
    / "9_results"
    / "tables"
    / "deployment"
)

INFERENCE_REPORT_PATH = (
    OUTPUT_TABLES_DIR
    / "inference_pipeline_test.json"
)


# =============================================================================
# KONFIGURASI
# =============================================================================

EXPECTED_DATASET = "Kompas"
EXPECTED_SCENARIO = "K2"

EXPECTED_SEQUENCE_LENGTH = 60
EXPECTED_NUM_CLASSES = 4

EXPECTED_INDEX_TO_LABEL = {
    0: "bola",
    1: "global",
    2: "money",
    3: "tekno",
}

DEFAULT_MODEL_FILENAMES = {
    "cnn": "cnn_k2.keras",
    "attention_bilstm": "attention_bilstm_k2.keras",
}

DEFAULT_SEPARATOR_TOKEN = "[SEP]"

PROBABILITY_ATOL = 1e-4

# Pemanasan model tidak dihitung sebagai waktu inference.
WARMUP_RUNS = 2

# Waktu inference dihitung dari rata-rata beberapa kali pengujian.
TIMING_RUNS = 5


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
    Menyimpan seluruh artefak yang dibutuhkan untuk inference.
    """

    deployment_config: dict[str, Any]
    vectorizer_config: dict[str, Any]
    vocabulary: list[str]
    label_mapping: dict[int, str]
    vectorizer: tf.keras.layers.TextVectorization
    models: dict[str, tf.keras.Model]
    model_display_names: dict[str, str]
    sequence_length: int
    num_classes: int
    separator_token: str
    required_fields: list[str]


# Menyimpan ID model yang sudah melalui warm-up.
WARMED_MODEL_IDS: set[int] = set()


# =============================================================================
# UTILITAS UMUM
# =============================================================================

def print_header(
    title: str,
) -> None:
    """
    Menampilkan header pada terminal.
    """

    print("=" * 80)
    print(title)
    print("=" * 80)


def create_output_directories() -> None:
    """
    Membuat folder output.
    """

    OUTPUT_TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def json_default(
    value: Any,
) -> Any:
    """
    Mengubah tipe NumPy dan Path menjadi tipe JSON.
    """

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, tuple):
        return list(value)

    raise TypeError(
        f"Tipe data tidak dapat disimpan ke JSON: {type(value)}"
    )


def write_json(
    output_path: Path,
    data: dict[str, Any],
) -> None:
    """
    Menyimpan data ke JSON.
    """

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


def load_json(
    file_path: Path,
) -> dict[str, Any]:
    """
    Membaca file JSON.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            "File JSON tidak ditemukan:\n"
            f"{file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            "Path JSON bukan file:\n"
            f"{file_path}"
        )

    if file_path.stat().st_size <= 0:
        raise ValueError(
            "File JSON kosong:\n"
            f"{file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(
            file
        )

    if not isinstance(data, dict):
        raise ValueError(
            "Isi JSON harus berupa dictionary:\n"
            f"{file_path}"
        )

    return data


def sha256_file(
    file_path: Path,
) -> str:
    """
    Menghitung hash SHA-256 file.
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


def normalize_shape(
    shape: Any,
    shape_name: str,
) -> tuple[Any, ...]:
    """
    Menormalisasi input atau output shape model.
    """

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


def find_recursive_values(
    data: Any,
    accepted_keys: set[str],
) -> list[Any]:
    """
    Mencari nilai berdasarkan nama key secara rekursif.
    """

    results: list[Any] = []

    if isinstance(data, dict):
        for key, value in data.items():
            normalized_key = (
                str(key)
                .strip()
                .lower()
            )

            if normalized_key in accepted_keys:
                results.append(
                    value
                )

            results.extend(
                find_recursive_values(
                    value,
                    accepted_keys,
                )
            )

    elif isinstance(data, list):
        for value in data:
            results.extend(
                find_recursive_values(
                    value,
                    accepted_keys,
                )
            )

    return results


# =============================================================================
# PREPROCESSING TEKS
# =============================================================================

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE,
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
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


def preprocess_text(
    text: Any,
) -> str:
    """
    Membersihkan teks untuk inference.

    Tahapan:
    1. Mengubah input menjadi string.
    2. HTML entity decoding.
    3. Unicode normalization NFKC.
    4. Normalisasi apostrophe.
    5. Menghapus HTML tag.
    6. Menghapus URL.
    7. Menghapus alamat email.
    8. Case folding.
    9. Menghapus karakter kontrol.
    10. Mempertahankan huruf Unicode, angka, apostrophe, dan spasi.
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
        unicode_category = unicodedata.category(
            character
        )

        if unicode_category.startswith("C"):
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
    separator_token: str,
    required_fields: list[str],
) -> dict[str, str]:
    """
    Membentuk representasi K2:
    Title + [SEP] + Description.
    """

    clean_title = preprocess_text(
        title
    )

    clean_description = preprocess_text(
        description
    )

    normalized_required_fields = {
        str(field)
        .strip()
        .lower()

        for field
        in required_fields
    }

    if (
        "title" in normalized_required_fields
        and not clean_title
    ):
        raise ValueError(
            "Title wajib diisi dan tidak boleh kosong "
            "setelah preprocessing."
        )

    if (
        "description" in normalized_required_fields
        and not clean_description
    ):
        raise ValueError(
            "Description wajib diisi dan tidak boleh kosong "
            "setelah preprocessing."
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
# DEPLOYMENT CONFIG
# =============================================================================

def load_deployment_config() -> dict[str, Any]:
    """
    Membaca dan memvalidasi deployment_config.json.
    """

    configuration = load_json(
        DEPLOYMENT_CONFIG_PATH
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

    if dataset.lower() != EXPECTED_DATASET.lower():
        raise ValueError(
            "Dataset deployment tidak sesuai.\n"
            f"Expected : {EXPECTED_DATASET}\n"
            f"Actual   : {dataset}"
        )

    if scenario_code.upper() != EXPECTED_SCENARIO.upper():
        raise ValueError(
            "Skenario deployment tidak sesuai.\n"
            f"Expected : {EXPECTED_SCENARIO}\n"
            f"Actual   : {scenario_code}"
        )

    if sequence_length != EXPECTED_SEQUENCE_LENGTH:
        raise ValueError(
            "Sequence length deployment tidak sesuai.\n"
            f"Expected : {EXPECTED_SEQUENCE_LENGTH}\n"
            f"Actual   : {sequence_length}"
        )

    if num_classes != EXPECTED_NUM_CLASSES:
        raise ValueError(
            "Jumlah kelas deployment tidak sesuai.\n"
            f"Expected : {EXPECTED_NUM_CLASSES}\n"
            f"Actual   : {num_classes}"
        )

    return configuration


def load_vectorizer_configuration() -> dict[str, Any]:
    """
    Membaca vectorizer_config.json.
    """

    return load_json(
        VECTORIZER_CONFIG_PATH
    )


def validate_vectorizer_configuration(
    configuration: dict[str, Any],
    sequence_length: int,
    vocabulary_size: int,
) -> None:
    """
    Memvalidasi konfigurasi vectorizer.
    """

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

    if sequence_length not in parsed_sequence_values:
        raise ValueError(
            "Sequence length vectorizer tidak sesuai.\n"
            f"Expected  : {sequence_length}\n"
            f"Ditemukan : {parsed_sequence_values}"
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
        except (
            TypeError,
            ValueError,
        ):
            continue

    for config_vocabulary_size in parsed_vocabulary_values:
        if config_vocabulary_size < vocabulary_size:
            raise ValueError(
                "Vocabulary size pada vectorizer config "
                "lebih kecil dari vocabulary aktual.\n"
                f"Vocabulary aktual : {vocabulary_size}\n"
                f"Nilai config      : {config_vocabulary_size}"
            )


# =============================================================================
# VOCABULARY
# =============================================================================

def load_vocabulary() -> list[str]:
    """
    Membaca vocabulary hasil training K2.
    """

    if not VOCABULARY_PATH.exists():
        raise FileNotFoundError(
            "Vocabulary deployment tidak ditemukan:\n"
            f"{VOCABULARY_PATH}"
        )

    vocabulary = (
        VOCABULARY_PATH
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

    if len(vocabulary) <= 2:
        raise ValueError(
            "Vocabulary deployment tidak valid.\n"
            f"Vocabulary size: {len(vocabulary)}"
        )

    if vocabulary[0] != "":
        raise ValueError(
            "Index 0 vocabulary harus merupakan token padding kosong.\n"
            f"Token index 0: {repr(vocabulary[0])}"
        )

    if vocabulary[1] != "[UNK]":
        raise ValueError(
            "Index 1 vocabulary harus merupakan token [UNK].\n"
            f"Token index 1: {repr(vocabulary[1])}"
        )

    return vocabulary


def build_text_vectorizer(
    vocabulary: list[str],
    sequence_length: int,
) -> tf.keras.layers.TextVectorization:
    """
    Membangun TextVectorization dengan vocabulary training.

    Dua token pertama tidak diberikan kepada constructor karena
    TextVectorization menambahkan token padding dan [UNK] secara otomatis.
    """

    vectorizer = tf.keras.layers.TextVectorization(
        standardize=None,
        split="whitespace",
        output_mode="int",
        output_sequence_length=sequence_length,
        vocabulary=vocabulary[2:],
        name="deployment_text_vectorizer",
    )

    reconstructed_vocabulary = (
        vectorizer.get_vocabulary()
    )

    if reconstructed_vocabulary != vocabulary:
        mismatch_positions: list[int] = []

        maximum_length = min(
            len(reconstructed_vocabulary),
            len(vocabulary),
        )

        for index in range(
            maximum_length
        ):
            if (
                reconstructed_vocabulary[index]
                != vocabulary[index]
            ):
                mismatch_positions.append(
                    index
                )

                if len(mismatch_positions) >= 10:
                    break

        raise ValueError(
            "Vocabulary inference tidak identik "
            "dengan vocabulary training.\n"
            f"Training size      : {len(vocabulary)}\n"
            f"Inference size     : {len(reconstructed_vocabulary)}\n"
            f"Posisi tidak sama  : {mismatch_positions}"
        )

    if (
        vectorizer.vocabulary_size()
        != len(vocabulary)
    ):
        raise ValueError(
            "Vocabulary size vectorizer tidak sesuai.\n"
            f"Expected : {len(vocabulary)}\n"
            f"Actual   : {vectorizer.vocabulary_size()}"
        )

    return vectorizer


def vectorize_text(
    text: str,
    vectorizer: tf.keras.layers.TextVectorization,
    sequence_length: int,
    vocabulary_size: int,
) -> dict[str, Any]:
    """
    Mengubah teks menjadi integer sequence dan memvalidasi hasilnya.
    """

    sequence_tensor = vectorizer(
        tf.constant(
            [text],
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
            "Shape sequence inference tidak sesuai.\n"
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

    if maximum_token_id >= vocabulary_size:
        raise ValueError(
            "Token ID melebihi vocabulary size.\n"
            f"Maximum token ID : {maximum_token_id}\n"
            f"Vocabulary size  : {vocabulary_size}"
        )

    non_padding_count = int(
        np.count_nonzero(
            sequence
        )
    )

    oov_count = int(
        np.count_nonzero(
            sequence == 1
        )
    )

    return {
        "sequence":
            sequence,

        "shape":
            list(
                sequence.shape
            ),

        "non_padding_tokens":
            non_padding_count,

        "padding_tokens":
            int(
                sequence_length
                - non_padding_count
            ),

        "oov_tokens":
            oov_count,

        "maximum_token_id":
            maximum_token_id,

        "truncated":
            non_padding_count
            == sequence_length,
    }


# =============================================================================
# LABEL MAPPING
# =============================================================================

def extract_index_to_label(
    data: dict[str, Any],
) -> dict[int, str]:
    """
    Membaca index_to_label dari beberapa format JSON.
    """

    mapping: Any = data

    for dataset_key in [
        "Kompas",
        "kompas",
        "KOMPAS",
    ]:
        if dataset_key in data:
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
    deployment_config: dict[str, Any],
) -> dict[int, str]:
    """
    Membaca dan memvalidasi label mapping deployment.
    """

    label_data = load_json(
        LABEL_MAPPING_PATH
    )

    file_mapping = extract_index_to_label(
        label_data
    )

    config_labels = deployment_config.get(
        "labels",
        {},
    )

    config_mapping = extract_index_to_label(
        config_labels
    )

    if file_mapping != config_mapping:
        raise ValueError(
            "Label mapping file dan deployment config berbeda.\n"
            f"File   : {file_mapping}\n"
            f"Config : {config_mapping}"
        )

    if file_mapping != EXPECTED_INDEX_TO_LABEL:
        raise ValueError(
            "Label mapping tidak sesuai dengan penelitian.\n"
            f"Expected : {EXPECTED_INDEX_TO_LABEL}\n"
            f"Actual   : {file_mapping}"
        )

    return file_mapping


# =============================================================================
# MODEL PATH
# =============================================================================

def resolve_model_information(
    deployment_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Mengambil nama file dan metadata model dari deployment config.
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

        filename = str(
            model_config.get(
                "filename",
                DEFAULT_MODEL_FILENAMES[
                    model_key
                ],
            )
        ).strip()

        if not filename:
            filename = DEFAULT_MODEL_FILENAMES[
                model_key
            ]

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
                        (
                            "CNN"
                            if model_key == "cnn"
                            else "Attention-BiLSTM"
                        ),
                    )
                ),

            "experiment_name":
                str(
                    model_config.get(
                        "experiment_name",
                        (
                            "cnn_k2"
                            if model_key == "cnn"
                            else "attention_bilstm_k2"
                        ),
                    )
                ),

            "expected_sha256":
                model_config.get(
                    "sha256"
                ),
        }

    return result


# =============================================================================
# MODEL VALIDATION
# =============================================================================

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
    Memvalidasi arsitektur model inference.
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

    if input_shape != expected_input_shape:
        raise ValueError(
            f"Input shape {model_key} tidak sesuai.\n"
            f"Expected : {expected_input_shape}\n"
            f"Actual   : {input_shape}"
        )

    if output_shape != expected_output_shape:
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

    if embedding_input_dim != vocabulary_size:
        raise ValueError(
            f"Embedding input_dim {model_key} "
            "tidak sesuai vocabulary size.\n"
            f"Embedding input_dim : {embedding_input_dim}\n"
            f"Vocabulary size     : {vocabulary_size}"
        )

    layer_types = {
        type(layer).__name__
        for layer
        in model.layers
    }

    if model_key == "cnn":
        required_layers = {
            "ZeroPaddingEmbeddingOutput",
            "MaskedGlobalMaxPooling1D",
        }

    elif model_key == "attention_bilstm":
        required_layers = {
            "AttentionPooling1D",
        }

    else:
        required_layers = set()

    missing_layers = (
        required_layers
        - layer_types
    )

    if missing_layers:
        raise ValueError(
            f"Custom layer {model_key} tidak lengkap.\n"
            f"Layer hilang: {sorted(missing_layers)}"
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

        "embedding_layer":
            embedding_layer.name,

        "embedding_input_dim":
            embedding_input_dim,

        "embedding_output_dim":
            int(
                embedding_layer.output_dim
            ),

        "layer_types":
            sorted(
                layer_types
            ),
    }


def load_models(
    deployment_config: dict[str, Any],
    sequence_length: int,
    num_classes: int,
    vocabulary_size: int,
) -> tuple[
    dict[str, tf.keras.Model],
    dict[str, str],
    dict[str, Any],
]:
    """
    Memuat dan memvalidasi CNN serta Attention-BiLSTM.
    """

    model_information = resolve_model_information(
        deployment_config
    )

    models: dict[
        str,
        tf.keras.Model
    ] = {}

    display_names: dict[
        str,
        str
    ] = {}

    metadata: dict[
        str,
        Any
    ] = {}

    for model_key, information in (
        model_information.items()
    ):
        model_path: Path = information[
            "path"
        ]

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model {model_key} tidak ditemukan:\n"
                f"{model_path}"
            )

        if model_path.stat().st_size <= 0:
            raise ValueError(
                f"File model {model_key} kosong:\n"
                f"{model_path}"
            )

        actual_sha256 = sha256_file(
            model_path
        )

        expected_sha256 = information.get(
            "expected_sha256"
        )

        if (
            expected_sha256
            and str(expected_sha256).strip()
            and actual_sha256
            != str(expected_sha256).strip()
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

        architecture = validate_model(
            model=model,
            model_key=model_key,
            sequence_length=sequence_length,
            num_classes=num_classes,
            vocabulary_size=vocabulary_size,
        )

        models[
            model_key
        ] = model

        display_names[
            model_key
        ] = information[
            "display_name"
        ]

        metadata[
            model_key
        ] = {
            "path":
                str(
                    model_path
                ),

            "filename":
                model_path.name,

            "experiment_name":
                information[
                    "experiment_name"
                ],

            "display_name":
                information[
                    "display_name"
                ],

            "size_bytes":
                int(
                    model_path.stat().st_size
                ),

            "sha256":
                actual_sha256,

            "architecture":
                architecture,
        }

    return (
        models,
        display_names,
        metadata,
    )


# =============================================================================
# RUNTIME LOADER
# =============================================================================

def load_inference_runtime() -> tuple[
    InferenceRuntime,
    dict[str, Any],
]:
    """
    Memuat seluruh artefak inference.
    """

    deployment_config = load_deployment_config()

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

    if len(vocabulary) != configured_vocabulary_size:
        raise ValueError(
            "Vocabulary size deployment config "
            "berbeda dengan file vocabulary.\n"
            f"Config : {configured_vocabulary_size}\n"
            f"File   : {len(vocabulary)}"
        )

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
            f"Separator: {separator_token}"
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
            "required_fields pada deployment config harus berupa list."
        )

    required_fields = [
        str(field)
        .strip()
        .lower()

        for field
        in required_fields
    ]

    vectorizer_config = load_vectorizer_configuration()

    validate_vectorizer_configuration(
        configuration=vectorizer_config,
        sequence_length=sequence_length,
        vocabulary_size=len(vocabulary),
    )

    vectorizer = build_text_vectorizer(
        vocabulary=vocabulary,
        sequence_length=sequence_length,
    )

    label_mapping = load_label_mapping(
        deployment_config
    )

    (
        models,
        model_display_names,
        model_metadata,
    ) = load_models(
        deployment_config=deployment_config,
        sequence_length=sequence_length,
        num_classes=num_classes,
        vocabulary_size=len(vocabulary),
    )

    runtime = InferenceRuntime(
        deployment_config=deployment_config,
        vectorizer_config=vectorizer_config,
        vocabulary=vocabulary,
        label_mapping=label_mapping,
        vectorizer=vectorizer,
        models=models,
        model_display_names=model_display_names,
        sequence_length=sequence_length,
        num_classes=num_classes,
        separator_token=separator_token,
        required_fields=required_fields,
    )

    metadata = {
        "deployment_config_path":
            str(
                DEPLOYMENT_CONFIG_PATH
            ),

        "vectorizer_config_path":
            str(
                VECTORIZER_CONFIG_PATH
            ),

        "vocabulary_path":
            str(
                VOCABULARY_PATH
            ),

        "label_mapping_path":
            str(
                LABEL_MAPPING_PATH
            ),

        "dataset":
            deployment_config[
                "dataset"
            ],

        "scenario_code":
            deployment_config[
                "scenario_code"
            ],

        "scenario_name":
            deployment_config.get(
                "scenario_name"
            ),

        "sequence_length":
            sequence_length,

        "num_classes":
            num_classes,

        "vocabulary_size":
            len(
                vocabulary
            ),

        "separator_token":
            separator_token,

        "required_fields":
            required_fields,

        "models":
            model_metadata,
    }

    return (
        runtime,
        metadata,
    )


# =============================================================================
# PREDICTION
# =============================================================================

def validate_probabilities(
    probabilities: np.ndarray,
    num_classes: int,
    context: str,
) -> np.ndarray:
    """
    Memvalidasi probabilitas keluaran model.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    ).reshape(
        -1
    )

    expected_shape = (
        num_classes,
    )

    if probabilities.shape != expected_shape:
        raise ValueError(
            f"Shape probabilitas {context} tidak sesuai.\n"
            f"Expected : {expected_shape}\n"
            f"Actual   : {probabilities.shape}"
        )

    if not np.all(
        np.isfinite(
            probabilities
        )
    ):
        raise ValueError(
            f"Probabilitas {context} mengandung NaN atau infinity."
        )

    if (
        np.any(
            probabilities < -1e-7
        )
        or np.any(
            probabilities > 1.0 + 1e-7
        )
    ):
        raise ValueError(
            f"Probabilitas {context} berada di luar rentang 0–1."
        )

    probability_sum = float(
        np.sum(
            probabilities
        )
    )

    if not np.isclose(
        probability_sum,
        1.0,
        atol=PROBABILITY_ATOL,
    ):
        raise ValueError(
            f"Jumlah probabilitas {context} tidak mendekati 1.\n"
            f"Jumlah: {probability_sum}"
        )

    return probabilities


def warm_up_model(
    model: tf.keras.Model,
    sequence: np.ndarray,
) -> None:
    """
    Melakukan warm-up agar waktu tracing tidak masuk perhitungan.
    """

    model_identifier = id(
        model
    )

    if model_identifier in WARMED_MODEL_IDS:
        return

    for _ in range(
        WARMUP_RUNS
    ):
        output = model(
            sequence,
            training=False,
        )

        _ = np.asarray(
            output
        )

    WARMED_MODEL_IDS.add(
        model_identifier
    )


def predict_single_model(
    model: tf.keras.Model,
    sequence: np.ndarray,
    label_mapping: dict[int, str],
    num_classes: int,
    model_name: str,
) -> dict[str, Any]:
    """
    Menjalankan prediksi satu model.
    """

    warm_up_model(
        model=model,
        sequence=sequence,
    )

    elapsed_times: list[float] = []
    final_probabilities: np.ndarray | None = None

    for _ in range(
        TIMING_RUNS
    ):
        start_time = time.perf_counter()

        output = model(
            sequence,
            training=False,
        )

        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        probabilities = np.asarray(
            output,
            dtype=np.float64,
        )[0]

        probabilities = validate_probabilities(
            probabilities=probabilities,
            num_classes=num_classes,
            context=model_name,
        )

        elapsed_times.append(
            elapsed_seconds
        )

        final_probabilities = probabilities

    if final_probabilities is None:
        raise RuntimeError(
            f"Prediksi {model_name} tidak menghasilkan probabilitas."
        )

    predicted_index = int(
        np.argmax(
            final_probabilities
        )
    )

    if predicted_index not in label_mapping:
        raise KeyError(
            f"Index prediksi {predicted_index} "
            "tidak ditemukan pada label mapping."
        )

    predicted_label = label_mapping[
        predicted_index
    ]

    class_probabilities = {
        label_mapping[index]:
            float(
                final_probabilities[
                    index
                ]
            )

        for index
        in range(
            num_classes
        )
    }

    elapsed_array = np.asarray(
        elapsed_times,
        dtype=np.float64,
    )

    return {
        "predicted_index":
            predicted_index,

        "predicted_label":
            predicted_label,

        "confidence":
            float(
                final_probabilities[
                    predicted_index
                ]
            ),

        "probabilities":
            class_probabilities,

        "inference_timing": {
            "warmup_runs":
                WARMUP_RUNS,

            "timing_runs":
                TIMING_RUNS,

            "mean_seconds":
                float(
                    np.mean(
                        elapsed_array
                    )
                ),

            "median_seconds":
                float(
                    np.median(
                        elapsed_array
                    )
                ),

            "minimum_seconds":
                float(
                    np.min(
                        elapsed_array
                    )
                ),

            "maximum_seconds":
                float(
                    np.max(
                        elapsed_array
                    )
                ),

            "mean_ms":
                float(
                    np.mean(
                        elapsed_array
                    )
                    * 1000
                ),

            "median_ms":
                float(
                    np.median(
                        elapsed_array
                    )
                    * 1000
                ),
        },
    }


def predict_news(
    title: Any,
    description: Any,
    runtime: InferenceRuntime,
) -> dict[str, Any]:
    """
    Menjalankan seluruh pipeline inference.
    """

    prepared_input = build_input_text(
        title=title,
        description=description,
        separator_token=runtime.separator_token,
        required_fields=runtime.required_fields,
    )

    vectorized = vectorize_text(
        text=prepared_input[
            "combined_text"
        ],
        vectorizer=runtime.vectorizer,
        sequence_length=runtime.sequence_length,
        vocabulary_size=len(
            runtime.vocabulary
        ),
    )

    sequence = vectorized[
        "sequence"
    ]

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
            label_mapping=runtime.label_mapping,
            num_classes=runtime.num_classes,
            model_name=runtime.model_display_names[
                model_key
            ],
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

    best_model_config = (
        runtime.deployment_config.get(
            "best_research_model",
            {},
        )
    )

    primary_experiment_name = str(
        best_model_config.get(
            "experiment_name",
            "cnn_k2",
        )
    )

    recommended_model_key = (
        "attention_bilstm"
        if primary_experiment_name
        == "attention_bilstm_k2"
        else "cnn"
    )

    recommended_result = model_results[
        recommended_model_key
    ]

    return {
        "generated_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "input": {
            **prepared_input,

            "sequence_shape":
                vectorized[
                    "shape"
                ],

            "non_padding_tokens":
                vectorized[
                    "non_padding_tokens"
                ],

            "padding_tokens":
                vectorized[
                    "padding_tokens"
                ],

            "oov_tokens":
                vectorized[
                    "oov_tokens"
                ],

            "maximum_token_id":
                vectorized[
                    "maximum_token_id"
                ],

            "possibly_truncated":
                vectorized[
                    "truncated"
                ],

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
                recommended_model_key,

            "source_model":
                runtime.model_display_names[
                    recommended_model_key
                ],

            "experiment_name":
                primary_experiment_name,

            "predicted_index":
                recommended_result[
                    "predicted_index"
                ],

            "predicted_label":
                recommended_result[
                    "predicted_label"
                ],

            "confidence":
                recommended_result[
                    "confidence"
                ],
        },
    }


# =============================================================================
# DISPLAY
# =============================================================================

def print_prediction_result(
    result: dict[str, Any],
    runtime: InferenceRuntime,
) -> None:
    """
    Menampilkan hasil prediksi di terminal.
    """

    print("\n")
    print_header(
        "HASIL INFERENCE"
    )

    print(
        f"\nTeks gabungan           : "
        f"{result['input']['combined_text']}"
    )

    print(
        f"Sequence shape          : "
        f"{result['input']['sequence_shape']}"
    )

    print(
        f"Token non-padding       : "
        f"{result['input']['non_padding_tokens']}"
    )

    print(
        f"Token padding           : "
        f"{result['input']['padding_tokens']}"
    )

    print(
        f"Token OOV               : "
        f"{result['input']['oov_tokens']}"
    )

    print(
        f"Terindikasi terpotong   : "
        f"{result['input']['possibly_truncated']}"
    )

    for model_key in [
        "cnn",
        "attention_bilstm",
    ]:
        model_result = result[
            model_key
        ]

        print(
            "\n" + "-" * 80
        )

        print(
            f"Model                   : "
            f"{runtime.model_display_names[model_key]}"
        )

        print(
            f"Prediksi                : "
            f"{model_result['predicted_label']}"
        )

        print(
            f"Confidence              : "
            f"{model_result['confidence']:.2%}"
        )

        print(
            f"Rata-rata waktu         : "
            f"{model_result['inference_timing']['mean_ms']:.3f} ms"
        )

        print(
            f"Median waktu            : "
            f"{model_result['inference_timing']['median_ms']:.3f} ms"
        )

        print(
            "Probabilitas:"
        )

        sorted_probabilities = sorted(
            model_result[
                "probabilities"
            ].items(),
            key=lambda item: item[1],
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
        f"Kedua model sepakat     : "
        f"{result['model_agreement']}"
    )

    print(
        f"Prediksi rekomendasi    : "
        f"{result['recommended_prediction']['predicted_label']}"
    )

    print(
        f"Model rekomendasi       : "
        f"{result['recommended_prediction']['source_model']}"
    )

    print(
        f"Confidence rekomendasi  : "
        f"{result['recommended_prediction']['confidence']:.2%}"
    )


# =============================================================================
# SAMPLE INPUT
# =============================================================================

def get_sample_inputs() -> list[dict[str, str]]:
    """
    Menyediakan contoh inference untuk empat kelas.
    """

    return [
        {
            "expected_topic":
                "bola",

            "title":
                "Persib Menang di Liga Indonesia",

            "description":
                (
                    "Persib meraih kemenangan dalam pertandingan liga "
                    "setelah mencetak dua gol pada babak kedua."
                ),
        },

        {
            "expected_topic":
                "money",

            "title":
                "Rupiah Menguat terhadap Dolar AS",

            "description":
                (
                    "Nilai tukar rupiah menguat setelah Bank Indonesia "
                    "mengumumkan kebijakan stabilisasi pasar keuangan."
                ),
        },

        {
            "expected_topic":
                "tekno",

            "title":
                "Samsung Meluncurkan Ponsel Galaxy Terbaru",

            "description":
                (
                    "Samsung memperkenalkan perangkat Galaxy dengan "
                    "teknologi kecerdasan buatan dan peningkatan "
                    "kapasitas baterai."
                ),
        },

        {
            "expected_topic":
                "global",

            "title":
                "Iran dan Amerika Serikat Membahas Gencatan Senjata",

            "description":
                (
                    "Perwakilan Iran dan Amerika Serikat membahas "
                    "perkembangan konflik dan rencana gencatan senjata."
                ),
        },
    ]


# =============================================================================
# PROGRAM UTAMA
# =============================================================================

def main() -> None:
    """
    Menguji pipeline inference menggunakan beberapa contoh berita.
    """

    print_header(
        "STEP 8.2 - INFERENCE PIPELINE"
    )

    create_output_directories()

    print(
        "\nMemuat artefak deployment..."
    )

    runtime, runtime_metadata = (
        load_inference_runtime()
    )

    print(
        f"Dataset                 : "
        f"{runtime_metadata['dataset']}"
    )

    print(
        f"Skenario                : "
        f"{runtime_metadata['scenario_code']}"
    )

    print(
        f"Representasi            : "
        f"{runtime_metadata['scenario_name']}"
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
        "\nModel deployment:"
    )

    for model_key in [
        "cnn",
        "attention_bilstm",
    ]:
        model_info = runtime_metadata[
            "models"
        ][
            model_key
        ]

        architecture = model_info[
            "architecture"
        ]

        print(
            f"- {model_info['display_name']}"
        )

        print(
            f"  File                  : "
            f"{model_info['filename']}"
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

    sample_inputs = get_sample_inputs()

    inference_results: list[
        dict[str, Any]
    ] = []

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
            f"\nTopik contoh           : "
            f"{sample['expected_topic']}"
        )

        print(
            f"Title                  : "
            f"{sample['title']}"
        )

        print(
            f"Description            : "
            f"{sample['description']}"
        )

        result = predict_news(
            title=sample[
                "title"
            ],
            description=sample[
                "description"
            ],
            runtime=runtime,
        )

        result[
            "expected_topic"
        ] = sample[
            "expected_topic"
        ]

        result[
            "recommended_matches_expected_topic"
        ] = (
            result[
                "recommended_prediction"
            ][
                "predicted_label"
            ]
            == sample[
                "expected_topic"
            ]
        )

        inference_results.append(
            result
        )

        print_prediction_result(
            result=result,
            runtime=runtime,
        )

    successful_expected_topics = sum(
        bool(
            result[
                "recommended_matches_expected_topic"
            ]
        )
        for result
        in inference_results
    )

    agreement_count = sum(
        bool(
            result[
                "model_agreement"
            ]
        )
        for result
        in inference_results
    )

    report = {
        "generated_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "step":
            "8.2",

        "name":
            "Inference Pipeline",

        "status":
            "success",

        "runtime":
            runtime_metadata,

        "configuration": {
            "warmup_runs":
                WARMUP_RUNS,

            "timing_runs":
                TIMING_RUNS,

            "probability_tolerance":
                PROBABILITY_ATOL,
        },

        "summary": {
            "number_of_samples":
                len(
                    inference_results
                ),

            "recommended_matches_expected_topic":
                successful_expected_topics,

            "model_agreement_count":
                agreement_count,
        },

        "results":
            inference_results,
    }

    write_json(
        INFERENCE_REPORT_PATH,
        report,
    )

    print("\n")
    print_header(
        "HASIL INFERENCE PIPELINE"
    )

    print(
        f"\nJumlah contoh           : "
        f"{len(inference_results)}"
    )

    print(
        f"Sesuai topik contoh     : "
        f"{successful_expected_topics}/"
        f"{len(inference_results)}"
    )

    print(
        f"Kedua model sepakat     : "
        f"{agreement_count}/"
        f"{len(inference_results)}"
    )

    print(
        "\nLaporan inference:"
    )

    print(
        INFERENCE_REPORT_PATH
    )

    print("\n")
    print_header(
        "Inference pipeline berhasil dijalankan."
    )


if __name__ == "__main__":
    main()
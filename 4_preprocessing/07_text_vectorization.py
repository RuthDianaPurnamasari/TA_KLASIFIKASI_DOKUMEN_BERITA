from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# ============================================================
# MENAMBAHKAN ROOT PROJECT KE PYTHON PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config import TABLES_DIR  # noqa: E402


# ============================================================
# FOLDER INPUT DAN OUTPUT
# ============================================================

SPLITS_DIR = (
    PROJECT_ROOT
    / "2_data"
    / "splits"
)

VECTORIZED_DIR = (
    PROJECT_ROOT
    / "2_data"
    / "vectorized"
)


# ============================================================
# INPUT KOMPAS
# ============================================================

KOMPAS_SCENARIO_PATHS = {
    "K1": (
        SPLITS_DIR
        / "kompas_k1_split.csv"
    ),
    "K2": (
        SPLITS_DIR
        / "kompas_k2_split.csv"
    ),
    "K3": (
        SPLITS_DIR
        / "kompas_k3_split.csv"
    ),
    "K4": (
        SPLITS_DIR
        / "kompas_k4_split.csv"
    ),
}


# ============================================================
# INPUT AG NEWS
# ============================================================

AGNEWS_SCENARIO_PATHS = {
    "A1": {
        "train_validation": (
            SPLITS_DIR
            / "agnews_a1_train_validation.csv"
        ),
        "test": (
            SPLITS_DIR
            / "agnews_a1_test.csv"
        ),
    },
    "A2": {
        "train_validation": (
            SPLITS_DIR
            / "agnews_a2_train_validation.csv"
        ),
        "test": (
            SPLITS_DIR
            / "agnews_a2_test.csv"
        ),
    },
}


# ============================================================
# OUTPUT LAPORAN
# ============================================================

VECTORIZATION_REPORT_PATH = (
    TABLES_DIR
    / "text_vectorization_report.csv"
)

VOCABULARY_REPORT_PATH = (
    TABLES_DIR
    / "vocabulary_report.csv"
)

LABEL_MAPPING_PATH = (
    TABLES_DIR
    / "label_mapping.json"
)

VECTORIZATION_CONFIGURATION_PATH = (
    TABLES_DIR
    / "text_vectorization_configuration.json"
)


# ============================================================
# KONFIGURASI
# ============================================================

RANDOM_SEED = 42

BATCH_SIZE = 1024

# Panjang sequence hasil analisis P95.
MAX_SEQUENCE_LENGTHS = {
    "K1": 20,
    "K2": 60, #diubah dari 40 menjadi 60 karena ada beberapa dokumen yang panjangnya 60 token
    "K3": 60,
    "K4": 525,
    "A1": 20,
    "A2": 60,
}

# Batas vocabulary.
# Angka ini termasuk token mask dan OOV.
MAX_VOCABULARY_SIZES = {
    "K1": 30_000,
    "K2": 40_000,
    "K3": 50_000,
    "K4": 80_000,
    "A1": 50_000,
    "A2": 80_000,
}

# Text sudah dibersihkan sebelumnya sehingga tidak perlu
# distandardisasi ulang oleh TextVectorization.
STANDARDIZE_MODE = None

SPLIT_MODE = "whitespace"

OUTPUT_MODE = "int"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_global_seed(
    seed: int,
) -> None:
    """
    Menetapkan random seed Python, NumPy, dan TensorFlow.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        # Tidak semua versi/perangkat mendukung determinism.
        pass


# ============================================================
# INFORMASI PERANGKAT
# ============================================================

def print_tensorflow_environment() -> None:
    """
    Menampilkan versi TensorFlow dan perangkat yang tersedia.
    """

    print("\nInformasi TensorFlow:")
    print(f"TensorFlow version : {tf.__version__}")

    gpu_devices = tf.config.list_physical_devices(
        "GPU"
    )

    if gpu_devices:
        print(
            f"GPU terdeteksi     : "
            f"{len(gpu_devices)}"
        )

        for device in gpu_devices:
            print(f"- {device.name}")
    else:
        print("GPU terdeteksi     : Tidak")
        print("Pemrosesan memakai : CPU")


# ============================================================
# MEMBACA DATASET
# ============================================================

def load_split_dataset(
    file_path: Path,
    dataset_name: str,
    scenario_code: str,
) -> pd.DataFrame:
    """
    Membaca dataset hasil split.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"File {dataset_name} {scenario_code} "
            f"tidak ditemukan:\n{file_path}"
        )

    dataframe = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    required_columns = [
        "document_id",
        "category",
        "scenario_code",
        "scenario_name",
        "text",
        "split",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom tidak lengkap pada "
            f"{dataset_name} {scenario_code}: "
            f"{missing_columns}"
        )

    if dataframe.empty:
        raise ValueError(
            f"Dataset {dataset_name} "
            f"{scenario_code} kosong."
        )

    dataframe["text"] = (
        dataframe["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    empty_text = int(
        dataframe["text"]
        .eq("")
        .sum()
    )

    if empty_text > 0:
        raise ValueError(
            f"Ditemukan {empty_text} teks kosong "
            f"pada {dataset_name} {scenario_code}."
        )

    return dataframe


# ============================================================
# MEMUAT AG NEWS TRAIN, VALIDATION, DAN TEST
# ============================================================

def load_agnews_scenario(
    scenario_code: str,
    paths: dict[str, Path],
) -> pd.DataFrame:
    """
    Menggabungkan file train-validation dan test AG News.
    """

    train_validation = load_split_dataset(
        file_path=paths["train_validation"],
        dataset_name="AG News",
        scenario_code=scenario_code,
    )

    test = load_split_dataset(
        file_path=paths["test"],
        dataset_name="AG News",
        scenario_code=scenario_code,
    )

    dataframe = pd.concat(
        [
            train_validation,
            test,
        ],
        ignore_index=True,
    )

    duplicate_ids = int(
        dataframe["document_id"]
        .duplicated()
        .sum()
    )

    if duplicate_ids > 0:
        raise ValueError(
            f"AG News {scenario_code} memiliki "
            f"{duplicate_ids} document_id duplikat."
        )

    return dataframe


# ============================================================
# LABEL MAPPING
# ============================================================

def create_label_mapping(
    categories: pd.Series,
) -> tuple[dict[str, int], dict[int, str]]:
    """
    Membuat mapping kategori ke angka dan sebaliknya.
    """

    unique_categories = sorted(
        categories
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    label_to_index = {
        category: index
        for index, category
        in enumerate(unique_categories)
    }

    index_to_label = {
        index: category
        for category, index
        in label_to_index.items()
    }

    return (
        label_to_index,
        index_to_label,
    )


def encode_labels(
    categories: pd.Series,
    label_to_index: dict[str, int],
) -> np.ndarray:
    """
    Mengubah nama kategori menjadi integer.
    """

    encoded = (
        categories
        .astype(str)
        .map(label_to_index)
    )

    invalid_labels = int(
        encoded.isna().sum()
    )

    if invalid_labels > 0:
        raise ValueError(
            f"Ditemukan {invalid_labels} "
            f"label yang tidak dapat dikodekan."
        )

    return encoded.to_numpy(
        dtype=np.int32
    )


# ============================================================
# MEMBUAT TEXT VECTORIZATION
# ============================================================

def create_vectorizer(
    scenario_code: str,
) -> keras.layers.TextVectorization:
    """
    Membuat TextVectorization untuk satu skenario.
    """

    return keras.layers.TextVectorization(
        max_tokens=(
            MAX_VOCABULARY_SIZES[
                scenario_code
            ]
        ),
        standardize=STANDARDIZE_MODE,
        split=SPLIT_MODE,
        output_mode=OUTPUT_MODE,
        output_sequence_length=(
            MAX_SEQUENCE_LENGTHS[
                scenario_code
            ]
        ),
        pad_to_max_tokens=False,
        name=(
            f"text_vectorization_"
            f"{scenario_code.lower()}"
        ),
    )


# ============================================================
# MEMBUAT DATASET TEKS UNTUK ADAPT
# ============================================================

def create_text_dataset(
    texts: pd.Series,
    batch_size: int = BATCH_SIZE,
) -> tf.data.Dataset:
    """
    Membuat tf.data.Dataset berisi teks.
    """

    text_array = (
        texts
        .fillna("")
        .astype(str)
        .to_numpy()
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        text_array
    )

    dataset = dataset.batch(
        batch_size
    )

    return dataset


# ============================================================
# FIT VOCABULARY HANYA PADA TRAIN
# ============================================================

def adapt_vectorizer_on_train(
    vectorizer: keras.layers.TextVectorization,
    train_texts: pd.Series,
) -> None:
    """
    Membentuk vocabulary hanya dari teks train.
    """

    train_dataset = create_text_dataset(
        train_texts
    )

    vectorizer.adapt(
        train_dataset
    )


# ============================================================
# MENGUBAH TEKS MENJADI ARRAY INTEGER
# ============================================================

def vectorize_texts(
    vectorizer: keras.layers.TextVectorization,
    texts: pd.Series,
) -> np.ndarray:
    """
    Mengubah seluruh teks menjadi sequence integer.

    Proses dilakukan per batch agar penggunaan RAM lebih aman.
    """

    text_dataset = create_text_dataset(
        texts
    )

    vectorized_batches: list[np.ndarray] = []

    for text_batch in text_dataset:
        vectorized_batch = vectorizer(
            text_batch
        )

        vectorized_batches.append(
            vectorized_batch.numpy()
        )

    if not vectorized_batches:
        return np.empty(
            (
                0,
                vectorizer.output_sequence_length,
            ),
            dtype=np.int32,
        )

    return np.concatenate(
        vectorized_batches,
        axis=0,
    ).astype(np.int32)


# ============================================================
# MENGHITUNG STATISTIK ARRAY
# ============================================================

def calculate_array_statistics(
    vectorized_array: np.ndarray,
) -> dict[str, float | int]:
    """
    Menghitung jumlah token non-padding dan tingkat padding.
    """

    if vectorized_array.size == 0:
        return {
            "jumlah_data": 0,
            "mean_non_padding_tokens": 0.0,
            "median_non_padding_tokens": 0.0,
            "maximum_non_padding_tokens": 0,
            "padding_percentage": 0.0,
        }

    non_padding_lengths = np.count_nonzero(
        vectorized_array,
        axis=1,
    )

    total_positions = (
        vectorized_array.shape[0]
        * vectorized_array.shape[1]
    )

    padding_positions = int(
        np.count_nonzero(
            vectorized_array == 0
        )
    )

    padding_percentage = (
        padding_positions
        / total_positions
        * 100
    )

    return {
        "jumlah_data": int(
            vectorized_array.shape[0]
        ),
        "mean_non_padding_tokens": round(
            float(
                non_padding_lengths.mean()
            ),
            2,
        ),
        "median_non_padding_tokens": round(
            float(
                np.median(
                    non_padding_lengths
                )
            ),
            2,
        ),
        "maximum_non_padding_tokens": int(
            non_padding_lengths.max()
        ),
        "padding_percentage": round(
            float(padding_percentage),
            2,
        ),
    }


# ============================================================
# MENYIMPAN ARRAY SPLIT
# ============================================================

# 

def save_split_arrays(
    output_directory: Path,
    split_name: str,
    source_dataframe: pd.DataFrame,
    vectorized_text: np.ndarray,
    encoded_labels: np.ndarray,
) -> Path:
    """
    Menyimpan X, y, document_id, dan category
    dalam format NPZ.

    Metadata string disimpan sebagai Unicode array,
    bukan object array, sehingga dapat dibaca kembali
    menggunakan allow_pickle=False.
    """

    output_path = (
        output_directory
        / f"{split_name}.npz"
    )

    # Document ID disimpan sebagai Unicode string array.
    document_ids = np.asarray(
        source_dataframe[
            "document_id"
        ]
        .astype(str)
        .tolist(),
        dtype=np.str_,
    )

    # Nama kategori disimpan sebagai Unicode string array.
    categories = np.asarray(
        source_dataframe[
            "category"
        ]
        .astype(str)
        .tolist(),
        dtype=np.str_,
    )

    np.savez_compressed(
        output_path,
        X=vectorized_text.astype(
            np.int32
        ),
        y=encoded_labels.astype(
            np.int32
        ),
        document_id=document_ids,
        category=categories,
    )

    return output_path

# ============================================================
# MENYIMPAN VOCABULARY
# ============================================================

def save_vocabulary(
    vectorizer: keras.layers.TextVectorization,
    output_directory: Path,
) -> tuple[Path, list[str]]:
    """
    Menyimpan vocabulary agar dapat digunakan kembali.
    """

    vocabulary = vectorizer.get_vocabulary()

    vocabulary_path = (
        output_directory
        / "vocabulary.txt"
    )

    with open(
        vocabulary_path,
        "w",
        encoding="utf-8",
    ) as file:
        for token in vocabulary:
            file.write(
                token.replace(
                    "\n",
                    " ",
                )
                + "\n"
            )

    return (
        vocabulary_path,
        vocabulary,
    )


# ============================================================
# MENYIMPAN KONFIGURASI VECTORIZER
# ============================================================

def save_vectorizer_config(
    output_directory: Path,
    dataset_name: str,
    scenario_code: str,
    scenario_name: str,
    vocabulary_size: int,
    label_to_index: dict[str, int],
    index_to_label: dict[int, str],
) -> Path:
    """
    Menyimpan metadata vectorizer satu skenario.
    """

    config_path = (
        output_directory
        / "vectorizer_config.json"
    )

    configuration = {
        "framework": "TensorFlow/Keras",
        "layer": "TextVectorization",
        "dataset": dataset_name,
        "scenario_code": scenario_code,
        "scenario_name": scenario_name,
        "adapt_split": "train",
        "standardize": STANDARDIZE_MODE,
        "split": SPLIT_MODE,
        "output_mode": OUTPUT_MODE,
        "max_tokens": (
            MAX_VOCABULARY_SIZES[
                scenario_code
            ]
        ),
        "actual_vocabulary_size":
            vocabulary_size,
        "output_sequence_length": (
            MAX_SEQUENCE_LENGTHS[
                scenario_code
            ]
        ),
        "padding_value": 0,
        "oov_index": 1,
        "separator_token": "[SEP]",
        "label_to_index":
            label_to_index,
        "index_to_label": {
            str(key): value
            for key, value
            in index_to_label.items()
        },
        "random_seed":
            RANDOM_SEED,
    }

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            ensure_ascii=False,
            indent=4,
        )

    return config_path


# ============================================================
# VALIDASI HASIL VECTORIZATION
# ============================================================

def validate_vectorized_split(
    vectorized_array: np.ndarray,
    labels: np.ndarray,
    expected_rows: int,
    expected_sequence_length: int,
    dataset_name: str,
    scenario_code: str,
    split_name: str,
) -> None:
    """
    Memastikan bentuk array dan label sesuai.
    """

    expected_shape = (
        expected_rows,
        expected_sequence_length,
    )

    if vectorized_array.shape != expected_shape:
        raise ValueError(
            f"Shape X {dataset_name} "
            f"{scenario_code} {split_name} "
            f"tidak sesuai. "
            f"Diperoleh {vectorized_array.shape}, "
            f"seharusnya {expected_shape}."
        )

    if labels.shape[0] != expected_rows:
        raise ValueError(
            f"Jumlah label {dataset_name} "
            f"{scenario_code} {split_name} "
            f"tidak sesuai."
        )

    all_padding_rows = int(
        np.all(
            vectorized_array == 0,
            axis=1,
        ).sum()
    )

    if all_padding_rows > 0:
        raise ValueError(
            f"Ditemukan {all_padding_rows} "
            f"sequence seluruhnya padding pada "
            f"{dataset_name} {scenario_code} "
            f"{split_name}."
        )


# ============================================================
# MEMPROSES SATU SKENARIO
# ============================================================

def process_scenario(
    dataframe: pd.DataFrame,
    dataset_name: str,
    scenario_code: str,
    label_to_index: dict[str, int],
    index_to_label: dict[int, str],
) -> tuple[list[dict], dict]:
    """
    Membuat vocabulary, vectorization, dan file NPZ
    untuk satu skenario.
    """

    scenario_name = (
        dataframe[
            "scenario_name"
        ].iloc[0]
    )

    output_directory = (
        VECTORIZED_DIR
        / (
            f"{dataset_name.lower().replace(' ', '')}_"
            f"{scenario_code.lower()}"
        )
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_data = (
        dataframe[
            dataframe["split"]
            == "train"
        ]
        .copy()
        .reset_index(drop=True)
    )

    validation_data = (
        dataframe[
            dataframe["split"]
            == "validation"
        ]
        .copy()
        .reset_index(drop=True)
    )

    test_data = (
        dataframe[
            dataframe["split"]
            == "test"
        ]
        .copy()
        .reset_index(drop=True)
    )

    if train_data.empty:
        raise ValueError(
            f"Data train {dataset_name} "
            f"{scenario_code} kosong."
        )

    print(
        f"\n{dataset_name} {scenario_code} "
        f"- {scenario_name}"
    )

    print(
        f"Fit vocabulary pada "
        f"{len(train_data):,} data train..."
    )

    vectorizer = create_vectorizer(
        scenario_code
    )

    adapt_vectorizer_on_train(
        vectorizer=vectorizer,
        train_texts=train_data["text"],
    )

    vocabulary_path, vocabulary = (
        save_vocabulary(
            vectorizer=vectorizer,
            output_directory=output_directory,
        )
    )

    vocabulary_size = len(
        vocabulary
    )

    print(
        f"Ukuran vocabulary : "
        f"{vocabulary_size:,}"
    )

    split_dataframes = {
        "train": train_data,
        "validation": validation_data,
        "test": test_data,
    }

    report_rows: list[dict] = []

    split_output_paths: dict[str, str] = {}

    for split_name, split_dataframe in (
        split_dataframes.items()
    ):
        if split_dataframe.empty:
            continue

        print(
            f"Vectorizing {split_name}: "
            f"{len(split_dataframe):,} data..."
        )

        vectorized_text = vectorize_texts(
            vectorizer=vectorizer,
            texts=split_dataframe["text"],
        )

        encoded_labels = encode_labels(
            categories=split_dataframe[
                "category"
            ],
            label_to_index=label_to_index,
        )

        validate_vectorized_split(
            vectorized_array=vectorized_text,
            labels=encoded_labels,
            expected_rows=len(
                split_dataframe
            ),
            expected_sequence_length=(
                MAX_SEQUENCE_LENGTHS[
                    scenario_code
                ]
            ),
            dataset_name=dataset_name,
            scenario_code=scenario_code,
            split_name=split_name,
        )

        output_path = save_split_arrays(
            output_directory=output_directory,
            split_name=split_name,
            source_dataframe=split_dataframe,
            vectorized_text=vectorized_text,
            encoded_labels=encoded_labels,
        )

        split_output_paths[
            split_name
        ] = str(output_path)

        statistics = (
            calculate_array_statistics(
                vectorized_text
            )
        )

        report_rows.append(
            {
                "dataset":
                    dataset_name,
                "scenario_code":
                    scenario_code,
                "scenario_name":
                    scenario_name,
                "split":
                    split_name,
                "jumlah_data":
                    statistics[
                        "jumlah_data"
                    ],
                "max_sequence_length": (
                    MAX_SEQUENCE_LENGTHS[
                        scenario_code
                    ]
                ),
                "vocabulary_size":
                    vocabulary_size,
                "mean_non_padding_tokens":
                    statistics[
                        "mean_non_padding_tokens"
                    ],
                "median_non_padding_tokens":
                    statistics[
                        "median_non_padding_tokens"
                    ],
                "maximum_non_padding_tokens":
                    statistics[
                        "maximum_non_padding_tokens"
                    ],
                "padding_percentage":
                    statistics[
                        "padding_percentage"
                    ],
                "output_path":
                    str(output_path),
            }
        )

    vectorizer_config_path = (
        save_vectorizer_config(
            output_directory=output_directory,
            dataset_name=dataset_name,
            scenario_code=scenario_code,
            scenario_name=scenario_name,
            vocabulary_size=vocabulary_size,
            label_to_index=label_to_index,
            index_to_label=index_to_label,
        )
    )

    vocabulary_report = {
        "dataset": dataset_name,
        "scenario_code": scenario_code,
        "scenario_name": scenario_name,
        "maximum_vocabulary_size": (
            MAX_VOCABULARY_SIZES[
                scenario_code
            ]
        ),
        "actual_vocabulary_size":
            vocabulary_size,
        "vocabulary_path":
            str(vocabulary_path),
        "vectorizer_config_path":
            str(vectorizer_config_path),
        "train_path":
            split_output_paths.get(
                "train",
                "",
            ),
        "validation_path":
            split_output_paths.get(
                "validation",
                "",
            ),
        "test_path":
            split_output_paths.get(
                "test",
                "",
            ),
    }

    return (
        report_rows,
        vocabulary_report,
    )


# ============================================================
# MENYIMPAN LABEL MAPPING
# ============================================================

def save_label_mappings(
    kompas_mapping: tuple[
        dict[str, int],
        dict[int, str],
    ],
    agnews_mapping: tuple[
        dict[str, int],
        dict[int, str],
    ],
) -> None:
    """
    Menyimpan label mapping kedua dataset.
    """

    (
        kompas_label_to_index,
        kompas_index_to_label,
    ) = kompas_mapping

    (
        agnews_label_to_index,
        agnews_index_to_label,
    ) = agnews_mapping

    content = {
        "Kompas": {
            "label_to_index":
                kompas_label_to_index,
            "index_to_label": {
                str(key): value
                for key, value
                in kompas_index_to_label.items()
            },
        },
        "AG News": {
            "label_to_index":
                agnews_label_to_index,
            "index_to_label": {
                str(key): value
                for key, value
                in agnews_index_to_label.items()
            },
        },
    }

    with open(
        LABEL_MAPPING_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            content,
            file,
            ensure_ascii=False,
            indent=4,
        )


# ============================================================
# MENYIMPAN KONFIGURASI GLOBAL
# ============================================================

def save_global_configuration() -> None:
    """
    Menyimpan konfigurasi vectorization seluruh eksperimen.
    """

    configuration = {
        "framework": "TensorFlow/Keras",
        "vectorizer": "TextVectorization",
        "adapt_policy": (
            "Vocabulary dibentuk hanya dari data train "
            "pada masing-masing skenario."
        ),
        "standardize": STANDARDIZE_MODE,
        "split": SPLIT_MODE,
        "output_mode": OUTPUT_MODE,
        "batch_size": BATCH_SIZE,
        "random_seed": RANDOM_SEED,
        "max_sequence_lengths":
            MAX_SEQUENCE_LENGTHS,
        "max_vocabulary_sizes":
            MAX_VOCABULARY_SIZES,
        "special_indices": {
            "padding": 0,
            "oov": 1,
        },
    }

    with open(
        VECTORIZATION_CONFIGURATION_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            ensure_ascii=False,
            indent=4,
        )


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main() -> None:
    """
    Menjalankan text vectorization semua skenario.
    """

    print("=" * 72)
    print("STEP 4.7 - TEXT VECTORIZATION")
    print("=" * 72)

    set_global_seed(
        RANDOM_SEED
    )

    print_tensorflow_environment()

    VECTORIZED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # MEMUAT SEMUA SKENARIO
    # ========================================================

    kompas_dataframes: dict[
        str,
        pd.DataFrame,
    ] = {}

    for scenario_code, file_path in (
        KOMPAS_SCENARIO_PATHS.items()
    ):
        kompas_dataframes[
            scenario_code
        ] = load_split_dataset(
            file_path=file_path,
            dataset_name="Kompas",
            scenario_code=scenario_code,
        )

    agnews_dataframes: dict[
        str,
        pd.DataFrame,
    ] = {}

    for scenario_code, paths in (
        AGNEWS_SCENARIO_PATHS.items()
    ):
        agnews_dataframes[
            scenario_code
        ] = load_agnews_scenario(
            scenario_code=scenario_code,
            paths=paths,
        )

    # ========================================================
    # LABEL MAPPING
    # ========================================================

    kompas_mapping = create_label_mapping(
        kompas_dataframes[
            "K1"
        ]["category"]
    )

    agnews_mapping = create_label_mapping(
        agnews_dataframes[
            "A1"
        ]["category"]
    )

    (
        kompas_label_to_index,
        kompas_index_to_label,
    ) = kompas_mapping

    (
        agnews_label_to_index,
        agnews_index_to_label,
    ) = agnews_mapping

    print("\nLabel mapping Kompas:")
    print(kompas_label_to_index)

    print("\nLabel mapping AG News:")
    print(agnews_label_to_index)

    # ========================================================
    # PROSES KOMPAS
    # ========================================================

    all_report_rows: list[dict] = []
    vocabulary_rows: list[dict] = []

    print("\n" + "=" * 72)
    print("VECTORIZATION KOMPAS")
    print("=" * 72)

    for scenario_code, dataframe in (
        kompas_dataframes.items()
    ):
        (
            report_rows,
            vocabulary_report,
        ) = process_scenario(
            dataframe=dataframe,
            dataset_name="Kompas",
            scenario_code=scenario_code,
            label_to_index=(
                kompas_label_to_index
            ),
            index_to_label=(
                kompas_index_to_label
            ),
        )

        all_report_rows.extend(
            report_rows
        )

        vocabulary_rows.append(
            vocabulary_report
        )

    # ========================================================
    # PROSES AG NEWS
    # ========================================================

    print("\n" + "=" * 72)
    print("VECTORIZATION AG NEWS")
    print("=" * 72)

    for scenario_code, dataframe in (
        agnews_dataframes.items()
    ):
        (
            report_rows,
            vocabulary_report,
        ) = process_scenario(
            dataframe=dataframe,
            dataset_name="AG News",
            scenario_code=scenario_code,
            label_to_index=(
                agnews_label_to_index
            ),
            index_to_label=(
                agnews_index_to_label
            ),
        )

        all_report_rows.extend(
            report_rows
        )

        vocabulary_rows.append(
            vocabulary_report
        )

    # ========================================================
    # MENYIMPAN LAPORAN
    # ========================================================

    vectorization_report = pd.DataFrame(
        all_report_rows
    )

    vocabulary_report = pd.DataFrame(
        vocabulary_rows
    )

    vectorization_report.to_csv(
        VECTORIZATION_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    vocabulary_report.to_csv(
        VOCABULARY_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    save_label_mappings(
        kompas_mapping=kompas_mapping,
        agnews_mapping=agnews_mapping,
    )

    save_global_configuration()

    # ========================================================
    # MENAMPILKAN HASIL
    # ========================================================

    print("\n" + "=" * 72)
    print("HASIL TEXT VECTORIZATION")
    print("=" * 72)

    display_columns = [
        "dataset",
        "scenario_code",
        "split",
        "jumlah_data",
        "max_sequence_length",
        "vocabulary_size",
        "mean_non_padding_tokens",
        "padding_percentage",
    ]

    print(
        vectorization_report[
            display_columns
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 72)
    print("OUTPUT TEXT VECTORIZATION")
    print("=" * 72)

    print("\nFolder vectorized:")
    print(VECTORIZED_DIR)

    print("\nLaporan vectorization:")
    print(VECTORIZATION_REPORT_PATH)

    print("\nLaporan vocabulary:")
    print(VOCABULARY_REPORT_PATH)

    print("\nLabel mapping:")
    print(LABEL_MAPPING_PATH)

    print("\nKonfigurasi vectorization:")
    print(
        VECTORIZATION_CONFIGURATION_PATH
    )

    print(
        "\nTahap text vectorization selesai."
    )


if __name__ == "__main__":
    main()
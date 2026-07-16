from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import TABLES_DIR  # noqa: E402

SPLITS_DIR = PROJECT_ROOT / "2_data" / "splits"
VECTORIZED_DIR = PROJECT_ROOT / "2_data" / "vectorized"
SHARED_VOCABULARIES_DIR = VECTORIZED_DIR / "_shared_vocabularies"

KOMPAS_SCENARIO_PATHS = {
    "K1": SPLITS_DIR / "kompas_k1_split.csv",
    "K2": SPLITS_DIR / "kompas_k2_split.csv",
    "K3": SPLITS_DIR / "kompas_k3_split.csv",
}

AGNEWS_SCENARIO_PATHS = {
    "A1": {
        "train_validation": SPLITS_DIR / "agnews_a1_train_validation.csv",
        "test": SPLITS_DIR / "agnews_a1_test.csv",
    },
    "A2": {
        "train_validation": SPLITS_DIR / "agnews_a2_train_validation.csv",
        "test": SPLITS_DIR / "agnews_a2_test.csv",
    },
}

VECTORIZATION_REPORT_PATH = TABLES_DIR / "text_vectorization_report.csv"
VOCABULARY_REPORT_PATH = TABLES_DIR / "vocabulary_report.csv"
LABEL_MAPPING_PATH = TABLES_DIR / "label_mapping.json"
VECTORIZATION_CONFIGURATION_PATH = TABLES_DIR / "text_vectorization_configuration.json"

RANDOM_SEED = 42
BATCH_SIZE = 1024
STANDARDIZE_MODE = None
SPLIT_MODE = "whitespace"
OUTPUT_MODE = "int"
SEPARATOR_TOKEN = "[SEP]"

MAX_SEQUENCE_LENGTHS = {
    "K1": 20,
    "K2": 60,
    "K3": 60,
    "A1": 60,
    "A2": 60,
}

VOCABULARY_GROUPS = {
    "kompas_shared": {
        "dataset": "Kompas",
        "adapt_scenario": "K2",
        "scenario_codes": ["K1", "K2", "K3"],
        "max_tokens": 50_000,
        "reason": (
            "Vocabulary dibentuk dari Kompas K2 train dan digunakan bersama "
            "oleh K1, K2, dan K3."
        ),
    },
    "agnews_shared": {
        "dataset": "AG News",
        "adapt_scenario": "A2",
        "scenario_codes": ["A1", "A2"],
        "max_tokens": 80_000,
        "reason": (
            "Vocabulary dibentuk dari AG News A2 train dan digunakan bersama "
            "oleh A1 dan A2."
        ),
    },
}

SCENARIO_TO_GROUP = {
    scenario_code: group_name
    for group_name, group_config in VOCABULARY_GROUPS.items()
    for scenario_code in group_config["scenario_codes"]
}

EXPECTED_ROWS = {
    "Kompas": 9_997,
    "AG News Train": 119_817,
    "AG News Test": 7_600,
}

EXPECTED_SPLIT_ROWS = {
    "Kompas": {"train": 7_997, "validation": 1_000, "test": 1_000},
    "AG News": {"train": 107_835, "validation": 11_982, "test": 7_600},
}

KOMPAS_LABEL_TO_INDEX = {
    "bola": 0,
    "global": 1,
    "money": 2,
    "tekno": 3,
}

AGNEWS_LABEL_TO_INDEX = {
    "world": 0,
    "sports": 1,
    "business": 2,
    "sci_tech": 3,
}

REQUIRED_COLUMNS = [
    "document_id",
    "category",
    "dataset",
    "scenario_code",
    "scenario_name",
    "text",
    "word_count",
    "split",
    "uses_yake",
    "comparison_group",
]


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def print_tensorflow_environment() -> None:
    print("\nInformasi TensorFlow:")
    print(f"TensorFlow version : {tf.__version__}")
    gpu_devices = tf.config.list_physical_devices("GPU")
    if gpu_devices:
        print(f"GPU terdeteksi     : {len(gpu_devices)}")
        for device in gpu_devices:
            print(f"- {device.name}")
    else:
        print("GPU terdeteksi     : Tidak")
        print("Pemrosesan memakai : CPU")


def load_split_dataset(
    file_path: Path,
    dataset_name: str,
    scenario_code: str,
    expected_rows: int,
    expected_splits: set[str],
) -> pd.DataFrame:
    label = f"{dataset_name} {scenario_code}"
    if not file_path.exists():
        raise FileNotFoundError(f"File {label} tidak ditemukan:\n{file_path}")
    dataframe = pd.read_csv(file_path, encoding="utf-8-sig", keep_default_na=False)
    if dataframe.empty:
        raise ValueError(f"Dataset {label} kosong.")
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in dataframe.columns]
    if missing_columns:
        raise ValueError(
            f"Kolom tidak lengkap pada {label}: {missing_columns}\n"
            f"Kolom tersedia: {list(dataframe.columns)}"
        )
    if len(dataframe) != expected_rows:
        raise ValueError(
            f"Jumlah data {label} tidak sesuai. "
            f"Seharusnya {expected_rows:,}, ditemukan {len(dataframe):,}."
        )
    actual_codes = set(dataframe["scenario_code"].astype(str).str.strip().unique())
    if actual_codes != {scenario_code}:
        raise ValueError(f"Scenario code {label} tidak sesuai: {actual_codes}")
    actual_splits = set(dataframe["split"].astype(str).str.strip().unique())
    if actual_splits != expected_splits:
        raise ValueError(
            f"Split {label} tidak sesuai. "
            f"Seharusnya {expected_splits}, ditemukan {actual_splits}."
        )
    dataframe["text"] = dataframe["text"].fillna("").astype(str).str.strip()
    empty_text = int(dataframe["text"].eq("").sum())
    if empty_text:
        raise ValueError(f"Ditemukan {empty_text:,} teks kosong pada {label}.")
    duplicate_ids = int(dataframe["document_id"].duplicated().sum())
    if duplicate_ids:
        raise ValueError(f"Ditemukan {duplicate_ids:,} document_id duplikat pada {label}.")
    return dataframe.reset_index(drop=True)


def load_agnews_scenario(scenario_code: str, paths: dict[str, Path]) -> pd.DataFrame:
    train_validation = load_split_dataset(
        paths["train_validation"],
        "AG News Train",
        scenario_code,
        EXPECTED_ROWS["AG News Train"],
        {"train", "validation"},
    )
    test = load_split_dataset(
        paths["test"],
        "AG News Test",
        scenario_code,
        EXPECTED_ROWS["AG News Test"],
        {"test"},
    )
    dataframe = pd.concat([train_validation, test], ignore_index=True)
    duplicate_ids = int(dataframe["document_id"].duplicated().sum())
    if duplicate_ids:
        raise ValueError(
            f"AG News {scenario_code} memiliki {duplicate_ids:,} document_id duplikat."
        )
    return dataframe


def validate_scenario_alignment(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    comparison_name: str,
) -> None:
    columns = ["document_id", "category", "split"]
    if "class_index" in reference.columns and "class_index" in comparison.columns:
        columns.append("class_index")
    left = (
        reference[columns]
        .sort_values("document_id", kind="stable")
        .reset_index(drop=True)
        .astype(str)
    )
    right = (
        comparison[columns]
        .sort_values("document_id", kind="stable")
        .reset_index(drop=True)
        .astype(str)
    )
    if not left.equals(right):
        raise ValueError(
            f"Dokumen, label, atau split pada {comparison_name} tidak selaras."
        )


def validate_split_counts(dataframe: pd.DataFrame, dataset_name: str) -> None:
    expected = EXPECTED_SPLIT_ROWS[dataset_name]
    actual = dataframe["split"].value_counts().to_dict()
    if actual != expected:
        raise ValueError(
            f"Jumlah split {dataset_name} tidak sesuai. "
            f"Seharusnya {expected}, ditemukan {actual}."
        )


def create_index_to_label(label_to_index: dict[str, int]) -> dict[int, str]:
    return {index: label for label, index in label_to_index.items()}


def validate_label_mapping(
    dataframe: pd.DataFrame,
    label_to_index: dict[str, int],
    dataset_name: str,
) -> None:
    actual_categories = set(dataframe["category"].astype(str).unique())
    expected_categories = set(label_to_index)
    if actual_categories != expected_categories:
        raise ValueError(
            f"Kategori {dataset_name} tidak sesuai. "
            f"Seharusnya {expected_categories}, ditemukan {actual_categories}."
        )


def encode_labels(categories: pd.Series, label_to_index: dict[str, int]) -> np.ndarray:
    encoded = categories.astype(str).map(label_to_index)
    invalid_labels = int(encoded.isna().sum())
    if invalid_labels:
        raise ValueError(f"Ditemukan {invalid_labels:,} label yang tidak dapat dikodekan.")
    return encoded.to_numpy(dtype=np.int32)


def create_text_dataset(texts: pd.Series, batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    text_array = texts.fillna("").astype(str).to_numpy(dtype=np.str_)
    return tf.data.Dataset.from_tensor_slices(text_array).batch(batch_size)


def create_adapt_vectorizer(max_tokens: int, layer_name: str) -> keras.layers.TextVectorization:
    return keras.layers.TextVectorization(
        max_tokens=max_tokens,
        standardize=STANDARDIZE_MODE,
        split=SPLIT_MODE,
        output_mode=OUTPUT_MODE,
        name=layer_name,
    )


def build_shared_vocabulary(
    train_texts: pd.Series,
    max_tokens: int,
    group_name: str,
) -> tuple[list[str], str]:
    vectorizer = create_adapt_vectorizer(
        max_tokens=max_tokens,
        layer_name=f"adapt_{group_name}",
    )
    vectorizer.adapt(create_text_dataset(train_texts))
    vocabulary = vectorizer.get_vocabulary()
    if len(vocabulary) < 3:
        raise ValueError(f"Vocabulary {group_name} terlalu kecil.")
    if vocabulary[0] != "" or vocabulary[1] != "[UNK]":
        raise ValueError(
            f"Indeks khusus vocabulary {group_name} tidak sesuai: {vocabulary[:2]}"
        )
    if SEPARATOR_TOKEN not in vocabulary:
        raise ValueError(
            f"Token separator {SEPARATOR_TOKEN} tidak ditemukan pada vocabulary {group_name}."
        )
    digest = hashlib.sha256("\n".join(vocabulary).encode("utf-8")).hexdigest()
    return vocabulary, digest


def create_vectorizer_from_vocabulary(
    scenario_code: str,
    shared_vocabulary: list[str],
    max_tokens: int,
) -> keras.layers.TextVectorization:
    vectorizer = keras.layers.TextVectorization(
        max_tokens=max_tokens,
        standardize=STANDARDIZE_MODE,
        split=SPLIT_MODE,
        output_mode=OUTPUT_MODE,
        output_sequence_length=MAX_SEQUENCE_LENGTHS[scenario_code],
        vocabulary=shared_vocabulary[2:],
        name=f"text_vectorization_{scenario_code.lower()}",
    )
    if vectorizer.get_vocabulary() != shared_vocabulary:
        raise ValueError(
            f"Vocabulary hasil konstruksi {scenario_code} tidak identik dengan vocabulary bersama."
        )
    return vectorizer


def save_vocabulary_file(vocabulary: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        for token in vocabulary:
            file.write(token.replace("\n", " ") + "\n")


def vectorize_texts(
    vectorizer: keras.layers.TextVectorization,
    texts: pd.Series,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    for text_batch in create_text_dataset(texts):
        batches.append(vectorizer(text_batch).numpy())
    if not batches:
        return np.empty((0, 0), dtype=np.int32)
    return np.concatenate(batches, axis=0).astype(np.int32)


def count_whitespace_tokens(value: Any) -> int:
    text = " ".join(str(value).strip().split())
    return len(text.split()) if text else 0


def calculate_array_statistics(
    vectorized_array: np.ndarray,
    source_texts: pd.Series,
    max_sequence_length: int,
) -> dict[str, float | int]:
    non_padding_mask = vectorized_array != 0
    non_padding_lengths = np.count_nonzero(non_padding_mask, axis=1)
    total_positions = int(vectorized_array.size)
    padding_positions = int(np.count_nonzero(vectorized_array == 0))
    oov_positions = int(np.count_nonzero(vectorized_array == 1))
    non_padding_positions = int(np.count_nonzero(non_padding_mask))
    source_lengths = source_texts.apply(count_whitespace_tokens).to_numpy(dtype=np.int32)
    truncated_mask = source_lengths > max_sequence_length
    return {
        "jumlah_data": int(vectorized_array.shape[0]),
        "mean_non_padding_tokens": round(float(non_padding_lengths.mean()), 4),
        "median_non_padding_tokens": round(float(np.median(non_padding_lengths)), 4),
        "maximum_non_padding_tokens": int(non_padding_lengths.max()),
        "padding_percentage": round(padding_positions / total_positions * 100, 4),
        "oov_token_count": oov_positions,
        "oov_percentage_non_padding": round(
            oov_positions / non_padding_positions * 100 if non_padding_positions else 0.0,
            6,
        ),
        "truncated_document_count": int(truncated_mask.sum()),
        "truncated_document_percentage": round(float(truncated_mask.mean() * 100), 4),
    }


def save_split_arrays(
    output_directory: Path,
    split_name: str,
    source_dataframe: pd.DataFrame,
    vectorized_text: np.ndarray,
    encoded_labels: np.ndarray,
) -> Path:
    output_path = output_directory / f"{split_name}.npz"
    document_ids = np.asarray(source_dataframe["document_id"].astype(str).tolist(), dtype=np.str_)
    categories = np.asarray(source_dataframe["category"].astype(str).tolist(), dtype=np.str_)
    np.savez_compressed(
        output_path,
        X=vectorized_text.astype(np.int32),
        y=encoded_labels.astype(np.int32),
        document_id=document_ids,
        category=categories,
    )
    return output_path


def validate_vectorized_split(
    vectorized_array: np.ndarray,
    labels: np.ndarray,
    expected_rows: int,
    expected_sequence_length: int,
    vocabulary_size: int,
    number_of_classes: int,
    dataset_name: str,
    scenario_code: str,
    split_name: str,
) -> None:
    expected_shape = (expected_rows, expected_sequence_length)
    if vectorized_array.shape != expected_shape:
        raise ValueError(
            f"Shape X {dataset_name} {scenario_code} {split_name} tidak sesuai. "
            f"Diperoleh {vectorized_array.shape}, seharusnya {expected_shape}."
        )
    if labels.shape != (expected_rows,):
        raise ValueError(
            f"Shape y {dataset_name} {scenario_code} {split_name} tidak sesuai: {labels.shape}."
        )
    if int(np.all(vectorized_array == 0, axis=1).sum()) > 0:
        raise ValueError(
            f"Terdapat sequence seluruhnya padding pada {dataset_name} "
            f"{scenario_code} {split_name}."
        )
    if int(vectorized_array.max()) >= vocabulary_size:
        raise ValueError(
            f"Indeks token melebihi ukuran vocabulary pada {dataset_name} "
            f"{scenario_code} {split_name}."
        )
    if labels.min() < 0 or labels.max() >= number_of_classes:
        raise ValueError(
            f"Indeks label tidak valid pada {dataset_name} {scenario_code} {split_name}."
        )


def save_vectorizer_config(
    output_directory: Path,
    dataset_name: str,
    scenario_code: str,
    scenario_name: str,
    group_name: str,
    group_config: dict,
    vocabulary_size: int,
    vocabulary_sha256: str,
    shared_vocabulary_path: Path,
    label_to_index: dict[str, int],
    index_to_label: dict[int, str],
) -> Path:
    config_path = output_directory / "vectorizer_config.json"
    configuration = {
        "framework": "TensorFlow/Keras",
        "layer": "TextVectorization",
        "dataset": dataset_name,
        "scenario_code": scenario_code,
        "scenario_name": scenario_name,
        "adapt_split": "train",
        "vocabulary_group": group_name,
        "vocabulary_adapt_scenario": group_config["adapt_scenario"],
        "shared_vocabulary_path": str(shared_vocabulary_path),
        "shared_vocabulary_sha256": vocabulary_sha256,
        "standardize": STANDARDIZE_MODE,
        "split": SPLIT_MODE,
        "output_mode": OUTPUT_MODE,
        "max_tokens": int(group_config["max_tokens"]),
        "actual_vocabulary_size": vocabulary_size,
        "output_sequence_length": MAX_SEQUENCE_LENGTHS[scenario_code],
        "padding_value": 0,
        "oov_index": 1,
        "separator_token": SEPARATOR_TOKEN,
        "label_to_index": label_to_index,
        "index_to_label": {str(k): v for k, v in index_to_label.items()},
        "random_seed": RANDOM_SEED,
    }
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(configuration, file, ensure_ascii=False, indent=4)
    return config_path


def process_scenario(
    dataframe: pd.DataFrame,
    dataset_name: str,
    scenario_code: str,
    shared_vocabulary: list[str],
    vocabulary_sha256: str,
    shared_vocabulary_path: Path,
    label_to_index: dict[str, int],
    index_to_label: dict[int, str],
) -> tuple[list[dict], dict]:
    scenario_name = str(dataframe["scenario_name"].iloc[0])
    group_name = SCENARIO_TO_GROUP[scenario_code]
    group_config = VOCABULARY_GROUPS[group_name]
    output_directory = (
        VECTORIZED_DIR
        / f"{dataset_name.lower().replace(' ', '')}_{scenario_code.lower()}"
    )
    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    scenario_vocabulary_path = output_directory / "vocabulary.txt"
    save_vocabulary_file(shared_vocabulary, scenario_vocabulary_path)

    vectorizer = create_vectorizer_from_vocabulary(
        scenario_code=scenario_code,
        shared_vocabulary=shared_vocabulary,
        max_tokens=int(group_config["max_tokens"]),
    )
    vocabulary_size = len(shared_vocabulary)
    separator_index = shared_vocabulary.index(SEPARATOR_TOKEN)

    print(f"\n{dataset_name} {scenario_code} - {scenario_name}")
    print(f"Vocabulary group    : {group_name}")
    print(f"Ukuran vocabulary  : {vocabulary_size:,}")
    print(f"Index token [SEP]  : {separator_index}")

    split_dataframes = {
        split_name: (
            dataframe[dataframe["split"] == split_name]
            .copy()
            .reset_index(drop=True)
        )
        for split_name in ["train", "validation", "test"]
    }

    report_rows: list[dict] = []
    split_output_paths: dict[str, str] = {}

    for split_name, split_dataframe in split_dataframes.items():
        if split_dataframe.empty:
            continue
        expected_rows = EXPECTED_SPLIT_ROWS[dataset_name][split_name]
        if len(split_dataframe) != expected_rows:
            raise ValueError(
                f"Jumlah {dataset_name} {scenario_code} {split_name} tidak sesuai. "
                f"Seharusnya {expected_rows:,}, ditemukan {len(split_dataframe):,}."
            )
        print(f"Vectorizing {split_name}: {len(split_dataframe):,} data...")
        vectorized_text = vectorize_texts(vectorizer, split_dataframe["text"])
        encoded_labels = encode_labels(split_dataframe["category"], label_to_index)
        validate_vectorized_split(
            vectorized_array=vectorized_text,
            labels=encoded_labels,
            expected_rows=expected_rows,
            expected_sequence_length=MAX_SEQUENCE_LENGTHS[scenario_code],
            vocabulary_size=vocabulary_size,
            number_of_classes=len(label_to_index),
            dataset_name=dataset_name,
            scenario_code=scenario_code,
            split_name=split_name,
        )
        output_path = save_split_arrays(
            output_directory,
            split_name,
            split_dataframe,
            vectorized_text,
            encoded_labels,
        )
        split_output_paths[split_name] = str(output_path)
        statistics = calculate_array_statistics(
            vectorized_text,
            split_dataframe["text"],
            MAX_SEQUENCE_LENGTHS[scenario_code],
        )
        report_rows.append(
            {
                "dataset": dataset_name,
                "scenario_code": scenario_code,
                "scenario_name": scenario_name,
                "uses_yake": bool(dataframe["uses_yake"].iloc[0]),
                "comparison_group": str(dataframe["comparison_group"].iloc[0]),
                "vocabulary_group": group_name,
                "vocabulary_sha256": vocabulary_sha256,
                "split": split_name,
                "jumlah_data": statistics["jumlah_data"],
                "max_sequence_length": MAX_SEQUENCE_LENGTHS[scenario_code],
                "maximum_vocabulary_size": int(group_config["max_tokens"]),
                "actual_vocabulary_size": vocabulary_size,
                "separator_index": separator_index,
                "mean_non_padding_tokens": statistics["mean_non_padding_tokens"],
                "median_non_padding_tokens": statistics["median_non_padding_tokens"],
                "maximum_non_padding_tokens": statistics["maximum_non_padding_tokens"],
                "padding_percentage": statistics["padding_percentage"],
                "oov_token_count": statistics["oov_token_count"],
                "oov_percentage_non_padding": statistics["oov_percentage_non_padding"],
                "truncated_document_count": statistics["truncated_document_count"],
                "truncated_document_percentage": statistics["truncated_document_percentage"],
                "output_path": str(output_path),
            }
        )

    vectorizer_config_path = save_vectorizer_config(
        output_directory=output_directory,
        dataset_name=dataset_name,
        scenario_code=scenario_code,
        scenario_name=scenario_name,
        group_name=group_name,
        group_config=group_config,
        vocabulary_size=vocabulary_size,
        vocabulary_sha256=vocabulary_sha256,
        shared_vocabulary_path=shared_vocabulary_path,
        label_to_index=label_to_index,
        index_to_label=index_to_label,
    )

    vocabulary_report = {
        "dataset": dataset_name,
        "scenario_code": scenario_code,
        "scenario_name": scenario_name,
        "vocabulary_group": group_name,
        "vocabulary_adapt_scenario": group_config["adapt_scenario"],
        "maximum_vocabulary_size": int(group_config["max_tokens"]),
        "actual_vocabulary_size": vocabulary_size,
        "separator_index": separator_index,
        "vocabulary_sha256": vocabulary_sha256,
        "shared_vocabulary_path": str(shared_vocabulary_path),
        "scenario_vocabulary_path": str(scenario_vocabulary_path),
        "vectorizer_config_path": str(vectorizer_config_path),
        "train_path": split_output_paths.get("train", ""),
        "validation_path": split_output_paths.get("validation", ""),
        "test_path": split_output_paths.get("test", ""),
    }
    return report_rows, vocabulary_report


def save_label_mappings() -> None:
    content = {
        "Kompas": {
            "label_to_index": KOMPAS_LABEL_TO_INDEX,
            "index_to_label": {
                str(k): v for k, v in create_index_to_label(KOMPAS_LABEL_TO_INDEX).items()
            },
        },
        "AG News": {
            "label_to_index": AGNEWS_LABEL_TO_INDEX,
            "index_to_label": {
                str(k): v for k, v in create_index_to_label(AGNEWS_LABEL_TO_INDEX).items()
            },
            "official_class_index_relation": {
                "1": "world",
                "2": "sports",
                "3": "business",
                "4": "sci_tech",
            },
        },
    }
    with open(LABEL_MAPPING_PATH, "w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=4)


def save_global_configuration(group_artifacts: dict[str, dict]) -> None:
    configuration = {
        "framework": "TensorFlow/Keras",
        "vectorizer": "TextVectorization",
        "adapt_policy": (
            "Vocabulary dibentuk hanya dari data train pada skenario adaptasi "
            "setiap kelompok, lalu digunakan bersama oleh seluruh skenario dalam kelompok."
        ),
        "standardize": STANDARDIZE_MODE,
        "split": SPLIT_MODE,
        "output_mode": OUTPUT_MODE,
        "batch_size": BATCH_SIZE,
        "random_seed": RANDOM_SEED,
        "max_sequence_lengths": MAX_SEQUENCE_LENGTHS,
        "vocabulary_groups": VOCABULARY_GROUPS,
        "scenario_to_group": SCENARIO_TO_GROUP,
        "group_artifacts": group_artifacts,
        "special_indices": {"padding": 0, "oov": 1},
        "separator_token": SEPARATOR_TOKEN,
        "k4_used": False,
        "fair_comparison_policy": {
            "K2_vs_K3": {
                "same_split": True,
                "same_max_sequence_length": True,
                "same_vocabulary_and_token_ids": True,
            },
            "A1_vs_A2": {
                "same_split": True,
                "same_max_sequence_length": True,
                "same_vocabulary_and_token_ids": True,
            },
        },
    }
    with open(VECTORIZATION_CONFIGURATION_PATH, "w", encoding="utf-8") as file:
        json.dump(configuration, file, ensure_ascii=False, indent=4)


def remove_legacy_k4_output() -> None:
    legacy_directory = VECTORIZED_DIR / "kompas_k4"
    if legacy_directory.exists():
        shutil.rmtree(legacy_directory)
        print(f"\nOutput vectorized K4 lama dihapus:\n{legacy_directory}")


def main() -> None:
    print("=" * 72)
    print("STEP 4.7 - TEXT VECTORIZATION")
    print("=" * 72)

    set_global_seed(RANDOM_SEED)
    print_tensorflow_environment()

    VECTORIZED_DIR.mkdir(parents=True, exist_ok=True)
    SHARED_VOCABULARIES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    remove_legacy_k4_output()

    kompas_dataframes = {
        scenario_code: load_split_dataset(
            file_path=file_path,
            dataset_name="Kompas",
            scenario_code=scenario_code,
            expected_rows=EXPECTED_ROWS["Kompas"],
            expected_splits={"train", "validation", "test"},
        )
        for scenario_code, file_path in KOMPAS_SCENARIO_PATHS.items()
    }

    agnews_dataframes = {
        scenario_code: load_agnews_scenario(scenario_code, paths)
        for scenario_code, paths in AGNEWS_SCENARIO_PATHS.items()
    }

    validate_scenario_alignment(kompas_dataframes["K1"], kompas_dataframes["K2"], "Kompas K2")
    validate_scenario_alignment(kompas_dataframes["K1"], kompas_dataframes["K3"], "Kompas K3")
    validate_scenario_alignment(agnews_dataframes["A1"], agnews_dataframes["A2"], "AG News A2")

    validate_split_counts(kompas_dataframes["K1"], "Kompas")
    validate_split_counts(agnews_dataframes["A1"], "AG News")
    validate_label_mapping(kompas_dataframes["K1"], KOMPAS_LABEL_TO_INDEX, "Kompas")
    validate_label_mapping(agnews_dataframes["A1"], AGNEWS_LABEL_TO_INDEX, "AG News")

    all_dataframes = {**kompas_dataframes, **agnews_dataframes}
    shared_vocabularies: dict[str, list[str]] = {}
    group_artifacts: dict[str, dict] = {}

    print("\n" + "=" * 72)
    print("BUILD SHARED VOCABULARIES")
    print("=" * 72)

    for group_name, group_config in VOCABULARY_GROUPS.items():
        adapt_code = str(group_config["adapt_scenario"])
        adapt_dataframe = all_dataframes[adapt_code]
        train_texts = adapt_dataframe.loc[adapt_dataframe["split"] == "train", "text"]
        print(
            f"\n{group_name}: adapt dari {group_config['dataset']} "
            f"{adapt_code} train ({len(train_texts):,} dokumen)..."
        )
        vocabulary, digest = build_shared_vocabulary(
            train_texts=train_texts,
            max_tokens=int(group_config["max_tokens"]),
            group_name=group_name,
        )
        group_directory = SHARED_VOCABULARIES_DIR / group_name
        vocabulary_path = group_directory / "vocabulary.txt"
        save_vocabulary_file(vocabulary, vocabulary_path)
        shared_vocabularies[group_name] = vocabulary
        group_artifacts[group_name] = {
            "dataset": group_config["dataset"],
            "adapt_scenario": adapt_code,
            "adapt_split": "train",
            "scenario_codes": group_config["scenario_codes"],
            "max_tokens": int(group_config["max_tokens"]),
            "actual_vocabulary_size": len(vocabulary),
            "separator_index": vocabulary.index(SEPARATOR_TOKEN),
            "vocabulary_sha256": digest,
            "vocabulary_path": str(vocabulary_path),
        }
        print(f"Ukuran vocabulary : {len(vocabulary):,}")
        print(f"Index [SEP]        : {vocabulary.index(SEPARATOR_TOKEN)}")

    all_report_rows: list[dict] = []
    vocabulary_rows: list[dict] = []

    print("\n" + "=" * 72)
    print("VECTORIZATION KOMPAS")
    print("=" * 72)

    kompas_index_to_label = create_index_to_label(KOMPAS_LABEL_TO_INDEX)
    for scenario_code in ["K1", "K2", "K3"]:
        group_name = SCENARIO_TO_GROUP[scenario_code]
        report_rows, vocabulary_report = process_scenario(
            dataframe=kompas_dataframes[scenario_code],
            dataset_name="Kompas",
            scenario_code=scenario_code,
            shared_vocabulary=shared_vocabularies[group_name],
            vocabulary_sha256=group_artifacts[group_name]["vocabulary_sha256"],
            shared_vocabulary_path=Path(group_artifacts[group_name]["vocabulary_path"]),
            label_to_index=KOMPAS_LABEL_TO_INDEX,
            index_to_label=kompas_index_to_label,
        )
        all_report_rows.extend(report_rows)
        vocabulary_rows.append(vocabulary_report)

    print("\n" + "=" * 72)
    print("VECTORIZATION AG NEWS")
    print("=" * 72)

    agnews_index_to_label = create_index_to_label(AGNEWS_LABEL_TO_INDEX)
    for scenario_code in ["A1", "A2"]:
        group_name = SCENARIO_TO_GROUP[scenario_code]
        report_rows, vocabulary_report = process_scenario(
            dataframe=agnews_dataframes[scenario_code],
            dataset_name="AG News",
            scenario_code=scenario_code,
            shared_vocabulary=shared_vocabularies[group_name],
            vocabulary_sha256=group_artifacts[group_name]["vocabulary_sha256"],
            shared_vocabulary_path=Path(group_artifacts[group_name]["vocabulary_path"]),
            label_to_index=AGNEWS_LABEL_TO_INDEX,
            index_to_label=agnews_index_to_label,
        )
        all_report_rows.extend(report_rows)
        vocabulary_rows.append(vocabulary_report)

    vectorization_report = pd.DataFrame(all_report_rows)
    vocabulary_report = pd.DataFrame(vocabulary_rows)

    k2_k3_hashes = set(
        vocabulary_report.loc[
            vocabulary_report["scenario_code"].isin(["K2", "K3"]),
            "vocabulary_sha256",
        ]
    )
    if len(k2_k3_hashes) != 1:
        raise ValueError("Vocabulary K2 dan K3 tidak identik.")

    a1_a2_hashes = set(
        vocabulary_report.loc[
            vocabulary_report["scenario_code"].isin(["A1", "A2"]),
            "vocabulary_sha256",
        ]
    )
    if len(a1_a2_hashes) != 1:
        raise ValueError("Vocabulary A1 dan A2 tidak identik.")

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
    save_label_mappings()
    save_global_configuration(group_artifacts)

    print("\n" + "=" * 72)
    print("HASIL TEXT VECTORIZATION")
    print("=" * 72)

    display_columns = [
        "dataset",
        "scenario_code",
        "split",
        "jumlah_data",
        "max_sequence_length",
        "actual_vocabulary_size",
        "mean_non_padding_tokens",
        "padding_percentage",
        "oov_percentage_non_padding",
        "truncated_document_count",
    ]
    print(vectorization_report[display_columns].to_string(index=False))

    print("\nValidasi eksperimen:")
    print("K2 dan K3 menggunakan vocabulary serta token ID yang sama.")
    print("A1 dan A2 menggunakan vocabulary serta token ID yang sama.")
    print("Vocabulary hanya diadaptasi dari data train.")
    print("K4 tidak digunakan.")

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
    print(VECTORIZATION_CONFIGURATION_PATH)
    print("\nTahap text vectorization selesai.")


if __name__ == "__main__":
    main()

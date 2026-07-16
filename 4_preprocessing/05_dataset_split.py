from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


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

SCENARIOS_DIR = (
    PROJECT_ROOT
    / "2_data"
    / "scenarios"
)

SPLITS_DIR = (
    PROJECT_ROOT
    / "2_data"
    / "splits"
)


# ============================================================
# INPUT KOMPAS
# ============================================================

KOMPAS_K1_PATH = (
    SCENARIOS_DIR
    / "kompas_scenario_k1_title.csv"
)

KOMPAS_K2_PATH = (
    SCENARIOS_DIR
    / "kompas_scenario_k2_title_description.csv"
)

KOMPAS_K3_PATH = (
    SCENARIOS_DIR
    / "kompas_scenario_k3_title_description_keyword.csv"
)

KOMPAS_K4_PATH = (
    SCENARIOS_DIR
    / "kompas_scenario_k4_weighted_news.csv"
)


# ============================================================
# INPUT AG NEWS
# ============================================================

AGNEWS_TRAIN_A1_PATH = (
    SCENARIOS_DIR
    / "agnews_train_scenario_a1_title.csv"
)

AGNEWS_TRAIN_A2_PATH = (
    SCENARIOS_DIR
    / "agnews_train_scenario_a2_title_description.csv"
)

AGNEWS_TEST_A1_PATH = (
    SCENARIOS_DIR
    / "agnews_test_scenario_a1_title.csv"
)

AGNEWS_TEST_A2_PATH = (
    SCENARIOS_DIR
    / "agnews_test_scenario_a2_title_description.csv"
)


# ============================================================
# OUTPUT KOMPAS
# ============================================================

KOMPAS_K1_SPLIT_PATH = (
    SPLITS_DIR
    / "kompas_k1_split.csv"
)

KOMPAS_K2_SPLIT_PATH = (
    SPLITS_DIR
    / "kompas_k2_split.csv"
)

KOMPAS_K3_SPLIT_PATH = (
    SPLITS_DIR
    / "kompas_k3_split.csv"
)

KOMPAS_K4_SPLIT_PATH = (
    SPLITS_DIR
    / "kompas_k4_split.csv"
)


# ============================================================
# OUTPUT AG NEWS
# ============================================================

AGNEWS_A1_TRAIN_VALIDATION_PATH = (
    SPLITS_DIR
    / "agnews_a1_train_validation.csv"
)

AGNEWS_A2_TRAIN_VALIDATION_PATH = (
    SPLITS_DIR
    / "agnews_a2_train_validation.csv"
)

AGNEWS_A1_TEST_PATH = (
    SPLITS_DIR
    / "agnews_a1_test.csv"
)

AGNEWS_A2_TEST_PATH = (
    SPLITS_DIR
    / "agnews_a2_test.csv"
)


# ============================================================
# OUTPUT ASSIGNMENT DAN LAPORAN
# ============================================================

KOMPAS_SPLIT_ASSIGNMENT_PATH = (
    SPLITS_DIR
    / "kompas_split_assignment.csv"
)

AGNEWS_SPLIT_ASSIGNMENT_PATH = (
    SPLITS_DIR
    / "agnews_split_assignment.csv"
)

DATASET_SPLIT_REPORT_PATH = (
    TABLES_DIR
    / "dataset_split_report.csv"
)

DATASET_SPLIT_CATEGORY_REPORT_PATH = (
    TABLES_DIR
    / "dataset_split_category_report.csv"
)

DATASET_SPLIT_CONFIGURATION_PATH = (
    TABLES_DIR
    / "dataset_split_configuration.json"
)


# ============================================================
# KONFIGURASI
# ============================================================

RANDOM_SEED = 42

KOMPAS_TRAIN_SIZE = 0.80
KOMPAS_VALIDATION_SIZE = 0.10
KOMPAS_TEST_SIZE = 0.10

AGNEWS_TRAIN_SIZE = 0.90
AGNEWS_VALIDATION_SIZE = 0.10


# ============================================================
# MEMBACA DATASET SKENARIO
# ============================================================

def load_scenario(
    file_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Membaca dataset skenario dan memastikan strukturnya benar.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"File skenario {dataset_name} tidak ditemukan:\n"
            f"{file_path}"
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
        "word_count",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset {dataset_name} tidak memiliki kolom: "
            f"{missing_columns}"
        )

    if dataframe.empty:
        raise ValueError(
            f"Dataset {dataset_name} kosong."
        )

    return dataframe


# ============================================================
# VALIDASI KESELARASAN ANTARSKENARIO
# ============================================================

def validate_scenario_alignment(
    reference_dataframe: pd.DataFrame,
    comparison_dataframe: pd.DataFrame,
    comparison_name: str,
) -> None:
    """
    Memastikan seluruh skenario menggunakan artikel dan label
    yang sama.

    Yang boleh berbeda hanya isi kolom text dan word_count.
    """

    reference = (
        reference_dataframe[
            [
                "document_id",
                "category",
            ]
        ]
        .sort_values("document_id")
        .reset_index(drop=True)
    )

    comparison = (
        comparison_dataframe[
            [
                "document_id",
                "category",
            ]
        ]
        .sort_values("document_id")
        .reset_index(drop=True)
    )

    if len(reference) != len(comparison):
        raise ValueError(
            f"Jumlah data {comparison_name} tidak sama "
            f"dengan skenario referensi."
        )

    if not reference.equals(comparison):
        raise ValueError(
            f"Document ID atau kategori pada {comparison_name} "
            f"tidak sama dengan skenario referensi."
        )


# ============================================================
# MEMBUAT SPLIT KOMPAS
# ============================================================

def create_kompas_split_assignment(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Membagi Kompas menjadi:
    80% train, 10% validation, dan 10% test.

    Stratify menjaga distribusi kategori tetap seimbang.
    """

    base_data = dataframe[
        [
            "document_id",
            "category",
        ]
    ].copy()

    train_data, temporary_data = train_test_split(
        base_data,
        test_size=(
            KOMPAS_VALIDATION_SIZE
            + KOMPAS_TEST_SIZE
        ),
        random_state=RANDOM_SEED,
        stratify=base_data["category"],
    )

    validation_data, test_data = train_test_split(
        temporary_data,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temporary_data["category"],
    )

    train_data = train_data.copy()
    validation_data = validation_data.copy()
    test_data = test_data.copy()

    train_data["split"] = "train"
    validation_data["split"] = "validation"
    test_data["split"] = "test"

    assignment = pd.concat(
        [
            train_data,
            validation_data,
            test_data,
        ],
        ignore_index=True,
    )

    assignment = assignment.sort_values(
        "document_id"
    ).reset_index(drop=True)

    return assignment


# ============================================================
# MEMBUAT SPLIT AG NEWS TRAIN
# ============================================================

def create_agnews_split_assignment(
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Membagi AG News Train menjadi:
    90% train dan 10% validation.

    AG News Test resmi tetap menjadi final test.
    """

    train_base = train_dataframe[
        [
            "document_id",
            "category",
        ]
    ].copy()

    train_part, validation_part = train_test_split(
        train_base,
        test_size=AGNEWS_VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=train_base["category"],
    )

    train_part = train_part.copy()
    validation_part = validation_part.copy()

    train_part["split"] = "train"
    validation_part["split"] = "validation"

    official_test = test_dataframe[
        [
            "document_id",
            "category",
        ]
    ].copy()

    official_test["split"] = "test"

    assignment = pd.concat(
        [
            train_part,
            validation_part,
            official_test,
        ],
        ignore_index=True,
    )

    assignment = assignment.sort_values(
        [
            "split",
            "document_id",
        ]
    ).reset_index(drop=True)

    return assignment


# ============================================================
# MENAMBAHKAN SPLIT KE DATASET SKENARIO
# ============================================================

def apply_split_assignment(
    scenario_dataframe: pd.DataFrame,
    split_assignment: pd.DataFrame,
) -> pd.DataFrame:
    """
    Menambahkan kolom split berdasarkan document_id.
    """

    assignment = split_assignment[
        [
            "document_id",
            "split",
        ]
    ].copy()

    result = scenario_dataframe.merge(
        assignment,
        on="document_id",
        how="left",
        validate="one_to_one",
    )

    missing_split = int(
        result["split"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if missing_split > 0:
        raise ValueError(
            f"Ditemukan {missing_split} artikel "
            f"yang tidak memperoleh split."
        )

    return result


# ============================================================
# VALIDASI SPLIT
# ============================================================

def validate_split_dataset(
    dataframe: pd.DataFrame,
    dataset_name: str,
    allowed_splits: set[str],
) -> None:
    """
    Memastikan hasil split tidak bermasalah.
    """

    actual_splits = set(
        dataframe["split"].unique()
    )

    invalid_splits = (
        actual_splits
        - allowed_splits
    )

    if invalid_splits:
        raise ValueError(
            f"Split tidak valid pada {dataset_name}: "
            f"{invalid_splits}"
        )

    duplicated_ids = int(
        dataframe["document_id"]
        .duplicated()
        .sum()
    )

    if duplicated_ids > 0:
        raise ValueError(
            f"Dataset {dataset_name} memiliki "
            f"{duplicated_ids} document_id duplikat."
        )

    empty_text = int(
        dataframe["text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_text > 0:
        raise ValueError(
            f"Dataset {dataset_name} memiliki "
            f"{empty_text} text kosong."
        )


# ============================================================
# LAPORAN JUMLAH DATA
# ============================================================

def create_split_report(
    dataframes: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Membuat laporan jumlah dan persentase data setiap split.
    """

    report_rows: list[dict] = []

    for dataframe in dataframes:
        dataset_name = dataframe["dataset"].iloc[0]
        scenario_code = dataframe["scenario_code"].iloc[0]
        scenario_name = dataframe["scenario_name"].iloc[0]

        total_data = len(dataframe)

        split_counts = (
            dataframe["split"]
            .value_counts()
        )

        for split_name, count in split_counts.items():
            report_rows.append(
                {
                    "dataset": dataset_name,
                    "scenario_code": scenario_code,
                    "scenario_name": scenario_name,
                    "split": split_name,
                    "jumlah_data": int(count),
                    "persentase": round(
                        count
                        / total_data
                        * 100,
                        2,
                    ),
                }
            )

    return pd.DataFrame(report_rows)


# ============================================================
# LAPORAN DISTRIBUSI KATEGORI
# ============================================================

def create_category_split_report(
    dataframes: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Membuat laporan distribusi kategori pada setiap split.
    """

    report_rows: list[dict] = []

    for dataframe in dataframes:
        dataset_name = dataframe["dataset"].iloc[0]
        scenario_code = dataframe["scenario_code"].iloc[0]

        grouped = (
            dataframe.groupby(
                [
                    "split",
                    "category",
                ]
            )
            .size()
            .reset_index(
                name="jumlah_data"
            )
        )

        for row in grouped.itertuples(index=False):
            split_total = len(
                dataframe[
                    dataframe["split"]
                    == row.split
                ]
            )

            report_rows.append(
                {
                    "dataset": dataset_name,
                    "scenario_code": scenario_code,
                    "split": row.split,
                    "category": row.category,
                    "jumlah_data": int(
                        row.jumlah_data
                    ),
                    "persentase_dalam_split": round(
                        row.jumlah_data
                        / split_total
                        * 100,
                        2,
                    ),
                }
            )

    return pd.DataFrame(report_rows)


# ============================================================
# MENYIMPAN KONFIGURASI
# ============================================================

def save_split_configuration() -> None:
    """
    Menyimpan konfigurasi agar eksperimen dapat direproduksi.
    """

    configuration = {
        "random_seed": RANDOM_SEED,
        "stratified_split": True,
        "split_key": "document_id",
        "kompas": {
            "train": KOMPAS_TRAIN_SIZE,
            "validation": KOMPAS_VALIDATION_SIZE,
            "test": KOMPAS_TEST_SIZE,
        },
        "ag_news": {
            "original_train": {
                "train": AGNEWS_TRAIN_SIZE,
                "validation": AGNEWS_VALIDATION_SIZE,
            },
            "official_test": {
                "test": 1.0,
            },
        },
        "important_note": (
            "Split assignment yang sama digunakan "
            "untuk seluruh skenario dalam dataset yang sama."
        ),
    }

    with open(
        DATASET_SPLIT_CONFIGURATION_PATH,
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
    Menjalankan pembagian dataset.
    """

    print("=" * 72)
    print("STEP 4.5 - DATASET SPLIT")
    print("=" * 72)

    SPLITS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # MEMUAT KOMPAS
    # ========================================================

    kompas_k1 = load_scenario(
        KOMPAS_K1_PATH,
        "Kompas K1",
    )

    kompas_k2 = load_scenario(
        KOMPAS_K2_PATH,
        "Kompas K2",
    )

    kompas_k3 = load_scenario(
        KOMPAS_K3_PATH,
        "Kompas K3",
    )

    kompas_k4 = load_scenario(
        KOMPAS_K4_PATH,
        "Kompas K4",
    )

    validate_scenario_alignment(
        kompas_k1,
        kompas_k2,
        "Kompas K2",
    )

    validate_scenario_alignment(
        kompas_k1,
        kompas_k3,
        "Kompas K3",
    )

    validate_scenario_alignment(
        kompas_k1,
        kompas_k4,
        "Kompas K4",
    )

    # ========================================================
    # MEMUAT AG NEWS
    # ========================================================

    agnews_train_a1 = load_scenario(
        AGNEWS_TRAIN_A1_PATH,
        "AG News Train A1",
    )

    agnews_train_a2 = load_scenario(
        AGNEWS_TRAIN_A2_PATH,
        "AG News Train A2",
    )

    agnews_test_a1 = load_scenario(
        AGNEWS_TEST_A1_PATH,
        "AG News Test A1",
    )

    agnews_test_a2 = load_scenario(
        AGNEWS_TEST_A2_PATH,
        "AG News Test A2",
    )

    validate_scenario_alignment(
        agnews_train_a1,
        agnews_train_a2,
        "AG News Train A2",
    )

    validate_scenario_alignment(
        agnews_test_a1,
        agnews_test_a2,
        "AG News Test A2",
    )

    # ========================================================
    # MEMBUAT ASSIGNMENT
    # ========================================================

    print("\nMembagi dataset Kompas 80:10:10...")

    kompas_assignment = (
        create_kompas_split_assignment(
            kompas_k1
        )
    )

    print(
        "Membagi AG News Train 90:10 "
        "dan mempertahankan test resmi..."
    )

    agnews_assignment = (
        create_agnews_split_assignment(
            train_dataframe=agnews_train_a1,
            test_dataframe=agnews_test_a1,
        )
    )

    # ========================================================
    # MENERAPKAN SPLIT KOMPAS
    # ========================================================

    kompas_k1_split = apply_split_assignment(
        kompas_k1,
        kompas_assignment,
    )

    kompas_k2_split = apply_split_assignment(
        kompas_k2,
        kompas_assignment,
    )

    kompas_k3_split = apply_split_assignment(
        kompas_k3,
        kompas_assignment,
    )

    kompas_k4_split = apply_split_assignment(
        kompas_k4,
        kompas_assignment,
    )

    # ========================================================
    # MENERAPKAN SPLIT AG NEWS
    # ========================================================

    agnews_train_assignment = (
        agnews_assignment[
            agnews_assignment["split"]
            .isin(
                [
                    "train",
                    "validation",
                ]
            )
        ]
        .copy()
    )

    agnews_test_assignment = (
        agnews_assignment[
            agnews_assignment["split"]
            == "test"
        ]
        .copy()
    )

    agnews_a1_train_validation = (
        apply_split_assignment(
            agnews_train_a1,
            agnews_train_assignment,
        )
    )

    agnews_a2_train_validation = (
        apply_split_assignment(
            agnews_train_a2,
            agnews_train_assignment,
        )
    )

    agnews_a1_test = apply_split_assignment(
        agnews_test_a1,
        agnews_test_assignment,
    )

    agnews_a2_test = apply_split_assignment(
        agnews_test_a2,
        agnews_test_assignment,
    )

    # ========================================================
    # VALIDASI
    # ========================================================

    kompas_split_datasets = [
        kompas_k1_split,
        kompas_k2_split,
        kompas_k3_split,
        kompas_k4_split,
    ]

    for dataframe in kompas_split_datasets:
        validate_split_dataset(
            dataframe,
            dataframe["scenario_code"].iloc[0],
            {
                "train",
                "validation",
                "test",
            },
        )

    validate_split_dataset(
        agnews_a1_train_validation,
        "AG News A1 Train Validation",
        {
            "train",
            "validation",
        },
    )

    validate_split_dataset(
        agnews_a2_train_validation,
        "AG News A2 Train Validation",
        {
            "train",
            "validation",
        },
    )

    validate_split_dataset(
        agnews_a1_test,
        "AG News A1 Test",
        {"test"},
    )

    validate_split_dataset(
        agnews_a2_test,
        "AG News A2 Test",
        {"test"},
    )

    # ========================================================
    # MENYIMPAN DATASET
    # ========================================================

    kompas_k1_split.to_csv(
        KOMPAS_K1_SPLIT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    kompas_k2_split.to_csv(
        KOMPAS_K2_SPLIT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    kompas_k3_split.to_csv(
        KOMPAS_K3_SPLIT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    kompas_k4_split.to_csv(
        KOMPAS_K4_SPLIT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    agnews_a1_train_validation.to_csv(
        AGNEWS_A1_TRAIN_VALIDATION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    agnews_a2_train_validation.to_csv(
        AGNEWS_A2_TRAIN_VALIDATION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    agnews_a1_test.to_csv(
        AGNEWS_A1_TEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    agnews_a2_test.to_csv(
        AGNEWS_A2_TEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    kompas_assignment.to_csv(
        KOMPAS_SPLIT_ASSIGNMENT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    agnews_assignment.to_csv(
        AGNEWS_SPLIT_ASSIGNMENT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # MEMBUAT LAPORAN
    # ========================================================

    all_split_datasets = [
        *kompas_split_datasets,
        agnews_a1_train_validation,
        agnews_a2_train_validation,
        agnews_a1_test,
        agnews_a2_test,
    ]

    split_report = create_split_report(
        all_split_datasets
    )

    category_report = (
        create_category_split_report(
            all_split_datasets
        )
    )

    split_report.to_csv(
        DATASET_SPLIT_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    category_report.to_csv(
        DATASET_SPLIT_CATEGORY_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    save_split_configuration()

    # ========================================================
    # MENAMPILKAN HASIL
    # ========================================================

    print("\n" + "=" * 72)
    print("HASIL DATASET SPLIT")
    print("=" * 72)

    print(
        split_report.to_string(
            index=False
        )
    )

    print("\nDistribusi kategori Kompas K1:")

    print(
        category_report[
            (
                category_report["dataset"]
                == "Kompas"
            )
            & (
                category_report["scenario_code"]
                == "K1"
            )
        ]
        .to_string(index=False)
    )

    print("\n" + "=" * 72)
    print("OUTPUT DATASET SPLIT")
    print("=" * 72)

    print("\nFolder split:")
    print(SPLITS_DIR)

    print("\nAssignment Kompas:")
    print(KOMPAS_SPLIT_ASSIGNMENT_PATH)

    print("\nAssignment AG News:")
    print(AGNEWS_SPLIT_ASSIGNMENT_PATH)

    print("\nLaporan split:")
    print(DATASET_SPLIT_REPORT_PATH)

    print("\nLaporan kategori per split:")
    print(DATASET_SPLIT_CATEGORY_REPORT_PATH)

    print("\nKonfigurasi split:")
    print(DATASET_SPLIT_CONFIGURATION_PATH)

    print("\nTahap dataset split selesai.")


if __name__ == "__main__":
    main()
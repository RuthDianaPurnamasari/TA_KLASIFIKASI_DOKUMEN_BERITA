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
# PATH INPUT KOMPAS
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


# ============================================================
# PATH INPUT AG NEWS
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
# PATH OUTPUT KOMPAS
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


# ============================================================
# PATH OUTPUT AG NEWS
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
# PATH OUTPUT ASSIGNMENT DAN LAPORAN
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
# FILE K4 LAMA
# ============================================================

LEGACY_KOMPAS_K4_SPLIT_PATH = (
    SPLITS_DIR
    / "kompas_k4_split.csv"
)


# ============================================================
# KONFIGURASI SPLIT
# ============================================================

RANDOM_SEED = 42

KOMPAS_TRAIN_SIZE = 0.80
KOMPAS_VALIDATION_SIZE = 0.10
KOMPAS_TEST_SIZE = 0.10

AGNEWS_TRAIN_SIZE = 0.90
AGNEWS_VALIDATION_SIZE = 0.10


# ============================================================
# JUMLAH DATA YANG DIHARAPKAN
# ============================================================

EXPECTED_ROW_COUNTS = {
    "Kompas": 9_997,
    "AG News Train": 119_817,
    "AG News Test": 7_600,
}


# ============================================================
# KOLOM WAJIB DATASET SKENARIO
# ============================================================

REQUIRED_SCENARIO_COLUMNS = [
    "document_id",
    "category",
    "dataset",
    "scenario_code",
    "scenario_name",
    "text",
    "word_count",
    "uses_yake",
    "includes_content",
    "yake_comparison_role",
    "comparison_group",
]


# ============================================================
# MEMBACA DATASET SKENARIO
# ============================================================

def load_scenario(
    file_path: Path,
    expected_dataset: str,
    expected_scenario_code: str,
    expected_rows: int,
) -> pd.DataFrame:
    """
    Membaca dataset skenario dan memvalidasi struktur,
    identitas skenario, serta jumlah datanya.
    """

    scenario_label = (
        f"{expected_dataset} "
        f"{expected_scenario_code}"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"File skenario {scenario_label} "
            f"tidak ditemukan:\n{file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Path skenario {scenario_label} "
            f"bukan file:\n{file_path}"
        )

    dataframe = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    if dataframe.empty:
        raise ValueError(
            f"Dataset {scenario_label} kosong."
        )

    missing_columns = [
        column
        for column in REQUIRED_SCENARIO_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset {scenario_label} tidak memiliki "
            f"kolom wajib: {missing_columns}\n"
            f"Kolom tersedia: {list(dataframe.columns)}"
        )

    if expected_dataset.startswith("AG News"):
        agnews_required_columns = [
            "source_row",
            "class_index",
        ]

        missing_agnews_columns = [
            column
            for column in agnews_required_columns
            if column not in dataframe.columns
        ]

        if missing_agnews_columns:
            raise ValueError(
                f"Dataset {scenario_label} tidak memiliki "
                f"kolom AG News: {missing_agnews_columns}"
            )

    if len(dataframe) != expected_rows:
        raise ValueError(
            f"Jumlah data {scenario_label} tidak sesuai.\n"
            f"Seharusnya: {expected_rows:,}\n"
            f"Ditemukan : {len(dataframe):,}"
        )

    actual_datasets = set(
        dataframe["dataset"]
        .astype(str)
        .str.strip()
        .unique()
    )

    if actual_datasets != {expected_dataset}:
        raise ValueError(
            f"Identitas dataset pada {scenario_label} "
            f"tidak sesuai: {actual_datasets}"
        )

    actual_scenario_codes = set(
        dataframe["scenario_code"]
        .astype(str)
        .str.strip()
        .unique()
    )

    if actual_scenario_codes != {
        expected_scenario_code
    }:
        raise ValueError(
            f"Scenario code pada {scenario_label} "
            f"tidak sesuai: {actual_scenario_codes}"
        )

    duplicated_ids = int(
        dataframe["document_id"]
        .duplicated()
        .sum()
    )

    if duplicated_ids > 0:
        raise ValueError(
            f"Dataset {scenario_label} memiliki "
            f"{duplicated_ids:,} document_id duplikat."
        )

    empty_document_ids = int(
        dataframe["document_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_document_ids > 0:
        raise ValueError(
            f"Dataset {scenario_label} memiliki "
            f"{empty_document_ids:,} document_id kosong."
        )

    empty_texts = int(
        dataframe["text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_texts > 0:
        raise ValueError(
            f"Dataset {scenario_label} memiliki "
            f"{empty_texts:,} teks kosong."
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
    Memastikan dua skenario menggunakan dokumen, urutan
    identitas, dan label yang sama.

    Isi text dan word_count boleh berbeda.
    """

    comparison_columns = [
        "document_id",
        "category",
    ]

    for optional_column in [
        "source_row",
        "class_index",
    ]:
        if (
            optional_column in reference_dataframe.columns
            and optional_column in comparison_dataframe.columns
        ):
            comparison_columns.append(
                optional_column
            )

    reference = (
        reference_dataframe[
            comparison_columns
        ]
        .sort_values(
            "document_id",
            kind="stable",
        )
        .reset_index(drop=True)
        .astype(str)
    )

    comparison = (
        comparison_dataframe[
            comparison_columns
        ]
        .sort_values(
            "document_id",
            kind="stable",
        )
        .reset_index(drop=True)
        .astype(str)
    )

    if len(reference) != len(comparison):
        raise ValueError(
            f"Jumlah data {comparison_name} tidak sama "
            "dengan skenario referensi."
        )

    if not reference.equals(comparison):
        raise ValueError(
            f"Dokumen atau label pada {comparison_name} "
            "tidak sama dengan skenario referensi."
        )


# ============================================================
# MENGAMBIL KOLOM ASSIGNMENT
# ============================================================

def get_assignment_base(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mengambil identitas dokumen dan label untuk assignment.
    """

    columns = [
        "document_id",
        "category",
    ]

    if "class_index" in dataframe.columns:
        columns.append(
            "class_index"
        )

    return dataframe[
        columns
    ].copy()


# ============================================================
# MEMBUAT SPLIT KOMPAS
# ============================================================

def create_kompas_split_assignment(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Membagi Kompas menjadi 80% train, 10% validation,
    dan 10% test dengan stratified split.
    """

    base_data = get_assignment_base(
        dataframe
    )

    train_data, temporary_data = train_test_split(
        base_data,
        test_size=(
            KOMPAS_VALIDATION_SIZE
            + KOMPAS_TEST_SIZE
        ),
        random_state=RANDOM_SEED,
        stratify=base_data["category"],
        shuffle=True,
    )

    relative_test_size = (
        KOMPAS_TEST_SIZE
        / (
            KOMPAS_VALIDATION_SIZE
            + KOMPAS_TEST_SIZE
        )
    )

    validation_data, test_data = train_test_split(
        temporary_data,
        test_size=relative_test_size,
        random_state=RANDOM_SEED,
        stratify=temporary_data["category"],
        shuffle=True,
    )

    train_data = train_data.copy()
    validation_data = validation_data.copy()
    test_data = test_data.copy()

    train_data["split"] = "train"
    validation_data["split"] = "validation"
    test_data["split"] = "test"

    train_data["source_partition"] = (
        "kompas_clean"
    )
    validation_data["source_partition"] = (
        "kompas_clean"
    )
    test_data["source_partition"] = (
        "kompas_clean"
    )

    assignment = pd.concat(
        [
            train_data,
            validation_data,
            test_data,
        ],
        ignore_index=True,
    )

    split_order = {
        "train": 0,
        "validation": 1,
        "test": 2,
    }

    assignment["_split_order"] = (
        assignment["split"]
        .map(split_order)
    )

    assignment = (
        assignment
        .sort_values(
            [
                "_split_order",
                "document_id",
            ],
            kind="stable",
        )
        .drop(
            columns=["_split_order"]
        )
        .reset_index(drop=True)
    )

    return assignment


# ============================================================
# MEMBUAT SPLIT AG NEWS
# ============================================================

def create_agnews_split_assignment(
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Membagi AG News clean train menjadi 90% train dan
    10% validation.

    AG News official test dipertahankan seluruhnya sebagai
    final test.
    """

    train_base = get_assignment_base(
        train_dataframe
    )

    official_test = get_assignment_base(
        test_dataframe
    )

    train_ids = set(
        train_base["document_id"]
        .astype(str)
    )

    test_ids = set(
        official_test["document_id"]
        .astype(str)
    )

    overlapping_ids = (
        train_ids.intersection(test_ids)
    )

    if overlapping_ids:
        raise ValueError(
            "Ditemukan document_id yang sama antara "
            f"AG News train dan test sebanyak "
            f"{len(overlapping_ids):,}."
        )

    train_part, validation_part = train_test_split(
        train_base,
        test_size=AGNEWS_VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=train_base["category"],
        shuffle=True,
    )

    train_part = train_part.copy()
    validation_part = validation_part.copy()
    official_test = official_test.copy()

    train_part["split"] = "train"
    validation_part["split"] = "validation"
    official_test["split"] = "test"

    train_part["source_partition"] = (
        "ag_news_clean_train"
    )

    validation_part["source_partition"] = (
        "ag_news_clean_train"
    )

    official_test["source_partition"] = (
        "ag_news_official_test"
    )

    assignment = pd.concat(
        [
            train_part,
            validation_part,
            official_test,
        ],
        ignore_index=True,
    )

    split_order = {
        "train": 0,
        "validation": 1,
        "test": 2,
    }

    assignment["_split_order"] = (
        assignment["split"]
        .map(split_order)
    )

    assignment = (
        assignment
        .sort_values(
            [
                "_split_order",
                "document_id",
            ],
            kind="stable",
        )
        .drop(
            columns=["_split_order"]
        )
        .reset_index(drop=True)
    )

    return assignment


# ============================================================
# VALIDASI ASSIGNMENT
# ============================================================

def validate_split_assignment(
    assignment: pd.DataFrame,
    assignment_name: str,
    expected_rows: int,
    expected_splits: set[str],
) -> None:
    """
    Memastikan assignment lengkap, unik, dan tidak memiliki
    dokumen yang ditempatkan pada lebih dari satu split.
    """

    if len(assignment) != expected_rows:
        raise ValueError(
            f"Jumlah assignment {assignment_name} "
            f"tidak sesuai.\n"
            f"Seharusnya: {expected_rows:,}\n"
            f"Ditemukan : {len(assignment):,}"
        )

    duplicated_ids = int(
        assignment["document_id"]
        .duplicated()
        .sum()
    )

    if duplicated_ids > 0:
        raise ValueError(
            f"Assignment {assignment_name} memiliki "
            f"{duplicated_ids:,} document_id duplikat."
        )

    actual_splits = set(
        assignment["split"]
        .astype(str)
        .unique()
    )

    if actual_splits != expected_splits:
        raise ValueError(
            f"Jenis split {assignment_name} tidak sesuai.\n"
            f"Seharusnya: {expected_splits}\n"
            f"Ditemukan : {actual_splits}"
        )

    missing_categories = int(
        assignment["category"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if missing_categories > 0:
        raise ValueError(
            f"Assignment {assignment_name} memiliki "
            f"{missing_categories:,} kategori kosong."
        )

    split_id_sets = {
        split_name: set(
            assignment.loc[
                assignment["split"] == split_name,
                "document_id",
            ].astype(str)
        )
        for split_name in expected_splits
    }

    split_names = sorted(
        split_id_sets.keys()
    )

    for first_index, first_split in enumerate(
        split_names
    ):
        for second_split in split_names[
            first_index + 1:
        ]:
            overlap = (
                split_id_sets[first_split]
                .intersection(
                    split_id_sets[second_split]
                )
            )

            if overlap:
                raise ValueError(
                    f"Assignment {assignment_name} memiliki "
                    f"{len(overlap):,} dokumen yang masuk "
                    f"ke split {first_split} dan "
                    f"{second_split}."
                )


# ============================================================
# MENERAPKAN ASSIGNMENT KE SKENARIO
# ============================================================

def apply_split_assignment(
    scenario_dataframe: pd.DataFrame,
    split_assignment: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Menambahkan split eksperimen berdasarkan document_id.

    Kolom split dari tahap sebelumnya, jika ada, dipertahankan
    sebagai source_split agar tidak bertabrakan saat merge.
    """

    scenario_data = (
        scenario_dataframe.copy()
        .reset_index(drop=True)
    )

    if "split" in scenario_data.columns:
        if "source_split" in scenario_data.columns:
            raise ValueError(
                f"Dataset {dataset_name} sudah memiliki "
                "kolom split dan source_split."
            )

        scenario_data = scenario_data.rename(
            columns={
                "split": "source_split",
            }
        )

    original_ids = (
        scenario_data["document_id"]
        .astype(str)
        .reset_index(drop=True)
    )

    scenario_data["_original_order"] = range(
        len(scenario_data)
    )

    assignment = (
        split_assignment[
            [
                "document_id",
                "split",
            ]
        ]
        .copy()
    )

    duplicated_assignment_ids = int(
        assignment["document_id"]
        .duplicated()
        .sum()
    )

    if duplicated_assignment_ids > 0:
        raise ValueError(
            f"Assignment untuk {dataset_name} memiliki "
            f"{duplicated_assignment_ids:,} ID duplikat."
        )

    result = scenario_data.merge(
        assignment,
        on="document_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    result = (
        result
        .sort_values(
            "_original_order",
            kind="stable",
        )
        .drop(
            columns=["_original_order"]
        )
        .reset_index(drop=True)
    )

    if len(result) != len(
        scenario_dataframe
    ):
        raise ValueError(
            f"Jumlah data {dataset_name} berubah "
            "setelah assignment diterapkan."
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
            f"Ditemukan {missing_split:,} dokumen "
            f"{dataset_name} yang tidak memperoleh split."
        )

    result_ids = (
        result["document_id"]
        .astype(str)
        .reset_index(drop=True)
    )

    if not original_ids.equals(
        result_ids
    ):
        raise ValueError(
            f"Urutan document_id {dataset_name} berubah "
            "setelah assignment diterapkan."
        )

    return result


# ============================================================
# VALIDASI DATASET HASIL SPLIT
# ============================================================

def validate_split_dataset(
    dataframe: pd.DataFrame,
    dataset_name: str,
    expected_rows: int,
    expected_splits: set[str],
) -> None:
    """
    Memastikan dataset hasil split lengkap dan valid.
    """

    if len(dataframe) != expected_rows:
        raise ValueError(
            f"Jumlah data {dataset_name} berubah.\n"
            f"Seharusnya: {expected_rows:,}\n"
            f"Ditemukan : {len(dataframe):,}"
        )

    actual_splits = set(
        dataframe["split"]
        .astype(str)
        .unique()
    )

    if actual_splits != expected_splits:
        raise ValueError(
            f"Jenis split pada {dataset_name} "
            f"tidak sesuai.\n"
            f"Seharusnya: {expected_splits}\n"
            f"Ditemukan : {actual_splits}"
        )

    duplicated_ids = int(
        dataframe["document_id"]
        .duplicated()
        .sum()
    )

    if duplicated_ids > 0:
        raise ValueError(
            f"Dataset {dataset_name} memiliki "
            f"{duplicated_ids:,} document_id duplikat."
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
            f"{empty_text:,} teks kosong."
        )

    invalid_word_count = int(
        pd.to_numeric(
            dataframe["word_count"],
            errors="coerce",
        )
        .fillna(0)
        .le(0)
        .sum()
    )

    if invalid_word_count > 0:
        raise ValueError(
            f"Dataset {dataset_name} memiliki "
            f"{invalid_word_count:,} word_count "
            "tidak valid."
        )


# ============================================================
# VALIDASI KESELARASAN SPLIT ANTARSKENARIO
# ============================================================

def validate_split_alignment(
    reference_dataframe: pd.DataFrame,
    comparison_dataframe: pd.DataFrame,
    comparison_name: str,
) -> None:
    """
    Memastikan skenario dalam dataset yang sama memperoleh
    assignment split yang identik.
    """

    comparison_columns = [
        "document_id",
        "category",
        "split",
    ]

    if (
        "class_index" in reference_dataframe.columns
        and "class_index" in comparison_dataframe.columns
    ):
        comparison_columns.append(
            "class_index"
        )

    reference = (
        reference_dataframe[
            comparison_columns
        ]
        .sort_values(
            "document_id",
            kind="stable",
        )
        .reset_index(drop=True)
        .astype(str)
    )

    comparison = (
        comparison_dataframe[
            comparison_columns
        ]
        .sort_values(
            "document_id",
            kind="stable",
        )
        .reset_index(drop=True)
        .astype(str)
    )

    if not reference.equals(
        comparison
    ):
        raise ValueError(
            f"Assignment split {comparison_name} "
            "tidak sama dengan skenario referensi."
        )


# ============================================================
# MEMBUAT LAPORAN JUMLAH DATA
# ============================================================

def create_split_report(
    dataframes: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Membuat laporan jumlah dan persentase setiap split.
    """

    report_rows: list[dict] = []

    for dataframe in dataframes:
        dataset_name = str(
            dataframe["dataset"].iloc[0]
        )

        scenario_code = str(
            dataframe["scenario_code"].iloc[0]
        )

        scenario_name = str(
            dataframe["scenario_name"].iloc[0]
        )

        uses_yake = dataframe[
            "uses_yake"
        ].iloc[0]

        includes_content = dataframe[
            "includes_content"
        ].iloc[0]

        yake_comparison_role = str(
            dataframe[
                "yake_comparison_role"
            ].iloc[0]
        )

        comparison_group = str(
            dataframe[
                "comparison_group"
            ].iloc[0]
        )

        total_data = len(dataframe)

        split_counts = (
            dataframe["split"]
            .value_counts()
        )

        preferred_order = [
            "train",
            "validation",
            "test",
        ]

        for split_name in preferred_order:
            if split_name not in split_counts:
                continue

            count = int(
                split_counts[
                    split_name
                ]
            )

            report_rows.append(
                {
                    "dataset": dataset_name,
                    "scenario_code": (
                        scenario_code
                    ),
                    "scenario_name": (
                        scenario_name
                    ),
                    "uses_yake": uses_yake,
                    "includes_content": (
                        includes_content
                    ),
                    "yake_comparison_role": (
                        yake_comparison_role
                    ),
                    "comparison_group": (
                        comparison_group
                    ),
                    "split": split_name,
                    "jumlah_data": count,
                    "persentase": round(
                        count
                        / total_data
                        * 100,
                        4,
                    ),
                }
            )

    return pd.DataFrame(
        report_rows
    )


# ============================================================
# MEMBUAT LAPORAN DISTRIBUSI KATEGORI
# ============================================================

def create_category_split_report(
    dataframes: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Membuat laporan distribusi kategori pada setiap split.
    """

    report_rows: list[dict] = []

    for dataframe in dataframes:
        dataset_name = str(
            dataframe["dataset"].iloc[0]
        )

        scenario_code = str(
            dataframe[
                "scenario_code"
            ].iloc[0]
        )

        scenario_name = str(
            dataframe[
                "scenario_name"
            ].iloc[0]
        )

        uses_yake = dataframe[
            "uses_yake"
        ].iloc[0]

        grouped = (
            dataframe
            .groupby(
                [
                    "split",
                    "category",
                ],
                dropna=False,
            )
            .size()
            .reset_index(
                name="jumlah_data"
            )
        )

        for row in grouped.itertuples(
            index=False
        ):
            split_total = int(
                (
                    dataframe["split"]
                    == row.split
                ).sum()
            )

            report_rows.append(
                {
                    "dataset": dataset_name,
                    "scenario_code": (
                        scenario_code
                    ),
                    "scenario_name": (
                        scenario_name
                    ),
                    "uses_yake": uses_yake,
                    "split": row.split,
                    "category": row.category,
                    "jumlah_data": int(
                        row.jumlah_data
                    ),
                    "persentase_dalam_split": round(
                        row.jumlah_data
                        / split_total
                        * 100,
                        4,
                    ),
                }
            )

    return pd.DataFrame(
        report_rows
    )


# ============================================================
# MENYIMPAN KONFIGURASI
# ============================================================

def save_split_configuration() -> None:
    """
    Menyimpan konfigurasi split agar eksperimen dapat
    direproduksi.
    """

    configuration = {
        "random_seed": RANDOM_SEED,
        "shuffle": True,
        "stratified_split": True,
        "stratify_column": "category",
        "assignment_key": "document_id",
        "kompas": {
            "expected_rows": (
                EXPECTED_ROW_COUNTS[
                    "Kompas"
                ]
            ),
            "scenarios": [
                "K1",
                "K2",
                "K3",
            ],
            "train": KOMPAS_TRAIN_SIZE,
            "validation": (
                KOMPAS_VALIDATION_SIZE
            ),
            "test": KOMPAS_TEST_SIZE,
            "same_assignment_for_all_scenarios": (
                True
            ),
        },
        "ag_news": {
            "clean_train_rows": (
                EXPECTED_ROW_COUNTS[
                    "AG News Train"
                ]
            ),
            "official_test_rows": (
                EXPECTED_ROW_COUNTS[
                    "AG News Test"
                ]
            ),
            "scenarios": [
                "A1",
                "A2",
            ],
            "clean_train_partition": {
                "train": AGNEWS_TRAIN_SIZE,
                "validation": (
                    AGNEWS_VALIDATION_SIZE
                ),
            },
            "official_test_partition": {
                "test": 1.0,
                "modified": False,
            },
            "same_assignment_for_all_scenarios": (
                True
            ),
        },
        "yake_ablation": {
            "baseline_without_yake": "K2",
            "treatment_with_yake": "K3",
            "same_document_assignment": True,
        },
        "data_leakage_control": {
            "assignment_created_once_per_dataset": (
                True
            ),
            "split_sets_are_disjoint": True,
            "official_agnews_test_preserved": True,
        },
        "k4_used": False,
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
# MENGHAPUS OUTPUT K4 LAMA
# ============================================================

def remove_legacy_k4_split_output() -> None:
    """
    Menghapus output split K4 lama agar tidak terbaca oleh
    tahap modeling atau dashboard.
    """

    if LEGACY_KOMPAS_K4_SPLIT_PATH.exists():
        LEGACY_KOMPAS_K4_SPLIT_PATH.unlink()

        print(
            "\nOutput K4 lama dihapus:"
        )

        print(
            LEGACY_KOMPAS_K4_SPLIT_PATH
        )


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main() -> None:
    """
    Menjalankan pembagian dataset Kompas dan AG News.
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

    remove_legacy_k4_split_output()

    # --------------------------------------------------------
    # MEMUAT KOMPAS
    # --------------------------------------------------------

    kompas_k1 = load_scenario(
        file_path=KOMPAS_K1_PATH,
        expected_dataset="Kompas",
        expected_scenario_code="K1",
        expected_rows=EXPECTED_ROW_COUNTS[
            "Kompas"
        ],
    )

    kompas_k2 = load_scenario(
        file_path=KOMPAS_K2_PATH,
        expected_dataset="Kompas",
        expected_scenario_code="K2",
        expected_rows=EXPECTED_ROW_COUNTS[
            "Kompas"
        ],
    )

    kompas_k3 = load_scenario(
        file_path=KOMPAS_K3_PATH,
        expected_dataset="Kompas",
        expected_scenario_code="K3",
        expected_rows=EXPECTED_ROW_COUNTS[
            "Kompas"
        ],
    )

    validate_scenario_alignment(
        reference_dataframe=kompas_k1,
        comparison_dataframe=kompas_k2,
        comparison_name="Kompas K2",
    )

    validate_scenario_alignment(
        reference_dataframe=kompas_k1,
        comparison_dataframe=kompas_k3,
        comparison_name="Kompas K3",
    )

    # --------------------------------------------------------
    # MEMUAT AG NEWS
    # --------------------------------------------------------

    agnews_train_a1 = load_scenario(
        file_path=AGNEWS_TRAIN_A1_PATH,
        expected_dataset="AG News Train",
        expected_scenario_code="A1",
        expected_rows=EXPECTED_ROW_COUNTS[
            "AG News Train"
        ],
    )

    agnews_train_a2 = load_scenario(
        file_path=AGNEWS_TRAIN_A2_PATH,
        expected_dataset="AG News Train",
        expected_scenario_code="A2",
        expected_rows=EXPECTED_ROW_COUNTS[
            "AG News Train"
        ],
    )

    agnews_test_a1 = load_scenario(
        file_path=AGNEWS_TEST_A1_PATH,
        expected_dataset="AG News Test",
        expected_scenario_code="A1",
        expected_rows=EXPECTED_ROW_COUNTS[
            "AG News Test"
        ],
    )

    agnews_test_a2 = load_scenario(
        file_path=AGNEWS_TEST_A2_PATH,
        expected_dataset="AG News Test",
        expected_scenario_code="A2",
        expected_rows=EXPECTED_ROW_COUNTS[
            "AG News Test"
        ],
    )

    validate_scenario_alignment(
        reference_dataframe=agnews_train_a1,
        comparison_dataframe=agnews_train_a2,
        comparison_name="AG News Train A2",
    )

    validate_scenario_alignment(
        reference_dataframe=agnews_test_a1,
        comparison_dataframe=agnews_test_a2,
        comparison_name="AG News Test A2",
    )

    # --------------------------------------------------------
    # MEMBUAT ASSIGNMENT
    # --------------------------------------------------------

    print(
        "\nMembagi dataset Kompas "
        "80% train, 10% validation, 10% test..."
    )

    kompas_assignment = (
        create_kompas_split_assignment(
            kompas_k1
        )
    )

    print(
        "Membagi AG News clean train menjadi "
        "90% train dan 10% validation..."
    )

    print(
        "AG News official test tetap "
        "dipertahankan sebagai final test..."
    )

    agnews_assignment = (
        create_agnews_split_assignment(
            train_dataframe=agnews_train_a1,
            test_dataframe=agnews_test_a1,
        )
    )

    validate_split_assignment(
        assignment=kompas_assignment,
        assignment_name="Kompas",
        expected_rows=EXPECTED_ROW_COUNTS[
            "Kompas"
        ],
        expected_splits={
            "train",
            "validation",
            "test",
        },
    )

    validate_split_assignment(
        assignment=agnews_assignment,
        assignment_name="AG News",
        expected_rows=(
            EXPECTED_ROW_COUNTS[
                "AG News Train"
            ]
            + EXPECTED_ROW_COUNTS[
                "AG News Test"
            ]
        ),
        expected_splits={
            "train",
            "validation",
            "test",
        },
    )

    # --------------------------------------------------------
    # MENERAPKAN SPLIT KOMPAS
    # --------------------------------------------------------

    kompas_k1_split = apply_split_assignment(
        scenario_dataframe=kompas_k1,
        split_assignment=kompas_assignment,
        dataset_name="Kompas K1",
    )

    kompas_k2_split = apply_split_assignment(
        scenario_dataframe=kompas_k2,
        split_assignment=kompas_assignment,
        dataset_name="Kompas K2",
    )

    kompas_k3_split = apply_split_assignment(
        scenario_dataframe=kompas_k3,
        split_assignment=kompas_assignment,
        dataset_name="Kompas K3",
    )

    # --------------------------------------------------------
    # MEMISAHKAN ASSIGNMENT AG NEWS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MENERAPKAN SPLIT AG NEWS
    # --------------------------------------------------------

    agnews_a1_train_validation = (
        apply_split_assignment(
            scenario_dataframe=(
                agnews_train_a1
            ),
            split_assignment=(
                agnews_train_assignment
            ),
            dataset_name=(
                "AG News A1 Train Validation"
            ),
        )
    )

    agnews_a2_train_validation = (
        apply_split_assignment(
            scenario_dataframe=(
                agnews_train_a2
            ),
            split_assignment=(
                agnews_train_assignment
            ),
            dataset_name=(
                "AG News A2 Train Validation"
            ),
        )
    )

    agnews_a1_test = (
        apply_split_assignment(
            scenario_dataframe=(
                agnews_test_a1
            ),
            split_assignment=(
                agnews_test_assignment
            ),
            dataset_name=(
                "AG News A1 Test"
            ),
        )
    )

    agnews_a2_test = (
        apply_split_assignment(
            scenario_dataframe=(
                agnews_test_a2
            ),
            split_assignment=(
                agnews_test_assignment
            ),
            dataset_name=(
                "AG News A2 Test"
            ),
        )
    )

    # --------------------------------------------------------
    # VALIDASI HASIL KOMPAS
    # --------------------------------------------------------

    kompas_split_datasets = [
        kompas_k1_split,
        kompas_k2_split,
        kompas_k3_split,
    ]

    for dataframe in kompas_split_datasets:
        scenario_code = str(
            dataframe[
                "scenario_code"
            ].iloc[0]
        )

        validate_split_dataset(
            dataframe=dataframe,
            dataset_name=(
                f"Kompas {scenario_code}"
            ),
            expected_rows=EXPECTED_ROW_COUNTS[
                "Kompas"
            ],
            expected_splits={
                "train",
                "validation",
                "test",
            },
        )

    validate_split_alignment(
        reference_dataframe=kompas_k1_split,
        comparison_dataframe=kompas_k2_split,
        comparison_name="Kompas K2",
    )

    validate_split_alignment(
        reference_dataframe=kompas_k1_split,
        comparison_dataframe=kompas_k3_split,
        comparison_name="Kompas K3",
    )

    # --------------------------------------------------------
    # VALIDASI HASIL AG NEWS
    # --------------------------------------------------------

    validate_split_dataset(
        dataframe=agnews_a1_train_validation,
        dataset_name=(
            "AG News A1 Train Validation"
        ),
        expected_rows=EXPECTED_ROW_COUNTS[
            "AG News Train"
        ],
        expected_splits={
            "train",
            "validation",
        },
    )

    validate_split_dataset(
        dataframe=agnews_a2_train_validation,
        dataset_name=(
            "AG News A2 Train Validation"
        ),
        expected_rows=EXPECTED_ROW_COUNTS[
            "AG News Train"
        ],
        expected_splits={
            "train",
            "validation",
        },
    )

    validate_split_dataset(
        dataframe=agnews_a1_test,
        dataset_name="AG News A1 Test",
        expected_rows=EXPECTED_ROW_COUNTS[
            "AG News Test"
        ],
        expected_splits={"test"},
    )

    validate_split_dataset(
        dataframe=agnews_a2_test,
        dataset_name="AG News A2 Test",
        expected_rows=EXPECTED_ROW_COUNTS[
            "AG News Test"
        ],
        expected_splits={"test"},
    )

    validate_split_alignment(
        reference_dataframe=(
            agnews_a1_train_validation
        ),
        comparison_dataframe=(
            agnews_a2_train_validation
        ),
        comparison_name=(
            "AG News Train Validation A2"
        ),
    )

    validate_split_alignment(
        reference_dataframe=(
            agnews_a1_test
        ),
        comparison_dataframe=(
            agnews_a2_test
        ),
        comparison_name="AG News Test A2",
    )

    if len(agnews_a1_test) != 7_600:
        raise AssertionError(
            "AG News official test tidak lagi "
            "berjumlah 7.600."
        )

    # --------------------------------------------------------
    # MENYIMPAN DATASET SPLIT
    # --------------------------------------------------------

    output_mapping = {
        KOMPAS_K1_SPLIT_PATH: (
            kompas_k1_split
        ),
        KOMPAS_K2_SPLIT_PATH: (
            kompas_k2_split
        ),
        KOMPAS_K3_SPLIT_PATH: (
            kompas_k3_split
        ),
        AGNEWS_A1_TRAIN_VALIDATION_PATH: (
            agnews_a1_train_validation
        ),
        AGNEWS_A2_TRAIN_VALIDATION_PATH: (
            agnews_a2_train_validation
        ),
        AGNEWS_A1_TEST_PATH: (
            agnews_a1_test
        ),
        AGNEWS_A2_TEST_PATH: (
            agnews_a2_test
        ),
    }

    for output_path, dataframe in (
        output_mapping.items()
    ):
        dataframe.to_csv(
            output_path,
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

    # --------------------------------------------------------
    # MEMBUAT LAPORAN
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MENAMPILKAN HASIL
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("HASIL DATASET SPLIT")
    print("=" * 72)

    print(
        split_report.to_string(
            index=False
        )
    )

    print(
        "\nDistribusi kategori Kompas K1:"
    )

    kompas_k1_category_report = (
        category_report[
            (
                category_report["dataset"]
                == "Kompas"
            )
            & (
                category_report[
                    "scenario_code"
                ]
                == "K1"
            )
        ]
    )

    print(
        kompas_k1_category_report
        .to_string(index=False)
    )

    print("\nRingkasan assignment:")

    print(
        "\nKompas:"
    )

    print(
        kompas_assignment["split"]
        .value_counts()
        .reindex(
            [
                "train",
                "validation",
                "test",
            ]
        )
        .to_string()
    )

    print(
        "\nAG News:"
    )

    print(
        agnews_assignment["split"]
        .value_counts()
        .reindex(
            [
                "train",
                "validation",
                "test",
            ]
        )
        .to_string()
    )

    print("\nValidasi perbandingan YAKE:")

    print(
        "K2 tanpa YAKE dan K3 dengan YAKE "
        "menggunakan assignment dokumen yang sama."
    )

    # --------------------------------------------------------
    # INFORMASI OUTPUT
    # --------------------------------------------------------

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
    print(
        DATASET_SPLIT_CATEGORY_REPORT_PATH
    )

    print("\nKonfigurasi split:")
    print(
        DATASET_SPLIT_CONFIGURATION_PATH
    )

    print(
        "\nTahap dataset split selesai."
    )


if __name__ == "__main__":
    main()
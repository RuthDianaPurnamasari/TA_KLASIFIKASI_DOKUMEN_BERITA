from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd


# ============================================================
# ROOT PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config import (  # noqa: E402
    AG_NEWS_TEST_PROCESSED_PATH,
    AG_NEWS_TRAIN_PROCESSED_PATH,
    KOMPAS_PROCESSED_PATH,
    PROCESSED_DATA_DIR,
    TABLES_DIR,
)


# ============================================================
# KEBIJAKAN KONFLIK LABEL
# ============================================================

# Pilihan:
# 1. "review_required"
#    - Konflik label hanya dilaporkan.
#    - Program berhenti sebelum menghasilkan dataset clean.
#
# 2. "remove_all_conflicts"
#    - Seluruh baris dalam kelompok teks identik
#      yang memiliki label berbeda dikeluarkan.
#
# Gunakan "review_required" sampai 22 kelompok konflik
# selesai diperiksa dan keputusan metodologis dikunci.

LABEL_CONFLICT_POLICY = "remove_all_conflicts"

ALLOWED_LABEL_CONFLICT_POLICIES = {
    "review_required",
    "remove_all_conflicts",
}


# ============================================================
# PATH OUTPUT DATASET CLEAN
# ============================================================

KOMPAS_CLEAN_PATH = (
    PROCESSED_DATA_DIR
    / "kompas_clean.csv"
)

AG_NEWS_TRAIN_CLEAN_PATH = (
    PROCESSED_DATA_DIR
    / "ag_news_train_clean.csv"
)

AG_NEWS_TEST_CLEAN_PATH = (
    PROCESSED_DATA_DIR
    / "ag_news_test_clean.csv"
)


# ============================================================
# PATH OUTPUT LAPORAN
# ============================================================

DATA_CLEANING_REPORT_PATH = (
    TABLES_DIR
    / "data_cleaning_report.csv"
)

REMOVED_DUPLICATES_PATH = (
    TABLES_DIR
    / "removed_duplicates.csv"
)

REMOVED_TRAIN_TEST_OVERLAP_PATH = (
    TABLES_DIR
    / "removed_agnews_train_test_overlap.csv"
)

# Laporan untuk proses peninjauan konflik.
# Tidak menimpa laporan audit dari 02_prepare_agnews.py.
AGNEWS_CONFLICT_REVIEW_PATH = (
    TABLES_DIR
    / "ag_news_label_conflicts_review.csv"
)

# Hanya berisi konflik yang benar-benar dikeluarkan.
REMOVED_AGNEWS_LABEL_CONFLICTS_PATH = (
    TABLES_DIR
    / "removed_ag_news_label_conflicts.csv"
)

CLASS_DISTRIBUTION_AFTER_CLEANING_PATH = (
    TABLES_DIR
    / "class_distribution_after_cleaning.csv"
)


# ============================================================
# KONSTANTA DATASET
# ============================================================

KOMPAS_REQUIRED_COLUMNS = [
    "title",
    "description",
    "content",
    "date",
    "category",
    "link",
]

AGNEWS_REQUIRED_COLUMNS = [
    "document_id",
    "source_row",
    "class_index",
    "category",
    "title",
    "description",
    "split",
]

EXPECTED_AGNEWS_ROW_COUNTS = {
    "train": 120_000,
    "test": 7_600,
}

EXPECTED_AGNEWS_CLASS_COUNTS = {
    "train": {
        1: 30_000,
        2: 30_000,
        3: 30_000,
        4: 30_000,
    },
    "test": {
        1: 1_900,
        2: 1_900,
        3: 1_900,
        4: 1_900,
    },
}


# ============================================================
# VALIDASI KEBIJAKAN
# ============================================================

def validate_label_conflict_policy() -> None:
    """
    Memastikan kebijakan konflik label valid.
    """

    if (
        LABEL_CONFLICT_POLICY
        not in ALLOWED_LABEL_CONFLICT_POLICIES
    ):
        raise ValueError(
            "LABEL_CONFLICT_POLICY tidak valid.\n"
            f"Nilai saat ini: {LABEL_CONFLICT_POLICY}\n"
            "Pilihan yang tersedia: "
            f"{sorted(ALLOWED_LABEL_CONFLICT_POLICIES)}"
        )


# ============================================================
# MEMBACA DATASET
# ============================================================

def load_dataset(
    file_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Membaca dataset CSV hasil tahap preparation.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset {dataset_name} tidak ditemukan:\n"
            f"{file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Path dataset {dataset_name} bukan file:\n"
            f"{file_path}"
        )

    dataframe = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
    )

    if dataframe.empty:
        raise ValueError(
            f"Dataset {dataset_name} kosong."
        )

    return dataframe


# ============================================================
# VALIDASI KOLOM
# ============================================================

def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:
    """
    Memastikan seluruh kolom wajib tersedia.
    """

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} tidak memiliki kolom wajib: "
            f"{missing_columns}\n"
            f"Kolom tersedia: {list(dataframe.columns)}"
        )


# ============================================================
# NORMALISASI TEKS TEKNIS
# ============================================================

def standardize_text(value: object) -> str:
    """
    Merapikan whitespace tanpa melakukan preprocessing NLP.

    Kapitalisasi, tanda baca, dan isi semantik teks
    tetap dipertahankan.
    """

    if pd.isna(value):
        return ""

    text = str(value)

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_for_duplicate(value: object) -> str:
    """
    Membuat representasi teks untuk pemeriksaan duplikasi.

    Normalisasi ini hanya digunakan sebagai kunci audit,
    bukan sebagai teks input model.
    """

    text = standardize_text(value)

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.casefold()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def standardize_text_columns(
    dataframe: pd.DataFrame,
    text_columns: list[str],
) -> pd.DataFrame:
    """
    Merapikan whitespace pada beberapa kolom teks.
    """

    dataframe = dataframe.copy()

    for column in text_columns:
        if column not in dataframe.columns:
            raise ValueError(
                f"Kolom '{column}' tidak ditemukan."
            )

        dataframe[column] = (
            dataframe[column]
            .apply(standardize_text)
        )

    return dataframe


# ============================================================
# VALIDASI TEKS WAJIB
# ============================================================

def count_empty_text(
    series: pd.Series,
) -> int:
    """
    Menghitung nilai teks kosong.
    """

    return int(
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )


def validate_required_text(
    dataframe: pd.DataFrame,
    dataset_name: str,
    required_text_columns: list[str],
) -> None:
    """
    Memastikan kolom teks wajib tidak kosong.
    """

    empty_summary: dict[str, int] = {}

    for column in required_text_columns:
        empty_count = count_empty_text(
            dataframe[column]
        )

        if empty_count > 0:
            empty_summary[column] = empty_count

    if empty_summary:
        raise ValueError(
            f"{dataset_name} memiliki teks wajib kosong: "
            f"{empty_summary}"
        )


# ============================================================
# VALIDASI INPUT AG NEWS
# ============================================================

def validate_agnews_input(
    dataframe: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """
    Memastikan dataset processed AG News berasal dari
    preparation yang benar.
    """

    dataframe = dataframe.copy()

    dataframe["class_index"] = pd.to_numeric(
        dataframe["class_index"],
        errors="coerce",
    ).astype("Int64")

    invalid_class_index = int(
        dataframe["class_index"]
        .isna()
        .sum()
    )

    if invalid_class_index > 0:
        raise ValueError(
            f"AG News {split_name} memiliki "
            f"{invalid_class_index:,} Class Index tidak valid."
        )

    expected_count = (
        EXPECTED_AGNEWS_ROW_COUNTS[
            split_name
        ]
    )

    if len(dataframe) != expected_count:
        raise ValueError(
            f"Jumlah AG News {split_name} tidak sesuai.\n"
            f"Seharusnya: {expected_count:,}\n"
            f"Ditemukan : {len(dataframe):,}"
        )

    actual_distribution = (
        dataframe["class_index"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    expected_distribution = (
        EXPECTED_AGNEWS_CLASS_COUNTS[
            split_name
        ]
    )

    if actual_distribution != expected_distribution:
        raise ValueError(
            f"Distribusi kelas AG News {split_name} "
            "tidak sesuai.\n"
            f"Seharusnya: {expected_distribution}\n"
            f"Ditemukan : {actual_distribution}"
        )

    expected_split_value = split_name

    actual_split_values = set(
        dataframe["split"]
        .dropna()
        .astype(str)
        .str.lower()
        .unique()
        .tolist()
    )

    if actual_split_values != {
        expected_split_value
    }:
        raise ValueError(
            f"Kolom split AG News {split_name} "
            f"tidak sesuai: {actual_split_values}"
        )

    if dataframe["document_id"].duplicated().any():
        raise ValueError(
            f"AG News {split_name} memiliki "
            "document_id duplikat."
        )

    if dataframe["source_row"].duplicated().any():
        raise ValueError(
            f"AG News {split_name} memiliki "
            "source_row duplikat."
        )

    return dataframe


# ============================================================
# PENGURUTAN DETERMINISTIK
# ============================================================

def sort_kompas_deterministically(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mengurutkan Kompas secara konsisten sebelum deduplikasi.
    """

    data = dataframe.copy()

    data["_sort_date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    )

    sort_columns = [
        "_sort_date",
    ]

    if "document_id" in data.columns:
        sort_columns.append(
            "document_id"
        )

    sort_columns.append(
        "link"
    )

    data = (
        data
        .sort_values(
            sort_columns,
            kind="stable",
            na_position="last",
        )
        .drop(
            columns=["_sort_date"]
        )
        .reset_index(drop=True)
    )

    return data


def sort_agnews_deterministically(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mengurutkan AG News berdasarkan identitas asli.
    """

    data = dataframe.copy()

    sort_columns = [
        "source_row",
        "document_id",
    ]

    return (
        data
        .sort_values(
            sort_columns,
            kind="stable",
            na_position="last",
        )
        .reset_index(drop=True)
    )


# ============================================================
# KUNCI TEKS AG NEWS
# ============================================================

def create_agnews_text_key(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Membuat kunci gabungan title dan description.
    """

    normalized_title = (
        dataframe["title"]
        .apply(normalize_for_duplicate)
    )

    normalized_description = (
        dataframe["description"]
        .apply(normalize_for_duplicate)
    )

    return (
        normalized_title
        + " || "
        + normalized_description
    )


def add_agnews_text_key(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Menambahkan kunci teks sementara.
    """

    data = dataframe.copy()

    data["_text_key"] = (
        create_agnews_text_key(data)
    )

    return data


# ============================================================
# AUDIT KONFLIK LABEL AG NEWS
# ============================================================

def find_agnews_label_conflicts(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Menemukan teks identik yang memiliki label berbeda.
    """

    data = add_agnews_text_key(
        dataframe
    )

    label_count = (
        data
        .groupby(
            "_text_key",
            dropna=False,
        )["class_index"]
        .transform("nunique")
    )

    conflicts = data.loc[
        label_count > 1
    ].copy()

    if conflicts.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "conflict_group",
                "group_size",
                "jumlah_label",
                "document_id",
                "source_row",
                "class_index",
                "category",
                "title",
                "description",
                "review_status",
            ]
        )

    conflicts["group_size"] = (
        conflicts
        .groupby(
            "_text_key",
            dropna=False,
        )["document_id"]
        .transform("size")
    )

    conflicts["jumlah_label"] = (
        conflicts
        .groupby(
            "_text_key",
            dropna=False,
        )["class_index"]
        .transform("nunique")
    )

    conflicts["conflict_group"] = (
        conflicts
        .groupby(
            "_text_key",
            dropna=False,
            sort=True,
        )
        .ngroup()
        + 1
    )

    conflicts["dataset"] = (
        dataset_name
    )

    conflicts["review_status"] = (
        "pending_review"
    )

    conflicts = conflicts[
        [
            "dataset",
            "conflict_group",
            "group_size",
            "jumlah_label",
            "document_id",
            "source_row",
            "class_index",
            "category",
            "title",
            "description",
            "review_status",
        ]
    ].sort_values(
        by=[
            "conflict_group",
            "class_index",
            "source_row",
        ],
        kind="stable",
    )

    return conflicts.reset_index(
        drop=True
    )


# ============================================================
# AUDIT DUPLIKASI AG NEWS
# ============================================================

def find_agnews_internal_duplicate_rows(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Menemukan seluruh anggota kelompok duplikasi
    dengan teks dan label yang sama.
    """

    data = sort_agnews_deterministically(
        dataframe
    )

    data = add_agnews_text_key(
        data
    )

    duplicate_subset = [
        "class_index",
        "_text_key",
    ]

    duplicate_mask = (
        data
        .duplicated(
            subset=duplicate_subset,
            keep=False,
        )
        & data["_text_key"].ne(" || ")
    )

    duplicates = data.loc[
        duplicate_mask
    ].copy()

    if duplicates.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "duplicate_group",
                "group_size",
                "document_id",
                "source_row",
                "class_index",
                "category",
                "title",
                "description",
            ]
        )

    duplicates["group_size"] = (
        duplicates
        .groupby(
            duplicate_subset,
            dropna=False,
        )["document_id"]
        .transform("size")
    )

    duplicates["duplicate_group"] = (
        duplicates
        .groupby(
            duplicate_subset,
            dropna=False,
            sort=True,
        )
        .ngroup()
        + 1
    )

    duplicates["dataset"] = (
        dataset_name
    )

    duplicates = duplicates[
        [
            "dataset",
            "duplicate_group",
            "group_size",
            "document_id",
            "source_row",
            "class_index",
            "category",
            "title",
            "description",
        ]
    ].sort_values(
        by=[
            "duplicate_group",
            "source_row",
        ],
        kind="stable",
    )

    return duplicates.reset_index(
        drop=True
    )


# ============================================================
# CLEANING DUPLIKASI KOMPAS
# ============================================================

def clean_kompas_duplicates(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Menghapus duplikasi Kompas berdasarkan:
    1. content;
    2. gabungan title dan description;
    3. link.

    Baris pertama berdasarkan pengurutan deterministik
    dipertahankan.
    """

    data = sort_kompas_deterministically(
        dataframe
    )

    data["_content_key"] = (
        data["content"]
        .apply(normalize_for_duplicate)
    )

    data["_title_description_key"] = (
        data["title"]
        .apply(normalize_for_duplicate)
        + " || "
        + data["description"]
        .apply(normalize_for_duplicate)
    )

    data["_link_key"] = (
        data["link"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    duplicate_content = (
        data
        .duplicated(
            subset=["_content_key"],
            keep="first",
        )
        & data["_content_key"].ne("")
    )

    duplicate_title_description = (
        data
        .duplicated(
            subset=[
                "_title_description_key",
            ],
            keep="first",
        )
        & data[
            "_title_description_key"
        ].ne(" || ")
    )

    duplicate_link = (
        data
        .duplicated(
            subset=["_link_key"],
            keep="first",
        )
        & data["_link_key"].ne("")
    )

    duplicate_mask = (
        duplicate_content
        | duplicate_title_description
        | duplicate_link
    )

    removed_duplicates = data.loc[
        duplicate_mask
    ].copy()

    cleaned_dataframe = data.loc[
        ~duplicate_mask
    ].copy()

    removed_duplicates[
        "duplicate_by_content"
    ] = duplicate_content.loc[
        duplicate_mask
    ].to_numpy()

    removed_duplicates[
        "duplicate_by_title_description"
    ] = duplicate_title_description.loc[
        duplicate_mask
    ].to_numpy()

    removed_duplicates[
        "duplicate_by_link"
    ] = duplicate_link.loc[
        duplicate_mask
    ].to_numpy()

    removed_duplicates[
        "duplicate_type"
    ] = "kompas_duplicate"

    removed_duplicates[
        "dataset"
    ] = "Kompas"

    removed_duplicates[
        "removal_reason"
    ] = "duplicate_document"

    temporary_columns = [
        "_content_key",
        "_title_description_key",
        "_link_key",
    ]

    cleaned_dataframe = (
        cleaned_dataframe
        .drop(
            columns=temporary_columns,
        )
        .reset_index(drop=True)
    )

    removed_duplicates = (
        removed_duplicates
        .drop(
            columns=temporary_columns,
        )
        .reset_index(drop=True)
    )

    return (
        cleaned_dataframe,
        removed_duplicates,
    )


# ============================================================
# MENGHAPUS KONFLIK LABEL AG NEWS
# ============================================================

def remove_agnews_label_conflicts(
    dataframe: pd.DataFrame,
    conflict_report: pd.DataFrame,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Menghapus seluruh kelompok konflik label.

    Fungsi ini hanya dijalankan apabila kebijakan
    LABEL_CONFLICT_POLICY adalah remove_all_conflicts.
    """

    if conflict_report.empty:
        empty_removed = dataframe.iloc[
            0:0
        ].copy()

        empty_removed["dataset"] = (
            pd.Series(dtype="string")
        )

        empty_removed["removal_reason"] = (
            pd.Series(dtype="string")
        )

        return (
            dataframe.copy(),
            empty_removed,
        )

    data = add_agnews_text_key(
        dataframe
    )

    conflict_keys = set(
        create_agnews_text_key(
            conflict_report
        )
    )

    conflict_mask = (
        data["_text_key"]
        .isin(conflict_keys)
    )

    removed_conflicts = data.loc[
        conflict_mask
    ].copy()

    cleaned_dataframe = data.loc[
        ~conflict_mask
    ].copy()

    removed_conflicts[
        "dataset"
    ] = dataset_name

    removed_conflicts[
        "removal_reason"
    ] = "unresolved_label_conflict"

    cleaned_dataframe = (
        cleaned_dataframe
        .drop(
            columns=["_text_key"],
        )
        .reset_index(drop=True)
    )

    removed_conflicts = (
        removed_conflicts
        .drop(
            columns=["_text_key"],
        )
        .reset_index(drop=True)
    )

    return (
        cleaned_dataframe,
        removed_conflicts,
    )


# ============================================================
# MENGHAPUS DUPLIKASI INTERNAL AG NEWS
# ============================================================

def remove_agnews_internal_duplicates(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Menghapus duplikasi dengan:
    - Class Index sama;
    - title-description sama setelah normalisasi.

    Baris dengan source_row terkecil dipertahankan.
    """

    data = sort_agnews_deterministically(
        dataframe
    )

    data = add_agnews_text_key(
        data
    )

    duplicate_subset = [
        "class_index",
        "_text_key",
    ]

    data["_duplicate_group_size"] = (
        data
        .groupby(
            duplicate_subset,
            dropna=False,
        )["document_id"]
        .transform("size")
    )

    data["_kept_document_id"] = (
        data
        .groupby(
            duplicate_subset,
            dropna=False,
        )["document_id"]
        .transform("first")
    )

    duplicate_mask = (
        data
        .duplicated(
            subset=duplicate_subset,
            keep="first",
        )
        & data["_text_key"].ne(" || ")
    )

    removed_duplicates = data.loc[
        duplicate_mask
    ].copy()

    cleaned_dataframe = data.loc[
        ~duplicate_mask
    ].copy()

    removed_duplicates[
        "duplicate_type"
    ] = "normalized_title_description"

    removed_duplicates[
        "dataset"
    ] = dataset_name

    removed_duplicates[
        "removal_reason"
    ] = "internal_duplicate"

    removed_duplicates[
        "duplicate_group_size"
    ] = removed_duplicates[
        "_duplicate_group_size"
    ]

    removed_duplicates[
        "kept_document_id"
    ] = removed_duplicates[
        "_kept_document_id"
    ]

    temporary_columns = [
        "_text_key",
        "_duplicate_group_size",
        "_kept_document_id",
    ]

    cleaned_dataframe = (
        cleaned_dataframe
        .drop(
            columns=temporary_columns,
        )
        .reset_index(drop=True)
    )

    removed_duplicates = (
        removed_duplicates
        .drop(
            columns=temporary_columns,
        )
        .reset_index(drop=True)
    )

    return (
        cleaned_dataframe,
        removed_duplicates,
    )


# ============================================================
# MENGHAPUS OVERLAP DARI TRAIN
# ============================================================

def remove_agnews_train_test_overlap(
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Menghapus dokumen dari train apabila teks identik
    terdapat pada test.

    Dataset test tidak diubah.
    """

    train_data = add_agnews_text_key(
        train_dataframe
    )

    test_data = add_agnews_text_key(
        test_dataframe
    )

    test_lookup = (
        test_data[
            [
                "_text_key",
                "document_id",
                "source_row",
                "class_index",
                "category",
            ]
        ]
        .drop_duplicates(
            subset=["_text_key"],
            keep="first",
        )
        .rename(
            columns={
                "document_id": (
                    "matched_test_document_id"
                ),
                "source_row": (
                    "matched_test_source_row"
                ),
                "class_index": (
                    "matched_test_class_index"
                ),
                "category": (
                    "matched_test_category"
                ),
            }
        )
    )

    train_data = train_data.merge(
        test_lookup,
        on="_text_key",
        how="left",
    )

    overlap_mask = (
        train_data[
            "matched_test_document_id"
        ].notna()
        & train_data["_text_key"].ne(" || ")
    )

    removed_overlap = train_data.loc[
        overlap_mask
    ].copy()

    cleaned_train = train_data.loc[
        ~overlap_mask
    ].copy()

    removed_overlap[
        "dataset"
    ] = "AG News Train"

    removed_overlap[
        "removal_reason"
    ] = "train_test_text_overlap"

    temporary_columns = [
        "_text_key",
        "matched_test_document_id",
        "matched_test_source_row",
        "matched_test_class_index",
        "matched_test_category",
    ]

    cleaned_train = (
        cleaned_train
        .drop(
            columns=temporary_columns,
        )
        .reset_index(drop=True)
    )

    # Metadata pasangan test dipertahankan
    # pada laporan removed_overlap.
    removed_overlap = (
        removed_overlap
        .drop(
            columns=["_text_key"],
        )
        .reset_index(drop=True)
    )

    return (
        cleaned_train,
        removed_overlap,
    )


# ============================================================
# VALIDASI HASIL KOMPAS
# ============================================================

def validate_no_remaining_kompas_duplicates(
    dataframe: pd.DataFrame,
) -> None:
    """
    Memastikan tidak ada duplikasi Kompas tersisa.
    """

    content_key = (
        dataframe["content"]
        .apply(normalize_for_duplicate)
    )

    title_description_key = (
        dataframe["title"]
        .apply(normalize_for_duplicate)
        + " || "
        + dataframe["description"]
        .apply(normalize_for_duplicate)
    )

    link_key = (
        dataframe["link"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    duplicate_content = (
        content_key.ne("")
        & content_key.duplicated(
            keep=False
        )
    )

    duplicate_title_description = (
        title_description_key.ne(" || ")
        & title_description_key.duplicated(
            keep=False
        )
    )

    duplicate_link = (
        link_key.ne("")
        & link_key.duplicated(
            keep=False
        )
    )

    if (
        duplicate_content.any()
        or duplicate_title_description.any()
        or duplicate_link.any()
    ):
        raise AssertionError(
            "Masih ditemukan duplikat Kompas "
            "setelah cleaning."
        )


# ============================================================
# VALIDASI HASIL AG NEWS
# ============================================================

def validate_no_remaining_agnews_duplicates(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Memastikan tidak ada duplikasi dengan teks
    dan label yang sama.
    """

    data = add_agnews_text_key(
        dataframe
    )

    duplicate_mask = (
        data["_text_key"].ne(" || ")
        & data.duplicated(
            subset=[
                "class_index",
                "_text_key",
            ],
            keep=False,
        )
    )

    if duplicate_mask.any():
        raise AssertionError(
            f"Masih ditemukan duplikat internal "
            f"pada {dataset_name}."
        )


def validate_no_remaining_agnews_conflicts(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Memastikan tidak ada teks identik dengan label berbeda.
    """

    conflicts = find_agnews_label_conflicts(
        dataframe,
        dataset_name,
    )

    if not conflicts.empty:
        raise AssertionError(
            f"Masih ditemukan {len(conflicts):,} "
            f"baris konflik label pada {dataset_name}."
        )


def validate_no_train_test_overlap(
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> None:
    """
    Memastikan train dan test tidak memiliki
    title-description identik.
    """

    train_keys = set(
        create_agnews_text_key(
            train_dataframe
        )
    )

    test_keys = set(
        create_agnews_text_key(
            test_dataframe
        )
    )

    train_keys.discard(" || ")
    test_keys.discard(" || ")

    remaining_overlap = (
        train_keys
        .intersection(test_keys)
    )

    if remaining_overlap:
        raise AssertionError(
            "Masih ditemukan overlap AG News "
            f"train-test sebanyak "
            f"{len(remaining_overlap):,} key."
        )


# ============================================================
# VALIDASI KONSISTENSI JUMLAH
# ============================================================

def validate_count_consistency(
    original_count: int,
    cleaned_count: int,
    duplicate_count: int,
    conflict_count: int,
    overlap_count: int,
    dataset_name: str,
) -> None:
    """
    Memastikan seluruh pengurangan data tercatat.
    """

    reconstructed_count = (
        cleaned_count
        + duplicate_count
        + conflict_count
        + overlap_count
    )

    if original_count != reconstructed_count:
        raise AssertionError(
            f"Jumlah {dataset_name} tidak konsisten.\n"
            f"Jumlah awal        : {original_count:,}\n"
            f"Jumlah akhir       : {cleaned_count:,}\n"
            f"Duplikat dihapus   : {duplicate_count:,}\n"
            f"Konflik dihapus    : {conflict_count:,}\n"
            f"Overlap dihapus    : {overlap_count:,}\n"
            f"Hasil rekonstruksi : {reconstructed_count:,}"
        )


# ============================================================
# LAPORAN DATA CLEANING
# ============================================================

def create_cleaning_report_row(
    dataset_name: str,
    original_count: int,
    cleaned_count: int,
    duplicates_removed: int,
    label_conflicts_removed: int,
    overlap_removed: int,
    empty_title_after_cleaning: int,
    empty_description_after_cleaning: int,
    empty_content_after_cleaning: int | None,
) -> dict:
    """
    Membuat satu baris laporan cleaning.
    """

    total_removed = (
        duplicates_removed
        + label_conflicts_removed
        + overlap_removed
    )

    removal_percentage = (
        total_removed
        / original_count
        * 100
        if original_count > 0
        else 0.0
    )

    return {
        "dataset": dataset_name,
        "label_conflict_policy": (
            LABEL_CONFLICT_POLICY
        ),
        "jumlah_data_awal": (
            original_count
        ),
        "duplikat_dihapus": (
            duplicates_removed
        ),
        "konflik_label_dihapus": (
            label_conflicts_removed
        ),
        "overlap_dihapus": (
            overlap_removed
        ),
        "total_data_dihapus": (
            total_removed
        ),
        "persentase_data_dihapus": (
            round(
                removal_percentage,
                6,
            )
        ),
        "jumlah_data_akhir": (
            cleaned_count
        ),
        "title_kosong_setelah_cleaning": (
            empty_title_after_cleaning
        ),
        "description_kosong_setelah_cleaning": (
            empty_description_after_cleaning
        ),
        "content_kosong_setelah_cleaning": (
            empty_content_after_cleaning
        ),
    }


# ============================================================
# LAPORAN DISTRIBUSI KELAS
# ============================================================

def build_class_distribution_report(
    kompas_clean: pd.DataFrame,
    agnews_train_clean: pd.DataFrame,
    agnews_test_clean: pd.DataFrame,
) -> pd.DataFrame:
    """
    Membuat distribusi kelas setelah cleaning.
    """

    reports: list[pd.DataFrame] = []

    kompas_distribution = (
        kompas_clean["category"]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "class_label"
        )
        .reset_index(
            name="jumlah_data"
        )
    )

    kompas_distribution[
        "dataset"
    ] = "Kompas"

    reports.append(
        kompas_distribution
    )

    for dataset_name, dataframe in [
        (
            "AG News Train",
            agnews_train_clean,
        ),
        (
            "AG News Test",
            agnews_test_clean,
        ),
    ]:
        distribution = (
            dataframe["class_index"]
            .value_counts(
                dropna=False
            )
            .sort_index()
            .rename_axis(
                "class_label"
            )
            .reset_index(
                name="jumlah_data"
            )
        )

        distribution[
            "dataset"
        ] = dataset_name

        reports.append(
            distribution
        )

    report = pd.concat(
        reports,
        ignore_index=True,
        sort=False,
    )

    return report[
        [
            "dataset",
            "class_label",
            "jumlah_data",
        ]
    ]


# ============================================================
# PENGGABUNGAN DATAFRAME LAPORAN
# ============================================================

def concat_non_empty(
    dataframes: list[pd.DataFrame],
    empty_columns: list[str],
) -> pd.DataFrame:
    """
    Menggabungkan laporan tanpa FutureWarning.
    """

    non_empty_dataframes = [
        dataframe
        for dataframe in dataframes
        if not dataframe.empty
    ]

    if not non_empty_dataframes:
        return pd.DataFrame(
            columns=empty_columns
        )

    return pd.concat(
        non_empty_dataframes,
        ignore_index=True,
        sort=False,
    )


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main() -> None:
    """
    Menjalankan data cleaning Kompas dan AG News.
    """

    print("=" * 80)
    print("STEP 4.1 - DATA CLEANING")
    print("=" * 80)

    validate_label_conflict_policy()

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # MEMBACA DATASET
    # --------------------------------------------------------

    kompas = load_dataset(
        KOMPAS_PROCESSED_PATH,
        "Kompas",
    )

    agnews_train = load_dataset(
        AG_NEWS_TRAIN_PROCESSED_PATH,
        "AG News Train",
    )

    agnews_test = load_dataset(
        AG_NEWS_TEST_PROCESSED_PATH,
        "AG News Test",
    )

    # --------------------------------------------------------
    # VALIDASI KOLOM
    # --------------------------------------------------------

    validate_required_columns(
        kompas,
        KOMPAS_REQUIRED_COLUMNS,
        "Kompas",
    )

    validate_required_columns(
        agnews_train,
        AGNEWS_REQUIRED_COLUMNS,
        "AG News Train",
    )

    validate_required_columns(
        agnews_test,
        AGNEWS_REQUIRED_COLUMNS,
        "AG News Test",
    )

    # --------------------------------------------------------
    # NORMALISASI TEKNIS
    # --------------------------------------------------------

    kompas = standardize_text_columns(
        kompas,
        [
            "title",
            "description",
            "content",
        ],
    )

    agnews_train = standardize_text_columns(
        agnews_train,
        [
            "title",
            "description",
        ],
    )

    agnews_test = standardize_text_columns(
        agnews_test,
        [
            "title",
            "description",
        ],
    )

    agnews_train = validate_agnews_input(
        agnews_train,
        "train",
    )

    agnews_test = validate_agnews_input(
        agnews_test,
        "test",
    )

    validate_required_text(
        kompas,
        "Kompas",
        ["title"],
    )

    validate_required_text(
        agnews_train,
        "AG News Train",
        ["title"],
    )

    validate_required_text(
        agnews_test,
        "AG News Test",
        ["title"],
    )

    # --------------------------------------------------------
    # MENYIMPAN JUMLAH AWAL
    # --------------------------------------------------------

    kompas_original_count = len(
        kompas
    )

    agnews_train_original_count = len(
        agnews_train
    )

    agnews_test_original_count = len(
        agnews_test
    )

    # --------------------------------------------------------
    # AUDIT KONFLIK LABEL
    # --------------------------------------------------------

    agnews_train_conflicts = (
        find_agnews_label_conflicts(
            agnews_train,
            "AG News Train",
        )
    )

    agnews_test_conflicts = (
        find_agnews_label_conflicts(
            agnews_test,
            "AG News Test",
        )
    )

    conflict_review = concat_non_empty(
        [
            agnews_train_conflicts,
            agnews_test_conflicts,
        ],
        empty_columns=[
            "dataset",
            "conflict_group",
            "group_size",
            "jumlah_label",
            "document_id",
            "source_row",
            "class_index",
            "category",
            "title",
            "description",
            "review_status",
        ],
    )

    conflict_review.to_csv(
        AGNEWS_CONFLICT_REVIEW_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # VALIDASI DATA TEST RESMI
    # --------------------------------------------------------

    agnews_test_duplicates = (
        find_agnews_internal_duplicate_rows(
            agnews_test,
            "AG News Test",
        )
    )

    if not agnews_test_conflicts.empty:
        raise RuntimeError(
            "Dataset test resmi memiliki konflik label. "
            "Cleaning dihentikan agar test tidak diubah.\n"
            f"Periksa: {AGNEWS_CONFLICT_REVIEW_PATH}"
        )

    if not agnews_test_duplicates.empty:
        raise RuntimeError(
            "Dataset test resmi memiliki duplikasi internal. "
            "Cleaning dihentikan agar test tidak diubah."
        )

    # --------------------------------------------------------
    # KEPUTUSAN KONFLIK LABEL TRAIN
    # --------------------------------------------------------

    if (
        not agnews_train_conflicts.empty
        and LABEL_CONFLICT_POLICY
        == "review_required"
    ):
        conflict_groups = int(
            agnews_train_conflicts[
                "conflict_group"
            ].nunique()
        )

        print("\n" + "=" * 80)
        print("CLEANING DIHENTIKAN UNTUK REVIEW KONFLIK LABEL")
        print("=" * 80)

        print(
            f"Baris konflik ditemukan   : "
            f"{len(agnews_train_conflicts):,}"
        )

        print(
            f"Kelompok konflik ditemukan: "
            f"{conflict_groups:,}"
        )

        print(
            "\nLaporan konflik tersimpan di:"
        )

        print(
            AGNEWS_CONFLICT_REVIEW_PATH
        )

        print(
            "\nPeriksa setiap kelompok konflik sebelum "
            "menetapkan keputusan metodologis."
        )

        print(
            "\nJika keputusan akhirnya adalah mengeluarkan "
            "seluruh kelompok ambigu, ubah:"
        )

        print(
            'LABEL_CONFLICT_POLICY = "remove_all_conflicts"'
        )

        raise RuntimeError(
            "Review konflik label belum diselesaikan."
        )

    # --------------------------------------------------------
    # CLEANING KOMPAS
    # --------------------------------------------------------

    (
        kompas_clean,
        kompas_removed_duplicates,
    ) = clean_kompas_duplicates(
        kompas
    )

    # --------------------------------------------------------
    # CLEANING KONFLIK LABEL AG NEWS
    # --------------------------------------------------------

    if (
        LABEL_CONFLICT_POLICY
        == "remove_all_conflicts"
    ):
        (
            agnews_train_without_conflicts,
            removed_train_conflicts,
        ) = remove_agnews_label_conflicts(
            agnews_train,
            agnews_train_conflicts,
            "AG News Train",
        )
    else:
        agnews_train_without_conflicts = (
            agnews_train.copy()
        )

        removed_train_conflicts = (
            agnews_train.iloc[0:0]
            .copy()
        )

        removed_train_conflicts[
            "dataset"
        ] = pd.Series(
            dtype="string"
        )

        removed_train_conflicts[
            "removal_reason"
        ] = pd.Series(
            dtype="string"
        )

    # Test dipertahankan.
    agnews_test_clean = (
        agnews_test.copy()
        .reset_index(drop=True)
    )

    removed_test_conflicts = (
        agnews_test.iloc[0:0]
        .copy()
    )

    removed_test_conflicts[
        "dataset"
    ] = pd.Series(
        dtype="string"
    )

    removed_test_conflicts[
        "removal_reason"
    ] = pd.Series(
        dtype="string"
    )

    # --------------------------------------------------------
    # CLEANING DUPLIKASI INTERNAL TRAIN
    # --------------------------------------------------------

    (
        agnews_train_deduplicated,
        agnews_train_removed_duplicates,
    ) = remove_agnews_internal_duplicates(
        agnews_train_without_conflicts,
        "AG News Train",
    )

    # --------------------------------------------------------
    # MENGHAPUS OVERLAP DARI TRAIN
    # --------------------------------------------------------

    (
        agnews_train_clean,
        removed_overlap,
    ) = remove_agnews_train_test_overlap(
        agnews_train_deduplicated,
        agnews_test_clean,
    )

    # --------------------------------------------------------
    # VALIDASI HASIL AKHIR
    # --------------------------------------------------------

    validate_count_consistency(
        original_count=kompas_original_count,
        cleaned_count=len(
            kompas_clean
        ),
        duplicate_count=len(
            kompas_removed_duplicates
        ),
        conflict_count=0,
        overlap_count=0,
        dataset_name="Kompas",
    )

    validate_count_consistency(
        original_count=agnews_train_original_count,
        cleaned_count=len(
            agnews_train_clean
        ),
        duplicate_count=len(
            agnews_train_removed_duplicates
        ),
        conflict_count=len(
            removed_train_conflicts
        ),
        overlap_count=len(
            removed_overlap
        ),
        dataset_name="AG News Train",
    )

    validate_count_consistency(
        original_count=agnews_test_original_count,
        cleaned_count=len(
            agnews_test_clean
        ),
        duplicate_count=0,
        conflict_count=0,
        overlap_count=0,
        dataset_name="AG News Test",
    )

    validate_no_remaining_kompas_duplicates(
        kompas_clean
    )

    validate_no_remaining_agnews_duplicates(
        agnews_train_clean,
        "AG News Train",
    )

    validate_no_remaining_agnews_conflicts(
        agnews_train_clean,
        "AG News Train",
    )

    validate_no_remaining_agnews_duplicates(
        agnews_test_clean,
        "AG News Test",
    )

    validate_no_remaining_agnews_conflicts(
        agnews_test_clean,
        "AG News Test",
    )

    validate_no_train_test_overlap(
        agnews_train_clean,
        agnews_test_clean,
    )

    if len(agnews_test_clean) != 7_600:
        raise AssertionError(
            "Jumlah AG News test berubah. "
            "Dataset test harus tetap 7.600 baris."
        )

    # --------------------------------------------------------
    # MENYUSUN LAPORAN
    # --------------------------------------------------------

    removed_duplicates = (
        concat_non_empty(
            [
                kompas_removed_duplicates,
                agnews_train_removed_duplicates,
            ],
            empty_columns=[
                "dataset",
                "document_id",
                "source_row",
                "class_index",
                "category",
                "title",
                "description",
                "duplicate_type",
                "removal_reason",
            ],
        )
    )

    removed_conflicts = (
        concat_non_empty(
            [
                removed_train_conflicts,
                removed_test_conflicts,
            ],
            empty_columns=[
                "dataset",
                "document_id",
                "source_row",
                "class_index",
                "category",
                "title",
                "description",
                "removal_reason",
            ],
        )
    )

    cleaning_report = pd.DataFrame(
        [
            create_cleaning_report_row(
                dataset_name="Kompas",
                original_count=kompas_original_count,
                cleaned_count=len(
                    kompas_clean
                ),
                duplicates_removed=len(
                    kompas_removed_duplicates
                ),
                label_conflicts_removed=0,
                overlap_removed=0,
                empty_title_after_cleaning=(
                    count_empty_text(
                        kompas_clean["title"]
                    )
                ),
                empty_description_after_cleaning=(
                    count_empty_text(
                        kompas_clean["description"]
                    )
                ),
                empty_content_after_cleaning=(
                    count_empty_text(
                        kompas_clean["content"]
                    )
                ),
            ),
            create_cleaning_report_row(
                dataset_name="AG News Train",
                original_count=(
                    agnews_train_original_count
                ),
                cleaned_count=len(
                    agnews_train_clean
                ),
                duplicates_removed=len(
                    agnews_train_removed_duplicates
                ),
                label_conflicts_removed=len(
                    removed_train_conflicts
                ),
                overlap_removed=len(
                    removed_overlap
                ),
                empty_title_after_cleaning=(
                    count_empty_text(
                        agnews_train_clean[
                            "title"
                        ]
                    )
                ),
                empty_description_after_cleaning=(
                    count_empty_text(
                        agnews_train_clean[
                            "description"
                        ]
                    )
                ),
                empty_content_after_cleaning=None,
            ),
            create_cleaning_report_row(
                dataset_name="AG News Test",
                original_count=(
                    agnews_test_original_count
                ),
                cleaned_count=len(
                    agnews_test_clean
                ),
                duplicates_removed=0,
                label_conflicts_removed=0,
                overlap_removed=0,
                empty_title_after_cleaning=(
                    count_empty_text(
                        agnews_test_clean[
                            "title"
                        ]
                    )
                ),
                empty_description_after_cleaning=(
                    count_empty_text(
                        agnews_test_clean[
                            "description"
                        ]
                    )
                ),
                empty_content_after_cleaning=None,
            ),
        ]
    )

    class_distribution_report = (
        build_class_distribution_report(
            kompas_clean,
            agnews_train_clean,
            agnews_test_clean,
        )
    )

    # --------------------------------------------------------
    # MENYIMPAN OUTPUT
    # --------------------------------------------------------

    kompas_clean.to_csv(
        KOMPAS_CLEAN_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    agnews_train_clean.to_csv(
        AG_NEWS_TRAIN_CLEAN_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    agnews_test_clean.to_csv(
        AG_NEWS_TEST_CLEAN_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    cleaning_report.to_csv(
        DATA_CLEANING_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    removed_duplicates.to_csv(
        REMOVED_DUPLICATES_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    removed_overlap.to_csv(
        REMOVED_TRAIN_TEST_OVERLAP_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    removed_conflicts.to_csv(
        REMOVED_AGNEWS_LABEL_CONFLICTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    class_distribution_report.to_csv(
        CLASS_DISTRIBUTION_AFTER_CLEANING_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # MENAMPILKAN HASIL
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("HASIL DATA CLEANING")
    print("=" * 80)

    print(
        cleaning_report.to_string(
            index=False
        )
    )

    print("\nDetail penghapusan:")

    print(
        "Kompas - duplikat dihapus          : "
        f"{len(kompas_removed_duplicates):,}"
    )

    print(
        "AG News Train - konflik dihapus    : "
        f"{len(removed_train_conflicts):,}"
    )

    print(
        "AG News Train - duplikat dihapus   : "
        f"{len(agnews_train_removed_duplicates):,}"
    )

    print(
        "AG News Train - overlap dihapus    : "
        f"{len(removed_overlap):,}"
    )

    print(
        "AG News Test - jumlah akhir        : "
        f"{len(agnews_test_clean):,}"
    )

    print(
        "\nDistribusi kelas setelah cleaning:"
    )

    print(
        class_distribution_report.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("OUTPUT DATA CLEANING")
    print("=" * 80)

    print("\nDataset Kompas clean:")
    print(KOMPAS_CLEAN_PATH)

    print("\nDataset AG News Train clean:")
    print(AG_NEWS_TRAIN_CLEAN_PATH)

    print("\nDataset AG News Test clean:")
    print(AG_NEWS_TEST_CLEAN_PATH)

    print("\nLaporan data cleaning:")
    print(DATA_CLEANING_REPORT_PATH)

    print("\nLog duplikat yang dihapus:")
    print(REMOVED_DUPLICATES_PATH)

    print("\nLog konflik label untuk review:")
    print(AGNEWS_CONFLICT_REVIEW_PATH)

    print("\nLog konflik label yang dihapus:")
    print(REMOVED_AGNEWS_LABEL_CONFLICTS_PATH)

    print("\nLog overlap yang dihapus dari train:")
    print(REMOVED_TRAIN_TEST_OVERLAP_PATH)

    print("\nDistribusi kelas setelah cleaning:")
    print(CLASS_DISTRIBUTION_AFTER_CLEANING_PATH)

    print("\nTahap data cleaning selesai.")


if __name__ == "__main__":
    main()
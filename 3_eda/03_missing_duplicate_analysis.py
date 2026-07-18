from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


# =============================================================================
# PROJECT PATH
# =============================================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# =============================================================================
# IMPORT PROJECT CONFIGURATION
# =============================================================================

from config import (  # noqa: E402
    AG_NEWS_TEST_PROCESSED_PATH,
    AG_NEWS_TRAIN_PROCESSED_PATH,
    KOMPAS_PROCESSED_PATH,
    TABLES_DIR,
)


# =============================================================================
# DATA DIRECTORY
# =============================================================================

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "2_data"
    / "processed"
)


# =============================================================================
# FINAL DATASET CANDIDATES
# =============================================================================

KOMPAS_FINAL_CANDIDATES = [
    PROCESSED_DATA_DIR / "kompas_clean.csv",
    PROCESSED_DATA_DIR / "kompas_cleaned.csv",
    PROCESSED_DATA_DIR / "kompas_final.csv",
]

AGNEWS_TRAIN_FINAL_CANDIDATES = [
    PROCESSED_DATA_DIR / "ag_news_train_clean.csv",
    PROCESSED_DATA_DIR / "agnews_train_clean.csv",
    PROCESSED_DATA_DIR / "ag_news_train_final.csv",
]

AGNEWS_TEST_FINAL_CANDIDATES = [
    PROCESSED_DATA_DIR / "ag_news_test_clean.csv",
    PROCESSED_DATA_DIR / "agnews_test_clean.csv",
    PROCESSED_DATA_DIR / "ag_news_test_final.csv",
]


# =============================================================================
# OUTPUT FILES
# =============================================================================

MISSING_VALUE_REPORT_PATH = (
    TABLES_DIR
    / "missing_value_report.csv"
)

DUPLICATE_REPORT_PATH = (
    TABLES_DIR
    / "duplicate_report.csv"
)

KOMPAS_DUPLICATE_DETAIL_PATH = (
    TABLES_DIR
    / "kompas_duplicate_detail.csv"
)

AGNEWS_TRAIN_DUPLICATE_DETAIL_PATH = (
    TABLES_DIR
    / "agnews_train_duplicate_detail.csv"
)

AGNEWS_TEST_DUPLICATE_DETAIL_PATH = (
    TABLES_DIR
    / "agnews_test_duplicate_detail.csv"
)

AGNEWS_OVERLAP_REPORT_PATH = (
    TABLES_DIR
    / "agnews_train_test_overlap.csv"
)

CLEANING_INTEGRITY_PATH = (
    TABLES_DIR
    / "cleaning_integrity_summary.csv"
)


# =============================================================================
# EXPECTED DATASET COUNTS
# =============================================================================

EXPECTED_INITIAL_COUNTS = {
    "kompas": 10_000,
    "ag_news_train": 120_000,
    "ag_news_test": 7_600,
}

EXPECTED_FINAL_COUNTS = {
    "kompas": 9_997,
    "ag_news_train": 119_817,
    "ag_news_test": 7_600,
}

VALIDATE_EXPECTED_COUNTS = True


# =============================================================================
# REQUIRED COLUMNS
# =============================================================================

REQUIRED_KOMPAS_COLUMNS = {
    "document_id",
    "title",
    "description",
    "content",
    "category",
    "link",
}

REQUIRED_AGNEWS_COLUMNS = {
    "document_id",
    "class_index",
    "category",
    "title",
    "description",
}


# =============================================================================
# NORMALIZATION CONFIGURATION
# =============================================================================

WHITESPACE_PATTERN = re.compile(
    r"\s+"
)


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def ensure_output_directory() -> None:
    """
    Memastikan folder tabel tersedia.
    """

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def is_valid_file(
    file_path: Path,
) -> bool:
    """
    Memeriksa apakah path merupakan file yang valid.
    """

    path = Path(
        file_path
    )

    return (
        path.exists()
        and path.is_file()
        and path.stat().st_size > 0
    )


def resolve_first_existing_file(
    candidates: list[Path],
    dataset_name: str,
) -> Path:
    """
    Memilih file dataset final pertama yang tersedia.
    """

    for candidate in candidates:
        if is_valid_file(
            candidate
        ):
            return candidate

    paths = "\n".join(
        f"- {path}"
        for path in candidates
    )

    raise FileNotFoundError(
        f"Dataset final {dataset_name} tidak ditemukan.\n\n"
        f"Path yang diperiksa:\n{paths}\n\n"
        "Pastikan tahap cleaning sudah dijalankan."
    )


def read_csv_with_fallback(
    file_path: Path,
) -> pd.DataFrame:
    """
    Membaca CSV dengan beberapa kemungkinan encoding.
    """

    last_error: Exception | None = None

    for encoding in [
        "utf-8-sig",
        "utf-8",
        "latin-1",
    ]:
        try:
            return pd.read_csv(
                file_path,
                encoding=encoding,
            )

        except UnicodeDecodeError as error:
            last_error = error

        except Exception as error:
            last_error = error
            break

    if last_error is not None:
        raise last_error

    return pd.DataFrame()


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    """
    Memastikan kolom yang diperlukan tersedia.
    """

    missing_columns = (
        required_columns
        - set(
            dataframe.columns
        )
    )

    if missing_columns:
        raise KeyError(
            f"Dataset {dataset_name} tidak memiliki "
            "kolom wajib.\n"
            f"Kolom hilang: {sorted(missing_columns)}"
        )


def validate_dataset_count(
    dataframe: pd.DataFrame,
    dataset_key: str,
    stage: str,
) -> None:
    """
    Memvalidasi jumlah data sebelum dan setelah cleaning.
    """

    if not VALIDATE_EXPECTED_COUNTS:
        return

    if stage == "sebelum_cleaning":
        expected_mapping = (
            EXPECTED_INITIAL_COUNTS
        )
    else:
        expected_mapping = (
            EXPECTED_FINAL_COUNTS
        )

    expected_count = expected_mapping[
        dataset_key
    ]

    actual_count = int(
        len(
            dataframe
        )
    )

    if actual_count != expected_count:
        raise ValueError(
            f"Jumlah {dataset_key} {stage} tidak sesuai.\n"
            f"Expected : {expected_count:,}\n"
            f"Actual   : {actual_count:,}\n\n"
            "Pastikan file yang digunakan sudah benar."
        )


def validate_output_file(
    file_path: Path,
    description: str,
) -> None:
    """
    Memastikan output berhasil dibuat.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Output {description} tidak berhasil dibuat:\n"
            f"{file_path}"
        )

    if file_path.stat().st_size <= 0:
        raise ValueError(
            f"Output {description} kosong:\n"
            f"{file_path}"
        )


# =============================================================================
# DATASET LOADER
# =============================================================================

def load_dataset(
    file_path: Path,
    dataset_name: str,
    dataset_key: str,
    stage: str,
    required_columns: set[str],
) -> pd.DataFrame:
    """
    Membaca dan memvalidasi dataset.
    """

    path = Path(
        file_path
    )

    if not is_valid_file(
        path
    ):
        raise FileNotFoundError(
            f"Dataset {dataset_name} tidak ditemukan "
            f"atau kosong:\n{path}"
        )

    dataframe = read_csv_with_fallback(
        path
    )

    if dataframe.empty:
        raise ValueError(
            f"Dataset {dataset_name} kosong."
        )

    dataframe.columns = [
        str(column).strip()
        for column
        in dataframe.columns
    ]

    validate_required_columns(
        dataframe=dataframe,
        required_columns=required_columns,
        dataset_name=dataset_name,
    )

    validate_dataset_count(
        dataframe=dataframe,
        dataset_key=dataset_key,
        stage=stage,
    )

    return dataframe


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def normalize_text_value(
    value: Any,
) -> str:
    """
    Menormalisasi teks menggunakan definisi yang sama
    dengan proses cleaning:

    - Unicode NFKC;
    - lowercase;
    - menghapus spasi berlebih;
    - mempertahankan tanda baca.
    """

    if value is None:
        return ""

    try:
        if pd.isna(
            value
        ):
            return ""
    except (
        TypeError,
        ValueError,
    ):
        pass

    text = unicodedata.normalize(
        "NFKC",
        str(
            value
        ),
    )

    text = WHITESPACE_PATTERN.sub(
        " ",
        text,
    )

    return (
        text
        .strip()
        .lower()
    )


def normalize_text_series(
    series: pd.Series,
) -> pd.Series:
    """
    Menormalisasi seluruh nilai dalam Series.
    """

    return series.apply(
        normalize_text_value
    )


# =============================================================================
# MISSING VALUE REPORT
# =============================================================================

def create_missing_value_report(
    dataframe: pd.DataFrame,
    dataset_name: str,
    stage: str,
    source_path: Path,
) -> pd.DataFrame:
    """
    Membuat laporan missing value dan string kosong
    pada setiap kolom.
    """

    records: list[
        dict[str, Any]
    ] = []

    for column in dataframe.columns:
        missing_mask = (
            dataframe[
                column
            ].isna()
        )

        empty_mask = pd.Series(
            False,
            index=dataframe.index,
        )

        if (
            pd.api.types.is_object_dtype(
                dataframe[
                    column
                ]
            )
            or pd.api.types.is_string_dtype(
                dataframe[
                    column
                ]
            )
        ):
            empty_mask = (
                dataframe[
                    column
                ].notna()
                & dataframe[
                    column
                ]
                .astype(str)
                .str.strip()
                .eq("")
            )

        problem_mask = (
            missing_mask
            | empty_mask
        )

        records.append(
            {
                "dataset":
                    dataset_name,

                "stage":
                    stage,

                "source_path":
                    str(
                        Path(
                            source_path
                        ).resolve()
                    ),

                "column":
                    column,

                "missing_value":
                    int(
                        missing_mask.sum()
                    ),

                "empty_string":
                    int(
                        empty_mask.sum()
                    ),

                "total_problem":
                    int(
                        problem_mask.sum()
                    ),

                "percentage_problem":
                    round(
                        float(
                            problem_mask.mean()
                            * 100
                        ),
                        4,
                    ),
            }
        )

    return pd.DataFrame(
        records
    )


# =============================================================================
# DUPLICATE HELPERS
# =============================================================================

def duplicate_row_count(
    series: pd.Series,
) -> int:
    """
    Menghitung jumlah baris duplikat setelah
    kemunculan pertama.
    """

    valid_mask = (
        series
        .astype(str)
        .str.strip()
        .ne("")
    )

    return int(
        series[
            valid_mask
        ]
        .duplicated(
            keep="first"
        )
        .sum()
    )


def duplicate_group_count(
    series: pd.Series,
) -> int:
    """
    Menghitung jumlah kelompok nilai duplikat.
    """

    valid_series = series[
        series
        .astype(str)
        .str.strip()
        .ne("")
    ]

    return int(
        (
            valid_series
            .value_counts()
            > 1
        ).sum()
    )


# =============================================================================
# KOMPAS DUPLICATE ANALYSIS
# =============================================================================

def create_kompas_duplicate_detail(
    dataframe: pd.DataFrame,
    stage: str,
) -> pd.DataFrame:
    """
    Membuat detail duplikat Title, Content, dan Link.
    """

    detail_frames: list[
        pd.DataFrame
    ] = []

    duplicate_columns = [
        (
            "title_normalized",
            "normalized_title",
        ),
        (
            "content_normalized",
            "normalized_content",
        ),
        (
            "link_normalized",
            "normalized_link",
        ),
    ]

    for (
        duplicate_type,
        normalized_column,
    ) in duplicate_columns:

        duplicate_mask = (
            dataframe[
                normalized_column
            ].ne("")
            & dataframe[
                normalized_column
            ].duplicated(
                keep=False
            )
        )

        detail = dataframe.loc[
            duplicate_mask,
            [
                "document_id",
                "title",
                "category",
                "link",
                normalized_column,
            ],
        ].copy()

        if detail.empty:
            continue

        detail = detail.rename(
            columns={
                normalized_column:
                    "duplicate_key"
            }
        )

        detail.insert(
            0,
            "stage",
            stage,
        )

        detail.insert(
            1,
            "duplicate_type",
            duplicate_type,
        )

        detail_frames.append(
            detail
        )

    if not detail_frames:
        return pd.DataFrame(
            columns=[
                "stage",
                "duplicate_type",
                "document_id",
                "title",
                "category",
                "link",
                "duplicate_key",
            ]
        )

    return (
        pd.concat(
            detail_frames,
            ignore_index=True,
        )
        .drop_duplicates()
        .sort_values(
            [
                "stage",
                "duplicate_type",
                "duplicate_key",
                "category",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def analyze_kompas_duplicates(
    dataframe: pd.DataFrame,
    stage: str,
    source_path: Path,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
]:
    """
    Menganalisis duplikat dataset Kompas.
    """

    data = dataframe.copy()

    data[
        "normalized_title"
    ] = normalize_text_series(
        data[
            "title"
        ]
    )

    data[
        "normalized_content"
    ] = normalize_text_series(
        data[
            "content"
        ]
    )

    data[
        "normalized_link"
    ] = normalize_text_series(
        data[
            "link"
        ]
    )

    exact_duplicate_rows = int(
        data.duplicated(
            subset=[
                "title",
                "description",
                "content",
                "category",
                "link",
            ],
            keep="first",
        ).sum()
    )

    title_cross_category = (
        data.loc[
            data[
                "normalized_title"
            ].ne("")
        ]
        .groupby(
            "normalized_title"
        )[
            "category"
        ]
        .nunique()
    )

    report = {
        "dataset":
            "kompas",

        "stage":
            stage,

        "source_path":
            str(
                Path(
                    source_path
                ).resolve()
            ),

        "jumlah_data":
            int(
                len(
                    data
                )
            ),

        "duplikat_artikel_exact":
            exact_duplicate_rows,

        "duplikat_title_normalized":
            duplicate_row_count(
                data[
                    "normalized_title"
                ]
            ),

        "kelompok_title_duplikat":
            duplicate_group_count(
                data[
                    "normalized_title"
                ]
            ),

        "duplikat_content_normalized":
            duplicate_row_count(
                data[
                    "normalized_content"
                ]
            ),

        "kelompok_content_duplikat":
            duplicate_group_count(
                data[
                    "normalized_content"
                ]
            ),

        "duplikat_link_normalized":
            duplicate_row_count(
                data[
                    "normalized_link"
                ]
            ),

        "kelompok_link_duplikat":
            duplicate_group_count(
                data[
                    "normalized_link"
                ]
            ),

        "judul_muncul_lintas_kategori":
            int(
                (
                    title_cross_category
                    > 1
                ).sum()
            ),

        "kelompok_konflik_label":
            0,

        "baris_konflik_label":
            0,
    }

    detail = create_kompas_duplicate_detail(
        dataframe=data,
        stage=stage,
    )

    return (
        report,
        detail,
    )


# =============================================================================
# AG NEWS DUPLICATE AND LABEL-CONFLICT ANALYSIS
# =============================================================================

def prepare_agnews_duplicate_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Menambahkan normalisasi Title, Description,
    dan article key pada AG News.
    """

    data = dataframe.copy()

    data[
        "normalized_title"
    ] = normalize_text_series(
        data[
            "title"
        ]
    )

    data[
        "normalized_description"
    ] = normalize_text_series(
        data[
            "description"
        ]
    )

    data[
        "article_key"
    ] = (
        data[
            "normalized_title"
        ]
        + " || "
        + data[
            "normalized_description"
        ]
    )

    valid_article_mask = (
        data[
            "article_key"
        ]
        .str.replace(
            " || ",
            "",
            regex=False,
        )
        .str.strip()
        .ne("")
    )

    data.loc[
        ~valid_article_mask,
        "article_key",
    ] = ""

    return data


def get_agnews_conflict_keys(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Mengambil article key yang mempunyai lebih dari
    satu label class_index.
    """

    label_counts = (
        dataframe.loc[
            dataframe[
                "article_key"
            ].ne("")
        ]
        .groupby(
            "article_key"
        )[
            "class_index"
        ]
        .nunique()
    )

    return (
        label_counts[
            label_counts
            > 1
        ]
        .index
        .astype(str)
        .tolist()
    )


def create_agnews_duplicate_detail(
    dataframe: pd.DataFrame,
    stage: str,
) -> pd.DataFrame:
    """
    Membuat detail duplikat berlabel sama
    dan konflik label.
    """

    detail_frames: list[
        pd.DataFrame
    ] = []

    same_label_mask = (
        dataframe[
            "article_key"
        ].ne("")
        & dataframe.duplicated(
            subset=[
                "class_index",
                "article_key",
            ],
            keep=False,
        )
    )

    same_label_detail = dataframe.loc[
        same_label_mask,
        [
            "document_id",
            "class_index",
            "category",
            "title",
            "description",
            "article_key",
        ],
    ].copy()

    if not same_label_detail.empty:
        same_label_detail.insert(
            0,
            "stage",
            stage,
        )

        same_label_detail.insert(
            1,
            "duplicate_type",
            "same_label_duplicate",
        )

        same_label_detail = (
            same_label_detail.rename(
                columns={
                    "article_key":
                        "duplicate_key"
                }
            )
        )

        detail_frames.append(
            same_label_detail
        )

    conflict_keys = (
        get_agnews_conflict_keys(
            dataframe
        )
    )

    conflict_mask = (
        dataframe[
            "article_key"
        ].isin(
            conflict_keys
        )
    )

    conflict_detail = dataframe.loc[
        conflict_mask,
        [
            "document_id",
            "class_index",
            "category",
            "title",
            "description",
            "article_key",
        ],
    ].copy()

    if not conflict_detail.empty:
        conflict_detail.insert(
            0,
            "stage",
            stage,
        )

        conflict_detail.insert(
            1,
            "duplicate_type",
            "label_conflict",
        )

        conflict_detail = (
            conflict_detail.rename(
                columns={
                    "article_key":
                        "duplicate_key"
                }
            )
        )

        detail_frames.append(
            conflict_detail
        )

    if not detail_frames:
        return pd.DataFrame(
            columns=[
                "stage",
                "duplicate_type",
                "document_id",
                "class_index",
                "category",
                "title",
                "description",
                "duplicate_key",
            ]
        )

    return (
        pd.concat(
            detail_frames,
            ignore_index=True,
        )
        .drop_duplicates()
        .sort_values(
            [
                "stage",
                "duplicate_type",
                "duplicate_key",
                "class_index",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def analyze_agnews_duplicates(
    dataframe: pd.DataFrame,
    dataset_name: str,
    stage: str,
    source_path: Path,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
]:
    """
    Menganalisis duplikat dan konflik label AG News.
    """

    data = prepare_agnews_duplicate_data(
        dataframe
    )

    valid_article_mask = (
        data[
            "article_key"
        ].ne("")
    )

    exact_same_label_duplicate_rows = int(
        data.duplicated(
            subset=[
                "class_index",
                "title",
                "description",
            ],
            keep="first",
        ).sum()
    )

    normalized_same_label_duplicate_rows = int(
        data.loc[
            valid_article_mask
        ]
        .duplicated(
            subset=[
                "class_index",
                "article_key",
            ],
            keep="first",
        )
        .sum()
    )

    normalized_article_duplicate_rows = (
        duplicate_row_count(
            data[
                "article_key"
            ]
        )
    )

    conflict_keys = (
        get_agnews_conflict_keys(
            data
        )
    )

    conflict_row_count = int(
        data[
            "article_key"
        ]
        .isin(
            conflict_keys
        )
        .sum()
    )

    title_cross_category = (
        data.loc[
            data[
                "normalized_title"
            ].ne("")
        ]
        .groupby(
            "normalized_title"
        )[
            "class_index"
        ]
        .nunique()
    )

    report = {
        "dataset":
            dataset_name,

        "stage":
            stage,

        "source_path":
            str(
                Path(
                    source_path
                ).resolve()
            ),

        "jumlah_data":
            int(
                len(
                    data
                )
            ),

        "duplikat_artikel_exact":
            exact_same_label_duplicate_rows,

        "duplikat_artikel_normalized":
            normalized_same_label_duplicate_rows,

        "duplikat_article_key_tanpa_label":
            normalized_article_duplicate_rows,

        "duplikat_title_normalized":
            duplicate_row_count(
                data[
                    "normalized_title"
                ]
            ),

        "kelompok_title_duplikat":
            duplicate_group_count(
                data[
                    "normalized_title"
                ]
            ),

        "duplikat_content_normalized":
            0,

        "kelompok_content_duplikat":
            0,

        "duplikat_link_normalized":
            0,

        "kelompok_link_duplikat":
            0,

        "judul_muncul_lintas_kategori":
            int(
                (
                    title_cross_category
                    > 1
                ).sum()
            ),

        "kelompok_konflik_label":
            int(
                len(
                    conflict_keys
                )
            ),

        "baris_konflik_label":
            conflict_row_count,
    }

    detail = create_agnews_duplicate_detail(
        dataframe=data,
        stage=stage,
    )

    return (
        report,
        detail,
    )


# =============================================================================
# AG NEWS TRAIN-TEST OVERLAP
# =============================================================================

def analyze_agnews_train_test_overlap(
    train: pd.DataFrame,
    test: pd.DataFrame,
    stage: str,
) -> pd.DataFrame:
    """
    Menganalisis artikel yang identik antara
    AG News train dan test.

    Pencocokan menggunakan:
    - class_index;
    - normalized Title;
    - normalized Description.
    """

    train_data = prepare_agnews_duplicate_data(
        train
    )

    test_data = prepare_agnews_duplicate_data(
        test
    )

    train_keys = (
        train_data[
            [
                "class_index",
                "article_key",
                "document_id",
            ]
        ]
        .rename(
            columns={
                "document_id":
                    "train_document_id"
            }
        )
        .drop_duplicates(
            subset=[
                "class_index",
                "article_key",
            ],
            keep="first",
        )
    )

    test_unique = (
        test_data
        .drop_duplicates(
            subset=[
                "class_index",
                "article_key",
            ],
            keep="first",
        )
    )

    overlap = test_unique.merge(
        train_keys,
        on=[
            "class_index",
            "article_key",
        ],
        how="inner",
    )

    result = overlap[
        [
            "train_document_id",
            "document_id",
            "class_index",
            "category",
            "title",
            "description",
            "article_key",
        ]
    ].copy()

    result = result.rename(
        columns={
            "document_id":
                "test_document_id"
        }
    )

    result.insert(
        0,
        "stage",
        stage,
    )

    return (
        result
        .sort_values(
            [
                "stage",
                "class_index",
                "article_key",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# TERMINAL DISPLAY
# =============================================================================

def print_missing_summary(
    report: pd.DataFrame,
    dataset_name: str,
    stage: str,
) -> None:
    """
    Menampilkan ringkasan missing value.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"MISSING VALUE — "
        f"{dataset_name.upper()} — "
        f"{stage.upper()}"
    )

    print(
        "=" * 80
    )

    problem_rows = report[
        report[
            "total_problem"
        ]
        > 0
    ]

    if problem_rows.empty:
        print(
            "Tidak ditemukan missing value "
            "atau string kosong."
        )
        return

    print(
        problem_rows[
            [
                "column",
                "missing_value",
                "empty_string",
                "total_problem",
                "percentage_problem",
            ]
        ]
        .to_string(
            index=False
        )
    )


def print_duplicate_summary(
    report: dict[str, Any],
) -> None:
    """
    Menampilkan ringkasan analisis duplikat.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"DUPLIKAT — "
        f"{report['dataset'].upper()} — "
        f"{report['stage'].upper()}"
    )

    print(
        "=" * 80
    )

    excluded_columns = {
        "dataset",
        "stage",
        "source_path",
    }

    for key, value in report.items():
        if key in excluded_columns:
            continue

        label = (
            key
            .replace(
                "_",
                " ",
            )
            .title()
        )

        if isinstance(
            value,
            int,
        ):
            print(
                f"{label:<44}: "
                f"{value:,}"
            )
        else:
            print(
                f"{label:<44}: "
                f"{value}"
            )


# =============================================================================
# CLEANING INTEGRITY SUMMARY
# =============================================================================

def create_cleaning_integrity_summary(
    initial_duplicate_report: pd.DataFrame,
    final_duplicate_report: pd.DataFrame,
    overlap_initial_count: int,
    overlap_final_count: int,
) -> pd.DataFrame:
    """
    Membandingkan kondisi sebelum dan setelah cleaning.
    """

    records: list[
        dict[str, Any]
    ] = []

    for dataset_name in [
        "kompas",
        "ag_news_train",
        "ag_news_test",
    ]:
        before = initial_duplicate_report[
            initial_duplicate_report[
                "dataset"
            ].eq(
                dataset_name
            )
        ].iloc[
            0
        ]

        after = final_duplicate_report[
            final_duplicate_report[
                "dataset"
            ].eq(
                dataset_name
            )
        ].iloc[
            0
        ]

        if dataset_name == "kompas":
            primary_issue_column = (
                "duplikat_content_normalized"
            )
        else:
            primary_issue_column = (
                "duplikat_artikel_normalized"
            )

        records.append(
            {
                "dataset":
                    dataset_name,

                "jumlah_sebelum_cleaning":
                    int(
                        before[
                            "jumlah_data"
                        ]
                    ),

                "jumlah_setelah_cleaning":
                    int(
                        after[
                            "jumlah_data"
                        ]
                    ),

                "jumlah_dihapus":
                    int(
                        before[
                            "jumlah_data"
                        ]
                        - after[
                            "jumlah_data"
                        ]
                    ),

                "masalah_utama":
                    primary_issue_column,

                "jumlah_masalah_sebelum":
                    int(
                        before[
                            primary_issue_column
                        ]
                    ),

                "jumlah_masalah_setelah":
                    int(
                        after[
                            primary_issue_column
                        ]
                    ),

                "kelompok_konflik_sebelum":
                    int(
                        before[
                            "kelompok_konflik_label"
                        ]
                    ),

                "kelompok_konflik_setelah":
                    int(
                        after[
                            "kelompok_konflik_label"
                        ]
                    ),

                "baris_konflik_sebelum":
                    int(
                        before[
                            "baris_konflik_label"
                        ]
                    ),

                "baris_konflik_setelah":
                    int(
                        after[
                            "baris_konflik_label"
                        ]
                    ),

                "overlap_train_test_sebelum":
                    (
                        overlap_initial_count
                        if dataset_name
                        == "ag_news_train"
                        else 0
                    ),

                "overlap_train_test_setelah":
                    (
                        overlap_final_count
                        if dataset_name
                        == "ag_news_train"
                        else 0
                    ),
            }
        )

    return pd.DataFrame(
        records
    )


# =============================================================================
# FINAL INTEGRITY VALIDATION
# =============================================================================

def validate_final_integrity(
    duplicate_report_final: pd.DataFrame,
    overlap_final: pd.DataFrame,
) -> None:
    """
    Memastikan dataset final tidak lagi memiliki masalah
    yang menjadi dasar kebijakan cleaning.
    """

    kompas = duplicate_report_final[
        duplicate_report_final[
            "dataset"
        ].eq(
            "kompas"
        )
    ].iloc[
        0
    ]

    agnews_train = duplicate_report_final[
        duplicate_report_final[
            "dataset"
        ].eq(
            "ag_news_train"
        )
    ].iloc[
        0
    ]

    agnews_test = duplicate_report_final[
        duplicate_report_final[
            "dataset"
        ].eq(
            "ag_news_test"
        )
    ].iloc[
        0
    ]

    errors: list[str] = []

    if int(
        kompas[
            "duplikat_content_normalized"
        ]
    ) != 0:
        errors.append(
            "Kompas final masih memiliki "
            "duplikat content."
        )

    for label, row in [
        (
            "AG News Train",
            agnews_train,
        ),
        (
            "AG News Test",
            agnews_test,
        ),
    ]:
        if int(
            row[
                "duplikat_artikel_normalized"
            ]
        ) != 0:
            errors.append(
                f"{label} final masih memiliki "
                "duplikat artikel dengan label sama."
            )

        if int(
            row[
                "kelompok_konflik_label"
            ]
        ) != 0:
            errors.append(
                f"{label} final masih memiliki "
                "konflik label."
            )

    if not overlap_final.empty:
        errors.append(
            "AG News final masih memiliki "
            "overlap train-test."
        )

    if errors:
        raise ValueError(
            "Validasi integritas dataset final gagal:\n- "
            + "\n- ".join(
                errors
            )
        )


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main() -> None:
    """
    Menjalankan EDA tahap 3.3:

    - missing value;
    - empty string;
    - duplikat Kompas;
    - duplikat AG News;
    - konflik label;
    - overlap train-test;
    - validasi dataset final.
    """

    print(
        "=" * 80
    )

    print(
        "STEP 3.3 - MISSING VALUE AND DUPLICATE ANALYSIS"
    )

    print(
        "=" * 80
    )

    ensure_output_directory()

    # =========================================================================
    # RESOLVE FINAL DATASET PATHS
    # =========================================================================

    kompas_final_path = (
        resolve_first_existing_file(
            candidates=KOMPAS_FINAL_CANDIDATES,
            dataset_name="Kompas",
        )
    )

    agnews_train_final_path = (
        resolve_first_existing_file(
            candidates=AGNEWS_TRAIN_FINAL_CANDIDATES,
            dataset_name="AG News Train",
        )
    )

    agnews_test_final_path = (
        resolve_first_existing_file(
            candidates=AGNEWS_TEST_FINAL_CANDIDATES,
            dataset_name="AG News Test",
        )
    )

    print(
        "\nDataset final yang digunakan:"
    )

    print(
        f"Kompas        : "
        f"{kompas_final_path}"
    )

    print(
        f"AG News Train : "
        f"{agnews_train_final_path}"
    )

    print(
        f"AG News Test  : "
        f"{agnews_test_final_path}"
    )

    # =========================================================================
    # DATASET SPECIFICATION
    # =========================================================================

    dataset_specs = [
        (
            "kompas",
            "Kompas",
            Path(
                KOMPAS_PROCESSED_PATH
            ),
            kompas_final_path,
            REQUIRED_KOMPAS_COLUMNS,
        ),
        (
            "ag_news_train",
            "AG News Train",
            Path(
                AG_NEWS_TRAIN_PROCESSED_PATH
            ),
            agnews_train_final_path,
            REQUIRED_AGNEWS_COLUMNS,
        ),
        (
            "ag_news_test",
            "AG News Test",
            Path(
                AG_NEWS_TEST_PROCESSED_PATH
            ),
            agnews_test_final_path,
            REQUIRED_AGNEWS_COLUMNS,
        ),
    ]

    datasets: dict[
        tuple[str, str],
        pd.DataFrame,
    ] = {}

    source_paths: dict[
        tuple[str, str],
        Path,
    ] = {}

    # =========================================================================
    # LOAD DATASETS BEFORE AND AFTER CLEANING
    # =========================================================================

    for (
        dataset_key,
        display_name,
        initial_path,
        final_path,
        required_columns,
    ) in dataset_specs:

        for stage, path in [
            (
                "sebelum_cleaning",
                initial_path,
            ),
            (
                "setelah_cleaning",
                final_path,
            ),
        ]:
            datasets[
                (
                    dataset_key,
                    stage,
                )
            ] = load_dataset(
                file_path=path,
                dataset_name=(
                    f"{display_name} "
                    f"{stage}"
                ),
                dataset_key=dataset_key,
                stage=stage,
                required_columns=required_columns,
            )

            source_paths[
                (
                    dataset_key,
                    stage,
                )
            ] = path

    print(
        "\nDataset berhasil dimuat:"
    )

    for (
        dataset_key,
        display_name,
        _,
        _,
        _,
    ) in dataset_specs:

        before_count = len(
            datasets[
                (
                    dataset_key,
                    "sebelum_cleaning",
                )
            ]
        )

        after_count = len(
            datasets[
                (
                    dataset_key,
                    "setelah_cleaning",
                )
            ]
        )

        print(
            f"{display_name:<15}: "
            f"{before_count:,} -> "
            f"{after_count:,}"
        )

    # =========================================================================
    # MISSING VALUE ANALYSIS
    # =========================================================================

    missing_frames: list[
        pd.DataFrame
    ] = []

    for (
        dataset_key,
        display_name,
        _,
        _,
        _,
    ) in dataset_specs:

        for stage in [
            "sebelum_cleaning",
            "setelah_cleaning",
        ]:
            report = (
                create_missing_value_report(
                    dataframe=datasets[
                        (
                            dataset_key,
                            stage,
                        )
                    ],
                    dataset_name=dataset_key,
                    stage=stage,
                    source_path=source_paths[
                        (
                            dataset_key,
                            stage,
                        )
                    ],
                )
            )

            missing_frames.append(
                report
            )

            print_missing_summary(
                report=report,
                dataset_name=display_name,
                stage=stage,
            )

    missing_report = pd.concat(
        missing_frames,
        ignore_index=True,
    )

    missing_report.to_csv(
        MISSING_VALUE_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # DUPLICATE ANALYSIS
    # =========================================================================

    duplicate_records: list[
        dict[str, Any]
    ] = []

    kompas_details: list[
        pd.DataFrame
    ] = []

    agnews_train_details: list[
        pd.DataFrame
    ] = []

    agnews_test_details: list[
        pd.DataFrame
    ] = []

    for stage in [
        "sebelum_cleaning",
        "setelah_cleaning",
    ]:
        kompas_report, kompas_detail = (
            analyze_kompas_duplicates(
                dataframe=datasets[
                    (
                        "kompas",
                        stage,
                    )
                ],
                stage=stage,
                source_path=source_paths[
                    (
                        "kompas",
                        stage,
                    )
                ],
            )
        )

        duplicate_records.append(
            kompas_report
        )

        kompas_details.append(
            kompas_detail
        )

        print_duplicate_summary(
            kompas_report
        )

        for (
            dataset_key,
            detail_target,
        ) in [
            (
                "ag_news_train",
                agnews_train_details,
            ),
            (
                "ag_news_test",
                agnews_test_details,
            ),
        ]:
            agnews_report, agnews_detail = (
                analyze_agnews_duplicates(
                    dataframe=datasets[
                        (
                            dataset_key,
                            stage,
                        )
                    ],
                    dataset_name=dataset_key,
                    stage=stage,
                    source_path=source_paths[
                        (
                            dataset_key,
                            stage,
                        )
                    ],
                )
            )

            duplicate_records.append(
                agnews_report
            )

            detail_target.append(
                agnews_detail
            )

            print_duplicate_summary(
                agnews_report
            )

    duplicate_report = pd.DataFrame(
        duplicate_records
    )

    duplicate_report.to_csv(
        DUPLICATE_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # SAVE DUPLICATE DETAILS
    # =========================================================================

    kompas_duplicate_detail = pd.concat(
        kompas_details,
        ignore_index=True,
    )

    agnews_train_duplicate_detail = pd.concat(
        agnews_train_details,
        ignore_index=True,
    )

    agnews_test_duplicate_detail = pd.concat(
        agnews_test_details,
        ignore_index=True,
    )

    kompas_duplicate_detail.to_csv(
        KOMPAS_DUPLICATE_DETAIL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    agnews_train_duplicate_detail.to_csv(
        AGNEWS_TRAIN_DUPLICATE_DETAIL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    agnews_test_duplicate_detail.to_csv(
        AGNEWS_TEST_DUPLICATE_DETAIL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # TRAIN-TEST OVERLAP ANALYSIS
    # =========================================================================

    overlap_initial = (
        analyze_agnews_train_test_overlap(
            train=datasets[
                (
                    "ag_news_train",
                    "sebelum_cleaning",
                )
            ],
            test=datasets[
                (
                    "ag_news_test",
                    "sebelum_cleaning",
                )
            ],
            stage="sebelum_cleaning",
        )
    )

    overlap_final = (
        analyze_agnews_train_test_overlap(
            train=datasets[
                (
                    "ag_news_train",
                    "setelah_cleaning",
                )
            ],
            test=datasets[
                (
                    "ag_news_test",
                    "setelah_cleaning",
                )
            ],
            stage="setelah_cleaning",
        )
    )

    overlap_report = pd.concat(
        [
            overlap_initial,
            overlap_final,
        ],
        ignore_index=True,
    )

    overlap_report.to_csv(
        AGNEWS_OVERLAP_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "AG NEWS TRAIN-TEST OVERLAP"
    )

    print(
        "=" * 80
    )

    print(
        f"Sebelum cleaning : "
        f"{len(overlap_initial):,} artikel test"
    )

    print(
        f"Setelah cleaning : "
        f"{len(overlap_final):,} artikel test"
    )

    if not overlap_initial.empty:
        print(
            "\nContoh overlap sebelum cleaning:"
        )

        print(
            overlap_initial[
                [
                    "train_document_id",
                    "test_document_id",
                    "category",
                    "title",
                ]
            ]
            .head(
                10
            )
            .to_string(
                index=False
            )
        )

    # =========================================================================
    # FINAL INTEGRITY VALIDATION
    # =========================================================================

    duplicate_report_initial = (
        duplicate_report[
            duplicate_report[
                "stage"
            ].eq(
                "sebelum_cleaning"
            )
        ].copy()
    )

    duplicate_report_final = (
        duplicate_report[
            duplicate_report[
                "stage"
            ].eq(
                "setelah_cleaning"
            )
        ].copy()
    )

    validate_final_integrity(
        duplicate_report_final=(
            duplicate_report_final
        ),
        overlap_final=overlap_final,
    )

    # =========================================================================
    # CLEANING INTEGRITY SUMMARY
    # =========================================================================

    cleaning_integrity = (
        create_cleaning_integrity_summary(
            initial_duplicate_report=(
                duplicate_report_initial
            ),
            final_duplicate_report=(
                duplicate_report_final
            ),
            overlap_initial_count=len(
                overlap_initial
            ),
            overlap_final_count=len(
                overlap_final
            ),
        )
    )

    cleaning_integrity.to_csv(
        CLEANING_INTEGRITY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # OUTPUT VALIDATION
    # =========================================================================

    output_files = [
        (
            MISSING_VALUE_REPORT_PATH,
            "laporan missing value",
        ),
        (
            DUPLICATE_REPORT_PATH,
            "ringkasan duplikat",
        ),
        (
            KOMPAS_DUPLICATE_DETAIL_PATH,
            "detail duplikat Kompas",
        ),
        (
            AGNEWS_TRAIN_DUPLICATE_DETAIL_PATH,
            "detail duplikat AG News train",
        ),
        (
            AGNEWS_TEST_DUPLICATE_DETAIL_PATH,
            "detail duplikat AG News test",
        ),
        (
            AGNEWS_OVERLAP_REPORT_PATH,
            "overlap AG News train-test",
        ),
        (
            CLEANING_INTEGRITY_PATH,
            "ringkasan integritas cleaning",
        ),
    ]

    for (
        file_path,
        description,
    ) in output_files:
        validate_output_file(
            file_path=file_path,
            description=description,
        )

    # =========================================================================
    # TERMINAL OUTPUT
    # =========================================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "OUTPUT MISSING AND DUPLICATE ANALYSIS"
    )

    print(
        "=" * 80
    )

    print(
        "\nLaporan missing value:"
    )

    print(
        MISSING_VALUE_REPORT_PATH
    )

    print(
        "\nRingkasan duplikat:"
    )

    print(
        DUPLICATE_REPORT_PATH
    )

    print(
        "\nDetail duplikat Kompas:"
    )

    print(
        KOMPAS_DUPLICATE_DETAIL_PATH
    )

    print(
        "\nDetail duplikat AG News train:"
    )

    print(
        AGNEWS_TRAIN_DUPLICATE_DETAIL_PATH
    )

    print(
        "\nDetail duplikat AG News test:"
    )

    print(
        AGNEWS_TEST_DUPLICATE_DETAIL_PATH
    )

    print(
        "\nOverlap AG News train-test:"
    )

    print(
        AGNEWS_OVERLAP_REPORT_PATH
    )

    print(
        "\nRingkasan integritas cleaning:"
    )

    print(
        CLEANING_INTEGRITY_PATH
    )

    print(
        "\nRingkasan integritas:"
    )

    print(
        cleaning_integrity.to_string(
            index=False
        )
    )

    print(
        "\nValidasi dataset final: LULUS"
    )

    print(
        "Tahap missing dan duplicate analysis selesai."
    )


if __name__ == "__main__":
    main()
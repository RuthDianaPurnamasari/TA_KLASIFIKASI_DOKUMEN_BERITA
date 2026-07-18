from __future__ import annotations

import html
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# PROJECT DIRECTORY
# =============================================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# IMPORT PROJECT CONFIGURATION
# =============================================================================

from config import (  # noqa: E402
    AG_NEWS_TEST_PROCESSED_PATH,
    AG_NEWS_TRAIN_PROCESSED_PATH,
    FIGURES_DIR,
    KOMPAS_PROCESSED_PATH,
    TABLES_DIR,
)


# =============================================================================
# DATA DIRECTORIES
# =============================================================================

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "2_data"
    / "processed"
)


# =============================================================================
# FINAL CLEANED DATASET CANDIDATES
# =============================================================================
# Kandidat pertama menjadi prioritas.
# Beberapa alternatif disediakan agar script tetap dapat menemukan file
# jika nama file sedikit berbeda.
# =============================================================================

KOMPAS_FINAL_CANDIDATES = [
    PROCESSED_DATA_DIR / "kompas_clean.csv",
    PROCESSED_DATA_DIR / "kompas_cleaned.csv",
    PROCESSED_DATA_DIR / "kompas_final.csv",
    PROCESSED_DATA_DIR / "kompas_processed_clean.csv",
]

AGNEWS_TRAIN_FINAL_CANDIDATES = [
    PROCESSED_DATA_DIR / "ag_news_train_clean.csv",
    PROCESSED_DATA_DIR / "agnews_train_clean.csv",
    PROCESSED_DATA_DIR / "ag_news_train_final.csv",
    PROCESSED_DATA_DIR / "ag_news_clean_train.csv",
]

AGNEWS_TEST_FINAL_CANDIDATES = [
    PROCESSED_DATA_DIR / "ag_news_test_clean.csv",
    PROCESSED_DATA_DIR / "agnews_test_clean.csv",
    PROCESSED_DATA_DIR / "ag_news_test_final.csv",
    PROCESSED_DATA_DIR / "ag_news_clean_test.csv",
]


# =============================================================================
# OUTPUT PATHS
# =============================================================================

DATASET_OVERVIEW_PATH = (
    TABLES_DIR
    / "dataset_overview.csv"
)

DATASET_OVERVIEW_ALL_STAGES_PATH = (
    TABLES_DIR
    / "dataset_overview_all_stages.csv"
)

DATA_CLEANING_COMPARISON_PATH = (
    TABLES_DIR
    / "data_cleaning_comparison.csv"
)

CLASS_DISTRIBUTION_PATH = (
    TABLES_DIR
    / "class_distribution.csv"
)

DATA_QUALITY_AUDIT_PATH = (
    TABLES_DIR
    / "dataset_quality_audit.csv"
)

KOMPAS_DISTRIBUTION_FIGURE = (
    FIGURES_DIR
    / "kompas_class_distribution.png"
)

AGNEWS_TRAIN_DISTRIBUTION_FIGURE = (
    FIGURES_DIR
    / "agnews_train_class_distribution.png"
)

AGNEWS_TEST_DISTRIBUTION_FIGURE = (
    FIGURES_DIR
    / "agnews_test_class_distribution.png"
)


# =============================================================================
# EXPECTED FINAL DATASET SIZE
# =============================================================================

EXPECTED_FINAL_COUNTS = {
    "kompas": 9_997,
    "ag_news_train": 119_817,
    "ag_news_test": 7_600,
}

VALIDATE_EXPECTED_FINAL_COUNTS = True


# =============================================================================
# REQUIRED COLUMNS
# =============================================================================

REQUIRED_KOMPAS_COLUMNS = {
    "document_id",
    "title",
    "description",
    "content",
    "date",
    "category",
}

REQUIRED_AGNEWS_COLUMNS = {
    "document_id",
    "class_index",
    "category",
    "title",
    "description",
}


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE,
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


def normalize_text_for_audit(
    value: Any,
) -> str:
    """
    Menormalisasi teks menggunakan definisi yang sama
    dengan proses data cleaning.

    Tanda baca dipertahankan agar artikel yang hanya
    memiliki kemiripan tidak dianggap sebagai duplikat.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = unicodedata.normalize(
        "NFKC",
        str(value),
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip().lower()

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

def ensure_output_directories() -> None:
    """
    Memastikan folder output sudah tersedia.
    """

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# FILE RESOLUTION
# =============================================================================

def is_valid_file(
    file_path: Path,
) -> bool:
    """
    Memeriksa apakah path merupakan file valid dan tidak kosong.
    """

    path = Path(file_path)

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
    Mencari file final pertama yang tersedia.
    """

    for candidate in candidates:
        candidate_path = Path(candidate)

        if is_valid_file(candidate_path):
            return candidate_path

    candidate_text = "\n".join(
        f"- {candidate}"
        for candidate in candidates
    )

    raise FileNotFoundError(
        f"Dataset final {dataset_name} tidak ditemukan.\n\n"
        "Path yang sudah diperiksa:\n"
        f"{candidate_text}\n\n"
        "Pastikan script cleaning final telah dijalankan "
        "dan nama file output sesuai."
    )


# =============================================================================
# CSV READER
# =============================================================================

def read_csv_with_fallback(
    file_path: Path,
) -> pd.DataFrame:
    """
    Membaca CSV menggunakan beberapa kemungkinan encoding.
    """

    encodings = [
        "utf-8-sig",
        "utf-8",
        "latin-1",
    ]

    last_error: Exception | None = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                file_path,
                encoding=encoding,
            )

        except UnicodeDecodeError as error:
            last_error = error
            continue

        except Exception as error:
            last_error = error
            break

    if last_error is not None:
        raise last_error

    return pd.DataFrame()


# =============================================================================
# COLUMN VALIDATION
# =============================================================================

def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    """
    Memastikan dataset mempunyai semua kolom penting.
    """

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            f"Dataset {dataset_name} tidak memiliki "
            "kolom yang dibutuhkan.\n"
            f"Kolom hilang: {sorted(missing_columns)}"
        )


# =============================================================================
# DATASET LOADER
# =============================================================================

def load_dataset(
    file_path: Path,
    dataset_name: str,
    required_columns: set[str],
) -> pd.DataFrame:
    """
    Membaca dan memvalidasi dataset.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset {dataset_name} tidak ditemukan:\n"
            f"{path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Path dataset {dataset_name} bukan file:\n"
            f"{path}"
        )

    if path.stat().st_size <= 0:
        raise ValueError(
            f"File dataset {dataset_name} kosong:\n"
            f"{path}"
        )

    dataframe = read_csv_with_fallback(
        path
    )

    if dataframe.empty:
        raise ValueError(
            f"Dataset {dataset_name} tidak memiliki baris data."
        )

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    validate_required_columns(
        dataframe=dataframe,
        required_columns=required_columns,
        dataset_name=dataset_name,
    )

    return dataframe


# =============================================================================
# DATA QUALITY UTILITIES
# =============================================================================

def count_empty_text(
    dataframe: pd.DataFrame,
    column_name: str,
) -> int:
    """
    Menghitung teks kosong, NaN, atau hanya whitespace.
    """

    if column_name not in dataframe.columns:
        return 0

    return int(
        dataframe[column_name]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )


def count_exact_duplicate_rows(
    dataframe: pd.DataFrame,
) -> int:
    """
    Menghitung duplikat berdasarkan seluruh kolom.
    """

    return int(
        dataframe.duplicated().sum()
    )


def build_normalized_article_key(
    dataframe: pd.DataFrame,
    text_columns: list[str],
) -> pd.Series:
    """
    Membentuk kunci artikel dari beberapa kolom teks.
    """

    available_columns = [
        column
        for column in text_columns
        if column in dataframe.columns
    ]

    if not available_columns:
        return pd.Series(
            "",
            index=dataframe.index,
            dtype="object",
        )

    normalized_components: list[pd.Series] = []

    for column in available_columns:
        normalized_components.append(
            dataframe[column].apply(
                normalize_text_for_audit
            )
        )

    article_key = normalized_components[0]

    for component in normalized_components[1:]:
        article_key = (
            article_key
            + " || "
            + component
        )

    return article_key


def count_normalized_duplicates(
    dataframe: pd.DataFrame,
    text_columns: list[str],
) -> int:
    """
    Menghitung duplikat artikel berdasarkan teks yang dinormalisasi.

    Nilai yang dikembalikan adalah jumlah baris duplikat
    setelah kemunculan pertama.
    """

    article_key = build_normalized_article_key(
        dataframe=dataframe,
        text_columns=text_columns,
    )

    valid_mask = (
        article_key
        .astype(str)
        .str.replace(
            " || ",
            "",
            regex=False,
        )
        .str.strip()
        .ne("")
    )

    valid_key = article_key[
        valid_mask
    ]

    return int(
        valid_key.duplicated(
            keep="first"
        ).sum()
    )


def count_kompas_duplicate_content(
    dataframe: pd.DataFrame,
) -> int:
    """
    Menghitung duplikat Kompas berdasarkan content.

    Pemeriksaan ini sesuai dengan temuan tiga artikel
    ber-content sama pada dataset awal.
    """

    if "content" not in dataframe.columns:
        return 0

    normalized_content = (
        dataframe["content"]
        .apply(
            normalize_text_for_audit
        )
    )

    valid_content = normalized_content[
        normalized_content.ne("")
    ]

    return int(
        valid_content.duplicated(
            keep="first"
        ).sum()
    )


def calculate_agnews_label_conflicts(
    dataframe: pd.DataFrame,
) -> tuple[int, int]:
    """
    Menghitung konflik label AG News.

    Konflik label terjadi ketika Title + Description yang sama
    mempunyai lebih dari satu label berbeda.

    Returns
    -------
    tuple[int, int]
        jumlah kelompok konflik dan jumlah baris yang terlibat.
    """

    label_column: str | None = None

    for candidate in [
        "class_index",
        "category",
        "label",
    ]:
        if candidate in dataframe.columns:
            label_column = candidate
            break

    if label_column is None:
        return 0, 0

    article_key = build_normalized_article_key(
        dataframe=dataframe,
        text_columns=[
            "title",
            "description",
        ],
    )

    audit_frame = pd.DataFrame(
        {
            "article_key": article_key,
            "label": (
                dataframe[label_column]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
            ),
        }
    )

    audit_frame = audit_frame[
        audit_frame["article_key"]
        .str.replace(
            " || ",
            "",
            regex=False,
        )
        .str.strip()
        .ne("")
    ]

    label_count_per_key = (
        audit_frame
        .groupby("article_key")["label"]
        .nunique()
    )

    conflict_keys = label_count_per_key[
        label_count_per_key > 1
    ].index

    conflict_group_count = int(
        len(conflict_keys)
    )

    conflict_row_count = int(
        audit_frame[
            audit_frame["article_key"].isin(
                conflict_keys
            )
        ].shape[0]
    )

    return (
        conflict_group_count,
        conflict_row_count,
    )


def calculate_date_range(
    dataframe: pd.DataFrame,
    date_column: str = "date",
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """
    Menghitung tanggal awal dan akhir dataset.
    """

    if date_column not in dataframe.columns:
        return None, None

    parsed_date = pd.to_datetime(
        dataframe[date_column],
        errors="coerce",
    )

    valid_dates = parsed_date.dropna()

    if valid_dates.empty:
        return None, None

    return (
        valid_dates.min(),
        valid_dates.max(),
    )


# =============================================================================
# DATASET AUDIT
# =============================================================================

def create_quality_audit(
    dataframe: pd.DataFrame,
    dataset_name: str,
    stage: str,
    source_path: Path,
) -> dict[str, Any]:
    """
    Membentuk audit kualitas dataset.
    """

    date_start, date_end = calculate_date_range(
        dataframe
    )

    normalized_duplicate_count = 0
    duplicate_content_count = 0
    conflict_group_count = 0
    conflict_row_count = 0

    if dataset_name == "kompas":
        normalized_duplicate_count = (
            count_normalized_duplicates(
                dataframe=dataframe,
                text_columns=[
                    "title",
                    "description",
                    "content",
                ],
            )
        )

        duplicate_content_count = (
            count_kompas_duplicate_content(
                dataframe
            )
        )

    elif dataset_name.startswith(
        "ag_news"
    ):
        normalized_duplicate_count = (
            count_normalized_duplicates(
                dataframe=dataframe,
                text_columns=[
                    "title",
                    "description",
                ],
            )
        )

        (
            conflict_group_count,
            conflict_row_count,
        ) = calculate_agnews_label_conflicts(
            dataframe
        )

    return {
        "dataset":
            dataset_name,

        "stage":
            stage,

        "source_path":
            str(
                Path(source_path).resolve()
            ),

        "jumlah_baris":
            int(
                len(dataframe)
            ),

        "jumlah_kolom":
            int(
                dataframe.shape[1]
            ),

        "jumlah_kategori":
            int(
                dataframe["category"].nunique(
                    dropna=True
                )
            ),

        "total_missing_value":
            int(
                dataframe.isna().sum().sum()
            ),

        "total_duplikat_baris_exact":
            count_exact_duplicate_rows(
                dataframe
            ),

        "duplikat_artikel_normalized":
            normalized_duplicate_count,

        "duplikat_content_normalized":
            duplicate_content_count,

        "kelompok_konflik_label":
            conflict_group_count,

        "baris_konflik_label":
            conflict_row_count,

        "title_kosong":
            count_empty_text(
                dataframe,
                "title",
            ),

        "description_kosong":
            count_empty_text(
                dataframe,
                "description",
            ),

        "content_kosong":
            count_empty_text(
                dataframe,
                "content",
            ),

        "tanggal_awal":
            (
                date_start.isoformat()
                if date_start is not None
                else ""
            ),

        "tanggal_akhir":
            (
                date_end.isoformat()
                if date_end is not None
                else ""
            ),

        "nama_kolom":
            ", ".join(
                dataframe.columns.astype(str)
            ),
    }


# =============================================================================
# TERMINAL DISPLAY
# =============================================================================

def display_dataset_overview(
    dataframe: pd.DataFrame,
    dataset_name: str,
    stage_name: str,
    source_path: Path,
) -> None:
    """
    Menampilkan karakteristik dataset pada terminal.
    """

    audit = create_quality_audit(
        dataframe=dataframe,
        dataset_name=dataset_name,
        stage=stage_name,
        source_path=source_path,
    )

    title = (
        f"{dataset_name.upper()} "
        f"— {stage_name.upper()}"
    )

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print(
        f"Path sumber                    : "
        f"{source_path}"
    )

    print(
        f"Jumlah data                    : "
        f"{audit['jumlah_baris']:,}"
    )

    print(
        f"Jumlah kolom                   : "
        f"{audit['jumlah_kolom']}"
    )

    print(
        f"Jumlah kategori                : "
        f"{audit['jumlah_kategori']}"
    )

    print(
        f"Total missing value            : "
        f"{audit['total_missing_value']:,}"
    )

    print(
        f"Duplikat seluruh baris         : "
        f"{audit['total_duplikat_baris_exact']:,}"
    )

    print(
        f"Duplikat artikel normalized    : "
        f"{audit['duplikat_artikel_normalized']:,}"
    )

    if dataset_name == "kompas":
        print(
            f"Duplikat berdasarkan content   : "
            f"{audit['duplikat_content_normalized']:,}"
        )

    if dataset_name.startswith(
        "ag_news"
    ):
        print(
            f"Kelompok konflik label         : "
            f"{audit['kelompok_konflik_label']:,}"
        )

        print(
            f"Baris terlibat konflik label   : "
            f"{audit['baris_konflik_label']:,}"
        )

    print(
        f"Title kosong                   : "
        f"{audit['title_kosong']:,}"
    )

    print(
        f"Description kosong             : "
        f"{audit['description_kosong']:,}"
    )

    if "content" in dataframe.columns:
        print(
            f"Content kosong                 : "
            f"{audit['content_kosong']:,}"
        )

    if (
        audit["tanggal_awal"]
        and audit["tanggal_akhir"]
    ):
        print(
            f"Tanggal awal                   : "
            f"{audit['tanggal_awal']}"
        )

        print(
            f"Tanggal akhir                  : "
            f"{audit['tanggal_akhir']}"
        )

    print("\nDistribusi kategori:")

    print(
        dataframe["category"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nPreview data:")

    preview_columns = [
        column
        for column in [
            "document_id",
            "class_index",
            "title",
            "category",
            "date",
        ]
        if column in dataframe.columns
    ]

    print(
        dataframe[preview_columns]
        .head()
        .to_string(
            index=False
        )
    )


# =============================================================================
# DATASET SUMMARY TABLE
# =============================================================================

def create_dataset_summary(
    audit_record: dict[str, Any],
) -> dict[str, Any]:
    """
    Membentuk ringkasan utama dataset untuk dashboard.
    """

    return {
        "dataset":
            audit_record["dataset"],

        "stage":
            audit_record["stage"],

        "jumlah_baris":
            audit_record["jumlah_baris"],

        "jumlah_kolom":
            audit_record["jumlah_kolom"],

        "jumlah_kategori":
            audit_record["jumlah_kategori"],

        "total_missing_value":
            audit_record["total_missing_value"],

        "total_duplikat_baris":
            audit_record[
                "total_duplikat_baris_exact"
            ],

        "duplikat_artikel_normalized":
            audit_record[
                "duplikat_artikel_normalized"
            ],

        "duplikat_content_normalized":
            audit_record[
                "duplikat_content_normalized"
            ],

        "kelompok_konflik_label":
            audit_record[
                "kelompok_konflik_label"
            ],

        "description_kosong":
            audit_record["description_kosong"],

        "tanggal_awal":
            audit_record["tanggal_awal"],

        "tanggal_akhir":
            audit_record["tanggal_akhir"],

        "nama_kolom":
            audit_record["nama_kolom"],
    }


# =============================================================================
# CLASS DISTRIBUTION TABLE
# =============================================================================

def create_class_distribution_table(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Membentuk tabel distribusi kelas dataset final.
    """

    distribution = (
        dataframe["category"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts()
        .sort_index()
        .rename_axis("category")
        .reset_index(
            name="jumlah_data"
        )
    )

    distribution.insert(
        0,
        "dataset",
        dataset_name,
    )

    distribution["persentase"] = (
        distribution["jumlah_data"]
        / len(dataframe)
        * 100
    ).round(4)

    return distribution


# =============================================================================
# CLEANING COMPARISON
# =============================================================================

def create_cleaning_comparison(
    dataset_name: str,
    initial_dataframe: pd.DataFrame,
    final_dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Membandingkan jumlah data sebelum dan setelah cleaning.
    """

    initial_count = int(
        len(initial_dataframe)
    )

    final_count = int(
        len(final_dataframe)
    )

    removed_count = (
        initial_count
        - final_count
    )

    removed_percentage = (
        removed_count
        / initial_count
        * 100
        if initial_count > 0
        else 0.0
    )

    return {
        "dataset":
            dataset_name,

        "jumlah_sebelum_cleaning":
            initial_count,

        "jumlah_setelah_cleaning":
            final_count,

        "jumlah_dihapus":
            removed_count,

        "persentase_dihapus":
            round(
                removed_percentage,
                6,
            ),
    }


# =============================================================================
# CLASS DISTRIBUTION FIGURE
# =============================================================================

def plot_class_distribution(
    dataframe: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    """
    Membuat grafik distribusi kelas dataset final.
    """

    category_counts = (
        dataframe["category"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts()
        .sort_index()
    )

    if category_counts.empty:
        raise ValueError(
            f"Distribusi kategori kosong untuk {title}."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(
            9,
            6,
        )
    )

    bars = axis.bar(
        category_counts.index,
        category_counts.values,
    )

    axis.set_title(
        title,
        fontsize=14,
        pad=15,
    )

    axis.set_xlabel(
        "Kategori",
        fontsize=11,
    )

    axis.set_ylabel(
        "Jumlah Data",
        fontsize=11,
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    axis.set_axisbelow(
        True
    )

    maximum_value = int(
        category_counts.max()
    )

    upper_limit = max(
        1,
        int(
            maximum_value
            * 1.12
        ),
    )

    axis.set_ylim(
        0,
        upper_limit,
    )

    for bar, value in zip(
        bars,
        category_counts.values,
    ):
        axis.text(
            bar.get_x()
            + bar.get_width()
            / 2,
            bar.get_height()
            + maximum_value
            * 0.01,
            f"{int(value):,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# =============================================================================
# EXPECTED FINAL COUNT VALIDATION
# =============================================================================

def validate_final_dataset_count(
    dataset_name: str,
    dataframe: pd.DataFrame,
) -> None:
    """
    Memvalidasi jumlah dataset final terhadap hasil cleaning penelitian.
    """

    if not VALIDATE_EXPECTED_FINAL_COUNTS:
        return

    expected_count = EXPECTED_FINAL_COUNTS[
        dataset_name
    ]

    actual_count = int(
        len(dataframe)
    )

    if actual_count != expected_count:
        raise ValueError(
            f"Jumlah dataset final {dataset_name} tidak sesuai.\n"
            f"Expected : {expected_count:,}\n"
            f"Actual   : {actual_count:,}\n\n"
            "Pastikan file yang dibaca merupakan hasil cleaning final, "
            "bukan dataset sebelum cleaning."
        )


# =============================================================================
# OUTPUT VALIDATION
# =============================================================================

def validate_output_file(
    file_path: Path,
    description: str,
) -> None:
    """
    Memastikan file output berhasil dibuat.
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
# MAIN PROGRAM
# =============================================================================

def main() -> None:
    """
    Menjalankan EDA tahap 3.1.

    Dataset sebelum cleaning digunakan untuk audit perbandingan.
    Dataset setelah cleaning digunakan sebagai sumber utama EDA
    dan dashboard.
    """

    print("=" * 80)
    print("STEP 3.1 - DATASET OVERVIEW")
    print("=" * 80)

    ensure_output_directories()

    # =========================================================================
    # RESOLVE FINAL DATASET PATHS
    # =========================================================================

    kompas_final_path = resolve_first_existing_file(
        candidates=KOMPAS_FINAL_CANDIDATES,
        dataset_name="Kompas",
    )

    agnews_train_final_path = resolve_first_existing_file(
        candidates=AGNEWS_TRAIN_FINAL_CANDIDATES,
        dataset_name="AG News Train",
    )

    agnews_test_final_path = resolve_first_existing_file(
        candidates=AGNEWS_TEST_FINAL_CANDIDATES,
        dataset_name="AG News Test",
    )

    print("\nDataset final yang dipilih:")

    print(
        f"Kompas        : {kompas_final_path}"
    )

    print(
        f"AG News Train : {agnews_train_final_path}"
    )

    print(
        f"AG News Test  : {agnews_test_final_path}"
    )

    # =========================================================================
    # LOAD DATASET BEFORE CLEANING
    # =========================================================================

    kompas_initial = load_dataset(
        file_path=KOMPAS_PROCESSED_PATH,
        dataset_name="Kompas sebelum cleaning",
        required_columns=REQUIRED_KOMPAS_COLUMNS,
    )

    agnews_train_initial = load_dataset(
        file_path=AG_NEWS_TRAIN_PROCESSED_PATH,
        dataset_name="AG News Train sebelum cleaning",
        required_columns=REQUIRED_AGNEWS_COLUMNS,
    )

    agnews_test_initial = load_dataset(
        file_path=AG_NEWS_TEST_PROCESSED_PATH,
        dataset_name="AG News Test sebelum cleaning",
        required_columns=REQUIRED_AGNEWS_COLUMNS,
    )

    # =========================================================================
    # LOAD FINAL CLEANED DATASET
    # =========================================================================

    kompas_final = load_dataset(
        file_path=kompas_final_path,
        dataset_name="Kompas setelah cleaning",
        required_columns=REQUIRED_KOMPAS_COLUMNS,
    )

    agnews_train_final = load_dataset(
        file_path=agnews_train_final_path,
        dataset_name="AG News Train setelah cleaning",
        required_columns=REQUIRED_AGNEWS_COLUMNS,
    )

    agnews_test_final = load_dataset(
        file_path=agnews_test_final_path,
        dataset_name="AG News Test setelah cleaning",
        required_columns=REQUIRED_AGNEWS_COLUMNS,
    )

    # =========================================================================
    # VALIDATE FINAL COUNTS
    # =========================================================================

    validate_final_dataset_count(
        dataset_name="kompas",
        dataframe=kompas_final,
    )

    validate_final_dataset_count(
        dataset_name="ag_news_train",
        dataframe=agnews_train_final,
    )

    validate_final_dataset_count(
        dataset_name="ag_news_test",
        dataframe=agnews_test_final,
    )

    # =========================================================================
    # DISPLAY OVERVIEW
    # =========================================================================

    display_dataset_overview(
        dataframe=kompas_initial,
        dataset_name="kompas",
        stage_name="sebelum_cleaning",
        source_path=KOMPAS_PROCESSED_PATH,
    )

    display_dataset_overview(
        dataframe=kompas_final,
        dataset_name="kompas",
        stage_name="setelah_cleaning",
        source_path=kompas_final_path,
    )

    display_dataset_overview(
        dataframe=agnews_train_initial,
        dataset_name="ag_news_train",
        stage_name="sebelum_cleaning",
        source_path=AG_NEWS_TRAIN_PROCESSED_PATH,
    )

    display_dataset_overview(
        dataframe=agnews_train_final,
        dataset_name="ag_news_train",
        stage_name="setelah_cleaning",
        source_path=agnews_train_final_path,
    )

    display_dataset_overview(
        dataframe=agnews_test_initial,
        dataset_name="ag_news_test",
        stage_name="sebelum_cleaning",
        source_path=AG_NEWS_TEST_PROCESSED_PATH,
    )

    display_dataset_overview(
        dataframe=agnews_test_final,
        dataset_name="ag_news_test",
        stage_name="setelah_cleaning",
        source_path=agnews_test_final_path,
    )

    # =========================================================================
    # CREATE AUDIT RECORDS
    # =========================================================================

    audit_records = [
        create_quality_audit(
            dataframe=kompas_initial,
            dataset_name="kompas",
            stage="sebelum_cleaning",
            source_path=KOMPAS_PROCESSED_PATH,
        ),
        create_quality_audit(
            dataframe=kompas_final,
            dataset_name="kompas",
            stage="setelah_cleaning",
            source_path=kompas_final_path,
        ),
        create_quality_audit(
            dataframe=agnews_train_initial,
            dataset_name="ag_news_train",
            stage="sebelum_cleaning",
            source_path=AG_NEWS_TRAIN_PROCESSED_PATH,
        ),
        create_quality_audit(
            dataframe=agnews_train_final,
            dataset_name="ag_news_train",
            stage="setelah_cleaning",
            source_path=agnews_train_final_path,
        ),
        create_quality_audit(
            dataframe=agnews_test_initial,
            dataset_name="ag_news_test",
            stage="sebelum_cleaning",
            source_path=AG_NEWS_TEST_PROCESSED_PATH,
        ),
        create_quality_audit(
            dataframe=agnews_test_final,
            dataset_name="ag_news_test",
            stage="setelah_cleaning",
            source_path=agnews_test_final_path,
        ),
    ]

    quality_audit_dataframe = pd.DataFrame(
        audit_records
    )

    quality_audit_dataframe.to_csv(
        DATA_QUALITY_AUDIT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # FINAL OVERVIEW FOR DASHBOARD
    # =========================================================================

    final_audit_records = [
        record
        for record in audit_records
        if record["stage"] == "setelah_cleaning"
    ]

    final_overview_dataframe = pd.DataFrame(
        [
            create_dataset_summary(
                record
            )
            for record in final_audit_records
        ]
    )

    final_overview_dataframe.to_csv(
        DATASET_OVERVIEW_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # OVERVIEW FOR ALL STAGES
    # =========================================================================

    all_stage_overview_dataframe = pd.DataFrame(
        [
            create_dataset_summary(
                record
            )
            for record in audit_records
        ]
    )

    all_stage_overview_dataframe.to_csv(
        DATASET_OVERVIEW_ALL_STAGES_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # CLEANING COMPARISON
    # =========================================================================

    cleaning_comparison_dataframe = pd.DataFrame(
        [
            create_cleaning_comparison(
                dataset_name="kompas",
                initial_dataframe=kompas_initial,
                final_dataframe=kompas_final,
            ),
            create_cleaning_comparison(
                dataset_name="ag_news_train",
                initial_dataframe=agnews_train_initial,
                final_dataframe=agnews_train_final,
            ),
            create_cleaning_comparison(
                dataset_name="ag_news_test",
                initial_dataframe=agnews_test_initial,
                final_dataframe=agnews_test_final,
            ),
        ]
    )

    cleaning_comparison_dataframe.to_csv(
        DATA_CLEANING_COMPARISON_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # FINAL CLASS DISTRIBUTION
    # =========================================================================

    class_distribution_dataframe = pd.concat(
        [
            create_class_distribution_table(
                dataframe=kompas_final,
                dataset_name="kompas",
            ),
            create_class_distribution_table(
                dataframe=agnews_train_final,
                dataset_name="ag_news_train",
            ),
            create_class_distribution_table(
                dataframe=agnews_test_final,
                dataset_name="ag_news_test",
            ),
        ],
        ignore_index=True,
    )

    class_distribution_dataframe.to_csv(
        CLASS_DISTRIBUTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # FINAL CLASS DISTRIBUTION FIGURES
    # =========================================================================

    plot_class_distribution(
        dataframe=kompas_final,
        title=(
            "Distribusi Kategori Dataset Kompas "
            "Setelah Cleaning"
        ),
        output_path=KOMPAS_DISTRIBUTION_FIGURE,
    )

    plot_class_distribution(
        dataframe=agnews_train_final,
        title=(
            "Distribusi Kategori AG News Train "
            "Setelah Cleaning"
        ),
        output_path=AGNEWS_TRAIN_DISTRIBUTION_FIGURE,
    )

    plot_class_distribution(
        dataframe=agnews_test_final,
        title=(
            "Distribusi Kategori AG News Test "
            "Setelah Cleaning"
        ),
        output_path=AGNEWS_TEST_DISTRIBUTION_FIGURE,
    )

    # =========================================================================
    # VALIDATE OUTPUTS
    # =========================================================================

    output_files = [
        (
            DATASET_OVERVIEW_PATH,
            "dataset overview final",
        ),
        (
            DATASET_OVERVIEW_ALL_STAGES_PATH,
            "dataset overview seluruh tahap",
        ),
        (
            DATA_CLEANING_COMPARISON_PATH,
            "perbandingan cleaning",
        ),
        (
            CLASS_DISTRIBUTION_PATH,
            "distribusi kelas final",
        ),
        (
            DATA_QUALITY_AUDIT_PATH,
            "audit kualitas dataset",
        ),
        (
            KOMPAS_DISTRIBUTION_FIGURE,
            "grafik distribusi Kompas",
        ),
        (
            AGNEWS_TRAIN_DISTRIBUTION_FIGURE,
            "grafik distribusi AG News train",
        ),
        (
            AGNEWS_TEST_DISTRIBUTION_FIGURE,
            "grafik distribusi AG News test",
        ),
    ]

    for file_path, description in output_files:
        validate_output_file(
            file_path=file_path,
            description=description,
        )

    # =========================================================================
    # OUTPUT INFORMATION
    # =========================================================================

    print("\n" + "=" * 80)
    print("OUTPUT DATASET OVERVIEW")
    print("=" * 80)

    print("\nRingkasan dataset final:")
    print(DATASET_OVERVIEW_PATH)

    print("\nRingkasan seluruh tahap:")
    print(DATASET_OVERVIEW_ALL_STAGES_PATH)

    print("\nPerbandingan sebelum dan setelah cleaning:")
    print(DATA_CLEANING_COMPARISON_PATH)

    print("\nAudit kualitas dataset:")
    print(DATA_QUALITY_AUDIT_PATH)

    print("\nDistribusi kelas final:")
    print(CLASS_DISTRIBUTION_PATH)

    print("\nGrafik distribusi Kompas:")
    print(KOMPAS_DISTRIBUTION_FIGURE)

    print("\nGrafik distribusi AG News train:")
    print(AGNEWS_TRAIN_DISTRIBUTION_FIGURE)

    print("\nGrafik distribusi AG News test:")
    print(AGNEWS_TEST_DISTRIBUTION_FIGURE)

    print("\nPerbandingan jumlah data:")

    print(
        cleaning_comparison_dataframe.to_string(
            index=False
        )
    )

    print("\nDistribusi kelas final:")

    print(
        class_distribution_dataframe.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("Tahap dataset overview selesai.")
    print("=" * 80)


if __name__ == "__main__":
    main()
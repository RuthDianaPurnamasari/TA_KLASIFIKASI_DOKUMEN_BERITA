from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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
    FIGURES_DIR,
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

TEXT_STATISTICS_PATH = (
    TABLES_DIR
    / "text_statistics.csv"
)

TEXT_STATISTICS_BY_CATEGORY_PATH = (
    TABLES_DIR
    / "text_statistics_by_category.csv"
)

SEQUENCE_LENGTH_COVERAGE_PATH = (
    TABLES_DIR
    / "sequence_length_coverage.csv"
)

KOMPAS_WORD_LENGTH_FIGURE = (
    FIGURES_DIR
    / "kompas_text_length_distribution.png"
)

AGNEWS_TRAIN_WORD_LENGTH_FIGURE = (
    FIGURES_DIR
    / "agnews_train_text_length_distribution.png"
)

AGNEWS_TEST_WORD_LENGTH_FIGURE = (
    FIGURES_DIR
    / "agnews_test_text_length_distribution.png"
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
    "category",
}

REQUIRED_AGNEWS_COLUMNS = {
    "document_id",
    "title",
    "description",
    "category",
}


# =============================================================================
# TEXT FIELD CONFIGURATION
# =============================================================================

KOMPAS_TEXT_FIELDS = [
    "title",
    "description",
    "content",
    "title_description",
]

AGNEWS_TEXT_FIELDS = [
    "title",
    "description",
    "title_description",
]

FIELD_DISPLAY_NAMES = {
    "title": "Title",
    "description": "Description",
    "content": "Content",
    "title_description": "Title + Description",
}


# =============================================================================
# SEQUENCE LENGTH CANDIDATES
# =============================================================================

SEQUENCE_LENGTH_CANDIDATES = [
    20,
    40,
    60,
    80,
    100,
    128,
    256,
]

SEQUENCE_ANALYSIS_FIELDS = [
    "title",
    "title_description",
]


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def ensure_output_directories() -> None:
    """
    Memastikan folder output tersedia.
    """

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def is_valid_file(
    file_path: Path,
) -> bool:
    """
    Memeriksa apakah file tersedia dan tidak kosong.
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
    Memilih file dataset final pertama yang tersedia.
    """

    for candidate in candidates:
        if is_valid_file(candidate):
            return candidate

    candidate_text = "\n".join(
        f"- {candidate}"
        for candidate in candidates
    )

    raise FileNotFoundError(
        f"Dataset final {dataset_name} tidak ditemukan.\n\n"
        f"Path yang diperiksa:\n{candidate_text}\n\n"
        "Pastikan tahap cleaning sudah dijalankan."
    )


def read_csv_with_fallback(
    file_path: Path,
) -> pd.DataFrame:
    """
    Membaca file CSV menggunakan beberapa encoding.
    """

    encoding_candidates = [
        "utf-8-sig",
        "utf-8",
        "latin-1",
    ]

    last_error: Exception | None = None

    for encoding in encoding_candidates:
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


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    """
    Memastikan semua kolom penting tersedia.
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


def validate_final_dataset_count(
    dataframe: pd.DataFrame,
    dataset_key: str,
) -> None:
    """
    Memastikan jumlah dataset sesuai hasil cleaning final.
    """

    if not VALIDATE_EXPECTED_FINAL_COUNTS:
        return

    expected_count = EXPECTED_FINAL_COUNTS[
        dataset_key
    ]

    actual_count = int(
        len(dataframe)
    )

    if actual_count != expected_count:
        raise ValueError(
            f"Jumlah dataset final {dataset_key} tidak sesuai.\n"
            f"Expected : {expected_count:,}\n"
            f"Actual   : {actual_count:,}\n\n"
            "Pastikan file yang digunakan merupakan dataset "
            "setelah cleaning."
        )


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
# DATASET LOADER
# =============================================================================

def load_dataset(
    file_path: Path,
    dataset_name: str,
    dataset_key: str,
    required_columns: set[str],
) -> pd.DataFrame:
    """
    Membaca dan memvalidasi dataset final.
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

    validate_final_dataset_count(
        dataframe=dataframe,
        dataset_key=dataset_key,
    )

    return dataframe


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def normalize_text_value(
    value: Any,
) -> str:
    """
    Merapikan whitespace tanpa mengubah isi substantif teks.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return " ".join(
        str(value)
        .strip()
        .split()
    )


def prepare_text_fields(
    dataframe: pd.DataFrame,
    include_content: bool,
) -> pd.DataFrame:
    """
    Menyiapkan Title, Description, Content, dan gabungan
    Title + Description.

    Gabungan Title + Description digunakan untuk mendukung
    analisis skenario K2, K3, dan A2.
    """

    result = dataframe.copy()

    base_columns = [
        "title",
        "description",
    ]

    if include_content:
        base_columns.append(
            "content"
        )

    for column in base_columns:
        result[column] = result[column].apply(
            normalize_text_value
        )

    result["title_description"] = (
        result["title"]
        + " "
        + result["description"]
    ).str.strip()

    return result


# =============================================================================
# TEXT LENGTH FEATURES
# =============================================================================

def add_text_length_features(
    dataframe: pd.DataFrame,
    text_fields: list[str],
) -> pd.DataFrame:
    """
    Menambahkan jumlah kata dan jumlah karakter.

    Contoh kolom:
    title_word_count
    title_char_count
    title_description_word_count
    """

    result = dataframe.copy()

    for field in text_fields:
        if field not in result.columns:
            raise KeyError(
                f"Kolom teks {field} tidak ditemukan."
            )

        result[field] = (
            result[field]
            .fillna("")
            .astype(str)
            .apply(normalize_text_value)
        )

        result[f"{field}_word_count"] = (
            result[field]
            .str.split()
            .str.len()
            .astype(int)
        )

        result[f"{field}_char_count"] = (
            result[field]
            .str.len()
            .astype(int)
        )

    return result


# =============================================================================
# OVERALL TEXT STATISTICS
# =============================================================================

def create_text_statistics(
    dataframe: pd.DataFrame,
    dataset_name: str,
    text_fields: list[str],
) -> pd.DataFrame:
    """
    Membuat statistik panjang teks secara keseluruhan.
    """

    records: list[dict[str, Any]] = []

    for field in text_fields:
        for unit in [
            "word_count",
            "char_count",
        ]:
            metric_column = (
                f"{field}_{unit}"
            )

            series = pd.to_numeric(
                dataframe[metric_column],
                errors="coerce",
            ).dropna()

            if series.empty:
                continue

            records.append(
                {
                    "dataset":
                        dataset_name,

                    "text_field":
                        field,

                    "text_field_display":
                        FIELD_DISPLAY_NAMES.get(
                            field,
                            field,
                        ),

                    "unit":
                        unit,

                    "jumlah_data":
                        int(
                            len(series)
                        ),

                    "minimum":
                        int(
                            series.min()
                        ),

                    "maksimum":
                        int(
                            series.max()
                        ),

                    "mean":
                        round(
                            float(
                                series.mean()
                            ),
                            4,
                        ),

                    "median":
                        round(
                            float(
                                series.median()
                            ),
                            4,
                        ),

                    "std":
                        round(
                            float(
                                series.std()
                            ),
                            4,
                        ),

                    "q1":
                        round(
                            float(
                                series.quantile(
                                    0.25
                                )
                            ),
                            4,
                        ),

                    "q3":
                        round(
                            float(
                                series.quantile(
                                    0.75
                                )
                            ),
                            4,
                        ),

                    "p90":
                        round(
                            float(
                                series.quantile(
                                    0.90
                                )
                            ),
                            4,
                        ),

                    "p95":
                        round(
                            float(
                                series.quantile(
                                    0.95
                                )
                            ),
                            4,
                        ),

                    "p99":
                        round(
                            float(
                                series.quantile(
                                    0.99
                                )
                            ),
                            4,
                        ),

                    "jumlah_kosong":
                        int(
                            series.eq(0).sum()
                        ),
                }
            )

    return pd.DataFrame(
        records
    )


# =============================================================================
# TEXT STATISTICS BY CATEGORY
# =============================================================================

def create_statistics_by_category(
    dataframe: pd.DataFrame,
    dataset_name: str,
    text_fields: list[str],
) -> pd.DataFrame:
    """
    Menghitung statistik panjang teks per kategori.
    """

    if "category" not in dataframe.columns:
        raise KeyError(
            f"Kolom category tidak ditemukan "
            f"pada dataset {dataset_name}."
        )

    records: list[dict[str, Any]] = []

    grouped = dataframe.groupby(
        "category",
        dropna=False,
        sort=True,
    )

    for category, group in grouped:
        for field in text_fields:
            word_column = (
                f"{field}_word_count"
            )

            char_column = (
                f"{field}_char_count"
            )

            word_series = pd.to_numeric(
                group[word_column],
                errors="coerce",
            ).dropna()

            char_series = pd.to_numeric(
                group[char_column],
                errors="coerce",
            ).dropna()

            records.append(
                {
                    "dataset":
                        dataset_name,

                    "category":
                        str(category),

                    "text_field":
                        field,

                    "text_field_display":
                        FIELD_DISPLAY_NAMES.get(
                            field,
                            field,
                        ),

                    "jumlah_data":
                        int(
                            len(group)
                        ),

                    "minimum_word_count":
                        int(
                            word_series.min()
                        ),

                    "maximum_word_count":
                        int(
                            word_series.max()
                        ),

                    "mean_word_count":
                        round(
                            float(
                                word_series.mean()
                            ),
                            4,
                        ),

                    "median_word_count":
                        round(
                            float(
                                word_series.median()
                            ),
                            4,
                        ),

                    "std_word_count":
                        round(
                            float(
                                word_series.std()
                            ),
                            4,
                        ),

                    "p95_word_count":
                        round(
                            float(
                                word_series.quantile(
                                    0.95
                                )
                            ),
                            4,
                        ),

                    "mean_char_count":
                        round(
                            float(
                                char_series.mean()
                            ),
                            4,
                        ),

                    "median_char_count":
                        round(
                            float(
                                char_series.median()
                            ),
                            4,
                        ),
                }
            )

    return pd.DataFrame(
        records
    )


# =============================================================================
# SEQUENCE LENGTH COVERAGE
# =============================================================================

def create_sequence_length_coverage(
    dataframe: pd.DataFrame,
    dataset_name: str,
    text_fields: list[str],
    sequence_lengths: list[int],
) -> pd.DataFrame:
    """
    Menghitung persentase dokumen yang panjang katanya dapat
    ditampung oleh kandidat sequence length.

    Catatan:
    Perhitungan menggunakan jumlah kata berbasis whitespace.
    Hasil ini merupakan estimasi EDA, bukan jumlah token persis
    dari TextVectorization.
    """

    records: list[dict[str, Any]] = []

    for field in text_fields:
        word_column = (
            f"{field}_word_count"
        )

        if word_column not in dataframe.columns:
            continue

        series = pd.to_numeric(
            dataframe[word_column],
            errors="coerce",
        ).dropna()

        total_data = int(
            len(series)
        )

        if total_data == 0:
            continue

        for sequence_length in sequence_lengths:
            within_count = int(
                series.le(
                    sequence_length
                ).sum()
            )

            truncated_count = (
                total_data
                - within_count
            )

            records.append(
                {
                    "dataset":
                        dataset_name,

                    "text_field":
                        field,

                    "text_field_display":
                        FIELD_DISPLAY_NAMES.get(
                            field,
                            field,
                        ),

                    "sequence_length":
                        int(
                            sequence_length
                        ),

                    "jumlah_data":
                        total_data,

                    "jumlah_tertampung":
                        within_count,

                    "persentase_tertampung":
                        round(
                            within_count
                            / total_data
                            * 100,
                            4,
                        ),

                    "jumlah_berpotensi_truncated":
                        truncated_count,

                    "persentase_berpotensi_truncated":
                        round(
                            truncated_count
                            / total_data
                            * 100,
                            4,
                        ),
                }
            )

    return pd.DataFrame(
        records
    )


# =============================================================================
# TERMINAL DISPLAY
# =============================================================================

def display_dataset_statistics(
    dataframe: pd.DataFrame,
    dataset_name: str,
    text_fields: list[str],
) -> None:
    """
    Menampilkan statistik utama pada terminal.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        dataset_name.upper()
    )

    print(
        "=" * 80
    )

    print(
        f"Jumlah data: {len(dataframe):,}"
    )

    for field in text_fields:
        word_column = (
            f"{field}_word_count"
        )

        char_column = (
            f"{field}_char_count"
        )

        word_series = dataframe[
            word_column
        ]

        char_series = dataframe[
            char_column
        ]

        print(
            f"\nKolom: "
            f"{FIELD_DISPLAY_NAMES.get(field, field)}"
        )

        print(
            f"Rata-rata kata     : "
            f"{word_series.mean():.2f}"
        )

        print(
            f"Median kata        : "
            f"{word_series.median():.2f}"
        )

        print(
            f"Minimum kata       : "
            f"{int(word_series.min()):,}"
        )

        print(
            f"Maksimum kata      : "
            f"{int(word_series.max()):,}"
        )

        print(
            f"Persentil 90       : "
            f"{word_series.quantile(0.90):.2f}"
        )

        print(
            f"Persentil 95       : "
            f"{word_series.quantile(0.95):.2f}"
        )

        print(
            f"Persentil 99       : "
            f"{word_series.quantile(0.99):.2f}"
        )

        print(
            f"Rata-rata karakter : "
            f"{char_series.mean():.2f}"
        )

        print(
            f"Teks kosong        : "
            f"{int(word_series.eq(0).sum()):,}"
        )


# =============================================================================
# HISTOGRAM
# =============================================================================

def plot_text_length_distribution(
    dataframe: pd.DataFrame,
    text_fields: list[str],
    dataset_name: str,
    output_path: Path,
) -> None:
    """
    Membuat histogram jumlah kata.

    Tampilan histogram dibatasi sampai persentil ke-99 agar
    outlier yang sangat panjang tidak membuat pola utama sulit dibaca.
    Statistik CSV tetap dihitung menggunakan seluruh data.
    """

    figure_count = len(
        text_fields
    )

    column_count = 2

    row_count = math.ceil(
        figure_count
        / column_count
    )

    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(
            14,
            5 * row_count,
        ),
    )

    axes_list = (
        list(
            axes.flatten()
        )
        if hasattr(
            axes,
            "flatten",
        )
        else [axes]
    )

    for axis, field in zip(
        axes_list,
        text_fields,
    ):
        word_column = (
            f"{field}_word_count"
        )

        series = pd.to_numeric(
            dataframe[word_column],
            errors="coerce",
        ).dropna()

        percentile_99 = float(
            series.quantile(
                0.99
            )
        )

        upper_limit = max(
            1,
            int(
                math.ceil(
                    percentile_99
                )
            ),
        )

        visible_series = series[
            series.le(
                upper_limit
            )
        ]

        outlier_count = int(
            series.gt(
                upper_limit
            ).sum()
        )

        axis.hist(
            visible_series,
            bins=40,
            edgecolor="black",
            alpha=0.8,
        )

        mean_value = float(
            series.mean()
        )

        median_value = float(
            series.median()
        )

        axis.axvline(
            mean_value,
            linestyle="--",
            linewidth=1.8,
            label=(
                f"Mean = {mean_value:.2f}"
            ),
        )

        axis.axvline(
            median_value,
            linestyle=":",
            linewidth=1.8,
            label=(
                f"Median = {median_value:.2f}"
            ),
        )

        field_name = FIELD_DISPLAY_NAMES.get(
            field,
            field,
        )

        axis.set_title(
            f"Distribusi Panjang {field_name}\n"
            f"{dataset_name} — ditampilkan hingga P99",
            fontsize=12,
            pad=12,
        )

        axis.set_xlabel(
            "Jumlah Kata",
            fontsize=10,
        )

        axis.set_ylabel(
            "Frekuensi",
            fontsize=10,
        )

        axis.grid(
            axis="y",
            linestyle="--",
            alpha=0.3,
        )

        axis.legend(
            fontsize=9,
        )

        axis.text(
            0.98,
            0.95,
            (
                f"P99 = {percentile_99:.2f}\n"
                f"Data > P99 = {outlier_count:,}"
            ),
            transform=axis.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
            fontsize=9,
            bbox={
                "boxstyle": "round",
                "alpha": 0.15,
            },
        )

    for unused_axis in axes_list[
        figure_count:
    ]:
        unused_axis.axis(
            "off"
        )

    figure.suptitle(
        f"Distribusi Panjang Teks — {dataset_name}",
        fontsize=15,
        y=1.01,
    )

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main() -> None:
    """
    Menjalankan EDA tahap 3.2:
    statistik panjang teks dataset final.
    """

    print(
        "=" * 80
    )

    print(
        "STEP 3.2 - TEXT STATISTICS"
    )

    print(
        "=" * 80
    )

    ensure_output_directories()

    # =========================================================================
    # RESOLVE FINAL DATASET PATHS
    # =========================================================================

    kompas_path = resolve_first_existing_file(
        candidates=KOMPAS_FINAL_CANDIDATES,
        dataset_name="Kompas",
    )

    agnews_train_path = resolve_first_existing_file(
        candidates=AGNEWS_TRAIN_FINAL_CANDIDATES,
        dataset_name="AG News Train",
    )

    agnews_test_path = resolve_first_existing_file(
        candidates=AGNEWS_TEST_FINAL_CANDIDATES,
        dataset_name="AG News Test",
    )

    print(
        "\nDataset final yang digunakan:"
    )

    print(
        f"Kompas        : {kompas_path}"
    )

    print(
        f"AG News Train : {agnews_train_path}"
    )

    print(
        f"AG News Test  : {agnews_test_path}"
    )

    # =========================================================================
    # LOAD FINAL DATASETS
    # =========================================================================

    kompas = load_dataset(
        file_path=kompas_path,
        dataset_name="Kompas",
        dataset_key="kompas",
        required_columns=REQUIRED_KOMPAS_COLUMNS,
    )

    agnews_train = load_dataset(
        file_path=agnews_train_path,
        dataset_name="AG News Train",
        dataset_key="ag_news_train",
        required_columns=REQUIRED_AGNEWS_COLUMNS,
    )

    agnews_test = load_dataset(
        file_path=agnews_test_path,
        dataset_name="AG News Test",
        dataset_key="ag_news_test",
        required_columns=REQUIRED_AGNEWS_COLUMNS,
    )

    # =========================================================================
    # PREPARE TEXT FIELDS
    # =========================================================================

    kompas = prepare_text_fields(
        dataframe=kompas,
        include_content=True,
    )

    agnews_train = prepare_text_fields(
        dataframe=agnews_train,
        include_content=False,
    )

    agnews_test = prepare_text_fields(
        dataframe=agnews_test,
        include_content=False,
    )

    # =========================================================================
    # ADD TEXT LENGTH FEATURES
    # =========================================================================

    kompas = add_text_length_features(
        dataframe=kompas,
        text_fields=KOMPAS_TEXT_FIELDS,
    )

    agnews_train = add_text_length_features(
        dataframe=agnews_train,
        text_fields=AGNEWS_TEXT_FIELDS,
    )

    agnews_test = add_text_length_features(
        dataframe=agnews_test,
        text_fields=AGNEWS_TEXT_FIELDS,
    )

    # =========================================================================
    # DISPLAY TERMINAL STATISTICS
    # =========================================================================

    display_dataset_statistics(
        dataframe=kompas,
        dataset_name="Dataset Kompas",
        text_fields=KOMPAS_TEXT_FIELDS,
    )

    display_dataset_statistics(
        dataframe=agnews_train,
        dataset_name="AG News Train",
        text_fields=AGNEWS_TEXT_FIELDS,
    )

    display_dataset_statistics(
        dataframe=agnews_test,
        dataset_name="AG News Test",
        text_fields=AGNEWS_TEXT_FIELDS,
    )

    # =========================================================================
    # OVERALL STATISTICS
    # =========================================================================

    text_statistics = pd.concat(
        [
            create_text_statistics(
                dataframe=kompas,
                dataset_name="kompas",
                text_fields=KOMPAS_TEXT_FIELDS,
            ),
            create_text_statistics(
                dataframe=agnews_train,
                dataset_name="ag_news_train",
                text_fields=AGNEWS_TEXT_FIELDS,
            ),
            create_text_statistics(
                dataframe=agnews_test,
                dataset_name="ag_news_test",
                text_fields=AGNEWS_TEXT_FIELDS,
            ),
        ],
        ignore_index=True,
    )

    text_statistics.to_csv(
        TEXT_STATISTICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # STATISTICS BY CATEGORY
    # =========================================================================

    text_statistics_by_category = pd.concat(
        [
            create_statistics_by_category(
                dataframe=kompas,
                dataset_name="kompas",
                text_fields=KOMPAS_TEXT_FIELDS,
            ),
            create_statistics_by_category(
                dataframe=agnews_train,
                dataset_name="ag_news_train",
                text_fields=AGNEWS_TEXT_FIELDS,
            ),
            create_statistics_by_category(
                dataframe=agnews_test,
                dataset_name="ag_news_test",
                text_fields=AGNEWS_TEXT_FIELDS,
            ),
        ],
        ignore_index=True,
    )

    text_statistics_by_category.to_csv(
        TEXT_STATISTICS_BY_CATEGORY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # SEQUENCE LENGTH COVERAGE
    # =========================================================================

    sequence_length_coverage = pd.concat(
        [
            create_sequence_length_coverage(
                dataframe=kompas,
                dataset_name="kompas",
                text_fields=SEQUENCE_ANALYSIS_FIELDS,
                sequence_lengths=SEQUENCE_LENGTH_CANDIDATES,
            ),
            create_sequence_length_coverage(
                dataframe=agnews_train,
                dataset_name="ag_news_train",
                text_fields=SEQUENCE_ANALYSIS_FIELDS,
                sequence_lengths=SEQUENCE_LENGTH_CANDIDATES,
            ),
            create_sequence_length_coverage(
                dataframe=agnews_test,
                dataset_name="ag_news_test",
                text_fields=SEQUENCE_ANALYSIS_FIELDS,
                sequence_lengths=SEQUENCE_LENGTH_CANDIDATES,
            ),
        ],
        ignore_index=True,
    )

    sequence_length_coverage.to_csv(
        SEQUENCE_LENGTH_COVERAGE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # HISTOGRAM FIGURES
    # =========================================================================

    plot_text_length_distribution(
        dataframe=kompas,
        text_fields=KOMPAS_TEXT_FIELDS,
        dataset_name="Kompas",
        output_path=KOMPAS_WORD_LENGTH_FIGURE,
    )

    plot_text_length_distribution(
        dataframe=agnews_train,
        text_fields=AGNEWS_TEXT_FIELDS,
        dataset_name="AG News Train",
        output_path=AGNEWS_TRAIN_WORD_LENGTH_FIGURE,
    )

    plot_text_length_distribution(
        dataframe=agnews_test,
        text_fields=AGNEWS_TEXT_FIELDS,
        dataset_name="AG News Test",
        output_path=AGNEWS_TEST_WORD_LENGTH_FIGURE,
    )

    # =========================================================================
    # OUTPUT VALIDATION
    # =========================================================================

    output_files = [
        (
            TEXT_STATISTICS_PATH,
            "statistik teks keseluruhan",
        ),
        (
            TEXT_STATISTICS_BY_CATEGORY_PATH,
            "statistik teks per kategori",
        ),
        (
            SEQUENCE_LENGTH_COVERAGE_PATH,
            "cakupan sequence length",
        ),
        (
            KOMPAS_WORD_LENGTH_FIGURE,
            "grafik panjang teks Kompas",
        ),
        (
            AGNEWS_TRAIN_WORD_LENGTH_FIGURE,
            "grafik panjang teks AG News train",
        ),
        (
            AGNEWS_TEST_WORD_LENGTH_FIGURE,
            "grafik panjang teks AG News test",
        ),
    ]

    for file_path, description in output_files:
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
        "OUTPUT TEXT STATISTICS"
    )

    print(
        "=" * 80
    )

    print(
        "\nTabel statistik keseluruhan:"
    )

    print(
        TEXT_STATISTICS_PATH
    )

    print(
        "\nTabel statistik per kategori:"
    )

    print(
        TEXT_STATISTICS_BY_CATEGORY_PATH
    )

    print(
        "\nTabel cakupan sequence length:"
    )

    print(
        SEQUENCE_LENGTH_COVERAGE_PATH
    )

    print(
        "\nGrafik distribusi teks Kompas:"
    )

    print(
        KOMPAS_WORD_LENGTH_FIGURE
    )

    print(
        "\nGrafik distribusi teks AG News train:"
    )

    print(
        AGNEWS_TRAIN_WORD_LENGTH_FIGURE
    )

    print(
        "\nGrafik distribusi teks AG News test:"
    )

    print(
        AGNEWS_TEST_WORD_LENGTH_FIGURE
    )

    print(
        "\nRingkasan rata-rata jumlah kata:"
    )

    word_statistics = text_statistics[
        text_statistics["unit"].eq(
            "word_count"
        )
    ][
        [
            "dataset",
            "text_field_display",
            "mean",
            "median",
            "p95",
            "p99",
            "maksimum",
        ]
    ].copy()

    print(
        word_statistics.to_string(
            index=False
        )
    )

    print(
        "\nCakupan sequence length utama:"
    )

    main_sequence_coverage = (
        sequence_length_coverage[
            (
                sequence_length_coverage[
                    "dataset"
                ].eq(
                    "kompas"
                )
                & sequence_length_coverage[
                    "text_field"
                ].eq(
                    "title"
                )
                & sequence_length_coverage[
                    "sequence_length"
                ].eq(
                    20
                )
            )
            |
            (
                sequence_length_coverage[
                    "dataset"
                ].eq(
                    "kompas"
                )
                & sequence_length_coverage[
                    "text_field"
                ].eq(
                    "title_description"
                )
                & sequence_length_coverage[
                    "sequence_length"
                ].eq(
                    60
                )
            )
            |
            (
                sequence_length_coverage[
                    "dataset"
                ].eq(
                    "ag_news_train"
                )
                & sequence_length_coverage[
                    "text_field"
                ].eq(
                    "title"
                )
                & sequence_length_coverage[
                    "sequence_length"
                ].eq(
                    20
                )
            )
            |
            (
                sequence_length_coverage[
                    "dataset"
                ].eq(
                    "ag_news_train"
                )
                & sequence_length_coverage[
                    "text_field"
                ].eq(
                    "title_description"
                )
                & sequence_length_coverage[
                    "sequence_length"
                ].eq(
                    60
                )
            )
        ]
    )

    print(
        main_sequence_coverage[
            [
                "dataset",
                "text_field_display",
                "sequence_length",
                "persentase_tertampung",
                "persentase_berpotensi_truncated",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "Tahap text statistics selesai."
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
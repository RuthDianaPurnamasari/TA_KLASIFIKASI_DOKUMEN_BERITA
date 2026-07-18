from __future__ import annotations

import sys
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

from config import TABLES_DIR  # noqa: E402


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
# EXPECTED FINAL COUNTS
# =============================================================================

EXPECTED_FINAL_COUNTS = {
    "kompas": 9_997,
    "ag_news_train": 119_817,
    "ag_news_test": 7_600,
}


# =============================================================================
# REQUIRED DATASET COLUMNS
# =============================================================================

REQUIRED_COLUMNS = {
    "kompas": {
        "document_id",
        "category",
        "title",
        "description",
        "content",
        "date",
    },
    "ag_news_train": {
        "document_id",
        "category",
        "title",
        "description",
    },
    "ag_news_test": {
        "document_id",
        "category",
        "title",
        "description",
    },
}


# =============================================================================
# REQUIRED OUTPUTS FROM PREVIOUS EDA STAGES
# =============================================================================

PREVIOUS_EDA_OUTPUTS = {
    "3.1 Dataset Overview": [
        TABLES_DIR / "dataset_overview.csv",
        TABLES_DIR / "class_distribution.csv",
        TABLES_DIR / "data_cleaning_comparison.csv",
    ],
    "3.2 Text Statistics": [
        TABLES_DIR / "text_statistics.csv",
        TABLES_DIR / "text_statistics_by_category.csv",
        TABLES_DIR / "sequence_length_coverage.csv",
    ],
    "3.3 Missing and Duplicate Analysis": [
        TABLES_DIR / "missing_value_report.csv",
        TABLES_DIR / "duplicate_report.csv",
        TABLES_DIR / "cleaning_integrity_summary.csv",
    ],
    "3.4 Word Frequency Analysis": [
        TABLES_DIR / "word_frequency_overall.csv",
        TABLES_DIR / "word_frequency_by_category.csv",
        TABLES_DIR / "word_frequency_summary.csv",
    ],
    "3.5 Temporal Analysis": [
        TABLES_DIR / "kompas_monthly_distribution.csv",
        TABLES_DIR / "kompas_monthly_category_distribution.csv",
        TABLES_DIR / "kompas_daily_distribution.csv",
        TABLES_DIR / "kompas_hourly_distribution.csv",
        TABLES_DIR / "kompas_weekday_distribution.csv",
        TABLES_DIR / "kompas_temporal_summary.csv",
    ],
    "3.6 Word Cloud Analysis": [
        TABLES_DIR / "wordcloud_summary.csv",
    ],
}


# =============================================================================
# INPUT TABLES USED IN SUMMARY
# =============================================================================

CLEANING_INTEGRITY_PATH = (
    TABLES_DIR
    / "cleaning_integrity_summary.csv"
)

WORD_FREQUENCY_OVERALL_PATH = (
    TABLES_DIR
    / "word_frequency_overall.csv"
)

WORDCLOUD_SUMMARY_PATH = (
    TABLES_DIR
    / "wordcloud_summary.csv"
)

MONTHLY_DISTRIBUTION_PATH = (
    TABLES_DIR
    / "kompas_monthly_distribution.csv"
)

TEMPORAL_SUMMARY_PATH = (
    TABLES_DIR
    / "kompas_temporal_summary.csv"
)


# =============================================================================
# OUTPUT FILES
# =============================================================================

EDA_SUMMARY_PATH = (
    TABLES_DIR
    / "eda_summary.csv"
)

EDA_RESEARCH_FINDINGS_PATH = (
    TABLES_DIR
    / "eda_research_findings.csv"
)

EDA_STAGE_VALIDATION_PATH = (
    TABLES_DIR
    / "eda_stage_validation.csv"
)


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

KOMPAS_SCENARIOS = {
    "K1": "Title",
    "K2": "Title + [SEP] + Description",
    "K3": (
        "Title + [SEP] + Description "
        "+ [SEP] + Keyword YAKE"
    ),
}

AGNEWS_SCENARIOS = {
    "A1": "Title",
    "A2": "Title + [SEP] + Description",
}

SEQUENCE_LENGTHS = {
    "title": 20,
    "title_description": 60,
    "title_description_yake": 60,
}


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def ensure_output_directory() -> None:
    """
    Memastikan folder output tersedia.
    """

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def is_valid_file(
    file_path: Path,
) -> bool:
    """
    Memeriksa apakah suatu file tersedia dan tidak kosong.
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
        "Pastikan tahap data cleaning telah dijalankan."
    )


def read_csv_with_fallback(
    file_path: Path,
) -> pd.DataFrame:
    """
    Membaca CSV menggunakan beberapa encoding.
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


def format_integer(
    value: Any,
) -> str:
    """
    Mengubah bilangan bulat ke format Indonesia.
    """

    return f"{int(value):,}".replace(
        ",",
        ".",
    )


def format_decimal(
    value: Any,
    decimals: int = 2,
) -> str:
    """
    Mengubah bilangan desimal ke format Indonesia.
    """

    formatted = f"{float(value):,.{decimals}f}"

    return (
        formatted
        .replace(",", "TEMP")
        .replace(".", ",")
        .replace("TEMP", ".")
    )


def normalize_category(
    series: pd.Series,
) -> pd.Series:
    """
    Menyeragamkan penamaan kategori.
    """

    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


# =============================================================================
# DATASET LOADER
# =============================================================================

def load_dataset(
    file_path: Path,
    dataset_name: str,
    dataset_key: str,
) -> pd.DataFrame:
    """
    Membaca dan memvalidasi dataset final.
    """

    path = Path(file_path)

    if not is_valid_file(path):
        raise FileNotFoundError(
            f"Dataset {dataset_name} tidak ditemukan "
            f"atau kosong:\n{path}"
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

    missing_columns = (
        REQUIRED_COLUMNS[dataset_key]
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            f"Dataset {dataset_name} tidak memiliki kolom wajib.\n"
            f"Kolom hilang: {sorted(missing_columns)}"
        )

    actual_count = int(
        len(dataframe)
    )

    expected_count = EXPECTED_FINAL_COUNTS[
        dataset_key
    ]

    if actual_count != expected_count:
        raise ValueError(
            f"Jumlah dataset final {dataset_name} tidak sesuai.\n"
            f"Expected : {expected_count:,}\n"
            f"Actual   : {actual_count:,}\n\n"
            "Pastikan file yang digunakan merupakan hasil "
            "cleaning final."
        )

    dataframe["category"] = normalize_category(
        dataframe["category"]
    )

    if dataframe["category"].eq("").any():
        raise ValueError(
            f"Ditemukan kategori kosong pada dataset {dataset_name}."
        )

    return dataframe


# =============================================================================
# TEXT STATISTICS
# =============================================================================

def normalize_text_series(
    series: pd.Series,
) -> pd.Series:
    """
    Menormalisasi nilai teks untuk statistik sederhana.
    """

    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )


def count_empty_text(
    series: pd.Series,
) -> int:
    """
    Menghitung jumlah string kosong dan NaN.
    """

    normalized = normalize_text_series(
        series
    )

    return int(
        normalized.eq("").sum()
    )


def word_count_series(
    series: pd.Series,
) -> pd.Series:
    """
    Menghitung jumlah kata berdasarkan pemisah whitespace.
    """

    normalized = normalize_text_series(
        series
    )

    return normalized.map(
        lambda text: (
            len(text.split())
            if text
            else 0
        )
    )


def average_word_count(
    series: pd.Series,
) -> float:
    """
    Menghitung rata-rata jumlah kata.
    """

    return round(
        float(
            word_count_series(
                series
            ).mean()
        ),
        4,
    )


def combine_title_description(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Menggabungkan Title dan Description.
    """

    title = normalize_text_series(
        dataframe["title"]
    )

    description = normalize_text_series(
        dataframe["description"]
    )

    return (
        title
        + " "
        + description
    ).str.strip()


def calculate_sequence_coverage(
    series: pd.Series,
    max_length: int,
) -> float:
    """
    Menghitung persentase dokumen yang tidak melebihi max length.
    """

    lengths = word_count_series(
        series
    )

    if len(lengths) == 0:
        return 0.0

    return round(
        float(
            lengths.le(
                max_length
            ).mean()
            * 100
        ),
        4,
    )


# =============================================================================
# CATEGORY DISTRIBUTION
# =============================================================================

def create_category_distribution_text(
    dataframe: pd.DataFrame,
) -> str:
    """
    Membentuk distribusi kategori dalam format teks.
    """

    distribution = (
        dataframe["category"]
        .value_counts()
        .sort_index()
    )

    return "; ".join(
        (
            f"{category}: "
            f"{format_integer(count)}"
        )
        for category, count
        in distribution.items()
    )


# =============================================================================
# DATASET SUMMARY
# =============================================================================

def create_dataset_summary(
    dataframe: pd.DataFrame,
    dataset_name: str,
    dataset_key: str,
) -> dict[str, Any]:
    """
    Membuat satu baris ringkasan dataset final.
    """

    title_average = average_word_count(
        dataframe["title"]
    )

    description_average = average_word_count(
        dataframe["description"]
    )

    combined_text = combine_title_description(
        dataframe
    )

    summary: dict[str, Any] = {
        "dataset":
            dataset_key,

        "dataset_display_name":
            dataset_name,

        "stage":
            "setelah_cleaning",

        "jumlah_data":
            int(
                len(dataframe)
            ),

        "jumlah_kategori":
            int(
                dataframe["category"]
                .nunique()
            ),

        "distribusi_kategori":
            create_category_distribution_text(
                dataframe
            ),

        "empty_title":
            count_empty_text(
                dataframe["title"]
            ),

        "empty_description":
            count_empty_text(
                dataframe["description"]
            ),

        "avg_words_title":
            title_average,

        "avg_words_description":
            description_average,

        "avg_words_title_description":
            average_word_count(
                combined_text
            ),

        "coverage_title_max_20_percent":
            calculate_sequence_coverage(
                dataframe["title"],
                SEQUENCE_LENGTHS["title"],
            ),

        "coverage_title_description_max_60_percent":
            calculate_sequence_coverage(
                combined_text,
                SEQUENCE_LENGTHS[
                    "title_description"
                ],
            ),
    }

    if "content" in dataframe.columns:
        summary["empty_content"] = (
            count_empty_text(
                dataframe["content"]
            )
        )

        summary["avg_words_content"] = (
            average_word_count(
                dataframe["content"]
            )
        )

    else:
        summary["empty_content"] = pd.NA
        summary["avg_words_content"] = pd.NA

    if "date" in dataframe.columns:
        parsed_date = pd.to_datetime(
            dataframe["date"],
            errors="coerce",
        )

        if parsed_date.isna().any():
            raise ValueError(
                f"Ditemukan tanggal tidak valid pada {dataset_name}."
            )

        summary["tanggal_awal"] = (
            parsed_date
            .min()
            .strftime("%Y-%m-%d %H:%M:%S")
        )

        summary["tanggal_akhir"] = (
            parsed_date
            .max()
            .strftime("%Y-%m-%d %H:%M:%S")
        )

    else:
        summary["tanggal_awal"] = pd.NA
        summary["tanggal_akhir"] = pd.NA

    return summary


# =============================================================================
# VALIDATE PREVIOUS EDA OUTPUTS
# =============================================================================

def validate_previous_eda_outputs() -> pd.DataFrame:
    """
    Memastikan seluruh output inti tahap 3.1 sampai 3.6 tersedia.
    """

    records: list[dict[str, Any]] = []
    missing_outputs: list[str] = []

    for stage, output_paths in PREVIOUS_EDA_OUTPUTS.items():
        for output_path in output_paths:
            exists = output_path.exists()
            is_file = output_path.is_file()
            size_bytes = (
                output_path.stat().st_size
                if exists and is_file
                else 0
            )

            valid = bool(
                exists
                and is_file
                and size_bytes > 0
            )

            records.append(
                {
                    "stage":
                        stage,

                    "file_name":
                        output_path.name,

                    "file_path":
                        str(
                            output_path.resolve()
                        ),

                    "exists":
                        exists,

                    "size_bytes":
                        int(
                            size_bytes
                        ),

                    "status":
                        (
                            "LULUS"
                            if valid
                            else "GAGAL"
                        ),
                }
            )

            if not valid:
                missing_outputs.append(
                    str(output_path)
                )

    validation = pd.DataFrame(
        records
    )

    if missing_outputs:
        missing_text = "\n".join(
            f"- {path}"
            for path in missing_outputs
        )

        raise FileNotFoundError(
            "Beberapa output EDA sebelumnya belum tersedia "
            "atau masih kosong:\n"
            f"{missing_text}"
        )

    return validation


# =============================================================================
# LOAD PREVIOUS EDA TABLES
# =============================================================================

def load_required_eda_table(
    file_path: Path,
    description: str,
) -> pd.DataFrame:
    """
    Membaca tabel EDA yang telah divalidasi.
    """

    if not is_valid_file(file_path):
        raise FileNotFoundError(
            f"Tabel {description} tidak ditemukan:\n"
            f"{file_path}"
        )

    dataframe = read_csv_with_fallback(
        file_path
    )

    if dataframe.empty:
        raise ValueError(
            f"Tabel {description} kosong:\n"
            f"{file_path}"
        )

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    return dataframe


# =============================================================================
# TABLE LOOKUP HELPERS
# =============================================================================

def get_dataset_row(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> pd.Series:
    """
    Mengambil satu baris berdasarkan nama dataset.
    """

    if "dataset" not in dataframe.columns:
        raise KeyError(
            "Tabel tidak memiliki kolom 'dataset'."
        )

    normalized_dataset = (
        dataframe["dataset"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    matching_rows = dataframe[
        normalized_dataset.eq(
            dataset_name.lower()
        )
    ]

    if matching_rows.empty:
        raise ValueError(
            f"Dataset '{dataset_name}' tidak ditemukan "
            "pada tabel."
        )

    return matching_rows.iloc[0]


def get_top_overall_word(
    word_frequency: pd.DataFrame,
    dataset_name: str,
) -> tuple[str, int]:
    """
    Mengambil token peringkat pertama suatu dataset.
    """

    required_columns = {
        "dataset",
        "rank",
        "word",
        "frequency",
    }

    missing_columns = (
        required_columns
        - set(word_frequency.columns)
    )

    if missing_columns:
        raise KeyError(
            "Tabel word frequency tidak memiliki kolom:\n"
            f"{sorted(missing_columns)}"
        )

    dataset_series = (
        word_frequency["dataset"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    filtered = word_frequency[
        dataset_series.eq(
            dataset_name.lower()
        )
    ].copy()

    if "category" in filtered.columns:
        category_series = (
            filtered["category"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        filtered = filtered[
            category_series.eq("all")
        ]

    if filtered.empty:
        raise ValueError(
            f"Frekuensi kata dataset {dataset_name} tidak ditemukan."
        )

    filtered["rank"] = pd.to_numeric(
        filtered["rank"],
        errors="coerce",
    )

    top_row = (
        filtered
        .sort_values(
            "rank"
        )
        .iloc[0]
    )

    return (
        str(
            top_row["word"]
        ),
        int(
            top_row["frequency"]
        ),
    )


def create_wordcloud_category_text(
    wordcloud_summary: pd.DataFrame,
    dataset_name: str,
) -> str:
    """
    Membentuk ringkasan token dominan per kategori.
    """

    required_columns = {
        "dataset",
        "category",
        "token_teratas",
        "frekuensi_token_teratas",
    }

    missing_columns = (
        required_columns
        - set(wordcloud_summary.columns)
    )

    if missing_columns:
        raise KeyError(
            "Tabel wordcloud summary tidak memiliki kolom:\n"
            f"{sorted(missing_columns)}"
        )

    dataset_series = (
        wordcloud_summary["dataset"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    filtered = wordcloud_summary[
        dataset_series.eq(
            dataset_name.lower()
        )
    ].copy()

    category_series = (
        filtered["category"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    filtered = filtered[
        ~category_series.eq("all")
    ]

    filtered = filtered.sort_values(
        "category"
    )

    if filtered.empty:
        raise ValueError(
            f"Ringkasan word cloud {dataset_name} tidak ditemukan."
        )

    return "; ".join(
        (
            f"{str(row.category).replace('_', '/').title()}: "
            f"{row.token_teratas} "
            f"({format_integer(row.frekuensi_token_teratas)})"
        )
        for row in filtered.itertuples(
            index=False
        )
    )


# =============================================================================
# RESEARCH FINDINGS
# =============================================================================

def create_research_findings(
    kompas: pd.DataFrame,
    agnews_train: pd.DataFrame,
    agnews_test: pd.DataFrame,
    cleaning_integrity: pd.DataFrame,
    word_frequency: pd.DataFrame,
    wordcloud_summary: pd.DataFrame,
    monthly_distribution: pd.DataFrame,
    temporal_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Membuat temuan EDA yang sesuai dengan dataset dan
    desain eksperimen final.
    """

    kompas_cleaning = get_dataset_row(
        cleaning_integrity,
        "kompas",
    )

    agnews_train_cleaning = get_dataset_row(
        cleaning_integrity,
        "ag_news_train",
    )

    agnews_test_cleaning = get_dataset_row(
        cleaning_integrity,
        "ag_news_test",
    )

    kompas_title_average = average_word_count(
        kompas["title"]
    )

    kompas_description_average = average_word_count(
        kompas["description"]
    )

    kompas_content_average = average_word_count(
        kompas["content"]
    )

    kompas_combined = combine_title_description(
        kompas
    )

    agnews_train_combined = combine_title_description(
        agnews_train
    )

    agnews_test_combined = combine_title_description(
        agnews_test
    )

    kompas_title_coverage = (
        calculate_sequence_coverage(
            kompas["title"],
            20,
        )
    )

    kompas_combined_coverage = (
        calculate_sequence_coverage(
            kompas_combined,
            60,
        )
    )

    agnews_train_title_coverage = (
        calculate_sequence_coverage(
            agnews_train["title"],
            20,
        )
    )

    agnews_train_combined_coverage = (
        calculate_sequence_coverage(
            agnews_train_combined,
            60,
        )
    )

    agnews_test_combined_coverage = (
        calculate_sequence_coverage(
            agnews_test_combined,
            60,
        )
    )

    kompas_top_word, kompas_top_frequency = (
        get_top_overall_word(
            word_frequency,
            "kompas",
        )
    )

    agnews_train_top_word, agnews_train_top_frequency = (
        get_top_overall_word(
            word_frequency,
            "ag_news_train",
        )
    )

    kompas_category_words = (
        create_wordcloud_category_text(
            wordcloud_summary,
            "kompas",
        )
    )

    agnews_category_words = (
        create_wordcloud_category_text(
            wordcloud_summary,
            "ag_news_train",
        )
    )

    temporal_row = temporal_summary.iloc[0]

    month_rows = monthly_distribution.copy()

    month_rows["month_period"] = (
        month_rows["month_period"]
        .astype(str)
    )

    monthly_text = "; ".join(
        (
            f"{row.month_period}: "
            f"{format_integer(row.jumlah_berita)} "
            f"({format_decimal(row.persentase_dataset)}%)"
        )
        for row in month_rows.itertuples(
            index=False
        )
    )

    kompas_scenario_text = "; ".join(
        f"{code} = {description}"
        for code, description
        in KOMPAS_SCENARIOS.items()
    )

    agnews_scenario_text = "; ".join(
        f"{code} = {description}"
        for code, description
        in AGNEWS_SCENARIOS.items()
    )

    all_empty_values = (
        count_empty_text(
            kompas["title"]
        )
        + count_empty_text(
            kompas["description"]
        )
        + count_empty_text(
            kompas["content"]
        )
        + count_empty_text(
            agnews_train["title"]
        )
        + count_empty_text(
            agnews_train["description"]
        )
        + count_empty_text(
            agnews_test["title"]
        )
        + count_empty_text(
            agnews_test["description"]
        )
    )

    findings = [
        {
            "no":
                1,

            "aspek":
                "Hasil data cleaning",

            "temuan":
                (
                    "Jumlah Kompas berubah dari "
                    f"{format_integer(kompas_cleaning['jumlah_sebelum_cleaning'])} "
                    "menjadi "
                    f"{format_integer(kompas_cleaning['jumlah_setelah_cleaning'])} "
                    "artikel. AG News Train berubah dari "
                    f"{format_integer(agnews_train_cleaning['jumlah_sebelum_cleaning'])} "
                    "menjadi "
                    f"{format_integer(agnews_train_cleaning['jumlah_setelah_cleaning'])} "
                    "artikel, sedangkan AG News Test tetap "
                    f"{format_integer(agnews_test_cleaning['jumlah_setelah_cleaning'])} "
                    "artikel."
                ),

            "implikasi_penelitian":
                (
                    "Seluruh tahap EDA, preprocessing, modeling, "
                    "evaluasi, dan deployment harus menggunakan "
                    "dataset setelah cleaning."
                ),

            "sumber_output":
                "cleaning_integrity_summary.csv",
        },
        {
            "no":
                2,

            "aspek":
                "Kualitas dataset final",

            "temuan":
                (
                    f"Jumlah nilai teks kosong pada seluruh dataset "
                    f"final adalah {format_integer(all_empty_values)}. "
                    "Audit akhir juga menunjukkan tidak terdapat "
                    "duplikat artikel, konflik label, maupun overlap "
                    "AG News Train-Test."
                ),

            "implikasi_penelitian":
                (
                    "Dataset final memenuhi pemeriksaan integritas "
                    "dasar dan dapat digunakan untuk eksperimen tanpa "
                    "imputasi teks kosong."
                ),

            "sumber_output":
                (
                    "missing_value_report.csv; "
                    "duplicate_report.csv; "
                    "cleaning_integrity_summary.csv"
                ),
        },
        {
            "no":
                3,

            "aspek":
                "Distribusi kelas Kompas",

            "temuan":
                (
                    "Dataset Kompas final terdiri dari "
                    f"{format_integer(len(kompas))} artikel: "
                    f"{create_category_distribution_text(kompas)}."
                ),

            "implikasi_penelitian":
                (
                    "Distribusi kelas hampir seimbang sehingga "
                    "eksperimen tidak memerlukan oversampling atau "
                    "undersampling."
                ),

            "sumber_output":
                "class_distribution.csv",
        },
        {
            "no":
                4,

            "aspek":
                "Distribusi kelas AG News",

            "temuan":
                (
                    "AG News Train final terdiri dari "
                    f"{format_integer(len(agnews_train))} artikel: "
                    f"{create_category_distribution_text(agnews_train)}. "
                    "AG News Test terdiri dari "
                    f"{format_integer(len(agnews_test))} artikel: "
                    f"{create_category_distribution_text(agnews_test)}."
                ),

            "implikasi_penelitian":
                (
                    "AG News digunakan sebagai benchmark eksternal "
                    "dengan distribusi kelas yang tetap sangat seimbang."
                ),

            "sumber_output":
                "class_distribution.csv",
        },
        {
            "no":
                5,

            "aspek":
                "Karakteristik panjang teks Kompas",

            "temuan":
                (
                    "Rata-rata panjang Title, Description, dan Content "
                    "Kompas masing-masing adalah "
                    f"{format_decimal(kompas_title_average)} kata, "
                    f"{format_decimal(kompas_description_average)} kata, "
                    "dan "
                    f"{format_decimal(kompas_content_average)} kata."
                ),

            "implikasi_penelitian":
                (
                    "Title digunakan sebagai representasi paling ringkas, "
                    "sedangkan Description menyediakan konteks tambahan. "
                    "Content tidak digunakan dalam skenario final karena "
                    "jauh lebih panjang dan berpotensi meningkatkan "
                    "kompleksitas secara tidak proporsional."
                ),

            "sumber_output":
                "text_statistics.csv",
        },
        {
            "no":
                6,

            "aspek":
                "Pemilihan panjang sequence",

            "temuan":
                (
                    "Cakupan Title Kompas pada panjang 20 adalah "
                    f"{format_decimal(kompas_title_coverage, 4)}%, "
                    "sedangkan Title + Description pada panjang 60 "
                    "mencakup "
                    f"{format_decimal(kompas_combined_coverage, 4)}%. "
                    "Pada AG News Train, cakupan Title panjang 20 adalah "
                    f"{format_decimal(agnews_train_title_coverage, 4)}% "
                    "dan Title + Description panjang 60 adalah "
                    f"{format_decimal(agnews_train_combined_coverage, 4)}%. "
                    "Cakupan gabungan pada AG News Test adalah "
                    f"{format_decimal(agnews_test_combined_coverage, 4)}%."
                ),

            "implikasi_penelitian":
                (
                    "Panjang sequence 20 digunakan untuk skenario Title, "
                    "sedangkan panjang 60 digunakan untuk skenario "
                    "Title + Description dan Title + Description + YAKE."
                ),

            "sumber_output":
                "sequence_length_coverage.csv",
        },
        {
            "no":
                7,

            "aspek":
                "Frekuensi kata keseluruhan",

            "temuan":
                (
                    f"Token teratas pada Kompas adalah "
                    f"'{kompas_top_word}' dengan "
                    f"{format_integer(kompas_top_frequency)} kemunculan. "
                    "Token teratas pada AG News Train adalah "
                    f"'{agnews_train_top_word}' dengan "
                    f"{format_integer(agnews_train_top_frequency)} "
                    "kemunculan."
                ),

            "implikasi_penelitian":
                (
                    "Frekuensi kata digunakan untuk memahami "
                    "karakteristik korpus dan tidak diinterpretasikan "
                    "sebagai feature importance model."
                ),

            "sumber_output":
                "word_frequency_overall.csv",
        },
        {
            "no":
                8,

            "aspek":
                "Kosakata dominan per kategori",

            "temuan":
                (
                    f"Token dominan Kompas per kategori adalah "
                    f"{kompas_category_words}. Token dominan AG News "
                    f"Train per kategori adalah {agnews_category_words}."
                ),

            "implikasi_penelitian":
                (
                    "Perbedaan token dominan menunjukkan bahwa setiap "
                    "kategori memiliki pola kosakata yang dapat "
                    "dipelajari oleh CNN dan Attention-BiLSTM."
                ),

            "sumber_output":
                "wordcloud_summary.csv",
        },
        {
            "no":
                9,

            "aspek":
                "Distribusi temporal Kompas",

            "temuan":
                (
                    "Artikel Kompas mencakup periode "
                    f"{temporal_row['tanggal_awal']} sampai "
                    f"{temporal_row['tanggal_akhir']}. Distribusi bulanan "
                    f"adalah {monthly_text}. Tanggal teraktif adalah "
                    f"{temporal_row['tanggal_teraktif']} dengan "
                    f"{format_integer(temporal_row['jumlah_berita_tanggal_teraktif'])} "
                    "artikel dan jam teraktif adalah "
                    f"{temporal_row['rentang_jam_teraktif']}."
                ),

            "implikasi_penelitian":
                (
                    "Distribusi waktu didokumentasikan sebagai "
                    "karakteristik dataset hasil crawling dan tidak "
                    "digeneralisasi sebagai tren produksi berita "
                    "Kompas secara keseluruhan."
                ),

            "sumber_output":
                (
                    "kompas_monthly_distribution.csv; "
                    "kompas_temporal_summary.csv"
                ),
        },
        {
            "no":
                10,

            "aspek":
                "Skenario representasi teks Kompas",

            "temuan":
                kompas_scenario_text,

            "implikasi_penelitian":
                (
                    "Eksperimen bertingkat digunakan untuk mengukur "
                    "kontribusi Description dan keyword YAKE terhadap "
                    "kinerja klasifikasi. Keyword YAKE diekstraksi dari "
                    "gabungan Title dan Description, bukan dari Content."
                ),

            "sumber_output":
                "desain eksperimen final",
        },
        {
            "no":
                11,

            "aspek":
                "Skenario representasi teks AG News",

            "temuan":
                agnews_scenario_text,

            "implikasi_penelitian":
                (
                    "AG News hanya diuji menggunakan Title dan "
                    "Title + Description karena dataset tidak menyediakan "
                    "Content dan keyword YAKE tidak diterapkan pada "
                    "benchmark."
                ),

            "sumber_output":
                "desain eksperimen final",
        },
        {
            "no":
                12,

            "aspek":
                "Implikasi terhadap perbandingan model",

            "temuan":
                (
                    "CNN dan Attention-BiLSTM diuji menggunakan split, "
                    "label mapping, representasi teks, vocabulary, dan "
                    "panjang sequence yang sama pada setiap skenario."
                ),

            "implikasi_penelitian":
                (
                    "Pengendalian konfigurasi tersebut memastikan "
                    "perbedaan hasil terutama berasal dari karakteristik "
                    "arsitektur model, bukan dari perbedaan data input."
                ),

            "sumber_output":
                "pipeline eksperimen final",
        },
    ]

    return pd.DataFrame(
        findings
    )


# =============================================================================
# TERMINAL DISPLAY
# =============================================================================

def print_dataset_summary(
    summary_dataframe: pd.DataFrame,
) -> None:
    """
    Menampilkan ringkasan dataset.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        "RINGKASAN DATASET FINAL"
    )

    print(
        "=" * 80
    )

    display_columns = [
        "dataset",
        "jumlah_data",
        "jumlah_kategori",
        "empty_title",
        "empty_description",
        "avg_words_title",
        "avg_words_description",
        "avg_words_title_description",
    ]

    print(
        summary_dataframe[
            display_columns
        ].to_string(
            index=False
        )
    )


def print_research_findings(
    findings_dataframe: pd.DataFrame,
) -> None:
    """
    Menampilkan temuan penelitian.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TEMUAN UTAMA EDA"
    )

    print(
        "=" * 80
    )

    for row in findings_dataframe.itertuples(
        index=False
    ):
        print(
            f"\n{row.no}. {row.aspek}"
        )

        print(
            f"   Temuan    : {row.temuan}"
        )

        print(
            f"   Implikasi : {row.implikasi_penelitian}"
        )

        print(
            f"   Sumber    : {row.sumber_output}"
        )


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main() -> None:
    """
    Menjalankan EDA tahap 3.7 — EDA Summary.
    """

    print(
        "=" * 80
    )

    print(
        "STEP 3.7 - EDA SUMMARY"
    )

    print(
        "=" * 80
    )

    ensure_output_directory()

    # =========================================================================
    # VALIDATE PREVIOUS EDA OUTPUTS
    # =========================================================================

    stage_validation = (
        validate_previous_eda_outputs()
    )

    # =========================================================================
    # RESOLVE FINAL DATASETS
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
    )

    agnews_train = load_dataset(
        file_path=agnews_train_path,
        dataset_name="AG News Train",
        dataset_key="ag_news_train",
    )

    agnews_test = load_dataset(
        file_path=agnews_test_path,
        dataset_name="AG News Test",
        dataset_key="ag_news_test",
    )

    print(
        "\nDataset berhasil dimuat:"
    )

    print(
        f"Kompas        : {len(kompas):,}"
    )

    print(
        f"AG News Train : {len(agnews_train):,}"
    )

    print(
        f"AG News Test  : {len(agnews_test):,}"
    )

    # =========================================================================
    # LOAD PREVIOUS EDA TABLES
    # =========================================================================

    cleaning_integrity = (
        load_required_eda_table(
            CLEANING_INTEGRITY_PATH,
            "integritas cleaning",
        )
    )

    word_frequency = (
        load_required_eda_table(
            WORD_FREQUENCY_OVERALL_PATH,
            "frekuensi kata keseluruhan",
        )
    )

    wordcloud_summary = (
        load_required_eda_table(
            WORDCLOUD_SUMMARY_PATH,
            "ringkasan word cloud",
        )
    )

    monthly_distribution = (
        load_required_eda_table(
            MONTHLY_DISTRIBUTION_PATH,
            "distribusi bulanan Kompas",
        )
    )

    temporal_summary = (
        load_required_eda_table(
            TEMPORAL_SUMMARY_PATH,
            "ringkasan temporal Kompas",
        )
    )

    # =========================================================================
    # CREATE DATASET SUMMARY
    # =========================================================================

    summary_rows = [
        create_dataset_summary(
            dataframe=kompas,
            dataset_name="Kompas",
            dataset_key="kompas",
        ),
        create_dataset_summary(
            dataframe=agnews_train,
            dataset_name="AG News Train",
            dataset_key="ag_news_train",
        ),
        create_dataset_summary(
            dataframe=agnews_test,
            dataset_name="AG News Test",
            dataset_key="ag_news_test",
        ),
    ]

    summary_dataframe = pd.DataFrame(
        summary_rows
    )

    # =========================================================================
    # CREATE RESEARCH FINDINGS
    # =========================================================================

    findings_dataframe = (
        create_research_findings(
            kompas=kompas,
            agnews_train=agnews_train,
            agnews_test=agnews_test,
            cleaning_integrity=cleaning_integrity,
            word_frequency=word_frequency,
            wordcloud_summary=wordcloud_summary,
            monthly_distribution=monthly_distribution,
            temporal_summary=temporal_summary,
        )
    )

    # =========================================================================
    # VALIDATE FINAL SUMMARY
    # =========================================================================

    if len(summary_dataframe) != 3:
        raise ValueError(
            "Ringkasan dataset harus terdiri dari tiga baris."
        )

    if summary_dataframe[
        [
            "empty_title",
            "empty_description",
        ]
    ].sum().sum() != 0:
        raise ValueError(
            "Masih terdapat Title atau Description kosong "
            "pada ringkasan dataset final."
        )

    if int(
        findings_dataframe["no"].nunique()
    ) != len(findings_dataframe):
        raise ValueError(
            "Nomor temuan EDA tidak unik."
        )

    if findings_dataframe[
        [
            "aspek",
            "temuan",
            "implikasi_penelitian",
            "sumber_output",
        ]
    ].isna().any().any():
        raise ValueError(
            "Ditemukan nilai kosong pada tabel temuan EDA."
        )

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================

    summary_dataframe.to_csv(
        EDA_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    findings_dataframe.to_csv(
        EDA_RESEARCH_FINDINGS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    stage_validation.to_csv(
        EDA_STAGE_VALIDATION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # VALIDATE OUTPUT FILES
    # =========================================================================

    output_files = [
        (
            EDA_SUMMARY_PATH,
            "ringkasan dataset EDA",
        ),
        (
            EDA_RESEARCH_FINDINGS_PATH,
            "temuan penelitian EDA",
        ),
        (
            EDA_STAGE_VALIDATION_PATH,
            "validasi tahap EDA",
        ),
    ]

    for file_path, description in output_files:
        validate_output_file(
            file_path=file_path,
            description=description,
        )

    # =========================================================================
    # DISPLAY RESULTS
    # =========================================================================

    print_dataset_summary(
        summary_dataframe
    )

    print_research_findings(
        findings_dataframe
    )

    # =========================================================================
    # OUTPUT INFORMATION
    # =========================================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "OUTPUT EDA SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        "\nRingkasan karakteristik dataset:"
    )

    print(
        EDA_SUMMARY_PATH
    )

    print(
        "\nTemuan dan implikasi penelitian:"
    )

    print(
        EDA_RESEARCH_FINDINGS_PATH
    )

    print(
        "\nValidasi output tahap EDA:"
    )

    print(
        EDA_STAGE_VALIDATION_PATH
    )

    print(
        "\nRingkasan validasi tahap:"
    )

    print(
        stage_validation
        .groupby(
            [
                "stage",
                "status",
            ]
        )
        .size()
        .reset_index(
            name="jumlah_file"
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nCatatan:"
    )

    print(
        "- Seluruh ringkasan menggunakan dataset setelah cleaning."
    )

    print(
        "- Kompas menggunakan tiga skenario K1, K2, dan K3."
    )

    print(
        "- AG News menggunakan dua skenario A1 dan A2."
    )

    print(
        "- YAKE diekstraksi dari Title dan Description."
    )

    print(
        "- Content dianalisis pada EDA tetapi tidak digunakan "
        "pada skenario eksperimen final."
    )

    print(
        "- AG News Test hanya digunakan untuk evaluasi akhir."
    )

    print(
        "\nValidasi seluruh output EDA: LULUS"
    )

    print(
        "Tahap EDA selesai."
    )


if __name__ == "__main__":
    main()
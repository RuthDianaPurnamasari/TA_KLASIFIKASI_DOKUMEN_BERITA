from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter
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

WORD_FREQUENCY_OVERALL_PATH = (
    TABLES_DIR
    / "word_frequency_overall.csv"
)

WORD_FREQUENCY_BY_CATEGORY_PATH = (
    TABLES_DIR
    / "word_frequency_by_category.csv"
)

WORD_FREQUENCY_SUMMARY_PATH = (
    TABLES_DIR
    / "word_frequency_summary.csv"
)

KOMPAS_WORD_FREQUENCY_FIGURE = (
    FIGURES_DIR
    / "kompas_top_words.png"
)

AGNEWS_TRAIN_WORD_FREQUENCY_FIGURE = (
    FIGURES_DIR
    / "agnews_train_top_words.png"
)

AGNEWS_TEST_WORD_FREQUENCY_FIGURE = (
    FIGURES_DIR
    / "agnews_test_top_words.png"
)


# =============================================================================
# EXPECTED FINAL DATASET COUNTS
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

REQUIRED_COLUMNS = {
    "document_id",
    "category",
    "title",
    "description",
}


# =============================================================================
# ANALYSIS CONFIGURATION
# =============================================================================

TEXT_COLUMNS = [
    "title",
    "description",
]

TOP_N_TABLE = 100
TOP_N_CATEGORY = 50
TOP_N_FIGURE = 20

MINIMUM_TOKEN_LENGTH = 3


# =============================================================================
# STOPWORDS INDONESIA
# =============================================================================

INDONESIAN_STOPWORDS = {
    "yang",
    "dan",
    "di",
    "ke",
    "dari",
    "untuk",
    "dengan",
    "pada",
    "dalam",
    "ini",
    "itu",
    "adalah",
    "sebagai",
    "oleh",
    "akan",
    "atau",
    "juga",
    "karena",
    "ada",
    "tidak",
    "sudah",
    "telah",
    "bisa",
    "dapat",
    "lebih",
    "setelah",
    "saat",
    "menjadi",
    "hingga",
    "antara",
    "terhadap",
    "sebuah",
    "para",
    "masih",
    "yakni",
    "yaitu",
    "ia",
    "mereka",
    "kami",
    "kita",
    "saya",
    "anda",
    "dia",
    "nya",
    "pun",
    "per",
    "jadi",
    "tak",
    "baru",
    "usai",
    "mulai",
    "hari",
    "tengah",
    "bakal",
    "sebut",
    "kata",
    "ungkap",
    "tahun",
    "hari",
    "tersebut",
    "terkait",
    "menurut",
    "namun",
    "agar",
    "jika",
    "maka",
    "salah",
    "satu",
    "kompas",
    "kompascom",
    "com",
}


# =============================================================================
# STOPWORDS INGGRIS
# =============================================================================

ENGLISH_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "from",
    "with",
    "by",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "he",
    "she",
    "they",
    "we",
    "you",
    "his",
    "her",
    "their",
    "our",
    "your",
    "not",
    "has",
    "have",
    "had",
    "will",
    "would",
    "can",
    "could",
    "may",
    "might",
    "do",
    "does",
    "did",
    "than",
    "after",
    "before",
    "into",
    "over",
    "under",
    "about",
    "up",
    "out",
    "new",
    "says",
    "said",
    "two",
    "first",
    "year",
    "years",
    "more",
    "one",
    "last",
    "today",
    "yesterday",
    "tomorrow",
    "who",
    "what",
    "when",
    "where",
    "which",
    "while",
    "also",
    "some",
    "other",
    "all",
    "any",
    "now",
    "only",
    "most",
    "much",
    "many",
    "reuters",
    "ap",
    "afp",
    "quot",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}


# =============================================================================
# REGEX
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

# Mengambil rangkaian karakter huruf Unicode.
# Angka dan underscore tidak disertakan.
LETTER_TOKEN_PATTERN = re.compile(
    r"[^\W\d_]+",
    flags=re.UNICODE,
)


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def ensure_output_directories() -> None:
    """
    Memastikan folder tabel dan grafik tersedia.
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
    Memilih dataset final pertama yang ditemukan.
    """

    for candidate in candidates:
        if is_valid_file(candidate):
            return candidate

    paths = "\n".join(
        f"- {candidate}"
        for candidate in candidates
    )

    raise FileNotFoundError(
        f"Dataset final {dataset_name} tidak ditemukan.\n\n"
        f"Path yang diperiksa:\n{paths}\n\n"
        "Pastikan tahap cleaning telah dijalankan."
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


def validate_required_columns(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Memastikan dataset mempunyai kolom yang dibutuhkan.
    """

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            f"Dataset {dataset_name} tidak memiliki "
            "kolom yang dibutuhkan.\n"
            f"Kolom hilang: {sorted(missing_columns)}"
        )


def validate_dataset_count(
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
            "Pastikan file yang digunakan adalah dataset "
            "setelah cleaning."
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
            f"Dataset {dataset_name} tidak mempunyai baris data."
        )

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    validate_required_columns(
        dataframe=dataframe,
        dataset_name=dataset_name,
    )

    validate_dataset_count(
        dataframe=dataframe,
        dataset_key=dataset_key,
    )

    return dataframe


# =============================================================================
# TEXT NORMALIZATION AND TOKENIZATION
# =============================================================================

def normalize_text_for_frequency(
    value: Any,
) -> str:
    """
    Menormalisasi teks khusus untuk EDA frekuensi kata.

    Proses ini tidak mengubah dataset asli dan tidak digunakan
    sebagai input langsung model.
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

    text = text.casefold()

    text = HTML_TAG_PATTERN.sub(
        " ",
        text,
    )

    text = URL_PATTERN.sub(
        " ",
        text,
    )

    text = WHITESPACE_PATTERN.sub(
        " ",
        text,
    )

    return text.strip()


def tokenize_text(
    value: Any,
    stopwords: set[str],
) -> list[str]:
    """
    Mengubah teks menjadi token untuk analisis frekuensi.

    Token yang dipertahankan:
    - hanya karakter alfabet;
    - panjang minimal tiga karakter;
    - bukan stopword.
    """

    text = normalize_text_for_frequency(
        value
    )

    tokens = LETTER_TOKEN_PATTERN.findall(
        text
    )

    filtered_tokens = [
        token
        for token in tokens
        if (
            len(token) >= MINIMUM_TOKEN_LENGTH
            and token not in stopwords
        )
    ]

    return filtered_tokens


# =============================================================================
# COMBINE TEXT FIELDS
# =============================================================================

def combine_text_columns(
    dataframe: pd.DataFrame,
    text_columns: list[str],
) -> pd.Series:
    """
    Menggabungkan Title dan Description.

    Content tidak digunakan agar frekuensi tidak didominasi
    artikel yang jauh lebih panjang.
    """

    missing_columns = [
        column
        for column in text_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            "Kolom teks tidak ditemukan: "
            f"{missing_columns}"
        )

    result = pd.Series(
        "",
        index=dataframe.index,
        dtype="object",
    )

    for column in text_columns:
        values = (
            dataframe[column]
            .fillna("")
            .astype(str)
        )

        result = (
            result
            + " "
            + values
        )

    return (
        result
        .str.strip()
    )


# =============================================================================
# WORD FREQUENCY CALCULATION
# =============================================================================

def calculate_word_frequency(
    texts: pd.Series,
    stopwords: set[str],
    top_n: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Menghitung:

    - jumlah kemunculan token;
    - persentase terhadap seluruh token;
    - jumlah dokumen yang mengandung token;
    - persentase dokumen yang mengandung token.
    """

    token_counter: Counter[str] = Counter()
    document_counter: Counter[str] = Counter()

    total_documents = int(
        len(texts)
    )

    total_tokens = 0
    empty_token_documents = 0

    for text in texts:
        tokens = tokenize_text(
            value=text,
            stopwords=stopwords,
        )

        if not tokens:
            empty_token_documents += 1
            continue

        token_counter.update(
            tokens
        )

        document_counter.update(
            set(tokens)
        )

        total_tokens += len(tokens)

    records: list[dict[str, Any]] = []

    for rank, (
        word,
        frequency,
    ) in enumerate(
        token_counter.most_common(
            top_n
        ),
        start=1,
    ):
        document_frequency = int(
            document_counter[word]
        )

        records.append(
            {
                "rank":
                    rank,

                "word":
                    word,

                "frequency":
                    int(
                        frequency
                    ),

                "relative_frequency_percent":
                    round(
                        frequency
                        / total_tokens
                        * 100,
                        6,
                    )
                    if total_tokens > 0
                    else 0.0,

                "document_frequency":
                    document_frequency,

                "document_percentage":
                    round(
                        document_frequency
                        / total_documents
                        * 100,
                        6,
                    )
                    if total_documents > 0
                    else 0.0,
            }
        )

    frequency = pd.DataFrame(
        records,
        columns=[
            "rank",
            "word",
            "frequency",
            "relative_frequency_percent",
            "document_frequency",
            "document_percentage",
        ],
    )

    summary = {
        "jumlah_dokumen":
            total_documents,

        "total_token_setelah_filter":
            int(
                total_tokens
            ),

        "jumlah_token_unik":
            int(
                len(token_counter)
            ),

        "rata_rata_token_per_dokumen":
            round(
                total_tokens
                / total_documents,
                4,
            )
            if total_documents > 0
            else 0.0,

        "dokumen_tanpa_token_setelah_filter":
            int(
                empty_token_documents
            ),
    }

    return (
        frequency,
        summary,
    )


# =============================================================================
# OVERALL FREQUENCY
# =============================================================================

def create_overall_frequency(
    dataframe: pd.DataFrame,
    dataset_name: str,
    text_columns: list[str],
    stopwords: set[str],
    top_n: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Menghasilkan frekuensi kata keseluruhan dataset.
    """

    combined_text = combine_text_columns(
        dataframe=dataframe,
        text_columns=text_columns,
    )

    frequency, summary = (
        calculate_word_frequency(
            texts=combined_text,
            stopwords=stopwords,
            top_n=top_n,
        )
    )

    frequency.insert(
        0,
        "dataset",
        dataset_name,
    )

    frequency.insert(
        1,
        "category",
        "all",
    )

    frequency.insert(
        2,
        "text_source",
        " + ".join(text_columns),
    )

    summary_record = {
        "dataset":
            dataset_name,

        "category":
            "all",

        "text_source":
            " + ".join(text_columns),

        **summary,
    }

    return (
        frequency,
        summary_record,
    )


# =============================================================================
# FREQUENCY BY CATEGORY
# =============================================================================

def create_frequency_by_category(
    dataframe: pd.DataFrame,
    dataset_name: str,
    text_columns: list[str],
    stopwords: set[str],
    top_n: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Menghasilkan frekuensi kata pada setiap kategori.
    """

    frequency_frames: list[pd.DataFrame] = []
    summary_records: list[dict[str, Any]] = []

    grouped = dataframe.groupby(
        "category",
        dropna=False,
        sort=True,
    )

    for category, group in grouped:
        category_name = str(
            category
        )

        combined_text = combine_text_columns(
            dataframe=group,
            text_columns=text_columns,
        )

        frequency, summary = (
            calculate_word_frequency(
                texts=combined_text,
                stopwords=stopwords,
                top_n=top_n,
            )
        )

        frequency.insert(
            0,
            "dataset",
            dataset_name,
        )

        frequency.insert(
            1,
            "category",
            category_name,
        )

        frequency.insert(
            2,
            "text_source",
            " + ".join(text_columns),
        )

        frequency_frames.append(
            frequency
        )

        summary_records.append(
            {
                "dataset":
                    dataset_name,

                "category":
                    category_name,

                "text_source":
                    " + ".join(text_columns),

                **summary,
            }
        )

    if not frequency_frames:
        return (
            pd.DataFrame(),
            summary_records,
        )

    return (
        pd.concat(
            frequency_frames,
            ignore_index=True,
        ),
        summary_records,
    )


# =============================================================================
# TERMINAL DISPLAY
# =============================================================================

def display_top_words(
    frequency: pd.DataFrame,
    dataset_name: str,
    top_n: int = 20,
) -> None:
    """
    Menampilkan kata teratas di terminal.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"TOP WORDS — {dataset_name.upper()}"
    )

    print(
        "=" * 80
    )

    if frequency.empty:
        print(
            "Data frekuensi kata kosong."
        )
        return

    print(
        frequency[
            [
                "rank",
                "word",
                "frequency",
                "relative_frequency_percent",
                "document_frequency",
                "document_percentage",
            ]
        ]
        .head(
            top_n
        )
        .to_string(
            index=False
        )
    )


# =============================================================================
# TOP-WORD FIGURE
# =============================================================================

def plot_top_words(
    frequency: pd.DataFrame,
    title: str,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """
    Membuat horizontal bar chart token teratas.
    """

    if frequency.empty:
        raise ValueError(
            f"Data frekuensi kosong untuk grafik {title}."
        )

    plot_data = (
        frequency
        .head(top_n)
        .sort_values(
            "frequency",
            ascending=True,
        )
        .copy()
    )

    figure, axis = plt.subplots(
        figsize=(
            11,
            8,
        )
    )

    bars = axis.barh(
        plot_data["word"],
        plot_data["frequency"],
    )

    axis.set_title(
        title,
        fontsize=14,
        pad=15,
    )

    axis.set_xlabel(
        "Frekuensi Kemunculan",
        fontsize=11,
    )

    axis.set_ylabel(
        "Token",
        fontsize=11,
    )

    axis.grid(
        axis="x",
        linestyle="--",
        alpha=0.3,
    )

    axis.set_axisbelow(
        True
    )

    maximum_value = int(
        plot_data["frequency"].max()
    )

    axis.set_xlim(
        0,
        max(
            1,
            int(
                maximum_value
                * 1.16
            ),
        ),
    )

    for bar, value in zip(
        bars,
        plot_data["frequency"],
    ):
        axis.text(
            float(value)
            + maximum_value
            * 0.01,
            bar.get_y()
            + bar.get_height()
            / 2,
            f"{int(value):,}",
            va="center",
            fontsize=9,
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
    Menjalankan EDA tahap 3.4:

    - frekuensi kata keseluruhan;
    - frekuensi kata per kategori;
    - document frequency;
    - ringkasan token;
    - grafik 20 token teratas.
    """

    print(
        "=" * 80
    )

    print(
        "STEP 3.4 - WORD FREQUENCY ANALYSIS"
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
    # OVERALL WORD FREQUENCY
    # =========================================================================

    (
        kompas_overall,
        kompas_summary,
    ) = create_overall_frequency(
        dataframe=kompas,
        dataset_name="kompas",
        text_columns=TEXT_COLUMNS,
        stopwords=INDONESIAN_STOPWORDS,
        top_n=TOP_N_TABLE,
    )

    (
        agnews_train_overall,
        agnews_train_summary,
    ) = create_overall_frequency(
        dataframe=agnews_train,
        dataset_name="ag_news_train",
        text_columns=TEXT_COLUMNS,
        stopwords=ENGLISH_STOPWORDS,
        top_n=TOP_N_TABLE,
    )

    (
        agnews_test_overall,
        agnews_test_summary,
    ) = create_overall_frequency(
        dataframe=agnews_test,
        dataset_name="ag_news_test",
        text_columns=TEXT_COLUMNS,
        stopwords=ENGLISH_STOPWORDS,
        top_n=TOP_N_TABLE,
    )

    overall_frequency = pd.concat(
        [
            kompas_overall,
            agnews_train_overall,
            agnews_test_overall,
        ],
        ignore_index=True,
    )

    overall_frequency.to_csv(
        WORD_FREQUENCY_OVERALL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # WORD FREQUENCY BY CATEGORY
    # =========================================================================

    (
        kompas_by_category,
        kompas_category_summary,
    ) = create_frequency_by_category(
        dataframe=kompas,
        dataset_name="kompas",
        text_columns=TEXT_COLUMNS,
        stopwords=INDONESIAN_STOPWORDS,
        top_n=TOP_N_CATEGORY,
    )

    (
        agnews_train_by_category,
        agnews_train_category_summary,
    ) = create_frequency_by_category(
        dataframe=agnews_train,
        dataset_name="ag_news_train",
        text_columns=TEXT_COLUMNS,
        stopwords=ENGLISH_STOPWORDS,
        top_n=TOP_N_CATEGORY,
    )

    (
        agnews_test_by_category,
        agnews_test_category_summary,
    ) = create_frequency_by_category(
        dataframe=agnews_test,
        dataset_name="ag_news_test",
        text_columns=TEXT_COLUMNS,
        stopwords=ENGLISH_STOPWORDS,
        top_n=TOP_N_CATEGORY,
    )

    frequency_by_category = pd.concat(
        [
            kompas_by_category,
            agnews_train_by_category,
            agnews_test_by_category,
        ],
        ignore_index=True,
    )

    frequency_by_category.to_csv(
        WORD_FREQUENCY_BY_CATEGORY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # WORD-FREQUENCY SUMMARY
    # =========================================================================

    summary_records = [
        kompas_summary,
        agnews_train_summary,
        agnews_test_summary,
        *kompas_category_summary,
        *agnews_train_category_summary,
        *agnews_test_category_summary,
    ]

    frequency_summary = pd.DataFrame(
        summary_records
    )

    frequency_summary.to_csv(
        WORD_FREQUENCY_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # TERMINAL DISPLAY
    # =========================================================================

    display_top_words(
        frequency=kompas_overall,
        dataset_name="Kompas",
        top_n=TOP_N_FIGURE,
    )

    display_top_words(
        frequency=agnews_train_overall,
        dataset_name="AG News Train",
        top_n=TOP_N_FIGURE,
    )

    display_top_words(
        frequency=agnews_test_overall,
        dataset_name="AG News Test",
        top_n=TOP_N_FIGURE,
    )

    # =========================================================================
    # FIGURES
    # =========================================================================

    plot_top_words(
        frequency=kompas_overall,
        title=(
            "20 Token Paling Sering Muncul "
            "pada Title dan Description Kompas"
        ),
        output_path=KOMPAS_WORD_FREQUENCY_FIGURE,
        top_n=TOP_N_FIGURE,
    )

    plot_top_words(
        frequency=agnews_train_overall,
        title=(
            "20 Token Paling Sering Muncul "
            "pada Title dan Description AG News Train"
        ),
        output_path=AGNEWS_TRAIN_WORD_FREQUENCY_FIGURE,
        top_n=TOP_N_FIGURE,
    )

    plot_top_words(
        frequency=agnews_test_overall,
        title=(
            "20 Token Paling Sering Muncul "
            "pada Title dan Description AG News Test"
        ),
        output_path=AGNEWS_TEST_WORD_FREQUENCY_FIGURE,
        top_n=TOP_N_FIGURE,
    )

    # =========================================================================
    # OUTPUT VALIDATION
    # =========================================================================

    output_files = [
        (
            WORD_FREQUENCY_OVERALL_PATH,
            "frekuensi kata keseluruhan",
        ),
        (
            WORD_FREQUENCY_BY_CATEGORY_PATH,
            "frekuensi kata per kategori",
        ),
        (
            WORD_FREQUENCY_SUMMARY_PATH,
            "ringkasan frekuensi kata",
        ),
        (
            KOMPAS_WORD_FREQUENCY_FIGURE,
            "grafik frekuensi kata Kompas",
        ),
        (
            AGNEWS_TRAIN_WORD_FREQUENCY_FIGURE,
            "grafik frekuensi kata AG News Train",
        ),
        (
            AGNEWS_TEST_WORD_FREQUENCY_FIGURE,
            "grafik frekuensi kata AG News Test",
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

    print(
        "\n"
        + "=" * 80
    )

    print(
        "OUTPUT WORD FREQUENCY ANALYSIS"
    )

    print(
        "=" * 80
    )

    print(
        "\nFrekuensi kata keseluruhan:"
    )

    print(
        WORD_FREQUENCY_OVERALL_PATH
    )

    print(
        "\nFrekuensi kata per kategori:"
    )

    print(
        WORD_FREQUENCY_BY_CATEGORY_PATH
    )

    print(
        "\nRingkasan frekuensi kata:"
    )

    print(
        WORD_FREQUENCY_SUMMARY_PATH
    )

    print(
        "\nGrafik Kompas:"
    )

    print(
        KOMPAS_WORD_FREQUENCY_FIGURE
    )

    print(
        "\nGrafik AG News Train:"
    )

    print(
        AGNEWS_TRAIN_WORD_FREQUENCY_FIGURE
    )

    print(
        "\nGrafik AG News Test:"
    )

    print(
        AGNEWS_TEST_WORD_FREQUENCY_FIGURE
    )

    print(
        "\nRingkasan token keseluruhan:"
    )

    overall_summary = frequency_summary[
        frequency_summary["category"].eq(
            "all"
        )
    ]

    print(
        overall_summary[
            [
                "dataset",
                "jumlah_dokumen",
                "total_token_setelah_filter",
                "jumlah_token_unik",
                "rata_rata_token_per_dokumen",
                "dokumen_tanpa_token_setelah_filter",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nCatatan:"
    )

    print(
        "- Analisis menggunakan gabungan Title dan Description."
    )

    print(
        "- Content Kompas tidak digunakan karena jauh lebih panjang."
    )

    print(
        "- Stopword removal hanya digunakan untuk EDA frekuensi kata."
    )

    print(
        "- Frekuensi kata bukan nilai feature importance model."
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "Tahap word frequency analysis selesai."
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
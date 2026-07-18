from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

try:
    from wordcloud import WordCloud
except ImportError as error:
    raise ImportError(
        "Library wordcloud belum terpasang.\n"
        "Jalankan: pip install wordcloud"
    ) from error


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


# =============================================================================
# OUTPUT DIRECTORY AND TABLE
# =============================================================================

WORDCLOUD_DIR = (
    FIGURES_DIR
    / "wordclouds"
)

WORDCLOUD_SUMMARY_PATH = (
    TABLES_DIR
    / "wordcloud_summary.csv"
)


# =============================================================================
# EXPECTED FINAL DATASET COUNTS
# =============================================================================

EXPECTED_FINAL_COUNTS = {
    "kompas": 9_997,
    "ag_news_train": 119_817,
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
# WORD CLOUD CONFIGURATION
# =============================================================================

TEXT_COLUMNS = [
    "title",
    "description",
]

MINIMUM_TOKEN_LENGTH = 3
MAX_WORDS = 150
RANDOM_STATE = 42

WORDCLOUD_WIDTH = 1_600
WORDCLOUD_HEIGHT = 900
WORDCLOUD_DPI = 300


# =============================================================================
# INDONESIAN STOPWORDS
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
# ENGLISH STOPWORDS
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
    "i",
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
# REGEX PATTERNS
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

LETTER_TOKEN_PATTERN = re.compile(
    r"[^\W\d_]+",
    flags=re.UNICODE,
)


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def ensure_output_directories() -> None:
    """
    Memastikan folder tabel dan word cloud tersedia.
    """

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    WORDCLOUD_DIR.mkdir(
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
    Memilih file dataset final pertama yang ditemukan.
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
    Membaca file CSV dengan beberapa kemungkinan encoding.
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
    Memastikan dataset memiliki semua kolom yang dibutuhkan.
    """

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            f"Dataset {dataset_name} tidak memiliki kolom wajib.\n"
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
            "Pastikan file yang digunakan merupakan dataset "
            "setelah cleaning."
        )


def validate_output_file(
    file_path: Path,
    description: str,
) -> None:
    """
    Memastikan file output berhasil dibuat dan tidak kosong.
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


def make_safe_filename(
    value: Any,
) -> str:
    """
    Membentuk nama file yang aman dari nama kategori.
    """

    text = unicodedata.normalize(
        "NFKC",
        str(value),
    )

    text = text.casefold().strip()

    text = re.sub(
        r"[^\w-]+",
        "_",
        text,
        flags=re.UNICODE,
    )

    text = re.sub(
        r"_+",
        "_",
        text,
    )

    return text.strip("_") or "unknown"


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

    validate_required_columns(
        dataframe=dataframe,
        dataset_name=dataset_name,
    )

    validate_dataset_count(
        dataframe=dataframe,
        dataset_key=dataset_key,
    )

    dataframe["category"] = (
        dataframe["category"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if dataframe["category"].eq("").any():
        raise ValueError(
            f"Ditemukan kategori kosong pada dataset {dataset_name}."
        )

    return dataframe


# =============================================================================
# TEXT NORMALIZATION AND TOKENIZATION
# =============================================================================

def normalize_text_for_wordcloud(
    value: Any,
) -> str:
    """
    Menormalisasi teks untuk kebutuhan word cloud.

    Proses hanya digunakan untuk visualisasi EDA
    dan tidak mengubah dataset utama.
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
    Mengubah teks menjadi token untuk visualisasi word cloud.

    Token yang dipertahankan:
    - hanya karakter alfabet;
    - panjang minimal tiga karakter;
    - bukan stopword.
    """

    text = normalize_text_for_wordcloud(
        value
    )

    tokens = LETTER_TOKEN_PATTERN.findall(
        text
    )

    return [
        token
        for token in tokens
        if (
            len(token) >= MINIMUM_TOKEN_LENGTH
            and token not in stopwords
        )
    ]


# =============================================================================
# COMBINE TEXT COLUMNS
# =============================================================================

def combine_text_columns(
    dataframe: pd.DataFrame,
    text_columns: list[str],
) -> pd.Series:
    """
    Menggabungkan Title dan Description.

    Content Kompas tidak digunakan karena panjangnya jauh lebih besar
    dan dapat mendominasi visualisasi.
    """

    missing_columns = [
        column
        for column in text_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Kolom teks tidak ditemukan: {missing_columns}"
        )

    combined = pd.Series(
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

        combined = (
            combined
            + " "
            + values
        )

    return combined.str.strip()


# =============================================================================
# WORD FREQUENCY FOR WORD CLOUD
# =============================================================================

def calculate_word_frequencies(
    texts: pd.Series,
    stopwords: set[str],
) -> tuple[Counter[str], dict[str, Any]]:
    """
    Menghitung frekuensi token untuk word cloud.
    """

    counter: Counter[str] = Counter()

    total_documents = int(
        len(texts)
    )

    total_tokens = 0
    documents_without_tokens = 0

    for text in texts:
        tokens = tokenize_text(
            value=text,
            stopwords=stopwords,
        )

        if not tokens:
            documents_without_tokens += 1
            continue

        counter.update(
            tokens
        )

        total_tokens += len(tokens)

    if not counter:
        raise ValueError(
            "Tidak terdapat token yang dapat digunakan "
            "untuk membuat word cloud."
        )

    most_common_word, most_common_frequency = (
        counter.most_common(1)[0]
    )

    summary = {
        "jumlah_dokumen":
            total_documents,

        "total_token_setelah_filter":
            int(total_tokens),

        "jumlah_token_unik":
            int(len(counter)),

        "rata_rata_token_per_dokumen":
            round(
                total_tokens / total_documents,
                4,
            )
            if total_documents > 0
            else 0.0,

        "dokumen_tanpa_token":
            int(documents_without_tokens),

        "token_teratas":
            most_common_word,

        "frekuensi_token_teratas":
            int(most_common_frequency),
    }

    return counter, summary


# =============================================================================
# GENERATE WORD CLOUD
# =============================================================================

def generate_wordcloud(
    frequencies: Counter[str],
    title: str,
    output_path: Path,
) -> None:
    """
    Membuat dan menyimpan word cloud berdasarkan frekuensi token.
    """

    if not frequencies:
        raise ValueError(
            f"Frekuensi token kosong untuk word cloud: {title}"
        )

    wordcloud = WordCloud(
        width=WORDCLOUD_WIDTH,
        height=WORDCLOUD_HEIGHT,
        background_color="white",
        max_words=MAX_WORDS,
        collocations=False,
        random_state=RANDOM_STATE,
        prefer_horizontal=0.9,
        relative_scaling=0.5,
    ).generate_from_frequencies(
        dict(frequencies)
    )

    figure, axis = plt.subplots(
        figsize=(
            16,
            9,
        )
    )

    axis.imshow(
        wordcloud,
        interpolation="bilinear",
    )

    axis.set_title(
        title,
        fontsize=18,
        pad=20,
    )

    axis.axis(
        "off"
    )

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=WORDCLOUD_DPI,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# =============================================================================
# CREATE ONE WORD CLOUD
# =============================================================================

def create_wordcloud_output(
    dataframe: pd.DataFrame,
    dataset_name: str,
    category_name: str,
    text_columns: list[str],
    stopwords: set[str],
    output_path: Path,
    title: str,
) -> dict[str, Any]:
    """
    Membuat satu word cloud dan menghasilkan ringkasan.
    """

    combined_text = combine_text_columns(
        dataframe=dataframe,
        text_columns=text_columns,
    )

    frequencies, summary = (
        calculate_word_frequencies(
            texts=combined_text,
            stopwords=stopwords,
        )
    )

    generate_wordcloud(
        frequencies=frequencies,
        title=title,
        output_path=output_path,
    )

    return {
        "dataset":
            dataset_name,

        "category":
            category_name,

        "text_source":
            " + ".join(text_columns),

        **summary,

        "max_words_wordcloud":
            MAX_WORDS,

        "output_path":
            str(output_path.resolve()),
    }


# =============================================================================
# CREATE WORD CLOUDS BY CATEGORY
# =============================================================================

def generate_wordclouds_by_category(
    dataframe: pd.DataFrame,
    dataset_name: str,
    dataset_display_name: str,
    filename_prefix: str,
    text_columns: list[str],
    stopwords: set[str],
) -> tuple[list[Path], list[dict[str, Any]]]:
    """
    Membuat word cloud untuk setiap kategori.

    dataset_name digunakan pada tabel ringkasan.
    filename_prefix digunakan pada nama file gambar.
    """

    output_paths: list[Path] = []
    summary_records: list[dict[str, Any]] = []

    grouped = dataframe.groupby(
        "category",
        dropna=False,
        sort=True,
    )

    for category, group in grouped:
        category_name = str(category)

        safe_category = make_safe_filename(
            category_name
        )

        output_path = (
            WORDCLOUD_DIR
            / (
                f"{filename_prefix}_"
                f"{safe_category}_wordcloud.png"
            )
        )

        title = (
            f"Word Cloud {dataset_display_name}\n"
            f"Kategori "
            f"{category_name.replace('_', ' ').title()} "
            f"(Title + Description)"
        )

        summary_record = create_wordcloud_output(
            dataframe=group,
            dataset_name=dataset_name,
            category_name=category_name,
            text_columns=text_columns,
            stopwords=stopwords,
            output_path=output_path,
            title=title,
        )

        output_paths.append(
            output_path
        )

        summary_records.append(
            summary_record
        )

    return output_paths, summary_records


# =============================================================================
# DISPLAY SUMMARY
# =============================================================================

def display_wordcloud_summary(
    summary: pd.DataFrame,
) -> None:
    """
    Menampilkan ringkasan word cloud pada terminal.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        "RINGKASAN WORD CLOUD"
    )

    print(
        "=" * 80
    )

    display_columns = [
        "dataset",
        "category",
        "jumlah_dokumen",
        "total_token_setelah_filter",
        "jumlah_token_unik",
        "token_teratas",
        "frekuensi_token_teratas",
        "dokumen_tanpa_token",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main() -> None:
    """
    Menjalankan EDA tahap 3.6:

    - word cloud keseluruhan Kompas;
    - word cloud Kompas per kategori;
    - word cloud keseluruhan AG News Train;
    - word cloud AG News Train per kategori;
    - ringkasan word cloud untuk dashboard.
    """

    print(
        "=" * 80
    )

    print(
        "STEP 3.6 - WORD CLOUD ANALYSIS"
    )

    print(
        "=" * 80
    )

    ensure_output_directories()

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

    print(
        "\nDataset final yang digunakan:"
    )

    print(
        f"Kompas        : {kompas_path}"
    )

    print(
        f"AG News Train : {agnews_train_path}"
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

    print(
        "\nDataset berhasil dimuat:"
    )

    print(
        f"Kompas        : {len(kompas):,}"
    )

    print(
        f"AG News Train : {len(agnews_train):,}"
    )

    # =========================================================================
    # OVERALL WORD CLOUDS
    # =========================================================================

    kompas_overall_output = (
        WORDCLOUD_DIR
        / "kompas_overall_wordcloud.png"
    )

    agnews_train_overall_output = (
        WORDCLOUD_DIR
        / "agnews_train_overall_wordcloud.png"
    )

    summary_records: list[dict[str, Any]] = []

    summary_records.append(
        create_wordcloud_output(
            dataframe=kompas,
            dataset_name="kompas",
            category_name="all",
            text_columns=TEXT_COLUMNS,
            stopwords=INDONESIAN_STOPWORDS,
            output_path=kompas_overall_output,
            title=(
                "Word Cloud Dataset Kompas "
                "(Title + Description)"
            ),
        )
    )

    summary_records.append(
        create_wordcloud_output(
            dataframe=agnews_train,
            dataset_name="ag_news_train",
            category_name="all",
            text_columns=TEXT_COLUMNS,
            stopwords=ENGLISH_STOPWORDS,
            output_path=agnews_train_overall_output,
            title=(
                "Word Cloud AG News Train "
                "(Title + Description)"
            ),
        )
    )

    # =========================================================================
    # WORD CLOUDS BY CATEGORY
    # =========================================================================

    (
        kompas_category_outputs,
        kompas_category_summaries,
    ) = generate_wordclouds_by_category(
        dataframe=kompas,
        dataset_name="kompas",
        dataset_display_name="Dataset Kompas",
        filename_prefix="kompas",
        text_columns=TEXT_COLUMNS,
        stopwords=INDONESIAN_STOPWORDS,
    )

    (
        agnews_category_outputs,
        agnews_category_summaries,
    ) = generate_wordclouds_by_category(
        dataframe=agnews_train,
        dataset_name="ag_news_train",
        dataset_display_name="AG News Train",
        filename_prefix="agnews_train",
        text_columns=TEXT_COLUMNS,
        stopwords=ENGLISH_STOPWORDS,
    )

    summary_records.extend(
        kompas_category_summaries
    )

    summary_records.extend(
        agnews_category_summaries
    )

    # =========================================================================
    # SAVE SUMMARY
    # =========================================================================

    wordcloud_summary = pd.DataFrame(
        summary_records
    )

    # Pengamanan apabila masih terdapat nama lama.
    wordcloud_summary["dataset"] = (
        wordcloud_summary["dataset"]
        .replace(
            {
                "agnews_train":
                    "ag_news_train",
            }
        )
    )

    wordcloud_summary = (
        wordcloud_summary
        .sort_values(
            [
                "dataset",
                "category",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    wordcloud_summary.to_csv(
        WORDCLOUD_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # VALIDATE OUTPUTS
    # =========================================================================

    all_image_outputs = [
        kompas_overall_output,
        agnews_train_overall_output,
        *kompas_category_outputs,
        *agnews_category_outputs,
    ]

    for output_path in all_image_outputs:
        validate_output_file(
            file_path=output_path,
            description=(
                f"word cloud {output_path.name}"
            ),
        )

    validate_output_file(
        file_path=WORDCLOUD_SUMMARY_PATH,
        description="ringkasan word cloud",
    )

    expected_image_count = (
        2
        + int(
            kompas["category"].nunique()
        )
        + int(
            agnews_train["category"].nunique()
        )
    )

    actual_image_count = len(
        all_image_outputs
    )

    if actual_image_count != expected_image_count:
        raise ValueError(
            "Jumlah output word cloud tidak sesuai.\n"
            f"Expected : {expected_image_count}\n"
            f"Actual   : {actual_image_count}"
        )

    expected_summary_rows = expected_image_count

    actual_summary_rows = int(
        len(wordcloud_summary)
    )

    if actual_summary_rows != expected_summary_rows:
        raise ValueError(
            "Jumlah baris ringkasan word cloud tidak sesuai.\n"
            f"Expected : {expected_summary_rows}\n"
            f"Actual   : {actual_summary_rows}"
        )

    valid_dataset_names = {
        "kompas",
        "ag_news_train",
    }

    unexpected_dataset_names = (
        set(
            wordcloud_summary[
                "dataset"
            ].unique()
        )
        - valid_dataset_names
    )

    if unexpected_dataset_names:
        raise ValueError(
            "Ditemukan nama dataset yang tidak konsisten "
            "pada wordcloud_summary.csv:\n"
            f"{sorted(unexpected_dataset_names)}"
        )

    # =========================================================================
    # TERMINAL DISPLAY
    # =========================================================================

    display_wordcloud_summary(
        wordcloud_summary
    )

    print(
        "\nWord cloud keseluruhan Kompas:"
    )

    print(
        kompas_overall_output
    )

    print(
        "\nWord cloud keseluruhan AG News Train:"
    )

    print(
        agnews_train_overall_output
    )

    print(
        "\nWord cloud Kompas per kategori:"
    )

    for output_path in kompas_category_outputs:
        print(
            output_path
        )

    print(
        "\nWord cloud AG News Train per kategori:"
    )

    for output_path in agnews_category_outputs:
        print(
            output_path
        )

    # =========================================================================
    # OUTPUT INFORMATION
    # =========================================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "OUTPUT WORD CLOUD ANALYSIS"
    )

    print(
        "=" * 80
    )

    print(
        "\nFolder seluruh gambar:"
    )

    print(
        WORDCLOUD_DIR
    )

    print(
        "\nRingkasan word cloud:"
    )

    print(
        WORDCLOUD_SUMMARY_PATH
    )

    print(
        f"\nJumlah gambar berhasil dibuat: "
        f"{actual_image_count}"
    )

    print(
        "\nCatatan:"
    )

    print(
        "- Word cloud menggunakan gabungan Title dan Description."
    )

    print(
        "- Content Kompas tidak digunakan karena jauh lebih panjang."
    )

    print(
        "- Ukuran token menggambarkan frekuensi kemunculan."
    )

    print(
        "- Word cloud bukan feature importance atau hasil SHAP."
    )

    print(
        "- AG News Test tidak digunakan karena merupakan data evaluasi."
    )

    print(
        "\nValidasi seluruh output: LULUS"
    )

    print(
        "Tahap word cloud analysis selesai."
    )


if __name__ == "__main__":
    main()
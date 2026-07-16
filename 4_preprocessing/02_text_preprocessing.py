from __future__ import annotations

import html
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd


# ============================================================
# MENAMBAHKAN ROOT PROJECT KE PYTHON PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config import (  # noqa: E402
    PROCESSED_DATA_DIR,
    TABLES_DIR,
)


# ============================================================
# PATH INPUT
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
# PATH OUTPUT
# ============================================================

KOMPAS_PREPROCESSED_PATH = (
    PROCESSED_DATA_DIR
    / "kompas_preprocessed.csv"
)

AG_NEWS_TRAIN_PREPROCESSED_PATH = (
    PROCESSED_DATA_DIR
    / "ag_news_train_preprocessed.csv"
)

AG_NEWS_TEST_PREPROCESSED_PATH = (
    PROCESSED_DATA_DIR
    / "ag_news_test_preprocessed.csv"
)


# ============================================================
# PATH LAPORAN
# ============================================================

TEXT_PREPROCESSING_REPORT_PATH = (
    TABLES_DIR
    / "text_preprocessing_report.csv"
)

TEXT_PREPROCESSING_SAMPLE_PATH = (
    TABLES_DIR
    / "text_preprocessing_samples.csv"
)


# ============================================================
# MEMBACA DATASET
# ============================================================

def load_dataset(
    file_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Membaca dataset clean.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset {dataset_name} tidak ditemukan:\n"
            f"{file_path}"
        )

    dataframe = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    if dataframe.empty:
        raise ValueError(
            f"Dataset {dataset_name} kosong."
        )

    return dataframe


# ============================================================
# REGEX PREPROCESSING
# ============================================================

URL_PATTERN = re.compile(
    r"""
    (?:
        https?://\S+
        |
        www\.\S+
    )
    """,
    flags=(
        re.IGNORECASE
        | re.VERBOSE
    ),
)


EMAIL_PATTERN = re.compile(
    r"""
    \b
    [A-Za-z0-9._%+-]+
    @
    [A-Za-z0-9.-]+
    \.
    [A-Za-z]{2,}
    \b
    """,
    flags=re.VERBOSE,
)


HTML_TAG_PATTERN = re.compile(
    r"<[^>]+>"
)


CONTROL_CHARACTER_PATTERN = re.compile(
    r"[\x00-\x1f\x7f-\x9f]"
)


MULTIPLE_WHITESPACE_PATTERN = re.compile(
    r"\s+"
)


# ============================================================
# NORMALISASI APOSTROPHE
# ============================================================

def normalize_apostrophes(
    text: str,
) -> str:
    """
    Menyamakan variasi apostrophe Unicode menjadi apostrophe
    standar.

    Contoh:
    don’t -> don't
    company’s -> company's
    """

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u02bc": "'",
        "\u0060": "'",
        "\u00b4": "'",
    }

    for old_character, new_character in replacements.items():
        text = text.replace(
            old_character,
            new_character,
        )

    return text


# ============================================================
# FUNGSI UTAMA PREPROCESSING TEKS
# ============================================================

def preprocess_text(
    value,
) -> str:
    """
    Melakukan light text preprocessing.

    Tahapan:
    1. Menangani nilai kosong.
    2. HTML entity decoding.
    3. Unicode normalization.
    4. Normalisasi apostrophe.
    5. Menghapus HTML tag.
    6. Menghapus URL.
    7. Menghapus email.
    8. Case folding.
    9. Menghapus karakter kontrol.
    10. Menormalisasi karakter non-alfanumerik.
    11. Merapikan whitespace.

    Catatan:
    - Angka dipertahankan.
    - Apostrophe di dalam kata dipertahankan.
    - Stopword tidak dihapus.
    - Stemming tidak dilakukan.
    """

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    text = str(value)

    if not text.strip():
        return ""

    # --------------------------------------------------------
    # 1. Decode HTML entity
    # --------------------------------------------------------

    text = html.unescape(
        text
    )

    # --------------------------------------------------------
    # 2. Unicode normalization
    # --------------------------------------------------------

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    # --------------------------------------------------------
    # 3. Normalisasi apostrophe
    # --------------------------------------------------------

    text = normalize_apostrophes(
        text
    )

    # --------------------------------------------------------
    # 4. Menghapus HTML tag
    # --------------------------------------------------------

    text = HTML_TAG_PATTERN.sub(
        " ",
        text,
    )

    # --------------------------------------------------------
    # 5. Menghapus URL
    # --------------------------------------------------------

    text = URL_PATTERN.sub(
        " ",
        text,
    )

    # --------------------------------------------------------
    # 6. Menghapus email
    # --------------------------------------------------------

    text = EMAIL_PATTERN.sub(
        " ",
        text,
    )

    # --------------------------------------------------------
    # 7. Case folding
    # --------------------------------------------------------

    text = text.lower()

    # --------------------------------------------------------
    # 8. Menghapus karakter kontrol
    # --------------------------------------------------------

    text = CONTROL_CHARACTER_PATTERN.sub(
        " ",
        text,
    )

    # --------------------------------------------------------
    # 9. Mempertahankan huruf, angka, dan apostrophe
    #
    # [^\w\s'] berarti:
    # selain huruf/angka/underscore, whitespace, apostrophe
    # akan diganti spasi.
    # --------------------------------------------------------

    text = re.sub(
        r"[^\w\s']",
        " ",
        text,
        flags=re.UNICODE,
    )

    # --------------------------------------------------------
    # 10. Menghapus underscore
    # --------------------------------------------------------

    text = text.replace(
        "_",
        " ",
    )

    # --------------------------------------------------------
    # 11. Membersihkan apostrophe yang berdiri sendiri
    # --------------------------------------------------------

    text = re.sub(
        r"(?<!\w)'|'(?!\w)",
        " ",
        text,
    )

    # --------------------------------------------------------
    # 12. Normalisasi whitespace
    # --------------------------------------------------------

    text = MULTIPLE_WHITESPACE_PATTERN.sub(
        " ",
        text,
    )

    return text.strip()


# ============================================================
# MENGHITUNG JUMLAH KATA
# ============================================================

def count_words(
    value,
) -> int:
    """
    Menghitung jumlah kata berdasarkan whitespace.
    """

    if value is None:
        return 0

    text = str(value).strip()

    if not text:
        return 0

    return len(
        text.split()
    )


# ============================================================
# MENGHITUNG TEKS KOSONG
# ============================================================

def count_empty_text(
    series: pd.Series,
) -> int:
    """
    Menghitung jumlah teks kosong.
    """

    normalized = (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return int(
        normalized.eq("").sum()
    )


# ============================================================
# PREPROCESSING KOLOM TEKS
# ============================================================

def preprocess_text_columns(
    dataframe: pd.DataFrame,
    text_columns: list[str],
) -> pd.DataFrame:
    """
    Membuat kolom baru dengan suffix '_preprocessed'.

    Kolom asli tidak ditimpa.
    """

    dataframe = dataframe.copy()

    for column in text_columns:

        if column not in dataframe.columns:
            raise ValueError(
                f"Kolom '{column}' tidak ditemukan."
            )

        output_column = (
            f"{column}_preprocessed"
        )

        print(
            f"Memproses kolom: "
            f"{column} -> {output_column}"
        )

        dataframe[
            output_column
        ] = (
            dataframe[column]
            .apply(preprocess_text)
        )

    return dataframe


# ============================================================
# MEMBUAT LAPORAN PREPROCESSING
# ============================================================

def create_preprocessing_report(
    dataframe: pd.DataFrame,
    dataset_name: str,
    text_columns: list[str],
) -> list[dict]:
    """
    Membuat laporan perbandingan sebelum dan sesudah
    preprocessing.
    """

    report_rows = []

    for column in text_columns:

        preprocessed_column = (
            f"{column}_preprocessed"
        )

        original_word_count = (
            dataframe[column]
            .apply(count_words)
        )

        preprocessed_word_count = (
            dataframe[
                preprocessed_column
            ]
            .apply(count_words)
        )

        original_empty = (
            count_empty_text(
                dataframe[column]
            )
        )

        preprocessed_empty = (
            count_empty_text(
                dataframe[
                    preprocessed_column
                ]
            )
        )

        report_rows.append(
            {
                "dataset": dataset_name,
                "column": column,
                "jumlah_data": len(
                    dataframe
                ),
                "avg_words_before": round(
                    float(
                        original_word_count.mean()
                    ),
                    2,
                ),
                "avg_words_after": round(
                    float(
                        preprocessed_word_count.mean()
                    ),
                    2,
                ),
                "empty_before":
                    original_empty,
                "empty_after":
                    preprocessed_empty,
                "new_empty_after_preprocessing": (
                    preprocessed_empty
                    - original_empty
                ),
            }
        )

    return report_rows


# ============================================================
# MEMBUAT CONTOH BEFORE-AFTER
# ============================================================

def create_preprocessing_samples(
    dataframe: pd.DataFrame,
    dataset_name: str,
    text_columns: list[str],
    sample_size: int = 5,
) -> pd.DataFrame:
    """
    Mengambil contoh teks sebelum dan sesudah preprocessing.

    Menggunakan random_state agar sampel reproducible.
    """

    actual_sample_size = min(
        sample_size,
        len(dataframe),
    )

    sampled_dataframe = (
        dataframe.sample(
            n=actual_sample_size,
            random_state=42,
        )
        .copy()
    )

    sample_rows = []

    for _, row in sampled_dataframe.iterrows():

        for column in text_columns:

            preprocessed_column = (
                f"{column}_preprocessed"
            )

            sample_rows.append(
                {
                    "dataset":
                        dataset_name,
                    "document_id":
                        row.get(
                            "document_id",
                            "",
                        ),
                    "column":
                        column,
                    "before":
                        row[column],
                    "after":
                        row[
                            preprocessed_column
                        ],
                }
            )

    return pd.DataFrame(
        sample_rows
    )


# ============================================================
# VALIDASI HASIL PREPROCESSING
# ============================================================

def validate_preprocessed_dataset(
    dataframe: pd.DataFrame,
    dataset_name: str,
    text_columns: list[str],
) -> None:
    """
    Memastikan:
    - jumlah baris tidak berubah;
    - kolom hasil preprocessing tersedia;
    - title tidak menjadi kosong.

    Description AG News Test boleh kosong karena sudah
    ditemukan pada dataset asli.
    """

    if dataframe.empty:
        raise ValueError(
            f"Dataset {dataset_name} kosong "
            f"setelah preprocessing."
        )

    for column in text_columns:

        preprocessed_column = (
            f"{column}_preprocessed"
        )

        if (
            preprocessed_column
            not in dataframe.columns
        ):
            raise ValueError(
                f"Kolom {preprocessed_column} "
                f"tidak ditemukan."
            )

    if "title_preprocessed" in dataframe.columns:

        empty_title = count_empty_text(
            dataframe[
                "title_preprocessed"
            ]
        )

        if empty_title > 0:
            raise ValueError(
                f"Dataset {dataset_name} memiliki "
                f"{empty_title} title kosong setelah "
                f"preprocessing."
            )


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main() -> None:

    print("=" * 72)
    print("STEP 4.2 - TEXT PREPROCESSING")
    print("=" * 72)

    # ========================================================
    # MEMUAT DATASET CLEAN
    # ========================================================

    kompas = load_dataset(
        KOMPAS_CLEAN_PATH,
        "Kompas",
    )

    agnews_train = load_dataset(
        AG_NEWS_TRAIN_CLEAN_PATH,
        "AG News Train",
    )

    agnews_test = load_dataset(
        AG_NEWS_TEST_CLEAN_PATH,
        "AG News Test",
    )

    # ========================================================
    # MENYIMPAN JUMLAH BARIS SEBELUM PREPROCESSING
    # ========================================================

    original_row_counts = {
        "Kompas": len(
            kompas
        ),
        "AG News Train": len(
            agnews_train
        ),
        "AG News Test": len(
            agnews_test
        ),
    }

    # ========================================================
    # PREPROCESSING KOMPAS
    # ========================================================

    print("\n" + "=" * 72)
    print("PREPROCESSING KOMPAS")
    print("=" * 72)

    kompas_text_columns = [
        "title",
        "description",
        "content",
    ]

    kompas = preprocess_text_columns(
        dataframe=kompas,
        text_columns=kompas_text_columns,
    )

    # ========================================================
    # PREPROCESSING AG NEWS TRAIN
    # ========================================================

    print("\n" + "=" * 72)
    print("PREPROCESSING AG NEWS TRAIN")
    print("=" * 72)

    agnews_text_columns = [
        "title",
        "description",
    ]

    agnews_train = preprocess_text_columns(
        dataframe=agnews_train,
        text_columns=agnews_text_columns,
    )

    # ========================================================
    # PREPROCESSING AG NEWS TEST
    # ========================================================

    print("\n" + "=" * 72)
    print("PREPROCESSING AG NEWS TEST")
    print("=" * 72)

    agnews_test = preprocess_text_columns(
        dataframe=agnews_test,
        text_columns=agnews_text_columns,
    )

    # ========================================================
    # VALIDASI JUMLAH BARIS
    # ========================================================

    if len(kompas) != original_row_counts["Kompas"]:
        raise ValueError(
            "Jumlah baris Kompas berubah "
            "saat preprocessing."
        )

    if (
        len(agnews_train)
        != original_row_counts["AG News Train"]
    ):
        raise ValueError(
            "Jumlah baris AG News Train berubah "
            "saat preprocessing."
        )

    if (
        len(agnews_test)
        != original_row_counts["AG News Test"]
    ):
        raise ValueError(
            "Jumlah baris AG News Test berubah "
            "saat preprocessing."
        )

    # ========================================================
    # VALIDASI DATASET
    # ========================================================

    validate_preprocessed_dataset(
        dataframe=kompas,
        dataset_name="Kompas",
        text_columns=kompas_text_columns,
    )

    validate_preprocessed_dataset(
        dataframe=agnews_train,
        dataset_name="AG News Train",
        text_columns=agnews_text_columns,
    )

    validate_preprocessed_dataset(
        dataframe=agnews_test,
        dataset_name="AG News Test",
        text_columns=agnews_text_columns,
    )

    # ========================================================
    # MEMBUAT LAPORAN
    # ========================================================

    report_rows = []

    report_rows.extend(
        create_preprocessing_report(
            dataframe=kompas,
            dataset_name="Kompas",
            text_columns=kompas_text_columns,
        )
    )

    report_rows.extend(
        create_preprocessing_report(
            dataframe=agnews_train,
            dataset_name="AG News Train",
            text_columns=agnews_text_columns,
        )
    )

    report_rows.extend(
        create_preprocessing_report(
            dataframe=agnews_test,
            dataset_name="AG News Test",
            text_columns=agnews_text_columns,
        )
    )

    preprocessing_report = pd.DataFrame(
        report_rows
    )

    # ========================================================
    # MEMBUAT SAMPEL BEFORE-AFTER
    # ========================================================

    sample_dataframes = [
        create_preprocessing_samples(
            dataframe=kompas,
            dataset_name="Kompas",
            text_columns=kompas_text_columns,
        ),
        create_preprocessing_samples(
            dataframe=agnews_train,
            dataset_name="AG News Train",
            text_columns=agnews_text_columns,
        ),
        create_preprocessing_samples(
            dataframe=agnews_test,
            dataset_name="AG News Test",
            text_columns=agnews_text_columns,
        ),
    ]

    preprocessing_samples = pd.concat(
        sample_dataframes,
        ignore_index=True,
    )

    # ========================================================
    # MEMBUAT FOLDER OUTPUT
    # ========================================================

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # MENYIMPAN DATASET
    # ========================================================

    kompas.to_csv(
        KOMPAS_PREPROCESSED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    agnews_train.to_csv(
        AG_NEWS_TRAIN_PREPROCESSED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    agnews_test.to_csv(
        AG_NEWS_TEST_PREPROCESSED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # MENYIMPAN LAPORAN
    # ========================================================

    preprocessing_report.to_csv(
        TEXT_PREPROCESSING_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    preprocessing_samples.to_csv(
        TEXT_PREPROCESSING_SAMPLE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # MENAMPILKAN HASIL
    # ========================================================

    print("\n" + "=" * 72)
    print("HASIL TEXT PREPROCESSING")
    print("=" * 72)

    print(
        preprocessing_report.to_string(
            index=False
        )
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print("\n" + "=" * 72)
    print("OUTPUT TEXT PREPROCESSING")
    print("=" * 72)

    print("\nDataset Kompas preprocessed:")
    print(
        KOMPAS_PREPROCESSED_PATH
    )

    print("\nDataset AG News Train preprocessed:")
    print(
        AG_NEWS_TRAIN_PREPROCESSED_PATH
    )

    print("\nDataset AG News Test preprocessed:")
    print(
        AG_NEWS_TEST_PREPROCESSED_PATH
    )

    print("\nLaporan preprocessing:")
    print(
        TEXT_PREPROCESSING_REPORT_PATH
    )

    print("\nContoh before-after:")
    print(
        TEXT_PREPROCESSING_SAMPLE_PATH
    )

    print(
        "\nTahap text preprocessing selesai."
    )


if __name__ == "__main__":
    main()
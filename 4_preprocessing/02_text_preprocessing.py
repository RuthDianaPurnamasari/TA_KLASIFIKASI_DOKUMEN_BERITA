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
# JUMLAH DATA YANG DIHARAPKAN
# ============================================================

EXPECTED_ROW_COUNTS = {
    "Kompas": 9_997,
    "AG News Train": 119_817,
    "AG News Test": 7_600,
}


# ============================================================
# KOLOM WAJIB
# ============================================================

KOMPAS_REQUIRED_COLUMNS = [
    "title",
    "description",
    "content",
    "category",
]

AG_NEWS_REQUIRED_COLUMNS = [
    "document_id",
    "source_row",
    "class_index",
    "category",
    "title",
    "description",
    "split",
]


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


# Entitas HTML numerik AG News yang kehilangan karakter "&".
#
# Contoh:
# #39;   seharusnya &#39;
# #36;   seharusnya &#36;
# #151;  seharusnya &#151;
# #x2014; seharusnya &#x2014;
MALFORMED_HTML_ENTITY_PATTERN = re.compile(
    r"(?<!&)#(?P<entity>x[0-9a-fA-F]+|\d+);",
    flags=re.IGNORECASE,
)


# ============================================================
# MEMBACA DATASET
# ============================================================

def load_dataset(
    file_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Membaca dataset hasil data cleaning.
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
        keep_default_na=False,
    )

    if dataframe.empty:
        raise ValueError(
            f"Dataset {dataset_name} kosong."
        )

    return dataframe


# ============================================================
# VALIDASI KOLOM WAJIB
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
            f"Dataset {dataset_name} tidak memiliki "
            f"kolom wajib: {missing_columns}\n"
            f"Kolom tersedia: {list(dataframe.columns)}"
        )


# ============================================================
# VALIDASI JUMLAH DATA
# ============================================================

def validate_expected_row_count(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Memastikan script membaca hasil data cleaning terbaru.
    """

    if dataset_name not in EXPECTED_ROW_COUNTS:
        raise KeyError(
            f"Jumlah data yang diharapkan untuk "
            f"{dataset_name} belum dikonfigurasi."
        )

    expected_count = EXPECTED_ROW_COUNTS[
        dataset_name
    ]

    actual_count = len(dataframe)

    if actual_count != expected_count:
        raise ValueError(
            f"Jumlah data {dataset_name} tidak sesuai.\n"
            f"Seharusnya: {expected_count:,}\n"
            f"Ditemukan : {actual_count:,}\n"
            "Pastikan 4_preprocessing/01_data_cleaning.py "
            "telah dijalankan menggunakan dataset terbaru."
        )


# ============================================================
# MEMPERBAIKI ENTITAS HTML AG NEWS
# ============================================================

MALFORMED_HTML_ENTITY_PATTERN = re.compile(
    r"(?<!&)#(?:x[0-9a-fA-F]+|\d+);",
    flags=re.IGNORECASE,
)


def repair_malformed_html_entities(
    text: str,
) -> str:
    """
    Memperbaiki entitas HTML numerik AG News yang
    kehilangan karakter ampersand.
    """

    return MALFORMED_HTML_ENTITY_PATTERN.sub(
        lambda match: f"&{match.group(0)}",
        text,
    )


# ============================================================
# NORMALISASI APOSTROPHE
# ============================================================

def normalize_apostrophes(
    text: str,
) -> str:
    """
    Menyeragamkan variasi apostrophe Unicode menjadi
    apostrophe standar.

    Contoh:
    don't    -> don't
    company's -> company's
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
    value: object,
) -> str:
    """
    Melakukan light text preprocessing untuk CNN dan
    Attention-BiLSTM.

    Tahapan:
    1. Menangani nilai kosong.
    2. Memperbaiki entitas HTML AG News yang tidak lengkap.
    3. Melakukan HTML entity decoding.
    4. Melakukan normalisasi Unicode.
    5. Menyeragamkan apostrophe.
    6. Menghapus HTML tag.
    7. Menghapus URL.
    8. Menghapus alamat email.
    9. Melakukan case folding.
    10. Menghapus karakter kontrol.
    11. Mengganti karakter non-alfanumerik dengan spasi.
    12. Menghapus underscore.
    13. Menghapus apostrophe yang berdiri sendiri.
    14. Menormalisasi whitespace.

    Ketentuan:
    - Angka dipertahankan.
    - Apostrophe di dalam kata dipertahankan.
    - Stopword tidak dihapus.
    - Stemming tidak dilakukan.
    - Lemmatization tidak dilakukan.
    """

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    text = str(value)

    if not text.strip():
        return ""

    # --------------------------------------------------------
    # 1. Memperbaiki entitas HTML numerik AG News
    # --------------------------------------------------------

    text = repair_malformed_html_entities(
        text
    )

    # --------------------------------------------------------
    # 2. Decode HTML entity
    # --------------------------------------------------------

    text = html.unescape(
        text
    )

    # --------------------------------------------------------
    # 3. Unicode normalization
    # --------------------------------------------------------

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    # --------------------------------------------------------
    # 4. Normalisasi apostrophe
    # --------------------------------------------------------

    text = normalize_apostrophes(
        text
    )

    # --------------------------------------------------------
    # 5. Menghapus HTML tag
    # --------------------------------------------------------

    text = HTML_TAG_PATTERN.sub(
        " ",
        text,
    )

    # --------------------------------------------------------
    # 6. Menghapus URL
    # --------------------------------------------------------

    text = URL_PATTERN.sub(
        " ",
        text,
    )

    # --------------------------------------------------------
    # 7. Menghapus email
    # --------------------------------------------------------

    text = EMAIL_PATTERN.sub(
        " ",
        text,
    )

    # --------------------------------------------------------
    # 8. Case folding
    # --------------------------------------------------------

    text = text.casefold()

    # --------------------------------------------------------
    # 9. Menghapus karakter kontrol
    # --------------------------------------------------------

    text = CONTROL_CHARACTER_PATTERN.sub(
        " ",
        text,
    )

    # --------------------------------------------------------
    # 10. Mempertahankan:
    # - huruf Unicode;
    # - angka;
    # - whitespace;
    # - apostrophe.
    #
    # Karakter lainnya diganti menjadi spasi.
    # --------------------------------------------------------

    text = re.sub(
        r"[^\w\s']",
        " ",
        text,
        flags=re.UNICODE,
    )

    # --------------------------------------------------------
    # 11. Menghapus underscore
    # --------------------------------------------------------

    text = text.replace(
        "_",
        " ",
    )

    # --------------------------------------------------------
    # 12. Menghapus apostrophe yang berdiri sendiri
    #
    # Apostrophe pada don't dan company's tetap dipertahankan.
    # --------------------------------------------------------

    text = re.sub(
        r"(?<!\w)'|'(?!\w)",
        " ",
        text,
    )

    # --------------------------------------------------------
    # 13. Normalisasi whitespace
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
    value: object,
) -> int:
    """
    Menghitung jumlah kata berdasarkan whitespace.
    """

    if value is None:
        return 0

    if pd.isna(value):
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
# MENGHITUNG TEKS YANG BERUBAH
# ============================================================

def count_changed_text(
    original_series: pd.Series,
    preprocessed_series: pd.Series,
) -> int:
    """
    Menghitung jumlah teks yang berubah setelah preprocessing.
    """

    original = (
        original_series
        .fillna("")
        .astype(str)
    )

    preprocessed = (
        preprocessed_series
        .fillna("")
        .astype(str)
    )

    return int(
        original.ne(preprocessed).sum()
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

    Kolom teks asli tidak ditimpa agar proses dapat diaudit.
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
# VALIDASI HASIL PREPROCESSING
# ============================================================

def validate_preprocessed_dataset(
    dataframe: pd.DataFrame,
    dataset_name: str,
    text_columns: list[str],
    expected_row_count: int,
) -> None:
    """
    Memastikan:
    1. dataset tidak kosong;
    2. jumlah baris tidak berubah;
    3. seluruh kolom hasil preprocessing tersedia;
    4. preprocessing tidak menghasilkan teks kosong baru;
    5. entitas HTML rusak tidak tersisa.
    """

    if dataframe.empty:
        raise ValueError(
            f"Dataset {dataset_name} kosong "
            f"setelah preprocessing."
        )

    if len(dataframe) != expected_row_count:
        raise ValueError(
            f"Jumlah baris {dataset_name} berubah "
            f"saat preprocessing.\n"
            f"Sebelum: {expected_row_count:,}\n"
            f"Sesudah: {len(dataframe):,}"
        )

    for column in text_columns:

        preprocessed_column = (
            f"{column}_preprocessed"
        )

        if preprocessed_column not in dataframe.columns:
            raise ValueError(
                f"Kolom {preprocessed_column} "
                f"tidak ditemukan pada {dataset_name}."
            )

        empty_before = count_empty_text(
            dataframe[column]
        )

        empty_after = count_empty_text(
            dataframe[
                preprocessed_column
            ]
        )

        new_empty = (
            empty_after
            - empty_before
        )

        if new_empty > 0:
            raise ValueError(
                f"Preprocessing menghasilkan "
                f"{new_empty:,} teks kosong baru pada "
                f"{dataset_name}, kolom {column}.\n"
                f"Kosong sebelum: {empty_before:,}\n"
                f"Kosong sesudah: {empty_after:,}"
            )

        malformed_entity_count = int(
            dataframe[
                preprocessed_column
            ]
            .fillna("")
            .astype(str)
            .str.contains(
                MALFORMED_HTML_ENTITY_PATTERN,
                regex=True,
            )
            .sum()
        )

        if malformed_entity_count > 0:
            raise ValueError(
                f"Masih ditemukan "
                f"{malformed_entity_count:,} teks dengan "
                f"entitas HTML tidak valid pada "
                f"{dataset_name}, kolom "
                f"{preprocessed_column}."
            )


# ============================================================
# VALIDASI IDENTITAS BARIS
# ============================================================

def validate_row_identity_preserved(
    dataframe_before: pd.DataFrame,
    dataframe_after: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Memastikan preprocessing tidak mengubah urutan atau
    identitas dokumen.
    """

    if len(dataframe_before) != len(dataframe_after):
        raise ValueError(
            f"Jumlah baris {dataset_name} berubah."
        )

    identity_columns = [
        column
        for column in [
            "document_id",
            "source_row",
        ]
        if column in dataframe_before.columns
        and column in dataframe_after.columns
    ]

    for column in identity_columns:

        before_values = (
            dataframe_before[column]
            .astype(str)
            .reset_index(drop=True)
        )

        after_values = (
            dataframe_after[column]
            .astype(str)
            .reset_index(drop=True)
        )

        if not before_values.equals(
            after_values
        ):
            raise ValueError(
                f"Urutan atau nilai {column} pada "
                f"{dataset_name} berubah selama "
                f"preprocessing."
            )


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

    report_rows: list[dict] = []

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

        original_empty = count_empty_text(
            dataframe[column]
        )

        preprocessed_empty = count_empty_text(
            dataframe[
                preprocessed_column
            ]
        )

        changed_text = count_changed_text(
            dataframe[column],
            dataframe[
                preprocessed_column
            ],
        )

        changed_percentage = (
            changed_text
            / len(dataframe)
            * 100
            if len(dataframe) > 0
            else 0.0
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
                "min_words_before": int(
                    original_word_count.min()
                ),
                "min_words_after": int(
                    preprocessed_word_count.min()
                ),
                "max_words_before": int(
                    original_word_count.max()
                ),
                "max_words_after": int(
                    preprocessed_word_count.max()
                ),
                "empty_before": (
                    original_empty
                ),
                "empty_after": (
                    preprocessed_empty
                ),
                "new_empty_after_preprocessing": (
                    preprocessed_empty
                    - original_empty
                ),
                "jumlah_teks_berubah": (
                    changed_text
                ),
                "persentase_teks_berubah": round(
                    changed_percentage,
                    6,
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

    Random state digunakan agar sampel dapat direproduksi.
    """

    actual_sample_size = min(
        sample_size,
        len(dataframe),
    )

    sampled_dataframe = (
        dataframe
        .sample(
            n=actual_sample_size,
            random_state=42,
        )
        .copy()
    )

    sample_rows: list[dict] = []

    for _, row in sampled_dataframe.iterrows():

        for column in text_columns:

            preprocessed_column = (
                f"{column}_preprocessed"
            )

            sample_rows.append(
                {
                    "dataset": dataset_name,
                    "document_id": row.get(
                        "document_id",
                        "",
                    ),
                    "source_row": row.get(
                        "source_row",
                        "",
                    ),
                    "column": column,
                    "before": row[column],
                    "after": row[
                        preprocessed_column
                    ],
                    "words_before": count_words(
                        row[column]
                    ),
                    "words_after": count_words(
                        row[
                            preprocessed_column
                        ]
                    ),
                }
            )

    return pd.DataFrame(
        sample_rows
    )


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main() -> None:
    """
    Menjalankan light text preprocessing terhadap
    dataset Kompas dan AG News.
    """

    print("=" * 72)
    print("STEP 4.2 - TEXT PREPROCESSING")
    print("=" * 72)

    # --------------------------------------------------------
    # MEMBUAT FOLDER OUTPUT
    # --------------------------------------------------------

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # MEMUAT DATASET CLEAN
    # --------------------------------------------------------

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
        AG_NEWS_REQUIRED_COLUMNS,
        "AG News Train",
    )

    validate_required_columns(
        agnews_test,
        AG_NEWS_REQUIRED_COLUMNS,
        "AG News Test",
    )

    # --------------------------------------------------------
    # VALIDASI JUMLAH INPUT
    # --------------------------------------------------------

    validate_expected_row_count(
        kompas,
        "Kompas",
    )

    validate_expected_row_count(
        agnews_train,
        "AG News Train",
    )

    validate_expected_row_count(
        agnews_test,
        "AG News Test",
    )

    # Menyimpan salinan sebelum preprocessing untuk
    # memvalidasi identitas dan urutan dokumen.
    kompas_before = kompas.copy()
    agnews_train_before = agnews_train.copy()
    agnews_test_before = agnews_test.copy()

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

    # --------------------------------------------------------
    # PREPROCESSING KOMPAS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PREPROCESSING AG NEWS TRAIN
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PREPROCESSING AG NEWS TEST
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("PREPROCESSING AG NEWS TEST")
    print("=" * 72)

    agnews_test = preprocess_text_columns(
        dataframe=agnews_test,
        text_columns=agnews_text_columns,
    )

    # --------------------------------------------------------
    # VALIDASI IDENTITAS BARIS
    # --------------------------------------------------------

    validate_row_identity_preserved(
        dataframe_before=kompas_before,
        dataframe_after=kompas,
        dataset_name="Kompas",
    )

    validate_row_identity_preserved(
        dataframe_before=agnews_train_before,
        dataframe_after=agnews_train,
        dataset_name="AG News Train",
    )

    validate_row_identity_preserved(
        dataframe_before=agnews_test_before,
        dataframe_after=agnews_test,
        dataset_name="AG News Test",
    )

    # --------------------------------------------------------
    # VALIDASI HASIL PREPROCESSING
    # --------------------------------------------------------

    validate_preprocessed_dataset(
        dataframe=kompas,
        dataset_name="Kompas",
        text_columns=kompas_text_columns,
        expected_row_count=(
            original_row_counts[
                "Kompas"
            ]
        ),
    )

    validate_preprocessed_dataset(
        dataframe=agnews_train,
        dataset_name="AG News Train",
        text_columns=agnews_text_columns,
        expected_row_count=(
            original_row_counts[
                "AG News Train"
            ]
        ),
    )

    validate_preprocessed_dataset(
        dataframe=agnews_test,
        dataset_name="AG News Test",
        text_columns=agnews_text_columns,
        expected_row_count=(
            original_row_counts[
                "AG News Test"
            ]
        ),
    )

    # --------------------------------------------------------
    # MEMBUAT LAPORAN PREPROCESSING
    # --------------------------------------------------------

    report_rows: list[dict] = []

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

    # --------------------------------------------------------
    # MEMBUAT SAMPEL BEFORE-AFTER
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MENYIMPAN DATASET
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MENYIMPAN LAPORAN
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MENAMPILKAN HASIL
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("HASIL TEXT PREPROCESSING")
    print("=" * 72)

    print(
        preprocessing_report.to_string(
            index=False
        )
    )

    print("\nValidasi jumlah data:")

    print(
        f"Kompas        : "
        f"{len(kompas):,}"
    )

    print(
        f"AG News Train : "
        f"{len(agnews_train):,}"
    )

    print(
        f"AG News Test  : "
        f"{len(agnews_test):,}"
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

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
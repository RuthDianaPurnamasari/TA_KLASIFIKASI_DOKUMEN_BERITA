from __future__ import annotations

import html
import json
import math
import re
import sys
import time
import unicodedata
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yake
except ImportError as error:
    raise ImportError(
        "Library YAKE belum terpasang. Jalankan:\n"
        "pip install yake"
    ) from error


# ============================================================
# ROOT PROJECT
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

KOMPAS_PREPROCESSED_PATH = (
    PROCESSED_DATA_DIR
    / "kompas_preprocessed.csv"
)


# ============================================================
# PATH OUTPUT
# ============================================================

KOMPAS_WITH_KEYWORDS_PATH = (
    PROCESSED_DATA_DIR
    / "kompas_with_keywords.csv"
)

YAKE_REPORT_PATH = (
    TABLES_DIR
    / "yake_keyword_report.csv"
)

YAKE_SAMPLES_PATH = (
    TABLES_DIR
    / "yake_keyword_samples.csv"
)

YAKE_CONFIG_PATH = (
    TABLES_DIR
    / "yake_configuration.json"
)


# ============================================================
# KONFIGURASI YAKE
# ============================================================

YAKE_LANGUAGE = "id"

YAKE_MAX_NGRAM_SIZE = 3

YAKE_DEDUPLICATION_LIMIT = 0.9

YAKE_DEDUPLICATION_FUNCTION = "seqm"

YAKE_WINDOW_SIZE = 1

YAKE_TOP_KEYWORDS = 5

KEYWORD_SEPARATOR = " | "

PROGRESS_INTERVAL = 500

RANDOM_SEED = 42


# ============================================================
# KONFIGURASI DATASET
# ============================================================

EXPECTED_ROW_COUNT = 9_997

EXPECTED_CATEGORY_COUNTS = {
    "bola": 2_500,
    "global": 2_500,
    "money": 2_500,
    "tekno": 2_497,
}

REQUIRED_COLUMNS = [
    "document_id",
    "title",
    "description",
    "title_preprocessed",
    "description_preprocessed",
    "category",
]


# ============================================================
# REGEX PEMBERSIHAN TEKNIS
# ============================================================

URL_PATTERN = re.compile(
    r"""
    (?:
        https?://\S+
        |
        www\.\S+
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
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

MALFORMED_HTML_ENTITY_PATTERN = re.compile(
    r"(?<!&)#(?:x[0-9a-fA-F]+|\d+);",
    flags=re.IGNORECASE,
)


# ============================================================
# INFORMASI VERSI YAKE
# ============================================================

def get_yake_version() -> str:
    """
    Mendapatkan versi library YAKE untuk reproducibility.
    """

    try:
        return version("yake")
    except PackageNotFoundError:
        return "unknown"


# ============================================================
# MEMBACA DATASET
# ============================================================

def load_dataset(
    file_path: Path,
) -> pd.DataFrame:
    """
    Membaca dataset Kompas hasil text preprocessing.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            "Dataset Kompas preprocessed tidak ditemukan:\n"
            f"{file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            "Path dataset Kompas bukan file:\n"
            f"{file_path}"
        )

    dataframe = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    if dataframe.empty:
        raise ValueError(
            "Dataset Kompas preprocessed kosong."
        )

    return dataframe


# ============================================================
# VALIDASI INPUT
# ============================================================

def validate_input_dataset(
    dataframe: pd.DataFrame,
) -> None:
    """
    Memastikan dataset input sesuai hasil tahap sebelumnya.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Kolom yang dibutuhkan tidak tersedia:\n"
            f"{missing_columns}\n"
            f"Kolom tersedia:\n{list(dataframe.columns)}"
        )

    if len(dataframe) != EXPECTED_ROW_COUNT:
        raise ValueError(
            "Jumlah data Kompas tidak sesuai.\n"
            f"Seharusnya: {EXPECTED_ROW_COUNT:,}\n"
            f"Ditemukan : {len(dataframe):,}\n"
            "Pastikan tahap data cleaning dan text "
            "preprocessing sudah dijalankan."
        )

    if dataframe["document_id"].duplicated().any():
        duplicate_count = int(
            dataframe["document_id"]
            .duplicated()
            .sum()
        )

        raise ValueError(
            "Ditemukan document_id duplikat sebanyak "
            f"{duplicate_count:,}."
        )

    actual_category_counts = (
        dataframe["category"]
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts()
        .sort_index()
        .to_dict()
    )

    expected_category_counts = dict(
        sorted(EXPECTED_CATEGORY_COUNTS.items())
    )

    if actual_category_counts != expected_category_counts:
        raise ValueError(
            "Distribusi kategori Kompas tidak sesuai.\n"
            f"Seharusnya: {expected_category_counts}\n"
            f"Ditemukan : {actual_category_counts}"
        )

    required_text_columns = [
        "title",
        "description",
        "title_preprocessed",
        "description_preprocessed",
    ]

    for column in required_text_columns:
        empty_count = int(
            dataframe[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        if empty_count > 0:
            raise ValueError(
                f"Kolom {column} memiliki "
                f"{empty_count:,} teks kosong."
            )


# ============================================================
# NORMALISASI KARAKTER
# ============================================================

def repair_malformed_html_entities(
    text: str,
) -> str:
    """
    Memperbaiki entitas numerik yang kehilangan ampersand.

    Contoh:
    #39; -> &#39;
    """

    return MALFORMED_HTML_ENTITY_PATTERN.sub(
        lambda match: f"&{match.group(0)}",
        text,
    )


def normalize_apostrophes(
    text: str,
) -> str:
    """
    Menyeragamkan apostrophe Unicode.
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
# MENYIAPKAN TEKS SUMBER YAKE
# ============================================================

def prepare_yake_source_component(
    value: Any,
) -> str:
    """
    Melakukan pembersihan teknis minimal untuk input YAKE.

    Kapitalisasi dan tanda baca tetap dipertahankan karena
    digunakan oleh karakteristik lokal YAKE.
    """

    if value is None or pd.isna(value):
        return ""

    text = str(value).strip()

    if not text:
        return ""

    text = repair_malformed_html_entities(
        text
    )

    text = html.unescape(
        text
    )

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = normalize_apostrophes(
        text
    )

    text = HTML_TAG_PATTERN.sub(
        " ",
        text,
    )

    text = URL_PATTERN.sub(
        " ",
        text,
    )

    text = EMAIL_PATTERN.sub(
        " ",
        text,
    )

    text = CONTROL_CHARACTER_PATTERN.sub(
        " ",
        text,
    )

    text = MULTIPLE_WHITESPACE_PATTERN.sub(
        " ",
        text,
    )

    return text.strip()


def build_yake_source_text(
    title: Any,
    description: Any,
) -> str:
    """
    Membentuk sumber YAKE dari title dan description asli.

    Content tidak digunakan agar keyword tidak membawa
    informasi dari skenario title + description + content.
    """

    title_text = prepare_yake_source_component(
        title
    )

    description_text = prepare_yake_source_component(
        description
    )

    if title_text and description_text:
        if title_text.endswith((".", "!", "?")):
            return (
                f"{title_text} "
                f"{description_text}"
            ).strip()

        return (
            f"{title_text}. "
            f"{description_text}"
        ).strip()

    if title_text:
        return title_text

    return description_text


# ============================================================
# NORMALISASI KEYWORD UNTUK INPUT MODEL
# ============================================================

def normalize_keyword_for_model(
    keyword: Any,
) -> str:
    """
    Menormalisasi keyword YAKE agar konsisten dengan hasil
    light text preprocessing untuk CNN dan Attention-BiLSTM.
    """

    if keyword is None or pd.isna(keyword):
        return ""

    text = str(keyword).strip()

    if not text:
        return ""

    text = repair_malformed_html_entities(
        text
    )

    text = html.unescape(
        text
    )

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = normalize_apostrophes(
        text
    )

    text = text.casefold()

    text = CONTROL_CHARACTER_PATTERN.sub(
        " ",
        text,
    )

    text = re.sub(
        r"[^\w\s']",
        " ",
        text,
        flags=re.UNICODE,
    )

    text = text.replace(
        "_",
        " ",
    )

    text = re.sub(
        r"(?<!\w)'|'(?!\w)",
        " ",
        text,
    )

    text = MULTIPLE_WHITESPACE_PATTERN.sub(
        " ",
        text,
    )

    return text.strip()


def normalize_keyword_for_display(
    keyword: Any,
) -> str:
    """
    Merapikan keyword mentah untuk kebutuhan audit.
    """

    if keyword is None or pd.isna(keyword):
        return ""

    return MULTIPLE_WHITESPACE_PATTERN.sub(
        " ",
        str(keyword).strip(),
    )


# ============================================================
# MEMBUAT YAKE EXTRACTOR
# ============================================================

def create_yake_extractor() -> yake.KeywordExtractor:
    """
    Membuat instance YAKE KeywordExtractor.
    """

    try:
        extractor = yake.KeywordExtractor(
            lan=YAKE_LANGUAGE,
            n=YAKE_MAX_NGRAM_SIZE,
            dedupLim=YAKE_DEDUPLICATION_LIMIT,
            dedupFunc=YAKE_DEDUPLICATION_FUNCTION,
            windowsSize=YAKE_WINDOW_SIZE,
            top=YAKE_TOP_KEYWORDS,
            features=None,
        )
    except Exception as error:
        raise RuntimeError(
            "Gagal membuat YAKE KeywordExtractor.\n"
            f"Bahasa: {YAKE_LANGUAGE}\n"
            f"Error  : {error}"
        ) from error

    return extractor


# ============================================================
# EKSTRAKSI KEYWORD SATU DOKUMEN
# ============================================================

def extract_keywords(
    text: str,
    extractor: yake.KeywordExtractor,
) -> list[dict]:
    """
    Mengekstraksi keyword dan skor YAKE.

    Skor yang lebih kecil menunjukkan keyword lebih penting.
    """

    if not text.strip():
        return []

    raw_results = extractor.extract_keywords(
        text
    )

    raw_results = sorted(
        raw_results,
        key=lambda item: float(item[1]),
    )

    keyword_records: list[dict] = []

    seen_keywords: set[str] = set()

    for raw_keyword, raw_score in raw_results:
        display_keyword = normalize_keyword_for_display(
            raw_keyword
        )

        model_keyword = normalize_keyword_for_model(
            raw_keyword
        )

        score = float(raw_score)

        if not display_keyword or not model_keyword:
            continue

        if not math.isfinite(score):
            continue

        duplicate_key = model_keyword.casefold()

        if duplicate_key in seen_keywords:
            continue

        seen_keywords.add(
            duplicate_key
        )

        keyword_records.append(
            {
                "keyword_raw": display_keyword,
                "keyword_model": model_keyword,
                "score": round(score, 10),
            }
        )

        if len(keyword_records) >= YAKE_TOP_KEYWORDS:
            break

    return keyword_records


# ============================================================
# MENERAPKAN YAKE KE DATASET
# ============================================================

def apply_yake_to_dataset(
    dataframe: pd.DataFrame,
    extractor: yake.KeywordExtractor,
) -> pd.DataFrame:
    """
    Menerapkan YAKE pada seluruh artikel Kompas.
    """

    result = dataframe.copy()

    source_texts: list[str] = []
    keyword_model_texts: list[str] = []
    keyword_display_texts: list[str] = []
    keyword_raw_display_texts: list[str] = []
    keyword_scores_json: list[str] = []
    keyword_pairs_json: list[str] = []
    keyword_counts: list[int] = []

    total_rows = len(result)

    start_time = time.perf_counter()

    for row_number, row in enumerate(
        result.itertuples(index=False),
        start=1,
    ):
        document_id = getattr(
            row,
            "document_id",
        )

        source_text = build_yake_source_text(
            title=getattr(row, "title"),
            description=getattr(
                row,
                "description",
            ),
        )

        try:
            keyword_records = extract_keywords(
                text=source_text,
                extractor=extractor,
            )
        except Exception as error:
            raise RuntimeError(
                "Ekstraksi YAKE gagal.\n"
                f"Nomor proses : {row_number:,}\n"
                f"Document ID  : {document_id}\n"
                f"Error        : {error}"
            ) from error

        raw_keywords = [
            record["keyword_raw"]
            for record in keyword_records
        ]

        model_keywords = [
            record["keyword_model"]
            for record in keyword_records
        ]

        scores = [
            record["score"]
            for record in keyword_records
        ]

        source_texts.append(
            source_text
        )

        # Kolom yang digunakan untuk skenario model.
        keyword_model_texts.append(
            " ".join(model_keywords)
        )

        # Tampilan phrase yang sudah dinormalisasi.
        keyword_display_texts.append(
            KEYWORD_SEPARATOR.join(
                model_keywords
            )
        )

        # Tampilan asli hasil YAKE untuk audit.
        keyword_raw_display_texts.append(
            KEYWORD_SEPARATOR.join(
                raw_keywords
            )
        )

        keyword_scores_json.append(
            json.dumps(
                scores,
                ensure_ascii=False,
            )
        )

        keyword_pairs_json.append(
            json.dumps(
                keyword_records,
                ensure_ascii=False,
            )
        )

        keyword_counts.append(
            len(keyword_records)
        )

        if (
            row_number % PROGRESS_INTERVAL == 0
            or row_number == total_rows
        ):
            elapsed = (
                time.perf_counter()
                - start_time
            )

            average_time = (
                elapsed / row_number
            )

            estimated_remaining = (
                average_time
                * (total_rows - row_number)
            )

            print(
                f"Progress: "
                f"{row_number:,}/{total_rows:,} artikel | "
                f"{elapsed:.2f} detik | "
                f"estimasi sisa "
                f"{estimated_remaining:.2f} detik"
            )

    result["yake_source_text"] = (
        source_texts
    )

    result["keyword_yake"] = (
        keyword_model_texts
    )

    result["keyword_yake_display"] = (
        keyword_display_texts
    )

    result["keyword_yake_raw_display"] = (
        keyword_raw_display_texts
    )

    result["keyword_yake_scores"] = (
        keyword_scores_json
    )

    result["keyword_yake_pairs_json"] = (
        keyword_pairs_json
    )

    result["keyword_yake_count"] = (
        keyword_counts
    )

    return result


# ============================================================
# VALIDASI HASIL YAKE
# ============================================================

def validate_yake_output(
    dataframe_before: pd.DataFrame,
    dataframe_after: pd.DataFrame,
) -> None:
    """
    Memastikan hasil YAKE lengkap dan konsisten.
    """

    if len(dataframe_before) != len(dataframe_after):
        raise ValueError(
            "Jumlah data berubah selama ekstraksi YAKE.\n"
            f"Sebelum: {len(dataframe_before):,}\n"
            f"Sesudah: {len(dataframe_after):,}"
        )

    required_output_columns = [
        "yake_source_text",
        "keyword_yake",
        "keyword_yake_display",
        "keyword_yake_raw_display",
        "keyword_yake_scores",
        "keyword_yake_pairs_json",
        "keyword_yake_count",
    ]

    missing_columns = [
        column
        for column in required_output_columns
        if column not in dataframe_after.columns
    ]

    if missing_columns:
        raise ValueError(
            "Kolom hasil YAKE tidak lengkap:\n"
            f"{missing_columns}"
        )

    before_ids = (
        dataframe_before["document_id"]
        .astype(str)
        .reset_index(drop=True)
    )

    after_ids = (
        dataframe_after["document_id"]
        .astype(str)
        .reset_index(drop=True)
    )

    if not before_ids.equals(after_ids):
        raise ValueError(
            "Urutan atau identitas dokumen berubah "
            "selama ekstraksi YAKE."
        )

    empty_source_count = int(
        dataframe_after["yake_source_text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_source_count > 0:
        raise ValueError(
            f"Ditemukan {empty_source_count:,} "
            "sumber teks YAKE kosong."
        )

    empty_keyword_count = int(
        dataframe_after["keyword_yake"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_keyword_count > 0:
        raise ValueError(
            f"Ditemukan {empty_keyword_count:,} "
            "artikel tanpa keyword YAKE."
        )

    invalid_count_mask = (
        dataframe_after["keyword_yake_count"]
        .lt(1)
        |
        dataframe_after["keyword_yake_count"]
        .gt(YAKE_TOP_KEYWORDS)
    )

    if invalid_count_mask.any():
        invalid_count = int(
            invalid_count_mask.sum()
        )

        raise ValueError(
            f"Ditemukan {invalid_count:,} artikel "
            "dengan jumlah keyword tidak valid."
        )

    for row_number, row in enumerate(
        dataframe_after.itertuples(index=False),
        start=1,
    ):
        keyword_count = int(
            getattr(
                row,
                "keyword_yake_count",
            )
        )

        try:
            scores = json.loads(
                getattr(
                    row,
                    "keyword_yake_scores",
                )
            )

            pairs = json.loads(
                getattr(
                    row,
                    "keyword_yake_pairs_json",
                )
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                "JSON hasil YAKE tidak valid pada "
                f"baris {row_number:,}."
            ) from error

        if len(scores) != keyword_count:
            raise ValueError(
                "Jumlah skor tidak sama dengan jumlah "
                f"keyword pada baris {row_number:,}."
            )

        if len(pairs) != keyword_count:
            raise ValueError(
                "Jumlah pasangan keyword-score tidak sama "
                f"pada baris {row_number:,}."
            )

        if any(
            not math.isfinite(float(score))
            for score in scores
        ):
            raise ValueError(
                "Ditemukan skor YAKE tidak valid pada "
                f"baris {row_number:,}."
            )


# ============================================================
# MEMBUAT LAPORAN YAKE
# ============================================================

def get_best_keyword_score(
    value: str,
) -> float | None:
    """
    Mengambil skor keyword terbaik dari JSON score.
    """

    try:
        scores = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None

    if not scores:
        return None

    return float(
        min(scores)
    )


def create_yake_report(
    dataframe: pd.DataFrame,
    elapsed_seconds: float,
) -> pd.DataFrame:
    """
    Membuat statistik hasil ekstraksi YAKE.
    """

    data = dataframe.copy()

    data["_best_keyword_score"] = (
        data["keyword_yake_scores"]
        .apply(get_best_keyword_score)
    )

    records: list[dict] = []

    grouped_data = list(
        data.groupby(
            "category",
            dropna=False,
            sort=True,
        )
    )

    grouped_data.append(
        ("ALL", data)
    )

    for category, group in grouped_data:
        empty_keyword_count = int(
            group["keyword_yake"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        record = {
            "dataset": "Kompas",
            "category": category,
            "jumlah_data": len(group),
            "avg_keyword_count": round(
                float(
                    group[
                        "keyword_yake_count"
                    ].mean()
                ),
                4,
            ),
            "minimum_keyword_count": int(
                group[
                    "keyword_yake_count"
                ].min()
            ),
            "maximum_keyword_count": int(
                group[
                    "keyword_yake_count"
                ].max()
            ),
            "avg_best_keyword_score": round(
                float(
                    group[
                        "_best_keyword_score"
                    ].mean()
                ),
                8,
            ),
            "artikel_tanpa_keyword": (
                empty_keyword_count
            ),
        }

        if category == "ALL":
            record[
                "processing_time_seconds"
            ] = round(
                elapsed_seconds,
                4,
            )

            record[
                "average_seconds_per_article"
            ] = round(
                elapsed_seconds
                / len(dataframe),
                8,
            )

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# MEMBUAT SAMPEL HASIL
# ============================================================

def create_yake_samples(
    dataframe: pd.DataFrame,
    samples_per_category: int = 5,
) -> pd.DataFrame:
    """
    Mengambil sampel keyword per kategori.
    """

    samples: list[pd.DataFrame] = []

    for category, group in dataframe.groupby(
        "category",
        dropna=False,
        sort=True,
    ):
        sample_size = min(
            samples_per_category,
            len(group),
        )

        sampled_group = (
            group.sample(
                n=sample_size,
                random_state=RANDOM_SEED,
            )
            [
                [
                    "document_id",
                    "category",
                    "title",
                    "description",
                    "yake_source_text",
                    "keyword_yake_raw_display",
                    "keyword_yake_display",
                    "keyword_yake",
                    "keyword_yake_scores",
                    "keyword_yake_count",
                ]
            ]
            .copy()
        )

        samples.append(
            sampled_group
        )

    return pd.concat(
        samples,
        ignore_index=True,
    )


# ============================================================
# MENYIMPAN KONFIGURASI
# ============================================================

def save_yake_configuration() -> None:
    """
    Menyimpan konfigurasi YAKE agar eksperimen reproducible.
    """

    configuration = {
        "algorithm": "YAKE",
        "library_version": get_yake_version(),
        "language": YAKE_LANGUAGE,
        "source_dataset": (
            "kompas_preprocessed.csv"
        ),
        "source_columns": [
            "title",
            "description",
        ],
        "source_text": (
            "original cleaned title + "
            "original cleaned description"
        ),
        "source_uses_lowercase_text": False,
        "source_preserves_casing": True,
        "source_preserves_sentence_punctuation": True,
        "content_used": False,
        "max_ngram_size": (
            YAKE_MAX_NGRAM_SIZE
        ),
        "deduplication_limit": (
            YAKE_DEDUPLICATION_LIMIT
        ),
        "deduplication_function": (
            YAKE_DEDUPLICATION_FUNCTION
        ),
        "window_size": (
            YAKE_WINDOW_SIZE
        ),
        "top_keywords": (
            YAKE_TOP_KEYWORDS
        ),
        "keyword_separator": (
            KEYWORD_SEPARATOR
        ),
        "model_keyword_column": (
            "keyword_yake"
        ),
        "model_keyword_normalization": (
            "light preprocessing consistent "
            "with text preprocessing"
        ),
        "score_interpretation": (
            "lower score means more important keyword"
        ),
        "random_seed_for_sampling": (
            RANDOM_SEED
        ),
        "expected_row_count": (
            EXPECTED_ROW_COUNT
        ),
    }

    with open(
        YAKE_CONFIG_PATH,
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
# PROGRAM UTAMA
# ============================================================

def main() -> None:
    """
    Menjalankan ekstraksi keyword YAKE pada Kompas.
    """

    print("=" * 72)
    print("STEP 4.3 - YAKE KEYWORD EXTRACTION")
    print("=" * 72)

    print("\nKonfigurasi YAKE:")
    print(
        f"Versi library         : "
        f"{get_yake_version()}"
    )
    print(
        f"Bahasa                : "
        f"{YAKE_LANGUAGE}"
    )
    print(
        f"Maksimum n-gram       : "
        f"{YAKE_MAX_NGRAM_SIZE}"
    )
    print(
        f"Jumlah keyword        : "
        f"{YAKE_TOP_KEYWORDS}"
    )
    print(
        f"Deduplication limit   : "
        f"{YAKE_DEDUPLICATION_LIMIT}"
    )
    print(
        f"Deduplication function: "
        f"{YAKE_DEDUPLICATION_FUNCTION}"
    )
    print(
        f"Window size           : "
        f"{YAKE_WINDOW_SIZE}"
    )
    print(
        "Sumber teks           : "
        "title asli + description asli"
    )
    print(
        "Content digunakan     : Tidak"
    )

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # MEMUAT DAN MEMVALIDASI DATA
    # --------------------------------------------------------

    kompas = load_dataset(
        KOMPAS_PREPROCESSED_PATH
    )

    validate_input_dataset(
        kompas
    )

    kompas_before = kompas.copy()

    print(
        f"\nJumlah artikel Kompas: "
        f"{len(kompas):,}"
    )

    print("\nDistribusi kategori:")

    print(
        kompas["category"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # MEMBUAT EXTRACTOR
    # --------------------------------------------------------

    extractor = create_yake_extractor()

    # --------------------------------------------------------
    # EKSTRAKSI KEYWORD
    # --------------------------------------------------------

    print("\nMemulai ekstraksi keyword YAKE...")

    start_time = time.perf_counter()

    kompas_with_keywords = apply_yake_to_dataset(
        dataframe=kompas,
        extractor=extractor,
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # VALIDASI HASIL
    # --------------------------------------------------------

    validate_yake_output(
        dataframe_before=kompas_before,
        dataframe_after=kompas_with_keywords,
    )

    # --------------------------------------------------------
    # LAPORAN
    # --------------------------------------------------------

    yake_report = create_yake_report(
        dataframe=kompas_with_keywords,
        elapsed_seconds=elapsed_seconds,
    )

    yake_samples = create_yake_samples(
        dataframe=kompas_with_keywords,
        samples_per_category=5,
    )

    # --------------------------------------------------------
    # MENYIMPAN OUTPUT
    # --------------------------------------------------------

    kompas_with_keywords.to_csv(
        KOMPAS_WITH_KEYWORDS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    yake_report.to_csv(
        YAKE_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    yake_samples.to_csv(
        YAKE_SAMPLES_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    save_yake_configuration()

    # --------------------------------------------------------
    # MENAMPILKAN HASIL
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("HASIL YAKE KEYWORD EXTRACTION")
    print("=" * 72)

    print(
        yake_report.to_string(
            index=False
        )
    )

    print(
        f"\nWaktu pemrosesan total: "
        f"{elapsed_seconds:.2f} detik"
    )

    print("\nContoh hasil keyword:")

    print(
        kompas_with_keywords[
            [
                "document_id",
                "category",
                "title",
                "keyword_yake_raw_display",
                "keyword_yake_display",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nValidasi hasil:")

    print(
        f"Jumlah data akhir     : "
        f"{len(kompas_with_keywords):,}"
    )

    print(
        "Artikel tanpa keyword: "
        f"{int(kompas_with_keywords['keyword_yake_count'].eq(0).sum()):,}"
    )

    print(
        "Rata-rata keyword     : "
        f"{kompas_with_keywords['keyword_yake_count'].mean():.2f}"
    )

    # --------------------------------------------------------
    # INFORMASI OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("OUTPUT YAKE KEYWORD EXTRACTION")
    print("=" * 72)

    print("\nDataset Kompas dengan keyword:")
    print(KOMPAS_WITH_KEYWORDS_PATH)

    print("\nLaporan ekstraksi keyword:")
    print(YAKE_REPORT_PATH)

    print("\nContoh hasil YAKE:")
    print(YAKE_SAMPLES_PATH)

    print("\nKonfigurasi YAKE:")
    print(YAKE_CONFIG_PATH)

    print(
        "\nTahap YAKE keyword extraction selesai."
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yake


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
# MEMBACA DATASET
# ============================================================

def load_dataset(
    file_path: Path,
) -> pd.DataFrame:
    """
    Membaca dataset Kompas hasil preprocessing.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            "Dataset Kompas preprocessed tidak ditemukan:\n"
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

    required_columns = [
        "document_id",
        "title_preprocessed",
        "description_preprocessed",
        "category",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Kolom yang dibutuhkan tidak tersedia: "
            f"{missing_columns}"
        )

    return dataframe


# ============================================================
# MEMBUAT YAKE EXTRACTOR
# ============================================================

def create_yake_extractor() -> yake.KeywordExtractor:
    """
    Membuat satu instance YAKE KeywordExtractor.

    Instance digunakan ulang untuk seluruh artikel agar
    proses ekstraksi lebih efisien.
    """

    return yake.KeywordExtractor(
        lan=YAKE_LANGUAGE,
        n=YAKE_MAX_NGRAM_SIZE,
        dedupLim=YAKE_DEDUPLICATION_LIMIT,
        dedupFunc=YAKE_DEDUPLICATION_FUNCTION,
        windowsSize=YAKE_WINDOW_SIZE,
        top=YAKE_TOP_KEYWORDS,
        features=None,
    )


# ============================================================
# MENGGABUNGKAN TITLE DAN DESCRIPTION
# ============================================================

def build_yake_source_text(
    title: Any,
    description: Any,
) -> str:
    """
    Membentuk input YAKE dari:

    title_preprocessed + description_preprocessed

    Content tidak digunakan agar skenario 3 tidak membawa
    informasi dari content.
    """

    title_text = (
        ""
        if title is None
        else str(title).strip()
    )

    description_text = (
        ""
        if description is None
        else str(description).strip()
    )

    parts = [
        text
        for text in [
            title_text,
            description_text,
        ]
        if text
    ]

    return " ".join(parts).strip()


# ============================================================
# MEMBERSIHKAN HASIL KEYWORD
# ============================================================

def normalize_keyword(
    keyword: Any,
) -> str:
    """
    Merapikan keyword hasil YAKE.
    """

    if keyword is None:
        return ""

    return " ".join(
        str(keyword)
        .strip()
        .split()
    )


# ============================================================
# EKSTRAKSI KEYWORD SATU DOKUMEN
# ============================================================

def extract_keywords(
    text: str,
    extractor: yake.KeywordExtractor,
) -> tuple[list[str], list[float]]:
    """
    Mengekstraksi keyword dan score YAKE.

    Score YAKE yang lebih kecil menunjukkan keyword yang
    dianggap lebih penting.
    """

    if not text.strip():
        return [], []

    raw_keywords = extractor.extract_keywords(
        text
    )

    keywords: list[str] = []
    scores: list[float] = []
    seen_keywords: set[str] = set()

    for keyword, score in raw_keywords:
        normalized_keyword = normalize_keyword(
            keyword
        )

        if not normalized_keyword:
            continue

        duplicate_key = normalized_keyword.lower()

        if duplicate_key in seen_keywords:
            continue

        seen_keywords.add(
            duplicate_key
        )

        keywords.append(
            normalized_keyword
        )

        scores.append(
            round(float(score), 8)
        )

        if len(keywords) >= YAKE_TOP_KEYWORDS:
            break

    return keywords, scores


# ============================================================
# MEMPROSES SELURUH DATASET
# ============================================================

def apply_yake_to_dataset(
    dataframe: pd.DataFrame,
    extractor: yake.KeywordExtractor,
) -> pd.DataFrame:
    """
    Menerapkan YAKE ke seluruh artikel Kompas.
    """

    dataframe = dataframe.copy()

    source_texts: list[str] = []
    keyword_texts: list[str] = []
    keyword_lists: list[str] = []
    keyword_scores: list[str] = []
    keyword_counts: list[int] = []

    total_rows = len(dataframe)

    start_time = time.perf_counter()

    for row_number, row in enumerate(
        dataframe.itertuples(index=False),
        start=1,
    ):
        source_text = build_yake_source_text(
            getattr(
                row,
                "title_preprocessed",
            ),
            getattr(
                row,
                "description_preprocessed",
            ),
        )

        keywords, scores = extract_keywords(
            text=source_text,
            extractor=extractor,
        )

        source_texts.append(
            source_text
        )

        keyword_texts.append(
            " ".join(keywords)
        )

        keyword_lists.append(
            KEYWORD_SEPARATOR.join(keywords)
        )

        keyword_scores.append(
            json.dumps(
                scores,
                ensure_ascii=False,
            )
        )

        keyword_counts.append(
            len(keywords)
        )

        if (
            row_number % PROGRESS_INTERVAL == 0
            or row_number == total_rows
        ):
            elapsed_time = (
                time.perf_counter()
                - start_time
            )

            print(
                f"Progress: "
                f"{row_number:,}/{total_rows:,} "
                f"artikel | "
                f"{elapsed_time:.2f} detik"
            )

    dataframe["yake_source_text"] = (
        source_texts
    )

    dataframe["keyword_yake"] = (
        keyword_texts
    )

    dataframe["keyword_yake_display"] = (
        keyword_lists
    )

    dataframe["keyword_yake_scores"] = (
        keyword_scores
    )

    dataframe["keyword_yake_count"] = (
        keyword_counts
    )

    return dataframe


# ============================================================
# VALIDASI HASIL YAKE
# ============================================================

def validate_yake_output(
    dataframe: pd.DataFrame,
) -> None:
    """
    Memastikan hasil ekstraksi YAKE dapat digunakan.
    """

    required_columns = [
        "yake_source_text",
        "keyword_yake",
        "keyword_yake_display",
        "keyword_yake_scores",
        "keyword_yake_count",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Kolom hasil YAKE tidak lengkap: "
            f"{missing_columns}"
        )

    empty_source_count = int(
        dataframe["yake_source_text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    empty_keyword_count = int(
        dataframe["keyword_yake"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_source_count > 0:
        raise ValueError(
            f"Ditemukan {empty_source_count} "
            "input YAKE kosong."
        )

    if empty_keyword_count > 0:
        print(
            f"Peringatan: terdapat "
            f"{empty_keyword_count} artikel "
            f"tanpa keyword YAKE."
        )


# ============================================================
# MEMBUAT LAPORAN YAKE
# ============================================================

def create_yake_report(
    dataframe: pd.DataFrame,
    elapsed_seconds: float,
) -> pd.DataFrame:
    """
    Membuat laporan statistik ekstraksi YAKE.
    """

    records: list[dict] = []

    for category, group in dataframe.groupby(
        "category",
        dropna=False,
    ):
        empty_keywords = int(
            group["keyword_yake"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        records.append(
            {
                "dataset": "Kompas",
                "category": category,
                "jumlah_data": len(group),
                "avg_keyword_count": round(
                    float(
                        group[
                            "keyword_yake_count"
                        ].mean()
                    ),
                    2,
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
                "artikel_tanpa_keyword":
                    empty_keywords,
            }
        )

    overall_empty_keywords = int(
        dataframe["keyword_yake"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    records.append(
        {
            "dataset": "Kompas",
            "category": "ALL",
            "jumlah_data": len(dataframe),
            "avg_keyword_count": round(
                float(
                    dataframe[
                        "keyword_yake_count"
                    ].mean()
                ),
                2,
            ),
            "minimum_keyword_count": int(
                dataframe[
                    "keyword_yake_count"
                ].min()
            ),
            "maximum_keyword_count": int(
                dataframe[
                    "keyword_yake_count"
                ].max()
            ),
            "artikel_tanpa_keyword":
                overall_empty_keywords,
            "processing_time_seconds": round(
                elapsed_seconds,
                2,
            ),
            "average_seconds_per_article": round(
                elapsed_seconds
                / len(dataframe),
                6,
            ),
        }
    )

    return pd.DataFrame(records)


# ============================================================
# MEMBUAT SAMPEL HASIL YAKE
# ============================================================

def create_yake_samples(
    dataframe: pd.DataFrame,
    samples_per_category: int = 5,
) -> pd.DataFrame:
    """
    Mengambil contoh hasil keyword per kategori.
    """

    samples: list[pd.DataFrame] = []

    for category, group in dataframe.groupby(
        "category",
        dropna=False,
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
                    "keyword_yake_display",
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
    Menyimpan parameter YAKE agar eksperimen reproducible.
    """

    configuration = {
        "algorithm": "YAKE",
        "language": YAKE_LANGUAGE,
        "source_text": (
            "title_preprocessed + "
            "description_preprocessed"
        ),
        "max_ngram_size":
            YAKE_MAX_NGRAM_SIZE,
        "deduplication_limit":
            YAKE_DEDUPLICATION_LIMIT,
        "deduplication_function":
            YAKE_DEDUPLICATION_FUNCTION,
        "window_size":
            YAKE_WINDOW_SIZE,
        "top_keywords":
            YAKE_TOP_KEYWORDS,
        "keyword_separator":
            KEYWORD_SEPARATOR,
        "content_used": False,
        "random_seed":
            RANDOM_SEED,
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
        f"Bahasa               : "
        f"{YAKE_LANGUAGE}"
    )
    print(
        f"Maksimum n-gram      : "
        f"{YAKE_MAX_NGRAM_SIZE}"
    )
    print(
        f"Jumlah keyword       : "
        f"{YAKE_TOP_KEYWORDS}"
    )
    print(
        f"Deduplication limit  : "
        f"{YAKE_DEDUPLICATION_LIMIT}"
    )
    print(
        f"Sumber teks          : "
        f"title_preprocessed + "
        f"description_preprocessed"
    )

    # ========================================================
    # MEMUAT DATASET
    # ========================================================

    kompas = load_dataset(
        KOMPAS_PREPROCESSED_PATH
    )

    print(
        f"\nJumlah artikel Kompas: "
        f"{len(kompas):,}"
    )

    # ========================================================
    # MEMBUAT EXTRACTOR
    # ========================================================

    extractor = create_yake_extractor()

    # ========================================================
    # EKSTRAKSI KEYWORD
    # ========================================================

    start_time = time.perf_counter()

    kompas_with_keywords = (
        apply_yake_to_dataset(
            dataframe=kompas,
            extractor=extractor,
        )
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    # ========================================================
    # VALIDASI
    # ========================================================

    validate_yake_output(
        kompas_with_keywords
    )

    # ========================================================
    # LAPORAN
    # ========================================================

    yake_report = create_yake_report(
        dataframe=kompas_with_keywords,
        elapsed_seconds=elapsed_seconds,
    )

    yake_samples = create_yake_samples(
        dataframe=kompas_with_keywords,
        samples_per_category=5,
    )

    # ========================================================
    # MEMBUAT FOLDER
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
    # MENYIMPAN OUTPUT
    # ========================================================

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

    # ========================================================
    # MENAMPILKAN HASIL
    # ========================================================

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
                "keyword_yake_display",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # ========================================================
    # INFORMASI OUTPUT
    # ========================================================

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
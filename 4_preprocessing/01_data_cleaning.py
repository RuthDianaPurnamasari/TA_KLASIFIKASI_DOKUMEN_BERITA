from __future__ import annotations

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
    AG_NEWS_TEST_PROCESSED_PATH,
    AG_NEWS_TRAIN_PROCESSED_PATH,
    KOMPAS_PROCESSED_PATH,
    PROCESSED_DATA_DIR,
    TABLES_DIR,
)


# ============================================================
# PATH OUTPUT DATASET CLEAN
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
# PATH OUTPUT LAPORAN
# ============================================================

DATA_CLEANING_REPORT_PATH = (
    TABLES_DIR
    / "data_cleaning_report.csv"
)

REMOVED_DUPLICATES_PATH = (
    TABLES_DIR
    / "removed_duplicates.csv"
)

REMOVED_TRAIN_TEST_OVERLAP_PATH = (
    TABLES_DIR
    / "removed_agnews_train_test_overlap.csv"
)


# ============================================================
# MEMBACA DATASET
# ============================================================

def load_dataset(
    file_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Membaca dataset processed.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset {dataset_name} tidak ditemukan:\n"
            f"{file_path}"
        )

    dataframe = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
    )

    if dataframe.empty:
        raise ValueError(
            f"Dataset {dataset_name} kosong."
        )

    return dataframe


# ============================================================
# STANDARDISASI TEKS DASAR
# ============================================================

def standardize_text(
    value,
) -> str:
    """
    Melakukan standardisasi struktur teks dasar.

    Proses:
    - NaN menjadi string kosong
    - Unicode dinormalisasi
    - karakter whitespace berlebih dirapikan

    Fungsi ini BELUM melakukan:
    - lowercase
    - stopword removal
    - stemming
    - lemmatization

    Tahapan NLP tersebut dilakukan pada script berikutnya.
    """

    if pd.isna(value):
        return ""

    text = str(value)

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# NORMALISASI UNTUK DETEKSI DUPLIKAT
# ============================================================

def normalize_for_duplicate(
    value,
) -> str:
    """
    Membuat representasi teks untuk mendeteksi duplikat.

    Normalisasi dibuat konsisten dengan tahap EDA:
    - menangani nilai kosong;
    - normalisasi Unicode;
    - lowercase;
    - menghapus spasi awal dan akhir;
    - merapikan whitespace berlebih.

    Tanda baca tidak dihapus agar artikel yang hanya mirip
    tidak dianggap sebagai artikel identik.
    """

    text = standardize_text(value)

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()

# ============================================================
# STANDARDISASI KOLOM TEKS
# ============================================================

def standardize_text_columns(
    dataframe: pd.DataFrame,
    text_columns: list[str],
) -> pd.DataFrame:
    """
    Melakukan standardisasi dasar pada kolom teks.
    """

    dataframe = dataframe.copy()

    for column in text_columns:

        if column not in dataframe.columns:
            raise ValueError(
                f"Kolom '{column}' tidak ditemukan."
            )

        dataframe[column] = (
            dataframe[column]
            .apply(standardize_text)
        )

    return dataframe


# ============================================================
# MEMBERSIHKAN DUPLIKAT KOMPAS
# ============================================================

def clean_kompas_duplicates(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Menghapus duplikat content Kompas berdasarkan
    content yang telah dinormalisasi.

    Title tidak digunakan sebagai dasar penghapusan karena
    hasil EDA menunjukkan tidak ada duplikat title.
    """

    dataframe = dataframe.copy()

    dataframe["_duplicate_key"] = (
        dataframe["content"]
        .apply(normalize_for_duplicate)
    )

    duplicate_mask = (
        dataframe.duplicated(
            subset=["_duplicate_key"],
            keep="first",
        )
        & dataframe["_duplicate_key"].ne("")
    )

    removed_duplicates = (
        dataframe.loc[
            duplicate_mask
        ]
        .copy()
    )

    cleaned_dataframe = (
        dataframe.loc[
            ~duplicate_mask
        ]
        .copy()
    )

    removed_duplicates[
        "duplicate_type"
    ] = "normalized_content"

    removed_duplicates[
        "dataset"
    ] = "Kompas"

    cleaned_dataframe = (
        cleaned_dataframe
        .drop(
            columns=["_duplicate_key"]
        )
        .reset_index(drop=True)
    )

    removed_duplicates = (
        removed_duplicates
        .drop(
            columns=["_duplicate_key"]
        )
        .reset_index(drop=True)
    )

    return (
        cleaned_dataframe,
        removed_duplicates,
    )


# ============================================================
# MEMBUAT KEY ARTIKEL AG NEWS
# ============================================================

def create_agnews_article_key(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Membuat kunci artikel AG News berdasarkan kombinasi:
    class_index + normalized title + normalized description.

    Label dimasukkan agar artikel dengan teks sama tetapi berada
    pada kategori berbeda tidak otomatis dianggap duplikat.
    """

    normalized_title = (
        dataframe["title"]
        .apply(normalize_for_duplicate)
    )

    normalized_description = (
        dataframe["description"]
        .apply(normalize_for_duplicate)
    )

    normalized_class = (
        dataframe["class_index"]
        .astype("Int64")
        .astype("string")
        .fillna("")
    )

    article_key = (
        normalized_class
        + " || "
        + normalized_title
        + " || "
        + normalized_description
    )

    return article_key

# ============================================================
# MEMBERSIHKAN DUPLIKAT INTERNAL AG NEWS
# ============================================================

def clean_agnews_duplicates(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Menghapus artikel duplikat berdasarkan kombinasi
    normalized title + normalized description.
    """

    dataframe = dataframe.copy()

    dataframe["_article_key"] = (
        create_agnews_article_key(
            dataframe
        )
    )

    duplicate_mask = (
        dataframe.duplicated(
            subset=["_article_key"],
            keep="first",
        )
    )

    removed_duplicates = (
        dataframe.loc[
            duplicate_mask
        ]
        .copy()
    )

    cleaned_dataframe = (
        dataframe.loc[
            ~duplicate_mask
        ]
        .copy()
    )

    removed_duplicates[
        "duplicate_type"
    ] = (
        "normalized_title_description"
    )

    removed_duplicates[
        "dataset"
    ] = dataset_name

    cleaned_dataframe = (
        cleaned_dataframe
        .drop(
            columns=["_article_key"]
        )
        .reset_index(drop=True)
    )

    removed_duplicates = (
        removed_duplicates
        .drop(
            columns=["_article_key"]
        )
        .reset_index(drop=True)
    )

    return (
        cleaned_dataframe,
        removed_duplicates,
    )


# ============================================================
# MENGHAPUS OVERLAP TRAIN-TEST AG NEWS
# ============================================================

def remove_agnews_train_test_overlap(
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Menghapus artikel pada test yang identik dengan artikel
    pada train berdasarkan normalized title + description.

    Train dipertahankan.
    Artikel overlap dihapus dari test.
    """

    train_dataframe = (
        train_dataframe.copy()
    )

    test_dataframe = (
        test_dataframe.copy()
    )

    train_keys = set(
        create_agnews_article_key(
            train_dataframe
        )
    )

    test_dataframe[
        "_article_key"
    ] = create_agnews_article_key(
        test_dataframe
    )

    overlap_mask = (
        test_dataframe[
            "_article_key"
        ]
        .isin(train_keys)
    )

    removed_overlap = (
        test_dataframe.loc[
            overlap_mask
        ]
        .copy()
    )

    cleaned_test = (
        test_dataframe.loc[
            ~overlap_mask
        ]
        .copy()
    )

    removed_overlap[
        "removal_reason"
    ] = "train_test_overlap"

    cleaned_test = (
        cleaned_test
        .drop(
            columns=["_article_key"]
        )
        .reset_index(drop=True)
    )

    removed_overlap = (
        removed_overlap
        .drop(
            columns=["_article_key"]
        )
        .reset_index(drop=True)
    )

    return (
        cleaned_test,
        removed_overlap,
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

    return int(
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )


# ============================================================
# MEMBUAT LAPORAN CLEANING
# ============================================================

def create_cleaning_report_row(
    dataset_name: str,
    original_count: int,
    cleaned_count: int,
    duplicates_removed: int,
    overlap_removed: int,
    empty_description_after_cleaning: int,
) -> dict:
    """
    Membuat satu baris laporan data cleaning.
    """

    return {
        "dataset": dataset_name,
        "jumlah_data_awal": original_count,
        "duplikat_dihapus": duplicates_removed,
        "overlap_dihapus": overlap_removed,
        "jumlah_data_akhir": cleaned_count,
        "description_kosong_setelah_cleaning":
            empty_description_after_cleaning,
    }


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main() -> None:

    print("=" * 72)
    print("STEP 4.1 - DATA CLEANING")
    print("=" * 72)

    # ========================================================
    # MEMUAT DATASET
    # ========================================================

    kompas = load_dataset(
        KOMPAS_PROCESSED_PATH,
        "Kompas",
    )

    agnews_train = load_dataset(
        AG_NEWS_TRAIN_PROCESSED_PATH,
        "AG News Train",
    )

    agnews_test = load_dataset(
        AG_NEWS_TEST_PROCESSED_PATH,
        "AG News Test",
    )

    # Menyimpan jumlah data awal
    kompas_original_count = len(
        kompas
    )

    agnews_train_original_count = len(
        agnews_train
    )

    agnews_test_original_count = len(
        agnews_test
    )

    # ========================================================
    # STANDARDISASI KOLOM TEKS
    # ========================================================

    kompas = standardize_text_columns(
        dataframe=kompas,
        text_columns=[
            "title",
            "description",
            "content",
        ],
    )

    agnews_train = standardize_text_columns(
        dataframe=agnews_train,
        text_columns=[
            "title",
            "description",
        ],
    )

    agnews_test = standardize_text_columns(
        dataframe=agnews_test,
        text_columns=[
            "title",
            "description",
        ],
    )

    # ========================================================
    # CLEANING DUPLIKAT KOMPAS
    # ========================================================

    (
        kompas_clean,
        kompas_removed_duplicates,
    ) = clean_kompas_duplicates(
        kompas
    )

    # ========================================================
    # CLEANING DUPLIKAT AG NEWS TRAIN
    # ========================================================

    (
        agnews_train_clean,
        agnews_train_removed_duplicates,
    ) = clean_agnews_duplicates(
        dataframe=agnews_train,
        dataset_name="AG News Train",
    )

    # ========================================================
    # CLEANING DUPLIKAT AG NEWS TEST
    # ========================================================

    (
        agnews_test_clean,
        agnews_test_removed_duplicates,
    ) = clean_agnews_duplicates(
        dataframe=agnews_test,
        dataset_name="AG News Test",
    )

    # ========================================================
    # MENGHAPUS TRAIN-TEST OVERLAP
    # ========================================================

    (
        agnews_test_clean,
        removed_overlap,
    ) = remove_agnews_train_test_overlap(
        train_dataframe=agnews_train_clean,
        test_dataframe=agnews_test_clean,
    )

    # ========================================================
    # MENGGABUNGKAN LOG DUPLIKAT
    # ========================================================

    removed_duplicates = pd.concat(
        [
            kompas_removed_duplicates,
            agnews_train_removed_duplicates,
            agnews_test_removed_duplicates,
        ],
        ignore_index=True,
        sort=False,
    )

    # ========================================================
    # MEMBUAT LAPORAN
    # ========================================================

    cleaning_report = pd.DataFrame(
        [
            create_cleaning_report_row(
                dataset_name="Kompas",
                original_count=(
                    kompas_original_count
                ),
                cleaned_count=len(
                    kompas_clean
                ),
                duplicates_removed=len(
                    kompas_removed_duplicates
                ),
                overlap_removed=0,
                empty_description_after_cleaning=(
                    count_empty_text(
                        kompas_clean[
                            "description"
                        ]
                    )
                ),
            ),
            create_cleaning_report_row(
                dataset_name="AG News Train",
                original_count=(
                    agnews_train_original_count
                ),
                cleaned_count=len(
                    agnews_train_clean
                ),
                duplicates_removed=len(
                    agnews_train_removed_duplicates
                ),
                overlap_removed=0,
                empty_description_after_cleaning=(
                    count_empty_text(
                        agnews_train_clean[
                            "description"
                        ]
                    )
                ),
            ),
            create_cleaning_report_row(
                dataset_name="AG News Test",
                original_count=(
                    agnews_test_original_count
                ),
                cleaned_count=len(
                    agnews_test_clean
                ),
                duplicates_removed=len(
                    agnews_test_removed_duplicates
                ),
                overlap_removed=len(
                    removed_overlap
                ),
                empty_description_after_cleaning=(
                    count_empty_text(
                        agnews_test_clean[
                            "description"
                        ]
                    )
                ),
            ),
        ]
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
    # MENYIMPAN DATASET CLEAN
    # ========================================================

    kompas_clean.to_csv(
        KOMPAS_CLEAN_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    agnews_train_clean.to_csv(
        AG_NEWS_TRAIN_CLEAN_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    agnews_test_clean.to_csv(
        AG_NEWS_TEST_CLEAN_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # MENYIMPAN LAPORAN
    # ========================================================

    cleaning_report.to_csv(
        DATA_CLEANING_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    removed_duplicates.to_csv(
        REMOVED_DUPLICATES_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    removed_overlap.to_csv(
        REMOVED_TRAIN_TEST_OVERLAP_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # MENAMPILKAN HASIL
    # ========================================================

    print("\n" + "=" * 72)
    print("HASIL DATA CLEANING")
    print("=" * 72)

    print(
        cleaning_report.to_string(
            index=False
        )
    )

    print("\nDetail penghapusan:")

    print(
        f"Kompas - duplikat dihapus      : "
        f"{len(kompas_removed_duplicates):,}"
    )

    print(
        f"AG News Train - duplikat       : "
        f"{len(agnews_train_removed_duplicates):,}"
    )

    print(
        f"AG News Test - duplikat        : "
        f"{len(agnews_test_removed_duplicates):,}"
    )

    print(
        f"AG News Test - overlap train   : "
        f"{len(removed_overlap):,}"
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print("\n" + "=" * 72)
    print("OUTPUT DATA CLEANING")
    print("=" * 72)

    print("\nDataset Kompas clean:")
    print(KOMPAS_CLEAN_PATH)

    print("\nDataset AG News Train clean:")
    print(AG_NEWS_TRAIN_CLEAN_PATH)

    print("\nDataset AG News Test clean:")
    print(AG_NEWS_TEST_CLEAN_PATH)

    print("\nLaporan data cleaning:")
    print(DATA_CLEANING_REPORT_PATH)

    print("\nLog duplikat yang dihapus:")
    print(REMOVED_DUPLICATES_PATH)

    print("\nLog overlap train-test yang dihapus:")
    print(REMOVED_TRAIN_TEST_OVERLAP_PATH)

    print("\nTahap data cleaning selesai.")


if __name__ == "__main__":
    main()
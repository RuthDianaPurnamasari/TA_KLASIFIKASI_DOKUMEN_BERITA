from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# MENAMBAHKAN ROOT PROJECT KE PYTHON PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config import (  # noqa: E402
    KOMPAS_EXPECTED_CATEGORIES,
    KOMPAS_EXPECTED_COLUMNS,
    KOMPAS_PROCESSED_PATH,
    KOMPAS_RAW_FILES,
    KOMPAS_VALIDATION_REPORT,
    RANDOM_SEED,
)


# ============================================================
# FUNGSI MEMBACA DATASET
# ============================================================

def read_kompas_file(file_path: Path) -> pd.DataFrame:
    """
    Membaca dataset mentah hasil crawling Kompas.

    Dataset Kompas menggunakan delimiter titik koma (;).
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan:\n{file_path}"
        )

    try:
        dataframe = pd.read_csv(
            file_path,
            sep=";",
            encoding="utf-8-sig",
        )
    except UnicodeDecodeError:
        dataframe = pd.read_csv(
            file_path,
            sep=";",
            encoding="latin-1",
        )

    return dataframe


# ============================================================
# FUNGSI NORMALISASI NAMA KOLOM
# ============================================================

def normalize_column_names(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Menyeragamkan nama kolom menjadi huruf kecil
    dan mengganti spasi dengan underscore.
    """

    dataframe = dataframe.copy()

    dataframe.columns = (
        dataframe.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    return dataframe


# ============================================================
# FUNGSI NORMALISASI NILAI TEKS
# ============================================================

def normalize_string_values(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merapikan spasi teknis pada kolom teks.

    Proses ini belum termasuk preprocessing NLP.
    Kata, tanda baca, dan isi teks belum dihapus.
    """

    dataframe = dataframe.copy()

    text_columns = [
        "title",
        "description",
        "content",
        "category",
        "link",
    ]

    for column in text_columns:
        dataframe[column] = (
            dataframe[column]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    dataframe["category"] = (
        dataframe["category"]
        .str.lower()
        .str.strip()
    )

    return dataframe


# ============================================================
# FUNGSI VALIDASI KOLOM
# ============================================================

def validate_columns(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Memastikan seluruh kolom wajib tersedia.
    """

    missing_columns = [
        column
        for column in KOMPAS_EXPECTED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset '{dataset_name}' tidak memiliki kolom: "
            f"{missing_columns}"
        )


# ============================================================
# FUNGSI VALIDASI KATEGORI
# ============================================================

def validate_category(
    dataframe: pd.DataFrame,
    expected_category: str,
) -> None:
    """
    Memastikan kategori di dalam file sesuai nama file.
    """

    actual_categories = set(
        dataframe["category"]
        .dropna()
        .str.lower()
        .unique()
    )

    if actual_categories != {expected_category}:
        raise ValueError(
            f"Kategori file '{expected_category}' tidak sesuai.\n"
            f"Kategori ditemukan: {actual_categories}"
        )


# ============================================================
# FUNGSI KONVERSI TANGGAL KOMPAS
# ============================================================

def parse_kompas_date(
    date_series: pd.Series,
) -> pd.Series:
    """
    Mengubah beberapa variasi tanggal Kompas menjadi datetime.

    Format yang ditangani:

    1. Kompas.com, 20 Mei 2026, 22:20 WIB
    2. Kompas.com , 20 Mei 2026, 22:20 WIB
    3. Kompas.com, Diperbarui 20/05/2026, 10:37 WIB

    Hasil:
    2026-05-20 22:20:00
    """

    month_mapping = {
        "januari": "01",
        "februari": "02",
        "maret": "03",
        "april": "04",
        "mei": "05",
        "juni": "06",
        "juli": "07",
        "agustus": "08",
        "september": "09",
        "oktober": "10",
        "november": "11",
        "desember": "12",
    }

    cleaned_date = (
        date_series
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Membuat Series kosong untuk menampung hasil akhir
    parsed_date = pd.Series(
        pd.NaT,
        index=date_series.index,
        dtype="datetime64[ns]",
    )

    # ========================================================
    # FORMAT 1: 20 Mei 2026, 22:20
    # ========================================================

    pattern_text_month = (
        r"(?P<day>\d{1,2})\s+"
        r"(?P<month>"
        r"Januari|Februari|Maret|April|Mei|Juni|Juli|"
        r"Agustus|September|Oktober|November|Desember"
        r")\s+"
        r"(?P<year>\d{4})\s*,\s*"
        r"(?P<hour>\d{1,2})\s*:\s*"
        r"(?P<minute>\d{2})"
    )

    extracted_text_month = cleaned_date.str.extract(
        pattern_text_month,
        flags=re.IGNORECASE,
    )

    valid_text_month = (
        extracted_text_month.notna().all(axis=1)
    )

    text_month_datetime = (
        extracted_text_month.loc[
            valid_text_month,
            "year",
        ]
        + "-"
        + extracted_text_month.loc[
            valid_text_month,
            "month",
        ]
        .str.lower()
        .map(month_mapping)
        + "-"
        + extracted_text_month.loc[
            valid_text_month,
            "day",
        ].str.zfill(2)
        + " "
        + extracted_text_month.loc[
            valid_text_month,
            "hour",
        ].str.zfill(2)
        + ":"
        + extracted_text_month.loc[
            valid_text_month,
            "minute",
        ]
    )

    parsed_date.loc[valid_text_month] = pd.to_datetime(
        text_month_datetime,
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )

    # ========================================================
    # FORMAT 2: Diperbarui 20/05/2026, 10:37
    # ========================================================

    pattern_numeric_month = (
        r"(?P<day>\d{1,2})/"
        r"(?P<month>\d{1,2})/"
        r"(?P<year>\d{4})\s*,\s*"
        r"(?P<hour>\d{1,2})\s*:\s*"
        r"(?P<minute>\d{2})"
    )

    extracted_numeric_month = cleaned_date.str.extract(
        pattern_numeric_month
    )

    valid_numeric_month = (
        parsed_date.isna()
        & extracted_numeric_month.notna().all(axis=1)
    )

    numeric_month_datetime = (
        extracted_numeric_month.loc[
            valid_numeric_month,
            "year",
        ]
        + "-"
        + extracted_numeric_month.loc[
            valid_numeric_month,
            "month",
        ].str.zfill(2)
        + "-"
        + extracted_numeric_month.loc[
            valid_numeric_month,
            "day",
        ].str.zfill(2)
        + " "
        + extracted_numeric_month.loc[
            valid_numeric_month,
            "hour",
        ].str.zfill(2)
        + ":"
        + extracted_numeric_month.loc[
            valid_numeric_month,
            "minute",
        ]
    )

    parsed_date.loc[valid_numeric_month] = pd.to_datetime(
        numeric_month_datetime,
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )

    return parsed_date

# ============================================================
# FUNGSI LAPORAN VALIDASI
# ============================================================

def create_validation_record(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> dict:
    """
    Membuat ringkasan kondisi setiap dataset Kompas.
    """

    return {
        "dataset": dataset_name,
        "jumlah_baris": len(dataframe),
        "jumlah_kolom": dataframe.shape[1],
        "missing_title": int(
            dataframe["title"].isna().sum()
        ),
        "missing_description": int(
            dataframe["description"].isna().sum()
        ),
        "missing_content": int(
            dataframe["content"].isna().sum()
        ),
        "missing_date": int(
            dataframe["date"].isna().sum()
        ),
        "missing_category": int(
            dataframe["category"].isna().sum()
        ),
        "missing_link": int(
            dataframe["link"].isna().sum()
        ),
        "duplikat_baris": int(
            dataframe.duplicated().sum()
        ),
        "duplikat_title": int(
            dataframe.duplicated(
                subset=["title"]
            ).sum()
        ),
        "duplikat_link": int(
            dataframe.duplicated(
                subset=["link"]
            ).sum()
        ),
    }


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main() -> None:
    """
    Menjalankan validasi dan penggabungan dataset Kompas.
    """

    print("=" * 72)
    print("PERSIAPAN DATASET KOMPAS")
    print("=" * 72)

    dataframes: list[pd.DataFrame] = []
    validation_records: list[dict] = []

    # ========================================================
    # MEMBACA SETIAP DATASET KATEGORI
    # ========================================================

    for category in KOMPAS_EXPECTED_CATEGORIES:
        file_path = KOMPAS_RAW_FILES[category]

        print(f"\nMembaca kategori : {category}")
        print(f"Lokasi file      : {file_path}")

        dataframe = read_kompas_file(file_path)
        dataframe = normalize_column_names(dataframe)

        validate_columns(
            dataframe=dataframe,
            dataset_name=category,
        )

        # Hanya mengambil kolom yang digunakan dalam penelitian
        dataframe = dataframe[
            KOMPAS_EXPECTED_COLUMNS
        ].copy()

        dataframe = normalize_string_values(dataframe)

        validate_category(
            dataframe=dataframe,
            expected_category=category,
        )

        validation_record = create_validation_record(
            dataframe=dataframe,
            dataset_name=category,
        )

        validation_records.append(validation_record)
        dataframes.append(dataframe)

        print(f"Jumlah baris      : {len(dataframe):,}")
        print(
            "Total missing     : "
            f"{int(dataframe.isna().sum().sum()):,}"
        )
        print(
            "Duplikat title    : "
            f"{int(dataframe.duplicated(subset=['title']).sum()):,}"
        )
        print(
            "Duplikat link     : "
            f"{int(dataframe.duplicated(subset=['link']).sum()):,}"
        )

    # ========================================================
    # MENGGABUNGKAN SELURUH DATASET
    # ========================================================

    kompas = pd.concat(
        dataframes,
        ignore_index=True,
    )

    # Menyimpan tanggal asli untuk keperluan validasi
    kompas["date_original"] = kompas["date"].copy()

    # Mengonversi tanggal Kompas menjadi datetime
    kompas["date"] = parse_kompas_date(
        kompas["date"]
    )

    invalid_dates = int(
        kompas["date"].isna().sum()
    )

    print(
        f"\nTanggal gagal dikonversi : "
        f"{invalid_dates:,}"
    )

    if invalid_dates > 0:
        print("\nContoh tanggal yang gagal dikonversi:")

        print(
            kompas.loc[
                kompas["date"].isna(),
                "date_original",
            ]
            .head(10)
            .to_string(index=False)
        )

    # Menghapus kolom sementara
    kompas = kompas.drop(
        columns=["date_original"]
    )

    # ========================================================
    # MENGECEK DUPLIKAT SETELAH DIGABUNG
    # ========================================================

    duplicate_rows = int(
        kompas.duplicated().sum()
    )

    duplicate_titles = int(
        kompas.duplicated(
            subset=["title"]
        ).sum()
    )

    duplicate_links = int(
        kompas.duplicated(
            subset=["link"]
        ).sum()
    )

    # ========================================================
    # MENGACAK URUTAN DATA
    # ========================================================

    kompas = (
        kompas
        .sample(
            frac=1,
            random_state=RANDOM_SEED,
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # MENAMBAHKAN ID UNIK
    # ========================================================

    kompas.insert(
        loc=0,
        column="document_id",
        value=[
            f"KMP-{index:05d}"
            for index in range(
                1,
                len(kompas) + 1,
            )
        ],
    )

    # ========================================================
    # MENYIMPAN DATASET PROCESSED
    # ========================================================

    kompas.to_csv(
        KOMPAS_PROCESSED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # MENYIMPAN LAPORAN VALIDASI
    # ========================================================

    total_record = {
        "dataset": "total_kompas",
        "jumlah_baris": len(kompas),
        "jumlah_kolom": len(
            KOMPAS_EXPECTED_COLUMNS
        ),
        "missing_title": int(
            kompas["title"].isna().sum()
        ),
        "missing_description": int(
            kompas["description"].isna().sum()
        ),
        "missing_content": int(
            kompas["content"].isna().sum()
        ),
        "missing_date": int(
            kompas["date"].isna().sum()
        ),
        "missing_category": int(
            kompas["category"].isna().sum()
        ),
        "missing_link": int(
            kompas["link"].isna().sum()
        ),
        "duplikat_baris": duplicate_rows,
        "duplikat_title": duplicate_titles,
        "duplikat_link": duplicate_links,
    }

    validation_records.append(total_record)

    validation_report = pd.DataFrame(
        validation_records
    )

    validation_report.to_csv(
        KOMPAS_VALIDATION_REPORT,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # MENAMPILKAN HASIL
    # ========================================================

    print("\n" + "=" * 72)
    print("HASIL PERSIAPAN DATASET KOMPAS")
    print("=" * 72)

    print(f"Jumlah data       : {len(kompas):,}")
    print(
        f"Jumlah kategori   : "
        f"{kompas['category'].nunique()}"
    )
    print(
        f"Missing tanggal   : "
        f"{int(kompas['date'].isna().sum()):,}"
    )
    print(f"Duplikat baris    : {duplicate_rows:,}")
    print(f"Duplikat title    : {duplicate_titles:,}")
    print(f"Duplikat link     : {duplicate_links:,}")

    print("\nDistribusi kategori:")
    print(
        kompas["category"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nPreview data:")
    print(
        kompas[
            [
                "document_id",
                "title",
                "category",
                "date",
            ]
        ]
        .head()
        .to_string(index=False)
    )

    print("\nDataset processed tersimpan di:")
    print(KOMPAS_PROCESSED_PATH)

    print("\nLaporan validasi tersimpan di:")
    print(KOMPAS_VALIDATION_REPORT)


if __name__ == "__main__":
    main()
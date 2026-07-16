from __future__ import annotations

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
    AG_NEWS_LABEL_MAPPING,
    AG_NEWS_TEST_PROCESSED_PATH,
    AG_NEWS_TEST_RAW,
    AG_NEWS_TRAIN_PROCESSED_PATH,
    AG_NEWS_TRAIN_RAW,
    AG_NEWS_VALIDATION_REPORT,
)


# ============================================================
# VALIDASI FILE
# ============================================================

def validate_file(file_path: Path) -> None:
    """
    Memastikan file dataset tersedia.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan:\n{file_path}"
        )


# ============================================================
# NORMALISASI NAMA KOLOM
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
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    return dataframe


# ============================================================
# NORMALISASI NILAI TEKS
# ============================================================

def normalize_text(
    series: pd.Series,
) -> pd.Series:
    """
    Merapikan spasi teknis tanpa melakukan preprocessing NLP.
    """

    return (
        series
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


# ============================================================
# MEMBACA DATASET TRAIN
# ============================================================

def read_ag_news_train(
    file_path: Path,
) -> pd.DataFrame:
    """
    Membaca dataset train AG News.

    File train menggunakan delimiter koma.
    """

    validate_file(file_path)

    dataframe = pd.read_csv(
        file_path,
        sep=",",
        encoding="utf-8-sig",
    )

    return dataframe


# ============================================================
# MEMBACA DATASET TEST
# ============================================================

def read_ag_news_test(
    file_path: Path,
) -> pd.DataFrame:
    """
    Membaca dataset test AG News.

    File test menggunakan delimiter titik koma.
    Beberapa versi file memiliki kolom kosong tambahan.
    Hanya tiga kolom utama yang digunakan:
    Class Index, Title, dan Description.
    """

    validate_file(file_path)

    dataframe = pd.read_csv(
        file_path,
        sep=";",
        encoding="utf-8-sig",
        dtype="string",
    )

    dataframe = normalize_column_names(dataframe)

    required_columns = [
        "class_index",
        "title",
        "description",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Kolom utama pada dataset test AG News tidak lengkap.\n"
            f"Kolom yang tidak ditemukan: {missing_columns}\n"
            f"Kolom yang tersedia: {list(dataframe.columns)}"
        )

    dataframe = dataframe[
        required_columns
    ].copy()

    return dataframe


# ============================================================
# MENYERAGAMKAN STRUKTUR DATASET
# ============================================================

def prepare_ag_news(
    dataframe: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """
    Menyeragamkan struktur dataset train dan test AG News.
    """

    dataframe = normalize_column_names(dataframe)

    required_columns = [
        "class_index",
        "title",
        "description",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset {split_name} tidak memiliki kolom: "
            f"{missing_columns}"
        )

    dataframe = dataframe[
        required_columns
    ].copy()

    dataframe["class_index"] = pd.to_numeric(
        dataframe["class_index"],
        errors="coerce",
    ).astype("Int64")

    dataframe["title"] = normalize_text(
        dataframe["title"]
    )

    dataframe["description"] = normalize_text(
        dataframe["description"].fillna("")
    )

    dataframe["category"] = (
        dataframe["class_index"]
        .map(AG_NEWS_LABEL_MAPPING)
        .astype("string")
    )

    invalid_class_index = int(
        dataframe["class_index"].isna().sum()
    )

    invalid_category = int(
        dataframe["category"].isna().sum()
    )

    if invalid_class_index > 0:
        raise ValueError(
            f"Ditemukan {invalid_class_index} Class Index "
            f"yang tidak valid pada dataset {split_name}."
        )

    if invalid_category > 0:
        raise ValueError(
            f"Ditemukan {invalid_category} kategori "
            f"yang tidak valid pada dataset {split_name}."
        )

    dataframe.insert(
        loc=0,
        column="document_id",
        value=[
            f"AGN-{split_name.upper()}-{index:06d}"
            for index in range(
                1,
                len(dataframe) + 1,
            )
        ],
    )

    dataframe["split"] = split_name

    return dataframe


# ============================================================
# MEMBUAT LAPORAN VALIDASI
# ============================================================

def create_validation_record(
    dataframe: pd.DataFrame,
    split_name: str,
) -> dict:
    """
    Membuat ringkasan kondisi dataset AG News.
    """

    description_empty = int(
        dataframe["description"]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    return {
        "dataset": f"ag_news_{split_name}",
        "jumlah_baris": len(dataframe),
        "jumlah_kolom": dataframe.shape[1],
        "jumlah_kategori": int(
            dataframe["category"].nunique()
        ),
        "missing_class_index": int(
            dataframe["class_index"].isna().sum()
        ),
        "missing_title": int(
            dataframe["title"].isna().sum()
        ),
        "description_kosong": description_empty,
        "missing_category": int(
            dataframe["category"].isna().sum()
        ),
        "duplikat_baris": int(
            dataframe[
                [
                    "class_index",
                    "title",
                    "description",
                ]
            ]
            .duplicated()
            .sum()
        ),
        "duplikat_title": int(
            dataframe
            .duplicated(subset=["title"])
            .sum()
        ),
    }


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main() -> None:
    """
    Menjalankan persiapan dataset AG News.
    """

    print("=" * 72)
    print("PERSIAPAN DATASET AG NEWS")
    print("=" * 72)

    print("\nMembaca dataset train:")
    print(AG_NEWS_TRAIN_RAW)

    train_raw = read_ag_news_train(
        AG_NEWS_TRAIN_RAW
    )

    print("\nMembaca dataset test:")
    print(AG_NEWS_TEST_RAW)

    test_raw = read_ag_news_test(
        AG_NEWS_TEST_RAW
    )

    train = prepare_ag_news(
        dataframe=train_raw,
        split_name="train",
    )

    test = prepare_ag_news(
        dataframe=test_raw,
        split_name="test",
    )

    train.to_csv(
        AG_NEWS_TRAIN_PROCESSED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    test.to_csv(
        AG_NEWS_TEST_PROCESSED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    validation_report = pd.DataFrame(
        [
            create_validation_record(
                train,
                "train",
            ),
            create_validation_record(
                test,
                "test",
            ),
        ]
    )

    validation_report.to_csv(
        AG_NEWS_VALIDATION_REPORT,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 72)
    print("HASIL PERSIAPAN DATASET AG NEWS")
    print("=" * 72)

    print("\nDataset train:")
    print(f"Jumlah data       : {len(train):,}")
    print(
        f"Jumlah kategori   : "
        f"{train['category'].nunique()}"
    )
    print(
        "Description kosong: "
        f"{int(train['description'].fillna('').str.strip().eq('').sum()):,}"
    )
    print(
        "Duplikat baris    : "
        f"{int(train[['class_index', 'title', 'description']].duplicated().sum()):,}"
    )
    print(
        "Duplikat title    : "
        f"{int(train.duplicated(subset=['title']).sum()):,}"
    )

    print("\nDistribusi kategori train:")
    print(
        train["category"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nDataset test:")
    print(f"Jumlah data       : {len(test):,}")
    print(
        f"Jumlah kategori   : "
        f"{test['category'].nunique()}"
    )
    print(
        "Description kosong: "
        f"{int(test['description'].fillna('').str.strip().eq('').sum()):,}"
    )
    print(
        "Duplikat baris    : "
        f"{int(test[['class_index', 'title', 'description']].duplicated().sum()):,}"
    )
    print(
        "Duplikat title    : "
        f"{int(test.duplicated(subset=['title']).sum()):,}"
    )

    print("\nDistribusi kategori test:")
    print(
        test["category"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nPreview train:")
    print(
        train[
            [
                "document_id",
                "class_index",
                "category",
                "title",
            ]
        ]
        .head()
        .to_string(index=False)
    )

    print("\nPreview test:")
    print(
        test[
            [
                "document_id",
                "class_index",
                "category",
                "title",
            ]
        ]
        .head()
        .to_string(index=False)
    )

    print("\nDataset train processed tersimpan di:")
    print(AG_NEWS_TRAIN_PROCESSED_PATH)

    print("\nDataset test processed tersimpan di:")
    print(AG_NEWS_TEST_PROCESSED_PATH)

    print("\nLaporan validasi tersimpan di:")
    print(AG_NEWS_VALIDATION_REPORT)


if __name__ == "__main__":
    main()
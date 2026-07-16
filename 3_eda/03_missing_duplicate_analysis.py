from __future__ import annotations

import sys
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
    TABLES_DIR,
)


# ============================================================
# OUTPUT FILE
# ============================================================

MISSING_VALUE_REPORT_PATH = (
    TABLES_DIR / "missing_value_report.csv"
)

DUPLICATE_REPORT_PATH = (
    TABLES_DIR / "duplicate_report.csv"
)

KOMPAS_DUPLICATE_DETAIL_PATH = (
    TABLES_DIR / "kompas_duplicate_detail.csv"
)

AGNEWS_TRAIN_DUPLICATE_DETAIL_PATH = (
    TABLES_DIR / "agnews_train_duplicate_detail.csv"
)

AGNEWS_TEST_DUPLICATE_DETAIL_PATH = (
    TABLES_DIR / "agnews_test_duplicate_detail.csv"
)

AGNEWS_OVERLAP_REPORT_PATH = (
    TABLES_DIR / "agnews_train_test_overlap.csv"
)


# ============================================================
# MEMBACA DATASET
# ============================================================

def load_dataset(
    file_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Membaca dataset processed dan memastikan file tersedia.
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
# NORMALISASI TEKS UNTUK ANALISIS DUPLIKAT
# ============================================================

def normalize_for_duplicate_check(
    series: pd.Series,
) -> pd.Series:
    """
    Menormalisasi teks hanya untuk kebutuhan analisis duplikat.

    Proses:
    - mengisi nilai kosong;
    - mengubah ke string;
    - lowercase;
    - menghapus spasi berlebih.

    Data asli tidak diubah.
    """

    return (
        series
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


# ============================================================
# LAPORAN MISSING VALUE
# ============================================================

def create_missing_value_report(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Membuat laporan missing value dan string kosong per kolom.

    Missing value dan string kosong dihitung secara terpisah
    agar baris yang sama tidak dihitung dua kali.
    """

    records: list[dict] = []

    for column in dataframe.columns:
        missing_mask = dataframe[column].isna()
        missing_count = int(missing_mask.sum())

        empty_mask = pd.Series(
            False,
            index=dataframe.index,
        )

        if (
            pd.api.types.is_object_dtype(dataframe[column])
            or pd.api.types.is_string_dtype(dataframe[column])
        ):
            empty_mask = (
                dataframe[column]
                .notna()
                & dataframe[column]
                .astype(str)
                .str.strip()
                .eq("")
            )

        empty_string_count = int(
            empty_mask.sum()
        )

        problem_mask = (
            missing_mask
            | empty_mask
        )

        total_problem = int(
            problem_mask.sum()
        )

        records.append(
            {
                "dataset": dataset_name,
                "column": column,
                "missing_value": missing_count,
                "empty_string": empty_string_count,
                "total_problem": total_problem,
                "percentage_problem": round(
                    total_problem
                    / len(dataframe)
                    * 100,
                    4,
                ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# ANALISIS DUPLIKAT KOMPAS
# ============================================================

def analyze_kompas_duplicates(
    dataframe: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """
    Menganalisis duplikat dataset Kompas.
    """

    data = dataframe.copy()

    data["normalized_title"] = (
        normalize_for_duplicate_check(
            data["title"]
        )
    )

    data["normalized_content"] = (
        normalize_for_duplicate_check(
            data["content"]
        )
    )

    data["normalized_link"] = (
        normalize_for_duplicate_check(
            data["link"]
        )
    )

    exact_duplicate_count = int(
        data.duplicated(
            subset=[
                "title",
                "description",
                "content",
                "category",
                "link",
            ]
        ).sum()
    )

    normalized_title_duplicate_count = int(
        data.duplicated(
            subset=["normalized_title"]
        ).sum()
    )

    normalized_content_duplicate_count = int(
        data.duplicated(
            subset=["normalized_content"]
        ).sum()
    )

    normalized_link_duplicate_count = int(
        data.duplicated(
            subset=["normalized_link"]
        ).sum()
    )

    cross_category_title = (
        data.groupby("normalized_title")["category"]
        .nunique()
    )

    cross_category_title_count = int(
        (cross_category_title > 1).sum()
    )

    duplicate_mask = (
        data.duplicated(
            subset=["normalized_title"],
            keep=False,
        )
        | data.duplicated(
            subset=["normalized_content"],
            keep=False,
        )
        | data.duplicated(
            subset=["normalized_link"],
            keep=False,
        )
    )

    duplicate_detail = data.loc[
        duplicate_mask,
        [
            "document_id",
            "title",
            "category",
            "link",
            "normalized_title",
        ],
    ].copy()

    duplicate_detail = duplicate_detail.sort_values(
        by=[
            "normalized_title",
            "category",
        ]
    )

    report = {
        "dataset": "kompas",
        "jumlah_data": len(data),
        "duplikat_artikel_exact": exact_duplicate_count,
        "duplikat_title_normalized": (
            normalized_title_duplicate_count
        ),
        "duplikat_content_normalized": (
            normalized_content_duplicate_count
        ),
        "duplikat_link_normalized": (
            normalized_link_duplicate_count
        ),
        "judul_muncul_lintas_kategori": (
            cross_category_title_count
        ),
    }

    return report, duplicate_detail


# ============================================================
# ANALISIS DUPLIKAT AG NEWS
# ============================================================

def analyze_agnews_duplicates(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> tuple[dict, pd.DataFrame]:
    """
    Menganalisis duplikat AG News train atau test.
    """

    data = dataframe.copy()

    data["normalized_title"] = (
        normalize_for_duplicate_check(
            data["title"]
        )
    )

    data["normalized_description"] = (
        normalize_for_duplicate_check(
            data["description"]
        )
    )

    exact_duplicate_count = int(
        data.duplicated(
            subset=[
                "class_index",
                "title",
                "description",
            ]
        ).sum()
    )

    normalized_article_duplicate_count = int(
        data.duplicated(
            subset=[
                "class_index",
                "normalized_title",
                "normalized_description",
            ]
        ).sum()
    )

    normalized_title_duplicate_count = int(
        data.duplicated(
            subset=["normalized_title"]
        ).sum()
    )

    cross_category_title = (
        data.groupby("normalized_title")["category"]
        .nunique()
    )

    cross_category_title_count = int(
        (cross_category_title > 1).sum()
    )

    duplicate_mask = data.duplicated(
        subset=[
            "class_index",
            "normalized_title",
            "normalized_description",
        ],
        keep=False,
    )

    duplicate_detail = data.loc[
        duplicate_mask,
        [
            "document_id",
            "class_index",
            "category",
            "title",
            "description",
            "normalized_title",
        ],
    ].copy()

    duplicate_detail = duplicate_detail.sort_values(
        by=[
            "normalized_title",
            "category",
        ]
    )

    report = {
        "dataset": dataset_name,
        "jumlah_data": len(data),
        "duplikat_artikel_exact": exact_duplicate_count,
        "duplikat_artikel_normalized": (
            normalized_article_duplicate_count
        ),
        "duplikat_title_normalized": (
            normalized_title_duplicate_count
        ),
        "judul_muncul_lintas_kategori": (
            cross_category_title_count
        ),
    }

    return report, duplicate_detail


# ============================================================
# ANALISIS OVERLAP TRAIN DAN TEST AG NEWS
# ============================================================

def analyze_agnews_train_test_overlap(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mengecek artikel yang sama antara AG News train dan test.

    Overlap dicek berdasarkan:
    - title yang dinormalisasi;
    - description yang dinormalisasi;
    - label kategori.
    """

    train_data = train.copy()
    test_data = test.copy()

    train_data["normalized_title"] = (
        normalize_for_duplicate_check(
            train_data["title"]
        )
    )

    train_data["normalized_description"] = (
        normalize_for_duplicate_check(
            train_data["description"]
        )
    )

    test_data["normalized_title"] = (
        normalize_for_duplicate_check(
            test_data["title"]
        )
    )

    test_data["normalized_description"] = (
        normalize_for_duplicate_check(
            test_data["description"]
        )
    )

    train_keys = train_data[
        [
            "class_index",
            "normalized_title",
            "normalized_description",
        ]
    ].drop_duplicates()

    overlap = test_data.merge(
        train_keys,
        on=[
            "class_index",
            "normalized_title",
            "normalized_description",
        ],
        how="inner",
    )

    result_columns = [
        "document_id",
        "class_index",
        "category",
        "title",
        "description",
    ]

    overlap = overlap[
        result_columns
    ].copy()

    return overlap


# ============================================================
# MENAMPILKAN HASIL
# ============================================================

def print_missing_summary(
    report: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Menampilkan ringkasan missing value.
    """

    print("\n" + "=" * 72)
    print(f"MISSING VALUE - {dataset_name.upper()}")
    print("=" * 72)

    problem_rows = report[
        report["total_problem"] > 0
    ]

    if problem_rows.empty:
        print("Tidak ditemukan missing value atau teks kosong.")
    else:
        print(
            problem_rows[
                [
                    "column",
                    "missing_value",
                    "empty_string",
                    "total_problem",
                    "percentage_problem",
                ]
            ].to_string(index=False)
        )


def print_duplicate_summary(
    report: dict,
) -> None:
    """
    Menampilkan ringkasan duplikat.
    """

    print("\n" + "=" * 72)
    print(f"DUPLIKAT - {report['dataset'].upper()}")
    print("=" * 72)

    for key, value in report.items():
        if key == "dataset":
            continue

        label = key.replace("_", " ").title()

        if isinstance(value, int):
            print(f"{label:<36}: {value:,}")
        else:
            print(f"{label:<36}: {value}")


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main() -> None:
    """
    Menjalankan analisis missing value, duplikat,
    dan overlap train-test.
    """

    print("=" * 72)
    print("STEP 3.3 - MISSING VALUE AND DUPLICATE ANALYSIS")
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

    # ========================================================
    # ANALISIS MISSING VALUE
    # ========================================================

    missing_kompas = create_missing_value_report(
        kompas,
        "kompas",
    )

    missing_agnews_train = create_missing_value_report(
        agnews_train,
        "ag_news_train",
    )

    missing_agnews_test = create_missing_value_report(
        agnews_test,
        "ag_news_test",
    )

    missing_report = pd.concat(
        [
            missing_kompas,
            missing_agnews_train,
            missing_agnews_test,
        ],
        ignore_index=True,
    )

    missing_report.to_csv(
        MISSING_VALUE_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print_missing_summary(
        missing_kompas,
        "Kompas",
    )

    print_missing_summary(
        missing_agnews_train,
        "AG News Train",
    )

    print_missing_summary(
        missing_agnews_test,
        "AG News Test",
    )

    # ========================================================
    # ANALISIS DUPLIKAT
    # ========================================================

    kompas_report, kompas_duplicate_detail = (
        analyze_kompas_duplicates(
            kompas
        )
    )

    agnews_train_report, agnews_train_duplicate_detail = (
        analyze_agnews_duplicates(
            agnews_train,
            "ag_news_train",
        )
    )

    agnews_test_report, agnews_test_duplicate_detail = (
        analyze_agnews_duplicates(
            agnews_test,
            "ag_news_test",
        )
    )

    duplicate_report = pd.DataFrame(
        [
            kompas_report,
            agnews_train_report,
            agnews_test_report,
        ]
    )

    duplicate_report.to_csv(
        DUPLICATE_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    kompas_duplicate_detail.to_csv(
        KOMPAS_DUPLICATE_DETAIL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    agnews_train_duplicate_detail.to_csv(
        AGNEWS_TRAIN_DUPLICATE_DETAIL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    agnews_test_duplicate_detail.to_csv(
        AGNEWS_TEST_DUPLICATE_DETAIL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print_duplicate_summary(
        kompas_report
    )

    print_duplicate_summary(
        agnews_train_report
    )

    print_duplicate_summary(
        agnews_test_report
    )

    # ========================================================
    # ANALISIS OVERLAP TRAIN DAN TEST
    # ========================================================

    overlap = analyze_agnews_train_test_overlap(
        agnews_train,
        agnews_test,
    )

    overlap.to_csv(
        AGNEWS_OVERLAP_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 72)
    print("AG NEWS TRAIN-TEST OVERLAP")
    print("=" * 72)

    print(
        "Jumlah artikel test yang identik dengan train: "
        f"{len(overlap):,}"
    )

    if not overlap.empty:
        print("\nContoh overlap:")
        print(
            overlap[
                [
                    "document_id",
                    "category",
                    "title",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )
    else:
        print(
            "Tidak ditemukan artikel identik antara train dan test."
        )

    # ========================================================
    # INFORMASI OUTPUT
    # ========================================================

    print("\n" + "=" * 72)
    print("OUTPUT MISSING AND DUPLICATE ANALYSIS")
    print("=" * 72)

    print("\nLaporan missing value:")
    print(MISSING_VALUE_REPORT_PATH)

    print("\nRingkasan duplikat:")
    print(DUPLICATE_REPORT_PATH)

    print("\nDetail duplikat Kompas:")
    print(KOMPAS_DUPLICATE_DETAIL_PATH)

    print("\nDetail duplikat AG News train:")
    print(AGNEWS_TRAIN_DUPLICATE_DETAIL_PATH)

    print("\nDetail duplikat AG News test:")
    print(AGNEWS_TEST_DUPLICATE_DETAIL_PATH)

    print("\nOverlap AG News train-test:")
    print(AGNEWS_OVERLAP_REPORT_PATH)

    print("\nTahap missing dan duplicate analysis selesai.")


if __name__ == "__main__":
    main()
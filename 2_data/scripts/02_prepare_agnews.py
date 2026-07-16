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
# PATH LAPORAN AUDIT
# ============================================================

AG_NEWS_DUPLICATE_REPORT = (
    AG_NEWS_VALIDATION_REPORT.parent
    / "ag_news_duplicate_report.csv"
)

AG_NEWS_LABEL_CONFLICT_REPORT = (
    AG_NEWS_VALIDATION_REPORT.parent
    / "ag_news_label_conflicts.csv"
)

AG_NEWS_TRAIN_TEST_OVERLAP_REPORT = (
    AG_NEWS_VALIDATION_REPORT.parent
    / "ag_news_train_test_overlap.csv"
)


# ============================================================
# KONSTANTA DATASET AG NEWS
# ============================================================

EXPECTED_ROW_COUNTS = {
    "train": 120_000,
    "test": 7_600,
}

EXPECTED_CLASS_COUNTS = {
    "train": {
        1: 30_000,
        2: 30_000,
        3: 30_000,
        4: 30_000,
    },
    "test": {
        1: 1_900,
        2: 1_900,
        3: 1_900,
        4: 1_900,
    },
}

REQUIRED_COLUMNS = [
    "class_index",
    "title",
    "description",
]

TEXT_KEY_COLUMNS = [
    "_title_key",
    "_description_key",
]

PROCESSED_OUTPUT_COLUMNS = [
    "document_id",
    "source_row",
    "class_index",
    "category",
    "title",
    "description",
    "split",
]


# ============================================================
# VALIDASI FILE
# ============================================================

def validate_file(file_path: Path) -> None:
    """
    Memastikan file dataset tersedia dan merupakan file.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan:\n{file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Path berikut bukan file:\n{file_path}"
        )


# ============================================================
# NORMALISASI NAMA KOLOM
# ============================================================

def normalize_column_names(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Menyeragamkan nama kolom menjadi lowercase
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
# NORMALISASI WHITESPACE TEKNIS
# ============================================================

def normalize_whitespace(
    series: pd.Series,
) -> pd.Series:
    """
    Merapikan whitespace teknis tanpa melakukan
    preprocessing NLP.

    Operasi yang dilakukan:
    1. mengubah nilai menjadi tipe string;
    2. menghapus whitespace awal dan akhir;
    3. mengubah rangkaian whitespace menjadi satu spasi.
    """

    return (
        series
        .astype("string")
        .str.strip()
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
    )


# ============================================================
# MEMBACA DATASET AG NEWS
# ============================================================

def read_ag_news(
    file_path: Path,
    split_name: str,
) -> pd.DataFrame:
    """
    Membaca dataset AG News Kaggle.

    Dataset train dan test menggunakan delimiter koma.
    """

    validate_file(file_path)

    try:
        dataframe = pd.read_csv(
            file_path,
            sep=",",
            encoding="utf-8-sig",
            dtype="string",
        )

    except UnicodeDecodeError:
        dataframe = pd.read_csv(
            file_path,
            sep=",",
            encoding="latin-1",
            dtype="string",
        )

    except pd.errors.ParserError as error:
        raise ValueError(
            f"Gagal membaca dataset AG News {split_name}.\n"
            f"Periksa delimiter dan struktur CSV.\n"
            f"Detail error: {error}"
        ) from error

    if dataframe.empty:
        raise ValueError(
            f"Dataset AG News {split_name} kosong."
        )

    return dataframe


# ============================================================
# MENYIAPKAN STRUKTUR DATASET
# ============================================================

def prepare_ag_news(
    dataframe: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """
    Menyeragamkan struktur dataset AG News tanpa
    menghapus satu pun baris data.
    """

    dataframe = normalize_column_names(
        dataframe
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset {split_name} tidak memiliki "
            f"kolom wajib: {missing_columns}\n"
            f"Kolom tersedia: {list(dataframe.columns)}"
        )

    dataframe = dataframe[
        REQUIRED_COLUMNS
    ].copy()

    # Menyimpan nomor baris asli pada file raw.
    dataframe.insert(
        loc=0,
        column="source_row",
        value=range(
            1,
            len(dataframe) + 1,
        ),
    )

    dataframe["class_index"] = pd.to_numeric(
        dataframe["class_index"],
        errors="coerce",
    ).astype("Int64")

    dataframe["title"] = normalize_whitespace(
        dataframe["title"]
    )

    dataframe["description"] = (
        normalize_whitespace(
            dataframe["description"].fillna("")
        )
    )

    invalid_class_index = int(
        dataframe["class_index"].isna().sum()
    )

    if invalid_class_index > 0:
        raise ValueError(
            f"Ditemukan {invalid_class_index:,} "
            f"Class Index tidak valid pada "
            f"dataset {split_name}."
        )

    dataframe["category"] = (
        dataframe["class_index"]
        .map(AG_NEWS_LABEL_MAPPING)
        .astype("string")
    )

    invalid_category = int(
        dataframe["category"].isna().sum()
    )

    if invalid_category > 0:
        invalid_labels = (
            dataframe.loc[
                dataframe["category"].isna(),
                "class_index",
            ]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Ditemukan {invalid_category:,} label "
            f"tidak dikenal pada dataset {split_name}.\n"
            f"Label tidak dikenal: {invalid_labels}"
        )

    missing_title = int(
        dataframe["title"].isna().sum()
    )

    empty_title = int(
        dataframe["title"]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    if missing_title > 0 or empty_title > 0:
        raise ValueError(
            f"Dataset {split_name} memiliki title "
            f"yang tidak valid.\n"
            f"Missing title: {missing_title:,}\n"
            f"Title kosong : {empty_title:,}"
        )

    dataframe["split"] = split_name

    return dataframe


# ============================================================
# VALIDASI DATASET RAW
# ============================================================

def validate_expected_dataset(
    dataframe: pd.DataFrame,
    split_name: str,
) -> None:
    """
    Memastikan dataset sesuai dengan distribusi resmi
    AG News yang digunakan dalam penelitian.
    """

    expected_rows = EXPECTED_ROW_COUNTS[
        split_name
    ]

    actual_rows = len(dataframe)

    if actual_rows != expected_rows:
        raise ValueError(
            f"Jumlah data AG News {split_name} "
            f"tidak sesuai.\n"
            f"Seharusnya : {expected_rows:,}\n"
            f"Ditemukan  : {actual_rows:,}"
        )

    actual_distribution = (
        dataframe["class_index"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    expected_distribution = (
        EXPECTED_CLASS_COUNTS[
            split_name
        ]
    )

    if actual_distribution != expected_distribution:
        raise ValueError(
            f"Distribusi kelas AG News {split_name} "
            f"tidak sesuai.\n"
            f"Seharusnya : {expected_distribution}\n"
            f"Ditemukan  : {actual_distribution}"
        )


# ============================================================
# MEMBUAT DOCUMENT ID
# ============================================================

def add_document_id(
    dataframe: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """
    Menambahkan identitas unik untuk setiap baris.

    Document ID tidak mengubah urutan maupun
    jumlah data asli.
    """

    dataframe = dataframe.copy()
    dataframe = dataframe.reset_index(
        drop=True
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

    return dataframe


# ============================================================
# MEMBUAT KUNCI TEKS UNTUK AUDIT
# ============================================================

def add_text_keys(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Membuat kunci teks ternormalisasi untuk audit.

    Lowercase hanya digunakan sebagai kunci audit.
    Isi title dan description pada dataset output
    tetap mempertahankan kapitalisasi aslinya.
    """

    dataframe = dataframe.copy()

    dataframe["_title_key"] = (
        dataframe["title"]
        .fillna("")
        .str.lower()
        .str.strip()
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
    )

    dataframe["_description_key"] = (
        dataframe["description"]
        .fillna("")
        .str.lower()
        .str.strip()
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
    )

    return dataframe


# ============================================================
# AUDIT DUPLIKASI
# ============================================================

def find_normalized_duplicates(
    dataframe: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """
    Menemukan dokumen yang memiliki:
    - Class Index sama;
    - title sama setelah normalisasi;
    - description sama setelah normalisasi.

    Fungsi ini hanya membuat laporan dan tidak
    menghapus baris.
    """

    audit_data = add_text_keys(
        dataframe
    )

    duplicate_subset = [
        "class_index",
        "_title_key",
        "_description_key",
    ]

    duplicate_mask = (
        audit_data
        .duplicated(
            subset=duplicate_subset,
            keep=False,
        )
    )

    duplicate_data = audit_data.loc[
        duplicate_mask
    ].copy()

    if duplicate_data.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "duplicate_group",
                "group_size",
                "document_id",
                "source_row",
                "class_index",
                "category",
                "title",
                "description",
            ]
        )

    duplicate_data["group_size"] = (
        duplicate_data
        .groupby(
            duplicate_subset,
            dropna=False,
        )["source_row"]
        .transform("size")
    )

    duplicate_data["duplicate_group"] = (
        duplicate_data
        .groupby(
            duplicate_subset,
            dropna=False,
        )
        .ngroup()
        + 1
    )

    duplicate_data["split"] = split_name

    duplicate_data = duplicate_data[
        [
            "split",
            "duplicate_group",
            "group_size",
            "document_id",
            "source_row",
            "class_index",
            "category",
            "title",
            "description",
        ]
    ].sort_values(
        by=[
            "duplicate_group",
            "source_row",
        ]
    )

    return duplicate_data.reset_index(
        drop=True
    )


# ============================================================
# AUDIT KONFLIK LABEL
# ============================================================

def find_label_conflicts(
    dataframe: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """
    Menemukan title dan description yang sama,
    tetapi memiliki Class Index berbeda.

    Fungsi ini tidak menghapus atau mengubah label.
    """

    audit_data = add_text_keys(
        dataframe
    )

    label_counts = (
        audit_data
        .groupby(
            TEXT_KEY_COLUMNS,
            dropna=False,
        )["class_index"]
        .transform("nunique")
    )

    conflict_data = audit_data.loc[
        label_counts > 1
    ].copy()

    if conflict_data.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "conflict_group",
                "group_size",
                "jumlah_label",
                "document_id",
                "source_row",
                "class_index",
                "category",
                "title",
                "description",
            ]
        )

    conflict_data["group_size"] = (
        conflict_data
        .groupby(
            TEXT_KEY_COLUMNS,
            dropna=False,
        )["source_row"]
        .transform("size")
    )

    conflict_data["jumlah_label"] = (
        conflict_data
        .groupby(
            TEXT_KEY_COLUMNS,
            dropna=False,
        )["class_index"]
        .transform("nunique")
    )

    conflict_data["conflict_group"] = (
        conflict_data
        .groupby(
            TEXT_KEY_COLUMNS,
            dropna=False,
        )
        .ngroup()
        + 1
    )

    conflict_data["split"] = split_name

    conflict_data = conflict_data[
        [
            "split",
            "conflict_group",
            "group_size",
            "jumlah_label",
            "document_id",
            "source_row",
            "class_index",
            "category",
            "title",
            "description",
        ]
    ].sort_values(
        by=[
            "conflict_group",
            "class_index",
            "source_row",
        ]
    )

    return conflict_data.reset_index(
        drop=True
    )


# ============================================================
# AUDIT OVERLAP TRAIN DAN TEST
# ============================================================

def find_train_test_overlap(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Menemukan dokumen yang terdapat pada train dan test
    berdasarkan title dan description ternormalisasi.

    Fungsi ini hanya membuat laporan dan tidak
    menghapus data train maupun test.
    """

    train_audit = add_text_keys(
        train
    )

    test_audit = add_text_keys(
        test
    )

    train_columns = train_audit[
        [
            "document_id",
            "source_row",
            "class_index",
            "category",
            "title",
            "description",
            "_title_key",
            "_description_key",
        ]
    ].rename(
        columns={
            "document_id": "train_document_id",
            "source_row": "train_source_row",
            "class_index": "train_class_index",
            "category": "train_category",
            "title": "train_title",
            "description": "train_description",
        }
    )

    test_columns = test_audit[
        [
            "document_id",
            "source_row",
            "class_index",
            "category",
            "title",
            "description",
            "_title_key",
            "_description_key",
        ]
    ].rename(
        columns={
            "document_id": "test_document_id",
            "source_row": "test_source_row",
            "class_index": "test_class_index",
            "category": "test_category",
            "title": "test_title",
            "description": "test_description",
        }
    )

    overlap = train_columns.merge(
        test_columns,
        on=TEXT_KEY_COLUMNS,
        how="inner",
    )

    if overlap.empty:
        return pd.DataFrame(
            columns=[
                "train_document_id",
                "train_source_row",
                "train_class_index",
                "train_category",
                "test_document_id",
                "test_source_row",
                "test_class_index",
                "test_category",
                "title",
                "description",
                "label_consistent",
            ]
        )

    overlap["title"] = (
        overlap["train_title"]
    )

    overlap["description"] = (
        overlap["train_description"]
    )

    overlap["label_consistent"] = (
        overlap["train_class_index"]
        == overlap["test_class_index"]
    )

    overlap = overlap[
        [
            "train_document_id",
            "train_source_row",
            "train_class_index",
            "train_category",
            "test_document_id",
            "test_source_row",
            "test_class_index",
            "test_category",
            "title",
            "description",
            "label_consistent",
        ]
    ].sort_values(
        by=[
            "train_source_row",
            "test_source_row",
        ]
    )

    return overlap.reset_index(
        drop=True
    )


# ============================================================
# MENGGABUNGKAN LAPORAN AUDIT
# ============================================================

def combine_audit_frames(
    dataframes: list[pd.DataFrame],
    empty_columns: list[str],
) -> pd.DataFrame:
    """
    Menggabungkan DataFrame audit tanpa memunculkan
    FutureWarning ketika salah satu DataFrame kosong.
    """

    non_empty_frames = [
        dataframe
        for dataframe in dataframes
        if not dataframe.empty
    ]

    if not non_empty_frames:
        return pd.DataFrame(
            columns=empty_columns
        )

    return pd.concat(
        non_empty_frames,
        ignore_index=True,
    )


# ============================================================
# MEMBUAT LAPORAN VALIDASI
# ============================================================

def create_validation_record(
    dataframe: pd.DataFrame,
    split_name: str,
    duplicate_report: pd.DataFrame,
    conflict_report: pd.DataFrame,
    overlap_report: pd.DataFrame,
) -> dict:
    """
    Membuat ringkasan kondisi dataset tanpa
    melakukan penghapusan data.
    """

    audit_data = add_text_keys(
        dataframe
    )

    exact_duplicates = int(
        dataframe[
            [
                "class_index",
                "title",
                "description",
            ]
        ]
        .duplicated()
        .sum()
    )

    normalized_duplicate_excess = int(
        audit_data
        .duplicated(
            subset=[
                "class_index",
                "_title_key",
                "_description_key",
            ]
        )
        .sum()
    )

    normalized_duplicate_groups = 0

    if not duplicate_report.empty:
        normalized_duplicate_groups = int(
            duplicate_report[
                "duplicate_group"
            ].nunique()
        )

    conflict_groups = 0

    if not conflict_report.empty:
        conflict_groups = int(
            conflict_report[
                "conflict_group"
            ].nunique()
        )

    unique_overlap_keys = 0

    if not overlap_report.empty:
        unique_overlap_keys = int(
            overlap_report[
                [
                    "title",
                    "description",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        )

    return {
        "dataset": f"ag_news_{split_name}",
        "jumlah_baris": len(dataframe),
        "jumlah_kolom": dataframe.shape[1],
        "jumlah_kategori": int(
            dataframe["category"].nunique()
        ),
        "missing_class_index": int(
            dataframe["class_index"]
            .isna()
            .sum()
        ),
        "missing_title": int(
            dataframe["title"]
            .isna()
            .sum()
        ),
        "title_kosong": int(
            dataframe["title"]
            .fillna("")
            .str.strip()
            .eq("")
            .sum()
        ),
        "description_kosong": int(
            dataframe["description"]
            .fillna("")
            .str.strip()
            .eq("")
            .sum()
        ),
        "missing_category": int(
            dataframe["category"]
            .isna()
            .sum()
        ),
        "duplikat_exact_kelebihan": (
            exact_duplicates
        ),
        "duplikat_ternormalisasi_kelebihan": (
            normalized_duplicate_excess
        ),
        "kelompok_duplikat_ternormalisasi": (
            normalized_duplicate_groups
        ),
        "baris_dalam_kelompok_duplikat": (
            len(duplicate_report)
        ),
        "baris_konflik_label": (
            len(conflict_report)
        ),
        "kelompok_konflik_label": (
            conflict_groups
        ),
        "pasangan_overlap_train_test": (
            len(overlap_report)
        ),
        "dokumen_unik_overlap_train_test": (
            unique_overlap_keys
        ),
        "duplikat_title": int(
            dataframe
            .duplicated(
                subset=["title"]
            )
            .sum()
        ),
    }


# ============================================================
# MENAMPILKAN RINGKASAN DATASET
# ============================================================

def print_dataset_summary(
    dataframe: pd.DataFrame,
    split_name: str,
    duplicate_report: pd.DataFrame,
    conflict_report: pd.DataFrame,
) -> None:
    """
    Menampilkan ringkasan preparation dan audit.
    """

    audit_data = add_text_keys(
        dataframe
    )

    normalized_duplicate_excess = int(
        audit_data
        .duplicated(
            subset=[
                "class_index",
                "_title_key",
                "_description_key",
            ]
        )
        .sum()
    )

    duplicate_groups = 0

    if not duplicate_report.empty:
        duplicate_groups = int(
            duplicate_report[
                "duplicate_group"
            ].nunique()
        )

    conflict_groups = 0

    if not conflict_report.empty:
        conflict_groups = int(
            conflict_report[
                "conflict_group"
            ].nunique()
        )

    print(f"\nDataset {split_name}:")
    print(
        f"Jumlah data                  : "
        f"{len(dataframe):,}"
    )
    print(
        f"Jumlah kategori              : "
        f"{dataframe['category'].nunique()}"
    )
    print(
        f"Description kosong           : "
        f"{int(dataframe['description'].fillna('').str.strip().eq('').sum()):,}"
    )
    print(
        f"Duplikat exact kelebihan     : "
        f"{int(dataframe[['class_index', 'title', 'description']].duplicated().sum()):,}"
    )
    print(
        f"Duplikat normalisasi kelebihan: "
        f"{normalized_duplicate_excess:,}"
    )
    print(
        f"Kelompok duplikat            : "
        f"{duplicate_groups:,}"
    )
    print(
        f"Baris konflik label          : "
        f"{len(conflict_report):,}"
    )
    print(
        f"Kelompok konflik label       : "
        f"{conflict_groups:,}"
    )
    print(
        f"Duplikat title               : "
        f"{int(dataframe.duplicated(subset=['title']).sum()):,}"
    )

    print(
        f"\nDistribusi kategori {split_name}:"
    )

    print(
        dataframe["category"]
        .value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main() -> None:
    """
    Menjalankan preparation dan audit dataset AG News.

    Script ini tidak menghapus data. Seluruh proses
    cleaning dilakukan pada:
    4_preprocessing/01_data_cleaning.py
    """

    print("=" * 72)
    print("PERSIAPAN DAN AUDIT DATASET AG NEWS")
    print("=" * 72)

    # --------------------------------------------------------
    # MEMASTIKAN FOLDER OUTPUT TERSEDIA
    # --------------------------------------------------------

    AG_NEWS_TRAIN_PROCESSED_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    AG_NEWS_VALIDATION_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # MEMBACA DATASET RAW
    # --------------------------------------------------------

    print("\nMembaca dataset train:")
    print(AG_NEWS_TRAIN_RAW)

    train_raw = read_ag_news(
        file_path=AG_NEWS_TRAIN_RAW,
        split_name="train",
    )

    print("\nMembaca dataset test:")
    print(AG_NEWS_TEST_RAW)

    test_raw = read_ag_news(
        file_path=AG_NEWS_TEST_RAW,
        split_name="test",
    )

    # --------------------------------------------------------
    # PREPARATION
    # --------------------------------------------------------

    train_prepared = prepare_ag_news(
        dataframe=train_raw,
        split_name="train",
    )

    test_prepared = prepare_ag_news(
        dataframe=test_raw,
        split_name="test",
    )

    validate_expected_dataset(
        dataframe=train_prepared,
        split_name="train",
    )

    validate_expected_dataset(
        dataframe=test_prepared,
        split_name="test",
    )

    train_prepared = add_document_id(
        dataframe=train_prepared,
        split_name="train",
    )

    test_prepared = add_document_id(
        dataframe=test_prepared,
        split_name="test",
    )

    train_prepared = train_prepared[
        PROCESSED_OUTPUT_COLUMNS
    ].copy()

    test_prepared = test_prepared[
        PROCESSED_OUTPUT_COLUMNS
    ].copy()

    # --------------------------------------------------------
    # AUDIT DUPLIKASI
    # --------------------------------------------------------

    train_duplicates = (
        find_normalized_duplicates(
            dataframe=train_prepared,
            split_name="train",
        )
    )

    test_duplicates = (
        find_normalized_duplicates(
            dataframe=test_prepared,
            split_name="test",
        )
    )

    duplicate_report = combine_audit_frames(
        dataframes=[
            train_duplicates,
            test_duplicates,
        ],
        empty_columns=[
            "split",
            "duplicate_group",
            "group_size",
            "document_id",
            "source_row",
            "class_index",
            "category",
            "title",
            "description",
        ],
    )

    duplicate_report.to_csv(
        AG_NEWS_DUPLICATE_REPORT,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # AUDIT KONFLIK LABEL
    # --------------------------------------------------------

    train_conflicts = find_label_conflicts(
        dataframe=train_prepared,
        split_name="train",
    )

    test_conflicts = find_label_conflicts(
        dataframe=test_prepared,
        split_name="test",
    )

    conflict_report = combine_audit_frames(
        dataframes=[
            train_conflicts,
            test_conflicts,
        ],
        empty_columns=[
            "split",
            "conflict_group",
            "group_size",
            "jumlah_label",
            "document_id",
            "source_row",
            "class_index",
            "category",
            "title",
            "description",
        ],
    )

    conflict_report.to_csv(
        AG_NEWS_LABEL_CONFLICT_REPORT,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # AUDIT OVERLAP TRAIN DAN TEST
    # --------------------------------------------------------

    overlap_report = (
        find_train_test_overlap(
            train=train_prepared,
            test=test_prepared,
        )
    )

    overlap_report.to_csv(
        AG_NEWS_TRAIN_TEST_OVERLAP_REPORT,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # MENYIMPAN DATASET PROCESSED
    # --------------------------------------------------------

    train_prepared.to_csv(
        AG_NEWS_TRAIN_PROCESSED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    test_prepared.to_csv(
        AG_NEWS_TEST_PROCESSED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # MEMBUAT LAPORAN VALIDASI
    # --------------------------------------------------------

    validation_report = pd.DataFrame(
        [
            create_validation_record(
                dataframe=train_prepared,
                split_name="train",
                duplicate_report=train_duplicates,
                conflict_report=train_conflicts,
                overlap_report=overlap_report,
            ),
            create_validation_record(
                dataframe=test_prepared,
                split_name="test",
                duplicate_report=test_duplicates,
                conflict_report=test_conflicts,
                overlap_report=overlap_report,
            ),
        ]
    )

    validation_report.to_csv(
        AG_NEWS_VALIDATION_REPORT,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # MENAMPILKAN HASIL
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("HASIL PERSIAPAN DAN AUDIT DATASET AG NEWS")
    print("=" * 72)

    print_dataset_summary(
        dataframe=train_prepared,
        split_name="train",
        duplicate_report=train_duplicates,
        conflict_report=train_conflicts,
    )

    print_dataset_summary(
        dataframe=test_prepared,
        split_name="test",
        duplicate_report=test_duplicates,
        conflict_report=test_conflicts,
    )

    print("\nAudit overlap train-test:")
    print(
        "Jumlah pasangan overlap : "
        f"{len(overlap_report):,}"
    )

    if not overlap_report.empty:
        print(
            "Label overlap konsisten: "
            f"{int(overlap_report['label_consistent'].sum()):,}"
        )
        print(
            "Label overlap berbeda  : "
            f"{int((~overlap_report['label_consistent']).sum()):,}"
        )

    print("\nPreview train:")
    print(
        train_prepared[
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
        test_prepared[
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

    print("\nLaporan duplikasi tersimpan di:")
    print(AG_NEWS_DUPLICATE_REPORT)

    print("\nLaporan konflik label tersimpan di:")
    print(AG_NEWS_LABEL_CONFLICT_REPORT)

    print("\nLaporan overlap train-test tersimpan di:")
    print(AG_NEWS_TRAIN_TEST_OVERLAP_REPORT)

    print("\nStatus:")
    print(
        "Preparation selesai. Tidak ada baris yang dihapus."
    )
    print(
        "Cleaning dilanjutkan melalui "
        "4_preprocessing/01_data_cleaning.py."
    )


if __name__ == "__main__":
    main()
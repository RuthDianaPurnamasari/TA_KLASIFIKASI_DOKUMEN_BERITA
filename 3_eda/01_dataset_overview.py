from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
    FIGURES_DIR,
    KOMPAS_PROCESSED_PATH,
    TABLES_DIR,
)


# ============================================================
# OUTPUT FILE
# ============================================================

DATASET_OVERVIEW_PATH = (
    TABLES_DIR / "dataset_overview.csv"
)

CLASS_DISTRIBUTION_PATH = (
    TABLES_DIR / "class_distribution.csv"
)

KOMPAS_DISTRIBUTION_FIGURE = (
    FIGURES_DIR / "kompas_class_distribution.png"
)

AGNEWS_TRAIN_DISTRIBUTION_FIGURE = (
    FIGURES_DIR / "agnews_train_class_distribution.png"
)

AGNEWS_TEST_DISTRIBUTION_FIGURE = (
    FIGURES_DIR / "agnews_test_class_distribution.png"
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
            f"{file_path}\n\n"
            "Pastikan tahap persiapan data sudah dijalankan."
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
# MEMBUAT RINGKASAN DATASET
# ============================================================

def create_dataset_summary(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> dict:
    """
    Membuat ringkasan karakteristik dasar dataset.
    """

    summary = {
        "dataset": dataset_name,
        "jumlah_baris": len(dataframe),
        "jumlah_kolom": dataframe.shape[1],
        "jumlah_kategori": (
            dataframe["category"].nunique()
            if "category" in dataframe.columns
            else 0
        ),
        "total_missing_value": int(
            dataframe.isna().sum().sum()
        ),
        "total_duplikat_baris": int(
            dataframe.duplicated().sum()
        ),
        "nama_kolom": ", ".join(
            dataframe.columns.astype(str)
        ),
    }

    return summary


# ============================================================
# MENAMPILKAN OVERVIEW KOMPAS
# ============================================================

def display_kompas_overview(
    dataframe: pd.DataFrame,
) -> None:
    """
    Menampilkan karakteristik awal dataset Kompas.
    """

    print("\n" + "=" * 72)
    print("DATASET KOMPAS")
    print("=" * 72)

    print(f"Jumlah data       : {len(dataframe):,}")
    print(f"Jumlah kolom      : {dataframe.shape[1]}")
    print(
        f"Jumlah kategori   : "
        f"{dataframe['category'].nunique()}"
    )

    print("\nNama kolom:")
    for index, column in enumerate(
        dataframe.columns,
        start=1,
    ):
        print(f"{index}. {column}")

    print("\nTipe data:")
    print(dataframe.dtypes.to_string())

    print("\nDistribusi kategori:")
    print(
        dataframe["category"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nMissing value per kolom:")
    print(
        dataframe.isna()
        .sum()
        .to_string()
    )

    print(
        "\nJumlah duplikat seluruh baris: "
        f"{int(dataframe.duplicated().sum()):,}"
    )

    if "date" in dataframe.columns:
        parsed_date = pd.to_datetime(
            dataframe["date"],
            errors="coerce",
        )

        valid_date = parsed_date.dropna()

        if not valid_date.empty:
            print(
                "\nRentang tanggal berita:"
            )
            print(
                f"Tanggal awal       : "
                f"{valid_date.min()}"
            )
            print(
                f"Tanggal akhir      : "
                f"{valid_date.max()}"
            )
        else:
            print(
                "\nRentang tanggal tidak dapat dihitung."
            )

    print("\nPreview data:")
    preview_columns = [
        column
        for column in [
            "document_id",
            "title",
            "category",
            "date",
        ]
        if column in dataframe.columns
    ]

    print(
        dataframe[preview_columns]
        .head()
        .to_string(index=False)
    )


# ============================================================
# MENAMPILKAN OVERVIEW AG NEWS
# ============================================================

def display_agnews_overview(
    dataframe: pd.DataFrame,
    split_name: str,
) -> None:
    """
    Menampilkan karakteristik dataset AG News.
    """

    title = f"AG NEWS {split_name.upper()}"

    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

    print(f"Jumlah data       : {len(dataframe):,}")
    print(f"Jumlah kolom      : {dataframe.shape[1]}")
    print(
        f"Jumlah kategori   : "
        f"{dataframe['category'].nunique()}"
    )

    print("\nNama kolom:")
    for index, column in enumerate(
        dataframe.columns,
        start=1,
    ):
        print(f"{index}. {column}")

    print("\nTipe data:")
    print(dataframe.dtypes.to_string())

    print("\nDistribusi kategori:")
    print(
        dataframe["category"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nMissing value per kolom:")
    print(
        dataframe.isna()
        .sum()
        .to_string()
    )

    empty_description = int(
        dataframe["description"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        "\nDescription kosong: "
        f"{empty_description:,}"
    )

    duplicate_articles = int(
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

    print(
        "Duplikat artikel  : "
        f"{duplicate_articles:,}"
    )

    print("\nPreview data:")
    preview_columns = [
        column
        for column in [
            "document_id",
            "class_index",
            "category",
            "title",
        ]
        if column in dataframe.columns
    ]

    print(
        dataframe[preview_columns]
        .head()
        .to_string(index=False)
    )


# ============================================================
# MEMBUAT TABEL DISTRIBUSI KATEGORI
# ============================================================

def create_class_distribution_table(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Membentuk tabel jumlah dan persentase setiap kategori.
    """

    distribution = (
        dataframe["category"]
        .value_counts()
        .sort_index()
        .rename_axis("category")
        .reset_index(name="jumlah_data")
    )

    distribution.insert(
        0,
        "dataset",
        dataset_name,
    )

    distribution["persentase"] = (
        distribution["jumlah_data"]
        / len(dataframe)
        * 100
    ).round(2)

    return distribution


# ============================================================
# MEMBUAT GRAFIK DISTRIBUSI KATEGORI
# ============================================================

def plot_class_distribution(
    dataframe: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    """
    Membuat bar chart distribusi kategori.
    """

    category_counts = (
        dataframe["category"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    bars = ax.bar(
        category_counts.index,
        category_counts.values,
    )

    ax.set_title(
        title,
        fontsize=14,
        pad=15,
    )

    ax.set_xlabel(
        "Kategori",
        fontsize=11,
    )

    ax.set_ylabel(
        "Jumlah Data",
        fontsize=11,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    for bar, value in zip(
        bars,
        category_counts.values,
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main() -> None:
    """
    Menjalankan EDA tahap pertama: dataset overview.
    """

    print("=" * 72)
    print("STEP 3.1 - DATASET OVERVIEW")
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

    print("\nDataset berhasil dimuat:")
    print(f"Kompas        : {len(kompas):,} data")
    print(
        f"AG News Train : "
        f"{len(agnews_train):,} data"
    )
    print(
        f"AG News Test  : "
        f"{len(agnews_test):,} data"
    )

    # ========================================================
    # MENAMPILKAN OVERVIEW
    # ========================================================

    display_kompas_overview(
        kompas
    )

    display_agnews_overview(
        agnews_train,
        "train",
    )

    display_agnews_overview(
        agnews_test,
        "test",
    )

    # ========================================================
    # MENYIMPAN TABEL OVERVIEW
    # ========================================================

    overview_records = [
        create_dataset_summary(
            kompas,
            "kompas",
        ),
        create_dataset_summary(
            agnews_train,
            "ag_news_train",
        ),
        create_dataset_summary(
            agnews_test,
            "ag_news_test",
        ),
    ]

    overview_dataframe = pd.DataFrame(
        overview_records
    )

    overview_dataframe.to_csv(
        DATASET_OVERVIEW_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # MENYIMPAN DISTRIBUSI KATEGORI
    # ========================================================

    class_distribution = pd.concat(
        [
            create_class_distribution_table(
                kompas,
                "kompas",
            ),
            create_class_distribution_table(
                agnews_train,
                "ag_news_train",
            ),
            create_class_distribution_table(
                agnews_test,
                "ag_news_test",
            ),
        ],
        ignore_index=True,
    )

    class_distribution.to_csv(
        CLASS_DISTRIBUTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # MEMBUAT GRAFIK
    # ========================================================

    plot_class_distribution(
        dataframe=kompas,
        title="Distribusi Kategori Dataset Kompas",
        output_path=KOMPAS_DISTRIBUTION_FIGURE,
    )

    plot_class_distribution(
        dataframe=agnews_train,
        title="Distribusi Kategori AG News Train",
        output_path=(
            AGNEWS_TRAIN_DISTRIBUTION_FIGURE
        ),
    )

    plot_class_distribution(
        dataframe=agnews_test,
        title="Distribusi Kategori AG News Test",
        output_path=(
            AGNEWS_TEST_DISTRIBUTION_FIGURE
        ),
    )

    # ========================================================
    # INFORMASI OUTPUT
    # ========================================================

    print("\n" + "=" * 72)
    print("OUTPUT DATASET OVERVIEW")
    print("=" * 72)

    print("\nTabel overview:")
    print(DATASET_OVERVIEW_PATH)

    print("\nTabel distribusi kategori:")
    print(CLASS_DISTRIBUTION_PATH)

    print("\nGrafik distribusi Kompas:")
    print(KOMPAS_DISTRIBUTION_FIGURE)

    print("\nGrafik distribusi AG News train:")
    print(
        AGNEWS_TRAIN_DISTRIBUTION_FIGURE
    )

    print("\nGrafik distribusi AG News test:")
    print(
        AGNEWS_TEST_DISTRIBUTION_FIGURE
    )

    print("\nTahap dataset overview selesai.")


if __name__ == "__main__":
    main()
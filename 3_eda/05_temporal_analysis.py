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
    FIGURES_DIR,
    KOMPAS_PROCESSED_PATH,
    TABLES_DIR,
)


# ============================================================
# OUTPUT FILE
# ============================================================

MONTHLY_DISTRIBUTION_PATH = (
    TABLES_DIR / "kompas_monthly_distribution.csv"
)

MONTHLY_CATEGORY_DISTRIBUTION_PATH = (
    TABLES_DIR / "kompas_monthly_category_distribution.csv"
)

DAILY_DISTRIBUTION_PATH = (
    TABLES_DIR / "kompas_daily_distribution.csv"
)

HOURLY_DISTRIBUTION_PATH = (
    TABLES_DIR / "kompas_hourly_distribution.csv"
)

TEMPORAL_SUMMARY_PATH = (
    TABLES_DIR / "kompas_temporal_summary.csv"
)

MONTHLY_DISTRIBUTION_FIGURE = (
    FIGURES_DIR / "kompas_monthly_distribution.png"
)

MONTHLY_CATEGORY_FIGURE = (
    FIGURES_DIR / "kompas_monthly_category_distribution.png"
)

DAILY_DISTRIBUTION_FIGURE = (
    FIGURES_DIR / "kompas_daily_distribution.png"
)

HOURLY_DISTRIBUTION_FIGURE = (
    FIGURES_DIR / "kompas_hourly_distribution.png"
)


# ============================================================
# MEMBACA DATASET
# ============================================================

def load_kompas_dataset(
    file_path: Path,
) -> pd.DataFrame:
    """
    Membaca dataset Kompas processed.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset Kompas tidak ditemukan:\n{file_path}"
        )

    dataframe = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
    )

    if dataframe.empty:
        raise ValueError(
            "Dataset Kompas kosong."
        )

    required_columns = [
        "document_id",
        "date",
        "category",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset Kompas tidak memiliki kolom: "
            f"{missing_columns}"
        )

    return dataframe


# ============================================================
# MENYIAPKAN FITUR WAKTU
# ============================================================

def prepare_temporal_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mengubah kolom tanggal menjadi datetime dan menambahkan
    atribut waktu untuk keperluan EDA.

    Atribut:
    - publication_date
    - year
    - month
    - month_period
    - hour
    - day_name
    """

    dataframe = dataframe.copy()

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="coerce",
    )

    invalid_dates = int(
        dataframe["date"].isna().sum()
    )

    if invalid_dates > 0:
        raise ValueError(
            f"Ditemukan {invalid_dates} tanggal yang tidak valid."
        )

    dataframe["publication_date"] = (
        dataframe["date"].dt.date
    )

    dataframe["year"] = (
        dataframe["date"].dt.year
    )

    dataframe["month"] = (
        dataframe["date"].dt.month
    )

    dataframe["month_period"] = (
        dataframe["date"]
        .dt.to_period("M")
        .astype(str)
    )

    dataframe["hour"] = (
        dataframe["date"].dt.hour
    )

    dataframe["day_name"] = (
        dataframe["date"].dt.day_name()
    )

    return dataframe


# ============================================================
# DISTRIBUSI BULANAN
# ============================================================

def create_monthly_distribution(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Menghitung jumlah berita per bulan.
    """

    monthly_distribution = (
        dataframe.groupby(
            "month_period"
        )
        .size()
        .reset_index(
            name="jumlah_berita"
        )
        .sort_values(
            "month_period"
        )
        .reset_index(drop=True)
    )

    monthly_distribution["persentase"] = (
        monthly_distribution["jumlah_berita"]
        / len(dataframe)
        * 100
    ).round(2)

    return monthly_distribution


# ============================================================
# DISTRIBUSI BULANAN PER KATEGORI
# ============================================================

def create_monthly_category_distribution(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Menghitung jumlah berita per bulan dan kategori.
    """

    result = (
        dataframe.groupby(
            [
                "month_period",
                "category",
            ]
        )
        .size()
        .reset_index(
            name="jumlah_berita"
        )
        .sort_values(
            [
                "month_period",
                "category",
            ]
        )
        .reset_index(drop=True)
    )

    return result


# ============================================================
# DISTRIBUSI HARIAN
# ============================================================

def create_daily_distribution(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Menghitung jumlah berita setiap tanggal.
    """

    result = (
        dataframe.groupby(
            "publication_date"
        )
        .size()
        .reset_index(
            name="jumlah_berita"
        )
        .sort_values(
            "publication_date"
        )
        .reset_index(drop=True)
    )

    result["publication_date"] = (
        pd.to_datetime(
            result["publication_date"]
        )
    )

    return result


# ============================================================
# DISTRIBUSI JAM PUBLIKASI
# ============================================================

def create_hourly_distribution(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Menghitung jumlah berita berdasarkan jam publikasi.
    """

    hourly_distribution = (
        dataframe.groupby(
            "hour"
        )
        .size()
        .reindex(
            range(24),
            fill_value=0,
        )
        .reset_index(
            name="jumlah_berita"
        )
    )

    hourly_distribution["persentase"] = (
        hourly_distribution["jumlah_berita"]
        / len(dataframe)
        * 100
    ).round(2)

    return hourly_distribution


# ============================================================
# RINGKASAN TEMPORAL
# ============================================================

def create_temporal_summary(
    dataframe: pd.DataFrame,
    monthly_distribution: pd.DataFrame,
    daily_distribution: pd.DataFrame,
    hourly_distribution: pd.DataFrame,
) -> pd.DataFrame:
    """
    Membuat ringkasan hasil analisis waktu.
    """

    most_active_month = monthly_distribution.loc[
        monthly_distribution[
            "jumlah_berita"
        ].idxmax()
    ]

    most_active_date = daily_distribution.loc[
        daily_distribution[
            "jumlah_berita"
        ].idxmax()
    ]

    most_active_hour = hourly_distribution.loc[
        hourly_distribution[
            "jumlah_berita"
        ].idxmax()
    ]

    summary = pd.DataFrame(
        [
            {
                "jumlah_data": len(dataframe),
                "tanggal_awal": (
                    dataframe["date"]
                    .min()
                    .strftime("%Y-%m-%d %H:%M:%S")
                ),
                "tanggal_akhir": (
                    dataframe["date"]
                    .max()
                    .strftime("%Y-%m-%d %H:%M:%S")
                ),
                "jumlah_hari_unik": int(
                    dataframe[
                        "publication_date"
                    ].nunique()
                ),
                "bulan_teraktif": (
                    most_active_month[
                        "month_period"
                    ]
                ),
                "jumlah_berita_bulan_teraktif": int(
                    most_active_month[
                        "jumlah_berita"
                    ]
                ),
                "tanggal_teraktif": (
                    most_active_date[
                        "publication_date"
                    ].strftime("%Y-%m-%d")
                ),
                "jumlah_berita_tanggal_teraktif": int(
                    most_active_date[
                        "jumlah_berita"
                    ]
                ),
                "jam_teraktif": int(
                    most_active_hour["hour"]
                ),
                "jumlah_berita_jam_teraktif": int(
                    most_active_hour[
                        "jumlah_berita"
                    ]
                ),
            }
        ]
    )

    return summary


# ============================================================
# GRAFIK DISTRIBUSI BULANAN
# ============================================================

def plot_monthly_distribution(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Membuat grafik jumlah berita per bulan.
    """

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars = ax.bar(
        dataframe["month_period"],
        dataframe["jumlah_berita"],
    )

    ax.set_title(
        "Distribusi Berita Kompas per Bulan",
        fontsize=14,
        pad=15,
    )

    ax.set_xlabel(
        "Bulan",
        fontsize=11,
    )

    ax.set_ylabel(
        "Jumlah Berita",
        fontsize=11,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    for bar, value in zip(
        bars,
        dataframe["jumlah_berita"],
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
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
# GRAFIK DISTRIBUSI BULANAN PER KATEGORI
# ============================================================

def plot_monthly_category_distribution(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Membuat line chart jumlah berita setiap kategori per bulan.
    """

    pivot_table = dataframe.pivot(
        index="month_period",
        columns="category",
        values="jumlah_berita",
    ).fillna(0)

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    for category in pivot_table.columns:
        ax.plot(
            pivot_table.index,
            pivot_table[category],
            marker="o",
            linewidth=2,
            label=category,
        )

    ax.set_title(
        "Distribusi Kategori Berita Kompas per Bulan",
        fontsize=14,
        pad=15,
    )

    ax.set_xlabel(
        "Bulan",
        fontsize=11,
    )

    ax.set_ylabel(
        "Jumlah Berita",
        fontsize=11,
    )

    ax.legend(
        title="Kategori"
    )

    ax.grid(
        linestyle="--",
        alpha=0.4,
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# GRAFIK DISTRIBUSI HARIAN
# ============================================================

def plot_daily_distribution(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Membuat grafik jumlah berita setiap tanggal.
    """

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.plot(
        dataframe["publication_date"],
        dataframe["jumlah_berita"],
        linewidth=1.5,
    )

    ax.set_title(
        "Distribusi Harian Berita Kompas",
        fontsize=14,
        pad=15,
    )

    ax.set_xlabel(
        "Tanggal",
        fontsize=11,
    )

    ax.set_ylabel(
        "Jumlah Berita",
        fontsize=11,
    )

    ax.grid(
        linestyle="--",
        alpha=0.4,
    )

    fig.autofmt_xdate()

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# GRAFIK DISTRIBUSI JAM
# ============================================================

def plot_hourly_distribution(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Membuat grafik distribusi jam publikasi.
    """

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    bars = ax.bar(
        dataframe["hour"],
        dataframe["jumlah_berita"],
    )

    ax.set_title(
        "Distribusi Jam Publikasi Berita Kompas",
        fontsize=14,
        pad=15,
    )

    ax.set_xlabel(
        "Jam Publikasi",
        fontsize=11,
    )

    ax.set_ylabel(
        "Jumlah Berita",
        fontsize=11,
    )

    ax.set_xticks(
        range(24)
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    for bar, value in zip(
        bars,
        dataframe["jumlah_berita"],
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
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
    Menjalankan analisis temporal dataset Kompas.
    """

    print("=" * 72)
    print("STEP 3.5 - TEMPORAL ANALYSIS")
    print("=" * 72)

    # ========================================================
    # MEMUAT DATASET
    # ========================================================

    kompas = load_kompas_dataset(
        KOMPAS_PROCESSED_PATH
    )

    kompas = prepare_temporal_features(
        kompas
    )

    # ========================================================
    # MEMBUAT HASIL ANALISIS
    # ========================================================

    monthly_distribution = (
        create_monthly_distribution(
            kompas
        )
    )

    monthly_category_distribution = (
        create_monthly_category_distribution(
            kompas
        )
    )

    daily_distribution = (
        create_daily_distribution(
            kompas
        )
    )

    hourly_distribution = (
        create_hourly_distribution(
            kompas
        )
    )

    temporal_summary = (
        create_temporal_summary(
            dataframe=kompas,
            monthly_distribution=monthly_distribution,
            daily_distribution=daily_distribution,
            hourly_distribution=hourly_distribution,
        )
    )

    # ========================================================
    # MENYIMPAN TABEL
    # ========================================================

    monthly_distribution.to_csv(
        MONTHLY_DISTRIBUTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    monthly_category_distribution.to_csv(
        MONTHLY_CATEGORY_DISTRIBUTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    daily_distribution.to_csv(
        DAILY_DISTRIBUTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    hourly_distribution.to_csv(
        HOURLY_DISTRIBUTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    temporal_summary.to_csv(
        TEMPORAL_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # MENAMPILKAN HASIL TERMINAL
    # ========================================================

    print("\nRentang tanggal:")
    print(
        f"Tanggal awal : "
        f"{kompas['date'].min()}"
    )
    print(
        f"Tanggal akhir: "
        f"{kompas['date'].max()}"
    )

    print("\nDistribusi berita per bulan:")
    print(
        monthly_distribution.to_string(
            index=False
        )
    )

    print("\nDistribusi berita per kategori dan bulan:")
    print(
        monthly_category_distribution.to_string(
            index=False
        )
    )

    print("\n10 tanggal dengan berita terbanyak:")
    print(
        daily_distribution
        .sort_values(
            "jumlah_berita",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nDistribusi jam publikasi:")
    print(
        hourly_distribution.to_string(
            index=False
        )
    )

    print("\nRingkasan temporal:")
    print(
        temporal_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # MEMBUAT GRAFIK
    # ========================================================

    plot_monthly_distribution(
        monthly_distribution,
        MONTHLY_DISTRIBUTION_FIGURE,
    )

    plot_monthly_category_distribution(
        monthly_category_distribution,
        MONTHLY_CATEGORY_FIGURE,
    )

    plot_daily_distribution(
        daily_distribution,
        DAILY_DISTRIBUTION_FIGURE,
    )

    plot_hourly_distribution(
        hourly_distribution,
        HOURLY_DISTRIBUTION_FIGURE,
    )

    # ========================================================
    # INFORMASI OUTPUT
    # ========================================================

    print("\n" + "=" * 72)
    print("OUTPUT TEMPORAL ANALYSIS")
    print("=" * 72)

    print("\nDistribusi bulanan:")
    print(MONTHLY_DISTRIBUTION_PATH)

    print("\nDistribusi bulanan per kategori:")
    print(
        MONTHLY_CATEGORY_DISTRIBUTION_PATH
    )

    print("\nDistribusi harian:")
    print(DAILY_DISTRIBUTION_PATH)

    print("\nDistribusi jam:")
    print(HOURLY_DISTRIBUTION_PATH)

    print("\nRingkasan temporal:")
    print(TEMPORAL_SUMMARY_PATH)

    print("\nGrafik distribusi bulanan:")
    print(MONTHLY_DISTRIBUTION_FIGURE)

    print("\nGrafik kategori per bulan:")
    print(MONTHLY_CATEGORY_FIGURE)

    print("\nGrafik distribusi harian:")
    print(DAILY_DISTRIBUTION_FIGURE)

    print("\nGrafik distribusi jam:")
    print(HOURLY_DISTRIBUTION_FIGURE)

    print("\nTahap temporal analysis selesai.")


if __name__ == "__main__":
    main()
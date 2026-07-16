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

TEXT_STATISTICS_PATH = (
    TABLES_DIR / "text_statistics.csv"
)

TEXT_STATISTICS_BY_CATEGORY_PATH = (
    TABLES_DIR / "text_statistics_by_category.csv"
)

KOMPAS_WORD_LENGTH_FIGURE = (
    FIGURES_DIR / "kompas_text_length_distribution.png"
)

AGNEWS_TRAIN_WORD_LENGTH_FIGURE = (
    FIGURES_DIR / "agnews_train_text_length_distribution.png"
)

AGNEWS_TEST_WORD_LENGTH_FIGURE = (
    FIGURES_DIR / "agnews_test_text_length_distribution.png"
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
# MENGHITUNG PANJANG TEKS
# ============================================================

def add_text_length_features(
    dataframe: pd.DataFrame,
    text_columns: list[str],
) -> pd.DataFrame:
    """
    Menambahkan jumlah kata dan karakter untuk setiap kolom teks.

    Contoh:
    title_word_count
    title_char_count
    """

    dataframe = dataframe.copy()

    for column in text_columns:
        dataframe[column] = (
            dataframe[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        dataframe[f"{column}_word_count"] = (
            dataframe[column]
            .str.split()
            .str.len()
        )

        dataframe[f"{column}_char_count"] = (
            dataframe[column]
            .str.len()
        )

    return dataframe


# ============================================================
# MEMBUAT STATISTIK RINGKAS
# ============================================================

def create_text_statistics(
    dataframe: pd.DataFrame,
    dataset_name: str,
    text_columns: list[str],
) -> pd.DataFrame:
    """
    Membuat statistik minimum, maksimum, rata-rata,
    median, dan standar deviasi.
    """

    records: list[dict] = []

    for column in text_columns:
        for unit in ["word_count", "char_count"]:
            metric_column = f"{column}_{unit}"

            series = dataframe[metric_column]

            records.append(
                {
                    "dataset": dataset_name,
                    "text_field": column,
                    "unit": unit,
                    "jumlah_data": len(series),
                    "minimum": int(series.min()),
                    "maksimum": int(series.max()),
                    "mean": round(float(series.mean()), 2),
                    "median": round(float(series.median()), 2),
                    "std": round(float(series.std()), 2),
                    "q1": round(float(series.quantile(0.25)), 2),
                    "q3": round(float(series.quantile(0.75)), 2),
                    "jumlah_kosong": int(
                        series.eq(0).sum()
                    ),
                }
            )

    return pd.DataFrame(records)


# ============================================================
# STATISTIK PER KATEGORI
# ============================================================

def create_statistics_by_category(
    dataframe: pd.DataFrame,
    dataset_name: str,
    text_columns: list[str],
) -> pd.DataFrame:
    """
    Menghitung rata-rata panjang teks per kategori.
    """

    records: list[dict] = []

    for category, group in dataframe.groupby(
        "category",
        dropna=False,
    ):
        for column in text_columns:
            word_column = f"{column}_word_count"
            char_column = f"{column}_char_count"

            records.append(
                {
                    "dataset": dataset_name,
                    "category": category,
                    "text_field": column,
                    "jumlah_data": len(group),
                    "mean_word_count": round(
                        float(group[word_column].mean()),
                        2,
                    ),
                    "median_word_count": round(
                        float(group[word_column].median()),
                        2,
                    ),
                    "mean_char_count": round(
                        float(group[char_column].mean()),
                        2,
                    ),
                    "median_char_count": round(
                        float(group[char_column].median()),
                        2,
                    ),
                }
            )

    return pd.DataFrame(records)


# ============================================================
# MENAMPILKAN STATISTIK
# ============================================================

def display_dataset_statistics(
    dataframe: pd.DataFrame,
    dataset_name: str,
    text_columns: list[str],
) -> None:
    """
    Menampilkan statistik panjang teks di terminal.
    """

    print("\n" + "=" * 72)
    print(dataset_name.upper())
    print("=" * 72)

    for column in text_columns:
        word_column = f"{column}_word_count"
        char_column = f"{column}_char_count"

        print(f"\nKolom: {column}")

        print(
            f"Rata-rata kata    : "
            f"{dataframe[word_column].mean():.2f}"
        )

        print(
            f"Median kata       : "
            f"{dataframe[word_column].median():.2f}"
        )

        print(
            f"Minimum kata      : "
            f"{dataframe[word_column].min():,}"
        )

        print(
            f"Maksimum kata     : "
            f"{dataframe[word_column].max():,}"
        )

        print(
            f"Rata-rata karakter: "
            f"{dataframe[char_column].mean():.2f}"
        )

        print(
            f"Teks kosong       : "
            f"{int(dataframe[word_column].eq(0).sum()):,}"
        )


# ============================================================
# MEMBUAT GRAFIK HISTOGRAM
# ============================================================

def plot_text_length_distribution(
    dataframe: pd.DataFrame,
    text_columns: list[str],
    dataset_name: str,
    output_path: Path,
) -> None:
    """
    Membuat histogram distribusi jumlah kata.
    """

    figure_count = len(text_columns)

    fig, axes = plt.subplots(
        figure_count,
        1,
        figsize=(10, 5 * figure_count),
    )

    if figure_count == 1:
        axes = [axes]

    for axis, column in zip(
        axes,
        text_columns,
    ):
        word_column = f"{column}_word_count"

        axis.hist(
            dataframe[word_column],
            bins=40,
            edgecolor="black",
            alpha=0.8,
        )

        axis.set_title(
            f"Distribusi Panjang {column.title()} - {dataset_name}",
            fontsize=13,
            pad=12,
        )

        axis.set_xlabel(
            "Jumlah Kata",
            fontsize=11,
        )

        axis.set_ylabel(
            "Frekuensi",
            fontsize=11,
        )

        axis.grid(
            axis="y",
            linestyle="--",
            alpha=0.3,
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
    Menjalankan EDA statistik panjang teks.
    """

    print("=" * 72)
    print("STEP 3.2 - TEXT STATISTICS")
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
    # MENENTUKAN KOLOM TEKS
    # ========================================================

    kompas_text_columns = [
        "title",
        "description",
        "content",
    ]

    agnews_text_columns = [
        "title",
        "description",
    ]

    # ========================================================
    # MENAMBAHKAN FITUR PANJANG TEKS
    # ========================================================

    kompas = add_text_length_features(
        kompas,
        kompas_text_columns,
    )

    agnews_train = add_text_length_features(
        agnews_train,
        agnews_text_columns,
    )

    agnews_test = add_text_length_features(
        agnews_test,
        agnews_text_columns,
    )

    # ========================================================
    # MENAMPILKAN HASIL
    # ========================================================

    display_dataset_statistics(
        kompas,
        "Dataset Kompas",
        kompas_text_columns,
    )

    display_dataset_statistics(
        agnews_train,
        "AG News Train",
        agnews_text_columns,
    )

    display_dataset_statistics(
        agnews_test,
        "AG News Test",
        agnews_text_columns,
    )

    # ========================================================
    # MEMBUAT TABEL STATISTIK KESELURUHAN
    # ========================================================

    text_statistics = pd.concat(
        [
            create_text_statistics(
                kompas,
                "kompas",
                kompas_text_columns,
            ),
            create_text_statistics(
                agnews_train,
                "ag_news_train",
                agnews_text_columns,
            ),
            create_text_statistics(
                agnews_test,
                "ag_news_test",
                agnews_text_columns,
            ),
        ],
        ignore_index=True,
    )

    text_statistics.to_csv(
        TEXT_STATISTICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # MEMBUAT TABEL STATISTIK PER KATEGORI
    # ========================================================

    text_statistics_by_category = pd.concat(
        [
            create_statistics_by_category(
                kompas,
                "kompas",
                kompas_text_columns,
            ),
            create_statistics_by_category(
                agnews_train,
                "ag_news_train",
                agnews_text_columns,
            ),
            create_statistics_by_category(
                agnews_test,
                "ag_news_test",
                agnews_text_columns,
            ),
        ],
        ignore_index=True,
    )

    text_statistics_by_category.to_csv(
        TEXT_STATISTICS_BY_CATEGORY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # MEMBUAT GRAFIK
    # ========================================================

    plot_text_length_distribution(
        dataframe=kompas,
        text_columns=kompas_text_columns,
        dataset_name="Kompas",
        output_path=KOMPAS_WORD_LENGTH_FIGURE,
    )

    plot_text_length_distribution(
        dataframe=agnews_train,
        text_columns=agnews_text_columns,
        dataset_name="AG News Train",
        output_path=AGNEWS_TRAIN_WORD_LENGTH_FIGURE,
    )

    plot_text_length_distribution(
        dataframe=agnews_test,
        text_columns=agnews_text_columns,
        dataset_name="AG News Test",
        output_path=AGNEWS_TEST_WORD_LENGTH_FIGURE,
    )

    # ========================================================
    # INFORMASI OUTPUT
    # ========================================================

    print("\n" + "=" * 72)
    print("OUTPUT TEXT STATISTICS")
    print("=" * 72)

    print("\nTabel statistik keseluruhan:")
    print(TEXT_STATISTICS_PATH)

    print("\nTabel statistik per kategori:")
    print(TEXT_STATISTICS_BY_CATEGORY_PATH)

    print("\nGrafik distribusi teks Kompas:")
    print(KOMPAS_WORD_LENGTH_FIGURE)

    print("\nGrafik distribusi teks AG News train:")
    print(AGNEWS_TRAIN_WORD_LENGTH_FIGURE)

    print("\nGrafik distribusi teks AG News test:")
    print(AGNEWS_TEST_WORD_LENGTH_FIGURE)

    print("\nTahap text statistics selesai.")


if __name__ == "__main__":
    main()
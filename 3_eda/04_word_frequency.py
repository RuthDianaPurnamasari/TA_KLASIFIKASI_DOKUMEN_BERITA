from __future__ import annotations

import re
import sys
from collections import Counter
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

WORD_FREQUENCY_OVERALL_PATH = (
    TABLES_DIR / "word_frequency_overall.csv"
)

WORD_FREQUENCY_BY_CATEGORY_PATH = (
    TABLES_DIR / "word_frequency_by_category.csv"
)

KOMPAS_WORD_FREQUENCY_FIGURE = (
    FIGURES_DIR / "kompas_top_words.png"
)

AGNEWS_TRAIN_WORD_FREQUENCY_FIGURE = (
    FIGURES_DIR / "agnews_train_top_words.png"
)

AGNEWS_TEST_WORD_FREQUENCY_FIGURE = (
    FIGURES_DIR / "agnews_test_top_words.png"
)


# ============================================================
# STOPWORD DASAR
# ============================================================

INDONESIAN_STOPWORDS = {
    "yang",
    "dan",
    "di",
    "ke",
    "dari",
    "untuk",
    "dengan",
    "pada",
    "dalam",
    "ini",
    "itu",
    "adalah",
    "sebagai",
    "oleh",
    "akan",
    "atau",
    "juga",
    "karena",
    "ada",
    "tidak",
    "sudah",
    "telah",
    "bisa",
    "dapat",
    "lebih",
    "setelah",
    "saat",
    "menjadi",
    "hingga",
    "antara",
    "terhadap",
    "sebuah",
    "para",
    "masih",
    "yakni",
    "yaitu",
    "ia",
    "mereka",
    "kami",
    "kita",
    "saya",
    "anda",
    "dia",
    "nya",
    "pun",
    "per",
    "kompas",
    "com",
}

INDONESIAN_STOPWORDS.update({
    "jadi",
    "tak",
    "baru",
    "usai",
    "mulai",
    "hari",
    "tengah",
    "bakal",
    "sebut",
    "kata",
    "ungkap",
    "tahun",
})

ENGLISH_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "from",
    "with",
    "by",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "he",
    "she",
    "they",
    "we",
    "you",
    "i",
    "his",
    "her",
    "their",
    "our",
    "your",
    "not",
    "has",
    "have",
    "had",
    "will",
    "would",
    "can",
    "could",
    "may",
    "might",
    "do",
    "does",
    "did",
    "than",
    "after",
    "before",
    "into",
    "over",
    "under",
    "about",
    "up",
    "out",
    "new",
    "says",
    "said",
    "reuters",
    "ap",
    "afp",
}

ENGLISH_STOPWORDS.update({
    "two",
    "first",
    "year",
    "more",
    "one",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "last",
    "yesterday",
    "today",
    "tomorrow",
    "who",
    "quot",
})


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
# NORMALISASI TEKS UNTUK FREKUENSI KATA
# ============================================================

def tokenize_text(
    text: str,
    stopwords: set[str],
) -> list[str]:
    """
    Mengubah teks menjadi token sederhana.

    Tahap:
    - lowercase;
    - menghapus URL;
    - mengambil token alfabet;
    - menghapus stopword;
    - menghapus token dengan panjang kurang dari 3 karakter.

    Proses ini hanya untuk analisis EDA.
    Dataset asli tidak diubah.
    """

    text = str(text).lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
    )

    tokens = re.findall(
        r"[a-zA-ZÀ-ÿ]+",
        text,
    )

    filtered_tokens = [
        token
        for token in tokens
        if (
            token not in stopwords
            and len(token) >= 3
        )
    ]

    return filtered_tokens


# ============================================================
# MENGGABUNGKAN KOLOM TEKS
# ============================================================

def combine_text_columns(
    dataframe: pd.DataFrame,
    text_columns: list[str],
) -> pd.Series:
    """
    Menggabungkan beberapa kolom teks menjadi satu Series.
    """

    combined = pd.Series(
        "",
        index=dataframe.index,
        dtype="string",
    )

    for column in text_columns:
        if column not in dataframe.columns:
            raise ValueError(
                f"Kolom '{column}' tidak tersedia."
            )

        combined = (
            combined.fillna("")
            + " "
            + dataframe[column]
            .fillna("")
            .astype(str)
        )

    return combined.str.strip()


# ============================================================
# MENGHITUNG FREKUENSI KATA
# ============================================================

def calculate_word_frequency(
    texts: pd.Series,
    stopwords: set[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Menghitung frekuensi kata teratas.
    """

    counter: Counter[str] = Counter()

    for text in texts:
        tokens = tokenize_text(
            text=text,
            stopwords=stopwords,
        )

        counter.update(tokens)

    frequency = pd.DataFrame(
        counter.most_common(top_n),
        columns=[
            "word",
            "frequency",
        ],
    )

    return frequency


# ============================================================
# FREKUENSI KATA KESELURUHAN
# ============================================================

def create_overall_frequency(
    dataframe: pd.DataFrame,
    dataset_name: str,
    text_columns: list[str],
    stopwords: set[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Menghasilkan tabel frekuensi kata seluruh dataset.
    """

    combined_text = combine_text_columns(
        dataframe,
        text_columns,
    )

    frequency = calculate_word_frequency(
        combined_text,
        stopwords,
        top_n,
    )

    frequency.insert(
        0,
        "dataset",
        dataset_name,
    )

    frequency.insert(
        1,
        "text_source",
        " + ".join(text_columns),
    )

    frequency.insert(
        2,
        "rank",
        range(1, len(frequency) + 1),
    )

    return frequency


# ============================================================
# FREKUENSI KATA PER KATEGORI
# ============================================================

def create_frequency_by_category(
    dataframe: pd.DataFrame,
    dataset_name: str,
    text_columns: list[str],
    stopwords: set[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Menghasilkan tabel frekuensi kata per kategori.
    """

    records: list[pd.DataFrame] = []

    for category, group in dataframe.groupby(
        "category",
        dropna=False,
    ):
        combined_text = combine_text_columns(
            group,
            text_columns,
        )

        frequency = calculate_word_frequency(
            combined_text,
            stopwords,
            top_n,
        )

        frequency.insert(
            0,
            "dataset",
            dataset_name,
        )

        frequency.insert(
            1,
            "category",
            category,
        )

        frequency.insert(
            2,
            "text_source",
            " + ".join(text_columns),
        )

        frequency.insert(
            3,
            "rank",
            range(1, len(frequency) + 1),
        )

        records.append(frequency)

    return pd.concat(
        records,
        ignore_index=True,
    )


# ============================================================
# MENAMPILKAN HASIL
# ============================================================

def display_top_words(
    frequency: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Menampilkan kata teratas di terminal.
    """

    print("\n" + "=" * 72)
    print(f"TOP WORDS - {dataset_name.upper()}")
    print("=" * 72)

    print(
        frequency[
            [
                "rank",
                "word",
                "frequency",
            ]
        ].to_string(index=False)
    )


# ============================================================
# MEMBUAT GRAFIK TOP WORDS
# ============================================================

def plot_top_words(
    frequency: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    """
    Membuat horizontal bar chart kata teratas.
    """

    plot_data = (
        frequency
        .sort_values(
            "frequency",
            ascending=True,
        )
        .copy()
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    bars = ax.barh(
        plot_data["word"],
        plot_data["frequency"],
    )

    ax.set_title(
        title,
        fontsize=14,
        pad=15,
    )

    ax.set_xlabel(
        "Frekuensi",
        fontsize=11,
    )

    ax.set_ylabel(
        "Kata",
        fontsize=11,
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.3,
    )

    for bar, value in zip(
        bars,
        plot_data["frequency"],
    ):
        ax.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {value:,}",
            va="center",
            fontsize=9,
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
    Menjalankan analisis frekuensi kata.
    """

    print("=" * 72)
    print("STEP 3.4 - WORD FREQUENCY ANALYSIS")
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
    # KOLOM YANG DIGUNAKAN
    # ========================================================

    kompas_text_columns = [
        "title",
        "description",
    ]

    agnews_text_columns = [
        "title",
        "description",
    ]

    # Content Kompas belum dimasukkan agar hasil frekuensi
    # tidak didominasi artikel yang sangat panjang.
    # Analisis content dapat dibuat terpisah bila dibutuhkan.

    # ========================================================
    # FREKUENSI KESELURUHAN
    # ========================================================

    kompas_overall = create_overall_frequency(
        dataframe=kompas,
        dataset_name="kompas",
        text_columns=kompas_text_columns,
        stopwords=INDONESIAN_STOPWORDS,
        top_n=20,
    )

    agnews_train_overall = create_overall_frequency(
        dataframe=agnews_train,
        dataset_name="ag_news_train",
        text_columns=agnews_text_columns,
        stopwords=ENGLISH_STOPWORDS,
        top_n=20,
    )

    agnews_test_overall = create_overall_frequency(
        dataframe=agnews_test,
        dataset_name="ag_news_test",
        text_columns=agnews_text_columns,
        stopwords=ENGLISH_STOPWORDS,
        top_n=20,
    )

    overall_frequency = pd.concat(
        [
            kompas_overall,
            agnews_train_overall,
            agnews_test_overall,
        ],
        ignore_index=True,
    )

    overall_frequency.to_csv(
        WORD_FREQUENCY_OVERALL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # FREKUENSI PER KATEGORI
    # ========================================================

    kompas_by_category = create_frequency_by_category(
        dataframe=kompas,
        dataset_name="kompas",
        text_columns=kompas_text_columns,
        stopwords=INDONESIAN_STOPWORDS,
        top_n=20,
    )

    agnews_train_by_category = (
        create_frequency_by_category(
            dataframe=agnews_train,
            dataset_name="ag_news_train",
            text_columns=agnews_text_columns,
            stopwords=ENGLISH_STOPWORDS,
            top_n=20,
        )
    )

    agnews_test_by_category = (
        create_frequency_by_category(
            dataframe=agnews_test,
            dataset_name="ag_news_test",
            text_columns=agnews_text_columns,
            stopwords=ENGLISH_STOPWORDS,
            top_n=20,
        )
    )

    frequency_by_category = pd.concat(
        [
            kompas_by_category,
            agnews_train_by_category,
            agnews_test_by_category,
        ],
        ignore_index=True,
    )

    frequency_by_category.to_csv(
        WORD_FREQUENCY_BY_CATEGORY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # MENAMPILKAN HASIL
    # ========================================================

    display_top_words(
        kompas_overall,
        "Kompas",
    )

    display_top_words(
        agnews_train_overall,
        "AG News Train",
    )

    display_top_words(
        agnews_test_overall,
        "AG News Test",
    )

    # ========================================================
    # MEMBUAT GRAFIK
    # ========================================================

    plot_top_words(
        frequency=kompas_overall,
        title=(
            "20 Kata Paling Sering Muncul "
            "pada Dataset Kompas"
        ),
        output_path=KOMPAS_WORD_FREQUENCY_FIGURE,
    )

    plot_top_words(
        frequency=agnews_train_overall,
        title=(
            "20 Kata Paling Sering Muncul "
            "pada AG News Train"
        ),
        output_path=(
            AGNEWS_TRAIN_WORD_FREQUENCY_FIGURE
        ),
    )

    plot_top_words(
        frequency=agnews_test_overall,
        title=(
            "20 Kata Paling Sering Muncul "
            "pada AG News Test"
        ),
        output_path=(
            AGNEWS_TEST_WORD_FREQUENCY_FIGURE
        ),
    )

    # ========================================================
    # INFORMASI OUTPUT
    # ========================================================

    print("\n" + "=" * 72)
    print("OUTPUT WORD FREQUENCY ANALYSIS")
    print("=" * 72)

    print("\nFrekuensi kata keseluruhan:")
    print(WORD_FREQUENCY_OVERALL_PATH)

    print("\nFrekuensi kata per kategori:")
    print(WORD_FREQUENCY_BY_CATEGORY_PATH)

    print("\nGrafik Kompas:")
    print(KOMPAS_WORD_FREQUENCY_FIGURE)

    print("\nGrafik AG News train:")
    print(AGNEWS_TRAIN_WORD_FREQUENCY_FIGURE)

    print("\nGrafik AG News test:")
    print(AGNEWS_TEST_WORD_FREQUENCY_FIGURE)

    print("\nTahap word frequency analysis selesai.")


if __name__ == "__main__":
    main()
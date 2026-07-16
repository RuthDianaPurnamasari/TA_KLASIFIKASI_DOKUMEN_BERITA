from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud


# ============================================================
# MENAMBAHKAN ROOT PROJECT KE PYTHON PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config import (  # noqa: E402
    AG_NEWS_TRAIN_PROCESSED_PATH,
    FIGURES_DIR,
    KOMPAS_PROCESSED_PATH,
)


# ============================================================
# FOLDER OUTPUT
# ============================================================

WORDCLOUD_DIR = FIGURES_DIR / "wordclouds"
WORDCLOUD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# STOPWORD
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
}

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
}


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
# MEMBERSIHKAN TEKS UNTUK WORD CLOUD
# ============================================================

def clean_text_for_wordcloud(
    text: str,
    stopwords: set[str],
) -> str:
    """
    Membersihkan teks khusus untuk visualisasi word cloud.

    Proses ini tidak mengubah dataset utama.
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

    return " ".join(filtered_tokens)


# ============================================================
# MENGGABUNGKAN KOLOM TEKS
# ============================================================

def combine_text_columns(
    dataframe: pd.DataFrame,
    text_columns: list[str],
) -> pd.Series:
    """
    Menggabungkan beberapa kolom teks.
    """

    combined = pd.Series(
        "",
        index=dataframe.index,
        dtype="string",
    )

    for column in text_columns:
        if column not in dataframe.columns:
            raise ValueError(
                f"Kolom '{column}' tidak ditemukan."
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
# MEMBUAT WORD CLOUD
# ============================================================

def generate_wordcloud(
    texts: pd.Series,
    stopwords: set[str],
    title: str,
    output_path: Path,
) -> None:
    """
    Membuat dan menyimpan word cloud.
    """

    cleaned_documents = []

    for text in texts:
        cleaned = clean_text_for_wordcloud(
            text,
            stopwords,
        )

        if cleaned:
            cleaned_documents.append(cleaned)

    combined_text = " ".join(
        cleaned_documents
    )

    if not combined_text.strip():
        raise ValueError(
            f"Teks kosong untuk word cloud: {title}"
        )

    wordcloud = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        max_words=150,
        collocations=False,
        random_state=42,
    ).generate(combined_text)

    fig, ax = plt.subplots(
        figsize=(16, 9)
    )

    ax.imshow(
        wordcloud,
        interpolation="bilinear",
    )

    ax.set_title(
        title,
        fontsize=18,
        pad=20,
    )

    ax.axis("off")

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MEMBUAT WORD CLOUD PER KATEGORI
# ============================================================

def generate_wordclouds_by_category(
    dataframe: pd.DataFrame,
    dataset_name: str,
    text_columns: list[str],
    stopwords: set[str],
) -> list[Path]:
    """
    Membuat satu word cloud untuk setiap kategori.
    """

    output_paths: list[Path] = []

    for category, group in dataframe.groupby(
        "category",
        dropna=False,
    ):
        combined_text = combine_text_columns(
            group,
            text_columns,
        )

        safe_category = (
            str(category)
            .lower()
            .replace("/", "_")
            .replace(" ", "_")
        )

        output_path = (
            WORDCLOUD_DIR
            / f"{dataset_name}_{safe_category}_wordcloud.png"
        )

        generate_wordcloud(
            texts=combined_text,
            stopwords=stopwords,
            title=(
                f"Word Cloud {dataset_name.replace('_', ' ').title()} "
                f"- Kategori {str(category).title()}"
            ),
            output_path=output_path,
        )

        output_paths.append(
            output_path
        )

    return output_paths


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main() -> None:
    """
    Menjalankan pembuatan word cloud.
    """

    print("=" * 72)
    print("STEP 3.6 - WORD CLOUD ANALYSIS")
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

    # ========================================================
    # SUMBER TEKS
    # ========================================================

    kompas_text_columns = [
        "title",
        "description",
    ]

    agnews_text_columns = [
        "title",
        "description",
    ]

    # ========================================================
    # WORD CLOUD KESELURUHAN
    # ========================================================

    kompas_overall_text = combine_text_columns(
        kompas,
        kompas_text_columns,
    )

    agnews_train_overall_text = combine_text_columns(
        agnews_train,
        agnews_text_columns,
    )

    kompas_overall_output = (
        WORDCLOUD_DIR
        / "kompas_overall_wordcloud.png"
    )

    agnews_train_overall_output = (
        WORDCLOUD_DIR
        / "agnews_train_overall_wordcloud.png"
    )

    generate_wordcloud(
        texts=kompas_overall_text,
        stopwords=INDONESIAN_STOPWORDS,
        title=(
            "Word Cloud Dataset Kompas "
            "(Title + Description)"
        ),
        output_path=kompas_overall_output,
    )

    generate_wordcloud(
        texts=agnews_train_overall_text,
        stopwords=ENGLISH_STOPWORDS,
        title=(
            "Word Cloud AG News Train "
            "(Title + Description)"
        ),
        output_path=agnews_train_overall_output,
    )

    # ========================================================
    # WORD CLOUD PER KATEGORI
    # ========================================================

    kompas_category_outputs = (
        generate_wordclouds_by_category(
            dataframe=kompas,
            dataset_name="kompas",
            text_columns=kompas_text_columns,
            stopwords=INDONESIAN_STOPWORDS,
        )
    )

    agnews_category_outputs = (
        generate_wordclouds_by_category(
            dataframe=agnews_train,
            dataset_name="agnews_train",
            text_columns=agnews_text_columns,
            stopwords=ENGLISH_STOPWORDS,
        )
    )

    # ========================================================
    # INFORMASI OUTPUT
    # ========================================================

    print("\nWord cloud keseluruhan Kompas:")
    print(kompas_overall_output)

    print("\nWord cloud keseluruhan AG News Train:")
    print(agnews_train_overall_output)

    print("\nWord cloud Kompas per kategori:")

    for output_path in kompas_category_outputs:
        print(output_path)

    print("\nWord cloud AG News Train per kategori:")

    for output_path in agnews_category_outputs:
        print(output_path)

    print("\n" + "=" * 72)
    print("OUTPUT WORD CLOUD ANALYSIS")
    print("=" * 72)

    print("\nSeluruh gambar tersimpan di:")
    print(WORDCLOUD_DIR)

    print("\nTahap word cloud analysis selesai.")


if __name__ == "__main__":
    main()
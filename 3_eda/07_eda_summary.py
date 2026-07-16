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
# PATH OUTPUT
# ============================================================

EDA_SUMMARY_PATH = (
    TABLES_DIR
    / "eda_summary.csv"
)

EDA_RESEARCH_FINDINGS_PATH = (
    TABLES_DIR
    / "eda_research_findings.csv"
)


# ============================================================
# MEMBACA DATASET
# ============================================================

def load_dataset(
    file_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Membaca dataset processed dan melakukan validasi dasar.
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
# MENGHITUNG TEKS KOSONG
# ============================================================

def count_empty_text(
    series: pd.Series,
) -> int:
    """
    Menghitung nilai NaN dan string kosong tanpa menghitung
    satu baris dua kali.
    """

    normalized = (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return int(
        normalized.eq("").sum()
    )


# ============================================================
# MENGHITUNG RATA-RATA PANJANG KATA
# ============================================================

def average_word_count(
    series: pd.Series,
) -> float:
    """
    Menghitung rata-rata jumlah kata pada kolom teks.
    """

    normalized = (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )

    word_count = normalized.apply(
        lambda text: (
            len(text.split())
            if text
            else 0
        )
    )

    return round(
        float(word_count.mean()),
        2,
    )


# ============================================================
# MEMBUAT RINGKASAN DATASET
# ============================================================

def create_dataset_summary(
    dataframe: pd.DataFrame,
    dataset_name: str,
    text_columns: list[str],
) -> dict:
    """
    Membuat satu baris ringkasan karakteristik dataset.
    """

    summary = {
        "dataset": dataset_name,
        "jumlah_data": len(dataframe),
        "jumlah_kategori": (
            dataframe["category"]
            .nunique()
        ),
    }

    # --------------------------------------------------------
    # Distribusi kategori
    # --------------------------------------------------------

    category_distribution = (
        dataframe["category"]
        .value_counts()
        .sort_index()
    )

    summary["distribusi_kategori"] = (
        "; ".join(
            [
                f"{category}: {count}"
                for category, count
                in category_distribution.items()
            ]
        )
    )

    # --------------------------------------------------------
    # Statistik teks
    # --------------------------------------------------------

    for column in text_columns:

        if column in dataframe.columns:

            summary[
                f"avg_words_{column}"
            ] = average_word_count(
                dataframe[column]
            )

            summary[
                f"empty_{column}"
            ] = count_empty_text(
                dataframe[column]
            )

    return summary


# ============================================================
# MEMBUAT TEMUAN PENELITIAN
# ============================================================

def create_research_findings(
    kompas: pd.DataFrame,
    agnews_train: pd.DataFrame,
    agnews_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Membuat tabel temuan EDA dan implikasinya
    terhadap desain eksperimen penelitian.
    """

    kompas_title_avg = average_word_count(
        kompas["title"]
    )

    kompas_description_avg = average_word_count(
        kompas["description"]
    )

    kompas_content_avg = average_word_count(
        kompas["content"]
    )

    agnews_title_avg = average_word_count(
        agnews_train["title"]
    )

    agnews_description_avg = average_word_count(
        agnews_train["description"]
    )

    agnews_test_empty_description = count_empty_text(
        agnews_test["description"]
    )

    findings = [
        {
            "no": 1,
            "aspek": "Keseimbangan kelas Kompas",
            "temuan": (
                "Dataset Kompas terdiri dari 10.000 artikel "
                "dengan empat kategori yang seimbang, yaitu "
                "bola, global, money, dan tekno, masing-masing "
                "sebanyak 2.500 artikel."
            ),
            "implikasi_penelitian": (
                "Dataset dapat digunakan untuk eksperimen "
                "klasifikasi multikelas tanpa memerlukan "
                "teknik penyeimbangan kelas pada tahap awal."
            ),
        },
        {
            "no": 2,
            "aspek": "Keseimbangan kelas AG News Train",
            "temuan": (
                "Dataset AG News Train terdiri dari 120.000 "
                "artikel dengan empat kategori seimbang, "
                "masing-masing sebanyak 30.000 artikel."
            ),
            "implikasi_penelitian": (
                "AG News dapat digunakan sebagai dataset "
                "benchmark untuk membandingkan kinerja model "
                "pada dataset internasional."
            ),
        },
        {
            "no": 3,
            "aspek": "Panjang title Kompas",
            "temuan": (
                f"Title Kompas memiliki rata-rata "
                f"{kompas_title_avg} kata."
            ),
            "implikasi_penelitian": (
                "Title merupakan representasi teks yang ringkas "
                "dan digunakan sebagai skenario dasar."
            ),
        },
        {
            "no": 4,
            "aspek": "Panjang description Kompas",
            "temuan": (
                f"Description Kompas memiliki rata-rata "
                f"{kompas_description_avg} kata."
            ),
            "implikasi_penelitian": (
                "Description menyediakan konteks tambahan "
                "yang diuji melalui kombinasi "
                "Title + Description."
            ),
        },
        {
            "no": 5,
            "aspek": "Panjang content Kompas",
            "temuan": (
                f"Content Kompas memiliki rata-rata "
                f"{kompas_content_avg} kata dan jauh lebih "
                "panjang dibandingkan title dan description."
            ),
            "implikasi_penelitian": (
                "Perbedaan panjang teks mendukung eksperimen "
                "representasi teks bertahap dan kebutuhan "
                "padding atau truncation untuk model "
                "deep learning."
            ),
        },
        {
            "no": 6,
            "aspek": "Keyword extraction",
            "temuan": (
                "Artikel Kompas memiliki beberapa komponen "
                "informasi dengan panjang berbeda, sementara "
                "content merupakan bagian teks paling panjang."
            ),
            "implikasi_penelitian": (
                "Keyword akan diekstraksi menggunakan YAKE "
                "dari gabungan title dan description sebagai "
                "representasi kata kunci tambahan pada "
                "dataset Kompas."
            ),
        },
        {
            "no": 7,
            "aspek": "Representasi teks Kompas",
            "temuan": (
                "Kompas menyediakan title, description, "
                "content, serta keyword yang akan dibentuk "
                "melalui YAKE."
            ),
            "implikasi_penelitian": (
                "Dataset Kompas diuji menggunakan empat "
                "skenario representasi teks bertahap."
            ),
        },
        {
            "no": 8,
            "aspek": "Representasi teks AG News",
            "temuan": (
                f"AG News Train memiliki rata-rata title "
                f"{agnews_title_avg} kata dan description "
                f"{agnews_description_avg} kata."
            ),
            "implikasi_penelitian": (
                "Karena AG News hanya menyediakan title dan "
                "description sebagai komponen teks utama, "
                "dataset benchmark diuji menggunakan dua "
                "skenario representasi teks."
            ),
        },
        {
            "no": 9,
            "aspek": "Missing description AG News Test",
            "temuan": (
                f"Terdapat "
                f"{agnews_test_empty_description:,} "
                f"description kosong pada AG News Test."
            ),
            "implikasi_penelitian": (
                "Nilai description kosong harus ditangani "
                "secara konsisten pada preprocessing tanpa "
                "mengubah label data test asli."
            ),
        },
        {
            "no": 10,
            "aspek": "Perbedaan kosakata antarkategori",
            "temuan": (
                "Analisis word frequency dan word cloud "
                "menunjukkan adanya perbedaan kata dominan "
                "pada setiap kategori berita."
            ),
            "implikasi_penelitian": (
                "Perbedaan pola kosakata menunjukkan bahwa "
                "informasi tekstual memiliki karakteristik "
                "diskriminatif yang dapat dipelajari oleh "
                "CNN dan Attention-BiLSTM."
            ),
        },
        {
            "no": 11,
            "aspek": "Temporal dataset Kompas",
            "temuan": (
                "Tanggal publikasi artikel Kompas mencakup "
                "periode 26 Januari 2026 hingga "
                "25 Mei 2026."
            ),
            "implikasi_penelitian": (
                "Rentang publikasi artikel didokumentasikan "
                "sebagai karakteristik dataset dan dibedakan "
                "dari periode pelaksanaan proses crawling."
            ),
        },
    ]

    return pd.DataFrame(
        findings
    )


# ============================================================
# MENAMPILKAN RINGKASAN
# ============================================================

def print_dataset_summary(
    summary_dataframe: pd.DataFrame,
) -> None:
    """
    Menampilkan ringkasan dataset ke terminal.
    """

    print("\n" + "=" * 72)
    print("RINGKASAN DATASET")
    print("=" * 72)

    display_columns = [
        "dataset",
        "jumlah_data",
        "jumlah_kategori",
    ]

    print(
        summary_dataframe[
            display_columns
        ].to_string(
            index=False
        )
    )


def print_research_findings(
    findings_dataframe: pd.DataFrame,
) -> None:
    """
    Menampilkan temuan utama EDA.
    """

    print("\n" + "=" * 72)
    print("TEMUAN UTAMA EDA")
    print("=" * 72)

    for _, row in findings_dataframe.iterrows():

        print(
            f"\n{row['no']}. "
            f"{row['aspek']}"
        )

        print(
            f"   Temuan     : "
            f"{row['temuan']}"
        )

        print(
            f"   Implikasi  : "
            f"{row['implikasi_penelitian']}"
        )


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main() -> None:

    print("=" * 72)
    print("STEP 3.7 - EDA SUMMARY")
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
    # MEMBUAT RINGKASAN DATASET
    # ========================================================

    summary_rows = [
        create_dataset_summary(
            dataframe=kompas,
            dataset_name="Kompas",
            text_columns=[
                "title",
                "description",
                "content",
            ],
        ),
        create_dataset_summary(
            dataframe=agnews_train,
            dataset_name="AG News Train",
            text_columns=[
                "title",
                "description",
            ],
        ),
        create_dataset_summary(
            dataframe=agnews_test,
            dataset_name="AG News Test",
            text_columns=[
                "title",
                "description",
            ],
        ),
    ]

    summary_dataframe = pd.DataFrame(
        summary_rows
    )

    # ========================================================
    # MEMBUAT TEMUAN PENELITIAN
    # ========================================================

    findings_dataframe = (
        create_research_findings(
            kompas=kompas,
            agnews_train=agnews_train,
            agnews_test=agnews_test,
        )
    )

    # ========================================================
    # MENYIMPAN OUTPUT
    # ========================================================

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_dataframe.to_csv(
        EDA_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    findings_dataframe.to_csv(
        EDA_RESEARCH_FINDINGS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # MENAMPILKAN HASIL
    # ========================================================

    print_dataset_summary(
        summary_dataframe
    )

    print_research_findings(
        findings_dataframe
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print("\n" + "=" * 72)
    print("OUTPUT EDA SUMMARY")
    print("=" * 72)

    print(
        "\nRingkasan karakteristik dataset:"
    )
    print(
        EDA_SUMMARY_PATH
    )

    print(
        "\nTemuan dan implikasi penelitian:"
    )
    print(
        EDA_RESEARCH_FINDINGS_PATH
    )

    print(
        "\nTahap EDA selesai."
    )


if __name__ == "__main__":
    main()
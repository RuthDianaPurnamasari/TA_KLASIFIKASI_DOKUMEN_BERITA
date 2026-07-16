import argparse
import html
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =====================================================
# KONFIGURASI DIREKTORI
# =====================================================

# Struktur:
# TA_KLASIFIKASI_DOKUMEN_BERITA/
# ├── 1_crawling_data/
# │   └── crawl_kompas.py
# └── 2_data/
#     └── raw/

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RAW_DATA_DIR = PROJECT_DIR / "2_data" / "raw"

RAW_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================
# KONFIGURASI CRAWLING
# =====================================================

# Target artikel valid per kategori.
TARGET_DATA = 2500

# Maksimal halaman indeks per tanggal.
MAX_PAGES_PER_DATE = 50

# Jeda permintaan agar tidak terlalu cepat.
ARTICLE_DELAY_SECONDS = 1.5
PAGE_DELAY_SECONDS = 2

# Penyimpanan checkpoint.
CHECKPOINT_INTERVAL = 25

# Minimal kata pada isi artikel.
MIN_CONTENT_WORDS = 30

# Description cadangan dari awal isi artikel.
FALLBACK_DESCRIPTION_WORDS = 40

# Dataset lama mencakup 20–25 Mei 2026.
# Oleh karena itu, crawling tambahan berhenti di 19 Mei 2026.
START_DATE = "2023-01-01"
END_DATE = "2026-05-19"


# =====================================================
# KONFIGURASI KATEGORI
# =====================================================

CATEGORY_CONFIG = {
    "bola": {
        "site": "bola",
        "label": "Bola",
        "old_filename": (
            "kompas_bola_2026-05-20_to_2026-05-25(1).csv"
        ),
    },
    "global": {
        "site": "global",
        "label": "Global",
        "old_filename": (
            "kompas_global_2026-05-20_to_2026-05-25(1).csv"
        ),
    },
    "money": {
        "site": "money",
        "label": "Money",
        "old_filename": (
            "kompas_money_2026-05-20_to_2026-05-25(1).csv"
        ),
    },
    "tekno": {
        "site": "tekno",
        "label": "Tekno",
        "old_filename": (
            "kompas_tekno_2026-05-20_to_2026-05-25(1).csv"
        ),
    },
}


# =====================================================
# HTTP HEADERS
# =====================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/149.0.0.0 Safari/537.36"
    ),
    "Accept-Language": (
        "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7"
    ),
}


# =====================================================
# ARGUMENT TERMINAL
# =====================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Crawling artikel Kompas sampai 2.500 "
            "artikel valid per kategori."
        )
    )

    parser.add_argument(
        "--category",
        required=True,
        choices=list(CATEGORY_CONFIG.keys()),
        help=(
            "Kategori yang dipilih: "
            "bola, global, money, atau tekno."
        ),
    )

    parser.add_argument(
        "--start-date",
        default=START_DATE,
        help="Tanggal awal dalam format YYYY-MM-DD.",
    )

    parser.add_argument(
        "--end-date",
        default=END_DATE,
        help="Tanggal akhir dalam format YYYY-MM-DD.",
    )

    parser.add_argument(
        "--target",
        type=int,
        default=TARGET_DATA,
        help="Target jumlah artikel valid.",
    )

    return parser.parse_args()


# =====================================================
# SESSION DAN RETRY
# =====================================================

def create_session():
    session = requests.Session()
    session.headers.update(HEADERS)

    retry_strategy = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=2,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy
    )

    session.mount(
        "http://",
        adapter,
    )

    session.mount(
        "https://",
        adapter,
    )

    return session


SESSION = create_session()


# =====================================================
# PEMBERSIHAN TEKS
# =====================================================

def clean_text(value):
    """
    Membersihkan tag HTML, karakter HTML,
    baris baru, tab, dan spasi berlebih.
    """

    if value is None:
        return ""

    text = html.unescape(
        str(value)
    )

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def create_fallback_description(
    content,
    max_words=FALLBACK_DESCRIPTION_WORDS,
):
    """
    Membuat description cadangan dari beberapa kata
    awal content.
    """

    words = clean_text(
        content
    ).split()

    if not words:
        return ""

    return " ".join(
        words[:max_words]
    )


# =====================================================
# VALIDASI DAN GENERATE TANGGAL
# =====================================================

def validate_date_range(
    start_date,
    end_date,
):
    try:
        start = datetime.strptime(
            start_date,
            "%Y-%m-%d",
        )

        end = datetime.strptime(
            end_date,
            "%Y-%m-%d",
        )

    except ValueError as error:
        raise ValueError(
            "Format tanggal harus YYYY-MM-DD."
        ) from error

    if start > end:
        raise ValueError(
            "Tanggal awal tidak boleh lebih besar "
            "daripada tanggal akhir."
        )


def generate_dates(
    start_date,
    end_date,
    newest_first=True,
):
    start = datetime.strptime(
        start_date,
        "%Y-%m-%d",
    )

    end = datetime.strptime(
        end_date,
        "%Y-%m-%d",
    )

    dates = []
    current_date = start

    while current_date <= end:
        dates.append(
            current_date.strftime("%Y-%m-%d")
        )

        current_date += timedelta(days=1)

    if newest_first:
        dates.reverse()

    return dates


# =====================================================
# URL INDEKS KOMPAS
# =====================================================

def build_index_url(
    category_site,
    date,
    page,
):
    base_url = "https://indeks.kompas.com/"

    if page == 1:
        return (
            f"{base_url}"
            f"?site={category_site}"
            f"&date={date}"
        )

    return (
        f"{base_url}"
        f"?site={category_site}"
        f"&date={date}"
        f"&page={page}"
    )


# =====================================================
# MEMBACA CSV
# =====================================================

def read_csv_flexible(file_path):
    """
    Membaca CSV menggunakan beberapa kemungkinan
    separator dan encoding.
    """

    if not file_path.exists():
        return pd.DataFrame()

    separators = [
        ";",
        ",",
    ]

    encodings = [
        "utf-8-sig",
        "utf-8",
        "latin-1",
    ]

    for encoding in encodings:
        for separator in separators:
            try:
                dataframe = pd.read_csv(
                    file_path,
                    sep=separator,
                    encoding=encoding,
                    engine="python",
                    on_bad_lines="skip",
                )

                if len(dataframe.columns) > 1:
                    return dataframe

            except (
                UnicodeDecodeError,
                pd.errors.ParserError,
                OSError,
            ):
                continue

    print(
        f"Gagal membaca file: {file_path}"
    )

    return pd.DataFrame()


# =====================================================
# STANDARDISASI DATAFRAME
# =====================================================

def standardize_dataframe(
    dataframe,
    category_label,
):
    """
    Menyamakan struktur dataset lama, checkpoint,
    dan hasil crawling baru.

    Dataset lama yang tidak memiliki description
    akan dibuatkan kolom description kosong.
    """

    required_columns = [
        "title",
        "description",
        "content",
        "date",
        "category",
        "link",
    ]

    if dataframe.empty:
        return pd.DataFrame(
            columns=required_columns
        )

    dataframe = dataframe.copy()

    # Menyeragamkan nama kolom.
    dataframe.columns = [
        clean_text(column)
        .lower()
        .replace(" ", "_")
        for column in dataframe.columns
    ]

    # Menghapus kolom tambahan seperti Unnamed: 0.
    dataframe = dataframe.loc[
        :,
        ~dataframe.columns.str.startswith("unnamed"),
    ]

    column_aliases = {
        "judul": "title",
        "isi": "content",
        "body": "content",
        "tanggal": "date",
        "kategori": "category",
        "url": "link",
        "desc": "description",
        "summary": "description",
    }

    rename_mapping = {}

    for old_name, new_name in column_aliases.items():
        if old_name in dataframe.columns:
            rename_mapping[old_name] = new_name

    dataframe = dataframe.rename(
        columns=rename_mapping
    )

    # Membuat kolom yang tidak tersedia.
    # Dataset lama akan mendapat kolom description kosong.
    for column in required_columns:
        if column not in dataframe.columns:
            dataframe[column] = ""

    # Membersihkan isi seluruh kolom.
    for column in required_columns:
        dataframe[column] = (
            dataframe[column]
            .fillna("")
            .apply(clean_text)
        )

    # Menyeragamkan kategori.
    dataframe["category"] = category_label

    # Menghapus data tanpa title.
    dataframe = dataframe[
        dataframe["title"].str.strip() != ""
    ]

    # Menghapus data tanpa link.
    dataframe = dataframe[
        dataframe["link"].str.strip() != ""
    ]

    # Menghapus duplikat link.
    dataframe = dataframe.drop_duplicates(
        subset=["link"],
        keep="first",
    )

    # Menghapus duplikat title tanpa memperhatikan kapital.
    dataframe["_normalized_title"] = (
        dataframe["title"]
        .str.lower()
        .str.strip()
    )

    dataframe = dataframe.drop_duplicates(
        subset=["_normalized_title"],
        keep="first",
    )

    dataframe = dataframe.drop(
        columns=["_normalized_title"]
    )

    dataframe = dataframe.reset_index(
        drop=True
    )

    return dataframe[required_columns]


# =====================================================
# META CONTENT
# =====================================================

def get_meta_content(
    soup,
    attribute_name,
    attribute_value,
):
    tag = soup.find(
        "meta",
        attrs={
            attribute_name: attribute_value
        },
    )

    if tag and tag.get("content"):
        return clean_text(
            tag.get("content")
        )

    return ""


# =====================================================
# EXTRACT TITLE
# =====================================================

def extract_title(article_soup):
    title_tag = article_soup.find("h1")

    if title_tag:
        title = clean_text(
            title_tag.get_text(
                " ",
                strip=True,
            )
        )

        if title:
            return title

    title = get_meta_content(
        article_soup,
        "property",
        "og:title",
    )

    if title:
        return title

    return get_meta_content(
        article_soup,
        "name",
        "title",
    )


# =====================================================
# EXTRACT DESCRIPTION
# =====================================================

def extract_description(
    article_soup,
    content="",
    allow_fallback=True,
):
    """
    Urutan pengambilan description:

    1. meta name="description"
    2. meta property="og:description"
    3. meta name="twitter:description"
    4. fallback dari content jika diizinkan
    """

    description = get_meta_content(
        article_soup,
        "name",
        "description",
    )

    if not description:
        description = get_meta_content(
            article_soup,
            "property",
            "og:description",
        )

    if not description:
        description = get_meta_content(
            article_soup,
            "name",
            "twitter:description",
        )

    if (
        not description
        and allow_fallback
    ):
        description = create_fallback_description(
            content
        )

    return clean_text(
        description
    )


# =====================================================
# EXTRACT CONTENT
# =====================================================

def extract_content(article_soup):
    selector_candidates = [
        "div.read__content",
        "div.article__body",
        "div.article-content",
        "[data-component-name='ArticleContent']",
        "article",
    ]

    content_div = None

    for selector in selector_candidates:
        candidate = article_soup.select_one(
            selector
        )

        if candidate:
            content_div = candidate
            break

    if content_div is None:
        return ""

    excluded_prefixes = (
        "baca juga",
        "simak breaking news",
        "dapatkan update",
        "ikuti berita",
        "kompas.com",
        "download aplikasi",
    )

    paragraphs = []

    for paragraph in content_div.find_all("p"):
        paragraph_text = clean_text(
            paragraph.get_text(
                " ",
                strip=True,
            )
        )

        if not paragraph_text:
            continue

        if paragraph_text.lower().startswith(
            excluded_prefixes
        ):
            continue

        paragraphs.append(
            paragraph_text
        )

    return clean_text(
        " ".join(paragraphs)
    )


# =====================================================
# EXTRACT TANGGAL
# =====================================================

def extract_date(
    article_soup,
    fallback_date,
):
    date_tag = article_soup.find(
        "div",
        class_="read__time",
    )

    if date_tag:
        date_text = clean_text(
            date_tag.get_text(
                " ",
                strip=True,
            )
        )

        if date_text:
            return date_text

    published_time = get_meta_content(
        article_soup,
        "property",
        "article:published_time",
    )

    if published_time:
        return published_time

    date_published = get_meta_content(
        article_soup,
        "itemprop",
        "datePublished",
    )

    if date_published:
        return date_published

    return fallback_date


# =====================================================
# VALIDASI URL
# =====================================================

def is_valid_article_url(url):
    if not url:
        return False

    url_lower = url.lower()

    if "kompas.com" not in url_lower:
        return False

    invalid_parts = [
        "/search/",
        "/tag/",
        "/video/",
        "/foto/",
        "/galeri/",
        "javascript:",
        "#",
    ]

    for invalid_part in invalid_parts:
        if invalid_part in url_lower:
            return False

    return True


# =====================================================
# SCRAPING ARTIKEL BARU
# =====================================================

def scrape_article(
    link,
    current_date,
    category_label,
):
    """
    Mengambil semua informasi artikel baru:
    title, description, content, date, category, link.
    """

    try:
        response = SESSION.get(
            link,
            timeout=30,
        )

        if response.status_code != 200:
            print(
                "  Gagal membuka artikel. "
                f"Status: {response.status_code}"
            )
            return None

        article_soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        title = extract_title(
            article_soup
        )

        content = extract_content(
            article_soup
        )

        description = extract_description(
            article_soup=article_soup,
            content=content,
            allow_fallback=True,
        )

        date_text = extract_date(
            article_soup,
            current_date,
        )

        if not title:
            print(
                "  Dilewati: title kosong."
            )
            return None

        content_word_count = len(
            content.split()
        )

        if content_word_count < MIN_CONTENT_WORDS:
            print(
                "  Dilewati: content terlalu pendek "
                f"({content_word_count} kata)."
            )
            return None

        if not description:
            print(
                "  Dilewati: description kosong."
            )
            return None

        return {
            "title": title,
            "description": description,
            "content": content,
            "date": date_text,
            "category": category_label,
            "link": link,
        }

    except requests.RequestException as error:
        print(
            f"  Request artikel gagal: {error}"
        )

        return None

    except Exception as error:
        print(
            f"  Error scraping artikel: {error}"
        )

        return None


# =====================================================
# MENGAMBIL DESCRIPTION DATA LAMA
# =====================================================

def fetch_description_from_link(
    link,
    current_content,
):
    """
    Khusus mengambil description data lama.

    Fungsi ini tidak mewajibkan content minimal 30 kata,
    karena hanya digunakan untuk memperbaiki description.
    """

    try:
        response = SESSION.get(
            link,
            timeout=30,
        )

        if response.status_code != 200:
            return create_fallback_description(
                current_content
            )

        article_soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        description = extract_description(
            article_soup=article_soup,
            content="",
            allow_fallback=False,
        )

        if description:
            return description

        return create_fallback_description(
            current_content
        )

    except requests.RequestException:
        return create_fallback_description(
            current_content
        )

    except Exception:
        return create_fallback_description(
            current_content
        )


# =====================================================
# MEMPERBAIKI DESCRIPTION DATA LAMA
# =====================================================

def enrich_old_descriptions(
    dataframe,
    category_label,
    checkpoint_path,
):
    """
    Menambahkan description ke data lama.

    Kondisi yang diperbaiki:
    1. kolom description sebelumnya tidak ada;
    2. description masih kosong;
    3. description merupakan 40 kata awal content
       dari kode lama.
    """

    if dataframe.empty:
        return dataframe

    dataframe = dataframe.copy()

    # Standardisasi sudah membuat kolom description
    # apabila sebelumnya tidak ada.
    if "description" not in dataframe.columns:
        dataframe["description"] = ""

    dataframe["description"] = (
        dataframe["description"]
        .fillna("")
        .apply(clean_text)
    )

    dataframe["content"] = (
        dataframe["content"]
        .fillna("")
        .apply(clean_text)
    )

    def description_needs_update(row):
        description = clean_text(
            row["description"]
        )

        content = clean_text(
            row["content"]
        )

        fallback_description = (
            create_fallback_description(
                content
            )
        )

        # Dataset lama tidak memiliki description.
        if description == "":
            return True

        # Checkpoint lama mungkin sudah terlanjur
        # berisi 40 kata awal content.
        if (
            fallback_description
            and description.lower()
            == fallback_description.lower()
        ):
            return True

        return False

    update_indices = dataframe[
        dataframe.apply(
            description_needs_update,
            axis=1,
        )
    ].index.tolist()

    if not update_indices:
        print(
            "Semua data sudah memiliki description."
        )

        return dataframe

    print(
        "\n======================================"
    )
    print("PENAMBAHAN DESCRIPTION DATA LAMA")
    print(
        "======================================"
    )

    print(
        f"Jumlah data yang akan diperbaiki: "
        f"{len(update_indices)}"
    )

    for order, index in enumerate(
        update_indices,
        start=1,
    ):
        link = clean_text(
            dataframe.at[index, "link"]
        )

        current_content = clean_text(
            dataframe.at[index, "content"]
        )

        print(
            f"\nDescription "
            f"{order}/{len(update_indices)}"
        )

        if not link:
            dataframe.at[
                index,
                "description",
            ] = create_fallback_description(
                current_content
            )

            print(
                "  Link kosong, menggunakan fallback."
            )

        else:
            description = fetch_description_from_link(
                link=link,
                current_content=current_content,
            )

            dataframe.at[
                index,
                "description",
            ] = description

            if description:
                print(
                    "  Description berhasil ditambahkan."
                )
            else:
                print(
                    "  Description tidak ditemukan."
                )

        # Simpan setiap 25 description.
        if (
            order % CHECKPOINT_INTERVAL == 0
        ):
            checkpoint_dataframe = (
                standardize_dataframe(
                    dataframe,
                    category_label,
                )
            )

            checkpoint_dataframe.to_csv(
                checkpoint_path,
                sep=";",
                index=False,
                encoding="utf-8-sig",
            )

            print(
                f"Checkpoint description disimpan: "
                f"{order}/"
                f"{len(update_indices)}"
            )

        time.sleep(
            ARTICLE_DELAY_SECONDS
        )

    result_dataframe = standardize_dataframe(
        dataframe,
        category_label,
    )

    # Fallback terakhir untuk description kosong.
    empty_description_mask = (
        result_dataframe["description"]
        .fillna("")
        .str.strip()
        == ""
    )

    result_dataframe.loc[
        empty_description_mask,
        "description",
    ] = result_dataframe.loc[
        empty_description_mask,
        "content",
    ].apply(
        create_fallback_description
    )

    result_dataframe.to_csv(
        checkpoint_path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nPenambahan description data lama selesai."
    )

    return result_dataframe


# =====================================================
# MEMBACA DATA LAMA ATAU CHECKPOINT
# =====================================================

def read_existing_dataset(
    checkpoint_path,
    output_path,
    old_dataset_path,
    category_label,
):
    """
    Urutan file yang dibaca:

    1. checkpoint;
    2. output final sebelumnya;
    3. dataset lama;
    4. dataframe kosong.
    """

    candidate_files = [
        checkpoint_path,
        output_path,
        old_dataset_path,
    ]

    for file_path in candidate_files:
        if not file_path.exists():
            continue

        dataframe = read_csv_flexible(
            file_path
        )

        if dataframe.empty:
            continue

        print(
            f"Dataset awal dibaca dari:\n{file_path}"
        )

        standardized_dataframe = (
            standardize_dataframe(
                dataframe,
                category_label,
            )
        )

        print(
            f"Jumlah data awal valid: "
            f"{len(standardized_dataframe)}"
        )

        return standardized_dataframe

    print(
        "Dataset lama dan checkpoint tidak ditemukan. "
        "Crawling dimulai dari nol."
    )

    return pd.DataFrame(
        columns=[
            "title",
            "description",
            "content",
            "date",
            "category",
            "link",
        ]
    )


# =====================================================
# MENYIMPAN CHECKPOINT
# =====================================================

def save_checkpoint(
    data,
    checkpoint_path,
    category_label,
):
    checkpoint_dataframe = pd.DataFrame(
        data
    )

    checkpoint_dataframe = standardize_dataframe(
        checkpoint_dataframe,
        category_label,
    )

    empty_description_mask = (
        checkpoint_dataframe["description"]
        .fillna("")
        .str.strip()
        == ""
    )

    checkpoint_dataframe.loc[
        empty_description_mask,
        "description",
    ] = checkpoint_dataframe.loc[
        empty_description_mask,
        "content",
    ].apply(
        create_fallback_description
    )

    checkpoint_dataframe.to_csv(
        checkpoint_path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Checkpoint disimpan: "
        f"{len(checkpoint_dataframe)} data"
    )


# =====================================================
# VALIDASI DATASET AKHIR
# =====================================================

def print_dataset_validation(
    dataframe,
    target,
    output_path,
):
    print(
        "\n======================================"
    )
    print("HASIL CRAWLING")
    print(
        "======================================"
    )

    print(
        f"Target data       : {target}"
    )

    print(
        f"Total data valid  : {len(dataframe)}"
    )

    print(
        f"File output       : {output_path}"
    )

    print(
        "\nJumlah nilai kosong:"
    )

    print(
        dataframe[
            [
                "title",
                "description",
                "content",
                "date",
                "category",
                "link",
            ]
        ]
        .replace("", pd.NA)
        .isna()
        .sum()
    )

    print(
        "\nDuplikat link:",
        dataframe.duplicated(
            subset=["link"]
        ).sum(),
    )

    normalized_titles = (
        dataframe["title"]
        .str.lower()
        .str.strip()
    )

    print(
        "Duplikat title:",
        normalized_titles.duplicated().sum(),
    )

    print(
        "\nDistribusi kategori:"
    )

    print(
        dataframe["category"].value_counts()
    )

    average_words = (
        dataframe["content"]
        .str.split()
        .str.len()
        .mean()
    )

    print(
        "\nRata-rata panjang content:"
    )

    if pd.isna(average_words):
        print("0 kata")
    else:
        print(
            f"{average_words:.2f} kata"
        )

    print(
        "\nContoh data:"
    )

    print(
        dataframe[
            [
                "title",
                "description",
                "category",
            ]
        ].head()
    )


# =====================================================
# PROSES UTAMA CRAWLING
# =====================================================

def crawl_category(
    category_key,
    start_date,
    end_date,
    target,
):
    config = CATEGORY_CONFIG[
        category_key
    ]

    category_site = config["site"]
    category_label = config["label"]

    old_dataset_path = (
        RAW_DATA_DIR
        / config["old_filename"]
    )

    output_path = (
        RAW_DATA_DIR
        / f"kompas_{category_key}_{target}.csv"
    )

    checkpoint_path = (
        RAW_DATA_DIR
        / f"checkpoint_kompas_{category_key}.csv"
    )

    validate_date_range(
        start_date,
        end_date,
    )

    # Membaca checkpoint atau dataset lama.
    existing_dataframe = read_existing_dataset(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        old_dataset_path=old_dataset_path,
        category_label=category_label,
    )

    # Menambahkan description pada dataset lama.
    existing_dataframe = enrich_old_descriptions(
        dataframe=existing_dataframe,
        category_label=category_label,
        checkpoint_path=checkpoint_path,
    )

    existing_dataframe = standardize_dataframe(
        existing_dataframe,
        category_label,
    )

    existing_dataframe = existing_dataframe.head(
        target
    )

    all_data = existing_dataframe.to_dict(
        "records"
    )

    known_links = set(
        existing_dataframe["link"]
        .fillna("")
        .str.strip()
    )

    known_titles = set(
        existing_dataframe["title"]
        .fillna("")
        .str.lower()
        .str.strip()
    )

    print(
        "\n======================================"
    )
    print("KONFIGURASI CRAWLING")
    print(
        "======================================"
    )

    print(
        f"Kategori        : {category_label}"
    )

    print(
        f"Site indeks     : {category_site}"
    )

    print(
        f"Tanggal awal    : {start_date}"
    )

    print(
        f"Tanggal akhir   : {end_date}"
    )

    print(
        f"Target data     : {target}"
    )

    print(
        f"Data awal valid : {len(all_data)}"
    )

    print(
        f"Perlu ditambah  : "
        f"{max(0, target - len(all_data))}"
    )

    # Jika checkpoint sudah mencapai target.
    if len(all_data) >= target:
        final_dataframe = pd.DataFrame(
            all_data
        )

        final_dataframe = standardize_dataframe(
            final_dataframe,
            category_label,
        ).head(target)

        final_dataframe.to_csv(
            output_path,
            sep=";",
            index=False,
            encoding="utf-8-sig",
        )

        print_dataset_validation(
            dataframe=final_dataframe,
            target=target,
            output_path=output_path,
        )

        return final_dataframe

    date_list = generate_dates(
        start_date=start_date,
        end_date=end_date,
        newest_first=True,
    )

    print(
        f"\nJumlah tanggal diperiksa: "
        f"{len(date_list)}"
    )

    # Loop setiap tanggal.
    for current_date in date_list:
        if len(all_data) >= target:
            break

        print(
            "\n======================================"
        )

        print(
            f"TANGGAL  : {current_date}"
        )

        print(
            f"PROGRESS : {len(all_data)}/{target}"
        )

        print(
            "======================================"
        )

        # Loop setiap halaman indeks.
        for page in range(
            1,
            MAX_PAGES_PER_DATE + 1,
        ):
            if len(all_data) >= target:
                break

            page_url = build_index_url(
                category_site=category_site,
                date=current_date,
                page=page,
            )

            print(
                f"\nHalaman {page}"
            )

            print(
                page_url
            )

            try:
                response = SESSION.get(
                    page_url,
                    timeout=30,
                )

            except requests.RequestException as error:
                print(
                    f"Gagal membuka indeks: {error}"
                )

                continue

            if response.status_code != 200:
                print(
                    "Gagal membuka halaman indeks. "
                    f"Status: {response.status_code}"
                )

                continue

            soup = BeautifulSoup(
                response.text,
                "html.parser",
            )

            articles = soup.find_all(
                "div",
                class_="articleItem",
            )

            print(
                f"Artikel ditemukan: {len(articles)}"
            )

            if len(articles) == 0:
                print(
                    "Tidak ada artikel lagi "
                    "pada tanggal ini."
                )

                break

            new_articles_this_page = 0

            # Loop setiap artikel.
            for article_index, article in enumerate(
                articles,
                start=1,
            ):
                if len(all_data) >= target:
                    break

                print(
                    f"\nArtikel "
                    f"{article_index}/{len(articles)}"
                )

                link_tag = article.find(
                    "a",
                    href=True,
                )

                if not link_tag:
                    print(
                        "  Link tidak ditemukan."
                    )

                    continue

                link = clean_text(
                    link_tag.get("href")
                )

                # Menghapus query parameter.
                link = link.split("?")[0]

                if not is_valid_article_url(
                    link
                ):
                    print(
                        "  URL bukan artikel valid."
                    )

                    continue

                if link in known_links:
                    print(
                        "  Link sudah tersedia, dilewati."
                    )

                    continue

                article_data = scrape_article(
                    link=link,
                    current_date=current_date,
                    category_label=category_label,
                )

                if article_data is None:
                    known_links.add(
                        link
                    )

                    continue

                normalized_title = (
                    article_data["title"]
                    .lower()
                    .strip()
                )

                if normalized_title in known_titles:
                    print(
                        "  Judul sudah tersedia, "
                        "dilewati."
                    )

                    known_links.add(
                        link
                    )

                    continue

                all_data.append(
                    article_data
                )

                known_links.add(
                    link
                )

                known_titles.add(
                    normalized_title
                )

                new_articles_this_page += 1

                print(
                    f"  BERHASIL: "
                    f"{len(all_data)}/{target}"
                )

                print(
                    f"  {article_data['title'][:100]}"
                )

                if (
                    len(all_data)
                    % CHECKPOINT_INTERVAL
                    == 0
                ):
                    save_checkpoint(
                        data=all_data,
                        checkpoint_path=checkpoint_path,
                        category_label=category_label,
                    )

                time.sleep(
                    ARTICLE_DELAY_SECONDS
                )

            print(
                f"\nArtikel baru halaman ini: "
                f"{new_articles_this_page}"
            )

            time.sleep(
                PAGE_DELAY_SECONDS
            )

    # Membuat dataframe akhir.
    final_dataframe = pd.DataFrame(
        all_data
    )

    final_dataframe = standardize_dataframe(
        final_dataframe,
        category_label,
    )

    final_dataframe = final_dataframe.head(
        target
    )

    # Mengisi description kosong sebagai pilihan terakhir.
    empty_description_mask = (
        final_dataframe["description"]
        .fillna("")
        .str.strip()
        == ""
    )

    final_dataframe.loc[
        empty_description_mask,
        "description",
    ] = final_dataframe.loc[
        empty_description_mask,
        "content",
    ].apply(
        create_fallback_description
    )

    final_dataframe.to_csv(
        output_path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    final_dataframe.to_csv(
        checkpoint_path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    print_dataset_validation(
        dataframe=final_dataframe,
        target=target,
        output_path=output_path,
    )

    if len(final_dataframe) < target:
        print(
            "\nPERINGATAN:"
        )

        print(
            f"Dataset baru mencapai "
            f"{len(final_dataframe)} dari {target} data."
        )

        print(
            "Perpanjang rentang tanggal jika jumlah "
            "artikel belum mencukupi."
        )

    return final_dataframe


# =====================================================
# MAIN
# =====================================================

def main():
    args = parse_arguments()

    crawl_category(
        category_key=args.category,
        start_date=args.start_date,
        end_date=args.end_date,
        target=args.target,
    )


if __name__ == "__main__":
    main()
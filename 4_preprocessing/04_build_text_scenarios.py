from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# MENAMBAHKAN ROOT PROJECT KE PYTHON PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config import (  # noqa: E402
    PROCESSED_DATA_DIR,
    TABLES_DIR,
)


# ============================================================
# PATH FOLDER SCENARIO
# ============================================================

SCENARIOS_DIR = (
    PROJECT_ROOT
    / "2_data"
    / "scenarios"
)


# ============================================================
# PATH INPUT
# ============================================================

KOMPAS_WITH_KEYWORDS_PATH = (
    PROCESSED_DATA_DIR
    / "kompas_with_keywords.csv"
)

AG_NEWS_TRAIN_PREPROCESSED_PATH = (
    PROCESSED_DATA_DIR
    / "ag_news_train_preprocessed.csv"
)

AG_NEWS_TEST_PREPROCESSED_PATH = (
    PROCESSED_DATA_DIR
    / "ag_news_test_preprocessed.csv"
)


# ============================================================
# PATH OUTPUT KOMPAS
# ============================================================

KOMPAS_K1_PATH = (
    SCENARIOS_DIR
    / "kompas_scenario_k1_title.csv"
)

KOMPAS_K2_PATH = (
    SCENARIOS_DIR
    / "kompas_scenario_k2_title_description.csv"
)

KOMPAS_K3_PATH = (
    SCENARIOS_DIR
    / "kompas_scenario_k3_title_description_keyword.csv"
)

KOMPAS_K4_PATH = (
    SCENARIOS_DIR
    / "kompas_scenario_k4_weighted_news.csv"
)


# ============================================================
# PATH OUTPUT AG NEWS TRAIN
# ============================================================

AGNEWS_TRAIN_A1_PATH = (
    SCENARIOS_DIR
    / "agnews_train_scenario_a1_title.csv"
)

AGNEWS_TRAIN_A2_PATH = (
    SCENARIOS_DIR
    / "agnews_train_scenario_a2_title_description.csv"
)


# ============================================================
# PATH OUTPUT AG NEWS TEST
# ============================================================

AGNEWS_TEST_A1_PATH = (
    SCENARIOS_DIR
    / "agnews_test_scenario_a1_title.csv"
)

AGNEWS_TEST_A2_PATH = (
    SCENARIOS_DIR
    / "agnews_test_scenario_a2_title_description.csv"
)


# ============================================================
# PATH LAPORAN
# ============================================================

SCENARIO_REPORT_PATH = (
    TABLES_DIR
    / "text_scenario_report.csv"
)

SCENARIO_SAMPLES_PATH = (
    TABLES_DIR
    / "text_scenario_samples.csv"
)

SCENARIO_CONFIGURATION_PATH = (
    TABLES_DIR
    / "text_scenario_configuration.json"
)


# ============================================================
# KONFIGURASI
# ============================================================

SEPARATOR = "[SEP]"

TITLE_WEIGHT = 3
KEYWORD_WEIGHT = 2
DESCRIPTION_WEIGHT = 1
CONTENT_WEIGHT = 1


# ============================================================
# MEMBACA DATASET
# ============================================================

def load_dataset(
    file_path: Path,
    dataset_name: str,
    required_columns: list[str],
) -> pd.DataFrame:
    """
    Membaca dataset dan memeriksa kolom wajib.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset {dataset_name} tidak ditemukan:\n"
            f"{file_path}"
        )

    dataframe = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    if dataframe.empty:
        raise ValueError(
            f"Dataset {dataset_name} kosong."
        )

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset {dataset_name} tidak memiliki "
            f"kolom berikut: {missing_columns}"
        )

    return dataframe


# ============================================================
# MEMBERSIHKAN NILAI TEKS
# ============================================================

def safe_text(
    value: Any,
) -> str:
    """
    Mengubah nilai menjadi teks yang aman untuk digabungkan.
    """

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return " ".join(
        str(value)
        .strip()
        .split()
    )


# ============================================================
# MENGGABUNGKAN KOMPONEN TEKS
# ============================================================

def combine_components(
    components: list[Any],
    separator: str = SEPARATOR,
) -> str:
    """
    Menggabungkan komponen teks yang tidak kosong.

    Contoh:
    title [SEP] description
    """

    valid_components = [
        safe_text(component)
        for component in components
        if safe_text(component)
    ]

    return (
        f" {separator} "
        .join(valid_components)
        .strip()
    )


# ============================================================
# MEMBUAT WEIGHTED NEWS REPRESENTATION
# ============================================================

def build_weighted_news_representation(
    title: Any,
    keyword: Any,
    description: Any,
    content: Any,
) -> str:
    """
    Membentuk Weighted News Representation:

    Title       x 3
    Keyword     x 2
    Description x 1
    Content     x 1
    """

    title_text = safe_text(title)
    keyword_text = safe_text(keyword)
    description_text = safe_text(description)
    content_text = safe_text(content)

    weighted_components = []

    # Title x 3
    weighted_components.extend(
        [title_text] * TITLE_WEIGHT
    )

    # Keyword x 2
    weighted_components.extend(
        [keyword_text] * KEYWORD_WEIGHT
    )

    # Description x 1
    weighted_components.extend(
        [description_text] * DESCRIPTION_WEIGHT
    )

    # Content x 1
    weighted_components.extend(
        [content_text] * CONTENT_WEIGHT
    )

    return combine_components(
        weighted_components
    )


# ============================================================
# MENGHITUNG JUMLAH KATA
# ============================================================

def count_words(
    text: Any,
) -> int:
    """
    Menghitung jumlah kata berdasarkan whitespace.
    """

    text = safe_text(text)

    if not text:
        return 0

    return len(
        text.split()
    )


# ============================================================
# MEMBUAT DATASET SKENARIO
# ============================================================

def create_scenario_dataframe(
    source_dataframe: pd.DataFrame,
    scenario_name: str,
    scenario_code: str,
    scenario_text: pd.Series,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Membentuk dataset skenario dengan format standar.
    """

    scenario_dataframe = pd.DataFrame()

    scenario_dataframe["document_id"] = (
        source_dataframe["document_id"]
    )

    if "class_index" in source_dataframe.columns:
        scenario_dataframe["class_index"] = (
            source_dataframe["class_index"]
        )

    scenario_dataframe["category"] = (
        source_dataframe["category"]
    )

    scenario_dataframe["dataset"] = (
        dataset_name
    )

    scenario_dataframe["scenario_code"] = (
        scenario_code
    )

    scenario_dataframe["scenario_name"] = (
        scenario_name
    )

    scenario_dataframe["text"] = (
        scenario_text
        .fillna("")
        .astype(str)
        .str.strip()
    )

    scenario_dataframe["word_count"] = (
        scenario_dataframe["text"]
        .apply(count_words)
    )

    return scenario_dataframe


# ============================================================
# MEMBUAT SKENARIO KOMPAS
# ============================================================

def build_kompas_scenarios(
    dataframe: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Membentuk empat skenario Kompas.
    """

    # --------------------------------------------------------
    # K1 - TITLE
    # --------------------------------------------------------

    k1_text = (
        dataframe["title_preprocessed"]
        .apply(safe_text)
    )

    k1 = create_scenario_dataframe(
        source_dataframe=dataframe,
        scenario_name="Title",
        scenario_code="K1",
        scenario_text=k1_text,
        dataset_name="Kompas",
    )

    # --------------------------------------------------------
    # K2 - TITLE + DESCRIPTION
    # --------------------------------------------------------

    k2_text = dataframe.apply(
        lambda row: combine_components(
            [
                row["title_preprocessed"],
                row["description_preprocessed"],
            ]
        ),
        axis=1,
    )

    k2 = create_scenario_dataframe(
        source_dataframe=dataframe,
        scenario_name="Title + Description",
        scenario_code="K2",
        scenario_text=k2_text,
        dataset_name="Kompas",
    )

    # --------------------------------------------------------
    # K3 - TITLE + DESCRIPTION + KEYWORD
    # --------------------------------------------------------

    k3_text = dataframe.apply(
        lambda row: combine_components(
            [
                row["title_preprocessed"],
                row["description_preprocessed"],
                row["keyword_yake"],
            ]
        ),
        axis=1,
    )

    k3 = create_scenario_dataframe(
        source_dataframe=dataframe,
        scenario_name=(
            "Title + Description + Keyword"
        ),
        scenario_code="K3",
        scenario_text=k3_text,
        dataset_name="Kompas",
    )

    # --------------------------------------------------------
    # K4 - TITLE + DESCRIPTION + KEYWORD + CONTENT
    # --------------------------------------------------------

    k4_text = dataframe.apply(
        lambda row: combine_components(
            [
                row["title_preprocessed"],
                row["description_preprocessed"],
                row["keyword_yake"],
                row["content_preprocessed"],
            ]
        ),
        axis=1,
    )

    k4 = create_scenario_dataframe(
        source_dataframe=dataframe,
        scenario_name=(
            "Title + Description + Keyword + Content"
        ),
        scenario_code="K4",
        scenario_text=k4_text,
        dataset_name="Kompas",
    )

    return {
        "K1": k1,
        "K2": k2,
        "K3": k3,
        "K4": k4,
    }

# ============================================================
# MEMBUAT SKENARIO AG NEWS
# ============================================================

def build_agnews_scenarios(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> dict[str, pd.DataFrame]:
    """
    Membentuk dua skenario AG News.
    """

    # --------------------------------------------------------
    # A1 - TITLE
    # --------------------------------------------------------

    a1_text = (
        dataframe["title_preprocessed"]
        .apply(safe_text)
    )

    a1 = create_scenario_dataframe(
        source_dataframe=dataframe,
        scenario_name="Title",
        scenario_code="A1",
        scenario_text=a1_text,
        dataset_name=dataset_name,
    )

    # --------------------------------------------------------
    # A2 - TITLE + DESCRIPTION
    # --------------------------------------------------------

    a2_text = dataframe.apply(
        lambda row: combine_components(
            [
                row["title_preprocessed"],
                row["description_preprocessed"],
            ]
        ),
        axis=1,
    )

    a2 = create_scenario_dataframe(
        source_dataframe=dataframe,
        scenario_name="Title + Description",
        scenario_code="A2",
        scenario_text=a2_text,
        dataset_name=dataset_name,
    )

    return {
        "A1": a1,
        "A2": a2,
    }


# ============================================================
# VALIDASI SKENARIO
# ============================================================

def validate_scenario(
    dataframe: pd.DataFrame,
    expected_rows: int,
    scenario_code: str,
) -> None:
    """
    Memastikan skenario tidak kehilangan data.
    """

    if len(dataframe) != expected_rows:
        raise ValueError(
            f"Jumlah data skenario {scenario_code} berubah."
        )

    empty_text_count = int(
        dataframe["text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_text_count > 0:
        raise ValueError(
            f"Skenario {scenario_code} memiliki "
            f"{empty_text_count} text kosong."
        )

    duplicate_document_id = int(
        dataframe["document_id"]
        .duplicated()
        .sum()
    )

    if duplicate_document_id > 0:
        raise ValueError(
            f"Skenario {scenario_code} memiliki "
            f"{duplicate_document_id} document_id duplikat."
        )


# ============================================================
# MEMBUAT LAPORAN SKENARIO
# ============================================================

def create_scenario_report(
    scenarios: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Membuat statistik setiap skenario.
    """

    report_rows = []

    for dataframe in scenarios:

        word_counts = (
            dataframe["word_count"]
        )

        report_rows.append(
            {
                "dataset":
                    dataframe["dataset"].iloc[0],
                "scenario_code":
                    dataframe[
                        "scenario_code"
                    ].iloc[0],
                "scenario_name":
                    dataframe[
                        "scenario_name"
                    ].iloc[0],
                "jumlah_data":
                    len(dataframe),
                "avg_word_count":
                    round(
                        float(
                            word_counts.mean()
                        ),
                        2,
                    ),
                "median_word_count":
                    round(
                        float(
                            word_counts.median()
                        ),
                        2,
                    ),
                "min_word_count":
                    int(
                        word_counts.min()
                    ),
                "max_word_count":
                    int(
                        word_counts.max()
                    ),
                "empty_text":
                    int(
                        dataframe["text"]
                        .fillna("")
                        .astype(str)
                        .str.strip()
                        .eq("")
                        .sum()
                    ),
            }
        )

    return pd.DataFrame(
        report_rows
    )


# ============================================================
# MEMBUAT CONTOH SKENARIO
# ============================================================

def create_scenario_samples(
    scenarios: list[pd.DataFrame],
    sample_size: int = 3,
) -> pd.DataFrame:
    """
    Mengambil contoh dari setiap skenario.
    """

    samples = []

    for dataframe in scenarios:

        actual_sample_size = min(
            sample_size,
            len(dataframe),
        )

        sample = (
            dataframe.sample(
                n=actual_sample_size,
                random_state=42,
            )
            [
                [
                    "document_id",
                    "dataset",
                    "scenario_code",
                    "scenario_name",
                    "category",
                    "text",
                    "word_count",
                ]
            ]
            .copy()
        )

        samples.append(
            sample
        )

    return pd.concat(
        samples,
        ignore_index=True,
    )


# ============================================================
# MENYIMPAN KONFIGURASI SKENARIO
# ============================================================

def save_scenario_configuration() -> None:
    """
    Menyimpan desain eksperimen representasi teks.
    """

    configuration = {
        "separator": SEPARATOR,
        "kompas": {
            "K1": {
                "name": "Title",
                "components": [
                    "title_preprocessed"
                ],
            },
            "K2": {
                "name": "Title + Description",
                "components": [
                    "title_preprocessed",
                    "description_preprocessed",
                ],
            },
            "K3": {
                "name": (
                    "Title + Description + Keyword"
                ),
                "components": [
                    "title_preprocessed",
                    "description_preprocessed",
                    "keyword_yake",
                ],
                "keyword_method": "YAKE",
            },
            "K4": {
                "name": (
                    "Weighted News Representation"
                ),
                "weights": {
                    "title": TITLE_WEIGHT,
                    "keyword": KEYWORD_WEIGHT,
                    "description":
                        DESCRIPTION_WEIGHT,
                    "content":
                        CONTENT_WEIGHT,
                },
            },
        },
        "ag_news": {
            "A1": {
                "name": "Title",
                "components": [
                    "title_preprocessed"
                ],
            },
            "A2": {
                "name": "Title + Description",
                "components": [
                    "title_preprocessed",
                    "description_preprocessed",
                ],
            },
        },
    }

    with open(
        SCENARIO_CONFIGURATION_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            ensure_ascii=False,
            indent=4,
        )


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main() -> None:

    print("=" * 72)
    print("STEP 4.4 - BUILD TEXT SCENARIOS")
    print("=" * 72)

    # ========================================================
    # MEMBUAT FOLDER OUTPUT
    # ========================================================

    SCENARIOS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # MEMUAT DATASET
    # ========================================================

    kompas = load_dataset(
        file_path=KOMPAS_WITH_KEYWORDS_PATH,
        dataset_name="Kompas",
        required_columns=[
            "document_id",
            "category",
            "title_preprocessed",
            "description_preprocessed",
            "content_preprocessed",
            "keyword_yake",
        ],
    )

    agnews_train = load_dataset(
        file_path=AG_NEWS_TRAIN_PREPROCESSED_PATH,
        dataset_name="AG News Train",
        required_columns=[
            "document_id",
            "class_index",
            "category",
            "title_preprocessed",
            "description_preprocessed",
        ],
    )

    agnews_test = load_dataset(
        file_path=AG_NEWS_TEST_PREPROCESSED_PATH,
        dataset_name="AG News Test",
        required_columns=[
            "document_id",
            "class_index",
            "category",
            "title_preprocessed",
            "description_preprocessed",
        ],
    )

    # ========================================================
    # MEMBENTUK SKENARIO
    # ========================================================

    print("\nMembentuk 4 skenario Kompas...")

    kompas_scenarios = (
        build_kompas_scenarios(
            kompas
        )
    )

    print(
        "Membentuk 2 skenario AG News Train..."
    )

    agnews_train_scenarios = (
        build_agnews_scenarios(
            dataframe=agnews_train,
            dataset_name="AG News Train",
        )
    )

    print(
        "Membentuk 2 skenario AG News Test..."
    )

    agnews_test_scenarios = (
        build_agnews_scenarios(
            dataframe=agnews_test,
            dataset_name="AG News Test",
        )
    )

    # ========================================================
    # VALIDASI
    # ========================================================

    for code, dataframe in (
        kompas_scenarios.items()
    ):
        validate_scenario(
            dataframe=dataframe,
            expected_rows=len(kompas),
            scenario_code=code,
        )

    for code, dataframe in (
        agnews_train_scenarios.items()
    ):
        validate_scenario(
            dataframe=dataframe,
            expected_rows=len(
                agnews_train
            ),
            scenario_code=(
                f"TRAIN-{code}"
            ),
        )

    for code, dataframe in (
        agnews_test_scenarios.items()
    ):
        validate_scenario(
            dataframe=dataframe,
            expected_rows=len(
                agnews_test
            ),
            scenario_code=(
                f"TEST-{code}"
            ),
        )

    # ========================================================
    # MENYIMPAN DATASET SKENARIO
    # ========================================================

    kompas_scenarios["K1"].to_csv(
        KOMPAS_K1_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    kompas_scenarios["K2"].to_csv(
        KOMPAS_K2_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    kompas_scenarios["K3"].to_csv(
        KOMPAS_K3_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    kompas_scenarios["K4"].to_csv(
        KOMPAS_K4_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    agnews_train_scenarios["A1"].to_csv(
        AGNEWS_TRAIN_A1_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    agnews_train_scenarios["A2"].to_csv(
        AGNEWS_TRAIN_A2_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    agnews_test_scenarios["A1"].to_csv(
        AGNEWS_TEST_A1_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    agnews_test_scenarios["A2"].to_csv(
        AGNEWS_TEST_A2_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # MENGGABUNGKAN SEMUA UNTUK LAPORAN
    # ========================================================

    all_scenarios = [
        *kompas_scenarios.values(),
        *agnews_train_scenarios.values(),
        *agnews_test_scenarios.values(),
    ]

    scenario_report = (
        create_scenario_report(
            all_scenarios
        )
    )

    scenario_samples = (
        create_scenario_samples(
            all_scenarios
        )
    )

    # ========================================================
    # MENYIMPAN LAPORAN
    # ========================================================

    scenario_report.to_csv(
        SCENARIO_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    scenario_samples.to_csv(
        SCENARIO_SAMPLES_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    save_scenario_configuration()

    # ========================================================
    # MENAMPILKAN HASIL
    # ========================================================

    print("\n" + "=" * 72)
    print("HASIL PEMBENTUKAN SKENARIO")
    print("=" * 72)

    print(
        scenario_report.to_string(
            index=False
        )
    )

    print("\n" + "=" * 72)
    print("OUTPUT TEXT SCENARIOS")
    print("=" * 72)

    print("\nFolder dataset skenario:")
    print(SCENARIOS_DIR)

    print("\nLaporan skenario:")
    print(SCENARIO_REPORT_PATH)

    print("\nContoh skenario:")
    print(SCENARIO_SAMPLES_PATH)

    print("\nKonfigurasi skenario:")
    print(SCENARIO_CONFIGURATION_PATH)

    print(
        "\nTahap pembentukan skenario selesai."
    )


if __name__ == "__main__":
    main()
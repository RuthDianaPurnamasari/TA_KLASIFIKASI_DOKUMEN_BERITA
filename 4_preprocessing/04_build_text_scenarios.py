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

# Nama lama dipertahankan agar tidak memutus referensi script
# lain. Isi K4 tetap berupa representasi bertingkat tanpa
# pembobotan manual.


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

YAKE_ABLATION_DESIGN_PATH = (
    TABLES_DIR
    / "yake_ablation_design.csv"
)

YAKE_ABLATION_SAMPLES_PATH = (
    TABLES_DIR
    / "yake_ablation_samples.csv"
)


# ============================================================
# KONFIGURASI
# ============================================================

SEPARATOR = "[SEP]"

RANDOM_SEED = 42

SCENARIO_SAMPLE_SIZE = 3

YAKE_SAMPLE_PER_CATEGORY = 3


# ============================================================
# JUMLAH DATA YANG DIHARAPKAN
# ============================================================

EXPECTED_ROW_COUNTS = {
    "Kompas": 9_997,
    "AG News Train": 119_817,
    "AG News Test": 7_600,
}


# ============================================================
# DEFINISI SKENARIO
# ============================================================

KOMPAS_SCENARIO_DEFINITIONS = {
    "K1": {
        "name": "Title",
        "components": [
            "title_preprocessed",
        ],
        "uses_yake": False,
        "includes_content": False,
        "yake_comparison_role": "not_applicable",
        "comparison_group": "representation_ablation",
    },
    "K2": {
        "name": "Title + Description",
        "components": [
            "title_preprocessed",
            "description_preprocessed",
        ],
        "uses_yake": False,
        "includes_content": False,
        "yake_comparison_role": "baseline_without_yake",
        "comparison_group": "K2_vs_K3_yake_ablation",
    },
    "K3": {
        "name": "Title + Description + Keyword YAKE",
        "components": [
            "title_preprocessed",
            "description_preprocessed",
            "keyword_yake",
        ],
        "uses_yake": True,
        "includes_content": False,
        "yake_comparison_role": "treatment_with_yake",
        "comparison_group": "K2_vs_K3_yake_ablation",
    },
}


AGNEWS_SCENARIO_DEFINITIONS = {
    "A1": {
        "name": "Title",
        "components": [
            "title_preprocessed",
        ],
        "uses_yake": False,
        "includes_content": False,
        "yake_comparison_role": "not_applicable",
        "comparison_group": "agnews_text_ablation",
    },
    "A2": {
        "name": "Title + Description",
        "components": [
            "title_preprocessed",
            "description_preprocessed",
        ],
        "uses_yake": False,
        "includes_content": False,
        "yake_comparison_role": "not_applicable",
        "comparison_group": "agnews_text_ablation",
    },
}


# ============================================================
# MEMBACA DATASET
# ============================================================

def load_dataset(
    file_path: Path,
    dataset_name: str,
    required_columns: list[str],
) -> pd.DataFrame:
    """
    Membaca dataset dan memeriksa kelengkapan input.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset {dataset_name} tidak ditemukan:\n"
            f"{file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Path dataset {dataset_name} bukan file:\n"
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
            f"kolom wajib: {missing_columns}\n"
            f"Kolom tersedia: {list(dataframe.columns)}"
        )

    expected_rows = EXPECTED_ROW_COUNTS[
        dataset_name
    ]

    if len(dataframe) != expected_rows:
        raise ValueError(
            f"Jumlah data {dataset_name} tidak sesuai.\n"
            f"Seharusnya: {expected_rows:,}\n"
            f"Ditemukan : {len(dataframe):,}"
        )

    duplicate_document_ids = int(
        dataframe["document_id"]
        .duplicated()
        .sum()
    )

    if duplicate_document_ids > 0:
        raise ValueError(
            f"Dataset {dataset_name} memiliki "
            f"{duplicate_document_ids:,} document_id "
            "duplikat."
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

    valid_components: list[str] = []

    for component in components:
        component_text = safe_text(
            component
        )

        if component_text:
            valid_components.append(
                component_text
            )

    return (
        f" {separator} "
        .join(valid_components)
        .strip()
    )


def build_scenario_text(
    dataframe: pd.DataFrame,
    component_columns: list[str],
) -> pd.Series:
    """
    Membentuk teks skenario berdasarkan daftar kolom.
    """

    missing_columns = [
        column
        for column in component_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Kolom komponen skenario tidak ditemukan: "
            f"{missing_columns}"
        )

    return dataframe[
        component_columns
    ].apply(
        lambda row: combine_components(
            row.tolist()
        ),
        axis=1,
    )


# ============================================================
# MENGHITUNG JUMLAH KATA
# ============================================================

def count_words(
    value: Any,
) -> int:
    """
    Menghitung token berdasarkan whitespace.

    Token separator [SEP] ikut dihitung karena nantinya
    menjadi bagian dari input model.
    """

    text = safe_text(
        value
    )

    if not text:
        return 0

    return len(
        text.split()
    )


def count_separator_tokens(
    value: Any,
) -> int:
    """
    Menghitung jumlah token separator dalam teks.
    """

    text = safe_text(
        value
    )

    if not text:
        return 0

    return text.split().count(
        SEPARATOR
    )


# ============================================================
# MEMBUAT DATASET SKENARIO
# ============================================================

def create_scenario_dataframe(
    source_dataframe: pd.DataFrame,
    dataset_name: str,
    scenario_code: str,
    scenario_definition: dict,
) -> pd.DataFrame:
    """
    Membentuk dataset skenario dengan metadata yang dapat
    digunakan oleh modeling, evaluation, dan dashboard.
    """

    component_columns = scenario_definition[
        "components"
    ]

    scenario_text = build_scenario_text(
        dataframe=source_dataframe,
        component_columns=component_columns,
    )

    scenario_dataframe = pd.DataFrame(
        index=source_dataframe.index
    )

    identity_columns = [
        "document_id",
        "source_row",
        "split",
        "class_index",
        "category",
    ]

    for column in identity_columns:
        if column in source_dataframe.columns:
            scenario_dataframe[column] = (
                source_dataframe[column]
                .values
            )

    scenario_dataframe["dataset"] = (
        dataset_name
    )

    scenario_dataframe["scenario_code"] = (
        scenario_code
    )

    scenario_dataframe["scenario_name"] = (
        scenario_definition["name"]
    )

    scenario_dataframe["component_signature"] = (
        " + ".join(component_columns)
    )

    scenario_dataframe["uses_yake"] = bool(
        scenario_definition[
            "uses_yake"
        ]
    )

    scenario_dataframe["includes_content"] = bool(
        scenario_definition[
            "includes_content"
        ]
    )

    scenario_dataframe[
        "yake_comparison_role"
    ] = scenario_definition[
        "yake_comparison_role"
    ]

    scenario_dataframe[
        "comparison_group"
    ] = scenario_definition[
        "comparison_group"
    ]

    scenario_dataframe["text"] = (
        scenario_text
        .fillna("")
        .astype(str)
        .str.strip()
        .values
    )

    scenario_dataframe["word_count"] = (
        scenario_dataframe["text"]
        .apply(count_words)
    )

    scenario_dataframe[
        "separator_count"
    ] = (
        scenario_dataframe["text"]
        .apply(count_separator_tokens)
    )

    return scenario_dataframe.reset_index(
        drop=True
    )


# ============================================================
# MEMBENTUK SEMUA SKENARIO
# ============================================================

def build_scenarios(
    source_dataframe: pd.DataFrame,
    dataset_name: str,
    scenario_definitions: dict[str, dict],
) -> dict[str, pd.DataFrame]:
    """
    Membentuk seluruh skenario berdasarkan konfigurasi.
    """

    scenarios: dict[str, pd.DataFrame] = {}

    for scenario_code, definition in (
        scenario_definitions.items()
    ):
        scenarios[scenario_code] = (
            create_scenario_dataframe(
                source_dataframe=source_dataframe,
                dataset_name=dataset_name,
                scenario_code=scenario_code,
                scenario_definition=definition,
            )
        )

    return scenarios


# ============================================================
# VALIDASI SKENARIO
# ============================================================

def validate_scenario(
    source_dataframe: pd.DataFrame,
    scenario_dataframe: pd.DataFrame,
    scenario_code: str,
) -> None:
    """
    Memastikan skenario tidak kehilangan atau mengubah data.
    """

    if len(scenario_dataframe) != len(
        source_dataframe
    ):
        raise ValueError(
            f"Jumlah data skenario {scenario_code} "
            "berubah."
        )

    empty_text_count = int(
        scenario_dataframe["text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_text_count > 0:
        raise ValueError(
            f"Skenario {scenario_code} memiliki "
            f"{empty_text_count:,} teks kosong."
        )

    duplicate_document_id = int(
        scenario_dataframe["document_id"]
        .duplicated()
        .sum()
    )

    if duplicate_document_id > 0:
        raise ValueError(
            f"Skenario {scenario_code} memiliki "
            f"{duplicate_document_id:,} document_id "
            "duplikat."
        )

    source_ids = (
        source_dataframe["document_id"]
        .astype(str)
        .reset_index(drop=True)
    )

    scenario_ids = (
        scenario_dataframe["document_id"]
        .astype(str)
        .reset_index(drop=True)
    )

    if not source_ids.equals(
        scenario_ids
    ):
        raise ValueError(
            f"Urutan document_id pada skenario "
            f"{scenario_code} berubah."
        )

    source_categories = (
        source_dataframe["category"]
        .astype(str)
        .reset_index(drop=True)
    )

    scenario_categories = (
        scenario_dataframe["category"]
        .astype(str)
        .reset_index(drop=True)
    )

    if not source_categories.equals(
        scenario_categories
    ):
        raise ValueError(
            f"Label kategori pada skenario "
            f"{scenario_code} berubah."
        )

    if (
        "class_index" in source_dataframe.columns
        and "class_index" in scenario_dataframe.columns
    ):
        source_labels = (
            source_dataframe["class_index"]
            .astype(str)
            .reset_index(drop=True)
        )

        scenario_labels = (
            scenario_dataframe["class_index"]
            .astype(str)
            .reset_index(drop=True)
        )

        if not source_labels.equals(
            scenario_labels
        ):
            raise ValueError(
                f"Class Index pada skenario "
                f"{scenario_code} berubah."
            )

    invalid_word_count = int(
        scenario_dataframe[
            "word_count"
        ].le(0).sum()
    )

    if invalid_word_count > 0:
        raise ValueError(
            f"Skenario {scenario_code} memiliki "
            f"{invalid_word_count:,} word_count "
            "tidak valid."
        )


# ============================================================
# VALIDASI PASANGAN K2 DAN K3
# ============================================================

def validate_yake_ablation_pair(
    k2_dataframe: pd.DataFrame,
    k3_dataframe: pd.DataFrame,
) -> None:
    """
    Memastikan perbandingan K2 dan K3 adil.

    Satu-satunya komponen tambahan pada K3 harus berupa
    keyword YAKE.
    """

    if len(k2_dataframe) != len(
        k3_dataframe
    ):
        raise ValueError(
            "Jumlah data K2 dan K3 berbeda."
        )

    k2_ids = (
        k2_dataframe["document_id"]
        .astype(str)
        .reset_index(drop=True)
    )

    k3_ids = (
        k3_dataframe["document_id"]
        .astype(str)
        .reset_index(drop=True)
    )

    if not k2_ids.equals(
        k3_ids
    ):
        raise ValueError(
            "Urutan dokumen K2 dan K3 berbeda."
        )

    if bool(
        k2_dataframe["uses_yake"].iloc[0]
    ):
        raise ValueError(
            "K2 seharusnya tidak menggunakan YAKE."
        )

    if not bool(
        k3_dataframe["uses_yake"].iloc[0]
    ):
        raise ValueError(
            "K3 seharusnya menggunakan YAKE."
        )

    expected_k2_signature = (
        "title_preprocessed + "
        "description_preprocessed"
    )

    expected_k3_signature = (
        "title_preprocessed + "
        "description_preprocessed + "
        "keyword_yake"
    )

    actual_k2_signature = (
        k2_dataframe[
            "component_signature"
        ].iloc[0]
    )

    actual_k3_signature = (
        k3_dataframe[
            "component_signature"
        ].iloc[0]
    )

    if actual_k2_signature != (
        expected_k2_signature
    ):
        raise ValueError(
            "Komponen K2 tidak sesuai untuk "
            "baseline YAKE."
        )

    if actual_k3_signature != (
        expected_k3_signature
    ):
        raise ValueError(
            "Komponen K3 tidak sesuai untuk "
            "eksperimen YAKE."
        )

    non_increasing_rows = int(
        (
            k3_dataframe["word_count"]
            <= k2_dataframe["word_count"]
        ).sum()
    )

    if non_increasing_rows > 0:
        raise ValueError(
            f"Ditemukan {non_increasing_rows:,} dokumen "
            "K3 yang tidak lebih panjang dari K2. "
            "Periksa kolom keyword_yake."
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

    report_rows: list[dict] = []

    for dataframe in scenarios:
        word_counts = dataframe[
            "word_count"
        ]

        report_rows.append(
            {
                "dataset": (
                    dataframe["dataset"].iloc[0]
                ),
                "scenario_code": (
                    dataframe[
                        "scenario_code"
                    ].iloc[0]
                ),
                "scenario_name": (
                    dataframe[
                        "scenario_name"
                    ].iloc[0]
                ),
                "component_signature": (
                    dataframe[
                        "component_signature"
                    ].iloc[0]
                ),
                "uses_yake": bool(
                    dataframe[
                        "uses_yake"
                    ].iloc[0]
                ),
                "includes_content": bool(
                    dataframe[
                        "includes_content"
                    ].iloc[0]
                ),
                "yake_comparison_role": (
                    dataframe[
                        "yake_comparison_role"
                    ].iloc[0]
                ),
                "comparison_group": (
                    dataframe[
                        "comparison_group"
                    ].iloc[0]
                ),
                "jumlah_data": len(
                    dataframe
                ),
                "avg_word_count": round(
                    float(
                        word_counts.mean()
                    ),
                    2,
                ),
                "median_word_count": round(
                    float(
                        word_counts.median()
                    ),
                    2,
                ),
                "min_word_count": int(
                    word_counts.min()
                ),
                "max_word_count": int(
                    word_counts.max()
                ),
                "avg_separator_count": round(
                    float(
                        dataframe[
                            "separator_count"
                        ].mean()
                    ),
                    2,
                ),
                "empty_text": int(
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
    sample_size: int = SCENARIO_SAMPLE_SIZE,
) -> pd.DataFrame:
    """
    Mengambil contoh dari setiap skenario.
    """

    samples: list[pd.DataFrame] = []

    for dataframe in scenarios:
        actual_sample_size = min(
            sample_size,
            len(dataframe),
        )

        sample_columns = [
            "document_id",
            "dataset",
            "scenario_code",
            "scenario_name",
            "uses_yake",
            "includes_content",
            "yake_comparison_role",
            "comparison_group",
            "category",
            "text",
            "word_count",
            "separator_count",
        ]

        sample = (
            dataframe
            .sample(
                n=actual_sample_size,
                random_state=RANDOM_SEED,
            )[sample_columns]
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
# MEMBUAT SAMPEL KHUSUS K2 VERSUS K3
# ============================================================

def create_yake_ablation_samples(
    source_dataframe: pd.DataFrame,
    k2_dataframe: pd.DataFrame,
    k3_dataframe: pd.DataFrame,
    samples_per_category: int = (
        YAKE_SAMPLE_PER_CATEGORY
    ),
) -> pd.DataFrame:
    """
    Membuat perbandingan teks K2 dan K3 pada dokumen yang sama.
    """

    sample_ids: list[str] = []

    for _, category_group in (
        source_dataframe.groupby(
            "category",
            sort=True,
            dropna=False,
        )
    ):
        sample_size = min(
            samples_per_category,
            len(category_group),
        )

        selected_ids = (
            category_group
            .sample(
                n=sample_size,
                random_state=RANDOM_SEED,
            )["document_id"]
            .astype(str)
            .tolist()
        )

        sample_ids.extend(
            selected_ids
        )

    source_lookup = (
        source_dataframe
        .set_index("document_id")
    )

    k2_lookup = (
        k2_dataframe
        .set_index("document_id")
    )

    k3_lookup = (
        k3_dataframe
        .set_index("document_id")
    )

    records: list[dict] = []

    for document_id in sample_ids:
        source_row = source_lookup.loc[
            document_id
        ]

        k2_row = k2_lookup.loc[
            document_id
        ]

        k3_row = k3_lookup.loc[
            document_id
        ]

        records.append(
            {
                "document_id": document_id,
                "category": source_row[
                    "category"
                ],
                "title": source_row.get(
                    "title",
                    "",
                ),
                "description": source_row.get(
                    "description",
                    "",
                ),
                "keyword_yake": source_row[
                    "keyword_yake"
                ],
                "k2_without_yake_text": (
                    k2_row["text"]
                ),
                "k3_with_yake_text": (
                    k3_row["text"]
                ),
                "k2_word_count": int(
                    k2_row["word_count"]
                ),
                "k3_word_count": int(
                    k3_row["word_count"]
                ),
                "additional_tokens_from_yake": int(
                    k3_row["word_count"]
                    - k2_row["word_count"]
                ),
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# DESAIN PERBANDINGAN YAKE
# ============================================================

def create_yake_ablation_design() -> pd.DataFrame:
    """
    Membuat metadata perbandingan yang nantinya digunakan
    pada evaluasi dan dashboard.
    """

    return pd.DataFrame(
        [
            {
                "dataset": "Kompas",
                "comparison_id": (
                    "K2_vs_K3_yake_ablation"
                ),
                "baseline_scenario": "K2",
                "baseline_name": (
                    "Title + Description"
                ),
                "baseline_uses_yake": False,
                "treatment_scenario": "K3",
                "treatment_name": (
                    "Title + Description + "
                    "Keyword YAKE"
                ),
                "treatment_uses_yake": True,
                "controlled_components": (
                    "title_preprocessed + "
                    "description_preprocessed"
                ),
                "additional_component": (
                    "keyword_yake"
                ),
                "required_models": (
                    "CNN | Attention-BiLSTM"
                ),
                "evaluation_metrics": (
                    "accuracy | precision_macro | "
                    "recall_macro | f1_macro"
                ),
                "dashboard_chart": (
                    "grouped_bar_chart"
                ),
                "comparison_formula": (
                    "metric_K3 - metric_K2"
                ),
            }
        ]
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
        "separator_is_model_token": True,
        "text_preprocessing": (
            "light preprocessing"
        ),
        "kompas": {
            code: definition
            for code, definition in (
                KOMPAS_SCENARIO_DEFINITIONS
                .items()
            )
        },
        "ag_news": {
            code: definition
            for code, definition in (
                AGNEWS_SCENARIO_DEFINITIONS
                .items()
            )
        },
        "yake_ablation": {
            "comparison_id": (
                "K2_vs_K3_yake_ablation"
            ),
            "baseline": "K2",
            "treatment": "K3",
            "only_difference": (
                "keyword_yake"
            ),
            "models": [
                "CNN",
                "Attention-BiLSTM",
            ],
            "dashboard_metrics": [
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
            ],
        },

        "random_seed": RANDOM_SEED,
        "expected_row_counts": (
            EXPECTED_ROW_COUNTS
        ),
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
    """
    Membentuk skenario representasi teks Kompas dan AG News.
    """

    print("=" * 72)
    print("STEP 4.4 - BUILD TEXT SCENARIOS")
    print("=" * 72)

    SCENARIOS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # MEMUAT DATASET
    # --------------------------------------------------------

    kompas = load_dataset(
        file_path=KOMPAS_WITH_KEYWORDS_PATH,
        dataset_name="Kompas",
        required_columns=[
            "document_id",
            "category",
            "title",
            "description",
            "title_preprocessed",
            "description_preprocessed",
            "keyword_yake",
        ],
    )

    agnews_train = load_dataset(
        file_path=AG_NEWS_TRAIN_PREPROCESSED_PATH,
        dataset_name="AG News Train",
        required_columns=[
            "document_id",
            "source_row",
            "split",
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
            "source_row",
            "split",
            "class_index",
            "category",
            "title_preprocessed",
            "description_preprocessed",
        ],
    )

    # --------------------------------------------------------
    # MEMBENTUK SKENARIO KOMPAS
    # --------------------------------------------------------

    print("\nMembentuk 3 skenario Kompas...")

    kompas_scenarios = build_scenarios(
        source_dataframe=kompas,
        dataset_name="Kompas",
        scenario_definitions=(
            KOMPAS_SCENARIO_DEFINITIONS
        ),
    )

    # --------------------------------------------------------
    # MEMBENTUK SKENARIO AG NEWS
    # --------------------------------------------------------

    print(
        "Membentuk 2 skenario AG News Train..."
    )

    agnews_train_scenarios = build_scenarios(
        source_dataframe=agnews_train,
        dataset_name="AG News Train",
        scenario_definitions=(
            AGNEWS_SCENARIO_DEFINITIONS
        ),
    )

    print(
        "Membentuk 2 skenario AG News Test..."
    )

    agnews_test_scenarios = build_scenarios(
        source_dataframe=agnews_test,
        dataset_name="AG News Test",
        scenario_definitions=(
            AGNEWS_SCENARIO_DEFINITIONS
        ),
    )

    # --------------------------------------------------------
    # VALIDASI SEMUA SKENARIO
    # --------------------------------------------------------

    for code, dataframe in (
        kompas_scenarios.items()
    ):
        validate_scenario(
            source_dataframe=kompas,
            scenario_dataframe=dataframe,
            scenario_code=code,
        )

    for code, dataframe in (
        agnews_train_scenarios.items()
    ):
        validate_scenario(
            source_dataframe=agnews_train,
            scenario_dataframe=dataframe,
            scenario_code=f"TRAIN-{code}",
        )

    for code, dataframe in (
        agnews_test_scenarios.items()
    ):
        validate_scenario(
            source_dataframe=agnews_test,
            scenario_dataframe=dataframe,
            scenario_code=f"TEST-{code}",
        )

    validate_yake_ablation_pair(
        k2_dataframe=(
            kompas_scenarios["K2"]
        ),
        k3_dataframe=(
            kompas_scenarios["K3"]
        ),
    )

    # --------------------------------------------------------
    # MENYIMPAN DATASET SKENARIO
    # --------------------------------------------------------

    output_mapping = {
        KOMPAS_K1_PATH: (
            kompas_scenarios["K1"]
        ),
        KOMPAS_K2_PATH: (
            kompas_scenarios["K2"]
        ),
        KOMPAS_K3_PATH: (
            kompas_scenarios["K3"]
        ),
        AGNEWS_TRAIN_A1_PATH: (
            agnews_train_scenarios["A1"]
        ),
        AGNEWS_TRAIN_A2_PATH: (
            agnews_train_scenarios["A2"]
        ),
        AGNEWS_TEST_A1_PATH: (
            agnews_test_scenarios["A1"]
        ),
        AGNEWS_TEST_A2_PATH: (
            agnews_test_scenarios["A2"]
        ),
    }

    for output_path, dataframe in (
        output_mapping.items()
    ):
        dataframe.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

    # --------------------------------------------------------
    # MEMBUAT LAPORAN
    # --------------------------------------------------------

    all_scenarios = [
        *kompas_scenarios.values(),
        *agnews_train_scenarios.values(),
        *agnews_test_scenarios.values(),
    ]

    scenario_report = create_scenario_report(
        all_scenarios
    )

    scenario_samples = (
        create_scenario_samples(
            all_scenarios
        )
    )

    yake_ablation_design = (
        create_yake_ablation_design()
    )

    yake_ablation_samples = (
        create_yake_ablation_samples(
            source_dataframe=kompas,
            k2_dataframe=(
                kompas_scenarios["K2"]
            ),
            k3_dataframe=(
                kompas_scenarios["K3"]
            ),
        )
    )

    # --------------------------------------------------------
    # MENYIMPAN LAPORAN
    # --------------------------------------------------------

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

    yake_ablation_design.to_csv(
        YAKE_ABLATION_DESIGN_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    yake_ablation_samples.to_csv(
        YAKE_ABLATION_SAMPLES_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    save_scenario_configuration()

    # --------------------------------------------------------
    # MENAMPILKAN HASIL
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("HASIL PEMBENTUKAN SKENARIO")
    print("=" * 72)

    print(
        scenario_report.to_string(
            index=False
        )
    )

    print("\nValidasi perbandingan YAKE:")

    print(
        "Baseline tanpa YAKE : "
        "K2 - Title + Description"
    )

    print(
    "Eksperimen YAKE     : "
    "K3 - Title + Description + Keyword YAKE"
    )

    print(
        "Jumlah dokumen K2   : "
        f"{len(kompas_scenarios['K2']):,}"
    )

    print(
        "Jumlah dokumen K3   : "
        f"{len(kompas_scenarios['K3']):,}"
    )

    avg_additional_tokens = (
        kompas_scenarios["K3"][
            "word_count"
        ]
        - kompas_scenarios["K2"][
            "word_count"
        ]
    ).mean()

    print(
        "Rata-rata tambahan token dari YAKE: "
        f"{avg_additional_tokens:.2f}"
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

    print("\nDesain perbandingan YAKE:")
    print(YAKE_ABLATION_DESIGN_PATH)

    print("\nContoh K2 versus K3:")
    print(YAKE_ABLATION_SAMPLES_PATH)

    print("\nKonfigurasi skenario:")
    print(SCENARIO_CONFIGURATION_PATH)

    print(
        "\nTahap pembentukan skenario selesai."
    )


if __name__ == "__main__":
    main()
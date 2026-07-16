from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# MENAMBAHKAN ROOT PROJECT KE PYTHON PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config import FIGURES_DIR, TABLES_DIR  # noqa: E402


# ============================================================
# FOLDER INPUT
# ============================================================

SPLITS_DIR = (
    PROJECT_ROOT
    / "2_data"
    / "splits"
)


# ============================================================
# PATH INPUT KOMPAS
# ============================================================

KOMPAS_K1_SPLIT_PATH = (
    SPLITS_DIR
    / "kompas_k1_split.csv"
)

KOMPAS_K2_SPLIT_PATH = (
    SPLITS_DIR
    / "kompas_k2_split.csv"
)

KOMPAS_K3_SPLIT_PATH = (
    SPLITS_DIR
    / "kompas_k3_split.csv"
)


# ============================================================
# PATH INPUT AG NEWS
# ============================================================

AGNEWS_A1_TRAIN_VALIDATION_PATH = (
    SPLITS_DIR
    / "agnews_a1_train_validation.csv"
)

AGNEWS_A2_TRAIN_VALIDATION_PATH = (
    SPLITS_DIR
    / "agnews_a2_train_validation.csv"
)


# ============================================================
# PATH OUTPUT
# ============================================================

SEQUENCE_LENGTH_REPORT_PATH = (
    TABLES_DIR
    / "sequence_length_analysis.csv"
)

SEQUENCE_LENGTH_RECOMMENDATION_PATH = (
    TABLES_DIR
    / "sequence_length_recommendation.csv"
)

SEQUENCE_LENGTH_CONFIGURATION_PATH = (
    TABLES_DIR
    / "sequence_length_configuration.json"
)

SEQUENCE_FIGURES_DIR = (
    FIGURES_DIR
    / "sequence_length"
)


# ============================================================
# KONFIGURASI ANALISIS
# ============================================================

ANALYSIS_SPLIT = "train"

PERCENTILES = {
    "p90": 0.90,
    "p95": 0.95,
    "p99": 0.99,
}

# Pembulatan max_length ke kelipatan 10.
ROUNDING_UNIT = 10


# ============================================================
# SPESIFIKASI SKENARIO
# ============================================================

SCENARIO_SPECS = {
    ("Kompas", "K1"): {
        "path": KOMPAS_K1_SPLIT_PATH,
        "expected_dataset_value": "Kompas",
        "expected_total_rows": 9_997,
        "expected_train_rows": 7_997,
        "recommendation_group": "kompas_k1",
        "comparison_purpose": "title_only",
    },
    ("Kompas", "K2"): {
        "path": KOMPAS_K2_SPLIT_PATH,
        "expected_dataset_value": "Kompas",
        "expected_total_rows": 9_997,
        "expected_train_rows": 7_997,
        "recommendation_group": (
            "kompas_k2_k3_yake_ablation"
        ),
        "comparison_purpose": (
            "baseline_without_yake"
        ),
    },
    ("Kompas", "K3"): {
        "path": KOMPAS_K3_SPLIT_PATH,
        "expected_dataset_value": "Kompas",
        "expected_total_rows": 9_997,
        "expected_train_rows": 7_997,
        "recommendation_group": (
            "kompas_k2_k3_yake_ablation"
        ),
        "comparison_purpose": (
            "treatment_with_yake"
        ),
    },
    ("AG News", "A1"): {
        "path": AGNEWS_A1_TRAIN_VALIDATION_PATH,
        "expected_dataset_value": "AG News Train",
        "expected_total_rows": 119_817,
        "expected_train_rows": 107_835,
        "recommendation_group": (
            "agnews_a1_a2_text_ablation"
        ),
        "comparison_purpose": "title_only",
    },
    ("AG News", "A2"): {
        "path": AGNEWS_A2_TRAIN_VALIDATION_PATH,
        "expected_dataset_value": "AG News Train",
        "expected_total_rows": 119_817,
        "expected_train_rows": 107_835,
        "recommendation_group": (
            "agnews_a1_a2_text_ablation"
        ),
        "comparison_purpose": (
            "title_description"
        ),
    },
}


# ============================================================
# DEFINISI KELOMPOK REKOMENDASI
# ============================================================

RECOMMENDATION_GROUPS = {
    "kompas_k1": [
        ("Kompas", "K1"),
    ],
    "kompas_k2_k3_yake_ablation": [
        ("Kompas", "K2"),
        ("Kompas", "K3"),
    ],
    "agnews_a1_a2_text_ablation": [
        ("AG News", "A1"),
        ("AG News", "A2"),
    ],
}


# ============================================================
# KOLOM WAJIB
# ============================================================

REQUIRED_COLUMNS = [
    "document_id",
    "category",
    "dataset",
    "scenario_code",
    "scenario_name",
    "text",
    "word_count",
    "split",
    "uses_yake",
    "comparison_group",
]


# ============================================================
# MEMBERSIHKAN NILAI TEKS
# ============================================================

def safe_text(
    value: Any,
) -> str:
    """
    Mengubah nilai menjadi teks yang aman.
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
# MENGHITUNG PANJANG SEQUENCE SEMENTARA
# ============================================================

def count_sequence_words(
    value: Any,
) -> int:
    """
    Menghitung panjang sequence berdasarkan pemisahan
    whitespace.

    Token [SEP] ikut dihitung sebagai satu token.

    Penghitungan ini sesuai dengan rancangan tahap
    TextVectorization berikutnya:

    standardize=None
    split="whitespace"
    """

    text = safe_text(
        value
    )

    if not text:
        return 0

    return len(
        text.split()
    )


# ============================================================
# MEMBACA DATASET HASIL SPLIT
# ============================================================

def load_split_dataset(
    file_path: Path,
    dataset_name: str,
    scenario_code: str,
    expected_dataset_value: str,
    expected_total_rows: int,
) -> pd.DataFrame:
    """
    Membaca dan memvalidasi dataset skenario hasil split.
    """

    scenario_label = (
        f"{dataset_name} {scenario_code}"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset {scenario_label} "
            f"tidak ditemukan:\n{file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Path dataset {scenario_label} "
            f"bukan file:\n{file_path}"
        )

    dataframe = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    if dataframe.empty:
        raise ValueError(
            f"Dataset {scenario_label} kosong."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom pada {scenario_label} "
            f"tidak lengkap: {missing_columns}\n"
            f"Kolom tersedia: "
            f"{list(dataframe.columns)}"
        )

    if len(dataframe) != expected_total_rows:
        raise ValueError(
            f"Jumlah data {scenario_label} "
            f"tidak sesuai.\n"
            f"Seharusnya: {expected_total_rows:,}\n"
            f"Ditemukan : {len(dataframe):,}"
        )

    actual_dataset_values = set(
        dataframe["dataset"]
        .astype(str)
        .str.strip()
        .unique()
    )

    if actual_dataset_values != {
        expected_dataset_value
    }:
        raise ValueError(
            f"Nilai kolom dataset pada "
            f"{scenario_label} tidak sesuai.\n"
            f"Seharusnya: {expected_dataset_value}\n"
            f"Ditemukan : {actual_dataset_values}"
        )

    actual_scenario_codes = set(
        dataframe["scenario_code"]
        .astype(str)
        .str.strip()
        .unique()
    )

    if actual_scenario_codes != {
        scenario_code
    }:
        raise ValueError(
            f"Scenario code pada "
            f"{scenario_label} tidak sesuai.\n"
            f"Ditemukan: {actual_scenario_codes}"
        )

    duplicate_ids = int(
        dataframe["document_id"]
        .duplicated()
        .sum()
    )

    if duplicate_ids > 0:
        raise ValueError(
            f"Dataset {scenario_label} memiliki "
            f"{duplicate_ids:,} document_id duplikat."
        )

    empty_texts = int(
        dataframe["text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_texts > 0:
        raise ValueError(
            f"Dataset {scenario_label} memiliki "
            f"{empty_texts:,} teks kosong."
        )

    return dataframe


# ============================================================
# MENGAMBIL DATA TRAIN
# ============================================================

def select_train_data(
    dataframe: pd.DataFrame,
    dataset_name: str,
    scenario_code: str,
    expected_train_rows: int,
) -> pd.DataFrame:
    """
    Mengambil hanya data train untuk menentukan
    max_sequence_length.

    Validation dan test tidak digunakan agar tidak terjadi
    kebocoran informasi.
    """

    scenario_label = (
        f"{dataset_name} {scenario_code}"
    )

    train_dataframe = (
        dataframe[
            dataframe["split"]
            .astype(str)
            .str.strip()
            .eq(ANALYSIS_SPLIT)
        ]
        .copy()
        .reset_index(drop=True)
    )

    if train_dataframe.empty:
        raise ValueError(
            f"Data train tidak ditemukan pada "
            f"{scenario_label}."
        )

    if len(train_dataframe) != expected_train_rows:
        raise ValueError(
            f"Jumlah data train {scenario_label} "
            f"tidak sesuai.\n"
            f"Seharusnya: {expected_train_rows:,}\n"
            f"Ditemukan : {len(train_dataframe):,}"
        )

    train_dataframe[
        "sequence_length"
    ] = (
        train_dataframe["text"]
        .apply(count_sequence_words)
    )

    empty_sequences = int(
        train_dataframe[
            "sequence_length"
        ]
        .eq(0)
        .sum()
    )

    if empty_sequences > 0:
        raise ValueError(
            f"Ditemukan {empty_sequences:,} "
            f"sequence kosong pada {scenario_label}."
        )

    stored_word_count = pd.to_numeric(
        train_dataframe["word_count"],
        errors="coerce",
    )

    mismatched_word_count = int(
        stored_word_count
        .ne(
            train_dataframe[
                "sequence_length"
            ]
        )
        .sum()
    )

    if mismatched_word_count > 0:
        raise ValueError(
            f"Ditemukan {mismatched_word_count:,} "
            f"perbedaan antara word_count dan "
            f"sequence_length pada {scenario_label}."
        )

    return train_dataframe


# ============================================================
# PEMBULATAN KE ATAS
# ============================================================

def round_up(
    value: float,
    unit: int = ROUNDING_UNIT,
) -> int:
    """
    Membulatkan nilai ke atas ke kelipatan unit.

    Contoh:
    53 dengan unit 10 menjadi 60.
    """

    if value <= 0:
        raise ValueError(
            "Nilai yang dibulatkan harus lebih dari 0."
        )

    if unit <= 0:
        raise ValueError(
            "Unit pembulatan harus lebih dari 0."
        )

    return int(
        math.ceil(value / unit)
        * unit
    )


# ============================================================
# MEMBUAT STATISTIK PANJANG SEQUENCE
# ============================================================

def create_sequence_statistics(
    dataframe: pd.DataFrame,
    dataset_name: str,
    scenario_code: str,
    recommendation_group: str,
    comparison_purpose: str,
) -> dict:
    """
    Membuat statistik distribusi panjang sequence
    pada data train.
    """

    lengths = dataframe[
        "sequence_length"
    ]

    scenario_name = str(
        dataframe[
            "scenario_name"
        ].iloc[0]
    )

    uses_yake = bool(
        dataframe[
            "uses_yake"
        ].iloc[0]
    )

    statistics = {
        "dataset": dataset_name,
        "scenario_code": scenario_code,
        "scenario_name": scenario_name,
        "uses_yake": uses_yake,
        "recommendation_group": (
            recommendation_group
        ),
        "comparison_purpose": (
            comparison_purpose
        ),
        "split_analyzed": ANALYSIS_SPLIT,
        "jumlah_data_train": len(
            dataframe
        ),
        "minimum": int(
            lengths.min()
        ),
        "mean": round(
            float(
                lengths.mean()
            ),
            4,
        ),
        "median": round(
            float(
                lengths.median()
            ),
            4,
        ),
        "maximum": int(
            lengths.max()
        ),
        "standard_deviation": round(
            float(
                lengths.std()
            ),
            4,
        ),
    }

    for percentile_name, percentile_value in (
        PERCENTILES.items()
    ):
        statistics[percentile_name] = round(
            float(
                lengths.quantile(
                    percentile_value
                )
            ),
            4,
        )

    return statistics


# ============================================================
# MEMBUAT REKOMENDASI PER KELOMPOK
# ============================================================

def create_group_recommendations(
    statistics_lookup: dict[
        tuple[str, str],
        dict
    ],
    train_lookup: dict[
        tuple[str, str],
        pd.DataFrame
    ],
) -> list[dict]:
    """
    Membuat rekomendasi max_length.

    Skenario dalam kelompok perbandingan yang sama memakai
    max_length yang sama.

    Contoh:
    - K2 dan K3 memakai max_length yang sama;
    - A1 dan A2 memakai max_length yang sama.
    """

    recommendation_rows: list[dict] = []

    for group_name, scenario_keys in (
        RECOMMENDATION_GROUPS.items()
    ):
        missing_keys = [
            key
            for key in scenario_keys
            if key not in statistics_lookup
            or key not in train_lookup
        ]

        if missing_keys:
            raise ValueError(
                f"Data kelompok {group_name} "
                f"tidak lengkap: {missing_keys}"
            )

        group_p95_values = [
            float(
                statistics_lookup[key][
                    "p95"
                ]
            )
            for key in scenario_keys
        ]

        group_raw_p95 = max(
            group_p95_values
        )

        shared_max_length = round_up(
            group_raw_p95,
            ROUNDING_UNIT,
        )

        for dataset_name, scenario_code in (
            scenario_keys
        ):
            statistics = statistics_lookup[
                (
                    dataset_name,
                    scenario_code,
                )
            ]

            train_dataframe = train_lookup[
                (
                    dataset_name,
                    scenario_code,
                )
            ]

            lengths = train_dataframe[
                "sequence_length"
            ]

            coverage = float(
                lengths
                .le(shared_max_length)
                .mean()
                * 100
            )

            truncated_count = int(
                lengths
                .gt(shared_max_length)
                .sum()
            )

            truncation_percentage = float(
                lengths
                .gt(shared_max_length)
                .mean()
                * 100
            )

            recommendation_rows.append(
                {
                    "dataset": dataset_name,
                    "scenario_code": (
                        scenario_code
                    ),
                    "scenario_name": (
                        statistics[
                            "scenario_name"
                        ]
                    ),
                    "uses_yake": (
                        statistics[
                            "uses_yake"
                        ]
                    ),
                    "recommendation_group": (
                        group_name
                    ),
                    "comparison_purpose": (
                        statistics[
                            "comparison_purpose"
                        ]
                    ),
                    "selection_basis": (
                        "maximum P95 among scenarios "
                        "in the same comparison group"
                    ),
                    "individual_raw_p95": float(
                        statistics["p95"]
                    ),
                    "group_raw_p95": round(
                        group_raw_p95,
                        4,
                    ),
                    "rounding_unit": (
                        ROUNDING_UNIT
                    ),
                    "recommended_max_length": (
                        shared_max_length
                    ),
                    "estimated_train_coverage": round(
                        coverage,
                        4,
                    ),
                    "jumlah_train_terpotong": (
                        truncated_count
                    ),
                    "persentase_train_terpotong": (
                        round(
                            truncation_percentage,
                            4,
                        )
                    ),
                }
            )

    return recommendation_rows


# ============================================================
# VALIDASI REKOMENDASI KELOMPOK
# ============================================================

def validate_shared_recommendations(
    recommendation_dataframe: pd.DataFrame,
) -> None:
    """
    Memastikan setiap kelompok perbandingan memakai
    max_length yang sama.
    """

    for group_name, group in (
        recommendation_dataframe.groupby(
            "recommendation_group",
            dropna=False,
        )
    ):
        unique_lengths = (
            group[
                "recommended_max_length"
            ]
            .nunique()
        )

        if unique_lengths != 1:
            raise ValueError(
                f"Kelompok {group_name} memiliki "
                "lebih dari satu recommended_max_length."
            )

    kompas_yake_group = (
        recommendation_dataframe[
            recommendation_dataframe[
                "recommendation_group"
            ]
            == "kompas_k2_k3_yake_ablation"
        ]
    )

    expected_codes = {
        "K2",
        "K3",
    }

    actual_codes = set(
        kompas_yake_group[
            "scenario_code"
        ]
        .astype(str)
    )

    if actual_codes != expected_codes:
        raise ValueError(
            "Kelompok perbandingan YAKE "
            "harus berisi K2 dan K3."
        )

    if (
        kompas_yake_group[
            "recommended_max_length"
        ]
        .nunique()
        != 1
    ):
        raise ValueError(
            "K2 dan K3 harus menggunakan "
            "max_length yang sama."
        )


# ============================================================
# MEMBUAT HISTOGRAM
# ============================================================

def plot_sequence_distribution(
    dataframe: pd.DataFrame,
    dataset_name: str,
    scenario_code: str,
    scenario_name: str,
    raw_p95: float,
    recommended_length: int,
    output_path: Path,
) -> None:
    """
    Membuat histogram distribusi panjang sequence
    berdasarkan data train.
    """

    lengths = dataframe[
        "sequence_length"
    ]

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.hist(
        lengths,
        bins=40,
        edgecolor="black",
        alpha=0.8,
    )

    axis.axvline(
        raw_p95,
        linestyle=":",
        linewidth=2,
        label=(
            f"P95 skenario = "
            f"{raw_p95:.2f}"
        ),
    )

    axis.axvline(
        recommended_length,
        linestyle="--",
        linewidth=2,
        label=(
            f"Recommended max length = "
            f"{recommended_length}"
        ),
    )

    axis.set_title(
        (
            "Distribusi Panjang Sequence Data Train\n"
            f"{dataset_name} {scenario_code} — "
            f"{scenario_name}"
        ),
        fontsize=13,
        pad=15,
    )

    axis.set_xlabel(
        "Jumlah Token Berdasarkan Whitespace",
        fontsize=11,
    )

    axis.set_ylabel(
        "Frekuensi Dokumen",
        fontsize=11,
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    axis.legend()

    plt.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# MEMBUAT SEMUA GRAFIK
# ============================================================

def create_all_figures(
    statistics_lookup: dict[
        tuple[str, str],
        dict
    ],
    train_lookup: dict[
        tuple[str, str],
        pd.DataFrame
    ],
    recommendation_dataframe: pd.DataFrame,
) -> None:
    """
    Membuat grafik distribusi untuk setiap skenario.
    """

    recommendation_lookup = {
        (
            str(row.dataset),
            str(row.scenario_code),
        ): row
        for row in (
            recommendation_dataframe
            .itertuples(index=False)
        )
    }

    for scenario_key, statistics in (
        statistics_lookup.items()
    ):
        dataset_name, scenario_code = (
            scenario_key
        )

        recommendation = (
            recommendation_lookup[
                scenario_key
            ]
        )

        safe_dataset_name = (
            dataset_name
            .lower()
            .replace(" ", "_")
        )

        output_path = (
            SEQUENCE_FIGURES_DIR
            / (
                f"{safe_dataset_name}_"
                f"{scenario_code.lower()}_"
                "sequence_length.png"
            )
        )

        plot_sequence_distribution(
            dataframe=(
                train_lookup[
                    scenario_key
                ]
            ),
            dataset_name=dataset_name,
            scenario_code=scenario_code,
            scenario_name=statistics[
                "scenario_name"
            ],
            raw_p95=float(
                statistics["p95"]
            ),
            recommended_length=int(
                recommendation
                .recommended_max_length
            ),
            output_path=output_path,
        )


# ============================================================
# MENYIMPAN KONFIGURASI
# ============================================================

def save_sequence_configuration(
    statistics_dataframe: pd.DataFrame,
    recommendation_dataframe: pd.DataFrame,
) -> None:
    """
    Menyimpan konfigurasi dan rekomendasi sequence
    agar eksperimen dapat direproduksi.
    """

    configuration = {
        "analysis_split": ANALYSIS_SPLIT,
        "data_leakage_control": {
            "train_used_for_analysis": True,
            "validation_used_for_analysis": False,
            "test_used_for_analysis": False,
        },
        "length_measurement": (
            "whitespace-based temporary token count"
        ),
        "future_vectorization_requirement": {
            "standardize": None,
            "split": "whitespace",
            "separator_token": "[SEP]",
            "separator_counted_as_one_token": True,
        },
        "percentiles": PERCENTILES,
        "selection_basis": (
            "Maximum P95 among scenarios in the same "
            "comparison group, rounded upward."
        ),
        "rounding_unit": ROUNDING_UNIT,
        "shared_length_policy": {
            "K1": (
                "Independent recommendation"
            ),
            "K2_and_K3": (
                "Same max_length for fair YAKE ablation"
            ),
            "A1_and_A2": (
                "Same max_length for fair text "
                "representation comparison"
            ),
        },
        "k4_used": False,
        "groups": {},
        "recommendations": {},
    }

    for group_name, group in (
        recommendation_dataframe.groupby(
            "recommendation_group",
            dropna=False,
        )
    ):
        configuration["groups"][
            str(group_name)
        ] = {
            "scenarios": (
                group["scenario_code"]
                .astype(str)
                .tolist()
            ),
            "datasets": (
                group["dataset"]
                .astype(str)
                .tolist()
            ),
            "group_raw_p95": float(
                group[
                    "group_raw_p95"
                ].iloc[0]
            ),
            "recommended_max_length": int(
                group[
                    "recommended_max_length"
                ].iloc[0]
            ),
        }

    for row in (
        recommendation_dataframe
        .itertuples(index=False)
    ):
        recommendation_key = (
            f"{row.dataset}_{row.scenario_code}"
            .lower()
            .replace(" ", "_")
        )

        configuration[
            "recommendations"
        ][recommendation_key] = {
            "dataset": row.dataset,
            "scenario_code": (
                row.scenario_code
            ),
            "scenario_name": (
                row.scenario_name
            ),
            "uses_yake": bool(
                row.uses_yake
            ),
            "recommendation_group": (
                row.recommendation_group
            ),
            "recommended_max_length": int(
                row.recommended_max_length
            ),
            "estimated_train_coverage": float(
                row.estimated_train_coverage
            ),
            "train_truncation_percentage": (
                float(
                    row
                    .persentase_train_terpotong
                )
            ),
        }

    configuration[
        "statistics_row_count"
    ] = len(
        statistics_dataframe
    )

    with open(
        SEQUENCE_LENGTH_CONFIGURATION_PATH,
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
    Menjalankan analisis panjang sequence berdasarkan
    data train.
    """

    print("=" * 72)
    print("STEP 4.6 - SEQUENCE LENGTH ANALYSIS")
    print("=" * 72)

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SEQUENCE_FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    statistics_lookup: dict[
        tuple[str, str],
        dict
    ] = {}

    train_lookup: dict[
        tuple[str, str],
        pd.DataFrame
    ] = {}

    # --------------------------------------------------------
    # MEMPROSES SELURUH SKENARIO
    # --------------------------------------------------------

    for (
        dataset_name,
        scenario_code,
    ), specification in (
        SCENARIO_SPECS.items()
    ):
        print(
            f"\nMenganalisis "
            f"{dataset_name} {scenario_code}..."
        )

        dataframe = load_split_dataset(
            file_path=specification["path"],
            dataset_name=dataset_name,
            scenario_code=scenario_code,
            expected_dataset_value=(
                specification[
                    "expected_dataset_value"
                ]
            ),
            expected_total_rows=(
                specification[
                    "expected_total_rows"
                ]
            ),
        )

        train_dataframe = select_train_data(
            dataframe=dataframe,
            dataset_name=dataset_name,
            scenario_code=scenario_code,
            expected_train_rows=(
                specification[
                    "expected_train_rows"
                ]
            ),
        )

        statistics = create_sequence_statistics(
            dataframe=train_dataframe,
            dataset_name=dataset_name,
            scenario_code=scenario_code,
            recommendation_group=(
                specification[
                    "recommendation_group"
                ]
            ),
            comparison_purpose=(
                specification[
                    "comparison_purpose"
                ]
            ),
        )

        scenario_key = (
            dataset_name,
            scenario_code,
        )

        statistics_lookup[
            scenario_key
        ] = statistics

        train_lookup[
            scenario_key
        ] = train_dataframe

    # --------------------------------------------------------
    # MEMBUAT DATAFRAME STATISTIK
    # --------------------------------------------------------

    statistics_dataframe = pd.DataFrame(
        list(
            statistics_lookup.values()
        )
    )

    # --------------------------------------------------------
    # MEMBUAT REKOMENDASI
    # --------------------------------------------------------

    recommendation_rows = (
        create_group_recommendations(
            statistics_lookup=(
                statistics_lookup
            ),
            train_lookup=train_lookup,
        )
    )

    recommendation_dataframe = pd.DataFrame(
        recommendation_rows
    )

    validate_shared_recommendations(
        recommendation_dataframe
    )

    # --------------------------------------------------------
    # MEMBUAT GRAFIK
    # --------------------------------------------------------

    create_all_figures(
        statistics_lookup=statistics_lookup,
        train_lookup=train_lookup,
        recommendation_dataframe=(
            recommendation_dataframe
        ),
    )

    # --------------------------------------------------------
    # MENYIMPAN OUTPUT
    # --------------------------------------------------------

    statistics_dataframe.to_csv(
        SEQUENCE_LENGTH_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    recommendation_dataframe.to_csv(
        SEQUENCE_LENGTH_RECOMMENDATION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    save_sequence_configuration(
        statistics_dataframe=(
            statistics_dataframe
        ),
        recommendation_dataframe=(
            recommendation_dataframe
        ),
    )

    # --------------------------------------------------------
    # MENAMPILKAN STATISTIK
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("STATISTIK PANJANG SEQUENCE DATA TRAIN")
    print("=" * 72)

    statistics_display_columns = [
        "dataset",
        "scenario_code",
        "scenario_name",
        "uses_yake",
        "jumlah_data_train",
        "mean",
        "median",
        "p90",
        "p95",
        "p99",
        "maximum",
    ]

    print(
        statistics_dataframe[
            statistics_display_columns
        ]
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # MENAMPILKAN REKOMENDASI
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("REKOMENDASI MAX SEQUENCE LENGTH")
    print("=" * 72)

    recommendation_display_columns = [
        "dataset",
        "scenario_code",
        "recommendation_group",
        "individual_raw_p95",
        "group_raw_p95",
        "recommended_max_length",
        "estimated_train_coverage",
        "jumlah_train_terpotong",
        "persentase_train_terpotong",
    ]

    print(
        recommendation_dataframe[
            recommendation_display_columns
        ]
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # VALIDASI PERBANDINGAN
    # --------------------------------------------------------

    print("\nValidasi panjang sequence bersama:")

    k2_k3_recommendations = (
        recommendation_dataframe[
            recommendation_dataframe[
                "scenario_code"
            ]
            .isin(
                [
                    "K2",
                    "K3",
                ]
            )
        ]
    )

    k2_k3_length = int(
        k2_k3_recommendations[
            "recommended_max_length"
        ]
        .iloc[0]
    )

    print(
        "Kompas K2 dan K3 menggunakan "
        f"max_length yang sama: "
        f"{k2_k3_length}"
    )

    a1_a2_recommendations = (
        recommendation_dataframe[
            recommendation_dataframe[
                "scenario_code"
            ]
            .isin(
                [
                    "A1",
                    "A2",
                ]
            )
        ]
    )

    a1_a2_length = int(
        a1_a2_recommendations[
            "recommended_max_length"
        ]
        .iloc[0]
    )

    print(
        "AG News A1 dan A2 menggunakan "
        f"max_length yang sama: "
        f"{a1_a2_length}"
    )

    # --------------------------------------------------------
    # INFORMASI OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("OUTPUT SEQUENCE LENGTH ANALYSIS")
    print("=" * 72)

    print("\nStatistik panjang sequence:")
    print(
        SEQUENCE_LENGTH_REPORT_PATH
    )

    print("\nRekomendasi max sequence:")
    print(
        SEQUENCE_LENGTH_RECOMMENDATION_PATH
    )

    print("\nKonfigurasi sequence:")
    print(
        SEQUENCE_LENGTH_CONFIGURATION_PATH
    )

    print("\nGrafik distribusi:")
    print(
        SEQUENCE_FIGURES_DIR
    )

    print(
        "\nTahap sequence length analysis selesai."
    )


if __name__ == "__main__":
    main()
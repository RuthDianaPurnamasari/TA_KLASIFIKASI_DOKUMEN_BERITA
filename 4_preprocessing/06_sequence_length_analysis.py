from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
# INPUT KOMPAS
# ============================================================

KOMPAS_SPLIT_PATHS = {
    "K1": SPLITS_DIR / "kompas_k1_split.csv",
    "K2": SPLITS_DIR / "kompas_k2_split.csv",
    "K3": SPLITS_DIR / "kompas_k3_split.csv",
    "K4": SPLITS_DIR / "kompas_k4_split.csv",
}


# ============================================================
# INPUT AG NEWS
# ============================================================

AGNEWS_SPLIT_PATHS = {
    "A1": (
        SPLITS_DIR
        / "agnews_a1_train_validation.csv"
    ),
    "A2": (
        SPLITS_DIR
        / "agnews_a2_train_validation.csv"
    ),
}


# ============================================================
# OUTPUT
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
# KONFIGURASI
# ============================================================

ANALYSIS_SPLIT = "train"

PERCENTILES = {
    "p90": 0.90,
    "p95": 0.95,
    "p99": 0.99,
}

# Pembulatan rekomendasi agar mudah digunakan sebagai max_length
ROUNDING_UNIT_SHORT = 10
ROUNDING_UNIT_LONG = 25

# Batas praktis awal untuk eksperimen pada perangkat lokal.
# Nilai ini bukan hasil analisis, hanya pengaman agar sequence
# yang ekstrem tidak langsung digunakan.
PRACTICAL_CAPS = {
    "K1": 50,
    "K2": 100,
    "K3": 150,
    "K4": 600,
    "A1": 50,
    "A2": 200,
}


# ============================================================
# MEMBACA DATASET
# ============================================================

def load_split_dataset(
    file_path: Path,
    dataset_name: str,
    scenario_code: str,
) -> pd.DataFrame:
    """
    Membaca dataset hasil split dan memvalidasi kolom.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset {dataset_name} {scenario_code} "
            f"tidak ditemukan:\n{file_path}"
        )

    dataframe = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    required_columns = [
        "document_id",
        "category",
        "scenario_code",
        "scenario_name",
        "text",
        "split",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom tidak lengkap pada "
            f"{dataset_name} {scenario_code}: "
            f"{missing_columns}"
        )

    if dataframe.empty:
        raise ValueError(
            f"Dataset {dataset_name} "
            f"{scenario_code} kosong."
        )

    return dataframe


# ============================================================
# MENGHITUNG PANJANG TEKS
# ============================================================

def count_sequence_words(
    text: str,
) -> int:
    """
    Menghitung panjang sequence sementara berdasarkan
    pemisahan whitespace.

    Token [SEP] ikut dihitung sebagai satu token karena
    nantinya berfungsi sebagai penanda batas komponen teks.
    """

    normalized_text = str(text).strip()

    if not normalized_text:
        return 0

    return len(
        normalized_text.split()
    )


# ============================================================
# MENGAMBIL DATA TRAIN
# ============================================================

def select_train_data(
    dataframe: pd.DataFrame,
    dataset_name: str,
    scenario_code: str,
) -> pd.DataFrame:
    """
    Mengambil hanya data train untuk analisis panjang sequence.
    """

    train_dataframe = (
        dataframe[
            dataframe["split"]
            == ANALYSIS_SPLIT
        ]
        .copy()
        .reset_index(drop=True)
    )

    if train_dataframe.empty:
        raise ValueError(
            f"Data train tidak ditemukan pada "
            f"{dataset_name} {scenario_code}."
        )

    train_dataframe["sequence_length"] = (
        train_dataframe["text"]
        .apply(count_sequence_words)
    )

    empty_sequences = int(
        train_dataframe["sequence_length"]
        .eq(0)
        .sum()
    )

    if empty_sequences > 0:
        raise ValueError(
            f"Ditemukan {empty_sequences} sequence kosong "
            f"pada {dataset_name} {scenario_code}."
        )

    return train_dataframe


# ============================================================
# PEMBULATAN PANJANG SEQUENCE
# ============================================================

def round_up(
    value: float,
    unit: int,
) -> int:
    """
    Membulatkan nilai ke atas berdasarkan unit tertentu.

    Contoh:
    93 dengan unit 10 menjadi 100.
    487 dengan unit 25 menjadi 500.
    """

    return int(
        math.ceil(value / unit) * unit
    )


# ============================================================
# MEMBUAT STATISTIK
# ============================================================

def create_sequence_statistics(
    dataframe: pd.DataFrame,
    dataset_name: str,
    scenario_code: str,
) -> dict:
    """
    Membuat statistik panjang sequence pada data train.
    """

    lengths = dataframe[
        "sequence_length"
    ]

    scenario_name = dataframe[
        "scenario_name"
    ].iloc[0]

    statistics = {
        "dataset": dataset_name,
        "scenario_code": scenario_code,
        "scenario_name": scenario_name,
        "split_analyzed": ANALYSIS_SPLIT,
        "jumlah_data_train": len(dataframe),
        "minimum": int(lengths.min()),
        "mean": round(
            float(lengths.mean()),
            2,
        ),
        "median": round(
            float(lengths.median()),
            2,
        ),
        "maximum": int(lengths.max()),
        "standard_deviation": round(
            float(lengths.std()),
            2,
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
            2,
        )

    return statistics


# ============================================================
# MEMBUAT REKOMENDASI MAX LENGTH
# ============================================================

def create_length_recommendation(
    statistics: dict,
) -> dict:
    """
    Membuat rekomendasi awal max_sequence_length.

    Rekomendasi utama menggunakan P95 karena mencakup sekitar
    95% data train tanpa harus mengikuti nilai maksimum ekstrem.
    """

    scenario_code = statistics[
        "scenario_code"
    ]

    p95_value = float(
        statistics["p95"]
    )

    rounding_unit = (
        ROUNDING_UNIT_LONG
        if p95_value > 200
        else ROUNDING_UNIT_SHORT
    )

    rounded_p95 = round_up(
        p95_value,
        rounding_unit,
    )

    practical_cap = PRACTICAL_CAPS[
        scenario_code
    ]

    recommended_length = min(
        rounded_p95,
        practical_cap,
    )

    estimated_coverage = None

    return {
        "dataset": statistics["dataset"],
        "scenario_code": scenario_code,
        "scenario_name": statistics[
            "scenario_name"
        ],
        "basis": "P95 data train",
        "raw_p95": p95_value,
        "rounded_p95": rounded_p95,
        "practical_cap": practical_cap,
        "recommended_max_length":
            recommended_length,
        "estimated_coverage": estimated_coverage,
        "note": (
            "Rekomendasi awal. Coverage aktual dihitung "
            "kembali setelah tokenizer difit pada data train."
        ),
    }


# ============================================================
# MENGHITUNG COVERAGE REKOMENDASI
# ============================================================

def add_actual_coverage(
    recommendation: dict,
    train_dataframe: pd.DataFrame,
) -> dict:
    """
    Menghitung persentase data train yang panjangnya tidak
    melebihi rekomendasi max_length.
    """

    max_length = recommendation[
        "recommended_max_length"
    ]

    coverage = (
        train_dataframe[
            "sequence_length"
        ]
        .le(max_length)
        .mean()
        * 100
    )

    truncated_count = int(
        train_dataframe[
            "sequence_length"
        ]
        .gt(max_length)
        .sum()
    )

    recommendation[
        "estimated_coverage"
    ] = round(
        float(coverage),
        2,
    )

    recommendation[
        "jumlah_train_terpotong"
    ] = truncated_count

    recommendation[
        "persentase_train_terpotong"
    ] = round(
        100 - float(coverage),
        2,
    )

    return recommendation


# ============================================================
# MEMBUAT HISTOGRAM
# ============================================================

def plot_sequence_distribution(
    dataframe: pd.DataFrame,
    dataset_name: str,
    scenario_code: str,
    scenario_name: str,
    recommended_length: int,
    output_path: Path,
) -> None:
    """
    Membuat histogram distribusi panjang sequence.
    """

    lengths = dataframe[
        "sequence_length"
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.hist(
        lengths,
        bins=50,
        edgecolor="black",
        alpha=0.8,
    )

    ax.axvline(
        recommended_length,
        linestyle="--",
        linewidth=2,
        label=(
            f"Rekomendasi max length = "
            f"{recommended_length}"
        ),
    )

    ax.set_title(
        (
            f"Distribusi Panjang Sequence "
            f"{dataset_name} {scenario_code}\n"
            f"{scenario_name}"
        ),
        fontsize=13,
        pad=15,
    )

    ax.set_xlabel(
        "Jumlah Token Sementara",
        fontsize=11,
    )

    ax.set_ylabel(
        "Frekuensi",
        fontsize=11,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    ax.legend()

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MEMPROSES SATU SKENARIO
# ============================================================

def analyze_scenario(
    file_path: Path,
    dataset_name: str,
    scenario_code: str,
) -> tuple[dict, dict, pd.DataFrame]:
    """
    Menjalankan analisis satu skenario.
    """

    dataframe = load_split_dataset(
        file_path=file_path,
        dataset_name=dataset_name,
        scenario_code=scenario_code,
    )

    train_dataframe = select_train_data(
        dataframe=dataframe,
        dataset_name=dataset_name,
        scenario_code=scenario_code,
    )

    statistics = create_sequence_statistics(
        dataframe=train_dataframe,
        dataset_name=dataset_name,
        scenario_code=scenario_code,
    )

    recommendation = (
        create_length_recommendation(
            statistics
        )
    )

    recommendation = add_actual_coverage(
        recommendation=recommendation,
        train_dataframe=train_dataframe,
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
            f"sequence_length.png"
        )
    )

    plot_sequence_distribution(
        dataframe=train_dataframe,
        dataset_name=dataset_name,
        scenario_code=scenario_code,
        scenario_name=statistics[
            "scenario_name"
        ],
        recommended_length=(
            recommendation[
                "recommended_max_length"
            ]
        ),
        output_path=output_path,
    )

    return (
        statistics,
        recommendation,
        train_dataframe,
    )


# ============================================================
# MENYIMPAN KONFIGURASI
# ============================================================

def save_sequence_configuration(
    recommendations: pd.DataFrame,
) -> None:
    """
    Menyimpan rekomendasi panjang sequence ke JSON.
    """

    configuration = {
        "analysis_split": ANALYSIS_SPLIT,
        "length_measurement": (
            "Whitespace-based temporary token count"
        ),
        "selection_basis": "P95 data train",
        "important_note": (
            "Nilai akan diverifikasi kembali menggunakan "
            "tokenizer yang difit hanya pada data train."
        ),
        "recommendations": {},
    }

    for row in recommendations.itertuples(
        index=False
    ):
        configuration[
            "recommendations"
        ][row.scenario_code] = {
            "dataset": row.dataset,
            "scenario_name":
                row.scenario_name,
            "recommended_max_length":
                int(
                    row.recommended_max_length
                ),
            "estimated_train_coverage":
                float(
                    row.estimated_coverage
                ),
            "train_truncation_percentage":
                float(
                    row.persentase_train_terpotong
                ),
        }

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
    Menjalankan analisis panjang sequence.
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

    statistics_rows: list[dict] = []
    recommendation_rows: list[dict] = []

    # ========================================================
    # KOMPAS
    # ========================================================

    print("\nMenganalisis skenario Kompas...")

    for scenario_code, file_path in (
        KOMPAS_SPLIT_PATHS.items()
    ):
        (
            statistics,
            recommendation,
            _,
        ) = analyze_scenario(
            file_path=file_path,
            dataset_name="Kompas",
            scenario_code=scenario_code,
        )

        statistics_rows.append(
            statistics
        )

        recommendation_rows.append(
            recommendation
        )

    # ========================================================
    # AG NEWS
    # ========================================================

    print("Menganalisis skenario AG News...")

    for scenario_code, file_path in (
        AGNEWS_SPLIT_PATHS.items()
    ):
        (
            statistics,
            recommendation,
            _,
        ) = analyze_scenario(
            file_path=file_path,
            dataset_name="AG News",
            scenario_code=scenario_code,
        )

        statistics_rows.append(
            statistics
        )

        recommendation_rows.append(
            recommendation
        )

    # ========================================================
    # DATAFRAME HASIL
    # ========================================================

    statistics_dataframe = pd.DataFrame(
        statistics_rows
    )

    recommendation_dataframe = pd.DataFrame(
        recommendation_rows
    )

    # ========================================================
    # MENYIMPAN HASIL
    # ========================================================

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
        recommendation_dataframe
    )

    # ========================================================
    # MENAMPILKAN HASIL
    # ========================================================

    print("\n" + "=" * 72)
    print("STATISTIK PANJANG SEQUENCE DATA TRAIN")
    print("=" * 72)

    display_columns = [
        "dataset",
        "scenario_code",
        "scenario_name",
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
            display_columns
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 72)
    print("REKOMENDASI MAX SEQUENCE LENGTH")
    print("=" * 72)

    recommendation_display_columns = [
        "dataset",
        "scenario_code",
        "raw_p95",
        "rounded_p95",
        "practical_cap",
        "recommended_max_length",
        "estimated_coverage",
        "jumlah_train_terpotong",
        "persentase_train_terpotong",
    ]

    print(
        recommendation_dataframe[
            recommendation_display_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print("\n" + "=" * 72)
    print("OUTPUT SEQUENCE LENGTH ANALYSIS")
    print("=" * 72)

    print("\nStatistik panjang sequence:")
    print(SEQUENCE_LENGTH_REPORT_PATH)

    print("\nRekomendasi max sequence:")
    print(
        SEQUENCE_LENGTH_RECOMMENDATION_PATH
    )

    print("\nKonfigurasi sequence:")
    print(
        SEQUENCE_LENGTH_CONFIGURATION_PATH
    )

    print("\nGrafik distribusi:")
    print(SEQUENCE_FIGURES_DIR)

    print(
        "\nTahap sequence length analysis selesai."
    )


if __name__ == "__main__":
    main()
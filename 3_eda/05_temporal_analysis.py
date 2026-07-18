from __future__ import annotations

import calendar
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# PROJECT PATH
# =============================================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# =============================================================================
# IMPORT PROJECT CONFIGURATION
# =============================================================================

from config import (  # noqa: E402
    FIGURES_DIR,
    TABLES_DIR,
)


# =============================================================================
# DATA DIRECTORY
# =============================================================================

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "2_data"
    / "processed"
)


# =============================================================================
# FINAL DATASET CANDIDATES
# =============================================================================

KOMPAS_FINAL_CANDIDATES = [
    PROCESSED_DATA_DIR / "kompas_clean.csv",
    PROCESSED_DATA_DIR / "kompas_cleaned.csv",
    PROCESSED_DATA_DIR / "kompas_final.csv",
]


# =============================================================================
# OUTPUT TABLES
# =============================================================================

MONTHLY_DISTRIBUTION_PATH = (
    TABLES_DIR
    / "kompas_monthly_distribution.csv"
)

MONTHLY_CATEGORY_DISTRIBUTION_PATH = (
    TABLES_DIR
    / "kompas_monthly_category_distribution.csv"
)

DAILY_DISTRIBUTION_PATH = (
    TABLES_DIR
    / "kompas_daily_distribution.csv"
)

HOURLY_DISTRIBUTION_PATH = (
    TABLES_DIR
    / "kompas_hourly_distribution.csv"
)

WEEKDAY_DISTRIBUTION_PATH = (
    TABLES_DIR
    / "kompas_weekday_distribution.csv"
)

TEMPORAL_SUMMARY_PATH = (
    TABLES_DIR
    / "kompas_temporal_summary.csv"
)


# =============================================================================
# OUTPUT FIGURES
# =============================================================================

MONTHLY_DISTRIBUTION_FIGURE = (
    FIGURES_DIR
    / "kompas_monthly_distribution.png"
)

MONTHLY_CATEGORY_FIGURE = (
    FIGURES_DIR
    / "kompas_monthly_category_distribution.png"
)

DAILY_DISTRIBUTION_FIGURE = (
    FIGURES_DIR
    / "kompas_daily_distribution.png"
)

HOURLY_DISTRIBUTION_FIGURE = (
    FIGURES_DIR
    / "kompas_hourly_distribution.png"
)

WEEKDAY_DISTRIBUTION_FIGURE = (
    FIGURES_DIR
    / "kompas_weekday_distribution.png"
)


# =============================================================================
# EXPECTED FINAL DATASET
# =============================================================================

EXPECTED_FINAL_COUNT = 9_997

REQUIRED_COLUMNS = {
    "document_id",
    "date",
    "category",
}


# =============================================================================
# DAY AND MONTH LABELS
# =============================================================================

INDONESIAN_DAY_NAMES = {
    0: "Senin",
    1: "Selasa",
    2: "Rabu",
    3: "Kamis",
    4: "Jumat",
    5: "Sabtu",
    6: "Minggu",
}

INDONESIAN_MONTH_NAMES = {
    1: "Januari",
    2: "Februari",
    3: "Maret",
    4: "April",
    5: "Mei",
    6: "Juni",
    7: "Juli",
    8: "Agustus",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Desember",
}


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def ensure_output_directories() -> None:
    """
    Memastikan folder tabel dan grafik tersedia.
    """

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def is_valid_file(
    file_path: Path,
) -> bool:
    """
    Memeriksa apakah path merupakan file valid dan tidak kosong.
    """

    path = Path(file_path)

    return (
        path.exists()
        and path.is_file()
        and path.stat().st_size > 0
    )


def resolve_first_existing_file(
    candidates: list[Path],
    dataset_name: str,
) -> Path:
    """
    Memilih file dataset final pertama yang tersedia.
    """

    for candidate in candidates:
        if is_valid_file(candidate):
            return candidate

    paths = "\n".join(
        f"- {candidate}"
        for candidate in candidates
    )

    raise FileNotFoundError(
        f"Dataset final {dataset_name} tidak ditemukan.\n\n"
        f"Path yang diperiksa:\n{paths}\n\n"
        "Pastikan tahap data cleaning telah dijalankan."
    )


def read_csv_with_fallback(
    file_path: Path,
) -> pd.DataFrame:
    """
    Membaca CSV menggunakan beberapa kemungkinan encoding.
    """

    last_error: Exception | None = None

    for encoding in [
        "utf-8-sig",
        "utf-8",
        "latin-1",
    ]:
        try:
            return pd.read_csv(
                file_path,
                encoding=encoding,
            )

        except UnicodeDecodeError as error:
            last_error = error

        except Exception as error:
            last_error = error
            break

    if last_error is not None:
        raise last_error

    return pd.DataFrame()


def validate_required_columns(
    dataframe: pd.DataFrame,
) -> None:
    """
    Memastikan kolom penting tersedia.
    """

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Dataset Kompas tidak memiliki kolom wajib.\n"
            f"Kolom hilang: {sorted(missing_columns)}"
        )


def validate_output_file(
    file_path: Path,
    description: str,
) -> None:
    """
    Memastikan file output berhasil dibuat.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Output {description} tidak berhasil dibuat:\n"
            f"{file_path}"
        )

    if file_path.stat().st_size <= 0:
        raise ValueError(
            f"Output {description} kosong:\n"
            f"{file_path}"
        )


# =============================================================================
# DATASET LOADER
# =============================================================================

def load_kompas_dataset(
    file_path: Path,
) -> pd.DataFrame:
    """
    Membaca dan memvalidasi dataset Kompas final.
    """

    path = Path(file_path)

    if not is_valid_file(path):
        raise FileNotFoundError(
            "Dataset Kompas tidak ditemukan atau kosong:\n"
            f"{path}"
        )

    dataframe = read_csv_with_fallback(
        path
    )

    if dataframe.empty:
        raise ValueError(
            "Dataset Kompas tidak memiliki baris data."
        )

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    validate_required_columns(
        dataframe
    )

    actual_count = int(
        len(dataframe)
    )

    if actual_count != EXPECTED_FINAL_COUNT:
        raise ValueError(
            "Jumlah dataset Kompas final tidak sesuai.\n"
            f"Expected : {EXPECTED_FINAL_COUNT:,}\n"
            f"Actual   : {actual_count:,}\n\n"
            "Pastikan file yang digunakan adalah kompas_clean.csv."
        )

    return dataframe


# =============================================================================
# PREPARE TEMPORAL FEATURES
# =============================================================================

def prepare_temporal_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mengubah kolom date menjadi datetime dan menambahkan
    atribut temporal untuk EDA.

    Timestamp dianalisis sesuai nilai yang tersimpan pada dataset.
    Tidak dilakukan konversi zona waktu.
    """

    result = dataframe.copy()

    result["date_original"] = result["date"]

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    )

    invalid_date_count = int(
        result["date"].isna().sum()
    )

    if invalid_date_count > 0:
        invalid_examples = (
            result.loc[
                result["date"].isna(),
                [
                    "document_id",
                    "date_original",
                ],
            ]
            .head(10)
        )

        raise ValueError(
            f"Ditemukan {invalid_date_count:,} tanggal tidak valid.\n\n"
            f"Contoh:\n{invalid_examples.to_string(index=False)}"
        )

    result["category"] = (
        result["category"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if result["category"].eq("").any():
        raise ValueError(
            "Ditemukan kategori kosong pada dataset Kompas."
        )

    result["publication_date"] = (
        result["date"]
        .dt.normalize()
    )

    result["year"] = (
        result["date"]
        .dt.year
        .astype(int)
    )

    result["month"] = (
        result["date"]
        .dt.month
        .astype(int)
    )

    result["month_period"] = (
        result["date"]
        .dt.to_period("M")
        .astype(str)
    )

    result["month_name"] = (
        result["month"]
        .map(INDONESIAN_MONTH_NAMES)
    )

    result["hour"] = (
        result["date"]
        .dt.hour
        .astype(int)
    )

    result["weekday_number"] = (
        result["date"]
        .dt.dayofweek
        .astype(int)
    )

    result["day_name"] = (
        result["weekday_number"]
        .map(INDONESIAN_DAY_NAMES)
    )

    return result


# =============================================================================
# MONTHLY DISTRIBUTION
# =============================================================================

def create_monthly_distribution(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Menghitung jumlah berita per bulan dan menandai bulan parsial.

    Januari dan Mei pada dataset ini merupakan bulan parsial karena
    periode observasi tidak dimulai pada tanggal 1 dan tidak berakhir
    pada hari terakhir bulan.
    """

    minimum_date = (
        dataframe["publication_date"]
        .min()
    )

    maximum_date = (
        dataframe["publication_date"]
        .max()
    )

    grouped = (
        dataframe
        .groupby(
            "month_period",
            sort=True,
        )
        .agg(
            jumlah_berita=(
                "document_id",
                "size",
            ),
            jumlah_hari_aktif=(
                "publication_date",
                "nunique",
            ),
        )
        .reset_index()
    )

    records: list[dict[str, Any]] = []

    for row in grouped.itertuples(
        index=False
    ):
        period = pd.Period(
            row.month_period,
            freq="M",
        )

        calendar_month_start = (
            period.start_time.normalize()
        )

        calendar_month_end = (
            period.end_time.normalize()
        )

        observation_start = max(
            calendar_month_start,
            minimum_date,
        )

        observation_end = min(
            calendar_month_end,
            maximum_date,
        )

        observed_calendar_days = int(
            (
                observation_end
                - observation_start
            ).days
            + 1
        )

        full_month_days = int(
            calendar.monthrange(
                period.year,
                period.month,
            )[1]
        )

        is_partial_month = bool(
            observation_start
            > calendar_month_start
            or observation_end
            < calendar_month_end
        )

        jumlah_berita = int(
            row.jumlah_berita
        )

        jumlah_hari_aktif = int(
            row.jumlah_hari_aktif
        )

        records.append(
            {
                "month_period":
                    row.month_period,

                "year":
                    int(period.year),

                "month":
                    int(period.month),

                "month_name":
                    INDONESIAN_MONTH_NAMES[
                        period.month
                    ],

                "jumlah_berita":
                    jumlah_berita,

                "persentase_dataset":
                    round(
                        jumlah_berita
                        / len(dataframe)
                        * 100,
                        4,
                    ),

                "tanggal_observasi_awal":
                    observation_start.strftime(
                        "%Y-%m-%d"
                    ),

                "tanggal_observasi_akhir":
                    observation_end.strftime(
                        "%Y-%m-%d"
                    ),

                "jumlah_hari_dalam_bulan":
                    full_month_days,

                "jumlah_hari_dalam_rentang_observasi":
                    observed_calendar_days,

                "jumlah_hari_aktif":
                    jumlah_hari_aktif,

                "cakupan_hari_aktif_persen":
                    round(
                        jumlah_hari_aktif
                        / observed_calendar_days
                        * 100,
                        4,
                    )
                    if observed_calendar_days > 0
                    else 0.0,

                "rata_rata_berita_per_hari_observasi":
                    round(
                        jumlah_berita
                        / observed_calendar_days,
                        4,
                    )
                    if observed_calendar_days > 0
                    else 0.0,

                "rata_rata_berita_per_hari_aktif":
                    round(
                        jumlah_berita
                        / jumlah_hari_aktif,
                        4,
                    )
                    if jumlah_hari_aktif > 0
                    else 0.0,

                "bulan_parsial":
                    is_partial_month,
            }
        )

    return pd.DataFrame(
        records
    )


# =============================================================================
# MONTHLY CATEGORY DISTRIBUTION
# =============================================================================

def create_monthly_category_distribution(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Menghitung jumlah berita per bulan dan kategori.

    Ditambahkan:
    - persentase kategori di dalam bulan;
    - persentase bulan di dalam kategori.
    """

    result = (
        dataframe
        .groupby(
            [
                "month_period",
                "category",
            ],
            sort=True,
        )
        .size()
        .reset_index(
            name="jumlah_berita"
        )
    )

    monthly_totals = (
        result
        .groupby(
            "month_period"
        )["jumlah_berita"]
        .transform("sum")
    )

    category_totals = (
        result
        .groupby(
            "category"
        )["jumlah_berita"]
        .transform("sum")
    )

    result["persentase_dalam_bulan"] = (
        result["jumlah_berita"]
        / monthly_totals
        * 100
    ).round(4)

    result["persentase_dalam_kategori"] = (
        result["jumlah_berita"]
        / category_totals
        * 100
    ).round(4)

    return (
        result
        .sort_values(
            [
                "month_period",
                "category",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# DAILY DISTRIBUTION
# =============================================================================

def create_daily_distribution(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Menghitung jumlah berita setiap tanggal.

    Semua tanggal pada rentang observasi dimunculkan.
    Tanggal tanpa berita diberi jumlah nol.
    """

    start_date = (
        dataframe["publication_date"]
        .min()
    )

    end_date = (
        dataframe["publication_date"]
        .max()
    )

    complete_dates = pd.DataFrame(
        {
            "publication_date":
                pd.date_range(
                    start=start_date,
                    end=end_date,
                    freq="D",
                )
        }
    )

    daily_counts = (
        dataframe
        .groupby(
            "publication_date"
        )
        .size()
        .reset_index(
            name="jumlah_berita"
        )
    )

    result = complete_dates.merge(
        daily_counts,
        on="publication_date",
        how="left",
    )

    result["jumlah_berita"] = (
        result["jumlah_berita"]
        .fillna(0)
        .astype(int)
    )

    result["weekday_number"] = (
        result["publication_date"]
        .dt.dayofweek
        .astype(int)
    )

    result["day_name"] = (
        result["weekday_number"]
        .map(INDONESIAN_DAY_NAMES)
    )

    result["month_period"] = (
        result["publication_date"]
        .dt.to_period("M")
        .astype(str)
    )

    result["is_active_day"] = (
        result["jumlah_berita"]
        .gt(0)
    )

    return result


# =============================================================================
# HOURLY DISTRIBUTION
# =============================================================================

def create_hourly_distribution(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Menghitung jumlah berita berdasarkan jam publikasi.
    """

    result = (
        dataframe
        .groupby(
            "hour"
        )
        .size()
        .reindex(
            range(24),
            fill_value=0,
        )
        .rename(
            "jumlah_berita"
        )
        .reset_index()
    )

    result["rentang_jam"] = (
        result["hour"]
        .map(
            lambda hour: (
                f"{hour:02d}:00–"
                f"{hour:02d}:59"
            )
        )
    )

    result["persentase"] = (
        result["jumlah_berita"]
        / len(dataframe)
        * 100
    ).round(4)

    return result


# =============================================================================
# WEEKDAY DISTRIBUTION
# =============================================================================

def create_weekday_distribution(
    dataframe: pd.DataFrame,
    daily_distribution: pd.DataFrame,
) -> pd.DataFrame:
    """
    Menghitung distribusi berita berdasarkan hari dalam minggu.
    """

    weekday_counts = (
        dataframe
        .groupby(
            "weekday_number"
        )
        .size()
        .reindex(
            range(7),
            fill_value=0,
        )
        .rename(
            "jumlah_berita"
        )
        .reset_index()
    )

    calendar_day_counts = (
        daily_distribution
        .groupby(
            "weekday_number"
        )
        .size()
        .reindex(
            range(7),
            fill_value=0,
        )
        .rename(
            "jumlah_hari_dalam_rentang"
        )
        .reset_index()
    )

    active_day_counts = (
        daily_distribution[
            daily_distribution[
                "is_active_day"
            ]
        ]
        .groupby(
            "weekday_number"
        )
        .size()
        .reindex(
            range(7),
            fill_value=0,
        )
        .rename(
            "jumlah_hari_aktif"
        )
        .reset_index()
    )

    result = (
        weekday_counts
        .merge(
            calendar_day_counts,
            on="weekday_number",
            how="left",
        )
        .merge(
            active_day_counts,
            on="weekday_number",
            how="left",
        )
    )

    result["day_name"] = (
        result["weekday_number"]
        .map(INDONESIAN_DAY_NAMES)
    )

    result["persentase"] = (
        result["jumlah_berita"]
        / len(dataframe)
        * 100
    ).round(4)

    result["rata_rata_berita_per_hari"] = (
        result["jumlah_berita"]
        / result["jumlah_hari_dalam_rentang"]
    ).round(4)

    result["rata_rata_berita_per_hari_aktif"] = (
        result["jumlah_berita"]
        / result["jumlah_hari_aktif"].replace(
            0,
            pd.NA,
        )
    ).round(4)

    return result[
        [
            "weekday_number",
            "day_name",
            "jumlah_berita",
            "persentase",
            "jumlah_hari_dalam_rentang",
            "jumlah_hari_aktif",
            "rata_rata_berita_per_hari",
            "rata_rata_berita_per_hari_aktif",
        ]
    ]


# =============================================================================
# TEMPORAL SUMMARY
# =============================================================================

def create_temporal_summary(
    dataframe: pd.DataFrame,
    monthly_distribution: pd.DataFrame,
    daily_distribution: pd.DataFrame,
    hourly_distribution: pd.DataFrame,
    weekday_distribution: pd.DataFrame,
) -> pd.DataFrame:
    """
    Membuat ringkasan hasil analisis temporal.
    """

    most_active_month = monthly_distribution.loc[
        monthly_distribution[
            "jumlah_berita"
        ].idxmax()
    ]

    highest_monthly_rate = monthly_distribution.loc[
        monthly_distribution[
            "rata_rata_berita_per_hari_observasi"
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

    most_active_weekday = weekday_distribution.loc[
        weekday_distribution[
            "jumlah_berita"
        ].idxmax()
    ]

    start_date = dataframe["date"].min()
    end_date = dataframe["date"].max()

    total_calendar_days = int(
        (
            end_date.normalize()
            - start_date.normalize()
        ).days
        + 1
    )

    active_days = int(
        dataframe["publication_date"]
        .nunique()
    )

    inactive_days = int(
        total_calendar_days
        - active_days
    )

    summary = pd.DataFrame(
        [
            {
                "dataset":
                    "kompas",

                "jumlah_data":
                    int(
                        len(dataframe)
                    ),

                "tanggal_awal":
                    start_date.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "tanggal_akhir":
                    end_date.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "jumlah_hari_dalam_rentang":
                    total_calendar_days,

                "jumlah_hari_aktif":
                    active_days,

                "jumlah_hari_tanpa_berita":
                    inactive_days,

                "cakupan_hari_aktif_persen":
                    round(
                        active_days
                        / total_calendar_days
                        * 100,
                        4,
                    ),

                "rata_rata_berita_per_hari":
                    round(
                        len(dataframe)
                        / total_calendar_days,
                        4,
                    ),

                "rata_rata_berita_per_hari_aktif":
                    round(
                        len(dataframe)
                        / active_days,
                        4,
                    ),

                "bulan_dengan_jumlah_terbanyak":
                    most_active_month[
                        "month_period"
                    ],

                "jumlah_berita_bulan_terbanyak":
                    int(
                        most_active_month[
                            "jumlah_berita"
                        ]
                    ),

                "bulan_dengan_rata_rata_harian_tertinggi":
                    highest_monthly_rate[
                        "month_period"
                    ],

                "rata_rata_harian_bulan_tertinggi":
                    float(
                        highest_monthly_rate[
                            "rata_rata_berita_per_hari_observasi"
                        ]
                    ),

                "tanggal_teraktif":
                    most_active_date[
                        "publication_date"
                    ].strftime(
                        "%Y-%m-%d"
                    ),

                "jumlah_berita_tanggal_teraktif":
                    int(
                        most_active_date[
                            "jumlah_berita"
                        ]
                    ),

                "jam_teraktif":
                    int(
                        most_active_hour[
                            "hour"
                        ]
                    ),

                "rentang_jam_teraktif":
                    most_active_hour[
                        "rentang_jam"
                    ],

                "jumlah_berita_jam_teraktif":
                    int(
                        most_active_hour[
                            "jumlah_berita"
                        ]
                    ),

                "hari_teraktif":
                    most_active_weekday[
                        "day_name"
                    ],

                "jumlah_berita_hari_teraktif":
                    int(
                        most_active_weekday[
                            "jumlah_berita"
                        ]
                    ),
            }
        ]
    )

    return summary


# =============================================================================
# CONSISTENCY VALIDATION
# =============================================================================

def validate_analysis_totals(
    dataframe: pd.DataFrame,
    monthly_distribution: pd.DataFrame,
    monthly_category_distribution: pd.DataFrame,
    daily_distribution: pd.DataFrame,
    hourly_distribution: pd.DataFrame,
    weekday_distribution: pd.DataFrame,
) -> None:
    """
    Memastikan seluruh agregasi menghasilkan total yang sama
    dengan jumlah dataset final.
    """

    expected_total = int(
        len(dataframe)
    )

    calculated_totals = {
        "monthly_distribution":
            int(
                monthly_distribution[
                    "jumlah_berita"
                ].sum()
            ),

        "monthly_category_distribution":
            int(
                monthly_category_distribution[
                    "jumlah_berita"
                ].sum()
            ),

        "daily_distribution":
            int(
                daily_distribution[
                    "jumlah_berita"
                ].sum()
            ),

        "hourly_distribution":
            int(
                hourly_distribution[
                    "jumlah_berita"
                ].sum()
            ),

        "weekday_distribution":
            int(
                weekday_distribution[
                    "jumlah_berita"
                ].sum()
            ),
    }

    errors = [
        (
            f"{name}: "
            f"{actual:,} != {expected_total:,}"
        )
        for name, actual
        in calculated_totals.items()
        if actual != expected_total
    ]

    if errors:
        raise ValueError(
            "Validasi total analisis temporal gagal:\n- "
            + "\n- ".join(errors)
        )


# =============================================================================
# FIGURE: MONTHLY DISTRIBUTION
# =============================================================================

def plot_monthly_distribution(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Membuat grafik jumlah berita per bulan.
    """

    figure, axis = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    bars = axis.bar(
        dataframe["month_period"],
        dataframe["jumlah_berita"],
    )

    axis.set_title(
        "Distribusi Berita Kompas per Bulan",
        fontsize=14,
        pad=15,
    )

    axis.set_xlabel(
        "Bulan",
        fontsize=11,
    )

    axis.set_ylabel(
        "Jumlah Berita",
        fontsize=11,
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    axis.set_axisbelow(
        True
    )

    maximum_value = int(
        dataframe["jumlah_berita"]
        .max()
    )

    axis.set_ylim(
        0,
        max(
            1,
            int(
                maximum_value
                * 1.15
            ),
        ),
    )

    for bar, row in zip(
        bars,
        dataframe.itertuples(
            index=False
        ),
    ):
        partial_label = (
            "\n(parsial)"
            if row.bulan_parsial
            else ""
        )

        axis.text(
            bar.get_x()
            + bar.get_width()
            / 2,
            bar.get_height()
            + maximum_value
            * 0.015,
            (
                f"{row.jumlah_berita:,}"
                f"{partial_label}"
            ),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axis.text(
        0.01,
        -0.16,
        (
            "Catatan: Januari dan Mei merupakan bulan parsial "
            "sesuai rentang pengumpulan data."
        ),
        transform=axis.transAxes,
        fontsize=9,
    )

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# =============================================================================
# FIGURE: MONTHLY CATEGORY DISTRIBUTION
# =============================================================================

def plot_monthly_category_distribution(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Membuat line chart kategori berita per bulan.
    """

    pivot_table = (
        dataframe
        .pivot(
            index="month_period",
            columns="category",
            values="jumlah_berita",
        )
        .fillna(0)
        .sort_index()
    )

    figure, axis = plt.subplots(
        figsize=(
            11,
            7,
        )
    )

    for category in pivot_table.columns:
        axis.plot(
            pivot_table.index,
            pivot_table[category],
            marker="o",
            linewidth=2,
            label=str(category).title(),
        )

    axis.set_title(
        "Distribusi Kategori Berita Kompas per Bulan",
        fontsize=14,
        pad=15,
    )

    axis.set_xlabel(
        "Bulan",
        fontsize=11,
    )

    axis.set_ylabel(
        "Jumlah Berita",
        fontsize=11,
    )

    axis.legend(
        title="Kategori"
    )

    axis.grid(
        linestyle="--",
        alpha=0.4,
    )

    axis.text(
        0.01,
        -0.14,
        (
            "Catatan: Januari dan Mei tidak mencakup "
            "satu bulan kalender penuh."
        ),
        transform=axis.transAxes,
        fontsize=9,
    )

    plt.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# =============================================================================
# FIGURE: DAILY DISTRIBUTION
# =============================================================================

def plot_daily_distribution(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Membuat grafik jumlah berita harian.
    """

    figure, axis = plt.subplots(
        figsize=(
            14,
            6,
        )
    )

    axis.plot(
        dataframe["publication_date"],
        dataframe["jumlah_berita"],
        linewidth=1.5,
    )

    mean_value = float(
        dataframe["jumlah_berita"]
        .mean()
    )

    axis.axhline(
        mean_value,
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Rata-rata = {mean_value:.2f}"
        ),
    )

    axis.set_title(
        "Distribusi Harian Berita Kompas",
        fontsize=14,
        pad=15,
    )

    axis.set_xlabel(
        "Tanggal",
        fontsize=11,
    )

    axis.set_ylabel(
        "Jumlah Berita",
        fontsize=11,
    )

    axis.grid(
        linestyle="--",
        alpha=0.4,
    )

    axis.legend()

    figure.autofmt_xdate()

    plt.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# =============================================================================
# FIGURE: HOURLY DISTRIBUTION
# =============================================================================

def plot_hourly_distribution(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Membuat grafik distribusi jam publikasi.
    """

    figure, axis = plt.subplots(
        figsize=(
            12,
            6,
        )
    )

    bars = axis.bar(
        dataframe["hour"],
        dataframe["jumlah_berita"],
    )

    axis.set_title(
        "Distribusi Jam Publikasi Berita Kompas",
        fontsize=14,
        pad=15,
    )

    axis.set_xlabel(
        "Jam Publikasi",
        fontsize=11,
    )

    axis.set_ylabel(
        "Jumlah Berita",
        fontsize=11,
    )

    axis.set_xticks(
        range(24)
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    axis.set_axisbelow(
        True
    )

    maximum_value = int(
        dataframe["jumlah_berita"]
        .max()
    )

    axis.set_ylim(
        0,
        max(
            1,
            int(
                maximum_value
                * 1.18
            ),
        ),
    )

    for bar, value in zip(
        bars,
        dataframe["jumlah_berita"],
    ):
        axis.text(
            bar.get_x()
            + bar.get_width()
            / 2,
            bar.get_height()
            + maximum_value
            * 0.01,
            f"{int(value):,}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    plt.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# =============================================================================
# FIGURE: WEEKDAY DISTRIBUTION
# =============================================================================

def plot_weekday_distribution(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Membuat grafik jumlah berita berdasarkan hari.
    """

    figure, axis = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    bars = axis.bar(
        dataframe["day_name"],
        dataframe["jumlah_berita"],
    )

    axis.set_title(
        "Distribusi Berita Kompas Berdasarkan Hari",
        fontsize=14,
        pad=15,
    )

    axis.set_xlabel(
        "Hari",
        fontsize=11,
    )

    axis.set_ylabel(
        "Jumlah Berita",
        fontsize=11,
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    axis.set_axisbelow(
        True
    )

    maximum_value = int(
        dataframe["jumlah_berita"]
        .max()
    )

    axis.set_ylim(
        0,
        max(
            1,
            int(
                maximum_value
                * 1.12
            ),
        ),
    )

    for bar, value in zip(
        bars,
        dataframe["jumlah_berita"],
    ):
        axis.text(
            bar.get_x()
            + bar.get_width()
            / 2,
            bar.get_height()
            + maximum_value
            * 0.01,
            f"{int(value):,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main() -> None:
    """
    Menjalankan EDA tahap 3.5 — Temporal Analysis.
    """

    print(
        "=" * 80
    )

    print(
        "STEP 3.5 - TEMPORAL ANALYSIS"
    )

    print(
        "=" * 80
    )

    ensure_output_directories()

    # =========================================================================
    # RESOLVE FINAL DATASET
    # =========================================================================

    kompas_path = resolve_first_existing_file(
        candidates=KOMPAS_FINAL_CANDIDATES,
        dataset_name="Kompas",
    )

    print(
        "\nDataset final yang digunakan:"
    )

    print(
        f"Kompas : {kompas_path}"
    )

    # =========================================================================
    # LOAD AND PREPARE DATASET
    # =========================================================================

    kompas = load_kompas_dataset(
        kompas_path
    )

    kompas = prepare_temporal_features(
        kompas
    )

    print(
        f"Jumlah data : {len(kompas):,}"
    )

    print(
        f"Jumlah kategori : "
        f"{kompas['category'].nunique():,}"
    )

    # =========================================================================
    # CREATE ANALYSIS TABLES
    # =========================================================================

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

    weekday_distribution = (
        create_weekday_distribution(
            dataframe=kompas,
            daily_distribution=daily_distribution,
        )
    )

    temporal_summary = (
        create_temporal_summary(
            dataframe=kompas,
            monthly_distribution=monthly_distribution,
            daily_distribution=daily_distribution,
            hourly_distribution=hourly_distribution,
            weekday_distribution=weekday_distribution,
        )
    )

    # =========================================================================
    # VALIDATE ANALYSIS TOTALS
    # =========================================================================

    validate_analysis_totals(
        dataframe=kompas,
        monthly_distribution=monthly_distribution,
        monthly_category_distribution=(
            monthly_category_distribution
        ),
        daily_distribution=daily_distribution,
        hourly_distribution=hourly_distribution,
        weekday_distribution=weekday_distribution,
    )

    # =========================================================================
    # SAVE TABLES
    # =========================================================================

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
        date_format="%Y-%m-%d",
    )

    hourly_distribution.to_csv(
        HOURLY_DISTRIBUTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    weekday_distribution.to_csv(
        WEEKDAY_DISTRIBUTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    temporal_summary.to_csv(
        TEMPORAL_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # TERMINAL DISPLAY
    # =========================================================================

    print(
        "\nRentang tanggal:"
    )

    print(
        f"Tanggal awal  : "
        f"{kompas['date'].min()}"
    )

    print(
        f"Tanggal akhir : "
        f"{kompas['date'].max()}"
    )

    print(
        "\nDistribusi berita per bulan:"
    )

    print(
        monthly_distribution.to_string(
            index=False
        )
    )

    print(
        "\nDistribusi berita per kategori dan bulan:"
    )

    print(
        monthly_category_distribution.to_string(
            index=False
        )
    )

    print(
        "\n10 tanggal dengan berita terbanyak:"
    )

    print(
        daily_distribution
        .sort_values(
            [
                "jumlah_berita",
                "publication_date",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    print(
        "\nDistribusi jam publikasi:"
    )

    print(
        hourly_distribution.to_string(
            index=False
        )
    )

    print(
        "\nDistribusi berdasarkan hari:"
    )

    print(
        weekday_distribution.to_string(
            index=False
        )
    )

    print(
        "\nRingkasan temporal:"
    )

    print(
        temporal_summary.to_string(
            index=False
        )
    )

    # =========================================================================
    # CREATE FIGURES
    # =========================================================================

    plot_monthly_distribution(
        dataframe=monthly_distribution,
        output_path=MONTHLY_DISTRIBUTION_FIGURE,
    )

    plot_monthly_category_distribution(
        dataframe=monthly_category_distribution,
        output_path=MONTHLY_CATEGORY_FIGURE,
    )

    plot_daily_distribution(
        dataframe=daily_distribution,
        output_path=DAILY_DISTRIBUTION_FIGURE,
    )

    plot_hourly_distribution(
        dataframe=hourly_distribution,
        output_path=HOURLY_DISTRIBUTION_FIGURE,
    )

    plot_weekday_distribution(
        dataframe=weekday_distribution,
        output_path=WEEKDAY_DISTRIBUTION_FIGURE,
    )

    # =========================================================================
    # OUTPUT VALIDATION
    # =========================================================================

    output_files = [
        (
            MONTHLY_DISTRIBUTION_PATH,
            "distribusi bulanan",
        ),
        (
            MONTHLY_CATEGORY_DISTRIBUTION_PATH,
            "distribusi bulanan per kategori",
        ),
        (
            DAILY_DISTRIBUTION_PATH,
            "distribusi harian",
        ),
        (
            HOURLY_DISTRIBUTION_PATH,
            "distribusi jam",
        ),
        (
            WEEKDAY_DISTRIBUTION_PATH,
            "distribusi hari",
        ),
        (
            TEMPORAL_SUMMARY_PATH,
            "ringkasan temporal",
        ),
        (
            MONTHLY_DISTRIBUTION_FIGURE,
            "grafik distribusi bulanan",
        ),
        (
            MONTHLY_CATEGORY_FIGURE,
            "grafik kategori per bulan",
        ),
        (
            DAILY_DISTRIBUTION_FIGURE,
            "grafik distribusi harian",
        ),
        (
            HOURLY_DISTRIBUTION_FIGURE,
            "grafik distribusi jam",
        ),
        (
            WEEKDAY_DISTRIBUTION_FIGURE,
            "grafik distribusi hari",
        ),
    ]

    for file_path, description in output_files:
        validate_output_file(
            file_path=file_path,
            description=description,
        )

    # =========================================================================
    # OUTPUT INFORMATION
    # =========================================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "OUTPUT TEMPORAL ANALYSIS"
    )

    print(
        "=" * 80
    )

    print(
        "\nDistribusi bulanan:"
    )

    print(
        MONTHLY_DISTRIBUTION_PATH
    )

    print(
        "\nDistribusi bulanan per kategori:"
    )

    print(
        MONTHLY_CATEGORY_DISTRIBUTION_PATH
    )

    print(
        "\nDistribusi harian:"
    )

    print(
        DAILY_DISTRIBUTION_PATH
    )

    print(
        "\nDistribusi jam:"
    )

    print(
        HOURLY_DISTRIBUTION_PATH
    )

    print(
        "\nDistribusi hari:"
    )

    print(
        WEEKDAY_DISTRIBUTION_PATH
    )

    print(
        "\nRingkasan temporal:"
    )

    print(
        TEMPORAL_SUMMARY_PATH
    )

    print(
        "\nGrafik distribusi bulanan:"
    )

    print(
        MONTHLY_DISTRIBUTION_FIGURE
    )

    print(
        "\nGrafik kategori per bulan:"
    )

    print(
        MONTHLY_CATEGORY_FIGURE
    )

    print(
        "\nGrafik distribusi harian:"
    )

    print(
        DAILY_DISTRIBUTION_FIGURE
    )

    print(
        "\nGrafik distribusi jam:"
    )

    print(
        HOURLY_DISTRIBUTION_FIGURE
    )

    print(
        "\nGrafik distribusi hari:"
    )

    print(
        WEEKDAY_DISTRIBUTION_FIGURE
    )

    print(
        "\nCatatan interpretasi:"
    )

    print(
        "- Januari dan Mei merupakan bulan parsial."
    )

    print(
        "- Jumlah bulanan harus dibaca bersama rata-rata per hari."
    )

    print(
        "- Distribusi waktu menggambarkan periode crawling dataset."
    )

    print(
        "- Timestamp dianalisis sesuai nilai yang tersimpan pada CSV."
    )

    print(
        "\nValidasi total agregasi: LULUS"
    )

    print(
        "Tahap temporal analysis selesai."
    )


if __name__ == "__main__":
    main()
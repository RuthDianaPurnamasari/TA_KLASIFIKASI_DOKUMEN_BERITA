# =============================================================================
# STEP 7.3 - SHAP WATERFALL PLOT
# =============================================================================
# File:
# 7_explainability/03_shap_waterfall.py
#
# Tujuan:
# Membuat visualisasi waterfall-style dari hasil local SHAP CNN K2.
#
# Input:
# 9_results/tables/shap/local/cnn_k2_local_shap_summary.csv
# 9_results/tables/shap/local/cnn_k2_local_token_contributions.csv
#
# Output:
# 9_results/figures/shap/waterfall/*.png
# 9_results/tables/shap/cnn_k2_waterfall_summary.csv
# 9_results/tables/shap/cnn_k2_waterfall_configuration.json
# =============================================================================

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# KONFIGURASI
# =============================================================================

MODEL_NAME = "cnn_k2"
DATASET_NAME = "Kompas"
SCENARIO_CODE = "K2"
SCENARIO_NAME = "Title + Description"

TOP_N_TOKENS = 15

SPECIAL_TOKENS = {
    "[PAD]",
    "[SEP]",
    "[UNK]",
    "[OOV]",
    "",
}


# =============================================================================
# PATH INPUT
# =============================================================================

LOCAL_SUMMARY_PATH = (
    PROJECT_ROOT
    / "9_results"
    / "tables"
    / "shap"
    / "local"
    / "cnn_k2_local_shap_summary.csv"
)

LOCAL_CONTRIBUTIONS_PATH = (
    PROJECT_ROOT
    / "9_results"
    / "tables"
    / "shap"
    / "local"
    / "cnn_k2_local_token_contributions.csv"
)


# =============================================================================
# PATH OUTPUT
# =============================================================================

OUTPUT_FIGURES_DIR = (
    PROJECT_ROOT
    / "9_results"
    / "figures"
    / "shap"
    / "waterfall"
)

OUTPUT_TABLES_DIR = (
    PROJECT_ROOT
    / "9_results"
    / "tables"
    / "shap"
)

WATERFALL_SUMMARY_PATH = (
    OUTPUT_TABLES_DIR
    / "cnn_k2_waterfall_summary.csv"
)

WATERFALL_CONFIGURATION_PATH = (
    OUTPUT_TABLES_DIR
    / "cnn_k2_waterfall_configuration.json"
)


# =============================================================================
# UTILITAS
# =============================================================================

def print_header(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)


def create_output_directories() -> None:
    OUTPUT_FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def safe_filename(text: str) -> str:
    value = str(text).strip()
    value = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        value,
    )
    value = re.sub(
        r"_+",
        "_",
        value,
    )
    return value.strip("_")


# =============================================================================
# MEMUAT DATA
# =============================================================================

def load_local_shap_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Memuat ringkasan dan kontribusi token dari STEP 7.2.
    """

    if not LOCAL_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "File ringkasan local SHAP tidak ditemukan:\n"
            f"{LOCAL_SUMMARY_PATH}\n\n"
            "Jalankan 02_shap_local.py terlebih dahulu."
        )

    if not LOCAL_CONTRIBUTIONS_PATH.exists():
        raise FileNotFoundError(
            "File kontribusi token local SHAP tidak ditemukan:\n"
            f"{LOCAL_CONTRIBUTIONS_PATH}\n\n"
            "Jalankan 02_shap_local.py terlebih dahulu."
        )

    summary = pd.read_csv(
        LOCAL_SUMMARY_PATH,
        encoding="utf-8-sig",
    )

    contributions = pd.read_csv(
        LOCAL_CONTRIBUTIONS_PATH,
        encoding="utf-8-sig",
    )

    required_summary_columns = {
        "document_id",
        "selection_type",
        "actual_label",
        "predicted_label",
        "is_correct",
        "prediction_confidence",
    }

    required_contribution_columns = {
        "document_id",
        "selection_type",
        "actual_label",
        "predicted_label",
        "token",
        "signed_shap",
        "absolute_shap",
        "direction",
    }

    missing_summary = (
        required_summary_columns
        - set(summary.columns)
    )

    missing_contributions = (
        required_contribution_columns
        - set(contributions.columns)
    )

    if missing_summary:
        raise KeyError(
            "Kolom ringkasan local SHAP tidak lengkap.\n"
            f"Kolom hilang: {missing_summary}"
        )

    if missing_contributions:
        raise KeyError(
            "Kolom kontribusi token tidak lengkap.\n"
            f"Kolom hilang: {missing_contributions}"
        )

    summary["prediction_confidence"] = pd.to_numeric(
        summary["prediction_confidence"],
        errors="coerce",
    )

    summary["is_correct"] = (
        summary["is_correct"]
        .astype(str)
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )

    contributions["signed_shap"] = pd.to_numeric(
        contributions["signed_shap"],
        errors="coerce",
    )

    contributions["absolute_shap"] = pd.to_numeric(
        contributions["absolute_shap"],
        errors="coerce",
    )

    contributions["token"] = (
        contributions["token"]
        .astype(str)
        .str.strip()
    )

    contributions = contributions.dropna(
        subset=[
            "signed_shap",
            "absolute_shap",
        ]
    )

    contributions = contributions[
        ~contributions["token"].isin(
            SPECIAL_TOKENS
        )
    ].copy()

    if summary.empty:
        raise ValueError(
            "Ringkasan local SHAP kosong."
        )

    if contributions.empty:
        raise ValueError(
            "Kontribusi token local SHAP kosong."
        )

    return summary, contributions


# =============================================================================
# MENYIAPKAN TOKEN
# =============================================================================

def aggregate_sample_tokens(
    sample_contributions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Menggabungkan token yang sama dalam satu artikel.
    """

    aggregated = (
        sample_contributions
        .groupby(
            "token",
            as_index=False,
        )
        .agg(
            signed_shap=(
                "signed_shap",
                "sum",
            ),
            absolute_shap=(
                "absolute_shap",
                "sum",
            ),
            occurrence_count=(
                "token",
                "size",
            ),
        )
    )

    aggregated = (
        aggregated
        .sort_values(
            "absolute_shap",
            ascending=False,
        )
        .head(
            TOP_N_TOKENS
        )
        .copy()
    )

    aggregated["direction"] = np.where(
        aggregated["signed_shap"] > 0,
        "mendukung_prediksi",
        np.where(
            aggregated["signed_shap"] < 0,
            "menahan_prediksi",
            "netral",
        ),
    )

    return aggregated


# =============================================================================
# MEMBUAT WATERFALL-STYLE PLOT
# =============================================================================

def create_waterfall_plot(
    summary_row: pd.Series,
    sample_contributions: pd.DataFrame,
) -> Path:
    """
    Membuat grafik waterfall-style kontribusi token.

    Nilai positif:
    mendukung kelas prediksi.

    Nilai negatif:
    menahan kelas prediksi.
    """

    token_data = aggregate_sample_tokens(
        sample_contributions
    )

    if token_data.empty:
        raise ValueError(
            "Tidak ada token yang dapat divisualisasikan."
        )

    plot_data = token_data.sort_values(
        "signed_shap",
        ascending=True,
    )

    document_id = str(
        summary_row["document_id"]
    )

    selection_type = str(
        summary_row["selection_type"]
    )

    actual_label = str(
        summary_row["actual_label"]
    )

    predicted_label = str(
        summary_row["predicted_label"]
    )

    is_correct = bool(
        summary_row["is_correct"]
    )

    confidence = float(
        summary_row["prediction_confidence"]
    )

    status = (
        "correct"
        if is_correct
        else "incorrect"
    )

    output_filename = (
        f"{safe_filename(document_id)}_"
        f"{safe_filename(selection_type)}_"
        f"actual_{safe_filename(actual_label)}_"
        f"pred_{safe_filename(predicted_label)}_"
        f"waterfall.png"
    )

    output_path = (
        OUTPUT_FIGURES_DIR
        / output_filename
    )

    values = plot_data[
        "signed_shap"
    ].to_numpy(
        dtype=float
    )

    tokens = plot_data[
        "token"
    ].astype(str).tolist()

    positions = np.arange(
        len(tokens)
    )

    positive_values = np.where(
        values > 0,
        values,
        0,
    )

    negative_values = np.where(
        values < 0,
        values,
        0,
    )

    figure, axis = plt.subplots(
        figsize=(11, 8)
    )

    axis.barh(
        positions,
        positive_values,
        label="Mendukung prediksi",
    )

    axis.barh(
        positions,
        negative_values,
        label="Menahan prediksi",
    )

    axis.axvline(
        0,
        linewidth=1,
    )

    axis.set_yticks(
        positions
    )

    axis.set_yticklabels(
        tokens
    )

    axis.set_xlabel(
        "Signed SHAP Contribution"
    )

    axis.set_ylabel(
        "Token"
    )

    status_text = (
        "BENAR"
        if is_correct
        else "SALAH"
    )

    axis.set_title(
        "Local SHAP Waterfall-Style Plot — CNN K2\n"
        f"Document: {document_id} | "
        f"Aktual: {actual_label} | "
        f"Prediksi: {predicted_label} | "
        f"Confidence: {confidence:.2%} | "
        f"{status_text}"
    )

    axis.grid(
        axis="x",
        alpha=0.3,
    )

    axis.legend()

    maximum_value = max(
        float(
            np.max(
                np.abs(
                    values
                )
            )
        ),
        0.001,
    )

    text_offset = (
        maximum_value
        * 0.025
    )

    for position, value in zip(
        positions,
        values,
    ):
        if value >= 0:
            text_x = (
                value
                + text_offset
            )
            horizontal_alignment = "left"
        else:
            text_x = (
                value
                - text_offset
            )
            horizontal_alignment = "right"

        axis.text(
            text_x,
            position,
            f"{value:+.4f}",
            va="center",
            ha=horizontal_alignment,
            fontsize=9,
        )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


# =============================================================================
# RINGKASAN WATERFALL
# =============================================================================

def build_waterfall_summary_row(
    summary_row: pd.Series,
    sample_contributions: pd.DataFrame,
    figure_path: Path,
) -> dict:
    """
    Membentuk satu baris ringkasan waterfall.
    """

    token_data = aggregate_sample_tokens(
        sample_contributions
    )

    supporting_tokens = (
        token_data[
            token_data[
                "signed_shap"
            ] > 0
        ]
        .sort_values(
            "signed_shap",
            ascending=False,
        )
        .head(5)
    )

    opposing_tokens = (
        token_data[
            token_data[
                "signed_shap"
            ] < 0
        ]
        .sort_values(
            "signed_shap",
            ascending=True,
        )
        .head(5)
    )

    supporting_text = "; ".join(
        f"{row.token} ({row.signed_shap:+.4f})"
        for row in supporting_tokens.itertuples(
            index=False
        )
    )

    opposing_text = "; ".join(
        f"{row.token} ({row.signed_shap:+.4f})"
        for row in opposing_tokens.itertuples(
            index=False
        )
    )

    return {
        "document_id":
            summary_row["document_id"],

        "selection_type":
            summary_row["selection_type"],

        "actual_label":
            summary_row["actual_label"],

        "predicted_label":
            summary_row["predicted_label"],

        "is_correct":
            bool(
                summary_row["is_correct"]
            ),

        "prediction_confidence":
            float(
                summary_row[
                    "prediction_confidence"
                ]
            ),

        "top_supporting_tokens":
            supporting_text,

        "top_opposing_tokens":
            opposing_text,

        "figure_path":
            str(
                figure_path
            ),
    }


# =============================================================================
# MENYIMPAN KONFIGURASI
# =============================================================================

def save_configuration(
    number_of_samples: int,
    success_count: int,
    failure_count: int,
) -> None:
    """
    Menyimpan konfigurasi waterfall.
    """

    configuration = {
        "model":
            MODEL_NAME,

        "dataset":
            DATASET_NAME,

        "scenario_code":
            SCENARIO_CODE,

        "scenario_name":
            SCENARIO_NAME,

        "source_summary":
            str(
                LOCAL_SUMMARY_PATH
            ),

        "source_token_contributions":
            str(
                LOCAL_CONTRIBUTIONS_PATH
            ),

        "top_n_tokens_per_plot":
            TOP_N_TOKENS,

        "number_of_samples":
            number_of_samples,

        "successful_plots":
            success_count,

        "failed_plots":
            failure_count,

        "special_tokens_excluded":
            sorted(
                SPECIAL_TOKENS
            ),

        "visualization_note": (
            "This is a waterfall-style local SHAP visualization "
            "built from signed token-level SHAP contributions. "
            "Positive values support the predicted class and "
            "negative values reduce the predicted class score."
        ),
    }

    with open(
        WATERFALL_CONFIGURATION_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            ensure_ascii=False,
            indent=4,
        )


# =============================================================================
# PROGRAM UTAMA
# =============================================================================

def main() -> None:
    """
    Membuat waterfall-style plot untuk delapan sampel local SHAP.
    """

    print_header(
        "STEP 7.3 - SHAP WATERFALL PLOT"
    )

    create_output_directories()

    print("\nKonfigurasi:")
    print(
        f"Model                : "
        f"{MODEL_NAME}"
    )
    print(
        f"Dataset              : "
        f"{DATASET_NAME}"
    )
    print(
        f"Skenario             : "
        f"{SCENARIO_CODE}"
    )
    print(
        f"Representasi         : "
        f"{SCENARIO_NAME}"
    )
    print(
        f"Top token per plot   : "
        f"{TOP_N_TOKENS}"
    )

    print(
        "\nMemuat hasil local SHAP..."
    )

    summary, contributions = (
        load_local_shap_data()
    )

    print(
        f"File ringkasan       : "
        f"{LOCAL_SUMMARY_PATH}"
    )
    print(
        f"File kontribusi      : "
        f"{LOCAL_CONTRIBUTIONS_PATH}"
    )
    print(
        f"Jumlah sampel        : "
        f"{len(summary)}"
    )
    print(
        f"Jumlah baris token   : "
        f"{len(contributions):,}"
    )

    success_count = 0
    failure_count = 0
    output_rows = []

    print(
        "\nMembuat waterfall plot..."
    )

    for number, summary_row in summary.iterrows():
        document_id = str(
            summary_row[
                "document_id"
            ]
        )

        selection_type = str(
            summary_row[
                "selection_type"
            ]
        )

        sample_contributions = contributions[
            (
                contributions[
                    "document_id"
                ].astype(str)
                == document_id
            )
            & (
                contributions[
                    "selection_type"
                ].astype(str)
                == selection_type
            )
        ].copy()

        print(
            "\n" + "-" * 80
        )
        print(
            f"{number + 1}/{len(summary)} "
            f"- {document_id}"
        )
        print(
            f"Jenis sampel         : "
            f"{selection_type}"
        )
        print(
            f"Label aktual         : "
            f"{summary_row['actual_label']}"
        )
        print(
            f"Label prediksi       : "
            f"{summary_row['predicted_label']}"
        )

        try:
            if sample_contributions.empty:
                raise ValueError(
                    "Kontribusi token untuk sampel ini tidak ditemukan."
                )

            figure_path = create_waterfall_plot(
                summary_row=summary_row,
                sample_contributions=(
                    sample_contributions
                ),
            )

            output_rows.append(
                build_waterfall_summary_row(
                    summary_row=summary_row,
                    sample_contributions=(
                        sample_contributions
                    ),
                    figure_path=figure_path,
                )
            )

            success_count += 1

            print(
                f"Grafik tersimpan     : "
                f"{figure_path}"
            )

        except Exception as error:
            failure_count += 1

            print(
                f"Gagal                : "
                f"{error}"
            )

    waterfall_summary = pd.DataFrame(
        output_rows
    )

    waterfall_summary.to_csv(
        WATERFALL_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    save_configuration(
        number_of_samples=len(
            summary
        ),
        success_count=success_count,
        failure_count=failure_count,
    )

    print("\n")
    print_header(
        "HASIL SHAP WATERFALL PLOT"
    )

    print(
        f"\nJumlah sampel       : "
        f"{len(summary)}"
    )
    print(
        f"Plot berhasil       : "
        f"{success_count}"
    )
    print(
        f"Plot gagal          : "
        f"{failure_count}"
    )

    print(
        "\nFolder waterfall:"
    )
    print(
        OUTPUT_FIGURES_DIR
    )

    print(
        "\nRingkasan waterfall:"
    )
    print(
        WATERFALL_SUMMARY_PATH
    )

    print(
        "\nKonfigurasi:"
    )
    print(
        WATERFALL_CONFIGURATION_PATH
    )

    print("\n")
    print_header(
        "Tahap SHAP waterfall selesai."
    )


if __name__ == "__main__":
    main()
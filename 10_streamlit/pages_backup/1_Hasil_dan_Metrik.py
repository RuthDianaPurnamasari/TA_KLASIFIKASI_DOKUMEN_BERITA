# =============================================================================
# PAGE 1 - HASIL DAN METRIK
# =============================================================================
# Halaman ini menampilkan:
# 1. Ringkasan hasil 10 eksperimen.
# 2. Grafik perbandingan accuracy dan macro F1.
# 3. Temuan kontribusi Description dan YAKE.
# 4. Detail metrik eksperimen terpilih.
# 5. Confusion matrix dan training curve.
# 6. Tabel hasil yang sudah dibersihkan dari kolom path teknis.
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================

CURRENT_FILE = Path(__file__).resolve()
STREAMLIT_DIR = CURRENT_FILE.parents[1]

if str(STREAMLIT_DIR) not in sys.path:
    sys.path.insert(0, str(STREAMLIT_DIR))


# =============================================================================
# IMPORT DASHBOARD MODULES
# =============================================================================

from utils.data_loader import (
    get_comparative_figure_path,
    get_confusion_matrix_path,
    get_training_curve_path,
    load_description_contribution,
    load_misclassification_analysis,
    load_model_comparison,
    load_scenario_comparison,
    load_test_evaluation,
    load_yake_contribution,
)

from utils.ui import (
    format_percentage,
    page_header,
    show_missing_figure_message,
)


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Hasil dan Metrik",
    page_icon="📊",
    layout="wide",
)


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown(
    """
    <style>
        .result-card {
            background: linear-gradient(
                135deg,
                rgba(240, 247, 255, 0.96),
                rgba(255, 255, 255, 0.96)
            );
            border: 1px solid rgba(75, 120, 180, 0.18);
            border-radius: 18px;
            padding: 20px 22px;
            margin-bottom: 14px;
            box-shadow: 0 5px 18px rgba(30, 60, 100, 0.07);
        }

        .finding-success {
            background: rgba(228, 248, 236, 0.95);
            border-left: 6px solid #16a05d;
            border-radius: 14px;
            padding: 18px 20px;
            min-height: 180px;
        }

        .finding-warning {
            background: rgba(255, 248, 220, 0.95);
            border-left: 6px solid #d59a00;
            border-radius: 14px;
            padding: 18px 20px;
            min-height: 180px;
        }

        .finding-info {
            background: rgba(231, 241, 255, 0.95);
            border-left: 6px solid #3676d1;
            border-radius: 14px;
            padding: 18px 20px;
            min-height: 180px;
        }

        .small-muted {
            color: #707785;
            font-size: 0.92rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(248, 250, 253, 0.92);
            border: 1px solid rgba(100, 115, 140, 0.14);
            border-radius: 15px;
            padding: 14px 16px;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

EXPERIMENTS = {
    "Kompas": {
        "CNN": {
            "K1": "cnn_k1",
            "K2": "cnn_k2",
            "K3": "cnn_k3",
        },
        "Attention-BiLSTM": {
            "K1": "attention_bilstm_k1",
            "K2": "attention_bilstm_k2",
            "K3": "attention_bilstm_k3",
        },
    },
    "AG News": {
        "CNN": {
            "A1": "cnn_a1",
            "A2": "cnn_a2",
        },
        "Attention-BiLSTM": {
            "A1": "attention_bilstm_a1",
            "A2": "attention_bilstm_a2",
        },
    },
}

SCENARIO_NAMES = {
    "K1": "Title",
    "K2": "Title + Description",
    "K3": "Title + Description + Keyword YAKE",
    "A1": "Title",
    "A2": "Title + Description",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """Mencari nama kolom berdasarkan beberapa kemungkinan nama."""

    normalized_columns = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        normalized_candidate = candidate.strip().lower()

        if normalized_candidate in normalized_columns:
            return normalized_columns[normalized_candidate]

    return None


def normalize_metric_value(
    value: object,
) -> float | None:
    """Menormalkan nilai metrik ke skala 0–1."""

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if numeric_value > 1:
        numeric_value = numeric_value / 100

    return numeric_value


def get_metric_value(
    dataframe: pd.DataFrame,
    experiment_name: str,
    candidates: list[str],
) -> float | None:
    """Mengambil nilai metrik dari satu eksperimen."""

    if dataframe.empty:
        return None

    experiment_column = find_column(
        dataframe,
        [
            "experiment_name",
            "experiment",
            "experiment_id",
            "model_scenario",
        ],
    )

    metric_column = find_column(
        dataframe,
        candidates,
    )

    if experiment_column is None or metric_column is None:
        return None

    selected = dataframe[
        dataframe[experiment_column]
        .astype(str)
        .str.strip()
        .str.lower()
        == experiment_name.lower()
    ]

    if selected.empty:
        return None

    return normalize_metric_value(
        selected.iloc[0][metric_column]
    )


def metric_display(
    value: float | None,
) -> str:
    """Menampilkan metrik sebagai persentase."""

    if value is None:
        return "-"

    return format_percentage(value)


def clean_evaluation_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Menghapus kolom teknis seperti checkpoint_path,
    test_path, prediction_path, dan model path.
    """

    if dataframe.empty:
        return dataframe

    technical_keywords = [
        "path",
        "checkpoint",
        "file",
        "directory",
        "folder",
    ]

    allowed_columns = []

    for column in dataframe.columns:
        normalized = str(column).strip().lower()

        is_technical = any(
            keyword in normalized
            for keyword in technical_keywords
        )

        if not is_technical:
            allowed_columns.append(column)

    cleaned = dataframe[allowed_columns].copy()

    preferred_order = [
        "experiment_name",
        "dataset",
        "model",
        "scenario_code",
        "scenario_name",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "inference_time_ms",
    ]

    ordered_columns = []

    for preferred in preferred_order:
        actual = find_column(cleaned, [preferred])

        if actual is not None and actual not in ordered_columns:
            ordered_columns.append(actual)

    remaining_columns = [
        column
        for column in cleaned.columns
        if column not in ordered_columns
    ]

    return cleaned[
        ordered_columns + remaining_columns
    ]


def format_metric_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Mengubah nilai metrik menjadi persentase untuk tampilan tabel."""

    if dataframe.empty:
        return dataframe

    formatted = dataframe.copy()

    metric_candidates = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "test_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ]

    for candidate in metric_candidates:
        actual_column = find_column(
            formatted,
            [candidate],
        )

        if actual_column is None:
            continue

        formatted[actual_column] = formatted[
            actual_column
        ].apply(
            lambda value: (
                metric_display(
                    normalize_metric_value(value)
                )
                if pd.notna(value)
                else "-"
            )
        )

    return formatted


def show_figure(
    path: Path,
    caption: str,
) -> None:
    """Menampilkan gambar grafik atau pesan jika tidak tersedia."""

    if path.exists():
        st.image(
            str(path),
            caption=caption,
            use_container_width=True,
        )
    else:
        show_missing_figure_message(
            figure_name=caption,
            figure_path=path,
        )


# =============================================================================
# LOAD DATA
# =============================================================================

evaluation_data = load_test_evaluation()
model_comparison = load_model_comparison()
scenario_comparison = load_scenario_comparison()
description_data = load_description_contribution()
yake_data = load_yake_contribution()
misclassification_data = load_misclassification_analysis()

clean_evaluation_data = clean_evaluation_table(
    evaluation_data
)

formatted_evaluation_data = format_metric_columns(
    clean_evaluation_data
)


# =============================================================================
# HEADER
# =============================================================================

page_header(
    title="📊 Hasil dan Metrik",
    subtitle=(
        "Ringkasan performa 10 eksperimen CNN dan "
        "Attention-BiLSTM pada dataset Kompas dan AG News."
    ),
)


# =============================================================================
# EXECUTIVE SUMMARY
# =============================================================================

st.markdown("## Ringkasan Hasil Penelitian")

summary_columns = st.columns(4)

with summary_columns[0]:
    st.metric(
        label="Eksperimen Utama",
        value="10",
        help="Lima skenario yang diuji pada dua model.",
    )

with summary_columns[1]:
    st.metric(
        label="Model Terbaik",
        value="CNN K2",
        help="CNN dengan representasi Title + Description.",
    )

with summary_columns[2]:
    st.metric(
        label="Accuracy Terbaik",
        value="95.80%",
        help="Hasil test set Kompas.",
    )

with summary_columns[3]:
    st.metric(
        label="Macro F1 Terbaik",
        value="95.81%",
        help="Rata-rata F1 seluruh kelas Kompas.",
    )

st.markdown(
    """
    <div class="result-card">
        <b>Kesimpulan utama:</b>
        representasi <b>Title + Description</b> memberikan hasil terbaik.
        CNN K2 memperoleh accuracy 95,80% dan menjadi model utama
        untuk implementasi dashboard serta analisis SHAP.
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# OVERALL COMPARISON CHARTS
# =============================================================================

st.divider()

st.markdown("## Perbandingan Seluruh Eksperimen")

st.caption(
    "Grafik berikut memberikan gambaran umum performa model "
    "sebelum melihat detail setiap eksperimen."
)

chart_tab_1, chart_tab_2, chart_tab_3 = st.tabs(
    [
        "Accuracy dan Macro F1",
        "Kontribusi Description",
        "Kontribusi YAKE",
    ]
)


with chart_tab_1:

    chart_column_1, chart_column_2 = st.columns(2)

    with chart_column_1:
        accuracy_figure = get_comparative_figure_path(
            "accuracy_comparison.png"
        )

        show_figure(
            accuracy_figure,
            "Perbandingan Accuracy Seluruh Eksperimen",
        )

    with chart_column_2:
        f1_figure = get_comparative_figure_path(
            "f1_macro_comparison.png"
        )

        show_figure(
            f1_figure,
            "Perbandingan Macro F1 Seluruh Eksperimen",
        )

    st.info(
        "CNN unggul pada Kompas K1, Kompas K2, AG News A1, "
        "dan AG News A2. Attention-BiLSTM sedikit unggul pada Kompas K3."
    )


with chart_tab_2:

    description_figure = get_comparative_figure_path(
        "description_contribution.png"
    )

    show_figure(
        description_figure,
        "Peningkatan Performa Setelah Penambahan Description",
    )

    st.success(
        "Description meningkatkan accuracy pada kedua model "
        "dan pada kedua dataset."
    )

    if not description_data.empty:
        with st.expander("Lihat data kontribusi Description"):
            description_display = format_metric_columns(
                clean_evaluation_table(description_data)
            )

            st.dataframe(
                description_display,
                use_container_width=True,
                hide_index=True,
            )


with chart_tab_3:

    yake_figure = get_comparative_figure_path(
        "yake_contribution.png"
    )

    show_figure(
        yake_figure,
        "Perubahan Accuracy Setelah Penambahan Keyword YAKE",
    )

    st.warning(
        "Pada konfigurasi penelitian ini, keyword YAKE belum "
        "meningkatkan accuracy dibandingkan Title + Description."
    )

    if not yake_data.empty:
        with st.expander("Lihat data kontribusi YAKE"):
            yake_display = format_metric_columns(
                clean_evaluation_table(yake_data)
            )

            st.dataframe(
                yake_display,
                use_container_width=True,
                hide_index=True,
            )


# =============================================================================
# MAIN FINDINGS
# =============================================================================

st.divider()

st.markdown("## Temuan Utama")

finding_column_1, finding_column_2, finding_column_3 = st.columns(3)

with finding_column_1:
    st.markdown(
        """
        <div class="finding-success">
            <h4>✅ Description Membantu</h4>
            <p>
                Penambahan Description meningkatkan performa
                CNN dan Attention-BiLSTM pada Kompas maupun AG News.
            </p>
            <b>Kompas CNN:</b> 94,70% → 95,80%
        </div>
        """,
        unsafe_allow_html=True,
    )

with finding_column_2:
    st.markdown(
        """
        <div class="finding-warning">
            <h4>⚠️ YAKE Belum Meningkat</h4>
            <p>
                Keyword YAKE belum meningkatkan accuracy karena
                informasi keyword kemungkinan sudah terkandung
                pada Title dan Description.
            </p>
            <b>CNN:</b> 95,80% → 95,00%
        </div>
        """,
        unsafe_allow_html=True,
    )

with finding_column_3:
    st.markdown(
        """
        <div class="finding-info">
            <h4>🏆 CNN K2 Terbaik</h4>
            <p>
                CNN dengan Title + Description menjadi konfigurasi
                terbaik pada dataset utama Kompas.
            </p>
            <b>Accuracy:</b> 95,80%<br>
            <b>Macro F1:</b> 95,81%
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# EXPERIMENT DETAIL
# =============================================================================

st.divider()

st.markdown("## Detail Eksperimen")

st.caption(
    "Pilih dataset, model, dan skenario untuk melihat metrik, "
    "confusion matrix, dan training curve."
)

filter_column_1, filter_column_2, filter_column_3 = st.columns(3)

with filter_column_1:
    selected_dataset = st.selectbox(
        "Dataset",
        ["Kompas", "AG News"],
        index=0,
    )

with filter_column_2:
    selected_model = st.selectbox(
        "Model",
        ["CNN", "Attention-BiLSTM"],
        index=0,
    )

available_scenarios = list(
    EXPERIMENTS[selected_dataset][selected_model].keys()
)

default_scenario_index = (
    1 if len(available_scenarios) > 1 else 0
)

with filter_column_3:
    selected_scenario = st.selectbox(
        "Skenario",
        available_scenarios,
        index=default_scenario_index,
        format_func=lambda code: (
            f"{code} — {SCENARIO_NAMES[code]}"
        ),
    )

experiment_name = (
    EXPERIMENTS[selected_dataset]
    [selected_model]
    [selected_scenario]
)

st.markdown(
    f"""
    <div class="result-card">
        <b>Eksperimen:</b> {experiment_name}<br>
        <b>Dataset:</b> {selected_dataset}<br>
        <b>Model:</b> {selected_model}<br>
        <b>Representasi:</b> {SCENARIO_NAMES[selected_scenario]}
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SELECTED METRICS
# =============================================================================

accuracy = get_metric_value(
    evaluation_data,
    experiment_name,
    ["accuracy", "test_accuracy"],
)

precision_macro = get_metric_value(
    evaluation_data,
    experiment_name,
    [
        "precision_macro",
        "macro_precision",
        "precision",
    ],
)

recall_macro = get_metric_value(
    evaluation_data,
    experiment_name,
    [
        "recall_macro",
        "macro_recall",
        "recall",
    ],
)

f1_macro = get_metric_value(
    evaluation_data,
    experiment_name,
    [
        "f1_macro",
        "macro_f1",
        "f1_score_macro",
        "f1_score",
    ],
)

metric_columns = st.columns(4)

with metric_columns[0]:
    st.metric(
        "Accuracy",
        metric_display(accuracy),
    )

with metric_columns[1]:
    st.metric(
        "Precision Macro",
        metric_display(precision_macro),
    )

with metric_columns[2]:
    st.metric(
        "Recall Macro",
        metric_display(recall_macro),
    )

with metric_columns[3]:
    st.metric(
        "F1-Score Macro",
        metric_display(f1_macro),
    )


# =============================================================================
# CONFUSION MATRIX AND TRAINING CURVE
# =============================================================================

visual_tab_1, visual_tab_2 = st.tabs(
    [
        "Confusion Matrix",
        "Training Curve",
    ]
)


with visual_tab_1:

    matrix_type = st.radio(
        "Jenis matriks",
        ["Jumlah", "Normalized"],
        horizontal=True,
    )

    normalized = matrix_type == "Normalized"

    confusion_path = get_confusion_matrix_path(
        experiment_name=experiment_name,
        normalized=normalized,
    )

    show_figure(
        confusion_path,
        f"Confusion Matrix {experiment_name} — {matrix_type}",
    )

    st.caption(
        "Diagonal menunjukkan prediksi benar. Nilai di luar "
        "diagonal menunjukkan kesalahan klasifikasi antar kelas."
    )


with visual_tab_2:

    training_path = get_training_curve_path(
        experiment_name
    )

    show_figure(
        training_path,
        f"Training Curve {experiment_name}",
    )

    st.caption(
        "Grafik digunakan untuk melihat perubahan accuracy dan loss "
        "pada data train dan validation selama proses training."
    )


# =============================================================================
# MISCLASSIFICATION ANALYSIS
# =============================================================================

if not misclassification_data.empty:

    st.divider()

    st.markdown("## Kesalahan Klasifikasi Penting")

    experiment_column = find_column(
        misclassification_data,
        ["experiment_name", "experiment"],
    )

    selected_errors = misclassification_data.copy()

    if experiment_column is not None:
        experiment_errors = misclassification_data[
            misclassification_data[experiment_column]
            .astype(str)
            .str.lower()
            == experiment_name.lower()
        ]

        if not experiment_errors.empty:
            selected_errors = experiment_errors

    selected_errors = clean_evaluation_table(
        selected_errors
    ).head(10)

    st.dataframe(
        selected_errors,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Tabel menunjukkan kelas aktual dan prediksi yang paling "
        "sering tertukar pada eksperimen terpilih."
    )


# =============================================================================
# CLEAN EVALUATION TABLE
# =============================================================================

st.divider()

st.markdown("## Ringkasan Metrik 10 Eksperimen")

st.caption(
    "Hanya informasi penelitian penting yang ditampilkan. "
    "Path checkpoint, test set, dan file prediksi disembunyikan."
)

if formatted_evaluation_data.empty:

    st.info(
        "Data metrik belum tersedia pada "
        "`9_results/metrics/model_test_metrics.csv`."
    )

else:

    search_text = st.text_input(
        "Cari eksperimen",
        placeholder="Contoh: cnn_k2 atau attention_bilstm_a2",
    )

    displayed_table = formatted_evaluation_data.copy()

    experiment_column = find_column(
        displayed_table,
        ["experiment_name", "experiment"],
    )

    if search_text and experiment_column is not None:
        displayed_table = displayed_table[
            displayed_table[experiment_column]
            .astype(str)
            .str.contains(
                search_text,
                case=False,
                na=False,
            )
        ]

    st.dataframe(
        displayed_table,
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
# OPTIONAL COMPARISON TABLES
# =============================================================================

with st.expander("Lihat tabel comparative analysis"):

    comparison_tab_1, comparison_tab_2 = st.tabs(
        [
            "Perbandingan Model",
            "Perbandingan Skenario",
        ]
    )

    with comparison_tab_1:
        if model_comparison.empty:
            st.info("Tabel perbandingan model belum tersedia.")
        else:
            st.dataframe(
                format_metric_columns(
                    clean_evaluation_table(model_comparison)
                ),
                use_container_width=True,
                hide_index=True,
            )

    with comparison_tab_2:
        if scenario_comparison.empty:
            st.info("Tabel perbandingan skenario belum tersedia.")
        else:
            st.dataframe(
                format_metric_columns(
                    clean_evaluation_table(scenario_comparison)
                ),
                use_container_width=True,
                hide_index=True,
            )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Hasil evaluasi CNN dan Attention-BiLSTM pada "
    "dataset Kompas dan AG News."
)
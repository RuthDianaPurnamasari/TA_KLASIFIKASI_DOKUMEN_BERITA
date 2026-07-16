from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix


# =============================================================================
# PATH PROJECT
# =============================================================================

CURRENT_FILE = Path(__file__).resolve()
STREAMLIT_DIR = CURRENT_FILE.parent
PROJECT_ROOT = CURRENT_FILE.parents[1]

if str(STREAMLIT_DIR) not in sys.path:
    sys.path.insert(0, str(STREAMLIT_DIR))


# =============================================================================
# IMPORT PROJECT
# =============================================================================

from config import (  # noqa: E402
    DATASET_INFORMATION,
    MODEL_PERFORMANCE,
    RESEARCH_TITLE,
)

from utils.data_loader import (  # noqa: E402
    load_description_contribution,
    load_global_shap,
    load_global_shap_by_class,
    load_local_shap_summary,
    load_local_token_contributions,
    load_misclassification_analysis,
    load_model_comparison,
    load_scenario_comparison,
    load_test_evaluation,
    load_waterfall_summary,
    load_yake_contribution,
)

from utils.inference import predict_news  # noqa: E402


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Dashboard Klasifikasi Berita",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =============================================================================
# LOAD EXTERNAL CSS
# =============================================================================

def load_css(css_path: Path) -> None:
    """Memuat style.css eksternal."""

    if not css_path.exists():
        return

    css_content = css_path.read_text(encoding="utf-8")
    st.markdown(
        f"<style>{css_content}</style>",
        unsafe_allow_html=True,
    )


STYLE_PATH = STREAMLIT_DIR / "assets" / "style.css"
load_css(STYLE_PATH)

# CSS cadangan agar sidebar multipage tidak muncul walaupun style.css belum ada.
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"],
        div[data-testid="collapsedControl"] {
            display: none;
        }

        .block-container {
            max-width: 1500px;
            padding-top: 1.6rem;
            padding-bottom: 4rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# CONSTANTS
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
    "K2": "Title + Description (tanpa YAKE)",
    "K3": "Title + Description + Keyword YAKE",
    "A1": "Title",
    "A2": "Title + Description",
}

CATEGORY_DISPLAY = {
    "bola": "Bola",
    "global": "Global",
    "money": "Money",
    "tekno": "Tekno",
    "business": "Business",
    "sci_tech": "Sci/Tech",
    "sports": "Sports",
    "world": "World",
}

LABELS_BY_DATASET = {
    "Kompas": ["bola", "global", "money", "tekno"],
    "AG News": ["business", "sci_tech", "sports", "world"],
}

NUMERIC_LABEL_MAPPING = {
    "Kompas": {
        0: "bola",
        1: "global",
        2: "money",
        3: "tekno",
    },
    "AG News": {
        0: "business",
        1: "sci_tech",
        2: "sports",
        3: "world",
    },
}

SPECIAL_TOKENS = {
    "[PAD]",
    "[SEP]",
    "[UNK]",
    "[OOV]",
    "",
}

KOMPAS_TEST_SIZE = 1000

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
    ],
}


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """Mencari nama kolom berdasarkan beberapa kemungkinan nama."""

    if dataframe.empty:
        return None

    mapping = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        normalized = str(candidate).strip().lower()
        if normalized in mapping:
            return mapping[normalized]

    return None


def normalize_metric(value: Any) -> float | None:
    """Menormalkan nilai metrik menjadi skala 0-1."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if np.isnan(numeric):
        return None

    return numeric / 100 if numeric > 1 else numeric


def percentage(value: Any, digits: int = 2) -> str:
    """Menampilkan nilai sebagai persentase."""

    normalized = normalize_metric(value)
    if normalized is None:
        return "-"

    return f"{normalized * 100:.{digits}f}%"


def number_id(value: int | float) -> str:
    """Format angka Indonesia dengan pemisah ribuan titik."""

    return f"{value:,.0f}".replace(",", ".")


def display_label(label: Any) -> str:
    """Merapikan nama label untuk tampilan."""

    normalized = str(label).strip().lower()
    return CATEGORY_DISPLAY.get(normalized, normalized.title())


def get_experiment_row(
    dataframe: pd.DataFrame,
    experiment_name: str,
) -> pd.DataFrame:
    """Mengambil satu baris eksperimen."""

    experiment_column = find_column(
        dataframe,
        ["experiment_name", "experiment", "experiment_id"],
    )

    if experiment_column is None:
        return pd.DataFrame()

    return dataframe[
        dataframe[experiment_column]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(experiment_name.lower())
    ].copy()


def get_metric(
    dataframe: pd.DataFrame,
    experiment_name: str,
    candidates: list[str],
) -> float | None:
    """Mengambil satu nilai metrik eksperimen."""

    selected = get_experiment_row(dataframe, experiment_name)
    if selected.empty:
        return None

    metric_column = find_column(selected, candidates)
    if metric_column is None:
        return None

    return normalize_metric(selected.iloc[0][metric_column])


def get_metric_bundle(
    dataframe: pd.DataFrame,
    experiment_name: str,
) -> dict[str, float | None]:
    """Mengambil accuracy, precision, recall, dan macro F1."""

    return {
        "accuracy": get_metric(
            dataframe,
            experiment_name,
            ["accuracy", "test_accuracy"],
        ),
        "precision": get_metric(
            dataframe,
            experiment_name,
            ["precision_macro", "macro_precision", "precision"],
        ),
        "recall": get_metric(
            dataframe,
            experiment_name,
            ["recall_macro", "macro_recall", "recall"],
        ),
        "f1": get_metric(
            dataframe,
            experiment_name,
            ["f1_macro", "macro_f1", "f1_score_macro", "f1_score"],
        ),
    }


def remove_technical_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Menghapus path dan kolom teknis file dari tampilan utama."""

    if dataframe.empty:
        return dataframe

    keywords = {
        "path",
        "checkpoint",
        "directory",
        "folder",
        "file",
    }

    visible_columns = [
        column
        for column in dataframe.columns
        if not any(
            keyword in str(column).strip().lower()
            for keyword in keywords
        )
    ]

    return dataframe[visible_columns].copy()


def clean_result_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Membersihkan tabel hasil evaluasi untuk dashboard."""

    table = remove_technical_columns(dataframe)

    if table.empty:
        return table

    rename_mapping = {
        "experiment_name": "Eksperimen",
        "dataset": "Dataset",
        "model": "Model",
        "scenario_code": "Skenario",
        "scenario_name": "Representasi Teks",
        "accuracy": "Accuracy",
        "precision_macro": "Precision Macro",
        "recall_macro": "Recall Macro",
        "f1_macro": "Macro F1",
        "inference_time_ms": "Waktu Inferensi (ms)",
    }

    actual_rename = {}
    for source, target in rename_mapping.items():
        column = find_column(table, [source])
        if column is not None:
            actual_rename[column] = target

    table = table.rename(columns=actual_rename)

    metric_columns = [
        column
        for column in table.columns
        if any(
            keyword in str(column).lower()
            for keyword in ["accuracy", "precision", "recall", "f1"]
        )
        and not any(
            keyword in str(column).lower()
            for keyword in ["change", "percentage_point", "count", "error"]
        )
    ]

    for column in metric_columns:
        table[column] = table[column].apply(
            lambda value: percentage(value) if pd.notna(value) else "-"
        )

    preferred = [
        "Eksperimen",
        "Dataset",
        "Model",
        "Skenario",
        "Representasi Teks",
        "Accuracy",
        "Precision Macro",
        "Recall Macro",
        "Macro F1",
        "Waktu Inferensi (ms)",
    ]

    ordered = [column for column in preferred if column in table.columns]
    remaining = [column for column in table.columns if column not in ordered]

    return table[ordered + remaining]


def prepare_chart_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Menyiapkan hasil eksperimen untuk grafik Plotly."""

    if dataframe.empty:
        return pd.DataFrame()

    experiment_column = find_column(
        dataframe,
        ["experiment_name", "experiment"],
    )
    accuracy_column = find_column(
        dataframe,
        ["accuracy", "test_accuracy"],
    )
    f1_column = find_column(
        dataframe,
        ["f1_macro", "macro_f1", "f1_score_macro"],
    )

    if experiment_column is None:
        return pd.DataFrame()

    result = dataframe.copy()
    result["Eksperimen"] = (
        result[experiment_column].astype(str).str.lower()
    )
    result["Model"] = np.where(
        result["Eksperimen"].str.startswith("cnn"),
        "CNN",
        "Attention-BiLSTM",
    )
    result["Dataset"] = np.where(
        result["Eksperimen"].str.contains("_k"),
        "Kompas",
        "AG News",
    )
    result["Skenario"] = (
        result["Eksperimen"].str.split("_").str[-1].str.upper()
    )

    if accuracy_column is not None:
        result["Accuracy (%)"] = (
            pd.to_numeric(result[accuracy_column], errors="coerce")
            .apply(
                lambda value: (
                    value * 100
                    if pd.notna(value) and value <= 1
                    else value
                )
            )
        )

    if f1_column is not None:
        result["Macro F1 (%)"] = (
            pd.to_numeric(result[f1_column], errors="coerce")
            .apply(
                lambda value: (
                    value * 100
                    if pd.notna(value) and value <= 1
                    else value
                )
            )
        )

    return result


def style_static_chart(
    figure: go.Figure,
    y_title: str,
    y_range: list[float] | None = None,
    height: int = 510,
) -> go.Figure:
    """Merapikan grafik Plotly tanpa animasi."""

    figure.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=30, r=30, t=80, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        legend_title_text="",
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
        ),
    )

    figure.update_xaxes(
        title=None,
        showgrid=False,
    )
    figure.update_yaxes(
        title=y_title,
        range=y_range,
        gridcolor="rgba(148, 163, 184, 0.20)",
        zeroline=False,
    )

    return figure


def add_bar_labels(figure: go.Figure) -> None:
    """Menambahkan label persentase di atas batang."""

    figure.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        cliponaxis=False,
    )


def show_result_explanation(
    how_to_read: str,
    main_result: str,
    interpretation: str,
    conclusion: str,
) -> None:
    """
    Menampilkan interpretasi secara ringkas agar halaman tetap terlihat
    seperti dashboard implementasi, bukan laporan panjang.
    """

    st.caption(f"**Ringkasan hasil:** {main_result}")

    with st.expander("Lihat cara membaca dan interpretasi"):
        st.markdown(f"**Cara membaca:** {how_to_read}")
        st.markdown(f"**Interpretasi:** {interpretation}")
        st.markdown(f"**Kesimpulan:** {conclusion}")


# =============================================================================
# DATA LOADERS UNTUK EDA, PREDICTION, DAN TRAINING HISTORY
# =============================================================================

@st.cache_data(show_spinner=False)
def load_first_csv(candidates: tuple[str, ...]) -> pd.DataFrame:
    """Membaca file CSV pertama yang ditemukan."""

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                continue

    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_kompas_eda_data() -> pd.DataFrame:
    """Membaca data Kompas untuk visualisasi EDA langsung di app.py."""

    processed_candidates = (
        str(PROJECT_ROOT / "2_data" / "processed" / "kompas_clean.csv"),
        str(PROJECT_ROOT / "2_data" / "processed" / "kompas_processed.csv"),
        str(PROJECT_ROOT / "2_data" / "processed" / "kompas_preprocessed.csv"),
        str(PROJECT_ROOT / "2_data" / "processed" / "kompas_with_keyword.csv"),
    )

    dataframe = load_first_csv(processed_candidates)

    if dataframe.empty:
        raw_frames: list[pd.DataFrame] = []
        raw_mapping = {
            "bola": PROJECT_ROOT / "2_data" / "raw" / "kompas_bola_2500.csv",
            "global": PROJECT_ROOT / "2_data" / "raw" / "kompas_global_2500.csv",
            "money": PROJECT_ROOT / "2_data" / "raw" / "kompas_money_2500.csv",
            "tekno": PROJECT_ROOT / "2_data" / "raw" / "kompas_tekno_2500.csv",
        }

        for category, path in raw_mapping.items():
            if not path.exists():
                continue

            try:
                frame = pd.read_csv(path)
            except Exception:
                continue

            category_column = find_column(
                frame,
                ["category", "kategori", "label"],
            )
            if category_column is None:
                frame["category"] = category

            raw_frames.append(frame)

        if raw_frames:
            dataframe = pd.concat(raw_frames, ignore_index=True)

    if dataframe.empty:
        return dataframe

    date_column = find_column(
        dataframe,
        [
            "date",
            "tanggal",
            "published_date",
            "publication_date",
            "published_at",
        ],
    )
    category_column = find_column(
        dataframe,
        ["category", "kategori", "label"],
    )

    if date_column is not None:
        dataframe[date_column] = pd.to_datetime(
            dataframe[date_column],
            errors="coerce",
        )
        dataframe = dataframe.rename(columns={date_column: "Tanggal"})

    if category_column is not None:
        dataframe = dataframe.rename(columns={category_column: "Kategori"})
        dataframe["Kategori"] = (
            dataframe["Kategori"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    return dataframe


@st.cache_data(show_spinner=False)
def load_agnews_eda_data() -> pd.DataFrame:
    """Membaca data train AG News untuk EDA langsung di dashboard."""

    candidates = (
        str(PROJECT_ROOT / "2_data" / "processed" / "ag_news_train_clean.csv"),
        str(PROJECT_ROOT / "2_data" / "processed" / "ag_news_train_processed.csv"),
        str(PROJECT_ROOT / "2_data" / "processed" / "ag_news_train_preprocessed.csv"),
        str(PROJECT_ROOT / "2_data" / "raw" / "train dataset AG News.csv"),
    )

    dataframe = load_first_csv(candidates)

    if dataframe.empty:
        return dataframe

    label_column = find_column(
        dataframe,
        ["category", "kategori", "label", "class", "class_index"],
    )

    if label_column is not None:
        dataframe = dataframe.rename(columns={label_column: "Kategori"})

        def normalize_ag_label(value: Any) -> str:
            if pd.isna(value):
                return ""

            text = str(value).strip().lower()

            try:
                numeric = int(float(text))
                mapping_one_based = {
                    1: "world",
                    2: "sports",
                    3: "business",
                    4: "sci_tech",
                }
                mapping_zero_based = {
                    0: "world",
                    1: "sports",
                    2: "business",
                    3: "sci_tech",
                }

                if numeric in mapping_one_based:
                    return mapping_one_based[numeric]

                if numeric in mapping_zero_based:
                    return mapping_zero_based[numeric]
            except (TypeError, ValueError):
                pass

            return (
                text
                .replace("sci/tech", "sci_tech")
                .replace("sci-tech", "sci_tech")
                .replace(" ", "_")
            )

        dataframe["Kategori"] = dataframe["Kategori"].apply(
            normalize_ag_label
        )

    return dataframe


def find_text_column(
    dataframe: pd.DataFrame,
    field_name: str,
) -> str | None:
    """Mencari kolom teks mentah atau hasil preprocessing."""

    aliases = {
        "title": [
            "title_clean",
            "clean_title",
            "title_processed",
            "processed_title",
            "title",
            "judul",
        ],
        "description": [
            "description_clean",
            "clean_description",
            "description_processed",
            "processed_description",
            "description",
            "desc",
            "deskripsi",
        ],
        "content": [
            "content_clean",
            "clean_content",
            "content_processed",
            "processed_content",
            "content",
            "isi",
        ],
    }

    return find_column(
        dataframe,
        aliases.get(field_name, [field_name]),
    )


def word_count_series(
    dataframe: pd.DataFrame,
    column_name: str | None,
) -> pd.Series:
    """Menghitung jumlah kata dengan aman."""

    if column_name is None or column_name not in dataframe.columns:
        return pd.Series(dtype="int64")

    return (
        dataframe[column_name]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
        .astype(int)
    )


def create_text_length_figure(
    dataframe: pd.DataFrame,
    dataset_name: str,
    include_content: bool,
) -> tuple[go.Figure | None, dict[str, float]]:
    """Membuat histogram panjang teks langsung dari DataFrame."""

    components = ["title", "description"]

    if include_content:
        components.append("content")

    frames: list[pd.DataFrame] = []
    summary: dict[str, float] = {}

    display_names = {
        "title": "Title",
        "description": "Description",
        "content": "Content",
    }

    for component in components:
        column_name = find_text_column(dataframe, component)
        counts = word_count_series(dataframe, column_name)

        if counts.empty:
            continue

        component_name = display_names[component]
        summary[component_name] = float(counts.mean())

        frames.append(
            pd.DataFrame(
                {
                    "Jumlah Kata": counts,
                    "Komponen": component_name,
                }
            )
        )

    if not frames:
        return None, summary

    length_data = pd.concat(frames, ignore_index=True)

    figure = px.histogram(
        length_data,
        x="Jumlah Kata",
        facet_row="Komponen",
        nbins=30,
        title=f"Distribusi Panjang Teks — {dataset_name}",
        labels={"count": "Frekuensi"},
    )
    figure.update_layout(
        template="plotly_white",
        height=250 + (220 * len(frames)),
        margin=dict(l=35, r=25, t=75, b=40),
        showlegend=False,
        bargap=0.08,
    )
    figure.update_xaxes(title="Jumlah Kata")
    figure.update_yaxes(title="Frekuensi")
    figure.for_each_annotation(
        lambda annotation: annotation.update(
            text=annotation.text.replace("Komponen=", "")
        )
    )

    return figure, summary


BASIC_STOPWORDS = {
    # Indonesia
    "yang", "dan", "di", "ke", "dari", "untuk", "pada", "dengan",
    "ini", "itu", "atau", "sebagai", "dalam", "adalah", "akan", "oleh",
    "karena", "juga", "tidak", "saat", "telah", "lebih", "setelah",
    "hingga", "antara", "terhadap", "bisa", "ada", "jadi", "kami",
    # English
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "of",
    "for", "from", "with", "by", "is", "are", "was", "were", "be",
    "been", "has", "have", "had", "as", "that", "this", "it", "its",
    "after", "new", "says", "said",
}


def create_top_words_figure(
    dataframe: pd.DataFrame,
    dataset_name: str,
    top_n: int = 20,
) -> tuple[go.Figure | None, pd.DataFrame]:
    """Membuat frekuensi kata langsung dari Title + Description."""

    text_columns = [
        find_text_column(dataframe, "title"),
        find_text_column(dataframe, "description"),
    ]
    text_columns = [
        column for column in text_columns
        if column is not None
    ]

    if not text_columns:
        return None, pd.DataFrame()

    combined_text = (
        dataframe[text_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )

    counter: Counter[str] = Counter()

    for text_value in combined_text:
        tokens = re.findall(
            r"[a-zA-ZÀ-ÿ0-9]+",
            text_value,
        )

        counter.update(
            token
            for token in tokens
            if len(token) > 2
            and token not in BASIC_STOPWORDS
            and not token.isdigit()
        )

    top_words = pd.DataFrame(
        counter.most_common(top_n),
        columns=["Kata", "Frekuensi"],
    )

    if top_words.empty:
        return None, top_words

    figure = px.bar(
        top_words.sort_values("Frekuensi"),
        x="Frekuensi",
        y="Kata",
        orientation="h",
        text="Frekuensi",
        title=f"{top_n} Kata Paling Sering Muncul — {dataset_name}",
    )
    figure.update_traces(
        textposition="outside",
        cliponaxis=False,
    )
    figure.update_layout(
        template="plotly_white",
        height=620,
        margin=dict(l=35, r=45, t=75, b=40),
        showlegend=False,
    )

    return figure, top_words


@st.cache_data(show_spinner=False)
def load_prediction_result(experiment_name: str) -> pd.DataFrame:
    """Membaca file prediksi eksperimen."""

    candidates = (
        str(
            PROJECT_ROOT
            / "9_results"
            / "predictions"
            / f"{experiment_name}_predictions.csv"
        ),
        str(
            PROJECT_ROOT
            / "9_results"
            / "backup_k2_length40"
            / f"{experiment_name}_predictions.csv"
        ),
    )

    return load_first_csv(candidates)


@st.cache_data(show_spinner=False)
def load_training_history(experiment_name: str) -> pd.DataFrame:
    """Membaca training history untuk membuat grafik langsung di app.py."""

    candidates = (
        str(
            PROJECT_ROOT
            / "9_results"
            / "training_history"
            / f"{experiment_name}_history.csv"
        ),
        str(
            PROJECT_ROOT
            / "9_results"
            / "training_history"
            / f"{experiment_name}.csv"
        ),
        str(
            PROJECT_ROOT
            / "9_results"
            / "logs"
            / f"{experiment_name}_training.log.csv"
        ),
        str(
            PROJECT_ROOT
            / "9_results"
            / "backup_k2_length40"
            / f"{experiment_name}_history.csv"
        ),
    )

    return load_first_csv(candidates)


# =============================================================================
# CONFUSION MATRIX HELPERS
# =============================================================================

def normalize_prediction_labels(
    values: pd.Series,
    dataset_name: str,
) -> pd.Series:
    """Mengubah label numerik atau teks menjadi label internal."""

    mapping = NUMERIC_LABEL_MAPPING[dataset_name]

    def normalize_one(value: Any) -> str:
        if pd.isna(value):
            return ""

        text = str(value).strip().lower()

        try:
            numeric = int(float(text))
            if numeric in mapping:
                return mapping[numeric]
        except (TypeError, ValueError):
            pass

        text = text.replace("sci/tech", "sci_tech")
        text = text.replace("sci-tech", "sci_tech")
        text = text.replace(" ", "_")

        return text

    return values.apply(normalize_one)


def create_confusion_matrix_chart(
    prediction_data: pd.DataFrame,
    dataset_name: str,
    normalized: bool,
) -> tuple[go.Figure | None, dict[str, Any] | None]:
    """Membuat confusion matrix Plotly dan ringkasan otomatis."""

    actual_column = find_column(
        prediction_data,
        [
            "actual_label",
            "actual_class",
            "y_true",
            "true_label",
            "actual",
            "label_actual",
        ],
    )
    predicted_column = find_column(
        prediction_data,
        [
            "predicted_label",
            "predicted_class",
            "y_pred",
            "prediction",
            "predicted",
            "label_predicted",
        ],
    )

    if actual_column is None or predicted_column is None:
        return None, None

    labels = LABELS_BY_DATASET[dataset_name]
    y_true = normalize_prediction_labels(
        prediction_data[actual_column],
        dataset_name,
    )
    y_pred = normalize_prediction_labels(
        prediction_data[predicted_column],
        dataset_name,
    )

    valid_mask = y_true.isin(labels) & y_pred.isin(labels)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if y_true.empty:
        return None, None

    matrix_count = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    total_data = int(matrix_count.sum())
    total_correct = int(np.trace(matrix_count))
    total_wrong = total_data - total_correct
    accuracy_value = total_correct / total_data if total_data else 0.0

    if normalized:
        row_total = matrix_count.sum(axis=1, keepdims=True)
        matrix_display = np.divide(
            matrix_count,
            row_total,
            out=np.zeros_like(matrix_count, dtype=float),
            where=row_total != 0,
        ) * 100
        annotations = np.array(
            [
                [f"{value:.1f}%" for value in row]
                for row in matrix_display
            ]
        )
        color_title = "Persentase"
        hover_template = (
            "<b>Aktual:</b> %{y}<br>"
            "<b>Prediksi:</b> %{x}<br>"
            "<b>Nilai:</b> %{z:.2f}%"
            "<extra></extra>"
        )
    else:
        matrix_display = matrix_count
        annotations = matrix_count.astype(str)
        color_title = "Jumlah Data"
        hover_template = (
            "<b>Aktual:</b> %{y}<br>"
            "<b>Prediksi:</b> %{x}<br>"
            "<b>Jumlah:</b> %{z:.0f} data"
            "<extra></extra>"
        )

    error_matrix = matrix_count.copy()
    np.fill_diagonal(error_matrix, 0)

    largest_error = int(error_matrix.max())
    if largest_error > 0:
        error_position = np.unravel_index(
            int(np.argmax(error_matrix)),
            error_matrix.shape,
        )
        actual_error_class = labels[error_position[0]]
        predicted_error_class = labels[error_position[1]]
    else:
        actual_error_class = "-"
        predicted_error_class = "-"

    display_labels = [display_label(label) for label in labels]

    figure = go.Figure(
        data=go.Heatmap(
            z=matrix_display,
            x=display_labels,
            y=display_labels,
            text=annotations,
            texttemplate="%{text}",
            hovertemplate=hover_template,
            colorbar=dict(title=color_title),
            colorscale="Blues",
        )
    )
    figure.update_layout(
        template="plotly_white",
        height=560,
        title=(
            "Confusion Matrix Normalized"
            if normalized
            else "Confusion Matrix Jumlah"
        ),
        xaxis_title="Kelas Prediksi",
        yaxis_title="Kelas Aktual",
        margin=dict(l=40, r=30, t=75, b=45),
    )
    figure.update_yaxes(autorange="reversed")

    summary = {
        "total_data": total_data,
        "correct": total_correct,
        "wrong": total_wrong,
        "accuracy": accuracy_value,
        "largest_error": largest_error,
        "actual_error_class": actual_error_class,
        "predicted_error_class": predicted_error_class,
    }

    return figure, summary


# =============================================================================
# TRAINING CURVE HELPER
# =============================================================================

def create_training_curve_charts(
    history: pd.DataFrame,
) -> tuple[go.Figure | None, go.Figure | None, dict[str, Any] | None]:
    """Membuat grafik accuracy dan loss dari training history."""

    if history.empty:
        return None, None, None

    epoch_column = find_column(history, ["epoch", "epochs"])
    accuracy_column = find_column(history, ["accuracy", "acc"])
    val_accuracy_column = find_column(
        history,
        ["val_accuracy", "validation_accuracy", "val_acc"],
    )
    loss_column = find_column(history, ["loss", "train_loss"])
    val_loss_column = find_column(
        history,
        ["val_loss", "validation_loss"],
    )

    if epoch_column is None:
        history = history.copy()
        history["Epoch"] = np.arange(1, len(history) + 1)
        epoch_column = "Epoch"
    else:
        history = history.copy()
        epoch_numeric = pd.to_numeric(history[epoch_column], errors="coerce")
        if epoch_numeric.min(skipna=True) == 0:
            history[epoch_column] = epoch_numeric + 1

    accuracy_figure: go.Figure | None = None
    loss_figure: go.Figure | None = None

    if accuracy_column is not None and val_accuracy_column is not None:
        accuracy_plot = history[
            [epoch_column, accuracy_column, val_accuracy_column]
        ].copy()
        accuracy_plot = accuracy_plot.rename(
            columns={
                accuracy_column: "Train Accuracy",
                val_accuracy_column: "Validation Accuracy",
            }
        )
        accuracy_long = accuracy_plot.melt(
            id_vars=[epoch_column],
            var_name="Kurva",
            value_name="Accuracy",
        )
        accuracy_figure = px.line(
            accuracy_long,
            x=epoch_column,
            y="Accuracy",
            color="Kurva",
            markers=True,
            title="Perubahan Accuracy Train dan Validation",
        )
        accuracy_figure = style_static_chart(
            accuracy_figure,
            y_title="Accuracy",
            height=460,
        )

    if loss_column is not None and val_loss_column is not None:
        loss_plot = history[
            [epoch_column, loss_column, val_loss_column]
        ].copy()
        loss_plot = loss_plot.rename(
            columns={
                loss_column: "Train Loss",
                val_loss_column: "Validation Loss",
            }
        )
        loss_long = loss_plot.melt(
            id_vars=[epoch_column],
            var_name="Kurva",
            value_name="Loss",
        )
        loss_figure = px.line(
            loss_long,
            x=epoch_column,
            y="Loss",
            color="Kurva",
            markers=True,
            title="Perubahan Loss Train dan Validation",
        )
        loss_figure = style_static_chart(
            loss_figure,
            y_title="Loss",
            height=460,
        )

    if val_loss_column is not None:
        val_loss_numeric = pd.to_numeric(
            history[val_loss_column],
            errors="coerce",
        )
        if val_loss_numeric.notna().any():
            best_position = int(val_loss_numeric.idxmin())
            best_epoch = history.loc[best_position, epoch_column]
            best_val_loss = float(val_loss_numeric.loc[best_position])
        else:
            best_epoch = "-"
            best_val_loss = np.nan
    else:
        best_epoch = "-"
        best_val_loss = np.nan

    best_val_accuracy = np.nan
    if val_accuracy_column is not None:
        val_accuracy_numeric = pd.to_numeric(
            history[val_accuracy_column],
            errors="coerce",
        )
        if val_accuracy_numeric.notna().any():
            best_val_accuracy = float(val_accuracy_numeric.max())

    summary = {
        "epochs": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
    }

    return accuracy_figure, loss_figure, summary


# =============================================================================
# SHAP TABLE HELPERS
# =============================================================================

def simplify_global_shap_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Menyederhanakan kolom Global SHAP untuk tampilan utama."""

    candidates = [
        "rank",
        "token",
        "total_abs_shap",
        "mean_abs_shap",
        "occurrence_count",
        "normalized_global_importance",
    ]
    columns = [column for column in candidates if column in dataframe.columns]
    result = dataframe[columns].copy()
    result = result.rename(
        columns={
            "rank": "Peringkat",
            "token": "Token",
            "total_abs_shap": "Total |SHAP|",
            "mean_abs_shap": "Rata-rata |SHAP|",
            "occurrence_count": "Jumlah Kemunculan",
            "normalized_global_importance": "Proporsi Importance",
        }
    )
    return result


def simplify_local_shap_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Menyederhanakan kolom Local SHAP untuk tampilan utama."""

    candidates = [
        "document_id",
        "selection_type",
        "actual_label",
        "predicted_label",
        "is_correct",
        "prediction_confidence",
    ]
    columns = [column for column in candidates if column in dataframe.columns]
    result = dataframe[columns].copy()
    result = result.rename(
        columns={
            "document_id": "ID Artikel",
            "selection_type": "Jenis Sampel",
            "actual_label": "Label Aktual",
            "predicted_label": "Label Prediksi",
            "is_correct": "Prediksi Benar",
            "prediction_confidence": "Confidence",
        }
    )

    if "Jenis Sampel" in result.columns:
        result["Jenis Sampel"] = result["Jenis Sampel"].replace(
            {
                "correct_high_confidence": "Benar — confidence tinggi",
                "incorrect_high_confidence": "Salah — confidence tinggi",
            }
        )

    if "Label Aktual" in result.columns:
        result["Label Aktual"] = result["Label Aktual"].apply(display_label)

    if "Label Prediksi" in result.columns:
        result["Label Prediksi"] = result["Label Prediksi"].apply(display_label)

    if "Confidence" in result.columns:
        result["Confidence"] = result["Confidence"].apply(percentage)

    return result


def parse_waterfall_filename(path: Path) -> dict[str, str]:
    """Mengambil ID, tipe sampel, label aktual, dan prediksi dari filename."""

    stem = path.stem
    pattern = re.compile(
        r"^(?P<document_id>KMP-\d+)_"
        r"(?P<selection_type>correct_high_confidence|incorrect_high_confidence)_"
        r"actual_(?P<actual>[^_]+)_pred_(?P<predicted>[^_]+)_waterfall$"
    )
    match = pattern.match(stem)

    if match is None:
        return {
            "document_id": stem,
            "selection_type": "-",
            "actual": "-",
            "predicted": "-",
        }

    return match.groupdict()


# =============================================================================
# LOAD RESEARCH RESULTS
# =============================================================================

evaluation_data = load_test_evaluation()
chart_data = prepare_chart_data(evaluation_data)

model_comparison = load_model_comparison()
scenario_comparison = load_scenario_comparison()
description_contribution = load_description_contribution()
yake_contribution = load_yake_contribution()
misclassification_data = load_misclassification_analysis()

global_shap = load_global_shap()
global_shap_by_class = load_global_shap_by_class()
local_shap_summary = load_local_shap_summary()
local_token_contributions = load_local_token_contributions()
waterfall_summary = load_waterfall_summary()

kompas_eda_data = load_kompas_eda_data()
agnews_eda_data = load_agnews_eda_data()


# =============================================================================
# HEADER
# =============================================================================

st.title("📰 Dashboard Klasifikasi Berita Berbahasa Indonesia")
st.caption(RESEARCH_TITLE)

st.info(
    "Dashboard ini mengintegrasikan dataset, hasil 10 eksperimen, "
    "perbandingan CNN dan Attention-BiLSTM, pengujian tanpa dan dengan "
    "keyword YAKE, prediksi berita baru, serta interpretasi SHAP."
)


# =============================================================================
# SUMMARY METRICS
# =============================================================================

kompas_info = DATASET_INFORMATION["Kompas"]
cnn_k2_info = MODEL_PERFORMANCE["CNN K2"]

summary_1, summary_2, summary_3, summary_4 = st.columns(4)

summary_1.metric(
    "Data Kompas Setelah Cleaning",
    f"{number_id(kompas_info['jumlah_data_setelah_cleaning'])} artikel",
    help=(
        "Jumlah dataset utama setelah tiga data duplikat dihapus "
        "dari 10.000 artikel awal."
    ),
)

summary_2.metric(
    "Eksperimen Utama",
    "10 eksperimen",
    help=(
        "Enam eksperimen pada Kompas dan empat eksperimen pada AG News."
    ),
)

summary_3.metric(
    "Model Terbaik pada Test Set",
    "CNN K2",
    help=(
        "CNN dengan representasi Title + Description tanpa keyword YAKE "
        "dan sequence length 60."
    ),
)

summary_4.metric(
    "Accuracy Test Terbaik",
    percentage(cnn_k2_info["accuracy"]),
    help=(
        "CNN K2 memprediksi benar 958 dari 1.000 artikel test Kompas."
    ),
)


# =============================================================================
# MAIN TABS
# =============================================================================

(
    tab_summary,
    tab_dataset,
    tab_results,
    tab_yake,
    tab_prediction,
    tab_shap,
) = st.tabs(
    [
        "🏠 Ringkasan",
        "📚 Dataset & EDA",
        "📊 Hasil & Metrik",
        "🔑 Perbandingan YAKE",
        "📰 Prediksi Berita",
        "🔍 Explainable AI",
    ]
)


# =============================================================================
# TAB 1 - RINGKASAN
# =============================================================================

with tab_summary:
    st.header("Ringkasan Penelitian")

    overview_1, overview_2 = st.columns(2)

    with overview_1:
        with st.container(border=True):
            st.subheader("Dataset Penelitian")
            st.markdown(
                """
                **Kompas** digunakan sebagai dataset utama karena berisi
                berita berbahasa Indonesia. Dataset terdiri dari kategori
                **Bola, Global, Money, dan Tekno**.

                **AG News** digunakan sebagai benchmark untuk melihat
                konsistensi pola performa kedua model pada dataset internasional.
                """
            )

    with overview_2:
        with st.container(border=True):
            st.subheader("Desain Eksperimen")
            st.markdown(
                """
                Dua arsitektur dibandingkan, yaitu **CNN** dan
                **Attention-BiLSTM**.

                - **K1:** Title
                - **K2:** Title + Description, tanpa YAKE
                - **K3:** Title + Description + Keyword YAKE
                - **A1:** Title
                - **A2:** Title + Description
                """
            )

    st.subheader("Temuan Utama")
    finding_1, finding_2, finding_3 = st.columns(3)

    with finding_1:
        st.success(
            "**Description meningkatkan performa**\n\n"
            "CNN Kompas meningkat dari 94,70% menjadi 95,80%."
        )

    with finding_2:
        st.warning(
            "**YAKE belum meningkatkan accuracy**\n\n"
            "CNN turun 0,80 pp dan Attention-BiLSTM turun 0,10 pp."
        )

    with finding_3:
        st.info(
            "**CNN K2 menjadi model terbaik**\n\n"
            "Accuracy 95,80% dan Macro F1 95,81%."
        )

    st.subheader("Perbandingan Model Utama pada Kompas")

    if chart_data.empty or "Accuracy (%)" not in chart_data.columns:
        st.info("Data grafik evaluasi belum tersedia.")
    else:
        kompas_chart = chart_data[
            chart_data["Dataset"].eq("Kompas")
        ].copy()

        figure = px.bar(
            kompas_chart,
            x="Skenario",
            y="Accuracy (%)",
            color="Model",
            barmode="group",
            text="Accuracy (%)",
            title="Accuracy CNN dan Attention-BiLSTM pada Dataset Kompas",
            category_orders={
                "Skenario": ["K1", "K2", "K3"],
                "Model": ["CNN", "Attention-BiLSTM"],
            },
            hover_data={
                "Eksperimen": True,
                "Accuracy (%)": ":.2f",
            },
        )
        add_bar_labels(figure)
        figure = style_static_chart(
            figure,
            y_title="Accuracy (%)",
            y_range=[90, 100],
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

        show_result_explanation(
            how_to_read=(
                "Sumbu horizontal menunjukkan K1, K2, dan K3. "
                "Setiap skenario memiliki dua batang untuk CNN dan "
                "Attention-BiLSTM. Sumbu vertikal menunjukkan accuracy test."
            ),
            main_result=(
                "CNN unggul pada K1 dan K2, sedangkan Attention-BiLSTM "
                "sedikit unggul pada K3. Nilai tertinggi adalah CNN K2 "
                "sebesar 95,80%."
            ),
            interpretation=(
                "Description menambah konteks yang tidak selalu tersedia "
                "pada Title. Keyword YAKE belum memberikan informasi baru "
                "yang cukup pada K3."
            ),
            conclusion=(
                "CNN K2 dengan Title + Description dipilih sebagai model final."
            ),
        )


# =============================================================================
# TAB 2 - DATASET & EDA
# =============================================================================

with tab_dataset:
    st.header("Dataset dan Exploratory Data Analysis")

    dataset_1, dataset_2 = st.columns(2)

    with dataset_1:
        with st.container(border=True):
            st.subheader("Dataset Kompas")
            st.markdown(
                f"""
                **Fungsi:** dataset utama penelitian  
                **Bahasa:** Indonesia  
                **Jumlah awal:** {number_id(kompas_info['jumlah_data_awal'])} artikel  
                **Setelah cleaning:** {number_id(kompas_info['jumlah_data_setelah_cleaning'])} artikel  
                **Kategori:** Bola, Global, Money, Tekno
                """
            )
            st.caption(
                "Data Kompas digunakan untuk melatih dan menguji "
                "klasifikasi berita berbahasa Indonesia."
            )

    agnews_info = DATASET_INFORMATION["AG News"]

    with dataset_2:
        with st.container(border=True):
            st.subheader("Dataset AG News")
            st.markdown(
                f"""
                **Fungsi:** dataset benchmark  
                **Bahasa:** Inggris  
                **Train setelah cleaning:** {number_id(agnews_info['jumlah_data_train_setelah_cleaning'])} artikel  
                **Test setelah cleaning:** {number_id(agnews_info['jumlah_data_test_setelah_cleaning'])} artikel  
                **Kategori:** Business, Sci/Tech, Sports, World
                """
            )
            st.caption(
                "AG News digunakan untuk membandingkan pola performa "
                "pada dataset internasional yang lebih besar."
            )

    eda_tab_1, eda_tab_2, eda_tab_3, eda_tab_4 = st.tabs(
        [
            "Distribusi Kelas",
            "Panjang Teks",
            "Frekuensi Kata",
            "Distribusi Waktu",
        ]
    )

    with eda_tab_1:
        col_1, col_2 = st.columns(2)

        with col_1:
            st.subheader("Distribusi Kelas Kompas")

            if (
                not kompas_eda_data.empty
                and "Kategori" in kompas_eda_data.columns
            ):
                class_data = (
                    kompas_eda_data["Kategori"]
                    .value_counts()
                    .rename_axis("Kategori")
                    .reset_index(name="Jumlah Artikel")
                )
                class_data["Kategori Tampilan"] = (
                    class_data["Kategori"].apply(display_label)
                )

                figure = px.bar(
                    class_data,
                    x="Kategori Tampilan",
                    y="Jumlah Artikel",
                    text="Jumlah Artikel",
                    title="Jumlah Artikel pada Setiap Kategori Kompas",
                )
                figure.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                )
                figure = style_static_chart(
                    figure,
                    y_title="Jumlah Artikel",
                    height=480,
                )

                st.plotly_chart(
                    figure,
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )

                largest = class_data.loc[
                    class_data["Jumlah Artikel"].idxmax()
                ]
                smallest = class_data.loc[
                    class_data["Jumlah Artikel"].idxmin()
                ]

                show_result_explanation(
                    how_to_read=(
                        "Sumbu horizontal menunjukkan kategori dan sumbu "
                        "vertikal menunjukkan jumlah artikel."
                    ),
                    main_result=(
                        f"Kategori terbanyak adalah "
                        f"**{display_label(largest['Kategori'])}** "
                        f"({number_id(largest['Jumlah Artikel'])} artikel), "
                        f"sedangkan yang paling sedikit adalah "
                        f"**{display_label(smallest['Kategori'])}** "
                        f"({number_id(smallest['Jumlah Artikel'])} artikel)."
                    ),
                    interpretation=(
                        "Selisih antarkelas sangat kecil karena proses crawling "
                        "menargetkan sekitar 2.500 artikel per kategori."
                    ),
                    conclusion=(
                        "Distribusi yang hampir seimbang mengurangi risiko "
                        "model terlalu dominan pada satu kategori."
                    ),
                )
            else:
                st.info(
                    "Data Kompas belum tersedia untuk membuat grafik "
                    "distribusi kelas."
                )

        with col_2:
            st.subheader("Distribusi Kelas AG News")

            if (
                not agnews_eda_data.empty
                and "Kategori" in agnews_eda_data.columns
            ):
                ag_class_data = (
                    agnews_eda_data["Kategori"]
                    .value_counts()
                    .rename_axis("Kategori")
                    .reset_index(name="Jumlah Artikel")
                )
                ag_class_data["Kategori Tampilan"] = (
                    ag_class_data["Kategori"].apply(display_label)
                )

                figure = px.bar(
                    ag_class_data,
                    x="Kategori Tampilan",
                    y="Jumlah Artikel",
                    text="Jumlah Artikel",
                    title="Jumlah Artikel pada Setiap Kategori AG News",
                )
                figure.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                )
                figure = style_static_chart(
                    figure,
                    y_title="Jumlah Artikel",
                    height=480,
                )

                st.plotly_chart(
                    figure,
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )

                max_count = int(
                    ag_class_data["Jumlah Artikel"].max()
                )
                min_count = int(
                    ag_class_data["Jumlah Artikel"].min()
                )

                show_result_explanation(
                    how_to_read=(
                        "Sumbu horizontal menunjukkan empat kategori AG News "
                        "dan sumbu vertikal menunjukkan jumlah data train."
                    ),
                    main_result=(
                        f"Jumlah per kelas berada pada rentang "
                        f"**{number_id(min_count)}–{number_id(max_count)} "
                        f"artikel**."
                    ),
                    interpretation=(
                        "Distribusi AG News disusun seimbang sehingga setiap "
                        "kelas mempunyai kontribusi yang setara saat training."
                    ),
                    conclusion=(
                        "AG News sesuai digunakan sebagai benchmark "
                        "perbandingan model."
                    ),
                )
            else:
                st.info(
                    "Data train AG News belum tersedia untuk membuat "
                    "grafik distribusi kelas."
                )

    with eda_tab_2:
        col_1, col_2 = st.columns(2)

        with col_1:
            st.subheader("Distribusi Panjang Teks Kompas")

            kompas_length_figure, kompas_length_summary = (
                create_text_length_figure(
                    kompas_eda_data,
                    dataset_name="Kompas",
                    include_content=True,
                )
            )

            if kompas_length_figure is None:
                st.info(
                    "Kolom Title, Description, atau Content Kompas "
                    "belum ditemukan."
                )
            else:
                st.plotly_chart(
                    kompas_length_figure,
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )

                summary_text = ", ".join(
                    f"{component} rata-rata "
                    f"{value:.2f} kata"
                    for component, value
                    in kompas_length_summary.items()
                )

                show_result_explanation(
                    how_to_read=(
                        "Setiap histogram menunjukkan banyaknya artikel "
                        "berdasarkan jumlah kata pada komponen teks."
                    ),
                    main_result=summary_text + ".",
                    interpretation=(
                        "Title paling ringkas, Description menambah konteks, "
                        "dan Content jauh lebih panjang."
                    ),
                    conclusion=(
                        "Penelitian final memakai Title + Description karena "
                        "lebih informatif namun tetap lebih ringkas dibanding "
                        "Content."
                    ),
                )

        with col_2:
            st.subheader("Distribusi Panjang Teks AG News")

            ag_length_figure, ag_length_summary = (
                create_text_length_figure(
                    agnews_eda_data,
                    dataset_name="AG News",
                    include_content=False,
                )
            )

            if ag_length_figure is None:
                st.info(
                    "Kolom Title atau Description AG News belum ditemukan."
                )
            else:
                st.plotly_chart(
                    ag_length_figure,
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )

                summary_text = ", ".join(
                    f"{component} rata-rata "
                    f"{value:.2f} kata"
                    for component, value
                    in ag_length_summary.items()
                )

                show_result_explanation(
                    how_to_read=(
                        "Histogram menunjukkan frekuensi artikel pada setiap "
                        "rentang jumlah kata."
                    ),
                    main_result=summary_text + ".",
                    interpretation=(
                        "Description AG News lebih panjang daripada Title "
                        "dan menyediakan konteks tambahan."
                    ),
                    conclusion=(
                        "Hal ini mendukung pengujian A2 sebagai representasi "
                        "yang lebih lengkap dibanding A1."
                    ),
                )

    with eda_tab_3:
        col_1, col_2 = st.columns(2)

        with col_1:
            st.subheader("Kata yang Sering Muncul pada Kompas")

            kompas_words_figure, kompas_words_data = (
                create_top_words_figure(
                    kompas_eda_data,
                    dataset_name="Kompas",
                    top_n=20,
                )
            )

            if kompas_words_figure is None:
                st.info(
                    "Kolom Title dan Description Kompas belum ditemukan."
                )
            else:
                st.plotly_chart(
                    kompas_words_figure,
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )

                top_word = kompas_words_data.iloc[0]

                show_result_explanation(
                    how_to_read=(
                        "Panjang batang menunjukkan frekuensi kata pada "
                        "gabungan Title dan Description."
                    ),
                    main_result=(
                        f"Kata paling sering muncul adalah "
                        f"**{top_word['Kata']}** sebanyak "
                        f"**{number_id(top_word['Frekuensi'])} kali**."
                    ),
                    interpretation=(
                        "Frekuensi tinggi menunjukkan topik yang dominan pada "
                        "periode crawling, bukan otomatis kata paling penting "
                        "bagi keputusan model."
                    ),
                    conclusion=(
                        "Pengaruh kata terhadap prediksi dijelaskan secara "
                        "terpisah menggunakan SHAP."
                    ),
                )

        with col_2:
            st.subheader("Kata yang Sering Muncul pada AG News")

            ag_words_figure, ag_words_data = create_top_words_figure(
                agnews_eda_data,
                dataset_name="AG News",
                top_n=20,
            )

            if ag_words_figure is None:
                st.info(
                    "Kolom Title dan Description AG News belum ditemukan."
                )
            else:
                st.plotly_chart(
                    ag_words_figure,
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )

                top_word = ag_words_data.iloc[0]

                show_result_explanation(
                    how_to_read=(
                        "Panjang batang menunjukkan frekuensi kata pada "
                        "Title dan Description data train AG News."
                    ),
                    main_result=(
                        f"Kata paling sering muncul adalah "
                        f"**{top_word['Kata']}** sebanyak "
                        f"**{number_id(top_word['Frekuensi'])} kali**."
                    ),
                    interpretation=(
                        "Kata-kata dominan mencerminkan topik Business, "
                        "Sci/Tech, Sports, dan World."
                    ),
                    conclusion=(
                        "Grafik ini digunakan untuk memahami karakteristik "
                        "dataset benchmark."
                    ),
                )

    with eda_tab_4:
        col_1, col_2 = st.columns(2)

        with col_1:
            st.subheader("Distribusi Bulanan Artikel Kompas")

            if (
                not kompas_eda_data.empty
                and "Tanggal" in kompas_eda_data.columns
                and "Kategori" in kompas_eda_data.columns
            ):
                valid_date_data = kompas_eda_data.dropna(
                    subset=["Tanggal"]
                ).copy()
                valid_date_data["Bulan"] = (
                    valid_date_data["Tanggal"]
                    .dt.to_period("M")
                    .astype(str)
                )

                monthly_data = (
                    valid_date_data
                    .groupby(["Bulan", "Kategori"], as_index=False)
                    .size()
                    .rename(columns={"size": "Jumlah Artikel"})
                )
                monthly_data["Kategori Tampilan"] = monthly_data[
                    "Kategori"
                ].apply(display_label)

                monthly_total = (
                    monthly_data.groupby("Bulan", as_index=False)[
                        "Jumlah Artikel"
                    ].sum()
                )
                peak_month = monthly_total.sort_values(
                    "Jumlah Artikel",
                    ascending=False,
                ).iloc[0]
                peak_category = monthly_data.sort_values(
                    "Jumlah Artikel",
                    ascending=False,
                ).iloc[0]

                figure = px.line(
                    monthly_data,
                    x="Bulan",
                    y="Jumlah Artikel",
                    color="Kategori Tampilan",
                    markers=True,
                    text="Jumlah Artikel",
                    title="Jumlah Artikel Kompas per Bulan dan Kategori",
                    category_orders={
                        "Kategori Tampilan": [
                            "Bola",
                            "Global",
                            "Money",
                            "Tekno",
                        ]
                    },
                )
                figure.update_traces(
                    textposition="top center",
                    hovertemplate=(
                        "<b>Bulan:</b> %{x}<br>"
                        "<b>Jumlah artikel:</b> %{y:,}"
                        "<extra></extra>"
                    ),
                )
                figure = style_static_chart(
                    figure,
                    y_title="Jumlah Artikel",
                    height=520,
                )

                st.plotly_chart(
                    figure,
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )

                show_result_explanation(
                    how_to_read=(
                        "Sumbu horizontal menunjukkan bulan publikasi, sumbu "
                        "vertikal menunjukkan jumlah artikel, dan setiap garis "
                        "menunjukkan satu kategori. Angka pada titik adalah "
                        "jumlah artikel kategori tersebut pada bulan terkait."
                    ),
                    main_result=(
                        f"Bulan dengan data terbanyak adalah **{peak_month['Bulan']}** "
                        f"dengan **{number_id(peak_month['Jumlah Artikel'])} artikel**. "
                        f"Kombinasi kategori-bulan tertinggi adalah "
                        f"**{display_label(peak_category['Kategori'])}** pada "
                        f"**{peak_category['Bulan']}**, yaitu "
                        f"**{number_id(peak_category['Jumlah Artikel'])} artikel**."
                    ),
                    interpretation=(
                        "Perbedaan jumlah antarb​​ulan dipengaruhi cakupan dan "
                        "periode crawling setiap kategori. Peningkatan pada bulan "
                        "terakhir tidak otomatis berarti produksi berita Kompas "
                        "secara umum selalu meningkat."
                    ),
                    conclusion=(
                        "Distribusi waktu hanya digunakan untuk memahami data, "
                        "bukan sebagai fitur input model final."
                    ),
                )
            else:
                st.info(
                    "Kolom tanggal atau kategori Kompas belum tersedia. "
                    "Grafik bulanan tidak ditampilkan."
                )

        with col_2:
            st.subheader("Distribusi Waktu Publikasi Kompas")

            if (
                not kompas_eda_data.empty
                and "Tanggal" in kompas_eda_data.columns
            ):
                valid_date_data = kompas_eda_data.dropna(
                    subset=["Tanggal"]
                ).copy()
                valid_date_data["Jam"] = valid_date_data["Tanggal"].dt.hour

                hourly_data = (
                    valid_date_data.groupby("Jam", as_index=False)
                    .size()
                    .rename(columns={"size": "Jumlah Artikel"})
                )
                all_hours = pd.DataFrame({"Jam": list(range(24))})
                hourly_data = (
                    all_hours.merge(hourly_data, on="Jam", how="left")
                    .fillna({"Jumlah Artikel": 0})
                )
                hourly_data["Jumlah Artikel"] = hourly_data[
                    "Jumlah Artikel"
                ].astype(int)

                peak_hour = hourly_data.sort_values(
                    "Jumlah Artikel",
                    ascending=False,
                ).iloc[0]
                positive_hours = hourly_data[
                    hourly_data["Jumlah Artikel"] > 0
                ]
                low_hour = positive_hours.sort_values(
                    "Jumlah Artikel",
                    ascending=True,
                ).iloc[0]

                figure = px.bar(
                    hourly_data,
                    x="Jam",
                    y="Jumlah Artikel",
                    text="Jumlah Artikel",
                    title="Jumlah Artikel Berdasarkan Jam Publikasi",
                )
                figure.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate=(
                        "<b>Jam:</b> %{x}:00<br>"
                        "<b>Jumlah artikel:</b> %{y:,}"
                        "<extra></extra>"
                    ),
                )
                figure.update_xaxes(tickmode="linear", dtick=1)
                figure = style_static_chart(
                    figure,
                    y_title="Jumlah Artikel",
                    height=520,
                )

                st.plotly_chart(
                    figure,
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )

                show_result_explanation(
                    how_to_read=(
                        "Sumbu horizontal menunjukkan jam 00.00–23.00, "
                        "sumbu vertikal menunjukkan jumlah artikel, dan angka "
                        "di atas batang adalah jumlah artikel pada jam tersebut."
                    ),
                    main_result=(
                        f"Publikasi terbanyak terdapat pada pukul "
                        f"**{int(peak_hour['Jam']):02d}.00**, yaitu "
                        f"**{number_id(peak_hour['Jumlah Artikel'])} artikel**. "
                        f"Jumlah terendah yang masih memiliki artikel terdapat "
                        f"pada pukul **{int(low_hour['Jam']):02d}.00**, yaitu "
                        f"**{number_id(low_hour['Jumlah Artikel'])} artikel**."
                    ),
                    interpretation=(
                        "Artikel lebih banyak diterbitkan pada jam aktif redaksi. "
                        "Jumlah rendah pada dini hari menunjukkan aktivitas "
                        "publikasi yang lebih sedikit pada periode tersebut."
                    ),
                    conclusion=(
                        "Jam publikasi digunakan untuk EDA dan tidak menjadi "
                        "fitur CNN maupun Attention-BiLSTM."
                    ),
                )
            else:
                st.info(
                    "Kolom tanggal Kompas belum tersedia. "
                    "Grafik jam publikasi tidak ditampilkan."
                )


# =============================================================================
# TAB 3 - HASIL & METRIK
# =============================================================================

with tab_results:
    st.header("Hasil dan Metrik 10 Eksperimen")
    st.caption(
        "Accuracy menunjukkan proporsi prediksi benar. Precision Macro, "
        "Recall Macro, dan Macro F1 memberi bobot yang sama pada setiap kelas."
    )

    result_tab_1, result_tab_2 = st.tabs(["Accuracy", "Macro F1"])

    with result_tab_1:
        if chart_data.empty or "Accuracy (%)" not in chart_data.columns:
            st.info("Data accuracy belum tersedia.")
        else:
            figure = px.bar(
                chart_data,
                x="Skenario",
                y="Accuracy (%)",
                color="Model",
                facet_col="Dataset",
                barmode="group",
                text="Accuracy (%)",
                title="Perbandingan Accuracy Seluruh Eksperimen",
                category_orders={
                    "Dataset": ["Kompas", "AG News"],
                    "Skenario": ["K1", "K2", "K3", "A1", "A2"],
                    "Model": ["CNN", "Attention-BiLSTM"],
                },
                hover_data={
                    "Eksperimen": True,
                    "Accuracy (%)": ":.2f",
                },
            )
            add_bar_labels(figure)
            figure = style_static_chart(
                figure,
                y_title="Accuracy (%)",
                y_range=[80, 100],
                height=540,
            )
            figure.for_each_annotation(
                lambda annotation: annotation.update(
                    text=annotation.text.replace("Dataset=", "")
                )
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )

            show_result_explanation(
                how_to_read=(
                    "Grafik dibagi menjadi Kompas dan AG News. Setiap skenario "
                    "memiliki batang CNN dan Attention-BiLSTM."
                ),
                main_result=(
                    "CNN unggul pada K1, K2, A1, dan A2. "
                    "Attention-BiLSTM sedikit unggul pada K3."
                ),
                interpretation=(
                    "CNN efektif menangkap pola lokal pada teks pendek. "
                    "Attention-BiLSTM tetap kompetitif ketika terdapat tambahan "
                    "urutan keyword pada K3."
                ),
                conclusion=(
                    "CNN K2 merupakan konfigurasi dengan accuracy tertinggi."
                ),
            )

    with result_tab_2:
        if chart_data.empty or "Macro F1 (%)" not in chart_data.columns:
            st.info("Data Macro F1 belum tersedia.")
        else:
            figure = px.bar(
                chart_data,
                x="Skenario",
                y="Macro F1 (%)",
                color="Model",
                facet_col="Dataset",
                barmode="group",
                text="Macro F1 (%)",
                title="Perbandingan Macro F1 Seluruh Eksperimen",
                category_orders={
                    "Dataset": ["Kompas", "AG News"],
                    "Skenario": ["K1", "K2", "K3", "A1", "A2"],
                    "Model": ["CNN", "Attention-BiLSTM"],
                },
                hover_data={
                    "Eksperimen": True,
                    "Macro F1 (%)": ":.2f",
                },
            )
            add_bar_labels(figure)
            figure = style_static_chart(
                figure,
                y_title="Macro F1 (%)",
                y_range=[80, 100],
                height=540,
            )
            figure.for_each_annotation(
                lambda annotation: annotation.update(
                    text=annotation.text.replace("Dataset=", "")
                )
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )

            show_result_explanation(
                how_to_read=(
                    "Macro F1 merupakan rata-rata F1 pada seluruh kelas dengan "
                    "bobot yang sama."
                ),
                main_result=(
                    "CNN K2 memperoleh Macro F1 sekitar 95,81%, sangat dekat "
                    "dengan accuracy 95,80%."
                ),
                interpretation=(
                    "Kedekatan accuracy dan Macro F1 menunjukkan performa "
                    "relatif konsisten pada empat kategori."
                ),
                conclusion=(
                    "Model tidak hanya baik secara keseluruhan, tetapi juga "
                    "cukup seimbang antarkelas."
                ),
            )

    st.subheader("Detail Eksperimen")

    selector_1, selector_2, selector_3 = st.columns(3)

    with selector_1:
        selected_dataset = st.selectbox(
            "Dataset",
            ["Kompas", "AG News"],
            key="result_dataset",
        )

    with selector_2:
        selected_model = st.selectbox(
            "Model",
            ["CNN", "Attention-BiLSTM"],
            key="result_model",
        )

    available_scenarios = list(
        EXPERIMENTS[selected_dataset][selected_model].keys()
    )

    with selector_3:
        selected_scenario = st.selectbox(
            "Skenario",
            available_scenarios,
            index=1 if len(available_scenarios) > 1 else 0,
            format_func=lambda code: f"{code} — {SCENARIO_NAMES[code]}",
            key="result_scenario",
        )

    experiment_name = (
        EXPERIMENTS[selected_dataset][selected_model][selected_scenario]
    )
    metrics = get_metric_bundle(evaluation_data, experiment_name)

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric(
        "Accuracy",
        percentage(metrics["accuracy"]),
        help="Proporsi seluruh data test yang diprediksi benar.",
    )
    metric_2.metric(
        "Precision Macro",
        percentage(metrics["precision"]),
        help="Rata-rata ketepatan prediksi setiap kelas.",
    )
    metric_3.metric(
        "Recall Macro",
        percentage(metrics["recall"]),
        help="Rata-rata kemampuan menemukan anggota setiap kelas.",
    )
    metric_4.metric(
        "Macro F1",
        percentage(metrics["f1"]),
        help="Keseimbangan precision dan recall pada seluruh kelas.",
    )

    visual_1, visual_2 = st.tabs(["Confusion Matrix", "Training Curve"])

    with visual_1:
        matrix_type = st.radio(
            "Tampilan",
            ["Jumlah", "Normalized"],
            horizontal=True,
            key="matrix_type",
        )

        prediction_data = load_prediction_result(experiment_name)

        if prediction_data.empty:
            st.info("Data prediksi eksperimen belum tersedia.")
        else:
            matrix_figure, matrix_summary = create_confusion_matrix_chart(
                prediction_data=prediction_data,
                dataset_name=selected_dataset,
                normalized=matrix_type == "Normalized",
            )

            if matrix_figure is None or matrix_summary is None:
                st.warning(
                    "Kolom label aktual atau prediksi tidak ditemukan pada CSV."
                )
            else:
                st.plotly_chart(
                    matrix_figure,
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )

                cm_1, cm_2, cm_3 = st.columns(3)
                cm_1.metric(
                    "Total Data Test",
                    number_id(matrix_summary["total_data"]),
                )
                cm_2.metric(
                    "Prediksi Benar",
                    number_id(matrix_summary["correct"]),
                )
                cm_3.metric(
                    "Prediksi Salah",
                    number_id(matrix_summary["wrong"]),
                )

                if matrix_type == "Jumlah":
                    how_to_read = (
                        "Baris menunjukkan kelas aktual dan kolom menunjukkan "
                        "kelas prediksi. Angka diagonal adalah jumlah prediksi "
                        "benar, sedangkan angka di luar diagonal adalah jumlah "
                        "kesalahan klasifikasi."
                    )
                else:
                    how_to_read = (
                        "Setiap baris dinormalisasi menjadi persentase. Nilai "
                        "diagonal menunjukkan persentase keberhasilan model "
                        "mengenali setiap kelas aktual."
                    )

                if matrix_summary["largest_error"] > 0:
                    error_interpretation = (
                        f"Kesalahan terbesar terjadi ketika kelas aktual "
                        f"**{display_label(matrix_summary['actual_error_class'])}** "
                        f"diprediksi sebagai "
                        f"**{display_label(matrix_summary['predicted_error_class'])}**, "
                        f"sebanyak **{matrix_summary['largest_error']} data**. "
                        "Kesalahan ini dapat terjadi karena kemiripan kosakata "
                        "atau konteks antara kedua kategori."
                    )
                else:
                    error_interpretation = (
                        "Tidak terdapat kesalahan klasifikasi pada data tersebut."
                    )

                show_result_explanation(
                    how_to_read=how_to_read,
                    main_result=(
                        f"Dari **{number_id(matrix_summary['total_data'])} data test**, "
                        f"model memprediksi **{number_id(matrix_summary['correct'])} "
                        f"data benar** dan **{number_id(matrix_summary['wrong'])} "
                        f"data salah**. Accuracy hasil perhitungan adalah "
                        f"**{matrix_summary['accuracy'] * 100:.2f}%**."
                    ),
                    interpretation=error_interpretation,
                    conclusion=(
                        "Nilai diagonal yang dominan menunjukkan bahwa model "
                        "mampu membedakan sebagian besar kategori dengan baik."
                    ),
                )

    with visual_2:
        history_data = load_training_history(experiment_name)
        accuracy_figure, loss_figure, training_summary = (
            create_training_curve_charts(history_data)
        )

        if (
            accuracy_figure is None
            and loss_figure is None
        ):
            st.info(
                "Training history CSV belum tersedia atau kolomnya belum "
                "sesuai. Gambar PNG tidak digunakan pada dashboard."
            )
        else:
            curve_1, curve_2 = st.columns(2)

            with curve_1:
                if accuracy_figure is not None:
                    st.plotly_chart(
                        accuracy_figure,
                        use_container_width=True,
                        config=PLOTLY_CONFIG,
                    )

            with curve_2:
                if loss_figure is not None:
                    st.plotly_chart(
                        loss_figure,
                        use_container_width=True,
                        config=PLOTLY_CONFIG,
                    )

            if training_summary is not None:
                best_accuracy_text = (
                    percentage(training_summary["best_val_accuracy"])
                    if pd.notna(training_summary["best_val_accuracy"])
                    else "-"
                )
                best_loss_text = (
                    f"{training_summary['best_val_loss']:.6f}"
                    if pd.notna(training_summary["best_val_loss"])
                    else "-"
                )

                show_result_explanation(
                    how_to_read=(
                        "Kurva accuracy menunjukkan perkembangan kemampuan model, "
                        "sedangkan kurva loss menunjukkan besar kesalahan. "
                        "Perbedaan train dan validation digunakan untuk melihat "
                        "indikasi overfitting."
                    ),
                    main_result=(
                        f"Training selesai dalam **{training_summary['epochs']} epoch**. "
                        f"Validation loss terbaik diperoleh pada epoch "
                        f"**{training_summary['best_epoch']}** sebesar "
                        f"**{best_loss_text}**, dengan validation accuracy "
                        f"tertinggi **{best_accuracy_text}**."
                    ),
                    interpretation=(
                        "Jika train accuracy terus meningkat tetapi validation loss "
                        "tidak membaik, model mulai menunjukkan indikasi overfitting."
                    ),
                    conclusion=(
                        "Checkpoint dan EarlyStopping memastikan model final "
                        "menggunakan bobot pada validation loss terbaik, bukan "
                        "bobot epoch terakhir."
                    ),
                )

    with st.expander(
        "Lihat ringkasan angka 10 eksperimen",
        expanded=False,
    ):
        if evaluation_data.empty:
            st.info("Data evaluasi belum tersedia.")
        else:
            compact_evaluation = clean_result_table(
                evaluation_data
            )

            preferred_columns = [
                "Dataset",
                "Model",
                "Skenario",
                "Representasi Teks",
                "Accuracy",
                "Macro F1",
            ]
            preferred_columns = [
                column
                for column in preferred_columns
                if column in compact_evaluation.columns
            ]

            st.dataframe(
                compact_evaluation[preferred_columns],
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Tabel ini hanya digunakan sebagai ringkasan angka. "
                "Penjelasan utama tetap disampaikan melalui grafik, "
                "confusion matrix, dan kartu metrik."
            )


# =============================================================================
# TAB 4 - PERBANDINGAN YAKE
# =============================================================================

with tab_yake:
    st.header("Perbandingan Tanpa YAKE dan Dengan YAKE")

    st.info(
        "**Perbandingan terkontrol:** K2 menggunakan Title + Description "
        "tanpa YAKE, sedangkan K3 menggunakan Title + Description + Keyword "
        "YAKE. Keduanya memakai split data yang sama, sequence length 60, "
        "arsitektur yang sama, dan konfigurasi training yang sama."
    )

    experiments_yake = [
        ("CNN", "cnn_k2", "cnn_k3"),
        (
            "Attention-BiLSTM",
            "attention_bilstm_k2",
            "attention_bilstm_k3",
        ),
    ]

    comparison_rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []

    for model_name, without_name, with_name in experiments_yake:
        without_metrics = get_metric_bundle(evaluation_data, without_name)
        with_metrics = get_metric_bundle(evaluation_data, with_name)

        without_accuracy = without_metrics["accuracy"]
        with_accuracy = with_metrics["accuracy"]

        change_pp = (
            (with_accuracy - without_accuracy) * 100
            if without_accuracy is not None and with_accuracy is not None
            else np.nan
        )

        errors_without = (
            round(KOMPAS_TEST_SIZE * (1 - without_accuracy))
            if without_accuracy is not None
            else np.nan
        )
        errors_with = (
            round(KOMPAS_TEST_SIZE * (1 - with_accuracy))
            if with_accuracy is not None
            else np.nan
        )

        comparison_rows.extend(
            [
                {
                    "Model": model_name,
                    "Kondisi": "Tanpa YAKE — K2",
                    "Accuracy (%)": (
                        without_accuracy * 100
                        if without_accuracy is not None
                        else np.nan
                    ),
                },
                {
                    "Model": model_name,
                    "Kondisi": "Dengan YAKE — K3",
                    "Accuracy (%)": (
                        with_accuracy * 100
                        if with_accuracy is not None
                        else np.nan
                    ),
                },
            ]
        )

        detailed_rows.extend(
            [
                {
                    "Model": model_name,
                    "Skenario": "K2",
                    "Penggunaan YAKE": "Tidak",
                    "Representasi": "Title + Description",
                    "Accuracy": percentage(without_metrics["accuracy"]),
                    "Precision Macro": percentage(without_metrics["precision"]),
                    "Recall Macro": percentage(without_metrics["recall"]),
                    "Macro F1": percentage(without_metrics["f1"]),
                    "Prediksi Salah": (
                        int(errors_without)
                        if pd.notna(errors_without)
                        else "-"
                    ),
                    "Perubahan Accuracy": "Baseline",
                },
                {
                    "Model": model_name,
                    "Skenario": "K3",
                    "Penggunaan YAKE": "Ya",
                    "Representasi": "Title + Description + Keyword YAKE",
                    "Accuracy": percentage(with_metrics["accuracy"]),
                    "Precision Macro": percentage(with_metrics["precision"]),
                    "Recall Macro": percentage(with_metrics["recall"]),
                    "Macro F1": percentage(with_metrics["f1"]),
                    "Prediksi Salah": (
                        int(errors_with)
                        if pd.notna(errors_with)
                        else "-"
                    ),
                    "Perubahan Accuracy": (
                        f"{change_pp:+.2f} pp"
                        if pd.notna(change_pp)
                        else "-"
                    ),
                },
            ]
        )

    comparison_frame = pd.DataFrame(comparison_rows)
    detailed_frame = pd.DataFrame(detailed_rows)

    if (
        not comparison_frame.empty
        and comparison_frame["Accuracy (%)"].notna().any()
    ):
        figure = px.bar(
            comparison_frame,
            x="Model",
            y="Accuracy (%)",
            color="Kondisi",
            barmode="group",
            text="Accuracy (%)",
            title="Perbandingan Accuracy Tanpa YAKE dan Dengan YAKE",
            category_orders={
                "Model": ["CNN", "Attention-BiLSTM"],
                "Kondisi": [
                    "Tanpa YAKE — K2",
                    "Dengan YAKE — K3",
                ],
            },
            hover_data={
                "Kondisi": True,
                "Accuracy (%)": ":.2f",
            },
        )
        add_bar_labels(figure)
        figure = style_static_chart(
            figure,
            y_title="Accuracy (%)",
            y_range=[93, 97],
            height=520,
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    cnn_without = get_metric(
        evaluation_data,
        "cnn_k2",
        ["accuracy", "test_accuracy"],
    )
    cnn_with = get_metric(
        evaluation_data,
        "cnn_k3",
        ["accuracy", "test_accuracy"],
    )
    attention_without = get_metric(
        evaluation_data,
        "attention_bilstm_k2",
        ["accuracy", "test_accuracy"],
    )
    attention_with = get_metric(
        evaluation_data,
        "attention_bilstm_k3",
        ["accuracy", "test_accuracy"],
    )

    cnn_change = (
        (cnn_with - cnn_without) * 100
        if cnn_without is not None and cnn_with is not None
        else np.nan
    )
    attention_change = (
        (attention_with - attention_without) * 100
        if attention_without is not None and attention_with is not None
        else np.nan
    )

    show_result_explanation(
        how_to_read=(
            "Setiap model memiliki dua batang. K2 menunjukkan hasil tanpa "
            "YAKE menggunakan Title + Description. K3 menunjukkan hasil "
            "setelah keyword YAKE ditambahkan."
        ),
        main_result=(
            f"CNN berubah dari **{percentage(cnn_without)}** menjadi "
            f"**{percentage(cnn_with)}** atau "
            f"**{cnn_change:+.2f} percentage point**. "
            f"Attention-BiLSTM berubah dari "
            f"**{percentage(attention_without)}** menjadi "
            f"**{percentage(attention_with)}** atau "
            f"**{attention_change:+.2f} percentage point**."
        ),
        interpretation=(
            "Keyword YAKE diekstraksi dari Title dan Description. Kata-kata "
            "penting kemungkinan sudah terdapat pada dua komponen tersebut, "
            "sehingga penambahan keyword menghasilkan pengulangan atau "
            "redundansi informasi."
        ),
        conclusion=(
            "Pada dataset dan konfigurasi penelitian ini, YAKE belum "
            "meningkatkan performa. Hasil ini tidak berarti YAKE selalu tidak "
            "bermanfaat, tetapi belum memberi informasi tambahan yang efektif "
            "pada konfigurasi yang diuji."
        ),
    )

    yake_metric_1, yake_metric_2 = st.columns(2)

    with yake_metric_1:
        with st.container(border=True):
            st.subheader("CNN")
            st.metric("Tanpa YAKE — K2", percentage(cnn_without))
            st.metric(
                "Dengan YAKE — K3",
                percentage(cnn_with),
                delta=(
                    f"{cnn_change:+.2f} pp"
                    if pd.notna(cnn_change)
                    else None
                ),
                delta_color="inverse",
            )
            if cnn_without is not None and cnn_with is not None:
                st.caption(
                    f"Prediksi salah berubah dari "
                    f"{round(KOMPAS_TEST_SIZE * (1 - cnn_without))} menjadi "
                    f"{round(KOMPAS_TEST_SIZE * (1 - cnn_with))} artikel."
                )

    with yake_metric_2:
        with st.container(border=True):
            st.subheader("Attention-BiLSTM")
            st.metric(
                "Tanpa YAKE — K2",
                percentage(attention_without),
            )
            st.metric(
                "Dengan YAKE — K3",
                percentage(attention_with),
                delta=(
                    f"{attention_change:+.2f} pp"
                    if pd.notna(attention_change)
                    else None
                ),
                delta_color="inverse",
            )
            if attention_without is not None and attention_with is not None:
                st.caption(
                    f"Prediksi salah berubah dari "
                    f"{round(KOMPAS_TEST_SIZE * (1 - attention_without))} "
                    f"menjadi "
                    f"{round(KOMPAS_TEST_SIZE * (1 - attention_with))} artikel."
                )

    st.caption(
        "Detail angka YAKE sudah diringkas pada grafik dan kartu metrik. "
        "Tabel CSV mentah tidak ditampilkan agar halaman tetap berfokus "
        "pada implementasi dan hasil utama."
    )


# =============================================================================
# TAB 5 - PREDIKSI BERITA
# =============================================================================

with tab_prediction:
    st.header("Prediksi Kategori Berita")

    st.info(
        "Masukkan Title dan Description. Input diproses menggunakan "
        "vocabulary saat training dan sequence length 60, lalu diprediksi "
        "oleh CNN K2 dan Attention-BiLSTM K2."
    )

    with st.form("prediction_form"):
        title_input = st.text_input(
            "Title",
            placeholder="Contoh: Rupiah Menguat terhadap Dolar AS",
        )
        description_input = st.text_area(
            "Description",
            placeholder=(
                "Contoh: Nilai tukar rupiah menguat setelah "
                "Bank Indonesia mengumumkan kebijakan..."
            ),
            height=140,
        )
        submitted = st.form_submit_button(
            "Prediksi Berita",
            use_container_width=True,
        )

    if submitted:
        if not title_input.strip():
            st.warning("Title tidak boleh kosong.")
        elif not description_input.strip():
            st.warning("Description tidak boleh kosong.")
        else:
            try:
                with st.spinner("Menjalankan CNN dan Attention-BiLSTM..."):
                    prediction_result = predict_news(
                        title=title_input,
                        description=description_input,
                    )

                recommendation = prediction_result["recommended_prediction"]
                recommended_label = display_label(
                    recommendation["predicted_label"]
                )

                result_1, result_2, result_3 = st.columns(3)
                result_1.metric("Prediksi Utama", recommended_label)
                result_2.metric(
                    "Model Rekomendasi",
                    recommendation["source_model"],
                )
                result_3.metric(
                    "Confidence",
                    percentage(recommendation["confidence"]),
                )

                cnn_result = prediction_result["cnn"]
                attention_result = prediction_result["attention_bilstm"]
                model_1, model_2 = st.columns(2)

                for column, model_name, model_result in [
                    (model_1, "CNN K2", cnn_result),
                    (model_2, "Attention-BiLSTM K2", attention_result),
                ]:
                    with column:
                        with st.container(border=True):
                            predicted_label = display_label(
                                model_result["predicted_label"]
                            )

                            st.subheader(model_name)
                            st.metric("Kategori", predicted_label)
                            st.metric(
                                "Confidence",
                                percentage(model_result["confidence"]),
                            )
                            st.caption(
                                f"Waktu inferensi: "
                                f"{model_result['inference_time_ms']:.2f} ms"
                            )

                            probability_frame = pd.DataFrame(
                                {
                                    "Kategori": [
                                        display_label(label)
                                        for label in model_result[
                                            "probabilities"
                                        ].keys()
                                    ],
                                    "Probabilitas (%)": [
                                        value * 100
                                        for value in model_result[
                                            "probabilities"
                                        ].values()
                                    ],
                                }
                            )

                            figure = px.bar(
                                probability_frame,
                                x="Kategori",
                                y="Probabilitas (%)",
                                text="Probabilitas (%)",
                                title=f"Probabilitas {model_name}",
                                hover_data={
                                    "Probabilitas (%)": ":.2f",
                                },
                            )
                            add_bar_labels(figure)
                            figure = style_static_chart(
                                figure,
                                y_title="Probabilitas (%)",
                                y_range=[0, 105],
                                height=440,
                            )

                            st.plotly_chart(
                                figure,
                                use_container_width=True,
                                config=PLOTLY_CONFIG,
                            )

                if prediction_result["model_agreement"]:
                    st.success(
                        "CNN dan Attention-BiLSTM memberikan kategori yang sama."
                    )
                else:
                    st.warning(
                        "Kedua model memberikan kategori berbeda. Rekomendasi "
                        "mengikuti CNN K2 karena memiliki performa test terbaik."
                    )

                show_result_explanation(
                    how_to_read=(
                        "Setiap batang menunjukkan probabilitas satu kategori. "
                        "Kategori dengan probabilitas tertinggi menjadi hasil "
                        "prediksi model."
                    ),
                    main_result=(
                        f"Rekomendasi sistem adalah **{recommended_label}** "
                        f"dengan confidence "
                        f"**{percentage(recommendation['confidence'])}**."
                    ),
                    interpretation=(
                        "Confidence merupakan keyakinan relatif model berdasarkan "
                        "distribusi probabilitas, bukan jaminan bahwa prediksi "
                        "selalu benar."
                    ),
                    conclusion=(
                        "CNN K2 digunakan sebagai sumber rekomendasi utama karena "
                        "mempunyai performa test set terbaik."
                    ),
                )

            except Exception as error:
                st.error(f"Prediksi gagal: {error}")


# =============================================================================
# TAB 6 - EXPLAINABLE AI
# =============================================================================

with tab_shap:
    st.header("Explainable AI Menggunakan SHAP")

    st.info(
        "SHAP diterapkan pada CNN K2 sebagai model terbaik. Fitur yang "
        "dianalisis adalah token atau kata pada Title dan Description."
    )

    with st.expander("Istilah penting dalam SHAP"):
        st.markdown(
            """
            - **Token:** kata atau unit teks yang masuk ke model.
            - **Nilai SHAP positif:** token mendukung kelas yang dijelaskan.
            - **Nilai SHAP negatif:** token mengurangi dukungan terhadap kelas.
            - **Nilai absolut SHAP:** besar pengaruh tanpa melihat arah positif
              atau negatif.
            - **Global explanation:** penjelasan pola model pada banyak artikel.
            - **Local explanation:** penjelasan satu artikel tertentu.
            """
        )

    shap_1, shap_2, shap_3, shap_4 = st.tabs(
        [
            "Global SHAP",
            "SHAP per Kelas",
            "Local SHAP",
            "Waterfall Plot",
        ]
    )

    with shap_1:
        if global_shap.empty:
            st.info("Data Global SHAP belum tersedia.")
        else:
            token_column = find_column(global_shap, ["token", "word"])
            importance_column = find_column(
                global_shap,
                ["total_abs_shap", "mean_abs_shap", "importance"],
            )

            top_n = st.slider(
                "Jumlah token yang ditampilkan",
                min_value=10,
                max_value=50,
                value=20,
                step=5,
            )

            if token_column is None or importance_column is None:
                st.warning("Kolom token atau importance tidak ditemukan.")
            else:
                semantic_global_shap = global_shap[
                    ~global_shap[token_column]
                    .astype(str)
                    .str.strip()
                    .isin(SPECIAL_TOKENS)
                ].copy()

                top_tokens = (
                    semantic_global_shap
                    .sort_values(importance_column, ascending=False)
                    .head(top_n)
                )

                figure = px.bar(
                    top_tokens.sort_values(
                        importance_column,
                        ascending=True,
                    ),
                    x=importance_column,
                    y=token_column,
                    orientation="h",
                    text=importance_column,
                    title=f"{top_n} Token Semantik Paling Berpengaruh Global",
                    hover_data={
                        importance_column: ":.4f",
                    },
                )
                figure.update_traces(
                    texttemplate="%{text:.4f}",
                    textposition="outside",
                    cliponaxis=False,
                )
                figure = style_static_chart(
                    figure,
                    y_title="Total Nilai Absolut SHAP",
                    height=620,
                )

                st.plotly_chart(
                    figure,
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )

                top_global_token = (
                    str(top_tokens.iloc[0][token_column])
                    if not top_tokens.empty
                    else "-"
                )
                top_global_value = (
                    float(top_tokens.iloc[0][importance_column])
                    if not top_tokens.empty
                    else 0.0
                )

                show_result_explanation(
                    how_to_read=(
                        "Sumbu vertikal menunjukkan token. Sumbu horizontal "
                        "menunjukkan total nilai absolut SHAP. Semakin panjang "
                        "batang, semakin besar pengaruh token terhadap keputusan "
                        "CNN K2 pada sampel yang dianalisis."
                    ),
                    main_result=(
                        f"Token semantik dengan pengaruh global tertinggi adalah "
                        f"**{top_global_token}** dengan nilai "
                        f"**{top_global_value:.4f}**."
                    ),
                    interpretation=(
                        "Global SHAP menunjukkan besar pengaruh secara keseluruhan, "
                        "tetapi belum menunjukkan token tersebut selalu mengarah "
                        "ke satu kelas. Hubungan dengan kategori dianalisis pada "
                        "SHAP per kelas."
                    ),
                    conclusion=(
                        "Model menggunakan kombinasi banyak token pada Title dan "
                        "Description, bukan hanya satu kata."
                    ),
                )

                st.caption(
                    "Nilai token sudah ditampilkan langsung pada grafik. "
                    "Tabel teknis Global SHAP tidak ditampilkan agar halaman "
                    "lebih ringkas saat demonstrasi."
                )

    with shap_2:
        if global_shap_by_class.empty:
            st.info("Data SHAP per kelas belum tersedia.")
        else:
            class_column = find_column(
                global_shap_by_class,
                ["class_name", "class", "label", "category"],
            )
            token_column = find_column(
                global_shap_by_class,
                ["token", "word"],
            )
            importance_column = find_column(
                global_shap_by_class,
                ["total_abs_shap", "mean_abs_shap", "importance"],
            )

            if (
                class_column is None
                or token_column is None
                or importance_column is None
            ):
                st.warning("Kolom SHAP per kelas belum lengkap.")
            else:
                classes = sorted(
                    global_shap_by_class[class_column]
                    .dropna()
                    .astype(str)
                    .unique()
                )

                selected_class = st.selectbox(
                    "Pilih kelas",
                    classes,
                    format_func=display_label,
                    key="shap_class",
                )

                class_data = global_shap_by_class[
                    global_shap_by_class[class_column]
                    .astype(str)
                    .eq(selected_class)
                ].copy()
                class_data = class_data[
                    ~class_data[token_column]
                    .astype(str)
                    .str.strip()
                    .isin(SPECIAL_TOKENS)
                ]
                class_data = (
                    class_data
                    .sort_values(importance_column, ascending=False)
                    .head(20)
                )

                figure = px.bar(
                    class_data.sort_values(
                        importance_column,
                        ascending=True,
                    ),
                    x=importance_column,
                    y=token_column,
                    orientation="h",
                    text=importance_column,
                    title=(
                        f"Token Penting untuk Kelas "
                        f"{display_label(selected_class)}"
                    ),
                    hover_data={importance_column: ":.4f"},
                )
                figure.update_traces(
                    texttemplate="%{text:.4f}",
                    textposition="outside",
                    cliponaxis=False,
                )
                figure = style_static_chart(
                    figure,
                    y_title="Total Nilai Absolut SHAP",
                    height=620,
                )

                st.plotly_chart(
                    figure,
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )

                top_class_token = (
                    str(class_data.iloc[0][token_column])
                    if not class_data.empty
                    else "-"
                )
                top_class_value = (
                    float(class_data.iloc[0][importance_column])
                    if not class_data.empty
                    else 0.0
                )

                if str(selected_class).lower() == "bola":
                    example_interpretation = (
                        "Token seperti 'vs', 'liga', 'madrid', 'barcelona', "
                        "dan 'champions' berkaitan dengan pertandingan dan klub "
                        "sehingga relevan dengan kategori Bola."
                    )
                elif str(selected_class).lower() == "global":
                    example_interpretation = (
                        "Token yang berkaitan dengan negara, tokoh internasional, "
                        "dan konflik membantu model mengenali kategori Global."
                    )
                elif str(selected_class).lower() == "money":
                    example_interpretation = (
                        "Token seperti rupiah, harga, bank, pasar, atau dolar "
                        "memiliki keterkaitan dengan kategori Money."
                    )
                else:
                    example_interpretation = (
                        "Token yang berkaitan dengan perangkat, teknologi, AI, "
                        "aplikasi, atau baterai membantu model mengenali Tekno."
                    )

                show_result_explanation(
                    how_to_read=(
                        "Sumbu vertikal menunjukkan token dan sumbu horizontal "
                        "menunjukkan besar pengaruh token terhadap kelas yang "
                        "dipilih."
                    ),
                    main_result=(
                        f"Untuk kelas **{display_label(selected_class)}**, token "
                        f"dengan pengaruh terbesar adalah "
                        f"**{top_class_token}** dengan nilai "
                        f"**{top_class_value:.4f}**."
                    ),
                    interpretation=example_interpretation,
                    conclusion=(
                        "SHAP per kelas menunjukkan bahwa model mempelajari "
                        "pola token yang relatif sesuai dengan konteks kategori."
                    ),
                )

                st.caption(
                    "Token dan nilai pengaruhnya sudah ditampilkan pada grafik "
                    "untuk kelas yang dipilih."
                )

    with shap_3:
        if local_shap_summary.empty:
            st.info("Data Local SHAP belum tersedia.")
        else:
            st.info(
                "Pilih satu ID artikel untuk melihat label aktual, hasil "
                "prediksi, confidence, dan kontribusi token. Tabel teknis "
                "seluruh sampel tidak ditampilkan pada halaman utama."
            )

            sample_column = find_column(
                local_shap_summary,
                ["document_id", "sample_id", "id"],
            )
            contribution_sample_column = find_column(
                local_token_contributions,
                ["document_id", "sample_id", "id"],
            )

            if (
                sample_column is not None
                and contribution_sample_column is not None
                and not local_token_contributions.empty
            ):
                selected_sample = st.selectbox(
                    "Pilih sampel",
                    local_shap_summary[sample_column].astype(str).tolist(),
                    key="local_sample",
                )

                selected_summary = local_shap_summary[
                    local_shap_summary[sample_column]
                    .astype(str)
                    .eq(str(selected_sample))
                ]

                if not selected_summary.empty:
                    selected_row = selected_summary.iloc[0]
                    actual_label = display_label(
                        selected_row.get("actual_label", "-")
                    )
                    predicted_label = display_label(
                        selected_row.get("predicted_label", "-")
                    )
                    confidence = normalize_metric(
                        selected_row.get("prediction_confidence", None)
                    )
                    is_correct_value = selected_row.get("is_correct", False)
                    is_correct = str(is_correct_value).lower() in {
                        "true",
                        "1",
                        "yes",
                    } or is_correct_value is True

                    local_1, local_2, local_3, local_4 = st.columns(4)
                    local_1.metric("ID Artikel", selected_sample)
                    local_2.metric("Label Aktual", actual_label)
                    local_3.metric("Label Prediksi", predicted_label)
                    local_4.metric("Confidence", percentage(confidence))

                    if is_correct:
                        st.success(
                            "Prediksi model sesuai dengan label aktual."
                        )
                    else:
                        st.warning(
                            "Prediksi tidak sesuai dengan label aktual. "
                            "Kontribusi token digunakan untuk menganalisis "
                            "penyebab kesalahan."
                        )

                sample_data = local_token_contributions[
                    local_token_contributions[
                        contribution_sample_column
                    ]
                    .astype(str)
                    .eq(str(selected_sample))
                ].copy()

                token_column = find_column(
                    sample_data,
                    ["token", "word"],
                )
                contribution_column = find_column(
                    sample_data,
                    [
                        "signed_shap",
                        "shap_value",
                        "contribution",
                        "token_contribution",
                    ],
                )

                if (
                    token_column is not None
                    and contribution_column is not None
                    and not sample_data.empty
                ):
                    sample_data = sample_data[
                        ~sample_data[token_column]
                        .astype(str)
                        .str.strip()
                        .isin(SPECIAL_TOKENS)
                    ].copy()
                    sample_data["absolute_contribution"] = (
                        pd.to_numeric(
                            sample_data[contribution_column],
                            errors="coerce",
                        ).abs()
                    )
                    sample_data = (
                        sample_data
                        .sort_values(
                            "absolute_contribution",
                            ascending=False,
                        )
                        .head(20)
                    )

                    figure = px.bar(
                        sample_data.sort_values(
                            contribution_column,
                            ascending=True,
                        ),
                        x=contribution_column,
                        y=token_column,
                        orientation="h",
                        text=contribution_column,
                        title=(
                            f"Kontribusi Token pada Artikel "
                            f"{selected_sample}"
                        ),
                        color=contribution_column,
                        color_continuous_scale="RdBu",
                        color_continuous_midpoint=0,
                    )
                    figure.update_traces(
                        texttemplate="%{text:+.4f}",
                        textposition="outside",
                        cliponaxis=False,
                    )
                    figure = style_static_chart(
                        figure,
                        y_title="Nilai SHAP",
                        height=620,
                    )

                    st.plotly_chart(
                        figure,
                        use_container_width=True,
                        config=PLOTLY_CONFIG,
                    )

                    strongest_row = sample_data.sort_values(
                        "absolute_contribution",
                        ascending=False,
                    ).iloc[0]
                    strongest_token = str(strongest_row[token_column])
                    strongest_value = float(
                        strongest_row[contribution_column]
                    )

                    show_result_explanation(
                        how_to_read=(
                            "Setiap batang menunjukkan kontribusi satu token "
                            "terhadap prediksi artikel terpilih. Nilai positif "
                            "mendukung kelas prediksi dan nilai negatif "
                            "mengurangi dukungan."
                        ),
                        main_result=(
                            f"Token dengan kontribusi terbesar adalah "
                            f"**{strongest_token}** dengan nilai SHAP "
                            f"**{strongest_value:+.4f}**."
                        ),
                        interpretation=(
                            "Model lebih dipengaruhi oleh token yang memiliki "
                            "hubungan kuat dengan kategori prediksi. Pada prediksi "
                            "salah, token tertentu dapat mendorong model ke kelas "
                            "lain walaupun label aktual berbeda."
                        ),
                        conclusion=(
                            "Local SHAP menjelaskan alasan prediksi satu artikel "
                            "secara individual."
                        ),
                    )

    with shap_4:
        if (
            local_shap_summary.empty
            or local_token_contributions.empty
        ):
            st.info("Data untuk waterfall SHAP belum tersedia.")
        else:
            sample_column = find_column(
                local_shap_summary,
                ["document_id", "sample_id", "id"],
            )
            contribution_sample_column = find_column(
                local_token_contributions,
                ["document_id", "sample_id", "id"],
            )

            if (
                sample_column is None
                or contribution_sample_column is None
            ):
                st.warning(
                    "Kolom ID artikel pada data SHAP belum ditemukan."
                )
            else:
                waterfall_sample = st.selectbox(
                    "Pilih artikel untuk waterfall",
                    local_shap_summary[
                        sample_column
                    ].astype(str).tolist(),
                    key="waterfall_direct_sample",
                )

                selected_summary = local_shap_summary[
                    local_shap_summary[sample_column]
                    .astype(str)
                    .eq(str(waterfall_sample))
                ]

                if not selected_summary.empty:
                    selected_row = selected_summary.iloc[0]

                    actual_label = display_label(
                        selected_row.get("actual_label", "-")
                    )
                    predicted_label = display_label(
                        selected_row.get("predicted_label", "-")
                    )
                    confidence = normalize_metric(
                        selected_row.get(
                            "prediction_confidence",
                            None,
                        )
                    )

                    waterfall_1, waterfall_2, waterfall_3, waterfall_4 = (
                        st.columns(4)
                    )
                    waterfall_1.metric(
                        "ID Artikel",
                        waterfall_sample,
                    )
                    waterfall_2.metric(
                        "Label Aktual",
                        actual_label,
                    )
                    waterfall_3.metric(
                        "Label Prediksi",
                        predicted_label,
                    )
                    waterfall_4.metric(
                        "Confidence",
                        percentage(confidence),
                    )

                    if actual_label == predicted_label:
                        st.success(
                            "Prediksi artikel ini sesuai dengan label aktual."
                        )
                    else:
                        st.warning(
                            "Prediksi artikel ini berbeda dari label aktual."
                        )

                sample_data = local_token_contributions[
                    local_token_contributions[
                        contribution_sample_column
                    ]
                    .astype(str)
                    .eq(str(waterfall_sample))
                ].copy()

                token_column = find_column(
                    sample_data,
                    ["token", "word"],
                )
                contribution_column = find_column(
                    sample_data,
                    [
                        "signed_shap",
                        "shap_value",
                        "contribution",
                        "token_contribution",
                    ],
                )

                if (
                    token_column is None
                    or contribution_column is None
                    or sample_data.empty
                ):
                    st.info(
                        "Kontribusi token untuk artikel terpilih "
                        "belum tersedia."
                    )
                else:
                    sample_data = sample_data[
                        ~sample_data[token_column]
                        .astype(str)
                        .str.strip()
                        .isin(SPECIAL_TOKENS)
                    ].copy()

                    sample_data[contribution_column] = pd.to_numeric(
                        sample_data[contribution_column],
                        errors="coerce",
                    )
                    sample_data = sample_data.dropna(
                        subset=[contribution_column]
                    )
                    sample_data["absolute_contribution"] = (
                        sample_data[contribution_column].abs()
                    )

                    waterfall_data = (
                        sample_data
                        .sort_values(
                            "absolute_contribution",
                            ascending=False,
                        )
                        .head(15)
                        .sort_values(
                            contribution_column,
                            ascending=True,
                        )
                    )

                    figure = go.Figure(
                        go.Waterfall(
                            orientation="h",
                            measure=[
                                "relative"
                                for _ in range(len(waterfall_data))
                            ],
                            y=waterfall_data[token_column],
                            x=waterfall_data[contribution_column],
                            text=[
                                f"{value:+.4f}"
                                for value in waterfall_data[
                                    contribution_column
                                ]
                            ],
                            textposition="outside",
                            connector={
                                "line": {
                                    "color": "rgba(100, 116, 139, 0.45)"
                                }
                            },
                            increasing={
                                "marker": {
                                    "color": "#2563eb"
                                }
                            },
                            decreasing={
                                "marker": {
                                    "color": "#ef4444"
                                }
                            },
                        )
                    )
                    figure.update_layout(
                        template="plotly_white",
                        height=620,
                        title=(
                            "Waterfall Kontribusi Token — "
                            f"{waterfall_sample}"
                        ),
                        xaxis_title="Akumulasi Nilai SHAP",
                        yaxis_title="Token",
                        margin=dict(
                            l=40,
                            r=55,
                            t=80,
                            b=45,
                        ),
                        showlegend=False,
                    )

                    st.plotly_chart(
                        figure,
                        use_container_width=True,
                        config=PLOTLY_CONFIG,
                    )

                    strongest_row = (
                        sample_data
                        .sort_values(
                            "absolute_contribution",
                            ascending=False,
                        )
                        .iloc[0]
                    )
                    strongest_token = str(
                        strongest_row[token_column]
                    )
                    strongest_value = float(
                        strongest_row[contribution_column]
                    )

                    show_result_explanation(
                        how_to_read=(
                            "Batang biru menambah dukungan terhadap kelas "
                            "prediksi, sedangkan batang merah mengurangi "
                            "dukungan. Posisi akhir merupakan akumulasi "
                            "kontribusi token yang ditampilkan."
                        ),
                        main_result=(
                            f"Token dengan kontribusi absolut terbesar adalah "
                            f"**{strongest_token}** dengan nilai "
                            f"**{strongest_value:+.4f}**."
                        ),
                        interpretation=(
                            "Prediksi dibentuk oleh gabungan beberapa token. "
                            "Pada prediksi salah, token yang lebih kuat dapat "
                            "mendorong model menuju kategori lain."
                        ),
                        conclusion=(
                            "Waterfall dibuat langsung oleh Streamlit dari "
                            "data kontribusi token, bukan dari gambar PNG."
                        ),
                    )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.caption(
    "Dashboard penelitian klasifikasi berita menggunakan CNN, "
    "Attention-BiLSTM, representasi Title + Description, pengujian "
    "keyword YAKE, dan Explainable AI SHAP."
)
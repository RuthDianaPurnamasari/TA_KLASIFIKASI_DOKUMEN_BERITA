from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix

try:
    from langdetect import (
        DetectorFactory,
        LangDetectException,
        detect,
    )

    DetectorFactory.seed = 42
    LANGDETECT_AVAILABLE = True

except ImportError:
    LANGDETECT_AVAILABLE = False

    class LangDetectException(Exception):
        """Fallback exception ketika langdetect belum terpasang."""



# =============================================================================
# PROJECT PATH
# =============================================================================

CURRENT_FILE = Path(__file__).resolve()
STREAMLIT_DIR = CURRENT_FILE.parent

# app.py biasanya berada pada project_root/10_streamlit/app.py.
# Saat file ini diuji di lokasi lain, fallback kedua tetap tersedia.
PROJECT_ROOT = (
    CURRENT_FILE.parents[1]
    if CURRENT_FILE.parent.name == "10_streamlit"
    else CURRENT_FILE.parent
)

if str(STREAMLIT_DIR) not in sys.path:
    sys.path.insert(0, str(STREAMLIT_DIR))


# =============================================================================
# PROJECT IMPORTS
# =============================================================================

try:
    from config import RESEARCH_TITLE  # type: ignore  # noqa: E402
except Exception:
    RESEARCH_TITLE = (
        "Perbandingan Kinerja CNN dan Attention-BiLSTM pada Klasifikasi "
        "Berita Bahasa Indonesia Berdasarkan Skenario Representasi Teks"
    )

from utils.data_loader import (  # type: ignore  # noqa: E402
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
from utils.inference import predict_news  # type: ignore  # noqa: E402


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
# DIRECTORIES
# =============================================================================

RESULTS_DIR = PROJECT_ROOT / "9_results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
WORDCLOUD_DIR = FIGURES_DIR / "wordclouds"
WATERFALL_DIR = FIGURES_DIR / "shap" / "waterfall"


# =============================================================================
# STYLING
# =============================================================================

def load_css(css_path: Path) -> None:
    """Memuat CSS eksternal apabila tersedia."""

    if css_path.exists() and css_path.is_file():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


load_css(STREAMLIT_DIR / "assets" / "style.css")



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
    "K2": "Title + Description",
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

# Mapping ini merupakan mapping keluaran model final, bukan class index mentah
# one-based dari dataset AG News asli.
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

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

KOMPAS_TEST_SIZE = 1_000


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def find_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:
    """Mencari nama kolom tanpa membedakan huruf besar-kecil."""

    if dataframe.empty:
        return None

    mapping = {
        str(column).strip().lower(): str(column)
        for column in dataframe.columns
    }

    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in mapping:
            return mapping[key]

    return None


def normalize_metric(value: Any) -> float | None:
    """Menormalkan nilai metrik menjadi skala 0–1."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(numeric):
        return None

    return numeric / 100.0 if numeric > 1.0 else numeric


def percentage(value: Any, digits: int = 2) -> str:
    """Memformat metrik menjadi persentase."""

    normalized = normalize_metric(value)
    if normalized is None:
        return "-"

    return f"{normalized * 100:.{digits}f}%"


def number_id(value: Any, digits: int = 0) -> str:
    """Memformat angka menggunakan pemisah Indonesia."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"

    if not np.isfinite(numeric):
        return "-"

    formatted = f"{numeric:,.{digits}f}"
    return (
        formatted
        .replace(",", "TEMP")
        .replace(".", ",")
        .replace("TEMP", ".")
    )


def display_label(value: Any) -> str:
    """Merapikan label kategori."""

    normalized = (
        str(value)
        .strip()
        .lower()
        .replace("sci/tech", "sci_tech")
        .replace("sci-tech", "sci_tech")
        .replace(" ", "_")
    )

    return CATEGORY_DISPLAY.get(
        normalized,
        normalized.replace("_", " ").title(),
    )


def safe_bool(value: Any) -> bool:
    """Mengubah beberapa representasi nilai menjadi boolean."""

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "ya",
        "benar",
    }


def first_value(
    row: pd.Series,
    candidates: Iterable[str],
    default: Any = None,
) -> Any:
    """Mengambil nilai pertama dari beberapa kandidat nama kolom."""

    lower_mapping = {
        str(column).strip().lower(): column
        for column in row.index
    }

    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in lower_mapping:
            value = row[lower_mapping[key]]
            if pd.notna(value):
                return value

    return default


def load_csv(path: Path) -> pd.DataFrame:
    """Membaca CSV dengan fallback encoding dan hasil kosong saat gagal."""

    path = Path(path)
    if not path.exists() or not path.is_file():
        return pd.DataFrame()

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_csv_cached(path_text: str) -> pd.DataFrame:
    """Versi cache dari loader CSV."""

    return load_csv(Path(path_text))


def load_table(filename: str) -> pd.DataFrame:
    """Membaca tabel dari 9_results/tables."""

    return load_csv_cached(str(TABLES_DIR / filename))


def get_experiment_row(
    dataframe: pd.DataFrame,
    experiment_name: str,
) -> pd.DataFrame:
    """Mengambil baris eksperimen tertentu."""

    experiment_column = find_column(
        dataframe,
        ["experiment_name", "experiment", "experiment_id"],
    )

    if experiment_column is None:
        return pd.DataFrame()

    return dataframe[
        dataframe[experiment_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(experiment_name.lower())
    ].copy()


def get_metric(
    dataframe: pd.DataFrame,
    experiment_name: str,
    candidates: Iterable[str],
) -> float | None:
    """Mengambil satu metrik eksperimen dalam skala 0–1."""

    selected = get_experiment_row(dataframe, experiment_name)
    if selected.empty:
        return None

    column = find_column(selected, candidates)
    if column is None:
        return None

    return normalize_metric(selected.iloc[0][column])


def get_raw_numeric(
    dataframe: pd.DataFrame,
    experiment_name: str,
    candidates: Iterable[str],
) -> float | None:
    """Mengambil satu nilai numerik tanpa normalisasi persentase."""

    selected = get_experiment_row(dataframe, experiment_name)
    if selected.empty:
        return None

    column = find_column(selected, candidates)
    if column is None:
        return None

    value = pd.to_numeric(
        pd.Series([selected.iloc[0][column]]),
        errors="coerce",
    ).iloc[0]

    return float(value) if pd.notna(value) else None


def get_metric_bundle(
    dataframe: pd.DataFrame,
    experiment_name: str,
) -> dict[str, float | None]:
    """Mengambil seluruh metrik utama satu eksperimen."""

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
        "log_loss": get_raw_numeric(
            dataframe,
            experiment_name,
            ["log_loss", "test_log_loss", "categorical_crossentropy"],
        ),
        "inference_time_ms": get_raw_numeric(
            dataframe,
            experiment_name,
            ["inference_time_ms", "mean_inference_time_ms"],
        ),
    }


def prepare_chart_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Menyiapkan tabel evaluasi menjadi format grafik."""

    if dataframe.empty:
        return pd.DataFrame()

    experiment_column = find_column(
        dataframe,
        ["experiment_name", "experiment", "experiment_id"],
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
        result[experiment_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    result["Model"] = np.where(
        result["Eksperimen"].str.startswith("cnn"),
        "CNN",
        "Attention-BiLSTM",
    )
    result["Dataset"] = np.where(
        result["Eksperimen"].str.contains(r"_k[123]$", regex=True),
        "Kompas",
        "AG News",
    )
    result["Skenario"] = (
        result["Eksperimen"].str.split("_").str[-1].str.upper()
    )
    result["Representasi Teks"] = result["Skenario"].map(
        SCENARIO_NAMES
    )

    if accuracy_column is not None:
        accuracy = pd.to_numeric(
            result[accuracy_column],
            errors="coerce",
        )
        result["Accuracy (%)"] = np.where(
            accuracy.le(1),
            accuracy * 100,
            accuracy,
        )

    if f1_column is not None:
        f1_value = pd.to_numeric(
            result[f1_column],
            errors="coerce",
        )
        result["Macro F1 (%)"] = np.where(
            f1_value.le(1),
            f1_value * 100,
            f1_value,
        )

    return result


def style_chart(
    figure: go.Figure,
    y_title: str,
    y_range: list[float] | None = None,
    height: int = 500,
) -> go.Figure:
    """Memberikan format konsisten dan ruang label yang cukup."""

    figure.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=45, r=35, t=100, b=55),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="left",
            x=0,
        ),
        legend_title_text="",
        hovermode="closest",
        uniformtext_minsize=11,
        uniformtext_mode="show",
    )
    figure.update_xaxes(
        showgrid=False,
        automargin=True,
    )
    figure.update_yaxes(
        title=y_title,
        range=y_range,
        gridcolor="rgba(148, 163, 184, 0.22)",
        zeroline=False,
        automargin=True,
    )

    return figure


def add_percentage_labels(figure: go.Figure) -> None:
    """Menambahkan label persen yang jelas pada bar chart."""

    figure.update_traces(
        texttemplate="<b>%{text:.2f}%</b>",
        textposition="outside",
        textfont=dict(size=13),
        cliponaxis=False,
    )


def percentage_axis_range(
    values: pd.Series,
    padding_bottom: float = 1.0,
    padding_top: float = 1.0,
) -> list[float]:
    """Membuat rentang sumbu persentase dengan ruang untuk label."""

    numeric = pd.to_numeric(values, errors="coerce").dropna()

    if numeric.empty:
        return [0, 101]

    lower = max(0.0, float(numeric.min()) - padding_bottom)
    upper = min(101.0, float(numeric.max()) + padding_top)

    if upper - lower < 4:
        lower = max(0.0, upper - 4)

    return [lower, upper]


def show_result_explanation(
    main_result: str,
    how_to_read: str,
    interpretation: str,
    conclusion: str,
) -> None:
    """Menampilkan ringkasan dan interpretasi hasil."""

    st.caption(f"**Ringkasan hasil:** {main_result}")

    with st.expander("Cara membaca dan interpretasi"):
        st.markdown(f"**Cara membaca:** {how_to_read}")
        st.markdown(f"**Interpretasi:** {interpretation}")
        st.markdown(f"**Kesimpulan:** {conclusion}")


def display_dataframe(
    dataframe: pd.DataFrame,
    columns: list[str] | None = None,
) -> None:
    """Menampilkan tabel secara aman."""

    if dataframe.empty:
        st.info("Data belum tersedia.")
        return

    table = dataframe.copy()
    if columns:
        selected = [column for column in columns if column in table.columns]
        if selected:
            table = table[selected]

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )


def show_technical_table(
    title: str,
    dataframe: pd.DataFrame,
    columns: list[str] | None = None,
) -> None:
    """Menempatkan tabel teknis di dalam expander tertutup."""

    if dataframe.empty:
        return

    with st.expander(title, expanded=False):
        display_dataframe(
            dataframe=dataframe,
            columns=columns,
        )


def dataset_display_name(value: Any) -> str:
    """Mengubah kode dataset menjadi nama yang mudah dibaca."""

    mapping = {
        "kompas": "Kompas",
        "ag_news_train": "AG News Train",
        "ag_news_test": "AG News Test",
    }

    normalized = str(value).strip().lower()
    return mapping.get(
        normalized,
        normalized.replace("_", " ").title(),
    )


def get_cleaning_counts(
    dataframe: pd.DataFrame,
    dataset_name: str,
    fallback_before: int,
    fallback_after: int,
) -> tuple[int, int]:
    """Mengambil jumlah sebelum dan setelah cleaning."""

    if dataframe.empty:
        return fallback_before, fallback_after

    dataset_column = find_column(dataframe, ["dataset"])
    before_column = find_column(
        dataframe,
        ["jumlah_sebelum_cleaning", "before_cleaning", "jumlah_awal"],
    )
    after_column = find_column(
        dataframe,
        ["jumlah_setelah_cleaning", "after_cleaning", "jumlah_final"],
    )

    if (
        dataset_column is None
        or before_column is None
        or after_column is None
    ):
        return fallback_before, fallback_after

    selected = dataframe[
        dataframe[dataset_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(dataset_name.lower())
    ]

    if selected.empty:
        return fallback_before, fallback_after

    before = pd.to_numeric(
        pd.Series([selected.iloc[0][before_column]]),
        errors="coerce",
    ).iloc[0]
    after = pd.to_numeric(
        pd.Series([selected.iloc[0][after_column]]),
        errors="coerce",
    ).iloc[0]

    return (
        int(before) if pd.notna(before) else fallback_before,
        int(after) if pd.notna(after) else fallback_after,
    )


def get_sequence_coverage(
    dataframe: pd.DataFrame,
    dataset_name: str,
    text_keyword: str,
    max_length: int,
    fallback: float,
) -> float:
    """Mengambil coverage sequence yang benar-benar digunakan."""

    if dataframe.empty:
        return fallback

    dataset_column = find_column(dataframe, ["dataset"])
    field_column = find_column(
        dataframe,
        ["text_field", "field", "component", "text_source"],
    )
    length_column = find_column(
        dataframe,
        ["max_length", "sequence_length", "panjang_sequence"],
    )
    coverage_column = find_column(
        dataframe,
        [
            "persentase_tertampung",
            "coverage_percent",
            "coverage_percentage",
            "persentase_cakupan",
            "coverage",
        ],
    )

    if (
        dataset_column is None
        or field_column is None
        or length_column is None
        or coverage_column is None
    ):
        return fallback

    dataset_values = (
        dataframe[dataset_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    field_values = (
        dataframe[field_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    lengths = pd.to_numeric(
        dataframe[length_column],
        errors="coerce",
    )

    selected = dataframe[
        dataset_values.eq(dataset_name.lower())
        & field_values.str.contains(text_keyword.lower(), regex=False)
        & lengths.eq(max_length)
    ]

    if selected.empty:
        return fallback

    value = pd.to_numeric(
        pd.Series([selected.iloc[0][coverage_column]]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(value):
        return fallback

    numeric = float(value)
    return numeric * 100 if numeric <= 1 else numeric



# =============================================================================
# INPUT LANGUAGE VALIDATION
# =============================================================================

INDONESIAN_COMMON_WORDS = {
    "yang",
    "dan",
    "di",
    "ke",
    "dari",
    "untuk",
    "dengan",
    "pada",
    "adalah",
    "dalam",
    "ini",
    "itu",
    "oleh",
    "setelah",
    "akan",
    "telah",
    "sebagai",
    "karena",
    "terhadap",
    "menjadi",
    "juga",
    "tidak",
    "lebih",
    "berita",
    "pemerintah",
    "indonesia",
}

ENGLISH_COMMON_WORDS = {
    "the",
    "and",
    "of",
    "to",
    "in",
    "for",
    "with",
    "on",
    "is",
    "was",
    "are",
    "from",
    "after",
    "this",
    "that",
    "has",
    "have",
    "will",
    "by",
    "as",
    "at",
    "its",
    "their",
    "report",
    "released",
}


def tokenize_for_language_validation(
    text: str,
) -> list[str]:
    """
    Tokenisasi ringan untuk validasi bahasa input dashboard.

    Fungsi ini bukan bagian dari preprocessing model.
    """

    return re.findall(
        r"[a-zA-ZÀ-ÿ]+",
        str(text).lower(),
    )


def detect_input_language(
    title: str,
    description: str,
) -> str:
    """
    Mendeteksi bahasa gabungan Title dan Description.

    Validasi ini hanya digunakan pada dashboard untuk mencegah input
    nonbahasa Indonesia masuk ke model Kompas. Validasi ini tidak mengubah
    model, vocabulary, preprocessing, atau hasil eksperimen penelitian.
    """

    combined_text = (
        f"{title.strip()} {description.strip()}"
    ).strip()

    if not combined_text:
        return "unknown"

    if LANGDETECT_AVAILABLE:
        try:
            return str(
                detect(combined_text)
            ).strip().lower()
        except LangDetectException:
            pass

    tokens = tokenize_for_language_validation(
        combined_text
    )

    if not tokens:
        return "unknown"

    indonesian_score = sum(
        token in INDONESIAN_COMMON_WORDS
        for token in tokens
    )
    english_score = sum(
        token in ENGLISH_COMMON_WORDS
        for token in tokens
    )

    if (
        indonesian_score >= 2
        and indonesian_score > english_score
    ):
        return "id"

    if (
        english_score >= 2
        and english_score > indonesian_score
    ):
        return "en"

    return "unknown"


# =============================================================================
# DATASET AND EDA HELPERS
# =============================================================================

def get_eda_dataset_row(
    eda_summary: pd.DataFrame,
    dataset_name: str,
) -> pd.Series | None:
    """Mengambil ringkasan satu dataset dari eda_summary.csv."""

    dataset_column = find_column(eda_summary, ["dataset"])
    if dataset_column is None:
        return None

    selected = eda_summary[
        eda_summary[dataset_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(dataset_name.lower())
    ]

    return None if selected.empty else selected.iloc[0]


def prepare_class_distribution(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Menyeragamkan tabel distribusi kelas."""

    if dataframe.empty:
        return pd.DataFrame()

    dataset_column = find_column(dataframe, ["dataset"])
    category_column = find_column(
        dataframe,
        ["category", "kategori", "class_name", "label"],
    )
    count_column = find_column(
        dataframe,
        ["count", "jumlah_data", "jumlah_artikel", "frequency", "jumlah"],
    )
    stage_column = find_column(dataframe, ["stage", "tahap"])

    if category_column is None or count_column is None:
        return pd.DataFrame()

    selected = dataframe.copy()

    if dataset_column is not None:
        selected = selected[
            selected[dataset_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .eq(dataset_name.lower())
        ]

    if stage_column is not None:
        stage_values = (
            selected[stage_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        final_mask = stage_values.isin(
            {"setelah_cleaning", "after_cleaning", "final", "clean"}
        )
        if final_mask.any():
            selected = selected[final_mask]

    result = pd.DataFrame(
        {
            "Kategori": selected[category_column].map(display_label),
            "Jumlah Artikel": pd.to_numeric(
                selected[count_column],
                errors="coerce",
            ),
        }
    ).dropna(subset=["Jumlah Artikel"])

    return result.sort_values("Kategori").reset_index(drop=True)


def prepare_word_frequency(
    dataframe: pd.DataFrame,
    dataset_name: str,
    top_n: int,
) -> pd.DataFrame:
    """Menyiapkan top words dari hasil tahap EDA 3.4."""

    if dataframe.empty:
        return pd.DataFrame()

    dataset_column = find_column(dataframe, ["dataset"])
    category_column = find_column(dataframe, ["category", "kategori"])
    rank_column = find_column(dataframe, ["rank", "peringkat"])
    word_column = find_column(dataframe, ["word", "token", "kata"])
    frequency_column = find_column(
        dataframe,
        ["frequency", "frekuensi", "count"],
    )
    document_frequency_column = find_column(
        dataframe,
        ["document_frequency", "jumlah_dokumen"],
    )
    document_percentage_column = find_column(
        dataframe,
        ["document_percentage", "persentase_dokumen"],
    )

    if (
        dataset_column is None
        or word_column is None
        or frequency_column is None
    ):
        return pd.DataFrame()

    selected = dataframe[
        dataframe[dataset_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(dataset_name.lower())
    ].copy()

    if category_column is not None:
        category_values = (
            selected[category_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        overall_mask = category_values.eq("all")
        if overall_mask.any():
            selected = selected[overall_mask]

    if rank_column is not None:
        selected[rank_column] = pd.to_numeric(
            selected[rank_column],
            errors="coerce",
        )
        selected = selected.sort_values(rank_column)
    else:
        selected[frequency_column] = pd.to_numeric(
            selected[frequency_column],
            errors="coerce",
        )
        selected = selected.sort_values(
            frequency_column,
            ascending=False,
        )

    result = pd.DataFrame(
        {
            "Kata": selected[word_column].astype(str),
            "Frekuensi": pd.to_numeric(
                selected[frequency_column],
                errors="coerce",
            ),
        }
    )

    if document_frequency_column is not None:
        result["Jumlah Dokumen"] = pd.to_numeric(
            selected[document_frequency_column],
            errors="coerce",
        ).to_numpy()

    if document_percentage_column is not None:
        result["Persentase Dokumen"] = pd.to_numeric(
            selected[document_percentage_column],
            errors="coerce",
        ).to_numpy()

    return result.dropna(subset=["Frekuensi"]).head(top_n)


def resolve_wordcloud_path(
    dataset_name: str,
    category_name: str,
    summary_row: pd.Series | None,
) -> Path | None:
    """Mencari file word cloud resmi."""

    if summary_row is not None:
        path_value = first_value(
            summary_row,
            ["output_path", "figure_path", "path"],
        )
        if path_value:
            path = Path(str(path_value))
            if path.exists():
                return path

    if dataset_name == "kompas":
        prefix = "kompas"
    else:
        prefix = "agnews_train"

    if category_name == "all":
        filename = f"{prefix}_overall_wordcloud.png"
    else:
        safe_category = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            category_name.lower(),
        ).strip("_")
        filename = f"{prefix}_{safe_category}_wordcloud.png"

    candidate = WORDCLOUD_DIR / filename
    return candidate if candidate.exists() else None


# =============================================================================
# PREDICTION, CONFUSION MATRIX, AND TRAINING CURVE HELPERS
# =============================================================================

@st.cache_data(show_spinner=False)
def load_prediction_result(experiment_name: str) -> pd.DataFrame:
    """Membaca file prediksi final suatu eksperimen."""

    path = RESULTS_DIR / "predictions" / f"{experiment_name}_predictions.csv"
    return load_csv(path)


@st.cache_data(show_spinner=False)
def load_training_history(experiment_name: str) -> pd.DataFrame:
    """Membaca training history final suatu eksperimen."""

    candidates = [
        RESULTS_DIR / "training_history" / f"{experiment_name}_history.csv",
        RESULTS_DIR / "training_history" / f"{experiment_name}.csv",
        RESULTS_DIR / "logs" / f"{experiment_name}_training.log.csv",
    ]

    for candidate in candidates:
        dataframe = load_csv(candidate)
        if not dataframe.empty:
            return dataframe

    return pd.DataFrame()


def normalize_prediction_labels(
    values: pd.Series,
    dataset_name: str,
) -> pd.Series:
    """Menormalisasi label keluaran model atau label teks."""

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

        return (
            text
            .replace("sci/tech", "sci_tech")
            .replace("sci-tech", "sci_tech")
            .replace(" ", "_")
        )

    return values.apply(normalize_one)


def create_confusion_matrix_chart(
    prediction_data: pd.DataFrame,
    dataset_name: str,
    normalized: bool,
) -> tuple[go.Figure | None, dict[str, Any] | None]:
    """Membuat confusion matrix dari prediksi final."""

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
        annotation = np.array(
            [[f"{value:.1f}%" for value in row] for row in matrix_display]
        )
        colorbar_title = "Persentase"
        hovertemplate = (
            "<b>Aktual:</b> %{y}<br>"
            "<b>Prediksi:</b> %{x}<br>"
            "<b>Nilai:</b> %{z:.2f}%"
            "<extra></extra>"
        )
    else:
        matrix_display = matrix_count
        annotation = matrix_count.astype(str)
        colorbar_title = "Jumlah Data"
        hovertemplate = (
            "<b>Aktual:</b> %{y}<br>"
            "<b>Prediksi:</b> %{x}<br>"
            "<b>Jumlah:</b> %{z:.0f}"
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
        actual_error = labels[error_position[0]]
        predicted_error = labels[error_position[1]]
    else:
        actual_error = "-"
        predicted_error = "-"

    display_labels = [display_label(label) for label in labels]

    figure = go.Figure(
        data=go.Heatmap(
            z=matrix_display,
            x=display_labels,
            y=display_labels,
            text=annotation,
            texttemplate="%{text}",
            hovertemplate=hovertemplate,
            colorbar=dict(title=colorbar_title),
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

    return figure, {
        "total_data": total_data,
        "correct": total_correct,
        "wrong": total_wrong,
        "accuracy": accuracy_value,
        "largest_error": largest_error,
        "actual_error_class": actual_error,
        "predicted_error_class": predicted_error,
    }


def create_training_curve_charts(
    history: pd.DataFrame,
) -> tuple[go.Figure | None, go.Figure | None, dict[str, Any] | None]:
    """Membuat grafik accuracy dan loss dari history CSV."""

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

    history = history.copy()

    if epoch_column is None:
        history["Epoch"] = np.arange(1, len(history) + 1)
        epoch_column = "Epoch"
    else:
        epoch_numeric = pd.to_numeric(
            history[epoch_column],
            errors="coerce",
        )
        if epoch_numeric.notna().any() and epoch_numeric.min() == 0:
            history[epoch_column] = epoch_numeric + 1

    accuracy_figure: go.Figure | None = None
    loss_figure: go.Figure | None = None

    if accuracy_column and val_accuracy_column:
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
            title="Accuracy Train dan Validation",
        )
        accuracy_figure = style_chart(
            accuracy_figure,
            y_title="Accuracy",
            height=440,
        )

    if loss_column and val_loss_column:
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
            title="Loss Train dan Validation",
        )
        loss_figure = style_chart(
            loss_figure,
            y_title="Loss",
            height=440,
        )

    best_epoch: Any = "-"
    best_val_loss = np.nan
    best_val_accuracy = np.nan

    if val_loss_column:
        val_loss = pd.to_numeric(
            history[val_loss_column],
            errors="coerce",
        )
        if val_loss.notna().any():
            best_index = val_loss.idxmin()
            best_epoch = history.loc[best_index, epoch_column]
            best_val_loss = float(val_loss.loc[best_index])

    if val_accuracy_column:
        val_accuracy = pd.to_numeric(
            history[val_accuracy_column],
            errors="coerce",
        )
        if val_accuracy.notna().any():
            best_val_accuracy = float(val_accuracy.max())

    return accuracy_figure, loss_figure, {
        "epochs": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
    }


# =============================================================================
# SHAP HELPERS
# =============================================================================

def prepare_local_token_data(
    contributions: pd.DataFrame,
    document_id: str,
    top_n: int = 20,
) -> tuple[pd.DataFrame, str | None, str | None]:
    """Menyiapkan kontribusi token local SHAP."""

    sample_column = find_column(
        contributions,
        ["document_id", "sample_id", "id"],
    )
    token_column = find_column(
        contributions,
        ["token", "word"],
    )
    contribution_column = find_column(
        contributions,
        ["signed_shap", "shap_value", "contribution", "token_contribution"],
    )

    if (
        sample_column is None
        or token_column is None
        or contribution_column is None
    ):
        return pd.DataFrame(), token_column, contribution_column

    selected = contributions[
        contributions[sample_column]
        .astype(str)
        .eq(str(document_id))
    ].copy()

    if selected.empty:
        return selected, token_column, contribution_column

    selected = selected[
        ~selected[token_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .isin(SPECIAL_TOKENS)
    ].copy()

    selected[contribution_column] = pd.to_numeric(
        selected[contribution_column],
        errors="coerce",
    )
    selected = selected.dropna(subset=[contribution_column])
    selected["absolute_contribution"] = selected[contribution_column].abs()
    selected = (
        selected
        .sort_values("absolute_contribution", ascending=False)
        .head(top_n)
    )

    return selected, token_column, contribution_column


def resolve_waterfall_image(row: pd.Series) -> Path | None:
    """Mencari gambar waterfall resmi berdasarkan ringkasan."""

    path_value = first_value(
        row,
        ["figure_path", "output_path", "waterfall_path", "image_path"],
    )

    if path_value:
        path = Path(str(path_value))
        if path.exists():
            return path

        # Saat path tersimpan absolut dari komputer lain, gunakan nama file.
        candidate = WATERFALL_DIR / path.name
        if candidate.exists():
            return candidate

    document_id = str(
        first_value(row, ["document_id", "sample_id", "id"], "")
    ).strip()

    if document_id and WATERFALL_DIR.exists():
        matches = sorted(WATERFALL_DIR.glob(f"{document_id}*_waterfall.png"))
        if matches:
            return matches[0]

    return None


# =============================================================================
# DATA LOADING
# =============================================================================

evaluation_data = load_test_evaluation()
chart_data = prepare_chart_data(evaluation_data)
eda_summary = load_table("eda_summary.csv")


# =============================================================================
# HEADER AND TOP METRICS
# =============================================================================

st.title("📰 Dashboard Klasifikasi Berita Berbahasa Indonesia")
st.caption(RESEARCH_TITLE)
st.info(
    "Dashboard menampilkan dataset final, EDA, hasil 10 eksperimen, "
    "perbandingan YAKE, prediksi berita baru, dan interpretasi SHAP. "
    "Seluruh hasil penelitian dibaca dari folder `9_results`."
)

kompas_row = get_eda_dataset_row(eda_summary, "kompas")
kompas_count = (
    int(first_value(kompas_row, ["jumlah_data"], 9_997))
    if kompas_row is not None
    else 9_997
)

best_experiment = "cnn_k2"
best_accuracy = get_metric(
    evaluation_data,
    best_experiment,
    ["accuracy", "test_accuracy"],
)
best_f1 = get_metric(
    evaluation_data,
    best_experiment,
    ["f1_macro", "macro_f1", "f1_score_macro"],
)
experiment_count = int(len(evaluation_data)) if not evaluation_data.empty else 10
correct_predictions = (
    round(KOMPAS_TEST_SIZE * best_accuracy)
    if best_accuracy is not None
    else 969
)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric(
    "Data Kompas Final",
    f"{number_id(kompas_count)} artikel",
)
metric_2.metric(
    "Jumlah Eksperimen",
    f"{number_id(experiment_count)} eksperimen",
)
metric_3.metric(
    "Model Deployment",
    "CNN K2",
)
metric_4.metric(
    "Accuracy CNN K2",
    percentage(best_accuracy),
    help=(
        f"CNN K2 memprediksi benar {correct_predictions} dari "
        f"{KOMPAS_TEST_SIZE} artikel test Kompas."
    ),
)


# =============================================================================
# HORIZONTAL NAVIGATION
# =============================================================================

PAGES = [
    "🏠 Ringkasan",
    "📚 Dataset & EDA",
    "📊 Hasil & Metrik",
    "🔑 Perbandingan YAKE",
    "📰 Prediksi Berita",
    "🔍 Explainable AI",
]

selected_page = st.radio(
    "Navigasi",
    PAGES,
    horizontal=True,
    label_visibility="collapsed",
    key="main_navigation",
)

st.divider()


# =============================================================================
# PAGE: SUMMARY
# =============================================================================

if selected_page == "🏠 Ringkasan":
    st.header("Ringkasan Penelitian")
    st.caption(
        "Ringkasan menampilkan alur data, desain eksperimen, dan hasil utama "
        "tanpa tabel teknis."
    )

    cleaning_summary = load_table("cleaning_integrity_summary.csv")

    kompas_before, kompas_after = get_cleaning_counts(
        cleaning_summary,
        "kompas",
        10_000,
        9_997,
    )
    ag_train_before, ag_train_after = get_cleaning_counts(
        cleaning_summary,
        "ag_news_train",
        120_000,
        119_817,
    )
    ag_test_before, ag_test_after = get_cleaning_counts(
        cleaning_summary,
        "ag_news_test",
        7_600,
        7_600,
    )

    st.subheader("Alur Data Penelitian")
    flow_1, flow_2, flow_3 = st.columns(3)

    with flow_1:
        with st.container(border=True):
            st.markdown("### Kompas")
            st.metric("Data Awal", f"{number_id(kompas_before)} artikel")
            st.markdown(
                f"**Cleaning:** {number_id(kompas_before - kompas_after)} "
                "artikel dihapus"
            )
            st.markdown(
                f"**Data Final:** {number_id(kompas_after)} artikel"
            )
            st.markdown(
                f"**Preprocessing:** {number_id(kompas_after)} artikel"
            )
            st.caption(
                "Dataset utama berbahasa Indonesia dengan kategori Bola, "
                "Global, Money, dan Tekno."
            )

    with flow_2:
        with st.container(border=True):
            st.markdown("### AG News Train")
            st.metric("Data Awal", f"{number_id(ag_train_before)} artikel")
            st.markdown(
                f"**Cleaning:** {number_id(ag_train_before - ag_train_after)} "
                "artikel dihapus"
            )
            st.markdown(
                f"**Data Final:** {number_id(ag_train_after)} artikel"
            )
            st.markdown(
                f"**Preprocessing:** {number_id(ag_train_after)} artikel"
            )
            st.caption(
                "Dataset benchmark berbahasa Inggris untuk proses pelatihan "
                "dan validasi."
            )

    with flow_3:
        with st.container(border=True):
            st.markdown("### AG News Test")
            st.metric("Data Awal", f"{number_id(ag_test_before)} artikel")
            st.markdown(
                f"**Cleaning:** {number_id(ag_test_before - ag_test_after)} "
                "artikel dihapus"
            )
            st.markdown(
                f"**Data Final:** {number_id(ag_test_after)} artikel"
            )
            st.markdown(
                f"**Preprocessing:** {number_id(ag_test_after)} artikel"
            )
            st.caption(
                "Data held-out yang hanya digunakan untuk evaluasi akhir."
            )

    st.info(
        "Cleaning memeriksa missing value, duplikat, konflik label, dan "
        "overlap train-test. Preprocessing mengubah teks menjadi input model "
        "tanpa mengurangi jumlah artikel final."
    )

    overview_1, overview_2 = st.columns(2)

    with overview_1:
        with st.container(border=True):
            st.subheader("Model yang Dibandingkan")
            st.markdown(
                """
                - **CNN** mempelajari pola lokal atau n-gram penting.
                - **Attention-BiLSTM** mempelajari urutan dua arah dan
                  memberikan bobot perhatian pada bagian teks.
                """
            )

    with overview_2:
        with st.container(border=True):
            st.subheader("Skenario Representasi Teks")
            st.markdown(
                """
                - **K1:** Title
                - **K2:** Title + Description
                - **K3:** Title + Description + Keyword YAKE
                - **A1:** Title
                - **A2:** Title + Description
                """
            )

    cnn_k1 = get_metric_bundle(evaluation_data, "cnn_k1")
    cnn_k2 = get_metric_bundle(evaluation_data, "cnn_k2")
    cnn_k3 = get_metric_bundle(evaluation_data, "cnn_k3")
    attention_k2 = get_metric_bundle(evaluation_data, "attention_bilstm_k2")
    attention_k3 = get_metric_bundle(evaluation_data, "attention_bilstm_k3")

    description_change = (
        (cnn_k2["accuracy"] - cnn_k1["accuracy"]) * 100
        if cnn_k1["accuracy"] is not None and cnn_k2["accuracy"] is not None
        else np.nan
    )
    cnn_yake_change = (
        (cnn_k3["accuracy"] - cnn_k2["accuracy"]) * 100
        if cnn_k2["accuracy"] is not None and cnn_k3["accuracy"] is not None
        else np.nan
    )
    attention_yake_change = (
        (attention_k3["accuracy"] - attention_k2["accuracy"]) * 100
        if (
            attention_k2["accuracy"] is not None
            and attention_k3["accuracy"] is not None
        )
        else np.nan
    )

    st.subheader("Temuan Utama")
    finding_1, finding_2, finding_3 = st.columns(3)

    with finding_1:
        st.success(
            "**Description membantu klasifikasi**\n\n"
            f"Accuracy CNN berubah dari {percentage(cnn_k1['accuracy'])} "
            f"menjadi {percentage(cnn_k2['accuracy'])} "
            f"({description_change:+.2f} pp)."
        )

    with finding_2:
        st.warning(
            "**YAKE belum mengungguli K2**\n\n"
            f"CNN berubah {cnn_yake_change:+.2f} pp dan "
            f"Attention-BiLSTM berubah {attention_yake_change:+.2f} pp."
        )

    with finding_3:
        st.info(
            "**Model deployment: CNN K2**\n\n"
            f"Accuracy {percentage(best_accuracy)} dan Macro F1 "
            f"{percentage(best_f1, 4)} pada test Kompas."
        )

    st.subheader("Perbandingan Model pada Dataset Kompas")

    kompas_chart = (
        chart_data[
            chart_data.get(
                "Dataset",
                pd.Series(dtype=str),
            ).eq("Kompas")
        ].copy()
        if not chart_data.empty
        else pd.DataFrame()
    )

    if kompas_chart.empty or "Accuracy (%)" not in kompas_chart.columns:
        st.info("Data evaluasi Kompas belum tersedia.")
    else:
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
        add_percentage_labels(figure)
        figure = style_chart(
            figure,
            y_title="Accuracy (%)",
            y_range=percentage_axis_range(
                kompas_chart["Accuracy (%)"],
                padding_bottom=1.0,
                padding_top=1.2,
            ),
            height=500,
        )
        st.plotly_chart(
            figure,
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

        best_row = kompas_chart.loc[
            kompas_chart["Accuracy (%)"].idxmax()
        ]

        show_result_explanation(
            main_result=(
                f"Eksperimen terbaik pada Kompas adalah "
                f"**{str(best_row['Eksperimen']).upper()}** dengan Accuracy "
                f"**{best_row['Accuracy (%)']:.2f}%**."
            ),
            how_to_read=(
                "Setiap kelompok skenario menampilkan satu batang CNN dan "
                "satu batang Attention-BiLSTM."
            ),
            interpretation=(
                "K2 memberi konteks lebih lengkap daripada Title saja, "
                "sedangkan penambahan keyword YAKE pada K3 tidak menghasilkan "
                "peningkatan lanjutan."
            ),
            conclusion=(
                "CNN K2 dipilih sebagai model deployment karena memperoleh "
                "hasil test terbaik pada dataset utama."
            ),
        )


# =============================================================================
# PAGE: DATASET AND EDA
# =============================================================================

elif selected_page == "📚 Dataset & EDA":
    st.header("Dataset dan Exploratory Data Analysis")
    st.caption(
        "Bagian ini membaca output resmi EDA tahap 3.1–3.7. "
        "Dashboard tidak menghitung ulang EDA dari dataset mentah."
    )

    eda_section = st.radio(
        "Bagian EDA",
        [
            "Ringkasan & Cleaning",
            "Distribusi Kelas",
            "Statistik Teks",
            "Frekuensi & Word Cloud",
            "Analisis Temporal",
        ],
        horizontal=True,
        key="eda_navigation",
    )

    if eda_section == "Ringkasan & Cleaning":
        cleaning_integrity = load_table("cleaning_integrity_summary.csv")
        stage_validation = load_table("eda_stage_validation.csv")

        kompas_before, kompas_after = get_cleaning_counts(
            cleaning_integrity,
            "kompas",
            10_000,
            9_997,
        )
        ag_train_before, ag_train_after = get_cleaning_counts(
            cleaning_integrity,
            "ag_news_train",
            120_000,
            119_817,
        )
        ag_test_before, ag_test_after = get_cleaning_counts(
            cleaning_integrity,
            "ag_news_test",
            7_600,
            7_600,
        )

        st.subheader("Alur Data: Awal → Cleaning → Preprocessing")

        dataset_cards = st.columns(3)
        flow_data = [
            (
                dataset_cards[0],
                "Kompas",
                kompas_before,
                kompas_after,
                "Dataset utama",
            ),
            (
                dataset_cards[1],
                "AG News Train",
                ag_train_before,
                ag_train_after,
                "Benchmark training",
            ),
            (
                dataset_cards[2],
                "AG News Test",
                ag_test_before,
                ag_test_after,
                "Evaluasi akhir",
            ),
        ]

        for column, name, before, after, role in flow_data:
            with column:
                with st.container(border=True):
                    st.markdown(f"### {name}")
                    st.metric("Data Awal", f"{number_id(before)} artikel")
                    st.markdown(
                        f"**Dihapus saat cleaning:** "
                        f"{number_id(before - after)}"
                    )
                    st.markdown(
                        f"**Setelah cleaning:** {number_id(after)}"
                    )
                    st.markdown(
                        f"**Setelah preprocessing:** {number_id(after)}"
                    )
                    st.caption(role)

        st.subheader("Kualitas Dataset Final")
        quality_1, quality_2, quality_3, quality_4 = st.columns(4)
        quality_1.metric("Teks Kosong", "0")
        quality_2.metric("Duplikat Artikel", "0")
        quality_3.metric("Konflik Label", "0")
        quality_4.metric("Overlap Train-Test", "0")

        st.success(
            "Dataset final telah melewati pemeriksaan missing value, "
            "duplikat artikel, konflik label, dan overlap train-test."
        )

        show_technical_table(
            "Lihat detail teknis cleaning",
            cleaning_integrity,
            columns=[
                "dataset",
                "jumlah_sebelum_cleaning",
                "jumlah_setelah_cleaning",
                "jumlah_dihapus",
                "status",
            ],
        )

        show_technical_table(
            "Lihat validasi output EDA",
            stage_validation,
            columns=[
                "stage",
                "file_name",
                "status",
            ],
        )

    elif eda_section == "Distribusi Kelas":
        class_distribution = load_table("class_distribution.csv")

        kompas_class = prepare_class_distribution(
            class_distribution,
            "kompas",
        )
        agnews_class = prepare_class_distribution(
            class_distribution,
            "ag_news_train",
        )

        class_1, class_2 = st.columns(2)

        with class_1:
            st.subheader("Kompas")
            if kompas_class.empty:
                st.info("Distribusi kelas Kompas belum tersedia.")
            else:
                figure = px.bar(
                    kompas_class,
                    x="Kategori",
                    y="Jumlah Artikel",
                    text="Jumlah Artikel",
                    title="Distribusi Kelas Kompas Final",
                )
                figure.update_traces(
                    texttemplate="<b>%{text:,.0f}</b>",
                    textposition="outside",
                    textfont=dict(size=13),
                    cliponaxis=False,
                )
                figure = style_chart(
                    figure,
                    y_title="Jumlah Artikel",
                    y_range=[
                        0,
                        float(kompas_class["Jumlah Artikel"].max()) * 1.15,
                    ],
                    height=450,
                )
                st.plotly_chart(
                    figure,
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )

        with class_2:
            st.subheader("AG News Train")
            if agnews_class.empty:
                st.info("Distribusi kelas AG News Train belum tersedia.")
            else:
                figure = px.bar(
                    agnews_class,
                    x="Kategori",
                    y="Jumlah Artikel",
                    text="Jumlah Artikel",
                    title="Distribusi Kelas AG News Train Final",
                )
                figure.update_traces(
                    texttemplate="<b>%{text:,.0f}</b>",
                    textposition="outside",
                    textfont=dict(size=13),
                    cliponaxis=False,
                )
                figure = style_chart(
                    figure,
                    y_title="Jumlah Artikel",
                    y_range=[
                        0,
                        float(agnews_class["Jumlah Artikel"].max()) * 1.15,
                    ],
                    height=450,
                )
                st.plotly_chart(
                    figure,
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )

        st.info(
            "Distribusi kelas kedua dataset hampir seimbang. Karena itu, "
            "eksperimen tidak memerlukan oversampling atau undersampling."
        )

    elif eda_section == "Statistik Teks":
        text_statistics = load_table("text_statistics.csv")
        sequence_coverage = load_table("sequence_length_coverage.csv")

        st.subheader("Rata-rata Panjang Teks")

        kompas_summary_row = get_eda_dataset_row(
            eda_summary,
            "kompas",
        )
        ag_train_summary_row = get_eda_dataset_row(
            eda_summary,
            "ag_news_train",
        )

        kompas_metrics = st.columns(4)
        kompas_metrics[0].metric(
            "Kompas — Title",
            f"{number_id(first_value(kompas_summary_row, ['avg_words_title'], 10.7146), 2)} kata",
        )
        kompas_metrics[1].metric(
            "Kompas — Description",
            f"{number_id(first_value(kompas_summary_row, ['avg_words_description'], 17.8679), 2)} kata",
        )
        kompas_metrics[2].metric(
            "Kompas — Gabungan",
            f"{number_id(first_value(kompas_summary_row, ['avg_words_title_description'], 28.5825), 2)} kata",
        )
        kompas_metrics[3].metric(
            "Kompas — Content",
            f"{number_id(first_value(kompas_summary_row, ['avg_words_content'], 330.3673), 2)} kata",
        )

        ag_metrics = st.columns(3)
        ag_metrics[0].metric(
            "AG News — Title",
            f"{number_id(first_value(ag_train_summary_row, ['avg_words_title'], 6.7841), 2)} kata",
        )
        ag_metrics[1].metric(
            "AG News — Description",
            f"{number_id(first_value(ag_train_summary_row, ['avg_words_description'], 31.0607), 2)} kata",
        )
        ag_metrics[2].metric(
            "AG News — Gabungan",
            f"{number_id(first_value(ag_train_summary_row, ['avg_words_title_description'], 37.8447), 2)} kata",
        )

        if text_statistics.empty:
            st.info("File `text_statistics.csv` belum tersedia.")
        else:
            dataset_column = find_column(text_statistics, ["dataset"])
            field_column = find_column(
                text_statistics,
                ["text_field", "field", "component", "kolom", "text_source"],
            )
            mean_column = find_column(
                text_statistics,
                ["mean", "average", "avg_words", "rata_rata"],
            )
            measurement_column = find_column(
                text_statistics,
                [
                    "measurement",
                    "measure",
                    "unit",
                    "metric_type",
                    "statistic_type",
                ],
            )

            if dataset_column and field_column and mean_column:
                chart_frame = text_statistics.copy()

                if measurement_column is not None:
                    measurement_values = (
                        chart_frame[measurement_column]
                        .fillna("")
                        .astype(str)
                        .str.strip()
                        .str.lower()
                    )
                    word_mask = measurement_values.str.contains(
                        "word",
                        regex=False,
                    )
                    if word_mask.any():
                        chart_frame = chart_frame[word_mask]

                chart_frame = chart_frame[
                    chart_frame[dataset_column]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(["kompas", "ag_news_train"])
                ].copy()

                field_values = (
                    chart_frame[field_column]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )
                chart_frame = chart_frame[
                    ~field_values.eq("content")
                ].copy()

                chart_frame["Dataset"] = chart_frame[dataset_column].map(
                    dataset_display_name
                )
                chart_frame["Komponen"] = (
                    chart_frame[field_column]
                    .astype(str)
                    .str.replace("_", " ", regex=False)
                    .str.title()
                    .str.replace(
                        "Title Description",
                        "Title + Description",
                        regex=False,
                    )
                )
                chart_frame["Rata-rata Kata"] = pd.to_numeric(
                    chart_frame[mean_column],
                    errors="coerce",
                )
                chart_frame = chart_frame.dropna(
                    subset=["Rata-rata Kata"]
                )

                figure = px.bar(
                    chart_frame,
                    x="Komponen",
                    y="Rata-rata Kata",
                    color="Dataset",
                    barmode="group",
                    text="Rata-rata Kata",
                    title="Rata-rata Kata pada Input Utama Model",
                    category_orders={
                        "Komponen": [
                            "Title",
                            "Description",
                            "Title + Description",
                        ],
                        "Dataset": [
                            "Kompas",
                            "AG News Train",
                        ],
                    },
                )
                figure.update_traces(
                    texttemplate="<b>%{text:.2f}</b>",
                    textposition="outside",
                    textfont=dict(size=13),
                    cliponaxis=False,
                )
                figure = style_chart(
                    figure,
                    y_title="Rata-rata Jumlah Kata",
                    y_range=[
                        0,
                        max(
                            45.0,
                            float(chart_frame["Rata-rata Kata"].max()) * 1.18,
                        ),
                    ],
                    height=480,
                )
                st.plotly_chart(
                    figure,
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )

        st.caption(
            "Content Kompas dianalisis pada EDA, tetapi tidak digunakan dalam "
            "skenario eksperimen final karena jauh lebih panjang daripada "
            "Title dan Description."
        )

        st.subheader("Cakupan Panjang Sequence yang Digunakan")

        sequence_values = [
            (
                "Kompas K1",
                20,
                get_sequence_coverage(
                    sequence_coverage,
                    "kompas",
                    "title",
                    20,
                    100.0,
                ),
            ),
            (
                "Kompas K2/K3",
                60,
                get_sequence_coverage(
                    sequence_coverage,
                    "kompas",
                    "title_description",
                    60,
                    100.0,
                ),
            ),
            (
                "AG News Train A2",
                60,
                get_sequence_coverage(
                    sequence_coverage,
                    "ag_news_train",
                    "title_description",
                    60,
                    97.5413,
                ),
            ),
            (
                "AG News Test A2",
                60,
                get_sequence_coverage(
                    sequence_coverage,
                    "ag_news_test",
                    "title_description",
                    60,
                    97.75,
                ),
            ),
        ]

        sequence_columns = st.columns(4)
        for column, (label, length, coverage) in zip(
            sequence_columns,
            sequence_values,
        ):
            column.metric(
                label,
                f"{coverage:.2f}%",
                help=f"Sequence length yang digunakan: {length}",
            )

        st.info(
            "Sequence length 20 digunakan untuk Title. Sequence length 60 "
            "digunakan untuk Title + Description dan K3. Sekitar 2,46% data "
            "AG News Train mengalami truncation pada length 60."
        )

        show_technical_table(
            "Lihat statistik deskriptif lengkap",
            text_statistics,
        )
        show_technical_table(
            "Lihat pengujian sequence length lengkap",
            sequence_coverage,
        )

    elif eda_section == "Frekuensi & Word Cloud":
        word_frequency = load_table("word_frequency_overall.csv")
        wordcloud_summary = load_table("wordcloud_summary.csv")

        frequency_1, frequency_2 = st.columns([1, 1])

        with frequency_1:
            selected_dataset = st.selectbox(
                "Dataset frekuensi kata",
                ["kompas", "ag_news_train"],
                format_func=lambda value: (
                    "Kompas" if value == "kompas" else "AG News Train"
                ),
                key="word_frequency_dataset",
            )
            top_n = st.slider(
                "Jumlah token",
                min_value=10,
                max_value=30,
                value=15,
                step=5,
                key="eda_top_words",
            )

            top_words = prepare_word_frequency(
                word_frequency,
                selected_dataset,
                top_n,
            )

            if top_words.empty:
                st.info("Data frekuensi kata belum tersedia.")
            else:
                figure = px.bar(
                    top_words.sort_values("Frekuensi"),
                    x="Frekuensi",
                    y="Kata",
                    orientation="h",
                    text="Frekuensi",
                    title=(
                        f"{top_n} Token Teratas — "
                        f"{'Kompas' if selected_dataset == 'kompas' else 'AG News Train'}"
                    ),
                )
                figure.update_traces(textposition="outside")
                figure = style_chart(
                    figure,
                    y_title="Frekuensi Kemunculan",
                    height=620,
                )
                st.plotly_chart(
                    figure,
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )
                st.caption(
                    "Nilai lengkap tersedia melalui tooltip pada setiap batang."
                )

        with frequency_2:
            st.subheader("Word Cloud")

            if wordcloud_summary.empty:
                st.info("Ringkasan word cloud belum tersedia.")
            else:
                dataset_column = find_column(wordcloud_summary, ["dataset"])
                category_column = find_column(wordcloud_summary, ["category"])

                if dataset_column is None or category_column is None:
                    st.warning("Kolom dataset atau category tidak ditemukan.")
                else:
                    wc_dataset = st.selectbox(
                        "Dataset word cloud",
                        ["kompas", "ag_news_train"],
                        format_func=lambda value: (
                            "Kompas" if value == "kompas" else "AG News Train"
                        ),
                        key="wordcloud_dataset",
                    )

                    available_rows = wordcloud_summary[
                        wordcloud_summary[dataset_column]
                        .fillna("")
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .eq(wc_dataset)
                    ].copy()

                    categories = (
                        available_rows[category_column]
                        .fillna("")
                        .astype(str)
                        .tolist()
                    )

                    if not categories:
                        st.info("Kategori word cloud tidak tersedia.")
                    else:
                        selected_category = st.selectbox(
                            "Kategori",
                            categories,
                            format_func=lambda value: (
                                "Keseluruhan"
                                if str(value).lower() == "all"
                                else display_label(value)
                            ),
                            key="wordcloud_category",
                        )

                        selected_rows = available_rows[
                            available_rows[category_column]
                            .astype(str)
                            .eq(str(selected_category))
                        ]
                        summary_row = (
                            selected_rows.iloc[0]
                            if not selected_rows.empty
                            else None
                        )
                        image_path = resolve_wordcloud_path(
                            wc_dataset,
                            str(selected_category),
                            summary_row,
                        )

                        if image_path is None:
                            st.info("Gambar word cloud tidak ditemukan.")
                        else:
                            st.image(
                                str(image_path),
                                use_container_width=True,
                            )

                        if summary_row is not None:
                            wc_1, wc_2, wc_3 = st.columns(3)
                            wc_1.metric(
                                "Jumlah Dokumen",
                                number_id(
                                    first_value(
                                        summary_row,
                                        ["jumlah_dokumen"],
                                        0,
                                    )
                                ),
                            )
                            wc_2.metric(
                                "Token Unik",
                                number_id(
                                    first_value(
                                        summary_row,
                                        ["jumlah_token_unik"],
                                        0,
                                    )
                                ),
                            )
                            wc_3.metric(
                                "Token Teratas",
                                str(
                                    first_value(
                                        summary_row,
                                        ["token_teratas"],
                                        "-",
                                    )
                                ),
                            )

        st.caption(
            "Frekuensi kata dan word cloud menggambarkan karakteristik korpus, "
            "bukan feature importance model."
        )

    elif eda_section == "Analisis Temporal":
        monthly = load_table("kompas_monthly_distribution.csv")
        monthly_category = load_table(
            "kompas_monthly_category_distribution.csv"
        )
        daily = load_table("kompas_daily_distribution.csv")
        hourly = load_table("kompas_hourly_distribution.csv")
        weekday = load_table("kompas_weekday_distribution.csv")
        temporal_summary = load_table("kompas_temporal_summary.csv")

        if not temporal_summary.empty:
            row = temporal_summary.iloc[0]
            temp_1, temp_2, temp_3, temp_4, temp_5 = st.columns(5)
            temp_1.metric(
                "Periode Awal",
                str(first_value(row, ["tanggal_awal"], "-"))[:10],
            )
            temp_2.metric(
                "Periode Akhir",
                str(first_value(row, ["tanggal_akhir"], "-"))[:10],
            )
            temp_3.metric(
                "Tanggal Teraktif",
                str(first_value(row, ["tanggal_teraktif"], "-")),
            )
            temp_4.metric(
                "Jam Teraktif",
                str(first_value(row, ["rentang_jam_teraktif"], "-")),
            )
            temp_5.metric(
                "Hari Teraktif",
                str(
                    first_value(
                        row,
                        ["hari_teraktif", "weekday_teraktif"],
                        "Rabu",
                    )
                ),
            )

        temporal_view = st.radio(
            "Visualisasi temporal",
            ["Bulanan", "Per Kategori", "Harian", "Jam", "Hari"],
            horizontal=True,
            key="temporal_view",
        )

        if temporal_view == "Bulanan" and not monthly.empty:
            chart_frame = monthly.copy()
            chart_frame["Bulan"] = (
                chart_frame["month_period"]
                .fillna("")
                .astype(str)
                .str[:7]
            )
            figure = px.bar(
                chart_frame,
                x="Bulan",
                y="jumlah_berita",
                text="jumlah_berita",
                title="Distribusi Artikel Kompas per Bulan",
                hover_data={
                    "persentase_dataset": ":.2f",
                    "bulan_parsial": True,
                },
            )
            figure.update_traces(
                texttemplate="<b>%{text:,.0f}</b>",
                textposition="outside",
                textfont=dict(size=13),
                cliponaxis=False,
            )
            figure = style_chart(
                figure,
                y_title="Jumlah Artikel",
                y_range=[
                    0,
                    float(chart_frame["jumlah_berita"].max()) * 1.15,
                ],
                height=480,
            )
            st.plotly_chart(
                figure,
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )

        elif temporal_view == "Per Kategori" and not monthly_category.empty:
            chart_frame = monthly_category.copy()
            chart_frame["Bulan"] = (
                chart_frame["month_period"]
                .fillna("")
                .astype(str)
                .str[:7]
            )
            chart_frame["Kategori"] = chart_frame["category"].map(
                display_label
            )
            figure = px.line(
                chart_frame,
                x="Bulan",
                y="jumlah_berita",
                color="Kategori",
                markers=True,
                title="Distribusi Kategori Kompas per Bulan",
            )
            figure = style_chart(
                figure,
                y_title="Jumlah Artikel",
                height=500,
            )
            st.plotly_chart(
                figure,
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )

        elif temporal_view == "Harian" and not daily.empty:
            chart_frame = daily.copy()
            date_column = find_column(
                chart_frame,
                ["publication_date"],
            )
            if date_column:
                chart_frame[date_column] = pd.to_datetime(
                    chart_frame[date_column],
                    errors="coerce",
                )
            figure = px.line(
                chart_frame,
                x=date_column or "publication_date",
                y="jumlah_berita",
                title="Distribusi Harian Artikel Kompas",
            )
            figure = style_chart(
                figure,
                y_title="Jumlah Artikel",
                height=480,
            )
            st.plotly_chart(
                figure,
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )

        elif temporal_view == "Jam" and not hourly.empty:
            figure = px.bar(
                hourly,
                x="hour",
                y="jumlah_berita",
                text="jumlah_berita",
                title="Distribusi Jam Publikasi Kompas",
            )
            figure.update_traces(
                texttemplate="<b>%{text:,.0f}</b>",
                textposition="outside",
                textfont=dict(size=12),
                cliponaxis=False,
            )
            figure.update_xaxes(dtick=1)
            figure = style_chart(
                figure,
                y_title="Jumlah Artikel",
                y_range=[
                    0,
                    float(hourly["jumlah_berita"].max()) * 1.18,
                ],
                height=480,
            )
            st.plotly_chart(
                figure,
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )

        elif temporal_view == "Hari" and not weekday.empty:
            figure = px.bar(
                weekday,
                x="day_name",
                y="jumlah_berita",
                text="jumlah_berita",
                title="Distribusi Artikel Kompas Berdasarkan Hari",
                category_orders={
                    "day_name": [
                        "Senin",
                        "Selasa",
                        "Rabu",
                        "Kamis",
                        "Jumat",
                        "Sabtu",
                        "Minggu",
                    ]
                },
            )
            figure.update_traces(
                texttemplate="<b>%{text:,.0f}</b>",
                textposition="outside",
                textfont=dict(size=13),
                cliponaxis=False,
            )
            figure = style_chart(
                figure,
                y_title="Jumlah Artikel",
                y_range=[
                    0,
                    float(weekday["jumlah_berita"].max()) * 1.15,
                ],
                height=480,
            )
            st.plotly_chart(
                figure,
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )

        else:
            st.info("Data visualisasi temporal yang dipilih belum tersedia.")

        st.warning(
            "Distribusi temporal menggambarkan karakteristik dataset hasil "
            "crawling. Januari dan Mei merupakan bulan parsial, sehingga "
            "grafik tidak digeneralisasi sebagai tren produksi seluruh berita "
            "Kompas."
        )


# =============================================================================
# PAGE: RESULTS AND METRICS
# =============================================================================

elif selected_page == "📊 Hasil & Metrik":
    st.header("Hasil dan Metrik 10 Eksperimen")

    if evaluation_data.empty:
        st.error("Tabel evaluasi final belum ditemukan.")
    else:
        chart_metric = st.radio(
            "Metrik grafik",
            ["Accuracy", "Macro F1"],
            horizontal=True,
            key="result_chart_metric",
        )

        metric_column = (
            "Accuracy (%)" if chart_metric == "Accuracy" else "Macro F1 (%)"
        )

        if metric_column not in chart_data.columns:
            st.info(f"Data {chart_metric} belum tersedia.")
        else:
            figure = px.bar(
                chart_data,
                x="Skenario",
                y=metric_column,
                color="Model",
                facet_col="Dataset",
                barmode="group",
                text=metric_column,
                title=f"Perbandingan {chart_metric} Seluruh Eksperimen",
                category_orders={
                    "Dataset": ["Kompas", "AG News"],
                    "Skenario": ["K1", "K2", "K3", "A1", "A2"],
                    "Model": ["CNN", "Attention-BiLSTM"],
                },
            )
            add_percentage_labels(figure)
            figure = style_chart(
                figure,
                y_title=f"{chart_metric} (%)",
                y_range=percentage_axis_range(
                    chart_data[metric_column],
                    padding_bottom=1.5,
                    padding_top=1.2,
                ),
                height=520,
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

            best_row = chart_data.loc[chart_data[metric_column].idxmax()]
            show_result_explanation(
                main_result=(
                    f"Nilai {chart_metric} tertinggi adalah "
                    f"**{best_row[metric_column]:.2f}%** pada "
                    f"**{best_row['Eksperimen']}**."
                ),
                how_to_read=(
                    "Grafik dipisahkan berdasarkan dataset. Setiap skenario "
                    "menampilkan hasil CNN dan Attention-BiLSTM."
                ),
                interpretation=(
                    "Perbandingan menunjukkan pengaruh skenario input dan "
                    "arsitektur model pada data test yang sama."
                ),
                conclusion=(
                    "Pemilihan model deployment didasarkan pada hasil dataset "
                    "utama Kompas, yaitu CNN K2."
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

        scenarios = list(EXPERIMENTS[selected_dataset][selected_model].keys())
        with selector_3:
            selected_scenario = st.selectbox(
                "Skenario",
                scenarios,
                index=1 if len(scenarios) > 1 else 0,
                format_func=lambda code: f"{code} — {SCENARIO_NAMES[code]}",
                key="result_scenario",
            )

        experiment_name = EXPERIMENTS[selected_dataset][selected_model][
            selected_scenario
        ]
        metrics = get_metric_bundle(evaluation_data, experiment_name)

        result_metrics = st.columns(6)
        result_metrics[0].metric("Accuracy", percentage(metrics["accuracy"]))
        result_metrics[1].metric(
            "Precision Macro",
            percentage(metrics["precision"]),
        )
        result_metrics[2].metric("Recall Macro", percentage(metrics["recall"]))
        result_metrics[3].metric("Macro F1", percentage(metrics["f1"]))
        result_metrics[4].metric(
            "Log Loss",
            (
                f"{metrics['log_loss']:.6f}"
                if metrics["log_loss"] is not None
                else "-"
            ),
        )
        result_metrics[5].metric(
            "Inferensi",
            (
                f"{metrics['inference_time_ms']:.2f} ms"
                if metrics["inference_time_ms"] is not None
                else "-"
            ),
        )

        detail_view = st.radio(
            "Visualisasi eksperimen",
            ["Confusion Matrix", "Training Curve", "Kesalahan Klasifikasi"],
            horizontal=True,
            key="result_detail_view",
        )

        if detail_view == "Confusion Matrix":
            matrix_type = st.radio(
                "Tampilan confusion matrix",
                ["Jumlah", "Normalized"],
                horizontal=True,
                key="matrix_type",
            )
            prediction_data = load_prediction_result(experiment_name)

            if prediction_data.empty:
                st.info(
                    f"File prediksi final `{experiment_name}_predictions.csv` "
                    "belum tersedia."
                )
            else:
                matrix_figure, matrix_summary = create_confusion_matrix_chart(
                    prediction_data,
                    selected_dataset,
                    normalized=matrix_type == "Normalized",
                )

                if matrix_figure is None or matrix_summary is None:
                    st.warning("Kolom label aktual atau prediksi tidak ditemukan.")
                else:
                    st.plotly_chart(
                        matrix_figure,
                        use_container_width=True,
                        config=PLOTLY_CONFIG,
                    )
                    cm_1, cm_2, cm_3, cm_4 = st.columns(4)
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
                    cm_4.metric(
                        "Accuracy",
                        percentage(matrix_summary["accuracy"]),
                    )

                    if matrix_summary["largest_error"] > 0:
                        st.caption(
                            "Kesalahan terbesar: kelas aktual "
                            f"**{display_label(matrix_summary['actual_error_class'])}** "
                            "diprediksi sebagai "
                            f"**{display_label(matrix_summary['predicted_error_class'])}** "
                            f"sebanyak **{matrix_summary['largest_error']} data**."
                        )

        elif detail_view == "Training Curve":
            history = load_training_history(experiment_name)
            accuracy_figure, loss_figure, training_summary = (
                create_training_curve_charts(history)
            )

            if accuracy_figure is None and loss_figure is None:
                st.info("Training history CSV belum tersedia.")
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

                if training_summary:
                    train_1, train_2, train_3 = st.columns(3)
                    train_1.metric("Jumlah Epoch", training_summary["epochs"])
                    train_2.metric(
                        "Epoch Validation Loss Terbaik",
                        training_summary["best_epoch"],
                    )
                    train_3.metric(
                        "Validation Loss Terbaik",
                        (
                            f"{training_summary['best_val_loss']:.6f}"
                            if pd.notna(training_summary["best_val_loss"])
                            else "-"
                        ),
                    )

        else:
            misclassification = load_misclassification_analysis()
            if misclassification.empty:
                st.info("Analisis kesalahan klasifikasi belum tersedia.")
            else:
                experiment_column = find_column(
                    misclassification,
                    ["experiment_name", "experiment"],
                )
                actual_column = find_column(
                    misclassification,
                    ["actual_class", "actual_label", "kelas_aktual"],
                )
                predicted_column = find_column(
                    misclassification,
                    [
                        "predicted_class",
                        "predicted_label",
                        "kelas_prediksi",
                    ],
                )
                count_column = find_column(
                    misclassification,
                    [
                        "misclassification_count",
                        "count",
                        "jumlah_salah",
                    ],
                )

                selected_misclassification = misclassification.copy()

                if experiment_column is not None:
                    filtered = misclassification[
                        misclassification[experiment_column]
                        .fillna("")
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .eq(experiment_name.lower())
                    ]
                    if not filtered.empty:
                        selected_misclassification = filtered.copy()

                if (
                    actual_column is None
                    or predicted_column is None
                    or count_column is None
                ):
                    st.info(
                        "Kolom ringkasan kesalahan klasifikasi belum lengkap."
                    )
                else:
                    error_frame = pd.DataFrame(
                        {
                            "Kelas Aktual": (
                                selected_misclassification[actual_column]
                                .map(display_label)
                            ),
                            "Diprediksi Sebagai": (
                                selected_misclassification[predicted_column]
                                .map(display_label)
                            ),
                            "Jumlah Salah": pd.to_numeric(
                                selected_misclassification[count_column],
                                errors="coerce",
                            ),
                        }
                    ).dropna(subset=["Jumlah Salah"])

                    error_frame["Pola Kesalahan"] = (
                        error_frame["Kelas Aktual"]
                        + " → "
                        + error_frame["Diprediksi Sebagai"]
                    )
                    error_frame = (
                        error_frame
                        .sort_values("Jumlah Salah", ascending=False)
                        .head(10)
                    )

                    figure = px.bar(
                        error_frame.sort_values("Jumlah Salah"),
                        x="Jumlah Salah",
                        y="Pola Kesalahan",
                        orientation="h",
                        text="Jumlah Salah",
                        title="Kesalahan Klasifikasi yang Paling Sering Terjadi",
                    )
                    figure.update_traces(
                        texttemplate="<b>%{text:.0f}</b>",
                        textposition="outside",
                        textfont=dict(size=13),
                        cliponaxis=False,
                    )
                    figure = style_chart(
                        figure,
                        y_title="Jumlah Salah",
                        y_range=None,
                        height=480,
                    )
                    st.plotly_chart(
                        figure,
                        use_container_width=True,
                        config=PLOTLY_CONFIG,
                    )

                    if not error_frame.empty:
                        top_error = error_frame.iloc[0]
                        st.caption(
                            "Kesalahan terbesar adalah kelas aktual "
                            f"**{top_error['Kelas Aktual']}** diprediksi sebagai "
                            f"**{top_error['Diprediksi Sebagai']}** sebanyak "
                            f"**{int(top_error['Jumlah Salah'])} artikel**."
                        )

        with st.expander("Ringkasan seluruh eksperimen"):
            preferred_columns = [
                "Eksperimen",
                "Dataset",
                "Model",
                "Skenario",
                "Representasi Teks",
                "Accuracy (%)",
                "Macro F1 (%)",
            ]
            display_dataframe(chart_data, preferred_columns)


# =============================================================================
# PAGE: YAKE COMPARISON
# =============================================================================

elif selected_page == "🔑 Perbandingan YAKE":
    st.header("Perbandingan Tanpa YAKE dan Dengan YAKE")
    st.info(
        "K2 menggunakan Title + Description. K3 menggunakan Title + "
        "Description + Keyword YAKE. Keduanya memakai split, vocabulary, "
        "panjang sequence 60, dan konfigurasi training yang setara untuk "
        "masing-masing arsitektur."
    )

    comparison_rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []

    for model_name, k2_name, k3_name in [
        ("CNN", "cnn_k2", "cnn_k3"),
        (
            "Attention-BiLSTM",
            "attention_bilstm_k2",
            "attention_bilstm_k3",
        ),
    ]:
        k2_metrics = get_metric_bundle(evaluation_data, k2_name)
        k3_metrics = get_metric_bundle(evaluation_data, k3_name)

        accuracy_change = (
            (k3_metrics["accuracy"] - k2_metrics["accuracy"]) * 100
            if (
                k2_metrics["accuracy"] is not None
                and k3_metrics["accuracy"] is not None
            )
            else np.nan
        )
        f1_change = (
            (k3_metrics["f1"] - k2_metrics["f1"]) * 100
            if k2_metrics["f1"] is not None and k3_metrics["f1"] is not None
            else np.nan
        )

        for scenario, condition, metrics in [
            ("K2", "Tanpa YAKE", k2_metrics),
            ("K3", "Dengan YAKE", k3_metrics),
        ]:
            comparison_rows.append(
                {
                    "Model": model_name,
                    "Skenario": scenario,
                    "Kondisi": condition,
                    "Accuracy (%)": (
                        metrics["accuracy"] * 100
                        if metrics["accuracy"] is not None
                        else np.nan
                    ),
                    "Macro F1 (%)": (
                        metrics["f1"] * 100
                        if metrics["f1"] is not None
                        else np.nan
                    ),
                }
            )

        detailed_rows.append(
            {
                "Model": model_name,
                "Accuracy K2": percentage(k2_metrics["accuracy"]),
                "Accuracy K3": percentage(k3_metrics["accuracy"]),
                "Perubahan Accuracy": (
                    f"{accuracy_change:+.2f} pp"
                    if pd.notna(accuracy_change)
                    else "-"
                ),
                "Macro F1 K2": percentage(k2_metrics["f1"]),
                "Macro F1 K3": percentage(k3_metrics["f1"]),
                "Perubahan Macro F1": (
                    f"{f1_change:+.2f} pp"
                    if pd.notna(f1_change)
                    else "-"
                ),
                "Prediksi Salah K2": (
                    round(KOMPAS_TEST_SIZE * (1 - k2_metrics["accuracy"]))
                    if k2_metrics["accuracy"] is not None
                    else "-"
                ),
                "Prediksi Salah K3": (
                    round(KOMPAS_TEST_SIZE * (1 - k3_metrics["accuracy"]))
                    if k3_metrics["accuracy"] is not None
                    else "-"
                ),
            }
        )

    comparison_frame = pd.DataFrame(comparison_rows)
    detailed_frame = pd.DataFrame(detailed_rows)

    metric_choice = st.radio(
        "Metrik YAKE",
        ["Accuracy (%)", "Macro F1 (%)"],
        horizontal=True,
        key="yake_metric",
    )

    if comparison_frame[metric_choice].notna().any():
        figure = px.bar(
            comparison_frame,
            x="Model",
            y=metric_choice,
            color="Kondisi",
            barmode="group",
            text=metric_choice,
            title=f"Perbandingan {metric_choice} K2 dan K3",
            category_orders={
                "Model": ["CNN", "Attention-BiLSTM"],
                "Kondisi": ["Tanpa YAKE", "Dengan YAKE"],
            },
        )
        add_percentage_labels(figure)
        figure = style_chart(
            figure,
            y_title=metric_choice,
            y_range=percentage_axis_range(
                comparison_frame[metric_choice],
                padding_bottom=1.0,
                padding_top=1.2,
            ),
            height=490,
        )
        st.plotly_chart(
            figure,
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    comparison_columns = st.columns(2)

    for column, model_name in zip(
        comparison_columns,
        ["CNN", "Attention-BiLSTM"],
    ):
        model_row = detailed_frame[
            detailed_frame["Model"].eq(model_name)
        ].iloc[0]

        with column:
            with st.container(border=True):
                st.markdown(f"### {model_name}")
                model_metrics = st.columns(3)
                model_metrics[0].metric(
                    "Accuracy K2",
                    model_row["Accuracy K2"],
                )
                model_metrics[1].metric(
                    "Accuracy K3",
                    model_row["Accuracy K3"],
                )
                model_metrics[2].metric(
                    "Perubahan",
                    model_row["Perubahan Accuracy"],
                )
                st.caption(
                    "Prediksi salah K2: "
                    f"{model_row['Prediksi Salah K2']} artikel · "
                    "Prediksi salah K3: "
                    f"{model_row['Prediksi Salah K3']} artikel"
                )

    cnn_change = detailed_frame.loc[
        detailed_frame["Model"].eq("CNN"),
        "Perubahan Accuracy",
    ].iloc[0]
    attention_change = detailed_frame.loc[
        detailed_frame["Model"].eq("Attention-BiLSTM"),
        "Perubahan Accuracy",
    ].iloc[0]

    show_result_explanation(
        main_result=(
            f"Perubahan Accuracy CNN adalah **{cnn_change}**, sedangkan "
            f"Attention-BiLSTM adalah **{attention_change}**."
        ),
        how_to_read=(
            "Batang K2 menunjukkan hasil tanpa YAKE dan batang K3 menunjukkan "
            "hasil setelah lima keyword YAKE ditambahkan."
        ),
        interpretation=(
            "Keyword diekstraksi dari Title dan Description. Informasi keyword "
            "dapat berulang dengan teks sumber sehingga belum tentu menambah "
            "informasi diskriminatif."
        ),
        conclusion=(
            "Pada dataset dan konfigurasi penelitian ini, YAKE belum "
            "mengungguli representasi Title + Description. Kesimpulan ini "
            "tidak digeneralisasi bahwa YAKE selalu tidak efektif."
        ),
    )

    st.subheader("Konfigurasi YAKE")
    config_1, config_2, config_3, config_4, config_5 = st.columns(5)
    config_1.metric("Sumber Teks", "Title + Description")
    config_2.metric("Bahasa", "Indonesia")
    config_3.metric("Maximum N-gram", "3")
    config_4.metric("Top Keyword", "5")
    config_5.metric("Deduplication", "0,9")
    st.caption("Content tidak digunakan dalam ekstraksi keyword YAKE.")

    with st.expander("Tabel analisis YAKE dari pipeline"):
        yake_contribution = load_yake_contribution()
        display_dataframe(yake_contribution)


# =============================================================================
# PAGE: PREDICTION
# =============================================================================

elif selected_page == "📰 Prediksi Berita":
    st.header("Prediksi Kategori Berita")
    st.info(
        "Masukkan title dan description berita untuk memprediksi kategorinya."
    )

    st.caption(
        "Validasi bahasa hanya merupakan pengaman pada dashboard. "
        "Validasi tersebut tidak mengubah preprocessing, model, vocabulary, "
        "atau hasil eksperimen penelitian."
    )

    with st.form("prediction_form"):
        title_input = st.text_input(
            "Title",
            placeholder=(
                "Contoh: Rupiah Menguat terhadap Dolar AS"
            ),
        )
        description_input = st.text_area(
            "Description",
            placeholder=(
                "Contoh: Nilai tukar rupiah menguat setelah Bank Indonesia "
                "mengumumkan kebijakan baru..."
            ),
            height=140,
        )
        submitted = st.form_submit_button(
            "Prediksi Berita",
            use_container_width=True,
        )

    if submitted:
        st.session_state.pop(
            "prediction_result",
            None,
        )
        st.session_state.pop(
            "prediction_language",
            None,
        )

        if not title_input.strip():
            st.warning(
                "Title tidak boleh kosong."
            )

        elif not description_input.strip():
            st.warning(
                "Description tidak boleh kosong."
            )

        else:
            detected_language = detect_input_language(
                title=title_input,
                description=description_input,
            )

            st.session_state[
                "prediction_language"
            ] = detected_language

            if detected_language != "id":
                language_name = {
                    "en": "Inggris",
                    "unknown": "Tidak dapat dipastikan",
                }.get(
                    detected_language,
                    detected_language.upper(),
                )

                st.error(
                    "Prediksi tidak dijalankan karena input bukan berita "
                    "berbahasa Indonesia."
                )
                st.info(
                    f"Bahasa terdeteksi: **{language_name}**. "
                    "Model deployment dilatih menggunakan berita Kompas "
                    "berbahasa Indonesia."
                )

            else:
                try:
                    with st.spinner(
                        "Menjalankan inference dua model..."
                    ):
                        result = predict_news(
                            title=title_input,
                            description=description_input,
                        )

                    result[
                        "detected_language"
                    ] = detected_language

                    st.session_state[
                        "prediction_result"
                    ] = result

                except Exception as error:
                    st.session_state.pop(
                        "prediction_result",
                        None,
                    )
                    st.error(
                        f"Prediksi gagal: {error}"
                    )

    prediction_result = st.session_state.get(
        "prediction_result"
    )

    if prediction_result:
        recommendation = prediction_result[
            "recommended_prediction"
        ]
        recommended_label = display_label(
            recommendation["predicted_label"]
        )

        result_1, result_2, result_3 = st.columns(3)
        result_1.metric(
            "Prediksi Utama",
            recommended_label,
        )
        result_2.metric(
            "Model Deployment",
            recommendation["source_model"],
        )
        result_3.metric(
            "Confidence",
            percentage(
                recommendation["confidence"]
            ),
        )

        st.success(
            "Prediksi utama mengikuti CNN K2 karena model tersebut dipilih "
            "sebagai model deployment berdasarkan hasil test Kompas."
        )

        model_columns = st.columns(2)
        model_results = [
            (
                "CNN K2",
                prediction_result["cnn"],
            ),
            (
                "Attention-BiLSTM K2",
                prediction_result["attention_bilstm"],
            ),
        ]

        for column, (
            model_name,
            model_result,
        ) in zip(
            model_columns,
            model_results,
        ):
            with column:
                with st.container(border=True):
                    st.subheader(model_name)
                    st.metric(
                        "Kategori",
                        display_label(
                            model_result[
                                "predicted_label"
                            ]
                        ),
                    )
                    st.metric(
                        "Confidence",
                        percentage(
                            model_result[
                                "confidence"
                            ]
                        ),
                    )
                    st.caption(
                        "Waktu inferensi: "
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
                                float(value) * 100
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
                        title=(
                            f"Probabilitas {model_name}"
                        ),
                    )
                    add_percentage_labels(
                        figure
                    )
                    figure = style_chart(
                        figure,
                        y_title="Probabilitas (%)",
                        y_range=[0, 105],
                        height=420,
                    )
                    st.plotly_chart(
                        figure,
                        use_container_width=True,
                        config=PLOTLY_CONFIG,
                    )

        if prediction_result[
            "model_agreement"
        ]:
            st.success(
                "CNN dan Attention-BiLSTM memberikan kategori yang sama."
            )
        else:
            st.warning(
                "CNN dan Attention-BiLSTM memberikan kategori berbeda. "
                "Hasil utama tetap mengikuti CNN K2 sebagai model deployment. "
                "Perbedaan ini menunjukkan bahwa input perlu ditinjau dengan "
                "lebih hati-hati."
            )

        st.warning(
            "Model hanya mengenal empat kategori, yaitu Bola, Global, Money, "
            "dan Tekno. Artikel di luar empat kategori tersebut tetap akan "
            "dipetakan ke salah satu kategori karena model bersifat closed-set."
        )

        st.caption(
            "Confidence merupakan probabilitas relatif model dan bukan "
            "jaminan bahwa prediksi selalu benar."
        )


# =============================================================================
# PAGE: EXPLAINABLE AI
# =============================================================================

elif selected_page == "🔍 Explainable AI":
    st.header("Explainable AI Menggunakan SHAP")
    st.info(
        "SHAP diterapkan pada CNN K2. Visualisasi pada halaman ini merupakan "
        "hasil **precomputed** dari sampel data test, bukan penjelasan real-time "
        "untuk berita yang dimasukkan pada halaman prediksi."
    )

    shap_view = st.radio(
        "Bagian SHAP",
        ["Global SHAP", "SHAP per Kelas", "Local SHAP", "Waterfall Resmi"],
        horizontal=True,
        key="shap_navigation",
    )

    if shap_view == "Global SHAP":
        global_shap = load_global_shap()

        if global_shap.empty:
            st.info("Data Global SHAP belum tersedia.")
        else:
            token_column = find_column(global_shap, ["token", "word"])
            importance_column = find_column(
                global_shap,
                ["total_abs_shap", "mean_abs_shap", "importance"],
            )
            occurrence_column = find_column(
                global_shap,
                ["occurrence_count", "document_frequency", "frequency"],
            )

            top_n = st.slider(
                "Jumlah token",
                min_value=10,
                max_value=50,
                value=20,
                step=5,
                key="global_shap_top_n",
            )

            if token_column is None or importance_column is None:
                st.warning("Kolom token atau importance belum tersedia.")
            else:
                selected = global_shap[
                    ~global_shap[token_column]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .isin(SPECIAL_TOKENS)
                ].copy()
                selected[importance_column] = pd.to_numeric(
                    selected[importance_column],
                    errors="coerce",
                )
                selected = (
                    selected
                    .dropna(subset=[importance_column])
                    .sort_values(importance_column, ascending=False)
                    .head(top_n)
                )

                figure = px.bar(
                    selected.sort_values(importance_column),
                    x=importance_column,
                    y=token_column,
                    orientation="h",
                    text=importance_column,
                    title=f"{top_n} Token dengan Pengaruh Global Terbesar",
                    hover_data=(
                        {occurrence_column: True}
                        if occurrence_column is not None
                        else None
                    ),
                )
                figure.update_traces(
                    texttemplate="%{text:.4f}",
                    textposition="outside",
                )
                figure = style_chart(
                    figure,
                    y_title="Total |SHAP|",
                    height=620,
                )
                st.plotly_chart(
                    figure,
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )

                if not selected.empty:
                    top_row = selected.iloc[0]
                    show_result_explanation(
                        main_result=(
                            f"Token dengan pengaruh global terbesar adalah "
                            f"**{top_row[token_column]}** dengan nilai "
                            f"**{float(top_row[importance_column]):.4f}**."
                        ),
                        how_to_read=(
                            "Semakin panjang batang, semakin besar total "
                            "pengaruh absolut token pada sampel global SHAP."
                        ),
                        interpretation=(
                            "Global SHAP menunjukkan besar pengaruh, tetapi "
                            "tidak langsung menunjukkan arah ke satu kelas."
                        ),
                        conclusion=(
                            "CNN K2 menggunakan kombinasi banyak token dalam "
                            "membentuk keputusan."
                        ),
                    )

    elif shap_view == "SHAP per Kelas":
        by_class = load_global_shap_by_class()

        if by_class.empty:
            st.info("Data SHAP per kelas belum tersedia.")
        else:
            class_column = find_column(
                by_class,
                ["class_name", "class", "label", "category"],
            )
            token_column = find_column(by_class, ["token", "word"])
            importance_column = find_column(
                by_class,
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
                    by_class[class_column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                selected_class = st.selectbox(
                    "Pilih kelas",
                    classes,
                    format_func=display_label,
                    key="shap_class",
                )

                selected = by_class[
                    by_class[class_column].astype(str).eq(selected_class)
                ].copy()
                selected = selected[
                    ~selected[token_column]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .isin(SPECIAL_TOKENS)
                ]
                selected[importance_column] = pd.to_numeric(
                    selected[importance_column],
                    errors="coerce",
                )
                selected = (
                    selected
                    .dropna(subset=[importance_column])
                    .sort_values(importance_column, ascending=False)
                    .head(20)
                )

                figure = px.bar(
                    selected.sort_values(importance_column),
                    x=importance_column,
                    y=token_column,
                    orientation="h",
                    text=importance_column,
                    title=(
                        "Token Penting untuk Kelas "
                        f"{display_label(selected_class)}"
                    ),
                )
                figure.update_traces(
                    texttemplate="%{text:.4f}",
                    textposition="outside",
                )
                figure = style_chart(
                    figure,
                    y_title="Total |SHAP|",
                    height=620,
                )
                st.plotly_chart(
                    figure,
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )

                if not selected.empty:
                    top_row = selected.iloc[0]
                    st.caption(
                        f"Token teratas untuk kelas "
                        f"**{display_label(selected_class)}** adalah "
                        f"**{top_row[token_column]}** dengan nilai "
                        f"**{float(top_row[importance_column]):.4f}**."
                    )

    elif shap_view == "Local SHAP":
        local_summary = load_local_shap_summary()
        local_contributions = load_local_token_contributions()

        if local_summary.empty or local_contributions.empty:
            st.info("Data Local SHAP belum tersedia.")
        else:
            sample_column = find_column(
                local_summary,
                ["document_id", "sample_id", "id"],
            )

            if sample_column is None:
                st.warning("Kolom ID artikel tidak ditemukan.")
            else:
                samples = local_summary[sample_column].astype(str).tolist()
                selected_sample = st.selectbox(
                    "Pilih artikel",
                    samples,
                    key="local_shap_sample",
                )

                selected_summary = local_summary[
                    local_summary[sample_column]
                    .astype(str)
                    .eq(selected_sample)
                ]
                summary_row = selected_summary.iloc[0]

                actual_label = display_label(
                    first_value(summary_row, ["actual_label"], "-")
                )
                predicted_label = display_label(
                    first_value(summary_row, ["predicted_label"], "-")
                )
                confidence = normalize_metric(
                    first_value(summary_row, ["prediction_confidence"], None)
                )
                is_correct = safe_bool(
                    first_value(summary_row, ["is_correct"], False)
                )

                local_1, local_2, local_3, local_4 = st.columns(4)
                local_1.metric("ID Artikel", selected_sample)
                local_2.metric("Label Aktual", actual_label)
                local_3.metric("Label Prediksi", predicted_label)
                local_4.metric("Confidence", percentage(confidence))

                if is_correct:
                    st.success("Prediksi artikel sesuai dengan label aktual.")
                else:
                    st.warning("Prediksi artikel berbeda dari label aktual.")

                token_data, token_column, contribution_column = (
                    prepare_local_token_data(
                        local_contributions,
                        selected_sample,
                        top_n=20,
                    )
                )

                if (
                    token_data.empty
                    or token_column is None
                    or contribution_column is None
                ):
                    st.info("Kontribusi token artikel belum tersedia.")
                else:
                    figure = px.bar(
                        token_data.sort_values(contribution_column),
                        x=contribution_column,
                        y=token_column,
                        orientation="h",
                        text=contribution_column,
                        color=contribution_column,
                        color_continuous_scale="RdBu",
                        color_continuous_midpoint=0,
                        title=f"Kontribusi Token — {selected_sample}",
                    )
                    figure.update_traces(
                        texttemplate="%{text:+.4f}",
                        textposition="outside",
                    )
                    figure = style_chart(
                        figure,
                        y_title="Signed SHAP",
                        height=620,
                    )
                    st.plotly_chart(
                        figure,
                        use_container_width=True,
                        config=PLOTLY_CONFIG,
                    )

                    strongest = token_data.iloc[
                        token_data["absolute_contribution"].argmax()
                    ]
                    st.caption(
                        f"Token dengan kontribusi absolut terbesar adalah "
                        f"**{strongest[token_column]}** dengan nilai "
                        f"**{float(strongest[contribution_column]):+.4f}**."
                    )

    else:
        waterfall_summary = load_waterfall_summary()

        if waterfall_summary.empty:
            st.info("Ringkasan waterfall resmi belum tersedia.")
        else:
            sample_column = find_column(
                waterfall_summary,
                ["document_id", "sample_id", "id"],
            )

            if sample_column is None:
                st.warning(
                    "Kolom ID artikel pada ringkasan waterfall tidak ada."
                )
            else:
                samples = (
                    waterfall_summary[sample_column]
                    .astype(str)
                    .tolist()
                )
                selected_sample = st.selectbox(
                    "Pilih artikel waterfall",
                    samples,
                    key="waterfall_sample",
                )
                selected_rows = waterfall_summary[
                    waterfall_summary[sample_column]
                    .astype(str)
                    .eq(selected_sample)
                ]
                row = selected_rows.iloc[0]

                actual_label = display_label(
                    first_value(row, ["actual_label"], "-")
                )
                predicted_label = display_label(
                    first_value(row, ["predicted_label"], "-")
                )
                confidence = normalize_metric(
                    first_value(
                        row,
                        ["prediction_confidence"],
                        None,
                    )
                )
                is_correct = safe_bool(
                    first_value(row, ["is_correct"], False)
                )

                baseline = first_value(
                    row,
                    [
                        "expected_value",
                        "base_value",
                        "baseline_value",
                        "baseline",
                    ],
                )
                model_output = first_value(
                    row,
                    [
                        "model_output",
                        "predicted_probability",
                        "prediction_probability",
                        "final_output",
                    ],
                    confidence,
                )
                reconstruction_error = first_value(
                    row,
                    [
                        "reconstruction_error",
                        "absolute_reconstruction_error",
                        "max_reconstruction_error",
                    ],
                    0.0,
                )

                waterfall_1, waterfall_2, waterfall_3, waterfall_4 = (
                    st.columns(4)
                )
                waterfall_1.metric("ID Artikel", selected_sample)
                waterfall_2.metric("Label Aktual", actual_label)
                waterfall_3.metric("Label Prediksi", predicted_label)
                waterfall_4.metric("Confidence", percentage(confidence))

                if is_correct:
                    st.success(
                        "Prediksi artikel sesuai dengan label aktual."
                    )
                else:
                    st.warning(
                        "Prediksi artikel berbeda dari label aktual."
                    )

                st.subheader("Ringkasan Kontribusi Token")

                local_contributions = load_local_token_contributions()
                (
                    waterfall_tokens,
                    waterfall_token_column,
                    waterfall_contribution_column,
                ) = prepare_local_token_data(
                    local_contributions,
                    selected_sample,
                    top_n=20,
                )

                supporting_text = "-"
                opposing_text = "-"

                if (
                    not waterfall_tokens.empty
                    and waterfall_token_column is not None
                    and waterfall_contribution_column is not None
                ):
                    positive_tokens = (
                        waterfall_tokens[
                            waterfall_tokens[
                                waterfall_contribution_column
                            ].gt(0)
                        ]
                        .sort_values(
                            waterfall_contribution_column,
                            ascending=False,
                        )
                        .head(5)
                    )
                    negative_tokens = (
                        waterfall_tokens[
                            waterfall_tokens[
                                waterfall_contribution_column
                            ].lt(0)
                        ]
                        .sort_values(
                            waterfall_contribution_column,
                            ascending=True,
                        )
                        .head(5)
                    )

                    if not positive_tokens.empty:
                        supporting_text = ", ".join(
                            positive_tokens[
                                waterfall_token_column
                            ].astype(str)
                        )

                    if not negative_tokens.empty:
                        opposing_text = ", ".join(
                            negative_tokens[
                                waterfall_token_column
                            ].astype(str)
                        )

                support_column, oppose_column = st.columns(2)

                with support_column:
                    st.success(
                        "**Token yang meningkatkan dukungan**\n\n"
                        f"{supporting_text}"
                    )

                with oppose_column:
                    st.info(
                        "**Token yang mengurangi dukungan**\n\n"
                        f"{opposing_text}"
                    )

                value_columns = st.columns(3)

                try:
                    baseline_text = f"{float(baseline):.4f}"
                except (TypeError, ValueError):
                    baseline_text = "-"

                try:
                    model_output_numeric = float(model_output)
                    model_output_text = (
                        f"{model_output_numeric * 100:.2f}%"
                        if model_output_numeric <= 1
                        else f"{model_output_numeric:.2f}%"
                    )
                except (TypeError, ValueError):
                    model_output_text = percentage(confidence)

                try:
                    error_text = f"{float(reconstruction_error):.2e}"
                except (TypeError, ValueError):
                    error_text = "-"

                value_columns[0].metric(
                    "Nilai Awal Model",
                    baseline_text,
                )
                value_columns[1].metric(
                    "Probabilitas Akhir",
                    model_output_text,
                )
                value_columns[2].metric(
                    "Error Rekonstruksi",
                    error_text,
                )

                st.markdown(
                    f"""
                    **Kesimpulan sederhana:** model memprediksi artikel sebagai
                    **{predicted_label}**. Token pada kotak hijau meningkatkan
                    dukungan terhadap kelas tersebut, sedangkan token pada kotak
                    biru mengurangi dukungannya. Setelah seluruh kontribusi
                    digabungkan, probabilitas akhir model adalah
                    **{model_output_text}**.
                    """
                )

                with st.expander(
                    "Lihat grafik Waterfall SHAP resmi",
                    expanded=False,
                ):
                    image_path = resolve_waterfall_image(row)

                    if image_path is None:
                        st.info(
                            "Gambar waterfall resmi tidak ditemukan."
                        )
                    else:
                        st.image(
                            str(image_path),
                            width=1100,
                        )

                    st.markdown(
                        """
                        **Cara membaca grafik:**

                        - **Merah** meningkatkan probabilitas kelas prediksi.
                        - **Biru** mengurangi probabilitas kelas prediksi.
                        - **Garis putus-putus** menunjukkan nilai awal model.
                        - **Garis akhir** menunjukkan probabilitas setelah
                          kontribusi seluruh token dijumlahkan.
                        - Tanda **×2** berarti token muncul dua kali dan
                          kontribusinya digabungkan.
                        """
                    )

                if (
                    actual_label == "Bola"
                    or predicted_label == "Bola"
                ):
                    st.warning(
                        "Baseline all-padding cenderung kuat ke kelas Bola. "
                        "Interpretasi sampel Bola harus membaca baseline, "
                        "kontribusi token, probabilitas akhir, dan error "
                        "rekonstruksi secara bersamaan."
                    )


# =============================================================================
# FOOTER
# =============================================================================

footer_text = (
    "Dashboard implementasi penelitian CNN dan Attention-BiLSTM untuk "
    "klasifikasi berita, perbandingan skenario representasi teks, pengujian "
    "YAKE, deployment prediksi, dan Explainable AI SHAP."
)

st.divider()
st.caption(footer_text)

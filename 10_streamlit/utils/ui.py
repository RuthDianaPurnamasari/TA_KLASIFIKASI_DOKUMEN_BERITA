# =============================================================================
# STREAMLIT UI UTILITIES
# =============================================================================
# File:
# 10_streamlit/utils/ui.py
#
# Fungsi:
# 1. Menyediakan komponen tampilan yang digunakan berulang.
# 2. Menyamakan format judul, metrik, tabel, dan pesan dashboard.
# 3. Menampilkan performa model dari deployment_config.json.
# 4. Menampilkan hasil prediksi CNN dan Attention-BiLSTM.
# 5. Memvalidasi data probabilitas sebelum ditampilkan.
# 6. Mengurangi duplikasi kode pada setiap halaman Streamlit.
# =============================================================================

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import streamlit as st


# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================

CURRENT_FILE = Path(
    __file__
).resolve()

UTILS_DIR = CURRENT_FILE.parent

STREAMLIT_DIR = CURRENT_FILE.parents[1]

PROJECT_ROOT = CURRENT_FILE.parents[2]

CONFIG_PATH = (
    STREAMLIT_DIR
    / "config.py"
)

DEPLOYMENT_CONFIG_PATH = (
    PROJECT_ROOT
    / "8_save_models"
    / "deployment"
    / "deployment_config.json"
)


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_DISPLAY_LABELS = {
    "bola": "Bola",
    "global": "Global",
    "money": "Money",
    "tekno": "Tekno",
}

EXPECTED_LABELS = {
    "bola",
    "global",
    "money",
    "tekno",
}

MODEL_DISPLAY_ORDER = [
    "CNN K2",
    "Attention-BiLSTM K2",
]


# =============================================================================
# DASHBOARD CONFIGURATION
# =============================================================================

def load_dashboard_config_module() -> Any | None:
    """
    Memuat config.py menggunakan path absolut.

    Cara ini mencegah bentrok dengan package lain
    yang juga memiliki nama config.
    """

    if not CONFIG_PATH.exists():
        return None

    module_name = (
        "ta_streamlit_ui_config"
    )

    if module_name in sys.modules:
        return sys.modules[
            module_name
        ]

    specification = (
        importlib.util.spec_from_file_location(
            module_name,
            CONFIG_PATH,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        return None

    module = (
        importlib.util.module_from_spec(
            specification
        )
    )

    sys.modules[
        module_name
    ] = module

    specification.loader.exec_module(
        module
    )

    return module


DASHBOARD_CONFIG = (
    load_dashboard_config_module()
)

if (
    DASHBOARD_CONFIG is not None
    and hasattr(
        DASHBOARD_CONFIG,
        "DISPLAY_LABELS",
    )
):
    DISPLAY_LABELS = {
        str(label)
        .strip()
        .lower():
            str(display_label)

        for label, display_label
        in DASHBOARD_CONFIG.DISPLAY_LABELS.items()
    }

else:
    DISPLAY_LABELS = dict(
        DEFAULT_DISPLAY_LABELS
    )


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def is_finite_number(
    value: Any,
) -> bool:
    """
    Memeriksa apakah nilai dapat dikonversi
    menjadi angka finite.
    """

    try:
        numeric_value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return False

    return math.isfinite(
        numeric_value
    )


def safe_float(
    value: Any,
    default: float | None = None,
) -> float | None:
    """
    Mengubah nilai menjadi float dengan aman.
    """

    if not is_finite_number(
        value
    ):
        return default

    return float(
        value
    )


def validate_dataframe(
    dataframe: pd.DataFrame | None,
) -> bool:
    """
    Memeriksa apakah DataFrame tersedia
    dan tidak kosong.
    """

    return (
        dataframe is not None
        and isinstance(
            dataframe,
            pd.DataFrame,
        )
        and not dataframe.empty
    )


# =============================================================================
# JSON LOADER
# =============================================================================

@st.cache_data(
    show_spinner=False,
)
def _read_json_cached(
    file_path_string: str,
    modified_time_ns: int,
    file_size_bytes: int,
) -> dict[str, Any]:
    """
    Membaca JSON menggunakan cache.

    Waktu modifikasi dan ukuran file menjadi
    bagian cache key.
    """

    del modified_time_ns
    del file_size_bytes

    file_path = Path(
        file_path_string
    )

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(
            file
        )

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            "Isi JSON harus berupa dictionary."
        )

    return data


def read_json_safe(
    file_path: Path,
) -> dict[str, Any]:
    """
    Membaca JSON dengan aman.

    Mengembalikan dictionary kosong apabila
    file belum tersedia atau gagal dibaca.
    """

    path = Path(
        file_path
    )

    if (
        not path.exists()
        or not path.is_file()
        or path.stat().st_size <= 0
    ):
        return {}

    try:
        stat = path.stat()

        return dict(
            _read_json_cached(
                file_path_string=str(
                    path.resolve()
                ),
                modified_time_ns=int(
                    stat.st_mtime_ns
                ),
                file_size_bytes=int(
                    stat.st_size
                ),
            )
        )

    except Exception:
        return {}


# =============================================================================
# PAGE HEADER
# =============================================================================

def page_header(
    title: str,
    subtitle: str | None = None,
    icon: str | None = None,
) -> None:
    """
    Menampilkan judul utama halaman.

    Parameters
    ----------
    title:
        Judul halaman.

    subtitle:
        Penjelasan singkat di bawah judul.

    icon:
        Ikon opsional yang ditempatkan sebelum judul.
    """

    display_title = (
        f"{icon} {title}"
        if icon
        else title
    )

    st.title(
        display_title
    )

    if subtitle:
        st.caption(
            subtitle
        )

    st.divider()


# =============================================================================
# SECTION HEADER
# =============================================================================

def section_header(
    title: str,
    description: str | None = None,
    icon: str | None = None,
) -> None:
    """
    Menampilkan judul bagian dalam halaman.
    """

    display_title = (
        f"{icon} {title}"
        if icon
        else title
    )

    st.subheader(
        display_title
    )

    if description:
        st.caption(
            description
        )


# =============================================================================
# EMPTY DATA MESSAGE
# =============================================================================

def show_empty_data_message(
    data_name: str,
    path_hint: str | None = None,
    show_path: bool = False,
) -> None:
    """
    Menampilkan informasi ketika data belum tersedia.

    Path hanya ditampilkan jika show_path=True,
    sehingga path lokal tidak terekspos pada
    dashboard publik.
    """

    message = (
        f"Data **{data_name}** belum tersedia. "
        "Pastikan tahap pemrosesan terkait "
        "sudah berhasil dijalankan."
    )

    if (
        show_path
        and path_hint
    ):
        message += (
            "\n\nPath yang diperiksa:\n"
            f"`{path_hint}`"
        )

    st.info(
        message
    )


# =============================================================================
# MISSING FIGURE MESSAGE
# =============================================================================

def show_missing_figure_message(
    figure_name: str,
    figure_path: Path | None = None,
    show_path: bool = False,
) -> None:
    """
    Menampilkan informasi jika grafik belum tersedia.
    """

    message = (
        f"Grafik **{figure_name}** belum tersedia. "
        "Pastikan tahap evaluasi atau visualisasi "
        "terkait sudah berhasil dijalankan."
    )

    if (
        show_path
        and figure_path is not None
    ):
        message += (
            "\n\nPath yang diperiksa:\n"
            f"`{figure_path}`"
        )

    st.info(
        message
    )


# =============================================================================
# DISPLAY LABEL
# =============================================================================

def format_label(
    label: Any,
) -> str:
    """
    Mengubah label internal menjadi label tampilan.

    Contoh:
    bola   -> Bola
    global -> Global
    money  -> Money
    tekno  -> Tekno
    """

    if label is None:
        return "-"

    normalized_label = (
        str(label)
        .strip()
        .lower()
    )

    if not normalized_label:
        return "-"

    return DISPLAY_LABELS.get(
        normalized_label,
        normalized_label
        .replace(
            "_",
            " ",
        )
        .title(),
    )


# =============================================================================
# PERCENTAGE FORMAT
# =============================================================================

def format_percentage(
    value: float | int | None,
    decimal_places: int = 2,
) -> str:
    """
    Mengubah angka menjadi format persentase.

    Nilai 0 sampai 1 dianggap sebagai proporsi:
    0.969 -> 96.90%

    Nilai lebih dari 1 dianggap sudah dalam persen:
    96.9 -> 96.90%
    """

    numeric_value = safe_float(
        value
    )

    if numeric_value is None:
        return "-"

    if numeric_value < 0:
        return "-"

    if numeric_value <= 1:
        numeric_value *= 100

    if numeric_value > 100:
        return "-"

    return (
        f"{numeric_value:.{decimal_places}f}%"
    )


# =============================================================================
# NUMERIC FORMAT
# =============================================================================

def format_number(
    value: Any,
    decimal_places: int = 2,
    default: str = "-",
) -> str:
    """
    Memformat nilai numerik.
    """

    numeric_value = safe_float(
        value
    )

    if numeric_value is None:
        return default

    return (
        f"{numeric_value:,.{decimal_places}f}"
    )


def format_integer(
    value: Any,
    default: str = "-",
) -> str:
    """
    Memformat bilangan bulat.
    """

    numeric_value = safe_float(
        value
    )

    if numeric_value is None:
        return default

    return (
        f"{int(round(numeric_value)):,}"
    )


def format_milliseconds(
    value: Any,
    decimal_places: int = 2,
) -> str:
    """
    Memformat waktu dalam milidetik.
    """

    numeric_value = safe_float(
        value
    )

    if (
        numeric_value is None
        or numeric_value < 0
    ):
        return "-"

    return (
        f"{numeric_value:.{decimal_places}f} ms"
    )


# =============================================================================
# METRIC CARD
# =============================================================================

def metric_card(
    label: str,
    value: Any,
    delta: str | None = None,
    help_text: str | None = None,
    delta_color: str = "normal",
) -> None:
    """
    Menampilkan satu kartu metrik Streamlit.
    """

    st.metric(
        label=label,
        value=value,
        delta=delta,
        help=help_text,
        delta_color=delta_color,
    )


# =============================================================================
# MODEL PERFORMANCE LOADER
# =============================================================================

def load_model_performance_from_deployment(
) -> dict[str, dict[str, Any]]:
    """
    Membaca performa CNN K2 dan Attention-BiLSTM K2
    dari deployment_config.json.

    File deployment menjadi sumber utama sehingga
    nilai dashboard tetap sama dengan hasil penelitian.
    """

    deployment_config = read_json_safe(
        DEPLOYMENT_CONFIG_PATH
    )

    if not deployment_config:
        return {}

    models = deployment_config.get(
        "models",
        {},
    )

    if not isinstance(
        models,
        dict,
    ):
        return {}

    scenario_name = str(
        deployment_config.get(
            "scenario_name",
            "Title + Description",
        )
    )

    model_mapping = {
        "cnn": "CNN K2",
        "attention_bilstm":
            "Attention-BiLSTM K2",
    }

    performance: dict[
        str,
        dict[str, Any]
    ] = {}

    for model_key, display_key in (
        model_mapping.items()
    ):
        model_data = models.get(
            model_key,
            {},
        )

        if not isinstance(
            model_data,
            dict,
        ):
            continue

        performance[
            display_key
        ] = {
            "experiment_name":
                model_data.get(
                    "experiment_name"
                ),

            "display_name":
                model_data.get(
                    "display_name",
                    display_key,
                ),

            "accuracy":
                safe_float(
                    model_data.get(
                        "accuracy"
                    )
                ),

            "precision_macro":
                safe_float(
                    model_data.get(
                        "precision_macro"
                    )
                ),

            "recall_macro":
                safe_float(
                    model_data.get(
                        "recall_macro"
                    )
                ),

            "f1_macro":
                safe_float(
                    model_data.get(
                        "f1_macro"
                    )
                ),

            "log_loss":
                safe_float(
                    model_data.get(
                        "log_loss"
                    )
                ),

            "average_inference_ms_per_sample":
                safe_float(
                    model_data.get(
                        "average_inference_ms_per_sample"
                    )
                ),

            "available":
                bool(
                    model_data.get(
                        "available",
                        True,
                    )
                ),

            "scenario_name":
                scenario_name,
        }

    return performance


# =============================================================================
# MODEL PERFORMANCE CARD
# =============================================================================

def _show_single_model_performance_card(
    model_name: str,
    model_data: Mapping[str, Any],
    is_primary: bool = False,
) -> None:
    """
    Menampilkan performa satu model.
    """

    title = (
        f"{model_name} — Model Utama"
        if is_primary
        else model_name
    )

    with st.container(
        border=True
    ):
        st.subheader(
            title
        )

        metric_columns = st.columns(
            2
        )

        with metric_columns[0]:
            metric_card(
                label="Accuracy",
                value=format_percentage(
                    model_data.get(
                        "accuracy"
                    )
                ),
                help_text=(
                    "Accuracy model pada test set Kompas."
                ),
            )

        with metric_columns[1]:
            metric_card(
                label="Macro F1",
                value=format_percentage(
                    model_data.get(
                        "f1_macro"
                    )
                ),
                help_text=(
                    "Rata-rata F1-score seluruh kelas."
                ),
            )

        detail_columns = st.columns(
            2
        )

        with detail_columns[0]:
            precision_value = (
                model_data.get(
                    "precision_macro"
                )
            )

            if precision_value is not None:
                metric_card(
                    label="Macro Precision",
                    value=format_percentage(
                        precision_value
                    ),
                )

        with detail_columns[1]:
            recall_value = (
                model_data.get(
                    "recall_macro"
                )
            )

            if recall_value is not None:
                metric_card(
                    label="Macro Recall",
                    value=format_percentage(
                        recall_value
                    ),
                )

        scenario_name = str(
            model_data.get(
                "scenario_name",
                "Title + Description",
            )
        )

        st.caption(
            f"Representasi: {scenario_name}."
        )


def show_model_performance_cards(
    model_performance: Mapping[
        str,
        Mapping[str, Any]
    ] | None = None,
) -> None:
    """
    Menampilkan performa CNN K2 dan Attention-BiLSTM K2.

    Jika model_performance tidak diberikan, data dibaca
    dari deployment_config.json.
    """

    if model_performance is None:
        model_performance = (
            load_model_performance_from_deployment()
        )

    if not model_performance:
        show_empty_data_message(
            "performa model deployment"
        )

        return

    cnn_data = model_performance.get(
        "CNN K2"
    )

    attention_data = model_performance.get(
        "Attention-BiLSTM K2"
    )

    if (
        cnn_data is None
        or attention_data is None
    ):
        st.warning(
            "Data performa CNN K2 atau "
            "Attention-BiLSTM K2 belum lengkap."
        )

        return

    column_cnn, column_attention = (
        st.columns(
            2
        )
    )

    with column_cnn:
        _show_single_model_performance_card(
            model_name="CNN K2",
            model_data=cnn_data,
            is_primary=True,
        )

    with column_attention:
        _show_single_model_performance_card(
            model_name="Attention-BiLSTM K2",
            model_data=attention_data,
            is_primary=False,
        )


# =============================================================================
# PROBABILITY VALIDATION
# =============================================================================

def validate_probability_mapping(
    probabilities: Mapping[str, Any],
    tolerance: float = 1e-3,
) -> dict[str, float]:
    """
    Memvalidasi dictionary probabilitas.

    Validasi:
    - dictionary tidak kosong;
    - label dikenali;
    - nilai finite;
    - nilai berada pada rentang 0 sampai 1;
    - jumlah probabilitas mendekati 1.
    """

    if not isinstance(
        probabilities,
        Mapping,
    ):
        raise TypeError(
            "Probabilitas harus berupa mapping."
        )

    if not probabilities:
        raise ValueError(
            "Data probabilitas kosong."
        )

    normalized: dict[
        str,
        float
    ] = {}

    for label, probability in (
        probabilities.items()
    ):
        normalized_label = (
            str(label)
            .strip()
            .lower()
        )

        numeric_probability = safe_float(
            probability
        )

        if numeric_probability is None:
            raise ValueError(
                "Probabilitas untuk label "
                f"{normalized_label} tidak valid."
            )

        if (
            numeric_probability
            < -tolerance
            or numeric_probability
            > 1.0 + tolerance
        ):
            raise ValueError(
                "Probabilitas untuk label "
                f"{normalized_label} berada "
                "di luar rentang 0 sampai 1."
            )

        numeric_probability = min(
            1.0,
            max(
                0.0,
                numeric_probability,
            ),
        )

        normalized[
            normalized_label
        ] = numeric_probability

    probability_sum = sum(
        normalized.values()
    )

    if not math.isclose(
        probability_sum,
        1.0,
        abs_tol=tolerance,
    ):
        raise ValueError(
            "Jumlah probabilitas tidak mendekati 1.\n"
            f"Jumlah: {probability_sum:.8f}"
        )

    return normalized


# =============================================================================
# PROBABILITY DATAFRAME
# =============================================================================

def probability_dataframe(
    probabilities: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Mengubah dictionary probabilitas menjadi DataFrame.
    """

    validated_probabilities = (
        validate_probability_mapping(
            probabilities
        )
    )

    rows = [
        {
            "Kategori":
                format_label(
                    label
                ),

            "Probabilitas":
                probability,

            "Probabilitas (%)":
                probability
                * 100,
        }

        for label, probability
        in validated_probabilities.items()
    ]

    dataframe = pd.DataFrame(
        rows
    )

    return (
        dataframe
        .sort_values(
            by="Probabilitas",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# PREDICTION RESULT VALIDATION
# =============================================================================

def validate_prediction_result(
    result: Mapping[str, Any],
) -> tuple[bool, str | None]:
    """
    Memvalidasi struktur hasil prediksi.
    """

    if not isinstance(
        result,
        Mapping,
    ):
        return (
            False,
            "Hasil prediksi bukan dictionary.",
        )

    required_keys = {
        "predicted_label",
        "confidence",
        "inference_time_ms",
        "probabilities",
    }

    missing_keys = (
        required_keys
        - set(
            result.keys()
        )
    )

    if missing_keys:
        return (
            False,
            "Data hasil prediksi tidak lengkap. "
            f"Key hilang: {sorted(missing_keys)}",
        )

    confidence = safe_float(
        result.get(
            "confidence"
        )
    )

    if (
        confidence is None
        or confidence < 0
        or confidence > 1
    ):
        return (
            False,
            "Nilai confidence tidak valid.",
        )

    inference_time = safe_float(
        result.get(
            "inference_time_ms"
        )
    )

    if (
        inference_time is None
        or inference_time < 0
    ):
        return (
            False,
            "Waktu inference tidak valid.",
        )

    try:
        validate_probability_mapping(
            result[
                "probabilities"
            ]
        )

    except Exception as error:
        return (
            False,
            str(error),
        )

    return (
        True,
        None,
    )


# =============================================================================
# PREDICTION RESULT CARD
# =============================================================================

def show_prediction_result_card(
    model_name: str,
    result: Mapping[str, Any],
) -> None:
    """
    Menampilkan hasil prediksi satu model.
    """

    valid, error_message = (
        validate_prediction_result(
            result
        )
    )

    if not valid:
        st.error(
            f"Hasil prediksi **{model_name}** "
            f"tidak dapat ditampilkan.\n\n"
            f"{error_message}"
        )

        return

    predicted_label = format_label(
        result[
            "predicted_label"
        ]
    )

    confidence = format_percentage(
        result[
            "confidence"
        ]
    )

    inference_time = (
        format_milliseconds(
            result[
                "inference_time_ms"
            ]
        )
    )

    probability_data = (
        probability_dataframe(
            result[
                "probabilities"
            ]
        )
    )

    with st.container(
        border=True
    ):
        st.subheader(
            model_name
        )

        metric_columns = st.columns(
            3
        )

        with metric_columns[0]:
            metric_card(
                label="Prediksi",
                value=predicted_label,
            )

        with metric_columns[1]:
            metric_card(
                label="Confidence",
                value=confidence,
            )

        with metric_columns[2]:
            metric_card(
                label="Waktu Inference",
                value=inference_time,
            )

        st.markdown(
            "#### Distribusi probabilitas"
        )

        chart_data = (
            probability_data
            .set_index(
                "Kategori"
            )[
                "Probabilitas (%)"
            ]
        )

        st.bar_chart(
            chart_data
        )

        display_dataframe = (
            probability_data[
                [
                    "Kategori",
                    "Probabilitas (%)",
                ]
            ]
            .copy()
        )

        st.dataframe(
            display_dataframe.style.format(
                {
                    "Probabilitas (%)":
                        "{:.2f}%"
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# MODEL AGREEMENT
# =============================================================================

def show_model_agreement(
    model_agreement: bool,
    cnn_label: Any,
    attention_label: Any,
) -> None:
    """
    Menampilkan kesepakatan dua model.
    """

    cnn_display = format_label(
        cnn_label
    )

    attention_display = format_label(
        attention_label
    )

    if model_agreement:
        st.success(
            "Kedua model memberikan kategori yang sama, "
            f"yaitu **{cnn_display}**."
        )

    else:
        st.warning(
            "Kedua model memberikan hasil yang berbeda.\n\n"
            f"- CNN K2: **{cnn_display}**\n"
            f"- Attention-BiLSTM K2: "
            f"**{attention_display}**"
        )


# =============================================================================
# RECOMMENDED PREDICTION
# =============================================================================

def show_recommended_prediction(
    recommended_prediction: Mapping[str, Any],
) -> None:
    """
    Menampilkan hasil rekomendasi utama sistem.
    """

    if not isinstance(
        recommended_prediction,
        Mapping,
    ):
        st.error(
            "Data rekomendasi sistem tidak valid."
        )

        return

    required_keys = {
        "predicted_label",
        "source_model",
        "confidence",
    }

    missing_keys = (
        required_keys
        - set(
            recommended_prediction.keys()
        )
    )

    if missing_keys:
        st.error(
            "Data rekomendasi belum lengkap. "
            f"Key hilang: {sorted(missing_keys)}"
        )

        return

    label = format_label(
        recommended_prediction[
            "predicted_label"
        ]
    )

    confidence = format_percentage(
        recommended_prediction[
            "confidence"
        ]
    )

    source_model = str(
        recommended_prediction[
            "source_model"
        ]
    )

    with st.container(
        border=True
    ):
        st.markdown(
            "### Hasil Rekomendasi Sistem"
        )

        metric_columns = st.columns(
            3
        )

        with metric_columns[0]:
            metric_card(
                label="Kategori",
                value=label,
            )

        with metric_columns[1]:
            metric_card(
                label="Confidence",
                value=confidence,
            )

        with metric_columns[2]:
            metric_card(
                label="Model Utama",
                value=source_model,
            )

        st.caption(
            "Rekomendasi utama mengikuti model dengan "
            "performa test set terbaik pada penelitian."
        )


# =============================================================================
# SEQUENCE INFORMATION
# =============================================================================

def show_sequence_information(
    input_information: Mapping[str, Any],
) -> None:
    """
    Menampilkan informasi sequence hasil vectorization.
    """

    if not isinstance(
        input_information,
        Mapping,
    ):
        return

    with st.expander(
        "Informasi pemrosesan teks",
        expanded=False,
    ):
        columns = st.columns(
            4
        )

        columns[0].metric(
            "Token Non-padding",
            format_integer(
                input_information.get(
                    "non_padding_tokens"
                )
            ),
        )

        columns[1].metric(
            "Token Padding",
            format_integer(
                input_information.get(
                    "padding_tokens"
                )
            ),
        )

        columns[2].metric(
            "Token OOV",
            format_integer(
                input_information.get(
                    "oov_tokens"
                )
            ),
        )

        columns[3].metric(
            "Terpotong",
            (
                "Ya"
                if bool(
                    input_information.get(
                        "possibly_truncated",
                        False,
                    )
                )
                else "Tidak"
            ),
        )

        clean_title = input_information.get(
            "clean_title"
        )

        clean_description = (
            input_information.get(
                "clean_description"
            )
        )

        combined_text = input_information.get(
            "combined_text"
        )

        if clean_title is not None:
            st.markdown(
                "**Title setelah preprocessing**"
            )

            st.code(
                str(
                    clean_title
                ),
                language=None,
            )

        if clean_description is not None:
            st.markdown(
                "**Description setelah preprocessing**"
            )

            st.code(
                str(
                    clean_description
                ),
                language=None,
            )

        if combined_text is not None:
            st.markdown(
                "**Teks gabungan K2**"
            )

            st.code(
                str(
                    combined_text
                ),
                language=None,
            )


# =============================================================================
# SHAP INFORMATION
# =============================================================================

def show_shap_information() -> None:
    """
    Menampilkan cara membaca hasil SHAP.
    """

    st.info(
        """
**Cara membaca SHAP**

- Fitur dalam klasifikasi teks berupa token atau kata.
- Kontribusi positif mendukung kelas yang sedang dijelaskan.
- Kontribusi negatif mengurangi dukungan terhadap kelas tersebut.
- Semakin besar nilai absolut SHAP, semakin kuat pengaruh token.
- SHAP menjelaskan keputusan model, bukan menentukan kebenaran berita.
        """
    )


# =============================================================================
# DATAFRAME PREVIEW
# =============================================================================

def show_dataframe_preview(
    dataframe: pd.DataFrame | None,
    title: str | None = None,
    max_rows: int = 20,
    column_order: list[str] | None = None,
) -> None:
    """
    Menampilkan preview DataFrame dengan jumlah baris terbatas.
    """

    if title:
        st.subheader(
            title
        )

    if not validate_dataframe(
        dataframe
    ):
        show_empty_data_message(
            title or "tabel"
        )

        return

    if max_rows <= 0:
        st.warning(
            "Jumlah maksimum baris harus lebih dari 0."
        )

        return

    assert dataframe is not None

    preview = dataframe.copy()

    if column_order:
        available_columns = [
            column
            for column
            in column_order
            if column
            in preview.columns
        ]

        if available_columns:
            preview = preview[
                available_columns
            ]

    preview = preview.head(
        max_rows
    )

    st.dataframe(
        preview,
        use_container_width=True,
        hide_index=True,
    )

    if len(dataframe) > max_rows:
        st.caption(
            f"Menampilkan {max_rows:,} dari "
            f"{len(dataframe):,} baris."
        )


# =============================================================================
# IMAGE DISPLAY
# =============================================================================

def show_image_safe(
    image_path: Path | None,
    caption: str | None = None,
    figure_name: str = "grafik",
) -> bool:
    """
    Menampilkan gambar dengan aman.
    """

    if image_path is None:
        show_missing_figure_message(
            figure_name
        )

        return False

    path = Path(
        image_path
    )

    if (
        not path.exists()
        or not path.is_file()
        or path.stat().st_size <= 0
    ):
        show_missing_figure_message(
            figure_name=figure_name,
            figure_path=path,
        )

        return False

    st.image(
        str(
            path
        ),
        caption=caption,
        use_container_width=True,
    )

    return True


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

def clear_ui_cache() -> None:
    """
    Menghapus cache yang digunakan utilitas UI.
    """

    _read_json_cached.clear()


# =============================================================================
# TERMINAL TEST
# =============================================================================

def main() -> None:
    """
    Menguji fungsi UI yang tidak membutuhkan
    runtime Streamlit aktif.
    """

    print("=" * 80)
    print(
        "STREAMLIT UI UTILITY TEST"
    )
    print("=" * 80)

    print(
        "\nFormat label:"
    )

    for label in [
        "bola",
        "global",
        "money",
        "tekno",
    ]:
        print(
            f"{label:<10} -> "
            f"{format_label(label)}"
        )

    print(
        "\nFormat persentase:"
    )

    for value in [
        0.969,
        0.968,
        96.90,
        None,
        float("nan"),
    ]:
        print(
            f"{str(value):<10} -> "
            f"{format_percentage(value)}"
        )

    print(
        "\nMembentuk probability DataFrame..."
    )

    sample_probabilities = {
        "bola": 0.01,
        "global": 0.02,
        "money": 0.95,
        "tekno": 0.02,
    }

    probability_data = (
        probability_dataframe(
            sample_probabilities
        )
    )

    print(
        probability_data.to_string(
            index=False
        )
    )

    print(
        "\nMembaca performa deployment..."
    )

    performance = (
        load_model_performance_from_deployment()
    )

    if performance:
        for model_name in MODEL_DISPLAY_ORDER:
            model_data = performance.get(
                model_name,
                {}
            )

            print(
                f"{model_name:<25}: "
                f"accuracy="
                f"{format_percentage(model_data.get('accuracy'))}, "
                f"f1_macro="
                f"{format_percentage(model_data.get('f1_macro'))}"
            )

    else:
        print(
            "Deployment config belum tersedia."
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "UI utility test selesai."
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
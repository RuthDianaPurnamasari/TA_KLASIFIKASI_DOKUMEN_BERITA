# # =============================================================================
# # STREAMLIT UI UTILITIES
# # =============================================================================
# # File:
# # 10_streamlit/utils/ui.py
# #
# # Fungsi:
# # 1. Menyediakan komponen tampilan yang dipakai berulang.
# # 2. Menyamakan format judul, metrik, tabel, dan pesan dashboard.
# # 3. Mengurangi duplikasi kode pada setiap halaman Streamlit.
# # =============================================================================

# from __future__ import annotations

# import sys
# from pathlib import Path
# from typing import Any

# import pandas as pd
# import streamlit as st


# # =============================================================================
# # DIRECTORY CONFIGURATION
# # =============================================================================

# CURRENT_FILE = Path(
#     __file__
# ).resolve()

# STREAMLIT_DIR = (
#     CURRENT_FILE
#     .parents[1]
# )


# # Menambahkan folder 10_streamlit ke sys.path
# # agar config.py khusus dashboard dapat di-import.
# if str(STREAMLIT_DIR) not in sys.path:
#     sys.path.insert(
#         0,
#         str(STREAMLIT_DIR),
#     )


# # =============================================================================
# # IMPORT DASHBOARD CONFIGURATION
# # =============================================================================

# from config import (
#     DISPLAY_LABELS,
#     MODEL_PERFORMANCE,
# )


# # =============================================================================
# # PAGE HEADER
# # =============================================================================

# def page_header(
#     title: str,
#     subtitle: str | None = None,
# ) -> None:
#     """
#     Menampilkan judul utama halaman.

#     Parameters
#     ----------
#     title:
#         Judul halaman.

#     subtitle:
#         Penjelasan singkat di bawah judul.
#     """

#     st.title(
#         title
#     )

#     if subtitle:
#         st.caption(
#             subtitle
#         )

#     st.divider()


# # =============================================================================
# # SECTION HEADER
# # =============================================================================

# def section_header(
#     title: str,
#     description: str | None = None,
# ) -> None:
#     """
#     Menampilkan judul untuk bagian tertentu
#     dalam satu halaman.
#     """

#     st.subheader(
#         title
#     )

#     if description:
#         st.caption(
#             description
#         )


# # =============================================================================
# # EMPTY DATA MESSAGE
# # =============================================================================

# def show_empty_data_message(
#     data_name: str,
#     path_hint: str | None = None,
# ) -> None:
#     """
#     Menampilkan pesan ketika file atau data
#     belum ditemukan.

#     Dashboard tidak langsung berhenti,
#     tetapi memberikan informasi kepada pengguna.
#     """

#     message = (
#         f"Data **{data_name}** belum ditemukan."
#     )

#     if path_hint:
#         message += (
#             "\n\nPeriksa lokasi file berikut:\n"
#             f"`{path_hint}`"
#         )

#     st.info(
#         message
#     )


# # =============================================================================
# # FILE NOT FOUND MESSAGE
# # =============================================================================

# def show_missing_figure_message(
#     figure_name: str,
#     figure_path: Path,
# ) -> None:
#     """
#     Menampilkan pesan jika grafik belum ditemukan.
#     """

#     st.warning(
#         f"Grafik **{figure_name}** belum ditemukan.\n\n"
#         f"Path yang diperiksa:\n"
#         f"`{figure_path}`"
#     )


# # =============================================================================
# # DISPLAY LABEL
# # =============================================================================

# def format_label(
#     label: str,
# ) -> str:
#     """
#     Mengubah label internal model menjadi label
#     yang lebih rapi untuk dashboard.

#     Contoh:
#     bola   -> Bola
#     global -> Global
#     money  -> Money
#     tekno  -> Tekno
#     """

#     normalized_label = (
#         str(label)
#         .strip()
#         .lower()
#     )

#     return DISPLAY_LABELS.get(
#         normalized_label,
#         normalized_label.title(),
#     )


# # =============================================================================
# # PERCENTAGE FORMAT
# # =============================================================================

# def format_percentage(
#     value: float | int | None,
#     decimal_places: int = 2,
# ) -> str:
#     """
#     Mengubah angka desimal menjadi format persentase.

#     Contoh:
#     0.958 -> 95.80%
#     """

#     if value is None:
#         return "-"

#     try:
#         numeric_value = float(value)

#     except (TypeError, ValueError):
#         return "-"

#     return f"{numeric_value * 100:.{decimal_places}f}%"


# # =============================================================================
# # METRIC CARD
# # =============================================================================

# def metric_card(
#     label: str,
#     value: Any,
#     delta: str | None = None,
#     help_text: str | None = None,
# ) -> None:
#     """
#     Menampilkan satu kartu metrik Streamlit.
#     """

#     st.metric(
#         label=label,
#         value=value,
#         delta=delta,
#         help=help_text,
#     )


# # =============================================================================
# # MODEL PERFORMANCE CARDS
# # =============================================================================

# def show_model_performance_cards() -> None:
#     """
#     Menampilkan ringkasan performa CNN K2
#     dan Attention-BiLSTM K2.
#     """

#     cnn_data = MODEL_PERFORMANCE[
#         "CNN K2"
#     ]

#     attention_data = MODEL_PERFORMANCE[
#         "Attention-BiLSTM K2"
#     ]

#     column_1, column_2 = st.columns(
#         2
#     )

#     with column_1:
#         st.markdown(
#             "### CNN K2"
#         )

#         metric_card(
#             label="Accuracy",
#             value=format_percentage(
#                 cnn_data[
#                     "accuracy"
#                 ]
#             ),
#             help_text=(
#                 "Accuracy CNN pada test set Kompas."
#             ),
#         )

#         metric_card(
#             label="Macro F1",
#             value=format_percentage(
#                 cnn_data[
#                     "f1_macro"
#                 ]
#             ),
#             help_text=(
#                 "Rata-rata F1-score seluruh kelas."
#             ),
#         )

#         st.caption(
#             cnn_data[
#                 "scenario_name"
#             ]
#         )

#     with column_2:
#         st.markdown(
#             "### Attention-BiLSTM K2"
#         )

#         metric_card(
#             label="Accuracy",
#             value=format_percentage(
#                 attention_data[
#                     "accuracy"
#                 ]
#             ),
#             help_text=(
#                 "Accuracy Attention-BiLSTM "
#                 "pada test set Kompas."
#             ),
#         )

#         metric_card(
#             label="Macro F1",
#             value=format_percentage(
#                 attention_data[
#                     "f1_macro"
#                 ]
#             ),
#             help_text=(
#                 "Rata-rata F1-score seluruh kelas."
#             ),
#         )

#         st.caption(
#             attention_data[
#                 "scenario_name"
#             ]
#         )


# # =============================================================================
# # PROBABILITY DATAFRAME
# # =============================================================================

# def probability_dataframe(
#     probabilities: dict[str, float],
# ) -> pd.DataFrame:
#     """
#     Mengubah dictionary probabilitas menjadi DataFrame.

#     Contoh input:
#     {
#         "bola": 0.05,
#         "global": 0.10,
#         "money": 0.80,
#         "tekno": 0.05
#     }

#     Output:
#     DataFrame kategori dan probabilitas.
#     """

#     probability_rows = []

#     for label, probability in probabilities.items():

#         probability_rows.append(
#             {
#                 "Kategori":
#                     format_label(
#                         label
#                     ),

#                 "Probabilitas":
#                     float(
#                         probability
#                     ),

#                 "Probabilitas (%)":
#                     float(
#                         probability
#                     )
#                     * 100,
#             }
#         )

#     dataframe = pd.DataFrame(
#         probability_rows
#     )

#     dataframe = (
#         dataframe
#         .sort_values(
#             "Probabilitas",
#             ascending=False,
#         )
#         .reset_index(
#             drop=True
#         )
#     )

#     return dataframe


# # =============================================================================
# # PREDICTION RESULT CARD
# # =============================================================================

# def show_prediction_result_card(
#     model_name: str,
#     result: dict[str, Any],
# ) -> None:
#     """
#     Menampilkan hasil prediksi satu model.

#     Result harus memiliki:
#     - predicted_label
#     - confidence
#     - inference_time_ms
#     - probabilities
#     """

#     predicted_label = format_label(
#         result[
#             "predicted_label"
#         ]
#     )

#     confidence = format_percentage(
#         result[
#             "confidence"
#         ]
#     )

#     inference_time = (
#         f"{result['inference_time_ms']:.2f} ms"
#     )

#     st.markdown(
#         f"### {model_name}"
#     )

#     metric_columns = st.columns(
#         3
#     )

#     with metric_columns[0]:
#         metric_card(
#             label="Prediksi",
#             value=predicted_label,
#         )

#     with metric_columns[1]:
#         metric_card(
#             label="Confidence",
#             value=confidence,
#         )

#     with metric_columns[2]:
#         metric_card(
#             label="Waktu inferensi",
#             value=inference_time,
#         )

#     probability_data = probability_dataframe(
#         result[
#             "probabilities"
#         ]
#     )

#     st.bar_chart(
#         probability_data.set_index(
#             "Kategori"
#         )[
#             "Probabilitas (%)"
#         ]
#     )

#     display_dataframe = probability_data[
#         [
#             "Kategori",
#             "Probabilitas (%)",
#         ]
#     ].copy()

#     st.dataframe(
#         display_dataframe.style.format(
#             {
#                 "Probabilitas (%)":
#                     "{:.2f}%"
#             }
#         ),
#         use_container_width=True,
#         hide_index=True,
#     )


# # =============================================================================
# # MODEL AGREEMENT MESSAGE
# # =============================================================================

# def show_model_agreement(
#     model_agreement: bool,
#     cnn_label: str,
#     attention_label: str,
# ) -> None:
#     """
#     Menampilkan apakah dua model menghasilkan
#     kategori yang sama atau berbeda.
#     """

#     cnn_display = format_label(
#         cnn_label
#     )

#     attention_display = format_label(
#         attention_label
#     )

#     if model_agreement:
#         st.success(
#             "Kedua model memberikan hasil yang sama, "
#             f"yaitu **{cnn_display}**."
#         )

#     else:
#         st.warning(
#             "Kedua model memberikan hasil berbeda.\n\n"
#             f"- CNN K2: **{cnn_display}**\n"
#             f"- Attention-BiLSTM K2: "
#             f"**{attention_display}**"
#         )


# # =============================================================================
# # RECOMMENDED PREDICTION MESSAGE
# # =============================================================================

# def show_recommended_prediction(
#     recommended_prediction: dict[str, Any],
# ) -> None:
#     """
#     Menampilkan hasil rekomendasi utama sistem.

#     CNN K2 digunakan sebagai sumber rekomendasi
#     karena memperoleh performa terbaik.
#     """

#     label = format_label(
#         recommended_prediction[
#             "predicted_label"
#         ]
#     )

#     confidence = format_percentage(
#         recommended_prediction[
#             "confidence"
#         ]
#     )

#     source_model = (
#         recommended_prediction[
#             "source_model"
#         ]
#     )

#     st.info(
#         f"**Rekomendasi sistem:** {label}\n\n"
#         f"Model utama: **{source_model}**\n\n"
#         f"Confidence: **{confidence}**"
#     )


# # =============================================================================
# # SHAP INFORMATION
# # =============================================================================

# def show_shap_information() -> None:
#     """
#     Menampilkan penjelasan singkat cara membaca SHAP.
#     """

#     st.info(
#         """
#         **Cara membaca SHAP**

#         - Nilai SHAP positif berarti token mendukung kelas prediksi.
#         - Nilai SHAP negatif berarti token mengurangi dukungan
#           terhadap kelas prediksi.
#         - Semakin besar nilai absolut SHAP, semakin kuat
#           pengaruh token terhadap keputusan model.
#         """
#     )


# # =============================================================================
# # DATAFRAME PREVIEW
# # =============================================================================

# def show_dataframe_preview(
#     dataframe: pd.DataFrame,
#     title: str | None = None,
#     max_rows: int = 20,
# ) -> None:
#     """
#     Menampilkan DataFrame dengan jumlah baris terbatas.
#     """

#     if title:
#         st.markdown(
#             f"#### {title}"
#         )

#     if dataframe.empty:
#         show_empty_data_message(
#             title or "DataFrame"
#         )
#         return

#     preview = dataframe.head(
#         max_rows
#     )

#     st.dataframe(
#         preview,
#         use_container_width=True,
#         hide_index=True,
#     )


# # =============================================================================
# # TERMINAL TEST
# # =============================================================================

# def main() -> None:
#     """
#     Menguji fungsi-fungsi yang tidak membutuhkan
#     tampilan Streamlit aktif.
#     """

#     print("=" * 80)
#     print(
#         "STREAMLIT UI UTILITY TEST"
#     )
#     print("=" * 80)

#     print("\nFormat label:")

#     for label in [
#         "bola",
#         "global",
#         "money",
#         "tekno",
#     ]:
#         print(
#             f"{label:<10} -> "
#             f"{format_label(label)}"
#         )

#     print("\nFormat persentase:")

#     test_values = [
#         0.958,
#         0.953,
#         0.887594,
#     ]

#     for value in test_values:
#         print(
#             f"{value:<10} -> "
#             f"{format_percentage(value)}"
#         )

#     print(
#         "\nMembentuk probability DataFrame..."
#     )

#     sample_probabilities = {
#         "bola": 0.01,
#         "global": 0.02,
#         "money": 0.95,
#         "tekno": 0.02,
#     }

#     probability_data = (
#         probability_dataframe(
#             sample_probabilities
#         )
#     )

#     print(
#         probability_data.to_string(
#             index=False
#         )
#     )

#     print("\n" + "=" * 80)
#     print(
#         "UI utility test selesai."
#     )
#     print("=" * 80)


# if __name__ == "__main__":
#     main()

# =============================================================================
# STREAMLIT UI UTILITIES
# =============================================================================
# Helper tampilan yang aman dan menggunakan komponen native Streamlit.
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


CURRENT_FILE = Path(__file__).resolve()
STREAMLIT_DIR = CURRENT_FILE.parents[1]

if str(STREAMLIT_DIR) not in sys.path:
    sys.path.insert(0, str(STREAMLIT_DIR))


from config import DISPLAY_LABELS, MODEL_PERFORMANCE


def page_header(
    title: str,
    subtitle: str | None = None,
) -> None:
    """Menampilkan judul halaman."""

    st.title(title)

    if subtitle:
        st.caption(subtitle)

    st.divider()


def section_header(
    title: str,
    description: str | None = None,
) -> None:
    """Menampilkan judul bagian."""

    st.subheader(title)

    if description:
        st.caption(description)


def show_empty_data_message(
    data_name: str,
    path_hint: str | None = None,
) -> None:
    """Menampilkan pesan data belum tersedia tanpa mengekspos path."""

    st.info(
        f"Data **{data_name}** belum tersedia. "
        "Pastikan tahap pemrosesan terkait sudah berhasil dijalankan."
    )


def show_missing_figure_message(
    figure_name: str,
    figure_path: Path | None = None,
) -> None:
    """Menampilkan pesan grafik belum tersedia tanpa menampilkan path."""

    st.info(
        f"Grafik **{figure_name}** belum tersedia. "
        "Pastikan tahap evaluasi atau visualisasi terkait sudah selesai."
    )


def format_label(label: str) -> str:
    """Mengubah label internal menjadi label tampilan."""

    normalized = str(label).strip().lower()

    return DISPLAY_LABELS.get(
        normalized,
        normalized.title(),
    )


def format_percentage(
    value: float | int | None,
    decimal_places: int = 2,
) -> str:
    """
    Mengubah nilai 0-1 atau 0-100 menjadi persentase.

    0.958 -> 95.80%
    95.8  -> 95.80%
    """

    if value is None:
        return "-"

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"

    if pd.isna(numeric):
        return "-"

    if numeric <= 1:
        numeric *= 100

    return f"{numeric:.{decimal_places}f}%"


def metric_card(
    label: str,
    value: Any,
    delta: str | None = None,
    help_text: str | None = None,
) -> None:
    """Menampilkan kartu metrik."""

    st.metric(
        label=label,
        value=value,
        delta=delta,
        help=help_text,
    )


def show_model_performance_cards() -> None:
    """Menampilkan performa CNN K2 dan Attention-BiLSTM K2."""

    cnn_data = MODEL_PERFORMANCE["CNN K2"]
    attention_data = MODEL_PERFORMANCE["Attention-BiLSTM K2"]

    column_1, column_2 = st.columns(2)

    with column_1:
        with st.container(border=True):
            st.subheader("CNN K2")
            metric_card(
                "Accuracy Test Kompas",
                format_percentage(cnn_data["accuracy"]),
                help_text=(
                    "CNN menggunakan Title + Description tanpa YAKE."
                ),
            )
            metric_card(
                "Macro F1",
                format_percentage(cnn_data["f1_macro"]),
            )
            st.caption("Representasi: Title + Description.")

    with column_2:
        with st.container(border=True):
            st.subheader("Attention-BiLSTM K2")
            metric_card(
                "Accuracy Test Kompas",
                format_percentage(attention_data["accuracy"]),
                help_text=(
                    "Attention-BiLSTM menggunakan "
                    "Title + Description tanpa YAKE."
                ),
            )
            metric_card(
                "Macro F1",
                format_percentage(attention_data["f1_macro"]),
            )
            st.caption("Representasi: Title + Description.")


def probability_dataframe(
    probabilities: dict[str, float],
) -> pd.DataFrame:
    """Mengubah probabilitas model menjadi DataFrame."""

    rows = [
        {
            "Kategori": format_label(label),
            "Probabilitas": float(probability),
            "Probabilitas (%)": float(probability) * 100,
        }
        for label, probability in probabilities.items()
    ]

    return (
        pd.DataFrame(rows)
        .sort_values("Probabilitas", ascending=False)
        .reset_index(drop=True)
    )


def show_prediction_result_card(
    model_name: str,
    result: dict[str, Any],
) -> None:
    """Menampilkan hasil satu model."""

    predicted_label = format_label(result["predicted_label"])

    with st.container(border=True):
        st.subheader(model_name)

        metric_columns = st.columns(3)

        metric_columns[0].metric(
            "Prediksi",
            predicted_label,
        )
        metric_columns[1].metric(
            "Confidence",
            format_percentage(result["confidence"]),
        )
        metric_columns[2].metric(
            "Waktu Inferensi",
            f"{result['inference_time_ms']:.2f} ms",
        )

        probability_data = probability_dataframe(
            result["probabilities"]
        )

        st.bar_chart(
            probability_data.set_index("Kategori")[
                "Probabilitas (%)"
            ]
        )

        st.dataframe(
            probability_data[
                ["Kategori", "Probabilitas (%)"]
            ].style.format(
                {"Probabilitas (%)": "{:.2f}%"}
            ),
            use_container_width=True,
            hide_index=True,
        )


def show_model_agreement(
    model_agreement: bool,
    cnn_label: str,
    attention_label: str,
) -> None:
    """Menampilkan kesepakatan dua model."""

    cnn_display = format_label(cnn_label)
    attention_display = format_label(attention_label)

    if model_agreement:
        st.success(
            "Kedua model memberikan kategori yang sama, "
            f"yaitu **{cnn_display}**."
        )
    else:
        st.warning(
            "Kedua model memberikan hasil berbeda.\n\n"
            f"- CNN K2: **{cnn_display}**\n"
            f"- Attention-BiLSTM K2: **{attention_display}**"
        )


def show_recommended_prediction(
    recommended_prediction: dict[str, Any],
) -> None:
    """Menampilkan rekomendasi utama sistem."""

    label = format_label(
        recommended_prediction["predicted_label"]
    )

    st.info(
        f"**Rekomendasi sistem:** {label}\n\n"
        f"Model utama: "
        f"**{recommended_prediction['source_model']}**\n\n"
        f"Confidence: "
        f"**{format_percentage(recommended_prediction['confidence'])}**"
    )


def show_shap_information() -> None:
    """Menampilkan cara membaca SHAP."""

    st.info(
        """
        **Cara membaca SHAP**

        - Fitur pada klasifikasi teks adalah token atau kata.
        - Kontribusi positif mendukung kelas prediksi.
        - Kontribusi negatif mengurangi dukungan terhadap kelas prediksi.
        - Semakin besar nilai absolut SHAP, semakin kuat pengaruh token.
        """
    )


def show_dataframe_preview(
    dataframe: pd.DataFrame,
    title: str | None = None,
    max_rows: int = 20,
) -> None:
    """Menampilkan preview tabel."""

    if title:
        st.subheader(title)

    if dataframe.empty:
        show_empty_data_message(title or "tabel")
        return

    st.dataframe(
        dataframe.head(max_rows),
        use_container_width=True,
        hide_index=True,
    )
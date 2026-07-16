# from __future__ import annotations

# import sys
# from pathlib import Path
# from typing import Any

# import numpy as np
# import pandas as pd
# import plotly.express as px
# import streamlit as st


# # =============================================================================
# # PATH PROJECT
# # =============================================================================

# CURRENT_FILE = Path(__file__).resolve()
# STREAMLIT_DIR = CURRENT_FILE.parent
# PROJECT_ROOT = CURRENT_FILE.parents[1]

# if str(STREAMLIT_DIR) not in sys.path:
#     sys.path.insert(0, str(STREAMLIT_DIR))


# # =============================================================================
# # IMPORT KONFIGURASI
# # =============================================================================

# from config import (
#     DATASET_INFORMATION,
#     FIGURES_DIR,
#     MODEL_PERFORMANCE,
#     PRIMARY_MODEL,
#     PRIMARY_SCENARIO,
#     RESEARCH_TITLE,
#     TABLES_DIR,
# )

# from utils.data_loader import (
#     get_comparative_figure_path,
#     get_confusion_matrix_path,
#     get_training_curve_path,
#     load_best_model_summary,
#     load_description_contribution,
#     load_global_shap,
#     load_global_shap_by_class,
#     load_inference_efficiency,
#     load_local_shap_summary,
#     load_local_token_contributions,
#     load_misclassification_analysis,
#     load_model_comparison,
#     load_scenario_comparison,
#     load_test_evaluation,
#     load_waterfall_summary,
#     load_yake_contribution,
# )

# from utils.inference import predict_news


# # =============================================================================
# # KONFIGURASI HALAMAN
# # =============================================================================

# st.set_page_config(
#     page_title="Dashboard Klasifikasi Berita",
#     page_icon="📰",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )


# # =============================================================================
# # CSS RINGAN
# # Hanya untuk memperhalus komponen bawaan Streamlit.
# # Tidak digunakan untuk menulis isi kartu menggunakan HTML.
# # =============================================================================

# st.markdown(
#     """
#     <style>
#         .block-container {
#             padding-top: 2rem;
#             padding-bottom: 4rem;
#             max-width: 1450px;
#         }

#         div[data-testid="stMetric"] {
#             background-color: #f8fafc;
#             border: 1px solid #e2e8f0;
#             border-radius: 12px;
#             padding: 16px;
#         }

#         div[data-testid="stDataFrame"] {
#             border: 1px solid #e2e8f0;
#             border-radius: 10px;
#             overflow: hidden;
#         }

#         section[data-testid="stSidebar"] {
#             border-right: 1px solid #e2e8f0;
#         }

#         button[data-baseweb="tab"] {
#             font-weight: 600;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )


# # =============================================================================
# # KONFIGURASI EKSPERIMEN
# # =============================================================================

# EXPERIMENTS = {
#     "Kompas": {
#         "CNN": {
#             "K1": "cnn_k1",
#             "K2": "cnn_k2",
#             "K3": "cnn_k3",
#         },
#         "Attention-BiLSTM": {
#             "K1": "attention_bilstm_k1",
#             "K2": "attention_bilstm_k2",
#             "K3": "attention_bilstm_k3",
#         },
#     },
#     "AG News": {
#         "CNN": {
#             "A1": "cnn_a1",
#             "A2": "cnn_a2",
#         },
#         "Attention-BiLSTM": {
#             "A1": "attention_bilstm_a1",
#             "A2": "attention_bilstm_a2",
#         },
#     },
# }

# SCENARIO_NAMES = {
#     "K1": "Title",
#     "K2": "Title + Description",
#     "K3": "Title + Description + Keyword YAKE",
#     "A1": "Title",
#     "A2": "Title + Description",
# }

# CATEGORY_DISPLAY = {
#     "bola": "Bola",
#     "global": "Global",
#     "money": "Money",
#     "tekno": "Tekno",
# }


# # =============================================================================
# # HELPER UMUM
# # =============================================================================

# def find_column(
#     dataframe: pd.DataFrame,
#     candidates: list[str],
# ) -> str | None:
#     """Mencari kolom berdasarkan beberapa kemungkinan nama."""

#     if dataframe.empty:
#         return None

#     column_mapping = {
#         str(column).strip().lower(): column
#         for column in dataframe.columns
#     }

#     for candidate in candidates:
#         normalized = str(candidate).strip().lower()

#         if normalized in column_mapping:
#             return column_mapping[normalized]

#     return None


# def normalize_metric(value: Any) -> float | None:
#     """Menormalkan nilai metrik ke skala 0–1."""

#     try:
#         numeric_value = float(value)
#     except (TypeError, ValueError):
#         return None

#     if numeric_value > 1:
#         numeric_value /= 100

#     return numeric_value


# def percentage(value: Any) -> str:
#     """Mengubah nilai menjadi persentase."""

#     normalized = normalize_metric(value)

#     if normalized is None:
#         return "-"

#     return f"{normalized * 100:.2f}%"


# def remove_technical_columns(
#     dataframe: pd.DataFrame,
# ) -> pd.DataFrame:
#     """Menghapus path dan kolom teknis dari tampilan dashboard."""

#     if dataframe.empty:
#         return dataframe

#     technical_keywords = {
#         "path",
#         "checkpoint",
#         "directory",
#         "folder",
#         "file",
#     }

#     visible_columns = []

#     for column in dataframe.columns:
#         normalized = str(column).strip().lower()

#         is_technical = any(
#             keyword in normalized
#             for keyword in technical_keywords
#         )

#         if not is_technical:
#             visible_columns.append(column)

#     return dataframe[visible_columns].copy()


# def format_metric_table(
#     dataframe: pd.DataFrame,
# ) -> pd.DataFrame:
#     """Memformat kolom metrik pada tabel."""

#     formatted = remove_technical_columns(dataframe)

#     if formatted.empty:
#         return formatted

#     metric_keywords = {
#         "accuracy",
#         "precision",
#         "recall",
#         "f1",
#     }

#     excluded_keywords = {
#         "change",
#         "percentage_point",
#         "error",
#         "count",
#     }

#     for column in formatted.columns:
#         normalized = str(column).strip().lower()

#         is_metric = any(
#             keyword in normalized
#             for keyword in metric_keywords
#         )

#         is_excluded = any(
#             keyword in normalized
#             for keyword in excluded_keywords
#         )

#         if is_metric and not is_excluded:
#             formatted[column] = formatted[column].apply(
#                 lambda value: (
#                     percentage(value)
#                     if pd.notna(value)
#                     else "-"
#                 )
#             )

#     return formatted


# def get_experiment_row(
#     dataframe: pd.DataFrame,
#     experiment_name: str,
# ) -> pd.DataFrame:
#     """Mengambil baris satu eksperimen."""

#     if dataframe.empty:
#         return pd.DataFrame()

#     experiment_column = find_column(
#         dataframe,
#         [
#             "experiment_name",
#             "experiment",
#             "experiment_id",
#             "model_scenario",
#         ],
#     )

#     if experiment_column is None:
#         return pd.DataFrame()

#     return dataframe[
#         dataframe[experiment_column]
#         .astype(str)
#         .str.strip()
#         .str.lower()
#         == experiment_name.lower()
#     ].copy()


# def get_metric(
#     dataframe: pd.DataFrame,
#     experiment_name: str,
#     candidates: list[str],
# ) -> float | None:
#     """Mengambil satu metrik dari satu eksperimen."""

#     selected = get_experiment_row(
#         dataframe,
#         experiment_name,
#     )

#     if selected.empty:
#         return None

#     metric_column = find_column(
#         selected,
#         candidates,
#     )

#     if metric_column is None:
#         return None

#     return normalize_metric(
#         selected.iloc[0][metric_column]
#     )


# def show_image(
#     image_path: Path,
#     caption: str,
# ) -> None:
#     """Menampilkan gambar jika tersedia."""

#     if image_path.exists():
#         st.image(
#             str(image_path),
#             caption=caption,
#             use_container_width=True,
#         )
#     else:
#         st.info(
#             f"Grafik **{caption}** belum tersedia."
#         )


# def page_title(
#     title: str,
#     description: str,
# ) -> None:
#     """Judul konsisten untuk setiap menu."""

#     st.title(title)
#     st.caption(description)
#     st.divider()


# # =============================================================================
# # PERSIAPAN DATA EVALUASI UNTUK PLOTLY
# # =============================================================================

# def prepare_evaluation_chart_data(
#     dataframe: pd.DataFrame,
# ) -> pd.DataFrame:
#     """Menyiapkan data evaluasi untuk grafik interaktif."""

#     if dataframe.empty:
#         return pd.DataFrame()

#     result = dataframe.copy()

#     experiment_column = find_column(
#         result,
#         [
#             "experiment_name",
#             "experiment",
#         ],
#     )

#     accuracy_column = find_column(
#         result,
#         [
#             "accuracy",
#             "test_accuracy",
#         ],
#     )

#     f1_column = find_column(
#         result,
#         [
#             "f1_macro",
#             "macro_f1",
#             "f1_score_macro",
#         ],
#     )

#     if experiment_column is None:
#         return pd.DataFrame()

#     result["Eksperimen"] = (
#         result[experiment_column]
#         .astype(str)
#         .str.lower()
#     )

#     result["Model"] = np.where(
#         result["Eksperimen"].str.startswith("cnn"),
#         "CNN",
#         "Attention-BiLSTM",
#     )

#     result["Dataset"] = np.where(
#         result["Eksperimen"].str.contains("_k"),
#         "Kompas",
#         "AG News",
#     )

#     result["Skenario"] = (
#         result["Eksperimen"]
#         .str.split("_")
#         .str[-1]
#         .str.upper()
#     )

#     if accuracy_column is not None:
#         result["Accuracy (%)"] = (
#             pd.to_numeric(
#                 result[accuracy_column],
#                 errors="coerce",
#             )
#             .apply(
#                 lambda value: (
#                     value * 100
#                     if pd.notna(value) and value <= 1
#                     else value
#                 )
#             )
#         )

#     if f1_column is not None:
#         result["Macro F1 (%)"] = (
#             pd.to_numeric(
#                 result[f1_column],
#                 errors="coerce",
#             )
#             .apply(
#                 lambda value: (
#                     value * 100
#                     if pd.notna(value) and value <= 1
#                     else value
#                 )
#             )
#         )

#     return result


# # =============================================================================
# # LOAD SELURUH HASIL PENELITIAN
# # =============================================================================

# evaluation_data = load_test_evaluation()
# evaluation_chart_data = prepare_evaluation_chart_data(
#     evaluation_data
# )

# model_comparison = load_model_comparison()
# scenario_comparison = load_scenario_comparison()
# description_contribution = load_description_contribution()
# yake_contribution = load_yake_contribution()
# best_model_summary = load_best_model_summary()
# inference_efficiency = load_inference_efficiency()
# misclassification_data = load_misclassification_analysis()

# global_shap = load_global_shap()
# global_shap_by_class = load_global_shap_by_class()
# local_shap_summary = load_local_shap_summary()
# local_token_contributions = load_local_token_contributions()
# waterfall_summary = load_waterfall_summary()


# # =============================================================================
# # SIDEBAR NAVIGASI
# # =============================================================================

# with st.sidebar:
#     st.title("📰 Klasifikasi Berita")

#     st.caption(
#         "CNN dan Attention-BiLSTM"
#     )

#     st.divider()

#     selected_menu = st.radio(
#         "Menu Dashboard",
#         [
#             "🏠 Beranda",
#             "📚 Dataset & EDA",
#             "📊 Hasil dan Metrik",
#             "⚖️ Perbandingan Model",
#             "📰 Prediksi Berita",
#             "🔍 Explainable AI",
#         ],
#     )

#     st.divider()

#     st.markdown("#### Konfigurasi Final")
#     st.write("**Dataset utama:** Kompas")
#     st.write("**Model utama:** CNN K2")
#     st.write("**Representasi:** Title + Description")
#     st.write("**Sequence length:** 60")

#     st.divider()

#     st.caption(
#         "Dashboard penelitian klasifikasi "
#         "berita berbahasa Indonesia."
#     )


# # =============================================================================
# # MENU 1 — BERANDA
# # =============================================================================

# if selected_menu == "🏠 Beranda":

#     page_title(
#         "📰 Dashboard Klasifikasi Berita",
#         RESEARCH_TITLE,
#     )

#     st.info(
#         "Dashboard ini menampilkan dataset, hasil evaluasi, "
#         "perbandingan model, prediksi berita baru, dan "
#         "interpretasi model menggunakan SHAP."
#     )

#     kompas_info = DATASET_INFORMATION["Kompas"]
#     cnn_info = MODEL_PERFORMANCE["CNN K2"]

#     col1, col2, col3, col4 = st.columns(4)

#     col1.metric(
#         "Data Kompas",
#         f"{kompas_info['jumlah_data_setelah_cleaning']:,}",
#     )

#     col2.metric(
#         "Eksperimen Utama",
#         "10",
#     )

#     col3.metric(
#         "Model Terbaik",
#         PRIMARY_MODEL,
#     )

#     col4.metric(
#         "Accuracy Terbaik",
#         percentage(cnn_info["accuracy"]),
#     )

#     st.markdown("### Ringkasan Penelitian")

#     overview_col1, overview_col2 = st.columns(2)

#     with overview_col1:
#         with st.container(border=True):
#             st.subheader("Dataset")

#             st.write(
#                 "Dataset utama menggunakan 9.997 artikel "
#                 "Kompas setelah cleaning."
#             )

#             st.write(
#                 "Kategori: Bola, Global, Money, dan Tekno."
#             )

#             st.write(
#                 "AG News digunakan sebagai dataset benchmark."
#             )

#     with overview_col2:
#         with st.container(border=True):
#             st.subheader("Model dan Representasi")

#             st.write(
#                 "Model yang dibandingkan adalah CNN "
#                 "dan Attention-BiLSTM."
#             )

#             st.write(
#                 "Representasi terbaik adalah "
#                 "Title + Description."
#             )

#             st.write(
#                 "CNN K2 dipilih sebagai model utama."
#             )

#     st.markdown("### Temuan Utama")

#     finding_col1, finding_col2, finding_col3 = st.columns(3)

#     with finding_col1:
#         st.success(
#             "**Description meningkatkan performa**\n\n"
#             "CNN Kompas meningkat dari 94,70% "
#             "menjadi 95,80%."
#         )

#     with finding_col2:
#         st.warning(
#             "**YAKE belum meningkatkan accuracy**\n\n"
#             "CNN berubah dari 95,80% menjadi 95,00%."
#         )

#     with finding_col3:
#         st.info(
#             "**CNN K2 menjadi model terbaik**\n\n"
#             "Accuracy 95,80% dan Macro F1 95,81%."
#         )


# # =============================================================================
# # MENU 2 — DATASET DAN EDA
# # =============================================================================

# elif selected_menu == "📚 Dataset & EDA":

#     page_title(
#         "📚 Dataset dan Exploratory Data Analysis",
#         "Karakteristik dataset Kompas dan AG News.",
#     )

#     kompas_info = DATASET_INFORMATION["Kompas"]
#     agnews_info = DATASET_INFORMATION["AG News"]

#     col1, col2, col3, col4 = st.columns(4)

#     col1.metric(
#         "Kompas Awal",
#         f"{kompas_info['jumlah_data_awal']:,}",
#     )

#     col2.metric(
#         "Kompas Setelah Cleaning",
#         f"{kompas_info['jumlah_data_setelah_cleaning']:,}",
#     )

#     col3.metric(
#         "AG News Train",
#         f"{agnews_info['jumlah_data_train_setelah_cleaning']:,}",
#     )

#     col4.metric(
#         "AG News Test",
#         f"{agnews_info['jumlah_data_test_setelah_cleaning']:,}",
#     )

#     distribution_tab, length_tab, words_tab, temporal_tab = st.tabs(
#         [
#             "Distribusi Kelas",
#             "Panjang Teks",
#             "Frekuensi Kata",
#             "Distribusi Waktu",
#         ]
#     )

#     with distribution_tab:
#         image_col1, image_col2 = st.columns(2)

#         with image_col1:
#             show_image(
#                 FIGURES_DIR / "kompas_class_distribution.png",
#                 "Distribusi Kelas Kompas",
#             )

#         with image_col2:
#             show_image(
#                 FIGURES_DIR / "agnews_train_class_distribution.png",
#                 "Distribusi Kelas AG News Train",
#             )

#     with length_tab:
#         image_col1, image_col2 = st.columns(2)

#         with image_col1:
#             show_image(
#                 FIGURES_DIR / "kompas_text_length_distribution.png",
#                 "Distribusi Panjang Teks Kompas",
#             )

#         with image_col2:
#             show_image(
#                 FIGURES_DIR / "agnews_train_text_length_distribution.png",
#                 "Distribusi Panjang Teks AG News",
#             )

#     with words_tab:
#         image_col1, image_col2 = st.columns(2)

#         with image_col1:
#             show_image(
#                 FIGURES_DIR / "kompas_top_words.png",
#                 "Kata Teratas Kompas",
#             )

#         with image_col2:
#             show_image(
#                 FIGURES_DIR / "agnews_train_top_words.png",
#                 "Kata Teratas AG News",
#             )

#     with temporal_tab:
#         image_col1, image_col2 = st.columns(2)

#         with image_col1:
#             show_image(
#                 FIGURES_DIR / "kompas_monthly_distribution.png",
#                 "Distribusi Bulanan Kompas",
#             )

#         with image_col2:
#             show_image(
#                 FIGURES_DIR / "kompas_hourly_distribution.png",
#                 "Distribusi Jam Publikasi Kompas",
#             )


# # =============================================================================
# # MENU 3 — HASIL DAN METRIK
# # =============================================================================

# elif selected_menu == "📊 Hasil dan Metrik":

#     page_title(
#         "📊 Hasil dan Metrik",
#         "Evaluasi 10 eksperimen utama pada test set.",
#     )

#     col1, col2, col3, col4 = st.columns(4)

#     col1.metric("Jumlah Eksperimen", "10")
#     col2.metric("Model Terbaik", "CNN K2")
#     col3.metric("Accuracy Terbaik", "95.80%")
#     col4.metric("Macro F1 Terbaik", "95.81%")

#     st.markdown("### Performa Seluruh Eksperimen")

#     accuracy_tab, f1_tab = st.tabs(
#         [
#             "Accuracy",
#             "Macro F1",
#         ]
#     )

#     with accuracy_tab:
#         if (
#             not evaluation_chart_data.empty
#             and "Accuracy (%)" in evaluation_chart_data.columns
#         ):
#             fig_accuracy = px.bar(
#                 evaluation_chart_data,
#                 x="Skenario",
#                 y="Accuracy (%)",
#                 color="Model",
#                 facet_col="Dataset",
#                 barmode="group",
#                 text="Accuracy (%)",
#                 title="Perbandingan Accuracy",
#             )

#             fig_accuracy.update_traces(
#                 texttemplate="%{text:.2f}%",
#                 textposition="outside",
#             )

#             fig_accuracy.update_yaxes(
#                 range=[75, 100]
#             )

#             st.plotly_chart(
#                 fig_accuracy,
#                 use_container_width=True,
#             )
#         else:
#             show_image(
#                 get_comparative_figure_path(
#                     "accuracy_comparison.png"
#                 ),
#                 "Perbandingan Accuracy",
#             )

#     with f1_tab:
#         if (
#             not evaluation_chart_data.empty
#             and "Macro F1 (%)" in evaluation_chart_data.columns
#         ):
#             fig_f1 = px.bar(
#                 evaluation_chart_data,
#                 x="Skenario",
#                 y="Macro F1 (%)",
#                 color="Model",
#                 facet_col="Dataset",
#                 barmode="group",
#                 text="Macro F1 (%)",
#                 title="Perbandingan Macro F1",
#             )

#             fig_f1.update_traces(
#                 texttemplate="%{text:.2f}%",
#                 textposition="outside",
#             )

#             fig_f1.update_yaxes(
#                 range=[75, 100]
#             )

#             st.plotly_chart(
#                 fig_f1,
#                 use_container_width=True,
#             )
#         else:
#             show_image(
#                 get_comparative_figure_path(
#                     "f1_macro_comparison.png"
#                 ),
#                 "Perbandingan Macro F1",
#             )

#     st.divider()
#     st.markdown("### Detail Eksperimen")

#     filter_col1, filter_col2, filter_col3 = st.columns(3)

#     with filter_col1:
#         selected_dataset = st.selectbox(
#             "Dataset",
#             ["Kompas", "AG News"],
#         )

#     with filter_col2:
#         selected_model = st.selectbox(
#             "Model",
#             ["CNN", "Attention-BiLSTM"],
#         )

#     available_scenarios = list(
#         EXPERIMENTS[selected_dataset][selected_model].keys()
#     )

#     with filter_col3:
#         selected_scenario = st.selectbox(
#             "Skenario",
#             available_scenarios,
#             index=(
#                 1
#                 if len(available_scenarios) > 1
#                 else 0
#             ),
#             format_func=lambda code: (
#                 f"{code} — {SCENARIO_NAMES[code]}"
#             ),
#         )

#     experiment_name = (
#         EXPERIMENTS[selected_dataset]
#         [selected_model]
#         [selected_scenario]
#     )

#     accuracy = get_metric(
#         evaluation_data,
#         experiment_name,
#         ["accuracy", "test_accuracy"],
#     )

#     precision_macro = get_metric(
#         evaluation_data,
#         experiment_name,
#         [
#             "precision_macro",
#             "macro_precision",
#             "precision",
#         ],
#     )

#     recall_macro = get_metric(
#         evaluation_data,
#         experiment_name,
#         [
#             "recall_macro",
#             "macro_recall",
#             "recall",
#         ],
#     )

#     f1_macro = get_metric(
#         evaluation_data,
#         experiment_name,
#         [
#             "f1_macro",
#             "macro_f1",
#             "f1_score_macro",
#         ],
#     )

#     metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

#     metric_col1.metric(
#         "Accuracy",
#         percentage(accuracy),
#     )

#     metric_col2.metric(
#         "Precision Macro",
#         percentage(precision_macro),
#     )

#     metric_col3.metric(
#         "Recall Macro",
#         percentage(recall_macro),
#     )

#     metric_col4.metric(
#         "F1-Score Macro",
#         percentage(f1_macro),
#     )

#     confusion_tab, training_tab = st.tabs(
#         [
#             "Confusion Matrix",
#             "Training Curve",
#         ]
#     )

#     with confusion_tab:
#         matrix_type = st.radio(
#             "Jenis Confusion Matrix",
#             ["Jumlah", "Normalized"],
#             horizontal=True,
#         )

#         confusion_path = get_confusion_matrix_path(
#             experiment_name,
#             normalized=(
#                 matrix_type == "Normalized"
#             ),
#         )

#         show_image(
#             confusion_path,
#             f"Confusion Matrix {experiment_name}",
#         )

#     with training_tab:
#         show_image(
#             get_training_curve_path(
#                 experiment_name
#             ),
#             f"Training Curve {experiment_name}",
#         )

#     st.markdown("### Ringkasan 10 Eksperimen")

#     if evaluation_data.empty:
#         st.info("Data evaluasi belum tersedia.")
#     else:
#         st.dataframe(
#             format_metric_table(
#                 evaluation_data
#             ),
#             use_container_width=True,
#             hide_index=True,
#         )


# # =============================================================================
# # MENU 4 — PERBANDINGAN MODEL
# # =============================================================================

# elif selected_menu == "⚖️ Perbandingan Model":

#     page_title(
#         "⚖️ Perbandingan Model",
#         "CNN vs Attention-BiLSTM dan pengaruh representasi teks.",
#     )

#     col1, col2, col3, col4 = st.columns(4)

#     col1.metric("Model Terbaik", "CNN K2")
#     col2.metric("Accuracy", "95.80%")
#     col3.metric("Macro F1", "95.81%")
#     col4.metric("Representasi", "Title + Description")

#     model_col1, model_col2 = st.columns(2)

#     with model_col1:
#         with st.container(border=True):
#             st.subheader("🧩 CNN")

#             st.write(
#                 "CNN menangkap pola lokal dan frasa penting "
#                 "melalui operasi convolution."
#             )

#             st.markdown(
#                 """
#                 **Hasil penelitian:**

#                 - unggul pada K1 dan K2;
#                 - unggul pada A1 dan A2;
#                 - menghasilkan model terbaik CNN K2;
#                 - digunakan sebagai model utama dan model SHAP.
#                 """
#             )

#     with model_col2:
#         with st.container(border=True):
#             st.subheader("🔁 Attention-BiLSTM")

#             st.write(
#                 "Attention-BiLSTM mempelajari konteks dua arah "
#                 "dan memberikan bobot pada token penting."
#             )

#             st.markdown(
#                 """
#                 **Hasil penelitian:**

#                 - performanya kompetitif;
#                 - memperoleh 95,30% pada K2;
#                 - sedikit unggul pada K3;
#                 - digunakan sebagai model pembanding.
#                 """
#             )

#     comparison_tab1, comparison_tab2, comparison_tab3 = st.tabs(
#         [
#             "Perbandingan Model",
#             "Kontribusi Description",
#             "Kontribusi YAKE",
#         ]
#     )

#     with comparison_tab1:
#         show_image(
#             get_comparative_figure_path(
#                 "accuracy_comparison.png"
#             ),
#             "Perbandingan Accuracy Model",
#         )

#         if not model_comparison.empty:
#             st.dataframe(
#                 format_metric_table(
#                     model_comparison
#                 ),
#                 use_container_width=True,
#                 hide_index=True,
#             )

#     with comparison_tab2:
#         show_image(
#             get_comparative_figure_path(
#                 "description_contribution.png"
#             ),
#             "Kontribusi Description",
#         )

#         st.success(
#             "Description meningkatkan performa kedua model "
#             "pada Kompas dan AG News."
#         )

#         if not description_contribution.empty:
#             st.dataframe(
#                 format_metric_table(
#                     description_contribution
#                 ),
#                 use_container_width=True,
#                 hide_index=True,
#             )

#     with comparison_tab3:
#         show_image(
#             get_comparative_figure_path(
#                 "yake_contribution.png"
#             ),
#             "Kontribusi Keyword YAKE",
#         )

#         st.warning(
#             "Pada konfigurasi penelitian ini, YAKE belum "
#             "meningkatkan accuracy dibandingkan "
#             "Title + Description."
#         )

#         if not yake_contribution.empty:
#             st.dataframe(
#                 format_metric_table(
#                     yake_contribution
#                 ),
#                 use_container_width=True,
#                 hide_index=True,
#             )

#     st.success(
#         "**Rekomendasi final:** CNN K2 dipilih karena "
#         "menghasilkan accuracy 95,80% dan Macro F1 95,81% "
#         "pada dataset utama Kompas."
#     )


# # =============================================================================
# # MENU 5 — PREDIKSI BERITA
# # =============================================================================

# elif selected_menu == "📰 Prediksi Berita":

#     page_title(
#         "📰 Prediksi Berita",
#         "Masukkan Title dan Description untuk mengklasifikasikan berita.",
#     )

#     st.info(
#         "Input akan diproses menggunakan representasi K2, "
#         "yaitu Title + Description dengan sequence length 60."
#     )

#     with st.form(
#         "prediction_form"
#     ):
#         title_input = st.text_input(
#             "Title",
#             placeholder=(
#                 "Contoh: Rupiah Menguat terhadap Dolar AS"
#             ),
#         )

#         description_input = st.text_area(
#             "Description",
#             placeholder=(
#                 "Contoh: Nilai tukar rupiah menguat setelah "
#                 "Bank Indonesia mengumumkan kebijakan..."
#             ),
#             height=140,
#         )

#         submitted = st.form_submit_button(
#             "🔍 Prediksi Berita",
#             use_container_width=True,
#         )

#     if submitted:
#         if not title_input.strip():
#             st.warning("Title tidak boleh kosong.")

#         elif not description_input.strip():
#             st.warning("Description tidak boleh kosong.")

#         else:
#             try:
#                 with st.spinner(
#                     "Menjalankan prediksi CNN dan Attention-BiLSTM..."
#                 ):
#                     prediction_result = predict_news(
#                         title=title_input,
#                         description=description_input,
#                     )

#                 st.success("Prediksi berhasil.")

#                 recommendation = prediction_result[
#                     "recommended_prediction"
#                 ]

#                 recommended_label = CATEGORY_DISPLAY.get(
#                     recommendation["predicted_label"],
#                     recommendation["predicted_label"].title(),
#                 )

#                 result_col1, result_col2, result_col3 = st.columns(3)

#                 result_col1.metric(
#                     "Prediksi Utama",
#                     recommended_label,
#                 )

#                 result_col2.metric(
#                     "Model Utama",
#                     recommendation["source_model"],
#                 )

#                 result_col3.metric(
#                     "Confidence",
#                     percentage(
#                         recommendation["confidence"]
#                     ),
#                 )

#                 cnn_result = prediction_result["cnn"]
#                 attention_result = prediction_result[
#                     "attention_bilstm"
#                 ]

#                 cnn_col, attention_col = st.columns(2)

#                 with cnn_col:
#                     with st.container(border=True):
#                         st.subheader("CNN K2")

#                         st.metric(
#                             "Prediksi",
#                             CATEGORY_DISPLAY.get(
#                                 cnn_result["predicted_label"],
#                                 cnn_result[
#                                     "predicted_label"
#                                 ].title(),
#                             ),
#                         )

#                         st.metric(
#                             "Confidence",
#                             percentage(
#                                 cnn_result["confidence"]
#                             ),
#                         )

#                         st.caption(
#                             "Waktu inferensi: "
#                             f"{cnn_result['inference_time_ms']:.2f} ms"
#                         )

#                         cnn_probability = pd.DataFrame(
#                             {
#                                 "Kategori": [
#                                     CATEGORY_DISPLAY.get(
#                                         label,
#                                         label.title(),
#                                     )
#                                     for label in cnn_result[
#                                         "probabilities"
#                                     ].keys()
#                                 ],
#                                 "Probabilitas (%)": [
#                                     value * 100
#                                     for value in cnn_result[
#                                         "probabilities"
#                                     ].values()
#                                 ],
#                             }
#                         )

#                         fig_cnn = px.bar(
#                             cnn_probability,
#                             x="Kategori",
#                             y="Probabilitas (%)",
#                             text="Probabilitas (%)",
#                             title="Probabilitas CNN",
#                         )

#                         fig_cnn.update_traces(
#                             texttemplate="%{text:.2f}%",
#                             textposition="outside",
#                         )

#                         st.plotly_chart(
#                             fig_cnn,
#                             use_container_width=True,
#                         )

#                 with attention_col:
#                     with st.container(border=True):
#                         st.subheader("Attention-BiLSTM K2")

#                         st.metric(
#                             "Prediksi",
#                             CATEGORY_DISPLAY.get(
#                                 attention_result[
#                                     "predicted_label"
#                                 ],
#                                 attention_result[
#                                     "predicted_label"
#                                 ].title(),
#                             ),
#                         )

#                         st.metric(
#                             "Confidence",
#                             percentage(
#                                 attention_result["confidence"]
#                             ),
#                         )

#                         st.caption(
#                             "Waktu inferensi: "
#                             f"{attention_result['inference_time_ms']:.2f} ms"
#                         )

#                         attention_probability = pd.DataFrame(
#                             {
#                                 "Kategori": [
#                                     CATEGORY_DISPLAY.get(
#                                         label,
#                                         label.title(),
#                                     )
#                                     for label in attention_result[
#                                         "probabilities"
#                                     ].keys()
#                                 ],
#                                 "Probabilitas (%)": [
#                                     value * 100
#                                     for value in attention_result[
#                                         "probabilities"
#                                     ].values()
#                                 ],
#                             }
#                         )

#                         fig_attention = px.bar(
#                             attention_probability,
#                             x="Kategori",
#                             y="Probabilitas (%)",
#                             text="Probabilitas (%)",
#                             title="Probabilitas Attention-BiLSTM",
#                         )

#                         fig_attention.update_traces(
#                             texttemplate="%{text:.2f}%",
#                             textposition="outside",
#                         )

#                         st.plotly_chart(
#                             fig_attention,
#                             use_container_width=True,
#                         )

#                 if prediction_result["model_agreement"]:
#                     st.success(
#                         "CNN dan Attention-BiLSTM memberikan "
#                         "kategori yang sama."
#                     )
#                 else:
#                     st.warning(
#                         "CNN dan Attention-BiLSTM memberikan "
#                         "hasil yang berbeda. Rekomendasi utama "
#                         "mengikuti CNN K2 sebagai model terbaik."
#                     )

#             except Exception as error:
#                 st.error(
#                     f"Prediksi gagal: {error}"
#                 )


# # =============================================================================
# # MENU 6 — EXPLAINABLE AI
# # =============================================================================

# elif selected_menu == "🔍 Explainable AI":

#     page_title(
#         "🔍 Explainable AI — SHAP",
#         "Interpretasi token yang memengaruhi keputusan CNN K2.",
#     )

#     st.info(
#         "SHAP diterapkan pada CNN K2 karena model tersebut "
#         "memperoleh performa terbaik pada dataset Kompas."
#     )

#     global_tab, class_tab, local_tab, waterfall_tab = st.tabs(
#         [
#             "Global SHAP",
#             "SHAP per Kelas",
#             "Local SHAP",
#             "Waterfall Plot",
#         ]
#     )

#     with global_tab:
#         st.subheader("Token Paling Berpengaruh")

#         if global_shap.empty:
#             st.info("Data global SHAP belum tersedia.")

#         else:
#             token_column = find_column(
#                 global_shap,
#                 ["token", "word"],
#             )

#             importance_column = find_column(
#                 global_shap,
#                 [
#                     "total_abs_shap",
#                     "mean_abs_shap",
#                     "importance",
#                 ],
#             )

#             top_n = st.slider(
#                 "Jumlah token",
#                 min_value=10,
#                 max_value=50,
#                 value=20,
#                 step=5,
#             )

#             if (
#                 token_column is not None
#                 and importance_column is not None
#             ):
#                 top_tokens = (
#                     global_shap
#                     .sort_values(
#                         importance_column,
#                         ascending=False,
#                     )
#                     .head(top_n)
#                     .sort_values(
#                         importance_column,
#                         ascending=True,
#                     )
#                 )

#                 fig_global = px.bar(
#                     top_tokens,
#                     x=importance_column,
#                     y=token_column,
#                     orientation="h",
#                     title=(
#                         f"{top_n} Token Paling "
#                         "Berpengaruh Secara Global"
#                     ),
#                 )

#                 st.plotly_chart(
#                     fig_global,
#                     use_container_width=True,
#                 )

#                 st.dataframe(
#                     remove_technical_columns(
#                         top_tokens.sort_values(
#                             importance_column,
#                             ascending=False,
#                         )
#                     ),
#                     use_container_width=True,
#                     hide_index=True,
#                 )

#     with class_tab:
#         if global_shap_by_class.empty:
#             st.info(
#                 "Data SHAP per kelas belum tersedia."
#             )

#         else:
#             class_column = find_column(
#                 global_shap_by_class,
#                 [
#                     "class_name",
#                     "class",
#                     "label",
#                     "category",
#                 ],
#             )

#             token_column = find_column(
#                 global_shap_by_class,
#                 ["token", "word"],
#             )

#             importance_column = find_column(
#                 global_shap_by_class,
#                 [
#                     "total_abs_shap",
#                     "mean_abs_shap",
#                     "importance",
#                 ],
#             )

#             if class_column is not None:
#                 available_classes = sorted(
#                     global_shap_by_class[
#                         class_column
#                     ]
#                     .dropna()
#                     .astype(str)
#                     .unique()
#                     .tolist()
#                 )

#                 selected_class = st.selectbox(
#                     "Pilih kelas",
#                     available_classes,
#                 )

#                 class_data = global_shap_by_class[
#                     global_shap_by_class[
#                         class_column
#                     ].astype(str)
#                     == selected_class
#                 ].copy()

#                 if (
#                     token_column is not None
#                     and importance_column is not None
#                 ):
#                     class_data = (
#                         class_data
#                         .sort_values(
#                             importance_column,
#                             ascending=False,
#                         )
#                         .head(20)
#                         .sort_values(
#                             importance_column,
#                             ascending=True,
#                         )
#                     )

#                     fig_class = px.bar(
#                         class_data,
#                         x=importance_column,
#                         y=token_column,
#                         orientation="h",
#                         title=(
#                             "Token Penting untuk Kelas "
#                             f"{selected_class}"
#                         ),
#                     )

#                     st.plotly_chart(
#                         fig_class,
#                         use_container_width=True,
#                     )

#     with local_tab:
#         if local_shap_summary.empty:
#             st.info(
#                 "Data local SHAP belum tersedia."
#             )

#         else:
#             st.subheader("Ringkasan Sampel Local SHAP")

#             st.dataframe(
#                 remove_technical_columns(
#                     local_shap_summary
#                 ),
#                 use_container_width=True,
#                 hide_index=True,
#             )

#             sample_column = find_column(
#                 local_shap_summary,
#                 [
#                     "document_id",
#                     "sample_id",
#                     "id",
#                 ],
#             )

#             contribution_sample_column = find_column(
#                 local_token_contributions,
#                 [
#                     "document_id",
#                     "sample_id",
#                     "id",
#                 ],
#             )

#             if (
#                 sample_column is not None
#                 and contribution_sample_column is not None
#                 and not local_token_contributions.empty
#             ):
#                 selected_sample = st.selectbox(
#                     "Pilih sampel",
#                     local_shap_summary[
#                         sample_column
#                     ]
#                     .astype(str)
#                     .tolist(),
#                 )

#                 sample_tokens = local_token_contributions[
#                     local_token_contributions[
#                         contribution_sample_column
#                     ].astype(str)
#                     == str(selected_sample)
#                 ].copy()

#                 token_column = find_column(
#                     sample_tokens,
#                     ["token", "word"],
#                 )

#                 contribution_column = find_column(
#                     sample_tokens,
#                     [
#                         "shap_value",
#                         "contribution",
#                         "token_contribution",
#                     ],
#                 )

#                 if (
#                     token_column is not None
#                     and contribution_column is not None
#                     and not sample_tokens.empty
#                 ):
#                     sample_tokens = (
#                         sample_tokens
#                         .assign(
#                             absolute_contribution=lambda data: (
#                                 pd.to_numeric(
#                                     data[contribution_column],
#                                     errors="coerce",
#                                 ).abs()
#                             )
#                         )
#                         .sort_values(
#                             "absolute_contribution",
#                             ascending=False,
#                         )
#                         .head(20)
#                     )

#                     fig_local = px.bar(
#                         sample_tokens,
#                         x=contribution_column,
#                         y=token_column,
#                         orientation="h",
#                         title=(
#                             f"Kontribusi Token Sampel "
#                             f"{selected_sample}"
#                         ),
#                     )

#                     st.plotly_chart(
#                         fig_local,
#                         use_container_width=True,
#                     )

#     with waterfall_tab:
#         waterfall_dir = (
#             FIGURES_DIR
#             / "shap"
#             / "waterfall"
#         )

#         waterfall_files = (
#             sorted(
#                 waterfall_dir.glob("*.png")
#             )
#             if waterfall_dir.exists()
#             else []
#         )

#         if not waterfall_files:
#             st.info(
#                 "Grafik waterfall belum tersedia."
#             )

#         else:
#             selected_waterfall = st.selectbox(
#                 "Pilih sampel waterfall",
#                 waterfall_files,
#                 format_func=lambda path: path.stem,
#             )

#             show_image(
#                 selected_waterfall,
#                 selected_waterfall.stem,
#             )

#             st.caption(
#                 "Token dengan kontribusi positif mendukung "
#                 "kelas prediksi, sedangkan kontribusi negatif "
#                 "mengurangi dukungan terhadap kelas tersebut."
#             )


# # =============================================================================
# # FOOTER
# # =============================================================================

# st.divider()

# st.caption(
#     "Dashboard penelitian klasifikasi berita — "
#     "CNN, Attention-BiLSTM, dan Explainable AI SHAP."
# )

# =============================================================================
# DASHBOARD KLASIFIKASI BERITA - SINGLE PAGE APPLICATION
# =============================================================================
# Jalankan hanya dengan:
# python -m streamlit run .\10_streamlit\app.py
#
# Catatan:
# - Tidak menggunakan folder pages untuk navigasi multipage.
# - Seluruh hasil penelitian dibaca dari file CSV melalui data_loader.py.
# - Model deep learning dimuat dari file .keras melalui inference.py.
# - Joblib tidak digunakan untuk model CNN/Attention-BiLSTM karena format
#   model yang benar adalah Keras (.keras). Joblib hanya cocok untuk objek
#   scikit-learn atau preprocessor yang memang disimpan dengan joblib.
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =============================================================================
# PATH PROJECT
# =============================================================================

CURRENT_FILE = Path(__file__).resolve()
STREAMLIT_DIR = CURRENT_FILE.parent

if str(STREAMLIT_DIR) not in sys.path:
    sys.path.insert(0, str(STREAMLIT_DIR))


# =============================================================================
# IMPORT PROJECT
# =============================================================================

from config import (
    DATASET_INFORMATION,
    FIGURES_DIR,
    MODEL_PERFORMANCE,
    RESEARCH_TITLE,
)

from utils.data_loader import (
    get_confusion_matrix_path,
    get_training_curve_path,
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

from utils.inference import predict_news


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
# STYLING
# Hanya CSS tampilan. Isi dashboard tetap menggunakan komponen native Streamlit.
# =============================================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1500px;
            padding-top: 1.6rem;
            padding-bottom: 4rem;
        }

        section[data-testid="stSidebar"] {
            display: none;
        }

        div[data-testid="collapsedControl"] {
            display: none;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, #ffffff, #f7f9fc);
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 16px 18px;
            min-height: 132px;
            box-shadow: 0 5px 16px rgba(15, 23, 42, 0.05);
        }

        div[data-testid="stMetricLabel"] {
            font-weight: 650;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            overflow: hidden;
        }

        div[data-testid="stForm"] {
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 20px;
            background: #fbfcfe;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            padding: 6px;
            background: #f1f5f9;
            border-radius: 14px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 48px;
            border-radius: 10px;
            padding-left: 18px;
            padding-right: 18px;
            font-weight: 650;
        }

        .stTabs [aria-selected="true"] {
            background: white;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
        }

        h1, h2, h3 {
            letter-spacing: -0.02em;
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
}

KOMPAS_TEST_SIZE = 1000


# =============================================================================
# HELPERS
# =============================================================================

def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """Mencari nama kolom secara fleksibel."""

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
    """Menampilkan nilai metrik sebagai persentase."""

    normalized = normalize_metric(value)
    if normalized is None:
        return "-"

    return f"{normalized * 100:.{digits}f}%"


def number_id(value: int | float) -> str:
    """Format angka Indonesia dengan pemisah ribuan titik."""

    return f"{value:,.0f}".replace(",", ".")


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
    """Menghapus path dan kolom teknis dari tabel dashboard."""

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
    """Membersihkan dan merapikan tabel hasil evaluasi."""

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
            lambda value: percentage(value)
            if pd.notna(value)
            else "-"
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

    ordered = [
        column
        for column in preferred
        if column in table.columns
    ]

    remaining = [
        column
        for column in table.columns
        if column not in ordered
    ]

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
        result[experiment_column]
        .astype(str)
        .str.lower()
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
        result["Eksperimen"]
        .str.split("_")
        .str[-1]
        .str.upper()
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


def first_existing_path(
    candidates: list[Path],
) -> Path | None:
    """Mengambil gambar pertama yang tersedia."""

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def show_research_figure(
    candidates: list[Path],
    title: str,
    explanation: str,
) -> None:
    """Menampilkan grafik penelitian beserta keterangannya."""

    selected_path = first_existing_path(candidates)

    st.subheader(title)

    if selected_path is None:
        st.info(f"Grafik {title} belum tersedia.")
    else:
        st.image(
            str(selected_path),
            use_container_width=True,
        )

    st.caption(explanation)


def add_bar_labels(figure) -> None:
    """Menambahkan label persentase pada grafik batang."""

    figure.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        cliponaxis=False,
    )


# =============================================================================
# LOAD CSV RESULTS
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
        "Enam eksperimen pada Kompas dan empat eksperimen "
        "pada AG News."
    ),
)

summary_3.metric(
    "Model Terbaik pada Test Set",
    "CNN K2",
    help=(
        "CNN menggunakan Title + Description tanpa keyword YAKE "
        "dengan sequence length 60."
    ),
)

summary_4.metric(
    "Accuracy Test Terbaik",
    percentage(cnn_k2_info["accuracy"]),
    help=(
        "CNN K2 memprediksi benar 958 dari 1.000 artikel "
        "test Kompas."
    ),
)


# =============================================================================
# MAIN TABS - SATU APP, TANPA MENU PAGES DI SIDEBAR
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
                berita berbahasa Indonesia. Dataset terdiri dari empat
                kategori: **Bola, Global, Money, dan Tekno**.

                **AG News** digunakan sebagai dataset benchmark berbahasa
                Inggris untuk melihat konsistensi performa model pada
                dataset yang lebih besar.
                """
            )

    with overview_2:
        with st.container(border=True):
            st.subheader("Desain Eksperimen")

            st.markdown(
                """
                Dua model yang dibandingkan adalah **CNN** dan
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
            "CNN pada Kompas meningkat dari 94,70% menjadi 95,80%."
        )

    with finding_2:
        st.warning(
            "**YAKE belum meningkatkan accuracy**\n\n"
            "CNN turun 0,80 percentage point dan "
            "Attention-BiLSTM turun 0,10 percentage point."
        )

    with finding_3:
        st.info(
            "**CNN K2 menjadi model terbaik**\n\n"
            "Accuracy 95,80% dan Macro F1 95,81%."
        )

    st.subheader("Perbandingan Model Utama")

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
        )
        add_bar_labels(figure)
        figure.update_yaxes(range=[90, 100])

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

        st.caption(
            "CNN unggul pada K1 dan K2. Attention-BiLSTM sedikit "
            "unggul pada K3, tetapi accuracy tertinggi keseluruhan "
            "tetap diperoleh CNN K2 sebesar 95,80%."
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
                "Dataset Kompas digunakan untuk melatih dan menguji "
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
                "CNN dan Attention-BiLSTM pada dataset internasional."
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
            show_research_figure(
                [
                    FIGURES_DIR / "kompas_class_distribution.png",
                ],
                "Distribusi Kelas Kompas",
                (
                    "Grafik menunjukkan jumlah artikel pada kategori "
                    "Bola, Global, Money, dan Tekno. Distribusi yang "
                    "hampir seimbang mengurangi dominasi satu kelas."
                ),
            )

        with col_2:
            show_research_figure(
                [
                    FIGURES_DIR / "agnews_train_class_distribution.png",
                ],
                "Distribusi Kelas AG News",
                (
                    "Grafik menunjukkan distribusi kelas Business, "
                    "Sci/Tech, Sports, dan World pada data train AG News."
                ),
            )

    with eda_tab_2:
        col_1, col_2 = st.columns(2)

        with col_1:
            show_research_figure(
                [
                    FIGURES_DIR / "kompas_text_length_distribution.png",
                ],
                "Distribusi Panjang Teks Kompas",
                (
                    "Grafik digunakan untuk memahami panjang Title, "
                    "Description, dan Content serta membantu menentukan "
                    "sequence length yang sesuai."
                ),
            )

        with col_2:
            show_research_figure(
                [
                    FIGURES_DIR / "agnews_train_text_length_distribution.png",
                ],
                "Distribusi Panjang Teks AG News",
                (
                    "Grafik menunjukkan karakteristik panjang Title dan "
                    "Description pada data benchmark AG News."
                ),
            )

    with eda_tab_3:
        col_1, col_2 = st.columns(2)

        with col_1:
            show_research_figure(
                [
                    FIGURES_DIR / "kompas_top_words.png",
                ],
                "Kata yang Sering Muncul pada Kompas",
                (
                    "Frekuensi kata menunjukkan kata yang sering muncul, "
                    "tetapi tidak selalu menunjukkan kata yang paling "
                    "memengaruhi keputusan model."
                ),
            )

        with col_2:
            show_research_figure(
                [
                    FIGURES_DIR / "agnews_train_top_words.png",
                ],
                "Kata yang Sering Muncul pada AG News",
                (
                    "Grafik memberikan gambaran topik dominan pada "
                    "dataset benchmark."
                ),
            )

    with eda_tab_4:
        col_1, col_2 = st.columns(2)

        with col_1:
            show_research_figure(
                [
                    FIGURES_DIR / "kompas_monthly_category_distribution.png",
                    FIGURES_DIR / "kompas_monthly_distribution.png",
                ],
                "Distribusi Bulanan Artikel Kompas",
                (
                    "Grafik menunjukkan cakupan waktu publikasi artikel. "
                    "Informasi waktu digunakan untuk EDA, bukan sebagai "
                    "fitur input model final."
                ),
            )

        with col_2:
            show_research_figure(
                [
                    FIGURES_DIR / "kompas_hourly_distribution.png",
                    FIGURES_DIR / "kompas_daily_distribution.png",
                ],
                "Distribusi Waktu Publikasi Kompas",
                (
                    "Grafik menunjukkan pola waktu artikel diterbitkan "
                    "selama periode crawling."
                ),
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

    result_tab_1, result_tab_2 = st.tabs(
        ["Accuracy", "Macro F1"]
    )

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
            )
            add_bar_labels(figure)
            figure.update_yaxes(range=[80, 100])

            st.plotly_chart(figure, use_container_width=True)

            st.caption(
                "CNN unggul pada K1, K2, A1, dan A2. "
                "Attention-BiLSTM sedikit unggul pada K3."
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
            )
            add_bar_labels(figure)
            figure.update_yaxes(range=[80, 100])

            st.plotly_chart(figure, use_container_width=True)

            st.caption(
                "Macro F1 digunakan untuk memastikan performa model "
                "tetap seimbang pada semua kategori."
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
            format_func=lambda code: (
                f"{code} — {SCENARIO_NAMES[code]}"
            ),
            key="result_scenario",
        )

    experiment_name = (
        EXPERIMENTS[selected_dataset]
        [selected_model]
        [selected_scenario]
    )

    metrics = get_metric_bundle(
        evaluation_data,
        experiment_name,
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    metric_1.metric("Accuracy", percentage(metrics["accuracy"]))
    metric_2.metric("Precision Macro", percentage(metrics["precision"]))
    metric_3.metric("Recall Macro", percentage(metrics["recall"]))
    metric_4.metric("Macro F1", percentage(metrics["f1"]))

    visual_1, visual_2 = st.tabs(
        ["Confusion Matrix", "Training Curve"]
    )

    with visual_1:
        matrix_type = st.radio(
            "Tampilan",
            ["Jumlah", "Normalized"],
            horizontal=True,
            key="matrix_type",
        )

        matrix_path = get_confusion_matrix_path(
            experiment_name,
            normalized=matrix_type == "Normalized",
        )

        if matrix_path.exists():
            st.image(str(matrix_path), use_container_width=True)
        else:
            st.info("Confusion matrix belum tersedia.")

        if matrix_type == "Jumlah":
            st.caption(
                "Confusion matrix jumlah menampilkan banyaknya data "
                "pada setiap kombinasi kelas aktual dan prediksi."
            )
        else:
            st.caption(
                "Confusion matrix normalized menampilkan proporsi "
                "atau persentase prediksi pada setiap kelas aktual."
            )

    with visual_2:
        training_path = get_training_curve_path(experiment_name)

        if training_path.exists():
            st.image(str(training_path), use_container_width=True)
        else:
            st.info("Training curve belum tersedia.")

        st.caption(
            "Training curve memperlihatkan perubahan accuracy dan loss "
            "pada data train dan validation. Model final menggunakan "
            "checkpoint pada validation loss terbaik."
        )

    st.subheader("Tabel Ringkasan Evaluasi")

    if evaluation_data.empty:
        st.info("Data evaluasi belum tersedia.")
    else:
        st.dataframe(
            clean_result_table(evaluation_data),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Tabel menampilkan hasil 10 eksperimen tanpa path internal. "
            "CNN K2 menjadi konfigurasi terbaik pada dataset Kompas."
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

    comparison_rows = []
    detailed_rows = []

    for model_name, without_name, with_name in experiments_yake:
        without_metrics = get_metric_bundle(
            evaluation_data,
            without_name,
        )
        with_metrics = get_metric_bundle(
            evaluation_data,
            with_name,
        )

        without_accuracy = without_metrics["accuracy"]
        with_accuracy = with_metrics["accuracy"]

        change_pp = (
            (with_accuracy - without_accuracy) * 100
            if without_accuracy is not None
            and with_accuracy is not None
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
                    "Precision Macro": percentage(
                        without_metrics["precision"]
                    ),
                    "Recall Macro": percentage(
                        without_metrics["recall"]
                    ),
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
                    "Representasi": (
                        "Title + Description + Keyword YAKE"
                    ),
                    "Accuracy": percentage(with_metrics["accuracy"]),
                    "Precision Macro": percentage(
                        with_metrics["precision"]
                    ),
                    "Recall Macro": percentage(
                        with_metrics["recall"]
                    ),
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

    if comparison_frame["Accuracy (%)"].notna().any():
        figure = px.bar(
            comparison_frame,
            x="Model",
            y="Accuracy (%)",
            color="Kondisi",
            barmode="group",
            text="Accuracy (%)",
            title=(
                "Accuracy Model Tanpa dan Dengan "
                "Penambahan Keyword YAKE"
            ),
        )
        add_bar_labels(figure)
        figure.update_yaxes(range=[93, 97])

        st.plotly_chart(figure, use_container_width=True)

    yake_metric_1, yake_metric_2 = st.columns(2)

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

    with yake_metric_1:
        with st.container(border=True):
            st.subheader("CNN")

            st.metric(
                "Tanpa YAKE — K2",
                percentage(cnn_without),
            )
            st.metric(
                "Dengan YAKE — K3",
                percentage(cnn_with),
                delta=(
                    f"{(cnn_with - cnn_without) * 100:+.2f} pp"
                    if cnn_without is not None and cnn_with is not None
                    else None
                ),
                delta_color="inverse",
            )

            if cnn_without is not None and cnn_with is not None:
                st.caption(
                    f"Prediksi salah berubah dari "
                    f"{round(KOMPAS_TEST_SIZE * (1 - cnn_without))} "
                    f"menjadi "
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
                    f"{(attention_with - attention_without) * 100:+.2f} pp"
                    if attention_without is not None
                    and attention_with is not None
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

    st.subheader("Tabel Perbandingan Lengkap")

    st.dataframe(
        detailed_frame,
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "**Interpretasi:** penambahan keyword YAKE belum meningkatkan "
        "performa pada konfigurasi penelitian ini. Pada CNN, accuracy "
        "berubah dari 95,80% menjadi 95,00%. Pada Attention-BiLSTM, "
        "accuracy berubah dari 95,30% menjadi 95,20%. Keyword YAKE "
        "diekstraksi dari Title dan Description sehingga sebagian "
        "informasinya kemungkinan sudah terdapat pada teks utama dan "
        "menimbulkan redundansi."
    )

    with st.expander("Lihat data CSV analisis YAKE"):
        if yake_contribution.empty:
            st.info("Data analisis YAKE belum tersedia.")
        else:
            st.dataframe(
                clean_result_table(yake_contribution),
                use_container_width=True,
                hide_index=True,
            )


# =============================================================================
# TAB 5 - PREDIKSI BERITA
# =============================================================================

with tab_prediction:
    st.header("Prediksi Kategori Berita")

    st.info(
        "Masukkan Title dan Description. Input akan diproses menggunakan "
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
                with st.spinner(
                    "Menjalankan CNN dan Attention-BiLSTM..."
                ):
                    prediction_result = predict_news(
                        title=title_input,
                        description=description_input,
                    )

                recommendation = prediction_result[
                    "recommended_prediction"
                ]
                recommended_label = CATEGORY_DISPLAY.get(
                    recommendation["predicted_label"],
                    recommendation["predicted_label"].title(),
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
                attention_result = prediction_result[
                    "attention_bilstm"
                ]

                model_1, model_2 = st.columns(2)

                for column, model_name, model_result in [
                    (model_1, "CNN K2", cnn_result),
                    (
                        model_2,
                        "Attention-BiLSTM K2",
                        attention_result,
                    ),
                ]:
                    with column:
                        with st.container(border=True):
                            predicted_label = CATEGORY_DISPLAY.get(
                                model_result["predicted_label"],
                                model_result[
                                    "predicted_label"
                                ].title(),
                            )

                            st.subheader(model_name)
                            st.metric("Kategori", predicted_label)
                            st.metric(
                                "Confidence",
                                percentage(
                                    model_result["confidence"]
                                ),
                            )
                            st.caption(
                                f"Waktu inferensi: "
                                f"{model_result['inference_time_ms']:.2f} ms"
                            )

                            probability_frame = pd.DataFrame(
                                {
                                    "Kategori": [
                                        CATEGORY_DISPLAY.get(
                                            label,
                                            label.title(),
                                        )
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
                            )
                            add_bar_labels(figure)
                            figure.update_yaxes(range=[0, 105])

                            st.plotly_chart(
                                figure,
                                use_container_width=True,
                            )

                if prediction_result["model_agreement"]:
                    st.success(
                        "CNN dan Attention-BiLSTM memberikan kategori "
                        "yang sama."
                    )
                else:
                    st.warning(
                        "Kedua model memberikan kategori berbeda. "
                        "Rekomendasi sistem mengikuti CNN K2 karena "
                        "memiliki performa test set terbaik."
                    )

                st.caption(
                    "Confidence menunjukkan keyakinan relatif model dan "
                    "bukan jaminan bahwa prediksi selalu benar."
                )

            except Exception as error:
                st.error(f"Prediksi gagal: {error}")


# =============================================================================
# TAB 6 - EXPLAINABLE AI
# =============================================================================

with tab_shap:
    st.header("Explainable AI Menggunakan SHAP")

    st.info(
        "SHAP diterapkan pada CNN K2 sebagai model terbaik. "
        "Fitur yang dianalisis adalah token atau kata pada "
        "Title dan Description."
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
            st.info("Data global SHAP belum tersedia.")
        else:
            token_column = find_column(global_shap, ["token", "word"])
            importance_column = find_column(
                global_shap,
                ["total_abs_shap", "mean_abs_shap", "importance"],
            )

            top_n = st.slider(
                "Jumlah token yang ditampilkan",
                10,
                50,
                20,
                5,
            )

            if token_column is not None and importance_column is not None:
                top_tokens = (
                    global_shap
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
                    title=(
                        f"{top_n} Token Paling Berpengaruh "
                        "Secara Global"
                    ),
                )

                st.plotly_chart(figure, use_container_width=True)

                st.caption(
                    "Semakin besar nilai absolut SHAP, semakin kuat "
                    "pengaruh token terhadap keputusan model. "
                    "Frekuensi kata dan SHAP importance memiliki "
                    "makna yang berbeda."
                )

                st.dataframe(
                    remove_technical_columns(top_tokens),
                    use_container_width=True,
                    hide_index=True,
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
                class_column is not None
                and token_column is not None
                and importance_column is not None
            ):
                classes = sorted(
                    global_shap_by_class[
                        class_column
                    ].dropna().astype(str).unique()
                )

                selected_class = st.selectbox(
                    "Pilih kelas",
                    classes,
                    key="shap_class",
                )

                class_data = (
                    global_shap_by_class[
                        global_shap_by_class[
                            class_column
                        ].astype(str).eq(selected_class)
                    ]
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
                    title=f"Token Penting untuk Kelas {selected_class}",
                )

                st.plotly_chart(figure, use_container_width=True)

                st.caption(
                    "Grafik menunjukkan token yang paling berpengaruh "
                    "terhadap masing-masing kategori berita."
                )

    with shap_3:
        if local_shap_summary.empty:
            st.info("Data local SHAP belum tersedia.")
        else:
            st.dataframe(
                remove_technical_columns(local_shap_summary),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Local SHAP menjelaskan satu artikel tertentu. "
                "Sampel benar menunjukkan alasan keputusan yang tepat, "
                "sedangkan sampel salah membantu menganalisis penyebab "
                "kesalahan model."
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
                    local_shap_summary[
                        sample_column
                    ].astype(str).tolist(),
                    key="local_sample",
                )

                sample_data = local_token_contributions[
                    local_token_contributions[
                        contribution_sample_column
                    ].astype(str).eq(str(selected_sample))
                ].copy()

                token_column = find_column(
                    sample_data,
                    ["token", "word"],
                )
                contribution_column = find_column(
                    sample_data,
                    [
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
                        sample_data,
                        x=contribution_column,
                        y=token_column,
                        orientation="h",
                        title=(
                            f"Kontribusi Token Sampel "
                            f"{selected_sample}"
                        ),
                    )

                    st.plotly_chart(
                        figure,
                        use_container_width=True,
                    )

    with shap_4:
        waterfall_directory = (
            FIGURES_DIR / "shap" / "waterfall"
        )
        waterfall_files = (
            sorted(waterfall_directory.glob("*.png"))
            if waterfall_directory.exists()
            else []
        )

        if not waterfall_files:
            st.info("Grafik waterfall belum tersedia.")
        else:
            selected_waterfall = st.selectbox(
                "Pilih sampel waterfall",
                waterfall_files,
                format_func=lambda path: path.stem,
                key="waterfall_sample",
            )

            st.image(
                str(selected_waterfall),
                use_container_width=True,
            )

            st.caption(
                "Kontribusi positif mendorong model menuju kelas "
                "prediksi, sedangkan kontribusi negatif mengurangi "
                "dukungan terhadap kelas tersebut."
            )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Dashboard penelitian klasifikasi berita menggunakan CNN, "
    "Attention-BiLSTM, representasi Title + Description, "
    "pengujian keyword YAKE, dan Explainable AI SHAP."
)
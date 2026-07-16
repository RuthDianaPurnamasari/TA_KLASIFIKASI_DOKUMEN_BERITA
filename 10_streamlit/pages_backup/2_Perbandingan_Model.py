# =============================================================================
# PAGE 2 - PERBANDINGAN MODEL
# =============================================================================
# Fungsi halaman:
# 1. Membandingkan CNN dan Attention-BiLSTM.
# 2. Membandingkan performa seluruh skenario eksperimen.
# 3. Menganalisis kontribusi Description.
# 4. Menganalisis kontribusi keyword YAKE.
# 5. Menampilkan model terbaik berdasarkan hasil penelitian.
#
# Halaman ini TIDAK melakukan training ulang.
# Seluruh data berasal dari hasil evaluasi yang sudah dihasilkan sebelumnya.
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pandas as pd
import streamlit as st


# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================

CURRENT_FILE = Path(__file__).resolve()

# File berada di:
# project_root/10_streamlit/pages/2_Perbandingan_Model.py
#
# parents[1]:
# pages -> 10_streamlit
STREAMLIT_DIR = CURRENT_FILE.parents[1]

if str(STREAMLIT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(STREAMLIT_DIR),
    )


# =============================================================================
# IMPORT DASHBOARD MODULES
# =============================================================================

from utils.data_loader import (
    get_comparative_figure_path,
    load_best_model_summary,
    load_description_contribution,
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
    page_title="Perbandingan Model",
    page_icon="⚖️",
    layout="wide",
)


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown(
    """
<style>

/* -------------------------------------------------------------------------- */
/* MAIN PAGE                                                                  */
/* -------------------------------------------------------------------------- */

.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
}


/* -------------------------------------------------------------------------- */
/* GENERAL CARDS                                                              */
/* -------------------------------------------------------------------------- */

.comparison-card {
    background: linear-gradient(
        135deg,
        rgba(244, 248, 255, 0.98),
        rgba(255, 255, 255, 0.98)
    );

    border: 1px solid rgba(73, 105, 160, 0.16);
    border-radius: 18px;

    padding: 22px 24px;
    margin-top: 8px;
    margin-bottom: 20px;

    box-shadow:
        0 6px 20px
        rgba(30, 55, 90, 0.06);
}

.comparison-card h3 {
    margin-top: 0;
    margin-bottom: 10px;
}

.comparison-card p {
    margin-bottom: 0;
    line-height: 1.7;
}


/* -------------------------------------------------------------------------- */
/* MODEL CARDS                                                                */
/* -------------------------------------------------------------------------- */

.model-card {
    background: linear-gradient(
        145deg,
        rgba(248, 250, 255, 0.98),
        rgba(255, 255, 255, 0.98)
    );

    border: 1px solid rgba(80, 105, 150, 0.16);
    border-radius: 18px;

    padding: 24px;

    min-height: 300px;

    box-shadow:
        0 5px 18px
        rgba(30, 55, 90, 0.05);
}

.model-card h2 {
    margin-top: 0;
    margin-bottom: 14px;
}

.model-card p {
    line-height: 1.65;
}

.model-card ul {
    padding-left: 22px;
    margin-bottom: 0;
}

.model-card li {
    margin-bottom: 8px;
}


/* -------------------------------------------------------------------------- */
/* FINDING CARDS                                                              */
/* -------------------------------------------------------------------------- */

.finding-positive {
    background: linear-gradient(
        135deg,
        rgba(226, 248, 236, 0.98),
        rgba(247, 255, 250, 0.98)
    );

    border: 1px solid rgba(22, 160, 93, 0.18);
    border-left: 6px solid #16a05d;
    border-radius: 16px;

    padding: 20px;

    min-height: 220px;
}

.finding-information {
    background: linear-gradient(
        135deg,
        rgba(231, 241, 255, 0.98),
        rgba(248, 251, 255, 0.98)
    );

    border: 1px solid rgba(54, 118, 209, 0.18);
    border-left: 6px solid #3676d1;
    border-radius: 16px;

    padding: 20px;

    min-height: 220px;
}

.finding-warning {
    background: linear-gradient(
        135deg,
        rgba(255, 248, 220, 0.98),
        rgba(255, 253, 245, 0.98)
    );

    border: 1px solid rgba(213, 154, 0, 0.18);
    border-left: 6px solid #d59a00;
    border-radius: 16px;

    padding: 20px;

    min-height: 220px;
}

.finding-positive h4,
.finding-information h4,
.finding-warning h4 {
    margin-top: 0;
    margin-bottom: 12px;
}

.finding-positive p,
.finding-information p,
.finding-warning p {
    line-height: 1.6;
}


/* -------------------------------------------------------------------------- */
/* FINAL WINNER CARD                                                          */
/* -------------------------------------------------------------------------- */

.winner-card {
    background: linear-gradient(
        135deg,
        rgba(225, 248, 235, 0.98),
        rgba(246, 255, 250, 0.98)
    );

    border: 1px solid rgba(22, 160, 93, 0.22);
    border-left: 7px solid #16a05d;
    border-radius: 20px;

    padding: 26px 28px;

    box-shadow:
        0 8px 24px
        rgba(22, 120, 75, 0.08);
}

.winner-card h2 {
    margin-top: 0;
    margin-bottom: 15px;
}

.winner-card p {
    line-height: 1.7;
}

.winner-card ul {
    padding-left: 22px;
    margin-bottom: 0;
}

.winner-card li {
    margin-bottom: 8px;
}


/* -------------------------------------------------------------------------- */
/* METRIC CARDS                                                               */
/* -------------------------------------------------------------------------- */

div[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        rgba(248, 250, 253, 0.98),
        rgba(255, 255, 255, 0.98)
    );

    border: 1px solid rgba(90, 110, 145, 0.14);
    border-radius: 16px;

    padding: 16px 18px;

    box-shadow:
        0 4px 14px
        rgba(30, 55, 90, 0.04);
}


/* -------------------------------------------------------------------------- */
/* DATAFRAME                                                                  */
/* -------------------------------------------------------------------------- */

div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}


/* -------------------------------------------------------------------------- */
/* TABS                                                                       */
/* -------------------------------------------------------------------------- */

button[data-baseweb="tab"] {
    font-weight: 600;
}


/* -------------------------------------------------------------------------- */
/* SECTION SPACING                                                            */
/* -------------------------------------------------------------------------- */

.section-description {
    color: #6f7785;
    font-size: 0.95rem;
    margin-bottom: 14px;
}

</style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# HELPER FUNCTION - HTML
# =============================================================================

def render_html(
    html_content: str,
) -> None:
    """
    Menampilkan HTML dengan benar.

    dedent() digunakan agar indentasi Python
    tidak dianggap sebagai code block oleh Streamlit.
    """

    cleaned_html = dedent(
        html_content
    ).strip()

    st.markdown(
        cleaned_html,
        unsafe_allow_html=True,
    )


# =============================================================================
# HELPER FUNCTION - COLUMN SEARCH
# =============================================================================

def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """
    Mencari nama kolom berdasarkan beberapa kemungkinan nama.
    """

    if dataframe.empty:
        return None

    normalized_columns = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:

        normalized_candidate = (
            str(candidate)
            .strip()
            .lower()
        )

        if normalized_candidate in normalized_columns:

            return normalized_columns[
                normalized_candidate
            ]

    return None


# =============================================================================
# HELPER FUNCTION - METRIC NORMALIZATION
# =============================================================================

def normalize_metric_value(
    value: object,
) -> float | None:
    """
    Menormalkan nilai metrik ke skala 0 sampai 1.

    Contoh:
    0.958  -> 0.958
    95.8   -> 0.958
    """

    try:

        numeric_value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return None

    if numeric_value > 1:

        numeric_value = (
            numeric_value / 100
        )

    return numeric_value


# =============================================================================
# HELPER FUNCTION - PERCENTAGE
# =============================================================================

def display_percentage(
    value: object,
) -> str:
    """
    Menampilkan nilai metrik dalam bentuk persentase.
    """

    normalized_value = (
        normalize_metric_value(
            value
        )
    )

    if normalized_value is None:

        return "-"

    return format_percentage(
        normalized_value
    )


# =============================================================================
# HELPER FUNCTION - CLEAN TABLE
# =============================================================================

def clean_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Menghapus kolom teknis yang tidak perlu ditampilkan
    pada dashboard.
    """

    if dataframe.empty:

        return dataframe

    technical_keywords = [
        "path",
        "checkpoint",
        "directory",
        "folder",
    ]

    selected_columns = []

    for column in dataframe.columns:

        normalized_column = (
            str(column)
            .strip()
            .lower()
        )

        is_technical = any(
            keyword in normalized_column
            for keyword in technical_keywords
        )

        if not is_technical:

            selected_columns.append(
                column
            )

    return dataframe[
        selected_columns
    ].copy()


# =============================================================================
# HELPER FUNCTION - FORMAT TABLE
# =============================================================================

def format_metric_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Memformat kolom metrik sebagai persentase.

    Kolom perubahan seperti accuracy_change_percentage_point
    tidak diformat sebagai persentase skala 0-1 karena nilainya
    sudah dalam satuan percentage point.
    """

    if dataframe.empty:

        return dataframe

    formatted = clean_table(
        dataframe
    )

    metric_keywords = [
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]

    excluded_keywords = [
        "change",
        "percentage_point",
        "error",
        "count",
    ]

    for column in formatted.columns:

        normalized_column = (
            str(column)
            .strip()
            .lower()
        )

        is_metric_column = any(
            keyword in normalized_column
            for keyword in metric_keywords
        )

        is_excluded_column = any(
            keyword in normalized_column
            for keyword in excluded_keywords
        )

        if (
            is_metric_column
            and not is_excluded_column
        ):

            formatted[column] = (
                formatted[column]
                .apply(
                    lambda value:
                    display_percentage(value)
                    if pd.notna(value)
                    else "-"
                )
            )

    return formatted


# =============================================================================
# HELPER FUNCTION - SHOW FIGURE
# =============================================================================

def show_figure(
    figure_path: Path,
    caption: str,
) -> None:
    """
    Menampilkan grafik jika tersedia.
    """

    if figure_path.exists():

        st.image(
            str(figure_path),
            caption=caption,
            use_container_width=True,
        )

    else:

        show_missing_figure_message(
            figure_name=caption,
            figure_path=figure_path,
        )


# =============================================================================
# LOAD RESEARCH RESULTS
# =============================================================================

evaluation_data = (
    load_test_evaluation()
)

model_comparison = (
    load_model_comparison()
)

scenario_comparison = (
    load_scenario_comparison()
)

description_contribution = (
    load_description_contribution()
)

yake_contribution = (
    load_yake_contribution()
)

best_model_summary = (
    load_best_model_summary()
)


# =============================================================================
# PAGE HEADER
# =============================================================================

page_header(
    title="⚖️ Perbandingan Model",
    subtitle=(
        "Analisis komparatif CNN dan Attention-BiLSTM "
        "pada berbagai representasi teks dan dataset."
    ),
)


# =============================================================================
# RESEARCH QUESTION
# =============================================================================

render_html(
    """
    <div class="comparison-card">
        <h3>🎯 Fokus Analisis</h3>

        <p>
            Bagaimana perbandingan kinerja
            <b>Convolutional Neural Network (CNN)</b> dan
            <b>Attention-BiLSTM</b>
            dalam melakukan klasifikasi berita,
            serta bagaimana perubahan representasi teks
            memengaruhi performa kedua model?
        </p>
    </div>
    """
)


# =============================================================================
# FINAL RESULT SUMMARY
# =============================================================================

st.markdown(
    "## Ringkasan Perbandingan"
)

st.caption(
    "Ringkasan konfigurasi terbaik berdasarkan hasil "
    "10 eksperimen utama."
)

summary_column_1, \
summary_column_2, \
summary_column_3, \
summary_column_4 = st.columns(
    4
)


with summary_column_1:

    st.metric(
        label="Model Terbaik",
        value="CNN K2",
    )


with summary_column_2:

    st.metric(
        label="Accuracy Terbaik",
        value="95.80%",
    )


with summary_column_3:

    st.metric(
        label="Macro F1 Terbaik",
        value="95.81%",
    )


with summary_column_4:

    st.metric(
        label="Representasi Terbaik",
        value="Title + Description",
    )


# =============================================================================
# CNN VS ATTENTION-BILSTM
# =============================================================================

st.divider()

st.markdown(
    "## CNN vs Attention-BiLSTM"
)

st.caption(
    "Perbandingan karakteristik kedua arsitektur "
    "berdasarkan penggunaannya dalam penelitian."
)


model_column_1, \
model_column_2 = st.columns(
    2
)


with model_column_1:

    render_html(
        """
        <div class="model-card">
            <h2>🧩 CNN</h2>

            <p>
                CNN mempelajari pola lokal pada teks
                melalui operasi convolution sehingga dapat
                mengenali kombinasi kata atau frasa yang
                relevan terhadap kategori berita.
            </p>

            <b>Kekuatan pada penelitian:</b>

            <ul>
                <li>Efektif menangkap pola lokal dan frasa penting.</li>
                <li>Memperoleh performa tinggi pada dataset Kompas.</li>
                <li>Menghasilkan konfigurasi terbaik pada skenario K2.</li>
                <li>Dipilih sebagai model utama penelitian.</li>
            </ul>
        </div>
        """
    )


with model_column_2:

    render_html(
        """
        <div class="model-card">
            <h2>🔁 Attention-BiLSTM</h2>

            <p>
                Attention-BiLSTM mempelajari urutan teks
                dari dua arah dan menggunakan mekanisme
                attention untuk memberikan perhatian lebih
                pada bagian teks yang relevan.
            </p>

            <b>Kekuatan pada penelitian:</b>

            <ul>
                <li>Mempelajari konteks teks dari dua arah.</li>
                <li>Menggunakan mekanisme attention.</li>
                <li>Menghasilkan performa yang kompetitif.</li>
                <li>Sedikit unggul dibandingkan CNN pada skenario K3.</li>
            </ul>
        </div>
        """
    )


# =============================================================================
# MAIN PERFORMANCE COMPARISON
# =============================================================================

st.divider()

st.markdown(
    "## Perbandingan Performa Seluruh Eksperimen"
)

st.caption(
    "Grafik berikut memperlihatkan performa CNN dan "
    "Attention-BiLSTM pada seluruh skenario eksperimen."
)


performance_tab_1, \
performance_tab_2 = st.tabs(
    [
        "📈 Accuracy",
        "📊 Macro F1",
    ]
)


# =============================================================================
# ACCURACY TAB
# =============================================================================

with performance_tab_1:

    accuracy_figure = (
        get_comparative_figure_path(
            "accuracy_comparison.png"
        )
    )

    show_figure(
        accuracy_figure,
        "Perbandingan Accuracy Seluruh Eksperimen",
    )

    st.info(
        "CNN memperoleh accuracy lebih tinggi pada "
        "Kompas K1, Kompas K2, AG News A1, dan AG News A2. "
        "Attention-BiLSTM sedikit lebih tinggi pada Kompas K3."
    )


# =============================================================================
# MACRO F1 TAB
# =============================================================================

with performance_tab_2:

    f1_figure = (
        get_comparative_figure_path(
            "f1_macro_comparison.png"
        )
    )

    show_figure(
        f1_figure,
        "Perbandingan Macro F1 Seluruh Eksperimen",
    )

    st.info(
        "Macro F1 digunakan untuk melihat keseimbangan "
        "performa model pada seluruh kelas."
    )


# =============================================================================
# INTERPRETATION OF MODEL COMPARISON
# =============================================================================

st.markdown(
    "### Interpretasi Perbandingan"
)


finding_column_1, \
finding_column_2, \
finding_column_3 = st.columns(
    3
)


with finding_column_1:

    render_html(
        """
        <div class="finding-positive">
            <h4>🏆 CNN K2 Terbaik</h4>

            <p>
                CNN K2 menghasilkan performa terbaik
                pada dataset utama Kompas.
            </p>

            <b>Accuracy: 95,80%</b>
            <br>
            <b>Macro F1: 95,81%</b>
        </div>
        """
    )


with finding_column_2:

    render_html(
        """
        <div class="finding-information">
            <h4>🔄 Performa Kompetitif</h4>

            <p>
                Kedua model menghasilkan performa tinggi.
                Namun, CNN unggul pada sebagian besar
                konfigurasi eksperimen yang diuji.
            </p>

            <b>
                Attention-BiLSTM tetap menunjukkan
                performa yang kompetitif.
            </b>
        </div>
        """
    )


with finding_column_3:

    render_html(
        """
        <div class="finding-warning">
            <h4>🧪 Hasil K3 Berbeda</h4>

            <p>
                Pada skenario K3, Attention-BiLSTM
                memperoleh accuracy 95,20%,
                sedikit lebih tinggi dibandingkan
                CNN sebesar 95,00%.
            </p>

            <b>
                Namun, K3 bukan konfigurasi terbaik
                secara keseluruhan.
            </b>
        </div>
        """
    )


# =============================================================================
# TEXT REPRESENTATION ANALYSIS
# =============================================================================

st.divider()

st.markdown(
    "## Analisis Representasi Teks"
)

st.caption(
    "Bagian ini menganalisis pengaruh penambahan "
    "Description dan keyword YAKE terhadap performa model."
)


representation_tab_1, \
representation_tab_2 = st.tabs(
    [
        "📝 Kontribusi Description",
        "🔑 Kontribusi Keyword YAKE",
    ]
)


# =============================================================================
# DESCRIPTION CONTRIBUTION
# =============================================================================

with representation_tab_1:

    description_figure = (
        get_comparative_figure_path(
            "description_contribution.png"
        )
    )

    show_figure(
        description_figure,
        "Kontribusi Description terhadap Accuracy",
    )

    st.success(
        "Penambahan Description meningkatkan accuracy "
        "pada CNN dan Attention-BiLSTM, baik pada dataset "
        "Kompas maupun AG News."
    )


    description_column_1, \
    description_column_2 = st.columns(
        2
    )


    with description_column_1:

        st.markdown(
            """
            ### 🇮🇩 Dataset Kompas

            **CNN**

            `K1 — Title`  
            **94,70%**

            ⬇️ ditambahkan Description

            `K2 — Title + Description`  
            **95,80%**

            **Peningkatan: +1,10 percentage point**
            """
        )


    with description_column_2:

        st.markdown(
            """
            ### 🌐 Dataset AG News

            **CNN**

            `A1 — Title`  
            **86,32%**

            ⬇️ ditambahkan Description

            `A2 — Title + Description`  
            **88,76%**

            **Peningkatan: +2,44 percentage point**
            """
        )


    st.markdown(
        "### Perubahan pada Kedua Model"
    )


    description_metric_1, \
    description_metric_2, \
    description_metric_3, \
    description_metric_4 = st.columns(
        4
    )


    with description_metric_1:

        st.metric(
            "Kompas CNN",
            "+1.10 pp",
        )


    with description_metric_2:

        st.metric(
            "Kompas Attention-BiLSTM",
            "+1.60 pp",
        )


    with description_metric_3:

        st.metric(
            "AG News CNN",
            "+2.44 pp",
        )


    with description_metric_4:

        st.metric(
            "AG News Attention-BiLSTM",
            "+3.09 pp",
        )


# =============================================================================
# YAKE CONTRIBUTION
# =============================================================================

with representation_tab_2:

    yake_figure = (
        get_comparative_figure_path(
            "yake_contribution.png"
        )
    )

    show_figure(
        yake_figure,
        "Kontribusi Keyword YAKE terhadap Accuracy",
    )

    st.warning(
        "Pada konfigurasi eksperimen penelitian ini, "
        "penambahan keyword hasil ekstraksi YAKE belum "
        "meningkatkan accuracy dibandingkan representasi "
        "Title + Description."
    )


    yake_column_1, \
    yake_column_2 = st.columns(
        2
    )


    with yake_column_1:

        st.markdown(
            """
            ### 🧩 CNN

            **K2 — Title + Description**

            **95,80%**

            ⬇️ ditambahkan keyword YAKE

            **K3 — Title + Description + Keyword**

            **95,00%**

            **Perubahan: -0,80 percentage point**
            """
        )


    with yake_column_2:

        st.markdown(
            """
            ### 🔁 Attention-BiLSTM

            **K2 — Title + Description**

            **95,30%**

            ⬇️ ditambahkan keyword YAKE

            **K3 — Title + Description + Keyword**

            **95,20%**

            **Perubahan: -0,10 percentage point**
            """
        )


    st.info(
        "Hasil ini tidak berarti YAKE merupakan metode yang buruk. "
        "Hasil menunjukkan bahwa pada dataset, preprocessing, model, "
        "dan konfigurasi penelitian ini, penambahan keyword YAKE "
        "belum memberikan peningkatan performa dibandingkan "
        "Title + Description."
    )


# =============================================================================
# FINAL MODEL RECOMMENDATION
# =============================================================================

st.divider()

st.markdown(
    "## Rekomendasi Model Final"
)


render_html(
    """
    <div class="winner-card">
        <h2>🏆 CNN K2 — Model Terbaik Penelitian</h2>

        <p>
            Berdasarkan hasil evaluasi terhadap seluruh eksperimen,
            model <b>Convolutional Neural Network (CNN)</b>
            dengan representasi
            <b>Title + Description</b>
            dipilih sebagai model utama penelitian.
        </p>

        <p>
            Model ini memperoleh
            <b>accuracy 95,80%</b>
            dan
            <b>Macro F1 95,81%</b>
            pada test set dataset Kompas.
        </p>

        <p>
            CNN K2 selanjutnya digunakan untuk:
        </p>

        <ul>
            <li>Implementasi klasifikasi berita baru pada dashboard.</li>
            <li>Model utama untuk hasil rekomendasi sistem.</li>
            <li>Analisis Explainable AI menggunakan SHAP.</li>
            <li>Interpretasi kontribusi token terhadap prediksi model.</li>
        </ul>
    </div>
    """
)


# =============================================================================
# SUPPORTING TABLES
# =============================================================================

st.divider()

st.markdown(
    "## Data Pendukung Analisis"
)

st.caption(
    "Tabel berikut berasal dari hasil evaluasi eksperimen. "
    "Kolom path dan informasi file internal tidak ditampilkan."
)


table_tab_1, \
table_tab_2, \
table_tab_3, \
table_tab_4 = st.tabs(
    [
        "Perbandingan Model",
        "Perbandingan Skenario",
        "Kontribusi Description",
        "Kontribusi YAKE",
    ]
)


# =============================================================================
# MODEL COMPARISON TABLE
# =============================================================================

with table_tab_1:

    if model_comparison.empty:

        st.info(
            "Data perbandingan model belum tersedia."
        )

    else:

        st.dataframe(
            format_metric_table(
                model_comparison
            ),
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# SCENARIO COMPARISON TABLE
# =============================================================================

with table_tab_2:

    if scenario_comparison.empty:

        st.info(
            "Data perbandingan skenario belum tersedia."
        )

    else:

        st.dataframe(
            format_metric_table(
                scenario_comparison
            ),
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# DESCRIPTION CONTRIBUTION TABLE
# =============================================================================

with table_tab_3:

    if description_contribution.empty:

        st.info(
            "Data kontribusi Description belum tersedia."
        )

    else:

        st.dataframe(
            format_metric_table(
                description_contribution
            ),
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# YAKE CONTRIBUTION TABLE
# =============================================================================

with table_tab_4:

    if yake_contribution.empty:

        st.info(
            "Data kontribusi YAKE belum tersedia."
        )

    else:

        st.dataframe(
            format_metric_table(
                yake_contribution
            ),
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# FINAL CONCLUSION
# =============================================================================

st.divider()

st.markdown(
    "## Kesimpulan Perbandingan"
)


st.success(
    """
    Berdasarkan 10 eksperimen utama, CNN K2 menjadi konfigurasi
    terbaik pada dataset utama Kompas dengan accuracy 95,80%
    dan Macro F1 95,81%. Penambahan Description meningkatkan
    performa CNN dan Attention-BiLSTM pada dataset Kompas dan
    AG News. Sementara itu, penambahan keyword hasil ekstraksi
    YAKE pada skenario K3 belum meningkatkan accuracy dibandingkan
    representasi Title + Description. Oleh karena itu, CNN K2
    dipilih sebagai model utama untuk implementasi klasifikasi
    berita dan analisis Explainable AI menggunakan SHAP.
    """
)


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Analisis perbandingan CNN dan Attention-BiLSTM "
    "berdasarkan hasil eksperimen penelitian."
)
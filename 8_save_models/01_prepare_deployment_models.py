"""
STEP 8.1 - PREPARE DEPLOYMENT MODELS

Menyiapkan seluruh artefak yang dibutuhkan untuk implementasi
sistem klasifikasi berita pada dashboard Streamlit.

Model deployment:
1. CNN K2
2. Attention-BiLSTM K2

Representasi teks:
Title + Description
"""

from pathlib import Path
import json
import shutil
import sys

import tensorflow as tf

# ============================================================
# IMPORT CUSTOM MODEL COMPONENTS
# ============================================================

MODELING_DIR = Path(__file__).resolve().parents[1] / "5_modeling"

if str(MODELING_DIR) not in sys.path:
    sys.path.insert(0, str(MODELING_DIR))

from attention_bilstm_model import AttentionPooling1D

# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "8_save_models"
    / "checkpoints"
)

FINAL_MODEL_DIR = (
    PROJECT_ROOT
    / "8_save_models"
    / "final_models"
)

DEPLOYMENT_DIR = (
    PROJECT_ROOT
    / "8_save_models"
    / "deployment"
)

VECTORIZED_DIR = (
    PROJECT_ROOT
    / "2_data"
    / "vectorized"
    / "kompas_k2"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "9_results"
    / "tables"
)

DEPLOYMENT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEPLOYMENT CONFIGURATION
# ============================================================

MODEL_NAMES = [
    "cnn_k2",
    "attention_bilstm_k2",
]

DEPLOYMENT_CONFIG = {
    "dataset": "Kompas",
    "scenario_code": "K2",
    "scenario_name": "Title + Description",
    "sequence_length": 60,
    "num_classes": 4,

    "models": {
        "cnn": {
            "experiment_name": "cnn_k2",
            "display_name": "CNN",
        },
        "attention_bilstm": {
            "experiment_name": "attention_bilstm_k2",
            "display_name": "Attention-BiLSTM",
        },
    },

    "best_research_model": {
        "experiment_name": "cnn_k2",
        "model_name": "CNN",
        "accuracy": 0.958,
    },

    "input": {
        "required_fields": [
            "title",
            "description",
        ],
        "text_combination": "title + [SEP] + description",
    },

    "labels": {
        "0": "bola",
        "1": "global",
        "2": "money",
        "3": "tekno",
    },
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(title):
    print("=" * 80)
    print(title)
    print("=" * 80)


def find_model_file(experiment_name):
    """
    Mencari model terbaik.

    Prioritas:
    1. checkpoint
    2. final model
    """

    search_directories = [
        CHECKPOINT_DIR,
        FINAL_MODEL_DIR,
    ]

    extensions = [
        ".keras",
        ".h5",
    ]

    candidate_names = [
        experiment_name,
        f"{experiment_name}_best",
        f"{experiment_name}_final",
        f"best_{experiment_name}",
    ]

    for directory in search_directories:

        if not directory.exists():
            continue

        # Exact candidate names
        for candidate_name in candidate_names:

            for extension in extensions:

                candidate_path = (
                    directory
                    / f"{candidate_name}{extension}"
                )

                if candidate_path.exists():
                    return candidate_path

        # Flexible search
        for extension in extensions:

            matches = list(
                directory.glob(
                    f"*{experiment_name}*{extension}"
                )
            )

            if matches:
                return matches[0]

    return None


def copy_file(
    source_path,
    destination_path,
):
    """
    Menyalin file deployment.
    """

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source_path,
        destination_path,
    )


# ============================================================
# PREPARE MODELS
# ============================================================

def prepare_models():

    print("\nMenyiapkan model deployment...")

    results = []

    for experiment_name in MODEL_NAMES:

        print("\n" + "-" * 80)
        print(
            f"Model: {experiment_name}"
        )

        source_model = find_model_file(
            experiment_name
        )

        if source_model is None:

            print(
                "Status : MODEL TIDAK DITEMUKAN"
            )

            results.append({
                "experiment_name":
                    experiment_name,

                "status":
                    "not_found",
            })

            continue

        print(
            f"Sumber : {source_model}"
        )

        destination_model = (
            DEPLOYMENT_DIR
            / f"{experiment_name}.keras"
        )

        print(
            "Validasi model..."
        )

        try:

            model = tf.keras.models.load_model(
                source_model,
                compile=False,
                custom_objects={
                    "AttentionPooling1D": AttentionPooling1D,
                    "TAKlasifikasiBerita>AttentionPooling1D": AttentionPooling1D,
                },
            )

            print(
                f"Input  : {model.input_shape}"
            )

            print(
                f"Output : {model.output_shape}"
            )

        except Exception as error:

            print(
                f"Gagal memuat model: {error}"
            )

            results.append({
                "experiment_name":
                    experiment_name,

                "status":
                    "load_failed",

                "error":
                    str(error),
            })

            continue

        copy_file(
            source_model,
            destination_model,
        )

        print(
            f"Output : {destination_model}"
        )

        print(
            "Status : BERHASIL"
        )

        results.append({
            "experiment_name":
                experiment_name,

            "status":
                "success",

            "source":
                str(source_model),

            "deployment_path":
                str(destination_model),

            "input_shape":
                str(model.input_shape),

            "output_shape":
                str(model.output_shape),
        })

    return results


# ============================================================
# PREPARE VOCABULARY
# ============================================================

def prepare_vocabulary():

    print("\n" + "-" * 80)
    print(
        "Menyiapkan vocabulary K2..."
    )

    source = (
        VECTORIZED_DIR
        / "vocabulary.txt"
    )

    destination = (
        DEPLOYMENT_DIR
        / "vocabulary.txt"
    )

    if not source.exists():

        print(
            f"Vocabulary tidak ditemukan: {source}"
        )

        return False

    copy_file(
        source,
        destination,
    )

    print(
        f"Sumber : {source}"
    )

    print(
        f"Output : {destination}"
    )

    return True


# ============================================================
# PREPARE VECTORIZER CONFIG
# ============================================================

def prepare_vectorizer_config():

    print("\n" + "-" * 80)
    print(
        "Menyiapkan vectorizer configuration..."
    )

    source = (
        VECTORIZED_DIR
        / "vectorizer_config.json"
    )

    destination = (
        DEPLOYMENT_DIR
        / "vectorizer_config.json"
    )

    if not source.exists():

        print(
            f"Vectorizer config tidak ditemukan: {source}"
        )

        return False

    copy_file(
        source,
        destination,
    )

    print(
        f"Sumber : {source}"
    )

    print(
        f"Output : {destination}"
    )

    return True


# ============================================================
# PREPARE LABEL MAPPING
# ============================================================

def prepare_label_mapping():

    print("\n" + "-" * 80)
    print(
        "Menyiapkan label mapping..."
    )

    source = (
        TABLE_DIR
        / "label_mapping.json"
    )

    destination = (
        DEPLOYMENT_DIR
        / "label_mapping.json"
    )

    if source.exists():

        copy_file(
            source,
            destination,
        )

        print(
            f"Sumber : {source}"
        )

        print(
            f"Output : {destination}"
        )

        return True

    print(
        "File label mapping utama tidak ditemukan."
    )

    print(
        "Membuat label mapping deployment..."
    )

    fallback_mapping = {
        "0": "bola",
        "1": "global",
        "2": "money",
        "3": "tekno",
    }

    with open(
        destination,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            fallback_mapping,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"Output : {destination}"
    )

    return True


# ============================================================
# SAVE DEPLOYMENT CONFIG
# ============================================================

def save_deployment_config():

    print("\n" + "-" * 80)
    print(
        "Menyimpan deployment configuration..."
    )

    output_path = (
        DEPLOYMENT_DIR
        / "deployment_config.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            DEPLOYMENT_CONFIG,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"Output : {output_path}"
    )

    return output_path


# ============================================================
# SAVE DEPLOYMENT REPORT
# ============================================================

def save_deployment_report(
    model_results,
):

    report = {
        "deployment_status":
            "prepared",

        "primary_model":
            "cnn_k2",

        "comparison_model":
            "attention_bilstm_k2",

        "scenario":
            "K2",

        "representation":
            "Title + Description",

        "models":
            model_results,
    }

    output_path = (
        DEPLOYMENT_DIR
        / "deployment_report.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "STEP 8.1 - PREPARE DEPLOYMENT MODELS"
    )

    print(
        "\nKonfigurasi deployment:"
    )

    print(
        "Dataset              : Kompas"
    )

    print(
        "Skenario             : K2"
    )

    print(
        "Representasi         : Title + Description"
    )

    print(
        "Sequence length      : 60"
    )

    print(
        "Model utama          : CNN K2"
    )

    print(
        "Model pembanding     : Attention-BiLSTM K2"
    )

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    model_results = prepare_models()

    # --------------------------------------------------------
    # SUPPORTING FILES
    # --------------------------------------------------------

    vocabulary_status = (
        prepare_vocabulary()
    )

    vectorizer_status = (
        prepare_vectorizer_config()
    )

    label_status = (
        prepare_label_mapping()
    )

    deployment_config_path = (
        save_deployment_config()
    )

    deployment_report_path = (
        save_deployment_report(
            model_results
        )
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    success_models = [
        result
        for result in model_results
        if result["status"] == "success"
    ]

    print("\n")
    print_header(
        "HASIL PREPARE DEPLOYMENT"
    )

    print(
        f"\nModel berhasil       : "
        f"{len(success_models)}/{len(MODEL_NAMES)}"
    )

    print(
        f"Vocabulary           : "
        f"{'OK' if vocabulary_status else 'GAGAL'}"
    )

    print(
        f"Vectorizer config     : "
        f"{'OK' if vectorizer_status else 'GAGAL'}"
    )

    print(
        f"Label mapping         : "
        f"{'OK' if label_status else 'GAGAL'}"
    )

    print(
        "\nFolder deployment:"
    )

    print(
        DEPLOYMENT_DIR
    )

    print(
        "\nDeployment config:"
    )

    print(
        deployment_config_path
    )

    print(
        "\nDeployment report:"
    )

    print(
        deployment_report_path
    )

    print("\n")
    print_header(
        "Tahap prepare deployment selesai."
    )


if __name__ == "__main__":
    main()
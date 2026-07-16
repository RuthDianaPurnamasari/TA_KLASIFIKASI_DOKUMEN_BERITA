# =============================================================================
# STEP 5.1 - CNN MODEL ARCHITECTURE
# =============================================================================
# File:
# 5_modeling/cnn_model.py
#
# Tujuan:
# Mendefinisikan arsitektur Convolutional Neural Network (CNN)
# untuk klasifikasi berita multikelas.
#
# Arsitektur:
# Input
#   ↓
# Embedding
#   ↓
# SpatialDropout1D
#   ↓
# Conv1D
#   ↓
# GlobalMaxPooling1D
#   ↓
# Dense
#   ↓
# Dropout
#   ↓
# Output Softmax
# =============================================================================

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    SpatialDropout1D,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout
)


def build_cnn_model(
    vocabulary_size,
    max_sequence_length,
    num_classes=4,
    embedding_dim=128,
    num_filters=128,
    kernel_size=5,
    dense_units=128,
    spatial_dropout_rate=0.2,
    dropout_rate=0.5,
    learning_rate=0.001,
    model_name="CNN_Text_Classifier"
):
    """
    Membuat model CNN untuk klasifikasi teks multikelas.

    Parameters
    ----------
    vocabulary_size : int
        Jumlah vocabulary yang digunakan pada skenario.

    max_sequence_length : int
        Panjang maksimum sequence input.

    num_classes : int, default=4
        Jumlah kelas target.

    embedding_dim : int, default=128
        Dimensi representasi embedding setiap token.

    num_filters : int, default=128
        Jumlah filter pada Conv1D.

    kernel_size : int, default=5
        Ukuran jendela konvolusi.

    dense_units : int, default=128
        Jumlah neuron pada Dense layer.

    spatial_dropout_rate : float, default=0.2
        Dropout pada hasil embedding.

    dropout_rate : float, default=0.5
        Dropout sebelum output layer.

    learning_rate : float, default=0.001
        Learning rate optimizer Adam.

    model_name : str
        Nama model.

    Returns
    -------
    tensorflow.keras.Model
        Model CNN yang sudah dikompilasi.
    """

    # -------------------------------------------------------------------------
    # 1. INPUT LAYER
    # -------------------------------------------------------------------------
    inputs = Input(
        shape=(max_sequence_length,),
        dtype="int32",
        name="input_sequence"
    )

    # -------------------------------------------------------------------------
    # 2. EMBEDDING LAYER
    # -------------------------------------------------------------------------
    x = Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_dim,
        mask_zero=False,
        name="embedding"
    )(inputs)

    # -------------------------------------------------------------------------
    # 3. SPATIAL DROPOUT
    # -------------------------------------------------------------------------
    x = SpatialDropout1D(
        rate=spatial_dropout_rate,
        name="spatial_dropout"
    )(x)

    # -------------------------------------------------------------------------
    # 4. CONVOLUTIONAL LAYER
    # -------------------------------------------------------------------------
    x = Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        activation="relu",
        padding="valid",
        name="conv1d"
    )(x)

    # -------------------------------------------------------------------------
    # 5. GLOBAL MAX POOLING
    # -------------------------------------------------------------------------
    x = GlobalMaxPooling1D(
        name="global_max_pooling"
    )(x)

    # -------------------------------------------------------------------------
    # 6. DENSE LAYER
    # -------------------------------------------------------------------------
    x = Dense(
        units=dense_units,
        activation="relu",
        name="dense"
    )(x)

    # -------------------------------------------------------------------------
    # 7. DROPOUT
    # -------------------------------------------------------------------------
    x = Dropout(
        rate=dropout_rate,
        name="dropout"
    )(x)

    # -------------------------------------------------------------------------
    # 8. OUTPUT LAYER
    # -------------------------------------------------------------------------
    outputs = Dense(
        units=num_classes,
        activation="softmax",
        name="output"
    )(x)

    # -------------------------------------------------------------------------
    # 9. BUILD MODEL
    # -------------------------------------------------------------------------
    model = Model(
        inputs=inputs,
        outputs=outputs,
        name=model_name
    )

    # -------------------------------------------------------------------------
    # 10. COMPILE MODEL
    # -------------------------------------------------------------------------
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =============================================================================
# TEST MODEL
# =============================================================================
if __name__ == "__main__":

    print("=" * 72)
    print("STEP 5.1 - CNN MODEL ARCHITECTURE TEST")
    print("=" * 72)

    # Contoh konfigurasi untuk pengujian arsitektur.
    # Ini hanya untuk memastikan model berhasil dibuat,
    # bukan proses training sebenarnya.
    test_model = build_cnn_model(
        vocabulary_size=20000,
        max_sequence_length=60,
        num_classes=4
    )

    print("\nKonfigurasi pengujian:")
    print(f"Vocabulary size       : {20000:,}")
    print(f"Max sequence length   : {60}")
    print(f"Jumlah kelas          : {4}")
    print(f"Embedding dimension   : {128}")
    print(f"Jumlah filter Conv1D  : {128}")
    print(f"Kernel size           : {5}")

    print("\nModel Summary:")
    test_model.summary()

    print("\n" + "=" * 72)
    print("Arsitektur CNN berhasil dibuat.")
    print("=" * 72)
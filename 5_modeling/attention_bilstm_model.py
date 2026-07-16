# =============================================================================
# STEP 5.2 - ATTENTION-BILSTM MODEL ARCHITECTURE
# =============================================================================
# File:
# 5_modeling/attention_bilstm_model.py
#
# Tujuan:
# Mendefinisikan arsitektur Attention-BiLSTM untuk klasifikasi
# berita multikelas.
#
# Arsitektur:
# Input
#   ↓
# Embedding
#   ↓
# SpatialDropout1D
#   ↓
# Bidirectional LSTM
#   ↓
# Attention Pooling
#   ↓
# Dense
#   ↓
# Dropout
#   ↓
# Output Softmax
# =============================================================================

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    SpatialDropout1D,
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
)


# =============================================================================
# CUSTOM ATTENTION LAYER
# =============================================================================

@keras.utils.register_keras_serializable(
    package="TAKlasifikasiBerita"
)
class AttentionPooling1D(keras.layers.Layer):
    """
    Attention pooling untuk sequence hasil BiLSTM.

    Layer ini memberi skor pada setiap posisi token, kemudian
    mengubah skor tersebut menjadi bobot attention menggunakan
    softmax.

    Output akhirnya adalah satu context vector yang merupakan
    gabungan berbobot dari seluruh timestep BiLSTM.
    """

    def __init__(
        self,
        attention_units: int = 64,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.attention_units = attention_units

        # Layer Dense pertama membentuk representasi attention.
        self.projection = Dense(
            units=attention_units,
            activation="tanh",
            name="attention_projection",
        )

        # Layer Dense kedua menghasilkan satu skor per timestep.
        self.score = Dense(
            units=1,
            use_bias=False,
            name="attention_score",
        )

        # Layer ini mendukung mask dari Embedding.
        self.supports_masking = True

    def call(
        self,
        inputs: tf.Tensor,
        mask: tf.Tensor | None = None,
        training: bool | None = None,
    ) -> tf.Tensor:
        """
        Menghitung context vector berbasis attention.

        Parameters
        ----------
        inputs:
            Tensor hasil BiLSTM dengan bentuk:
            (batch_size, sequence_length, hidden_features)

        mask:
            Penanda posisi token asli dan padding.

        Returns
        -------
        Tensor dengan bentuk:
            (batch_size, hidden_features)
        """

        # -----------------------------------------------------
        # 1. REPRESENTASI ATTENTION
        # -----------------------------------------------------

        projected_inputs = self.projection(
            inputs,
            training=training,
        )

        # -----------------------------------------------------
        # 2. SKOR SETIAP TIMESTEP
        # -----------------------------------------------------

        attention_scores = self.score(
            projected_inputs,
            training=training,
        )

        # Bentuk awal:
        # (batch_size, sequence_length, 1)
        #
        # Setelah squeeze:
        # (batch_size, sequence_length)

        attention_scores = tf.squeeze(
            attention_scores,
            axis=-1,
        )

        # -----------------------------------------------------
        # 3. MENGABAIKAN POSISI PADDING
        # -----------------------------------------------------

        if mask is not None:
            mask = tf.cast(
                mask,
                dtype=attention_scores.dtype,
            )

            # Posisi padding diberi nilai negatif sangat besar
            # agar bobotnya mendekati nol setelah softmax.
            negative_value = tf.cast(
                -1e9,
                dtype=attention_scores.dtype,
            )

            attention_scores = (
                attention_scores
                + (1.0 - mask) * negative_value
            )

        # -----------------------------------------------------
        # 4. BOBOT ATTENTION
        # -----------------------------------------------------

        attention_weights = tf.nn.softmax(
            attention_scores,
            axis=1,
        )

        # Bentuk:
        # (batch_size, sequence_length, 1)

        attention_weights = tf.expand_dims(
            attention_weights,
            axis=-1,
        )

        # -----------------------------------------------------
        # 5. CONTEXT VECTOR
        # -----------------------------------------------------

        weighted_sequence = (
            inputs * attention_weights
        )

        context_vector = tf.reduce_sum(
            weighted_sequence,
            axis=1,
        )

        return context_vector

    def compute_mask(
        self,
        inputs: tf.Tensor,
        mask: tf.Tensor | None = None,
    ) -> None:
        """
        Output attention sudah tidak berupa sequence,
        sehingga mask tidak diteruskan.
        """

        return None

    def get_config(self) -> dict:
        """
        Menyimpan konfigurasi agar model bisa dimuat kembali.
        """

        config = super().get_config()

        config.update(
            {
                "attention_units":
                    self.attention_units,
            }
        )

        return config


# =============================================================================
# MEMBANGUN MODEL ATTENTION-BILSTM
# =============================================================================

def build_attention_bilstm_model(
    vocabulary_size: int,
    max_sequence_length: int,
    num_classes: int = 4,
    embedding_dim: int = 128,
    lstm_units: int = 64,
    attention_units: int = 64,
    dense_units: int = 128,
    spatial_dropout_rate: float = 0.2,
    recurrent_dropout_rate: float = 0.0,
    dropout_rate: float = 0.5,
    learning_rate: float = 0.001,
    model_name: str = "Attention_BiLSTM_Text_Classifier",
) -> Model:
    """
    Membuat dan mengompilasi model Attention-BiLSTM.

    Parameters
    ----------
    vocabulary_size:
        Jumlah token dalam vocabulary skenario.

    max_sequence_length:
        Panjang input sequence.

    num_classes:
        Jumlah kelas target. Penelitian ini menggunakan 4 kelas.

    embedding_dim:
        Ukuran vektor embedding setiap token.

    lstm_units:
        Jumlah unit pada masing-masing arah LSTM.

        Karena menggunakan Bidirectional dengan merge_mode='concat',
        output fitur menjadi:
        2 × lstm_units.

    attention_units:
        Jumlah unit pada projection layer attention.

    dense_units:
        Jumlah neuron Dense setelah attention.

    spatial_dropout_rate:
        Dropout pada output embedding.

    recurrent_dropout_rate:
        Dropout koneksi recurrent LSTM.

        Default 0.0 agar implementasi LSTM tetap lebih efisien,
        terutama jika nanti dijalankan pada GPU.

    dropout_rate:
        Dropout sebelum output klasifikasi.

    learning_rate:
        Learning rate optimizer Adam.

    model_name:
        Nama model.

    Returns
    -------
    tensorflow.keras.Model
        Model Attention-BiLSTM yang sudah dikompilasi.
    """

    # -------------------------------------------------------------------------
    # VALIDASI PARAMETER
    # -------------------------------------------------------------------------

    if vocabulary_size <= 2:
        raise ValueError(
            "vocabulary_size harus lebih besar dari 2."
        )

    if max_sequence_length <= 0:
        raise ValueError(
            "max_sequence_length harus lebih besar dari 0."
        )

    if num_classes <= 1:
        raise ValueError(
            "num_classes harus lebih besar dari 1."
        )

    # -------------------------------------------------------------------------
    # 1. INPUT LAYER
    # -------------------------------------------------------------------------

    inputs = Input(
        shape=(max_sequence_length,),
        dtype="int32",
        name="input_sequence",
    )

    # -------------------------------------------------------------------------
    # 2. EMBEDDING LAYER
    # -------------------------------------------------------------------------
    #
    # mask_zero=True:
    # indeks 0 dianggap sebagai padding.
    # Mask tersebut diteruskan ke BiLSTM dan Attention.

    x = Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="embedding",
    )(inputs)

    # -------------------------------------------------------------------------
    # 3. SPATIAL DROPOUT
    # -------------------------------------------------------------------------

    x = SpatialDropout1D(
        rate=spatial_dropout_rate,
        name="spatial_dropout",
    )(x)

    # -------------------------------------------------------------------------
    # 4. BIDIRECTIONAL LSTM
    # -------------------------------------------------------------------------
    #
    # return_sequences=True wajib karena attention memerlukan
    # output untuk setiap timestep/token.
    #
    # merge_mode="concat":
    # output forward dan backward digabungkan.

    x = Bidirectional(
        LSTM(
            units=lstm_units,
            return_sequences=True,
            dropout=0.0,
            recurrent_dropout=recurrent_dropout_rate,
            name="lstm",
        ),
        merge_mode="concat",
        name="bidirectional_lstm",
    )(x)

    # -------------------------------------------------------------------------
    # 5. ATTENTION POOLING
    # -------------------------------------------------------------------------

    x = AttentionPooling1D(
        attention_units=attention_units,
        name="attention_pooling",
    )(x)

    # -------------------------------------------------------------------------
    # 6. DENSE LAYER
    # -------------------------------------------------------------------------

    x = Dense(
        units=dense_units,
        activation="relu",
        name="dense",
    )(x)

    # -------------------------------------------------------------------------
    # 7. DROPOUT
    # -------------------------------------------------------------------------

    x = Dropout(
        rate=dropout_rate,
        name="dropout",
    )(x)

    # -------------------------------------------------------------------------
    # 8. OUTPUT LAYER
    # -------------------------------------------------------------------------

    outputs = Dense(
        units=num_classes,
        activation="softmax",
        name="output",
    )(x)

    # -------------------------------------------------------------------------
    # 9. BUILD MODEL
    # -------------------------------------------------------------------------

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name=model_name,
    )

    # -------------------------------------------------------------------------
    # 10. COMPILE MODEL
    # -------------------------------------------------------------------------

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# =============================================================================
# TEST MODEL
# =============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print(
        "STEP 5.2 - ATTENTION-BILSTM "
        "MODEL ARCHITECTURE TEST"
    )
    print("=" * 72)

    # Konfigurasi ini hanya untuk menguji apakah arsitektur
    # berhasil dibuat. Belum digunakan untuk training final.

    TEST_VOCABULARY_SIZE = 20_000
    TEST_MAX_SEQUENCE_LENGTH = 60
    TEST_NUM_CLASSES = 4

    test_model = build_attention_bilstm_model(
        vocabulary_size=TEST_VOCABULARY_SIZE,
        max_sequence_length=(
            TEST_MAX_SEQUENCE_LENGTH
        ),
        num_classes=TEST_NUM_CLASSES,
        embedding_dim=128,
        lstm_units=64,
        attention_units=64,
        dense_units=128,
        spatial_dropout_rate=0.2,
        recurrent_dropout_rate=0.0,
        dropout_rate=0.5,
        learning_rate=0.001,
    )

    print("\nKonfigurasi pengujian:")

    print(
        f"Vocabulary size       : "
        f"{TEST_VOCABULARY_SIZE:,}"
    )

    print(
        f"Max sequence length   : "
        f"{TEST_MAX_SEQUENCE_LENGTH}"
    )

    print(
        f"Jumlah kelas          : "
        f"{TEST_NUM_CLASSES}"
    )

    print(
        f"Embedding dimension   : "
        f"{128}"
    )

    print(
        f"LSTM units per arah   : "
        f"{64}"
    )

    print(
        f"Output BiLSTM         : "
        f"{64 * 2}"
    )

    print(
        f"Attention units       : "
        f"{64}"
    )

    print(
        f"Dense units           : "
        f"{128}"
    )

    print("\nModel Summary:")

    test_model.summary()

    # -------------------------------------------------------------------------
    # TEST FORWARD PASS
    # -------------------------------------------------------------------------
    #
    # Memastikan model bukan hanya berhasil dibuat,
    # tetapi juga dapat menerima input dan menghasilkan output.

    dummy_input = tf.ones(
        shape=(
            2,
            TEST_MAX_SEQUENCE_LENGTH,
        ),
        dtype=tf.int32,
    )

    dummy_output = test_model(
        dummy_input,
        training=False,
    )

    print("\nPengujian forward pass:")

    print(
        f"Shape dummy input     : "
        f"{dummy_input.shape}"
    )

    print(
        f"Shape dummy output    : "
        f"{dummy_output.shape}"
    )

    print(
        "Jumlah probabilitas "
        "setiap sampel:"
    )

    print(
        tf.reduce_sum(
            dummy_output,
            axis=1,
        ).numpy()
    )

    print("\n" + "=" * 72)
    print(
        "Arsitektur Attention-BiLSTM "
        "berhasil dibuat."
    )
    print("=" * 72)
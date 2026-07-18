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
#
# Penanganan padding:
# - indeks token 0 dianggap sebagai padding;
# - Embedding menggunakan mask_zero=True;
# - mask diteruskan ke BiLSTM;
# - AttentionPooling1D mengabaikan seluruh posisi padding.
# =============================================================================

from __future__ import annotations

import tempfile
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    Input,
    LSTM,
    SpatialDropout1D,
)


# =============================================================================
# CUSTOM ATTENTION LAYER
# =============================================================================

@keras.utils.register_keras_serializable(
    package="TAKlasifikasiBerita"
)
class AttentionPooling1D(keras.layers.Layer):
    """
    Attention pooling untuk sequence hasil Bidirectional LSTM.

    Layer ini menghitung skor pada setiap timestep, mengubah
    skor menjadi bobot attention menggunakan softmax, kemudian
    menghasilkan satu context vector melalui weighted sum.

    Posisi padding diabaikan menggunakan mask yang berasal dari:

        Embedding(mask_zero=True)
    """

    def __init__(
        self,
        attention_units: int = 64,
        **kwargs,
    ) -> None:
        """
        Menginisialisasi attention pooling.

        Parameters
        ----------
        attention_units:
            Jumlah neuron pada projection layer attention.
        """

        super().__init__(**kwargs)

        if (
            not isinstance(
                attention_units,
                Integral,
            )
            or isinstance(
                attention_units,
                bool,
            )
        ):
            raise TypeError(
                "attention_units harus bertipe integer."
            )

        if attention_units <= 0:
            raise ValueError(
                "attention_units harus lebih besar dari 0."
            )

        self.attention_units = int(
            attention_units
        )

        # Membentuk representasi internal attention.
        self.projection = Dense(
            units=self.attention_units,
            activation="tanh",
            name="attention_projection",
        )

        # Menghasilkan satu skor pada setiap timestep.
        self.score = Dense(
            units=1,
            use_bias=False,
            name="attention_score",
        )

        # Layer dapat menerima mask dari layer sebelumnya.
        self.supports_masking = True

    def build(
        self,
        input_shape,
    ) -> None:
        """
        Membangun sublayer attention berdasarkan bentuk input.

        Input harus berbentuk:

            (
                batch_size,
                sequence_length,
                hidden_features,
            )
        """

        input_shape = tf.TensorShape(
            input_shape
        )

        if input_shape.rank != 3:
            raise ValueError(
                "AttentionPooling1D membutuhkan input "
                "tiga dimensi dengan bentuk "
                "(batch_size, sequence_length, "
                "hidden_features).\n"
                f"Shape ditemukan: {input_shape}"
            )

        hidden_features = input_shape[-1]

        if hidden_features is None:
            raise ValueError(
                "Dimensi hidden_features pada input "
                "AttentionPooling1D harus diketahui."
            )

        # Projection menerima output BiLSTM.
        self.projection.build(
            input_shape
        )

        # Score menerima output projection.
        projected_shape = tf.TensorShape(
            [
                input_shape[0],
                input_shape[1],
                self.attention_units,
            ]
        )

        self.score.build(
            projected_shape
        )

        super().build(
            input_shape
        )

    def call(
        self,
        inputs: tf.Tensor,
        mask: tf.Tensor | None = None,
        training: bool | None = None,
    ) -> tf.Tensor:
        """
        Menghasilkan context vector berbasis attention.

        Parameters
        ----------
        inputs:
            Tensor hasil BiLSTM dengan bentuk:

                (
                    batch_size,
                    sequence_length,
                    hidden_features,
                )

        mask:
            Tensor boolean dengan bentuk:

                (
                    batch_size,
                    sequence_length,
                )

            True menunjukkan token asli.
            False menunjukkan posisi padding.

        training:
            Status mode training atau inference.

        Returns
        -------
        tf.Tensor
            Context vector dengan bentuk:

                (
                    batch_size,
                    hidden_features,
                )
        """

        # ---------------------------------------------------------------------
        # 1. PROJECTION ATTENTION
        # ---------------------------------------------------------------------

        projected_inputs = self.projection(
            inputs,
            training=training,
        )

        # Bentuk:
        # (
        #     batch_size,
        #     sequence_length,
        #     attention_units,
        # )

        # ---------------------------------------------------------------------
        # 2. MENGHITUNG SKOR ATTENTION
        # ---------------------------------------------------------------------

        attention_scores = self.score(
            projected_inputs,
            training=training,
        )

        # Bentuk awal:
        # (
        #     batch_size,
        #     sequence_length,
        #     1,
        # )

        attention_scores = tf.squeeze(
            attention_scores,
            axis=-1,
        )

        # Bentuk:
        # (
        #     batch_size,
        #     sequence_length,
        # )

        # ---------------------------------------------------------------------
        # 3. MENGABAIKAN POSISI PADDING
        # ---------------------------------------------------------------------

        if mask is not None:
            boolean_mask = tf.cast(
                mask,
                dtype=tf.bool,
            )

            # Posisi padding diberi skor negatif besar agar
            # memperoleh bobot mendekati nol setelah softmax.
            negative_value = tf.cast(
                -1e4,
                dtype=attention_scores.dtype,
            )

            negative_scores = tf.fill(
                dims=tf.shape(
                    attention_scores
                ),
                value=negative_value,
            )

            attention_scores = tf.where(
                boolean_mask,
                attention_scores,
                negative_scores,
            )

        # ---------------------------------------------------------------------
        # 4. MENGHITUNG BOBOT ATTENTION
        # ---------------------------------------------------------------------

        attention_weights = tf.nn.softmax(
            attention_scores,
            axis=1,
        )

        # Bentuk:
        # (
        #     batch_size,
        #     sequence_length,
        # )

        attention_weights = tf.expand_dims(
            attention_weights,
            axis=-1,
        )

        # Bentuk:
        # (
        #     batch_size,
        #     sequence_length,
        #     1,
        # )

        # ---------------------------------------------------------------------
        # 5. WEIGHTED SEQUENCE
        # ---------------------------------------------------------------------

        weighted_sequence = (
            inputs
            * attention_weights
        )

        # ---------------------------------------------------------------------
        # 6. CONTEXT VECTOR
        # ---------------------------------------------------------------------

        context_vector = tf.reduce_sum(
            weighted_sequence,
            axis=1,
        )

        # Bentuk:
        # (
        #     batch_size,
        #     hidden_features,
        # )

        return context_vector

    def compute_mask(
        self,
        inputs: tf.Tensor,
        mask: tf.Tensor | None = None,
    ) -> None:
        """
        Menghentikan penerusan mask.

        Output AttentionPooling1D sudah berupa context vector,
        sehingga tidak lagi memiliki dimensi sequence.
        """

        return None

    def compute_output_shape(
        self,
        input_shape,
    ):
        """
        Menghitung bentuk output layer.

        Input:
            (
                batch_size,
                sequence_length,
                hidden_features,
            )

        Output:
            (
                batch_size,
                hidden_features,
            )
        """

        input_shape = tf.TensorShape(
            input_shape
        )

        return tf.TensorShape(
            [
                input_shape[0],
                input_shape[-1],
            ]
        )

    def get_config(self) -> dict:
        """
        Menyimpan konfigurasi custom layer agar model dapat
        disimpan dan dimuat kembali.
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
# VALIDASI PARAMETER MODEL
# =============================================================================

def validate_model_parameters(
    vocabulary_size: int,
    max_sequence_length: int,
    num_classes: int,
    embedding_dim: int,
    lstm_units: int,
    attention_units: int,
    dense_units: int,
    spatial_dropout_rate: float,
    recurrent_dropout_rate: float,
    dropout_rate: float,
    learning_rate: float,
) -> None:
    """
    Memvalidasi seluruh parameter model Attention-BiLSTM.
    """

    integer_parameters = {
        "vocabulary_size":
            vocabulary_size,
        "max_sequence_length":
            max_sequence_length,
        "num_classes":
            num_classes,
        "embedding_dim":
            embedding_dim,
        "lstm_units":
            lstm_units,
        "attention_units":
            attention_units,
        "dense_units":
            dense_units,
    }

    for parameter_name, parameter_value in (
        integer_parameters.items()
    ):
        if (
            not isinstance(
                parameter_value,
                Integral,
            )
            or isinstance(
                parameter_value,
                bool,
            )
        ):
            raise TypeError(
                f"{parameter_name} harus bertipe integer."
            )

        if parameter_value <= 0:
            raise ValueError(
                f"{parameter_name} harus lebih besar dari 0."
            )

    if vocabulary_size <= 2:
        raise ValueError(
            "vocabulary_size harus lebih besar dari 2 "
            "karena indeks 0 digunakan untuk padding dan "
            "indeks 1 digunakan untuk token OOV."
        )

    if num_classes <= 1:
        raise ValueError(
            "num_classes harus lebih besar dari 1."
        )

    dropout_parameters = {
        "spatial_dropout_rate":
            spatial_dropout_rate,
        "recurrent_dropout_rate":
            recurrent_dropout_rate,
        "dropout_rate":
            dropout_rate,
    }

    for parameter_name, parameter_value in (
        dropout_parameters.items()
    ):
        if (
            not isinstance(
                parameter_value,
                Real,
            )
            or isinstance(
                parameter_value,
                bool,
            )
        ):
            raise TypeError(
                f"{parameter_name} harus berupa angka."
            )

        if not 0.0 <= parameter_value < 1.0:
            raise ValueError(
                f"{parameter_name} harus berada pada "
                "rentang 0 <= rate < 1."
            )

    if (
        not isinstance(
            learning_rate,
            Real,
        )
        or isinstance(
            learning_rate,
            bool,
        )
    ):
        raise TypeError(
            "learning_rate harus berupa angka."
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate harus lebih besar dari 0."
        )


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
    model_name: str = (
        "Attention_BiLSTM_Text_Classifier"
    ),
) -> Model:
    """
    Membuat dan mengompilasi model Attention-BiLSTM.

    Parameters
    ----------
    vocabulary_size:
        Jumlah token aktual dalam vocabulary.

        Nilai sudah mencakup:
        - indeks 0 untuk padding;
        - indeks 1 untuk OOV.

    max_sequence_length:
        Panjang tetap sequence input.

    num_classes:
        Jumlah kelas target.

    embedding_dim:
        Dimensi vektor embedding setiap token.

    lstm_units:
        Jumlah unit LSTM pada setiap arah.

        Karena menggunakan Bidirectional dengan merge_mode
        concat, jumlah fitur keluaran menjadi:

            2 × lstm_units

    attention_units:
        Jumlah neuron pada projection layer attention.

    dense_units:
        Jumlah neuron Dense setelah attention pooling.

    spatial_dropout_rate:
        Dropout pada output embedding.

    recurrent_dropout_rate:
        Dropout pada recurrent state LSTM.

        Default 0.0 agar implementasi LSTM lebih efisien.

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

    validate_model_parameters(
        vocabulary_size=vocabulary_size,
        max_sequence_length=(
            max_sequence_length
        ),
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        attention_units=attention_units,
        dense_units=dense_units,
        spatial_dropout_rate=(
            spatial_dropout_rate
        ),
        recurrent_dropout_rate=(
            recurrent_dropout_rate
        ),
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )

    if not isinstance(
        model_name,
        str,
    ):
        raise TypeError(
            "model_name harus bertipe string."
        )

    if not model_name.strip():
        raise ValueError(
            "model_name tidak boleh kosong."
        )

    # -------------------------------------------------------------------------
    # 1. INPUT LAYER
    # -------------------------------------------------------------------------

    inputs = Input(
        shape=(
            max_sequence_length,
        ),
        dtype=tf.int32,
        name="input_sequence",
    )

    # -------------------------------------------------------------------------
    # 2. EMBEDDING LAYER
    # -------------------------------------------------------------------------
    #
    # mask_zero=True:
    # - token ID 0 dianggap sebagai padding;
    # - mask diteruskan ke Bidirectional LSTM;
    # - mask diteruskan ke AttentionPooling1D.

    x = Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="embedding",
    )(inputs)

    # Bentuk:
    # (
    #     batch_size,
    #     max_sequence_length,
    #     embedding_dim,
    # )

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
    # return_sequences=True diperlukan karena attention
    # membutuhkan output pada setiap timestep.
    #
    # merge_mode="concat" menggabungkan output forward dan
    # backward.

    x = Bidirectional(
        LSTM(
            units=lstm_units,
            return_sequences=True,
            dropout=0.0,
            recurrent_dropout=(
                recurrent_dropout_rate
            ),
            name="lstm",
        ),
        merge_mode="concat",
        name="bidirectional_lstm",
    )(x)

    # Bentuk:
    # (
    #     batch_size,
    #     max_sequence_length,
    #     2 * lstm_units,
    # )

    # -------------------------------------------------------------------------
    # 5. ATTENTION POOLING
    # -------------------------------------------------------------------------

    x = AttentionPooling1D(
        attention_units=attention_units,
        name="attention_pooling",
    )(x)

    # Bentuk:
    # (
    #     batch_size,
    #     2 * lstm_units,
    # )

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
    # 8. OUTPUT SOFTMAX
    # -------------------------------------------------------------------------

    outputs = Dense(
        units=num_classes,
        activation="softmax",
        name="output",
    )(x)

    # -------------------------------------------------------------------------
    # 9. MEMBENTUK MODEL
    # -------------------------------------------------------------------------

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name=model_name,
    )

    # -------------------------------------------------------------------------
    # 10. OPTIMIZER
    # -------------------------------------------------------------------------

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        name="adam",
    )

    # -------------------------------------------------------------------------
    # 11. COMPILE MODEL
    # -------------------------------------------------------------------------
    #
    # sparse_categorical_crossentropy digunakan karena label
    # target berupa integer:
    #
    # 0, 1, 2, dan 3.

    model.compile(
        optimizer=optimizer,
        loss=(
            "sparse_categorical_crossentropy"
        ),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(
                name="accuracy",
            )
        ],
    )

    return model


# =============================================================================
# MEMBUAT DUMMY INPUT DENGAN PADDING
# =============================================================================

def create_dummy_padded_input(
    max_sequence_length: int,
) -> tf.Tensor:
    """
    Membuat input dummy yang berisi token asli dan padding.

    Sampel pertama mempunyai enam token.
    Sampel kedua mempunyai empat token.
    """

    if (
        not isinstance(
            max_sequence_length,
            Integral,
        )
        or isinstance(
            max_sequence_length,
            bool,
        )
    ):
        raise TypeError(
            "max_sequence_length harus bertipe integer."
        )

    if max_sequence_length < 6:
        raise ValueError(
            "max_sequence_length pengujian minimal 6."
        )

    dummy_array = np.zeros(
        shape=(
            2,
            int(
                max_sequence_length
            ),
        ),
        dtype=np.int32,
    )

    dummy_array[
        0,
        :6,
    ] = [
        2,
        3,
        4,
        5,
        6,
        7,
    ]

    dummy_array[
        1,
        :4,
    ] = [
        8,
        9,
        10,
        11,
    ]

    return tf.convert_to_tensor(
        dummy_array,
        dtype=tf.int32,
    )


# =============================================================================
# VALIDASI FORWARD PASS
# =============================================================================

def validate_forward_pass(
    model: Model,
    dummy_input: tf.Tensor,
    expected_num_classes: int,
) -> tf.Tensor:
    """
    Memastikan model menghasilkan probabilitas yang valid.
    """

    dummy_output = model(
        dummy_input,
        training=False,
    )

    expected_shape = (
        int(
            dummy_input.shape[0]
        ),
        int(
            expected_num_classes
        ),
    )

    actual_shape = tuple(
        dummy_output.shape
    )

    if actual_shape != expected_shape:
        raise ValueError(
            "Shape output model tidak sesuai.\n"
            f"Diperoleh : {actual_shape}\n"
            f"Seharusnya: {expected_shape}"
        )

    output_array = dummy_output.numpy()

    if not np.isfinite(
        output_array
    ).all():
        raise ValueError(
            "Output model mengandung NaN atau infinity."
        )

    probability_sums = np.sum(
        output_array,
        axis=1,
    )

    np.testing.assert_allclose(
        probability_sums,
        np.ones_like(
            probability_sums
        ),
        rtol=1e-5,
        atol=1e-6,
        err_msg=(
            "Jumlah probabilitas setiap sampel "
            "tidak sama dengan 1."
        ),
    )

    return dummy_output


# =============================================================================
# VALIDASI SAVE DAN LOAD
# =============================================================================

def validate_model_serialization(
    model: Model,
    dummy_input: tf.Tensor,
    original_output: tf.Tensor,
) -> None:
    """
    Menyimpan dan memuat ulang model untuk memastikan custom
    attention layer dapat diserialisasi dengan benar.

    Model dimuat menggunakan compile=False karena pengujian ini
    hanya memvalidasi arsitektur, bobot, dan output model.
    """

    with tempfile.TemporaryDirectory() as (
        temporary_directory
    ):
        model_path = (
            Path(
                temporary_directory
            )
            / "attention_bilstm_test.keras"
        )

        model.save(
            model_path
        )

        reloaded_model = (
            keras.models.load_model(
                model_path,
                compile=False,
            )
        )

        reloaded_output = reloaded_model(
            dummy_input,
            training=False,
        )

        np.testing.assert_allclose(
            original_output.numpy(),
            reloaded_output.numpy(),
            rtol=1e-5,
            atol=1e-6,
            err_msg=(
                "Output model berubah setelah "
                "proses save dan load."
            ),
        )


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

    # Konfigurasi ini hanya digunakan untuk menguji arsitektur.
    # Training final membaca vocabulary_size dan sequence length
    # dari hasil text vectorization.

    TEST_VOCABULARY_SIZE = 20_000
    TEST_MAX_SEQUENCE_LENGTH = 60
    TEST_NUM_CLASSES = 4

    TEST_EMBEDDING_DIM = 128
    TEST_LSTM_UNITS = 64
    TEST_ATTENTION_UNITS = 64
    TEST_DENSE_UNITS = 128

    TEST_SPATIAL_DROPOUT_RATE = 0.2
    TEST_RECURRENT_DROPOUT_RATE = 0.0
    TEST_DROPOUT_RATE = 0.5
    TEST_LEARNING_RATE = 0.001

    # -------------------------------------------------------------------------
    # MEMBANGUN MODEL
    # -------------------------------------------------------------------------

    test_model = build_attention_bilstm_model(
        vocabulary_size=(
            TEST_VOCABULARY_SIZE
        ),
        max_sequence_length=(
            TEST_MAX_SEQUENCE_LENGTH
        ),
        num_classes=(
            TEST_NUM_CLASSES
        ),
        embedding_dim=(
            TEST_EMBEDDING_DIM
        ),
        lstm_units=(
            TEST_LSTM_UNITS
        ),
        attention_units=(
            TEST_ATTENTION_UNITS
        ),
        dense_units=(
            TEST_DENSE_UNITS
        ),
        spatial_dropout_rate=(
            TEST_SPATIAL_DROPOUT_RATE
        ),
        recurrent_dropout_rate=(
            TEST_RECURRENT_DROPOUT_RATE
        ),
        dropout_rate=(
            TEST_DROPOUT_RATE
        ),
        learning_rate=(
            TEST_LEARNING_RATE
        ),
        model_name=(
            "Attention_BiLSTM_Architecture_Test"
        ),
    )

    # -------------------------------------------------------------------------
    # MENAMPILKAN KONFIGURASI
    # -------------------------------------------------------------------------

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
        f"{TEST_EMBEDDING_DIM}"
    )

    print(
        f"LSTM units per arah   : "
        f"{TEST_LSTM_UNITS}"
    )

    print(
        f"Output BiLSTM         : "
        f"{TEST_LSTM_UNITS * 2}"
    )

    print(
        f"Attention units       : "
        f"{TEST_ATTENTION_UNITS}"
    )

    print(
        f"Dense units           : "
        f"{TEST_DENSE_UNITS}"
    )

    print(
        f"Spatial dropout       : "
        f"{TEST_SPATIAL_DROPOUT_RATE}"
    )

    print(
        f"Recurrent dropout     : "
        f"{TEST_RECURRENT_DROPOUT_RATE}"
    )

    print(
        f"Dropout               : "
        f"{TEST_DROPOUT_RATE}"
    )

    print(
        f"Learning rate         : "
        f"{TEST_LEARNING_RATE}"
    )

    # -------------------------------------------------------------------------
    # MODEL SUMMARY
    # -------------------------------------------------------------------------

    print("\nModel Summary:")

    test_model.summary()

    # -------------------------------------------------------------------------
    # MEMBUAT DUMMY INPUT DENGAN PADDING
    # -------------------------------------------------------------------------

    dummy_input = create_dummy_padded_input(
        max_sequence_length=(
            TEST_MAX_SEQUENCE_LENGTH
        )
    )

    # -------------------------------------------------------------------------
    # VALIDASI FORWARD PASS
    # -------------------------------------------------------------------------

    dummy_output = validate_forward_pass(
        model=test_model,
        dummy_input=dummy_input,
        expected_num_classes=(
            TEST_NUM_CLASSES
        ),
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
        "Jumlah probabilitas setiap sampel:"
    )

    print(
        tf.reduce_sum(
            dummy_output,
            axis=1,
        ).numpy()
    )

    print(
        "Forward pass dengan padding : berhasil"
    )

    # -------------------------------------------------------------------------
    # VALIDASI SAVE DAN LOAD
    # -------------------------------------------------------------------------

    validate_model_serialization(
        model=test_model,
        dummy_input=dummy_input,
        original_output=dummy_output,
    )

    print(
        "Pengujian save-load          : berhasil"
    )

    # -------------------------------------------------------------------------
    # HASIL AKHIR
    # -------------------------------------------------------------------------

    print("\n" + "=" * 72)

    print(
        "Arsitektur Attention-BiLSTM "
        "berhasil dibuat dan divalidasi."
    )

    print("=" * 72)
# =============================================================================
# STEP 5.1 - CNN MODEL ARCHITECTURE
# =============================================================================
# File:
# 5_modeling/cnn_model.py
#
# Tujuan:
# Mendefinisikan arsitektur CNN untuk klasifikasi berita
# multikelas.
#
# Arsitektur:
# Input
#   ↓
# Embedding
#   ↓
# Zero Padding Embedding Output
#   ↓
# SpatialDropout1D
#   ↓
# Conv1D
#   ↓
# Masked Global Max Pooling
#   ↓
# Dense
#   ↓
# Dropout
#   ↓
# Output Softmax
#
# Penanganan padding:
# - indeks token 0 merupakan padding;
# - output embedding pada padding dibuat nol;
# - jendela konvolusi yang seluruhnya hanya berisi padding
#   tidak disertakan dalam global max pooling.
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
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    Input,
    SpatialDropout1D,
)


# =============================================================================
# CUSTOM LAYER: MENGOSONGKAN EMBEDDING PADDING
# =============================================================================

@keras.utils.register_keras_serializable(
    package="TAKlasifikasiBerita"
)
class ZeroPaddingEmbeddingOutput(
    keras.layers.Layer
):
    """
    Mengubah output embedding pada posisi padding menjadi nol.

    Input layer terdiri dari:
    1. embedding tensor;
    2. token ID asli.

    Token dengan indeks 0 dianggap sebagai padding.
    """

    def __init__(
        self,
        padding_index: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(
            padding_index,
            Integral,
        ) or isinstance(
            padding_index,
            bool,
        ):
            raise TypeError(
                "padding_index harus bertipe integer."
            )

        if padding_index < 0:
            raise ValueError(
                "padding_index tidak boleh negatif."
            )

        self.padding_index = int(
            padding_index
        )

    def call(
        self,
        inputs: list[tf.Tensor]
        | tuple[tf.Tensor, tf.Tensor],
    ) -> tf.Tensor:
        """
        Menghasilkan embedding dengan nilai nol pada padding.

        Parameters
        ----------
        inputs:
            Daftar yang berisi:
            - embedding_output:
              (batch, sequence_length, embedding_dim)
            - token_ids:
              (batch, sequence_length)

        Returns
        -------
        tf.Tensor
            Embedding yang telah diberi mask padding.
        """

        if not isinstance(
            inputs,
            (list, tuple),
        ) or len(inputs) != 2:
            raise ValueError(
                "ZeroPaddingEmbeddingOutput harus "
                "menerima [embedding_output, token_ids]."
            )

        embedding_output, token_ids = inputs

        token_mask = tf.not_equal(
            token_ids,
            self.padding_index,
        )

        token_mask = tf.cast(
            token_mask,
            dtype=embedding_output.dtype,
        )

        token_mask = tf.expand_dims(
            token_mask,
            axis=-1,
        )

        return (
            embedding_output
            * token_mask
        )

    def compute_output_shape(
        self,
        input_shape,
    ):
        """
        Bentuk output sama dengan bentuk embedding.
        """

        return input_shape[0]

    def get_config(self) -> dict:
        """
        Menyimpan konfigurasi custom layer.
        """

        config = super().get_config()

        config.update(
            {
                "padding_index":
                    self.padding_index,
            }
        )

        return config


# =============================================================================
# CUSTOM LAYER: MASKED GLOBAL MAX POOLING
# =============================================================================

@keras.utils.register_keras_serializable(
    package="TAKlasifikasiBerita"
)
class MaskedGlobalMaxPooling1D(
    keras.layers.Layer
):
    """
    Global max pooling yang mengabaikan jendela konvolusi
    yang seluruhnya hanya mengandung padding.

    Jendela yang masih mengandung setidaknya satu token asli
    tetap dianggap valid. Hal ini penting karena terdapat
    judul pendek yang jumlah tokennya lebih kecil dari
    kernel_size.
    """

    def __init__(
        self,
        kernel_size: int,
        padding_index: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(
            kernel_size,
            Integral,
        ) or isinstance(
            kernel_size,
            bool,
        ):
            raise TypeError(
                "kernel_size harus bertipe integer."
            )

        if kernel_size <= 0:
            raise ValueError(
                "kernel_size harus lebih besar dari 0."
            )

        if not isinstance(
            padding_index,
            Integral,
        ) or isinstance(
            padding_index,
            bool,
        ):
            raise TypeError(
                "padding_index harus bertipe integer."
            )

        if padding_index < 0:
            raise ValueError(
                "padding_index tidak boleh negatif."
            )

        self.kernel_size = int(
            kernel_size
        )

        self.padding_index = int(
            padding_index
        )

    def call(
        self,
        inputs: list[tf.Tensor]
        | tuple[tf.Tensor, tf.Tensor],
    ) -> tf.Tensor:
        """
        Melakukan max pooling hanya pada jendela valid.

        Parameters
        ----------
        inputs:
            Daftar yang berisi:
            - convolution_output:
              (batch, conv_steps, num_filters)
            - token_ids:
              (batch, original_sequence_length)

        Returns
        -------
        tf.Tensor
            Tensor hasil pooling:
            (batch, num_filters)
        """

        if not isinstance(
            inputs,
            (list, tuple),
        ) or len(inputs) != 2:
            raise ValueError(
                "MaskedGlobalMaxPooling1D harus "
                "menerima [convolution_output, token_ids]."
            )

        convolution_output, token_ids = (
            inputs
        )

        # -----------------------------------------------------
        # 1. MEMBUAT MASK TOKEN
        # -----------------------------------------------------

        token_mask = tf.not_equal(
            token_ids,
            self.padding_index,
        )

        token_mask = tf.cast(
            token_mask,
            dtype=convolution_output.dtype,
        )

        token_mask = tf.expand_dims(
            token_mask,
            axis=-1,
        )

        # Bentuk:
        # (batch, sequence_length, 1)

        # -----------------------------------------------------
        # 2. MEMERIKSA SETIAP JENDELA KONVOLUSI
        # -----------------------------------------------------
        #
        # Kernel bernilai satu digunakan untuk menghitung
        # jumlah token asli dalam setiap jendela Conv1D.

        mask_kernel = tf.ones(
            shape=(
                self.kernel_size,
                1,
                1,
            ),
            dtype=token_mask.dtype,
        )

        valid_token_counts = tf.nn.conv1d(
            input=token_mask,
            filters=mask_kernel,
            stride=1,
            padding="VALID",
        )

        # Jendela dianggap valid apabila mempunyai minimal
        # satu token asli.
        valid_window_mask = tf.greater(
            valid_token_counts,
            0,
        )

        # Membuat bentuk mask sama dengan output Conv1D.
        valid_window_mask = tf.broadcast_to(
            valid_window_mask,
            tf.shape(
                convolution_output
            ),
        )

        # -----------------------------------------------------
        # 3. MENONAKTIFKAN JENDELA FULL-PADDING
        # -----------------------------------------------------

        negative_value = tf.cast(
            -1e4,
            dtype=convolution_output.dtype,
        )

        negative_tensor = tf.fill(
            dims=tf.shape(
                convolution_output
            ),
            value=negative_value,
        )

        masked_convolution = tf.where(
            valid_window_mask,
            convolution_output,
            negative_tensor,
        )

        # -----------------------------------------------------
        # 4. GLOBAL MAX POOLING
        # -----------------------------------------------------

        pooled_output = tf.reduce_max(
            masked_convolution,
            axis=1,
        )

        return pooled_output

    def compute_output_shape(
        self,
        input_shape,
    ):
        """
        Output pooling berbentuk:
        (batch_size, num_filters).
        """

        convolution_shape = input_shape[0]

        return (
            convolution_shape[0],
            convolution_shape[-1],
        )

    def get_config(self) -> dict:
        """
        Menyimpan konfigurasi custom layer.
        """

        config = super().get_config()

        config.update(
            {
                "kernel_size":
                    self.kernel_size,
                "padding_index":
                    self.padding_index,
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
    num_filters: int,
    kernel_size: int,
    dense_units: int,
    spatial_dropout_rate: float,
    dropout_rate: float,
    learning_rate: float,
) -> None:
    """
    Memvalidasi seluruh hyperparameter model CNN.
    """

    integer_parameters = {
        "vocabulary_size": vocabulary_size,
        "max_sequence_length":
            max_sequence_length,
        "num_classes": num_classes,
        "embedding_dim": embedding_dim,
        "num_filters": num_filters,
        "kernel_size": kernel_size,
        "dense_units": dense_units,
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
                f"{parameter_name} harus "
                "bertipe integer."
            )

        if parameter_value <= 0:
            raise ValueError(
                f"{parameter_name} harus "
                "lebih besar dari 0."
            )

    if vocabulary_size <= 2:
        raise ValueError(
            "vocabulary_size harus lebih besar "
            "dari 2 karena indeks 0 digunakan "
            "untuk padding dan indeks 1 untuk OOV."
        )

    if num_classes <= 1:
        raise ValueError(
            "num_classes harus lebih besar dari 1."
        )

    if kernel_size > max_sequence_length:
        raise ValueError(
            "kernel_size tidak boleh lebih besar "
            "dari max_sequence_length."
        )

    dropout_parameters = {
        "spatial_dropout_rate":
            spatial_dropout_rate,
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
                f"{parameter_name} harus "
                "berupa angka."
            )

        if not 0.0 <= parameter_value < 1.0:
            raise ValueError(
                f"{parameter_name} harus berada "
                "pada rentang 0 <= rate < 1."
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
# MEMBANGUN MODEL CNN
# =============================================================================

def build_cnn_model(
    vocabulary_size: int,
    max_sequence_length: int,
    num_classes: int = 4,
    embedding_dim: int = 128,
    num_filters: int = 128,
    kernel_size: int = 5,
    dense_units: int = 128,
    spatial_dropout_rate: float = 0.2,
    dropout_rate: float = 0.5,
    learning_rate: float = 0.001,
    model_name: str = "CNN_Text_Classifier",
) -> Model:
    """
    Membuat dan mengompilasi CNN untuk klasifikasi teks.

    Parameters
    ----------
    vocabulary_size:
        Jumlah token aktual pada vocabulary.

        Nilai tersebut sudah mencakup:
        - indeks 0 untuk padding;
        - indeks 1 untuk OOV.

    max_sequence_length:
        Panjang tetap sequence input.

    num_classes:
        Jumlah kelas klasifikasi.

    embedding_dim:
        Dimensi vektor embedding setiap token.

    num_filters:
        Jumlah filter pada Conv1D.

    kernel_size:
        Ukuran jendela Conv1D.

    dense_units:
        Jumlah neuron Dense setelah pooling.

    spatial_dropout_rate:
        Tingkat dropout pada output embedding.

    dropout_rate:
        Tingkat dropout sebelum output klasifikasi.

    learning_rate:
        Learning rate optimizer Adam.

    model_name:
        Nama model.

    Returns
    -------
    tensorflow.keras.Model
        Model CNN yang sudah dikompilasi.
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
        num_filters=num_filters,
        kernel_size=kernel_size,
        dense_units=dense_units,
        spatial_dropout_rate=(
            spatial_dropout_rate
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
        shape=(max_sequence_length,),
        dtype=tf.int32,
        name="input_sequence",
    )

    # -------------------------------------------------------------------------
    # 2. EMBEDDING LAYER
    # -------------------------------------------------------------------------
    #
    # mask_zero=False digunakan karena Conv1D tidak memproses
    # mask Keras secara langsung.
    #
    # Padding ditangani oleh custom layer setelah embedding.

    embedding_output = Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_dim,
        mask_zero=False,
        name="embedding",
    )(inputs)

    # Bentuk:
    # (
    #     batch_size,
    #     max_sequence_length,
    #     embedding_dim,
    # )

    # -------------------------------------------------------------------------
    # 3. MENGOSONGKAN EMBEDDING PADDING
    # -------------------------------------------------------------------------

    x = ZeroPaddingEmbeddingOutput(
        padding_index=0,
        name="zero_padding_embedding",
    )(
        [
            embedding_output,
            inputs,
        ]
    )

    # -------------------------------------------------------------------------
    # 4. SPATIAL DROPOUT
    # -------------------------------------------------------------------------

    x = SpatialDropout1D(
        rate=spatial_dropout_rate,
        name="spatial_dropout",
    )(x)

    # -------------------------------------------------------------------------
    # 5. CONVOLUTIONAL LAYER
    # -------------------------------------------------------------------------

    convolution_output = Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        activation="relu",
        padding="valid",
        strides=1,
        name="conv1d",
    )(x)

    # Output length:
    #
    # max_sequence_length - kernel_size + 1

    # -------------------------------------------------------------------------
    # 6. MASKED GLOBAL MAX POOLING
    # -------------------------------------------------------------------------

    x = MaskedGlobalMaxPooling1D(
        kernel_size=kernel_size,
        padding_index=0,
        name="masked_global_max_pooling",
    )(
        [
            convolution_output,
            inputs,
        ]
    )

    # Bentuk:
    # (batch_size, num_filters)

    # -------------------------------------------------------------------------
    # 7. DENSE LAYER
    # -------------------------------------------------------------------------

    x = Dense(
        units=dense_units,
        activation="relu",
        name="dense",
    )(x)

    # -------------------------------------------------------------------------
    # 8. DROPOUT
    # -------------------------------------------------------------------------

    x = Dropout(
        rate=dropout_rate,
        name="dropout",
    )(x)

    # -------------------------------------------------------------------------
    # 9. OUTPUT SOFTMAX
    # -------------------------------------------------------------------------

    outputs = Dense(
        units=num_classes,
        activation="softmax",
        name="output",
    )(x)

    # -------------------------------------------------------------------------
    # 10. MEMBENTUK MODEL
    # -------------------------------------------------------------------------

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name=model_name,
    )

    # -------------------------------------------------------------------------
    # 11. OPTIMIZER
    # -------------------------------------------------------------------------

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        name="adam",
    )

    # -------------------------------------------------------------------------
    # 12. COMPILE
    # -------------------------------------------------------------------------
    #
    # Label target berbentuk integer:
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
    Membuat dua sequence dummy.

    Sampel pertama mempunyai enam token.
    Sampel kedua hanya mempunyai dua token sehingga juga
    menguji dokumen yang lebih pendek daripada kernel_size.
    """

    if max_sequence_length < 6:
        raise ValueError(
            "max_sequence_length pengujian minimal 6."
        )

    dummy_array = np.zeros(
        shape=(
            2,
            max_sequence_length,
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
        :2,
    ] = [
        8,
        9,
    ]

    return tf.convert_to_tensor(
        dummy_array,
        dtype=tf.int32,
    )


# =============================================================================
# VALIDASI OUTPUT EMBEDDING PADDING
# =============================================================================

def validate_zero_padding_embedding(
    model: Model,
    dummy_input: tf.Tensor,
) -> None:
    """
    Memastikan seluruh output embedding pada indeks padding
    bernilai nol.
    """

    padding_model = Model(
        inputs=model.input,
        outputs=model.get_layer(
            "zero_padding_embedding"
        ).output,
    )

    masked_embedding = padding_model(
        dummy_input,
        training=False,
    ).numpy()

    padding_positions = (
        dummy_input.numpy()
        == 0
    )

    padding_vectors = masked_embedding[
        padding_positions
    ]

    if padding_vectors.size == 0:
        raise ValueError(
            "Dummy input tidak mempunyai padding."
        )

    if not np.allclose(
        padding_vectors,
        0.0,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "Output embedding pada posisi padding "
            "belum seluruhnya bernilai nol."
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
    Memastikan model dapat melakukan forward pass dan
    menghasilkan probabilitas yang valid.
    """

    dummy_output = model(
        dummy_input,
        training=False,
    )

    expected_shape = (
        int(
            dummy_input.shape[0]
        ),
        expected_num_classes,
    )

    actual_shape = tuple(
        dummy_output.shape
    )

    if actual_shape != expected_shape:
        raise ValueError(
            "Shape output CNN tidak sesuai.\n"
            f"Diperoleh : {actual_shape}\n"
            f"Seharusnya: {expected_shape}"
        )

    output_array = dummy_output.numpy()

    if not np.isfinite(
        output_array
    ).all():
        raise ValueError(
            "Output CNN mengandung NaN "
            "atau infinity."
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
    Menyimpan dan memuat ulang model untuk memastikan
    custom layer CNN dapat diserialisasi.
    """

    with tempfile.TemporaryDirectory() as (
        temporary_directory
    ):
        model_path = (
            Path(temporary_directory)
            / "cnn_model_test.keras"
        )

        model.save(
            model_path
        )

        reloaded_model = keras.models.load_model(
            model_path,
            compile=False,
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
                "Output CNN berubah setelah "
                "proses save dan load."
            ),
        )


# =============================================================================
# TEST MODEL
# =============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print(
        "STEP 5.1 - CNN MODEL "
        "ARCHITECTURE TEST"
    )
    print("=" * 72)

    # Konfigurasi pengujian arsitektur.
    # Training final akan membaca vocabulary_size dan
    # max_sequence_length dari artefak vectorization.

    TEST_VOCABULARY_SIZE = 20_000
    TEST_MAX_SEQUENCE_LENGTH = 60
    TEST_NUM_CLASSES = 4

    TEST_EMBEDDING_DIM = 128
    TEST_NUM_FILTERS = 128
    TEST_KERNEL_SIZE = 5
    TEST_DENSE_UNITS = 128

    TEST_SPATIAL_DROPOUT_RATE = 0.2
    TEST_DROPOUT_RATE = 0.5
    TEST_LEARNING_RATE = 0.001

    # -------------------------------------------------------------------------
    # MEMBANGUN MODEL
    # -------------------------------------------------------------------------

    test_model = build_cnn_model(
        vocabulary_size=(
            TEST_VOCABULARY_SIZE
        ),
        max_sequence_length=(
            TEST_MAX_SEQUENCE_LENGTH
        ),
        num_classes=TEST_NUM_CLASSES,
        embedding_dim=TEST_EMBEDDING_DIM,
        num_filters=TEST_NUM_FILTERS,
        kernel_size=TEST_KERNEL_SIZE,
        dense_units=TEST_DENSE_UNITS,
        spatial_dropout_rate=(
            TEST_SPATIAL_DROPOUT_RATE
        ),
        dropout_rate=TEST_DROPOUT_RATE,
        learning_rate=TEST_LEARNING_RATE,
        model_name=(
            "CNN_Architecture_Test"
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
        f"Jumlah filter Conv1D  : "
        f"{TEST_NUM_FILTERS}"
    )

    print(
        f"Kernel size           : "
        f"{TEST_KERNEL_SIZE}"
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
    # MEMBUAT DUMMY INPUT
    # -------------------------------------------------------------------------

    dummy_input = create_dummy_padded_input(
        max_sequence_length=(
            TEST_MAX_SEQUENCE_LENGTH
        )
    )

    # -------------------------------------------------------------------------
    # VALIDASI PADDING EMBEDDING
    # -------------------------------------------------------------------------

    validate_zero_padding_embedding(
        model=test_model,
        dummy_input=dummy_input,
    )

    print(
        "\nPadding embedding      : berhasil dibuat nol"
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
        "Jumlah probabilitas "
        "setiap sampel:"
    )

    print(
        tf.reduce_sum(
            dummy_output,
            axis=1,
        ).numpy()
    )

    print(
        "Forward pass          : berhasil"
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
        "Pengujian save-load  : berhasil"
    )

    # -------------------------------------------------------------------------
    # HASIL AKHIR
    # -------------------------------------------------------------------------

    print("\n" + "=" * 72)
    print(
        "Arsitektur CNN berhasil dibuat "
        "dan divalidasi."
    )
    print("=" * 72)
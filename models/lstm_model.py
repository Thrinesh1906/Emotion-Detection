"""
LSTM architectures for emotional transition detection.

Uses two inputs:
  1) Padded utterance context (sequence)
  2) Previous emotion (one-hot) — required for meaningful transitions at inference
"""
from __future__ import annotations

from typing import Optional

from tensorflow import keras
from tensorflow.keras import layers

from utils.config import Config, get_config


def build_bilstm_model(
    num_classes: int,
    config: Optional[Config] = None,
    num_emotions: Optional[int] = None,
) -> keras.Model:
    """
    BiLSTM + previous-emotion fusion for next-state / transition prediction.
    """
    config = config or get_config()
    num_emotions = num_emotions or len(config.emotions)

    seq_input = keras.Input(shape=(config.max_sequence_length,), name="context_sequence")
    emo_input = keras.Input(shape=(num_emotions,), name="prev_emotion_onehot")

    x = layers.Embedding(
        input_dim=config.vocab_size,
        output_dim=config.embedding_dim,
        mask_zero=True,
        name="embedding",
    )(seq_input)
    x = layers.Bidirectional(
        layers.LSTM(config.lstm_units, return_sequences=True),
        name="bilstm_1",
    )(x)
    x = layers.Dropout(config.dropout_rate, name="dropout_1")(x)
    x = layers.Bidirectional(
        layers.LSTM(config.lstm_units // 2, return_sequences=False),
        name="bilstm_2",
    )(x)
    x = layers.Dropout(config.dropout_rate, name="dropout_2")(x)

    emo_dense = layers.Dense(32, activation="relu", name="emotion_embed")(emo_input)
    merged = layers.Concatenate(name="context_emotion_merge")([x, emo_dense])
    merged = layers.Dense(128, activation="relu", name="dense_1")(merged)
    merged = layers.Dropout(config.dropout_rate / 2, name="dropout_3")(merged)
    merged = layers.Dense(64, activation="relu", name="dense_2")(merged)
    outputs = layers.Dense(num_classes, activation="softmax", name="output_softmax")(merged)

    model = keras.Model(
        inputs=[seq_input, emo_input],
        outputs=outputs,
        name="EmotionTransitionBiLSTM",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_simple_rnn_model(
    num_classes: int,
    config: Optional[Config] = None,
    num_emotions: Optional[int] = None,
) -> keras.Model:
    """Vanilla RNN with previous-emotion input (comparison baseline)."""
    config = config or get_config()
    num_emotions = num_emotions or len(config.emotions)

    seq_input = keras.Input(shape=(config.max_sequence_length,), name="context_sequence")
    emo_input = keras.Input(shape=(num_emotions,), name="prev_emotion_onehot")

    x = layers.Embedding(
        input_dim=config.vocab_size,
        output_dim=config.embedding_dim,
        mask_zero=True,
    )(seq_input)
    x = layers.SimpleRNN(config.lstm_units, return_sequences=False, name="simple_rnn")(x)
    x = layers.Dropout(config.dropout_rate)(x)
    emo_dense = layers.Dense(32, activation="relu")(emo_input)
    merged = layers.Concatenate()([x, emo_dense])
    merged = layers.Dense(64, activation="relu")(merged)
    outputs = layers.Dense(num_classes, activation="softmax")(merged)

    model = keras.Model(inputs=[seq_input, emo_input], outputs=outputs, name="EmotionTransitionRNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

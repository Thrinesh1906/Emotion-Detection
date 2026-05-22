"""
Train the BiLSTM emotional transition detection model.
Run: python -m training.train
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tensorflow import keras

from models.lstm_model import build_bilstm_model, build_simple_rnn_model
from utils.config import get_config
from utils.data_loader import (
    build_emotion_tensors,
    build_sequence_tensors,
    compute_class_weights,
    encode_labels,
    prepare_dataset,
)
from utils.visualization import plot_training_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_model(
    epochs: int | None = None,
    model_type: str = "bilstm",
    quick: bool = False,
    best: bool = False,
) -> dict:
    """Full training pipeline."""
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    config = get_config()
    if epochs:
        config.epochs = epochs
    if quick:
        config.epochs = min(5, config.epochs)
        config.batch_size = 128
    elif best:
        config.epochs = epochs or 25
        config.batch_size = 48
        config.embedding_dim = 128
        config.lstm_units = 128
        config.dropout_rate = 0.3
        config.learning_rate = 0.001
        logger.info(
            "Best mode: epochs=%d, embed=%d, lstm=%d, batch=%d, dropout=%.2f",
            config.epochs,
            config.embedding_dim,
            config.lstm_units,
            config.batch_size,
            config.dropout_rate,
        )

    logger.info("Preparing dataset...")
    train_df, val_df, test_df = prepare_dataset(config)

    logger.info("Encoding labels and building sequences...")
    # Predict next emotion (7 classes); transition = prev -> predicted next at inference
    y_train, y_val, y_test, label_encoder, idx_to_label = encode_labels(
        train_df, val_df, test_df, label_col="next_emotion"
    )
    X_train, X_val, X_test, builder = build_sequence_tensors(train_df, val_df, test_df, config)
    E_train, E_val, E_test = build_emotion_tensors(train_df, val_df, test_df, config)
    class_weights = compute_class_weights(y_train)

    num_classes = len(label_encoder.classes_)
    logger.info("Number of transition classes: %d", num_classes)

    if model_type == "rnn":
        model = build_simple_rnn_model(num_classes, config)
        model_path = config.saved_models_dir / "emotion_transition_rnn.keras"
    else:
        model = build_bilstm_model(num_classes, config)
        model_path = config.model_path

    model.summary(print_fn=logger.info)

    patience = 10 if best else 4
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            str(model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    logger.info("Training %s for %d epochs...", model_type, config.epochs)
    history_obj = model.fit(
        [X_train, E_train],
        y_train,
        validation_data=([X_val, E_val], y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    history = {k: [float(v) for v in vals] for k, vals in history_obj.history.items()}

    # Always evaluate the best checkpoint saved to disk (not in-memory last epoch)
    if model_path.exists():
        logger.info("Reloading best checkpoint: %s", model_path)
        model = keras.models.load_model(model_path)

    plot_training_history(history)

    # Save artifacts
    joblib.dump(label_encoder, config.label_encoder_path)
    builder.save_tokenizer()

    metadata = {
        "model_type": model_type,
        "prediction_target": "next_emotion",
        "num_classes": num_classes,
        "emotions": list(config.emotions),
        "max_sequence_length": config.max_sequence_length,
        "vocab_size": config.vocab_size,
        "idx_to_label": {str(k): v for k, v in idx_to_label.items()},
        "label_to_idx": {v: int(i) for i, v in enumerate(label_encoder.classes_)},
        "final_train_accuracy": history["accuracy"][-1],
        "final_val_accuracy": history["val_accuracy"][-1],
    }
    with open(config.metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Evaluate on test set (best checkpoint + optional finetune)
    model = keras.models.load_model(model_path)
    test_loss, test_acc = model.evaluate([X_test, E_test], y_test, verbose=0)
    logger.info("Test accuracy: %.4f | Test loss: %.4f", test_acc, test_loss)
    metadata["test_accuracy"] = float(test_acc)
    metadata["test_loss"] = float(test_loss)
    with open(config.metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train emotion transition LSTM")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--model", choices=["bilstm", "rnn"], default="bilstm")
    parser.add_argument("--quick", action="store_true", help="Fast training for demo")
    parser.add_argument("--best", action="store_true", help="Maximum-quality training settings")
    args = parser.parse_args()
    train_model(
        epochs=args.epochs,
        model_type=args.model,
        quick=args.quick,
        best=args.best,
    )


if __name__ == "__main__":
    main()

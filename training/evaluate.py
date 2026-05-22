"""
Model evaluation: metrics, confusion matrix, analytics.
Run: python -m training.evaluate
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tensorflow import keras

from utils.config import get_config
from utils.data_loader import (
    build_emotion_tensors,
    build_sequence_tensors,
    encode_labels,
    prepare_dataset,
)
from utils.preprocessing import build_transition_label
from utils.transitions import emotion_transition_matrix
from utils.visualization import (
    plot_confusion_matrix,
    plot_emotion_distribution,
    plotly_transition_heatmap,
    save_plotly_figure,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model() -> dict:
    """Run full evaluation and generate output artifacts."""
    config = get_config()
    if not config.model_path.exists():
        raise FileNotFoundError(f"Model not found at {config.model_path}. Run training first.")

    train_df, val_df, test_df = prepare_dataset(config)
    _, _, y_test, label_encoder, idx_to_label = encode_labels(
        train_df, val_df, test_df, label_col="next_emotion"
    )
    X_train, X_val, X_test, _ = build_sequence_tensors(train_df, val_df, test_df, config, fit_tokenizer=False)
    _, _, E_test = build_emotion_tensors(train_df, val_df, test_df, config)

    model = keras.models.load_model(config.model_path)
    probs = model.predict([X_test, E_test], verbose=0)
    y_pred = np.argmax(probs, axis=1)

    labels = list(label_encoder.classes_)
    # Build transition strings for heatmap / confusion on transitions
    prev_emotions = test_df["prev_emotion"].tolist()
    true_transitions = [
        build_transition_label(p, labels[y]) for p, y in zip(prev_emotions, y_test)
    ]
    pred_transitions = [
        build_transition_label(p, labels[y]) for p, y in zip(prev_emotions, y_pred)
    ]
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    logger.info("Macro F1: %.4f", macro_f1)
    logger.info("Test samples: %d", len(y_test))

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, labels)

    # Emotion distribution from test set
    emotions = test_df["next_emotion"].tolist()
    plot_emotion_distribution(emotions, title="Test Set Next-Emotion Distribution")

    matrix = emotion_transition_matrix(true_transitions)
    fig = plotly_transition_heatmap(matrix)
    save_plotly_figure(fig, config.outputs_dir / "transition_heatmap.html")

    # Save metrics
    metrics = {
        "macro_f1": float(macro_f1),
        "accuracy": float(np.mean(y_pred == y_test)),
        "num_test_samples": len(y_test),
        "classification_report": report,
    }
    metrics_path = config.outputs_dir / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main() -> None:
    evaluate_model()


if __name__ == "__main__":
    main()

"""
Training and evaluation visualization utilities.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils.config import Config, get_config


plt.style.use("seaborn-v0_8-darkgrid")


def plot_training_history(
    history: Dict,
    save_dir: Optional[Path] = None,
) -> List[Path]:
    """Plot accuracy and loss curves from Keras history."""
    config = get_config()
    save_dir = save_dir or config.outputs_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["accuracy"]) + 1)

    axes[0].plot(epochs, history["accuracy"], "b-o", label="Train")
    axes[0].plot(epochs, history["val_accuracy"], "r-s", label="Validation")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["loss"], "b-o", label="Train")
    axes[1].plot(epochs, history["val_loss"], "r-s", label="Validation")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    acc_path = save_dir / "training_accuracy_loss.png"
    plt.savefig(acc_path, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(acc_path)

    with open(config.history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return paths


def plot_emotion_distribution(
    emotions: List[str],
    save_path: Optional[Path] = None,
    title: str = "Emotion Distribution",
) -> Path:
    """Bar chart of emotion frequencies."""
    config = get_config()
    save_path = save_path or config.outputs_dir / "emotion_distribution.png"
    counts = pd.Series(emotions).value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("husl", len(counts))
    counts.plot(kind="bar", ax=ax, color=colors)
    ax.set_title(title)
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    save_path: Optional[Path] = None,
    max_labels: int = 20,
) -> Path:
    """Plot confusion matrix (top classes if too many)."""
    config = get_config()
    save_path = save_path or config.outputs_dir / "confusion_matrix.png"

    unique = sorted(set(y_true) | set(y_pred))
    if len(unique) > max_labels:
        from collections import Counter
        top = [c for c, _ in Counter(y_true).most_common(max_labels)]
        mask_true = np.isin(y_true, top)
        mask_pred = np.isin(y_pred, top)
        mask = mask_true & mask_pred
        y_true_f = y_true[mask]
        y_pred_f = y_pred[mask]
        label_names = [labels[i] for i in top if i < len(labels)]
        cm = confusion_matrix(y_true_f, y_pred_f, labels=top)
    else:
        cm = confusion_matrix(y_true, y_pred, labels=unique)
        label_names = [labels[i][:20] for i in unique]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_title("Confusion Matrix (Emotional Transitions)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plotly_confidence_bars(
    labels: List[str],
    probabilities: np.ndarray,
    title: str = "Transition Prediction Confidence",
) -> go.Figure:
    """Interactive horizontal bar chart for confidence scores."""
    sorted_idx = np.argsort(probabilities)[::-1][:10]
    top_labels = [labels[i] for i in sorted_idx]
    top_probs = probabilities[sorted_idx]
    fig = go.Figure(
        go.Bar(
            x=top_probs,
            y=top_labels,
            orientation="h",
            marker=dict(
                color=top_probs,
                colorscale="Viridis",
                showscale=True,
            ),
            text=[f"{p:.1%}" for p in top_probs],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Confidence",
        yaxis_title="Transition",
        template="plotly_dark",
        height=400,
    )
    return fig


def plotly_emotion_timeline(timeline_df: pd.DataFrame) -> go.Figure:
    """Interactive emotion timeline across conversation steps."""
    emotion_colors = {
        "joy": "#FFD700",
        "sadness": "#4169E1",
        "anger": "#FF4500",
        "fear": "#9370DB",
        "surprise": "#00CED1",
        "disgust": "#32CD32",
        "neutral": "#A9A9A9",
    }
    colors = [emotion_colors.get(e, "#FFFFFF") for e in timeline_df["emotion"]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timeline_df["step"],
            y=timeline_df["emotion"],
            mode="lines+markers",
            marker=dict(size=14, color=colors),
            line=dict(color="#636EFA", width=2),
            name="Emotion",
        )
    )
    if "confidence" in timeline_df.columns:
        fig.add_trace(
            go.Scatter(
                x=timeline_df["step"],
                y=timeline_df["confidence"],
                mode="lines",
                yaxis="y2",
                name="Confidence",
                line=dict(dash="dot", color="#EF553B"),
            )
        )
        fig.update_layout(
            yaxis2=dict(title="Confidence", overlaying="y", side="right", range=[0, 1]),
        )
    fig.update_layout(
        title="Emotional Transition Timeline",
        xaxis_title="Conversation Step",
        yaxis_title="Emotion",
        template="plotly_dark",
        height=420,
    )
    return fig


def plotly_transition_heatmap(matrix_df: pd.DataFrame) -> go.Figure:
    """Heatmap of emotion transition frequencies."""
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_df.values,
            x=matrix_df.columns.tolist(),
            y=matrix_df.index.tolist(),
            colorscale="YlOrRd",
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title="Emotion Transition Heatmap",
        template="plotly_dark",
        height=450,
    )
    return fig


def save_plotly_figure(fig: go.Figure, path: Path) -> None:
    """Export plotly figure to HTML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))

"""
Emotion transition analytics and timeline utilities.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.config import Config, get_config
from utils.preprocessing import build_transition_label, parse_transition_label


def emotion_transition_matrix(
    transitions: List[str],
    emotions: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build transition count matrix (source x target)."""
    config = get_config()
    emotions = emotions or list(config.emotions)
    matrix = pd.DataFrame(0, index=emotions, columns=emotions)
    for t in transitions:
        src, tgt = parse_transition_label(t)
        if src in matrix.index and tgt in matrix.columns:
            matrix.loc[src, tgt] += 1
    return matrix


def compute_transition_frequencies(df: pd.DataFrame, col: str = "transition") -> pd.Series:
    """Count frequency of each transition type."""
    return df[col].value_counts()


def build_emotion_timeline(
    emotions: List[str],
    confidences: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Create timeline DataFrame for visualization."""
    data = {
        "step": list(range(1, len(emotions) + 1)),
        "emotion": emotions,
    }
    if confidences:
        data["confidence"] = confidences
    return pd.DataFrame(data)


def detect_transitions_from_emotions(emotions: List[str]) -> List[Dict]:
    """Derive transition events from an emotion sequence."""
    events = []
    for i in range(1, len(emotions)):
        prev, curr = emotions[i - 1], emotions[i]
        events.append(
            {
                "step": i + 1,
                "from": prev,
                "to": curr,
                "transition": build_transition_label(prev, curr),
                "changed": prev != curr,
            }
        )
    return events


def top_k_transitions(
    predictions: List[str],
    k: int = 5,
) -> List[Tuple[str, int]]:
    """Return top-k most common predicted transitions."""
    return Counter(predictions).most_common(k)


def transition_entropy(probabilities: np.ndarray) -> float:
    """Shannon entropy of prediction distribution."""
    p = probabilities[probabilities > 0]
    return float(-np.sum(p * np.log2(p + 1e-12)))

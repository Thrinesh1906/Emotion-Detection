"""
Encode emotion labels for model input (previous emotional state).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np

from utils.config import Config, get_config


def normalize_emotion(emotion: str, config: Optional[Config] = None) -> str:
    """Map emotion string to a known label."""
    config = config or get_config()
    emo = str(emotion).lower().strip()
    if emo in config.emotions:
        return emo
    return "neutral"


def encode_prev_emotions(
    emotions: Sequence[str],
    config: Optional[Config] = None,
) -> np.ndarray:
    """One-hot encode previous emotion per sample. Shape: (n_samples, num_emotions)."""
    config = config or get_config()
    n = len(config.emotions)
    emo_to_idx = {e: i for i, e in enumerate(config.emotions)}
    out = np.zeros((len(emotions), n), dtype=np.float32)
    for i, emo in enumerate(emotions):
        idx = emo_to_idx.get(normalize_emotion(emo, config), emo_to_idx["neutral"])
        out[i, idx] = 1.0
    return out


def decode_emotion_index(idx: int, config: Optional[Config] = None) -> str:
    config = config or get_config()
    if 0 <= idx < len(config.emotions):
        return config.emotions[idx]
    return "neutral"

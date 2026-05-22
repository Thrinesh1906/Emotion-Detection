"""
MELD dataset download, loading, and preprocessing pipeline.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from utils.config import Config, get_config
from utils.preprocessing import SequenceBuilder, TextPreprocessor, create_conversation_windows

logger = logging.getLogger(__name__)


def download_file(url: str, dest: Path, timeout: int = 120) -> bool:
    """Download a single file with error handling."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 1000:
            logger.info("Already exists: %s", dest.name)
            return True
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        dest.write_bytes(response.content)
        logger.info("Downloaded: %s", dest.name)
        return True
    except Exception as exc:
        logger.warning("Download failed for %s: %s", url, exc)
        return False


def download_meld_dataset(config: Optional[Config] = None) -> bool:
    """Download MELD CSV splits from GitHub."""
    config = config or get_config()
    success = True
    for filename in config.meld_files:
        url = f"{config.meld_base_url}/{filename}"
        dest = config.raw_dir / filename
        if not download_file(url, dest):
            success = False
    return success


def load_meld_raw(config: Optional[Config] = None) -> pd.DataFrame:
    """Load and concatenate all MELD splits."""
    config = config or get_config()
    frames: List[pd.DataFrame] = []
    for filename in config.meld_files:
        path = config.raw_dir / filename
        if not path.exists():
            download_meld_dataset(config)
        if path.exists():
            df = pd.read_csv(path)
            df["split"] = filename.replace("_sent_emo.csv", "")
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            "MELD dataset not found. Run download_meld_dataset() or place CSVs in dataset/raw/"
        )
    combined = pd.concat(frames, ignore_index=True)
    combined["Emotion"] = combined["Emotion"].str.lower().str.strip()
    return combined


def generate_synthetic_meld(n_dialogues: int = 200, utterances_per: int = 6) -> pd.DataFrame:
    """
    Generate synthetic conversational data when MELD download is unavailable.
    Ensures the project remains runnable offline.
    """
    import random

    config = get_config()
    random.seed(config.random_seed)
    emotions = list(config.emotions)
    templates = {
        "joy": ["I am so happy today!", "This is wonderful news.", "We did it!"],
        "sadness": ["I feel terrible about this.", "Nothing seems to go right.", "I miss them so much."],
        "anger": ["This is completely unacceptable!", "I cannot believe you did that.", "Stop doing that now!"],
        "fear": ["I am really scared right now.", "What if something bad happens?", "I do not feel safe."],
        "surprise": ["Oh my goodness, really?", "I did not expect that at all!", "That is shocking!"],
        "disgust": ["That is absolutely revolting.", "I cannot stand this anymore.", "How disgusting."],
        "neutral": ["Okay, I understand.", "Let me think about it.", "Sure, that works for me."],
    }
    rows = []
    for d in range(n_dialogues):
        prev = random.choice(emotions)
        for u in range(utterances_per):
            if random.random() < 0.6:
                emo = prev
            else:
                emo = random.choice(emotions)
            text = random.choice(templates[emo])
            rows.append(
                {
                    "Utterance": text,
                    "Speaker": f"Speaker_{u % 2}",
                    "Emotion": emo,
                    "Sentiment": "positive" if emo in ("joy", "surprise") else "negative" if emo in ("anger", "sadness", "disgust", "fear") else "neutral",
                    "Dialogue_ID": d,
                    "Utterance_ID": u,
                    "split": "synthetic",
                }
            )
            prev = emo
    return pd.DataFrame(rows)


def prepare_dataset(
    config: Optional[Config] = None,
    use_synthetic_fallback: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: load MELD -> build windows -> save processed CSVs.
    Returns train, val, test DataFrames of transition samples.
    """
    config = config or get_config()
    try:
        raw = load_meld_raw(config)
        logger.info("Loaded MELD: %d utterances", len(raw))
    except Exception as exc:
        logger.warning("MELD load failed (%s). Using synthetic data.", exc)
        if not use_synthetic_fallback:
            raise
        raw = generate_synthetic_meld()

    windows = create_conversation_windows(raw)
    windows = windows[windows["transition"].notna()]
    windows.to_csv(config.processed_dir / "transitions_all.csv", index=False)

    # Split by dialogue to avoid leakage
    dialogue_ids = windows["dialogue_id"].unique()
    import numpy as np

    rng = np.random.default_rng(config.random_seed)
    rng.shuffle(dialogue_ids)
    n = len(dialogue_ids)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    train_ids = set(dialogue_ids[:train_end])
    val_ids = set(dialogue_ids[train_end:val_end])
    test_ids = set(dialogue_ids[val_end:])

    train_df = windows[windows["dialogue_id"].isin(train_ids)].reset_index(drop=True)
    val_df = windows[windows["dialogue_id"].isin(val_ids)].reset_index(drop=True)
    test_df = windows[windows["dialogue_id"].isin(test_ids)].reset_index(drop=True)

    train_df.to_csv(config.processed_dir / "train_transitions.csv", index=False)
    val_df.to_csv(config.processed_dir / "val_transitions.csv", index=False)
    test_df.to_csv(config.processed_dir / "test_transitions.csv", index=False)

    logger.info("Train: %d | Val: %d | Test: %d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


def encode_labels(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = "transition",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, object, Dict[int, str]]:
    """Label-encode transition classes."""
    from sklearn.preprocessing import LabelEncoder
    import numpy as np

    le = LabelEncoder()
    y_train = le.fit_transform(train_df[label_col])
    y_val = le.transform(val_df[label_col])
    y_test = le.transform(test_df[label_col])
    idx_to_label = {i: lbl for i, lbl in enumerate(le.classes_)}
    return y_train, y_val, y_test, le, idx_to_label


def build_sequence_tensors(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Optional[Config] = None,
    fit_tokenizer: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SequenceBuilder]:
    """Tokenize context texts into padded sequences."""
    import numpy as np

    config = config or get_config()
    builder = SequenceBuilder(config)
    all_contexts = (
        train_df["context"].tolist()
        + val_df["context"].tolist()
        + test_df["context"].tolist()
    )
    if fit_tokenizer:
        builder.fit_tokenizer(all_contexts)
        builder.save_tokenizer()
    elif config.tokenizer_path.exists():
        builder.load_tokenizer()
    else:
        builder.fit_tokenizer(all_contexts)
        builder.save_tokenizer()

    X_train = builder.texts_to_sequences(train_df["context"].tolist())
    X_val = builder.texts_to_sequences(val_df["context"].tolist())
    X_test = builder.texts_to_sequences(test_df["context"].tolist())
    return X_train, X_val, X_test, builder


def build_emotion_tensors(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Optional[Config] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-hot encode previous emotion column for each split."""
    from utils.emotion_encoding import encode_prev_emotions

    return (
        encode_prev_emotions(train_df["prev_emotion"].tolist(), config),
        encode_prev_emotions(val_df["prev_emotion"].tolist(), config),
        encode_prev_emotions(test_df["prev_emotion"].tolist(), config),
    )


def compute_class_weights(y_train: np.ndarray) -> dict:
    """Sklearn-balanced class weights for imbalanced emotion labels."""
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y_train)
    balanced = compute_class_weight("balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, balanced)}

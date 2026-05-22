"""
Global configuration for the Emotional Transition Detection project.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    """Central project configuration."""

    project_root: Path = PROJECT_ROOT
    dataset_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "dataset")
    raw_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "dataset" / "raw")
    processed_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "dataset" / "processed")
    saved_models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "saved_models")
    outputs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")
    screenshots_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "screenshots")

    # MELD dataset URLs (Multimodal EmotionLines Dataset)
    meld_base_url: str = (
        "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD"
    )
    meld_files: tuple = ("train_sent_emo.csv", "dev_sent_emo.csv", "test_sent_emo.csv")

    # Model hyperparameters
    max_sequence_length: int = 50
    max_conversation_length: int = 8
    vocab_size: int = 15000
    embedding_dim: int = 128
    lstm_units: int = 128
    dropout_rate: float = 0.4
    batch_size: int = 64
    epochs: int = 15
    learning_rate: float = 0.001
    validation_split: float = 0.15
    random_seed: int = 42

    # Emotion labels in MELD
    emotions: tuple = (
        "anger",
        "disgust",
        "fear",
        "joy",
        "neutral",
        "sadness",
        "surprise",
    )

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    def __post_init__(self) -> None:
        for path in (
            self.dataset_dir,
            self.raw_dir,
            self.processed_dir,
            self.saved_models_dir,
            self.outputs_dir,
            self.screenshots_dir,
            (self.project_root / "models"),
            (self.project_root / "training"),
            (self.project_root / "app"),
            (self.project_root / "notebooks"),
            (self.project_root / "report"),
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def model_path(self) -> Path:
        return self.saved_models_dir / "emotion_transition_lstm.keras"

    @property
    def tokenizer_path(self) -> Path:
        return self.saved_models_dir / "tokenizer.json"

    @property
    def label_encoder_path(self) -> Path:
        return self.saved_models_dir / "label_encoder.joblib"

    @property
    def metadata_path(self) -> Path:
        return self.saved_models_dir / "model_metadata.json"

    @property
    def history_path(self) -> Path:
        return self.outputs_dir / "training_history.json"


def get_config() -> Config:
    """Return singleton-style config instance."""
    return Config()

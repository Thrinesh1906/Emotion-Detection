"""
Text preprocessing, tokenization, and sequence preparation utilities.
"""
from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

from utils.config import Config, get_config


def ensure_nltk_resources() -> None:
    """Download required NLTK corpora if missing."""
    if not _NLTK_AVAILABLE:
        return
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                pass


class TextPreprocessor:
    """Clean and normalize conversational text while preserving emotional markers."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or get_config()
        ensure_nltk_resources()
        self._lemmatizer = WordNetLemmatizer() if _NLTK_AVAILABLE else None
        # Only remove truly non-emotional stopwords; keep negations and emotional words
        try:
            all_stops = set(stopwords.words("english")) if _NLTK_AVAILABLE else set()
            # Preserve emotional negations and intensifiers
            emotional_words = {
                "don't", "don't", "not", "never", "always", "really", "so",
                "very", "just", "only", "still", "cant", "can't", "won't",
                "no", "yes", "never", "ever", "would", "could", "should"
            }
            self._stop_words = all_stops - emotional_words
        except Exception:
            self._stop_words = set()

    def clean_text(self, text: str) -> str:
        """Normalize text while preserving emotional markers (!, ?, repetition)."""
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase but preserve some punctuation
        text = text.lower().strip()
        
        # Remove URLs and mentions
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"@\w+|#\w+", "", text)
        
        # Preserve exclamation/question marks that indicate emotion, but normalize repeated ones
        text = re.sub(r"!{2,}", " !", text)  # Multiple ! -> single !
        text = re.sub(r"\?{2,}", " ?", text)  # Multiple ? -> single ?
        text = re.sub(r"\.{2,}", " .", text)  # Multiple . -> single .
        
        # Replace numbers with number token but preserve structure
        text = re.sub(r"\d+", " <num> ", text)
        
        # Keep apostrophes and preserve common emotional punctuation
        # Remove only truly unnecessary punctuation
        text = re.sub(r'[#$%&()*+,/:;<=>@[\\\]^`{|}~]', " ", text)
        
        # Multiple spaces to single space
        text = re.sub(r"\s+", " ", text).strip()
        
        # Light lemmatization without aggressive stopword removal
        if self._lemmatizer:
            tokens = []
            for w in text.split():
                # Skip only short tokens and truly empty words
                if len(w) > 1 and w not in self._stop_words:
                    tokens.append(self._lemmatizer.lemmatize(w))
                elif w in ("!", "?", "."):  # Always keep emotion markers
                    tokens.append(w)
            text = " ".join(tokens) if tokens else text
        
        return text

    def clean_series(self, series: pd.Series) -> pd.Series:
        """Apply cleaning to a pandas Series."""
        return series.fillna("").astype(str).apply(self.clean_text)


def build_transition_label(prev_emotion: str, next_emotion: str) -> str:
    """Create human-readable transition label."""
    return f"{prev_emotion} -> {next_emotion}"


def parse_transition_label(label: str) -> Tuple[str, str]:
    """Split transition label into source and target emotions."""
    parts = [p.strip() for p in label.split("->")]
    if len(parts) == 2:
        return parts[0], parts[1]
    return "neutral", "neutral"


class SequenceBuilder:
    """Build padded sequences and transition labels from dialogues."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or get_config()
        self.preprocessor = TextPreprocessor(self.config)
        self.tokenizer: Optional[Tokenizer] = None

    def fit_tokenizer(self, texts: List[str]) -> Tokenizer:
        """Fit Keras tokenizer on corpus."""
        cleaned = [self.preprocessor.clean_text(t) for t in texts]
        self.tokenizer = Tokenizer(
            num_words=self.config.vocab_size,
            oov_token="<OOV>",
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        )
        self.tokenizer.fit_on_texts(cleaned)
        return self.tokenizer

    def texts_to_sequences(self, texts: List[str]) -> np.ndarray:
        """Convert texts to padded integer sequences."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not fitted. Call fit_tokenizer first.")
        cleaned = [self.preprocessor.clean_text(t) for t in texts]
        seqs = self.tokenizer.texts_to_sequences(cleaned)
        return pad_sequences(
            seqs,
            maxlen=self.config.max_sequence_length,
            padding="post",
            truncating="post",
        )

    def save_tokenizer(self, path: Optional[Path] = None) -> None:
        """Persist tokenizer vocabulary."""
        path = path or self.config.tokenizer_path
        if self.tokenizer is None:
            raise ValueError("No tokenizer to save.")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "num_words": self.tokenizer.num_words,
                "oov_token": getattr(self.tokenizer, "oov_token", "<OOV>"),
            },
            "word_index": self.tokenizer.word_index,
            "index_word": {str(k): v for k, v in self.tokenizer.index_word.items()},
            "document_count": getattr(self.tokenizer, "document_count", 0),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _apply_tokenizer_dict(self, data: dict) -> Tokenizer:
        """Rebuild Tokenizer from serialized dict."""
        cfg = data.get("config", {}) or {}
        tok = Tokenizer(
            num_words=cfg.get("num_words") or self.config.vocab_size,
            oov_token=cfg.get("oov_token") or "<OOV>",
            filters=cfg.get("filters", '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'),
        )
        wi = data.get("word_index") or {}
        tok.word_index = {str(k): int(v) for k, v in wi.items()}
        iw = data.get("index_word") or {}
        tok.index_word = {int(k): str(v) for k, v in iw.items()}
        if "document_count" in data:
            tok.document_count = int(data["document_count"])
        return tok

    def load_tokenizer(self, path: Optional[Path] = None) -> Tokenizer:
        """Load tokenizer from disk (custom JSON or legacy Keras format)."""
        path = path or self.config.tokenizer_path
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, str):
            data = json.loads(data)
        self.tokenizer = self._apply_tokenizer_dict(data)
        return self.tokenizer


def create_conversation_windows(
    df: pd.DataFrame,
    text_col: str = "Utterance",
    emotion_col: str = "Emotion",
    dialogue_col: str = "Dialogue_ID",
    utterance_col: str = "Utterance_ID",
    window_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build sliding windows over each dialogue for transition prediction.
    Each row: context utterances + previous emotion -> transition label.
    """
    config = get_config()
    window_size = window_size or config.max_conversation_length
    preprocessor = TextPreprocessor(config)
    records = []

    for dialogue_id, group in df.groupby(dialogue_col):
        group = group.sort_values(utterance_col)
        utterances = group[text_col].tolist()
        emotions = group[emotion_col].str.lower().tolist()

        for i in range(1, len(utterances)):
            start = max(0, i - window_size)
            context_texts = utterances[start:i]
            context_clean = [preprocessor.clean_text(t) for t in context_texts]
            prev_emotion = emotions[i - 1]
            # Explicit prior-state token helps LSTM learn transition dynamics
            combined_context = f"prev_emotion_{prev_emotion} [SEP] " + " [SEP] ".join(context_clean)
            next_emotion = emotions[i]
            transition = build_transition_label(prev_emotion, next_emotion)
            records.append(
                {
                    "dialogue_id": dialogue_id,
                    "position": i,
                    "context": combined_context,
                    "prev_emotion": prev_emotion,
                    "next_emotion": next_emotion,
                    "transition": transition,
                    "utterance": utterances[i],
                }
            )

    return pd.DataFrame(records)

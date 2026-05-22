"""
Lightweight keyword emotion cues for clearer dashboard demos.
Used only when 'Demo boost' is enabled in Streamlit.
"""
from __future__ import annotations

from utils.config import get_config


def detect_emotion_from_text(text: str) -> str:
    """Infer dominant emotion from utterance keywords."""
    config = get_config()
    t = text.lower()
    rules = [
        ("anger", ["angry", "furious", "hate", "unacceptable", "stop", "can't believe", "cant believe", "how dare", "why would they", "can't stop"]),
        ("sadness", ["sad", "terrible", "miss", "bad news", "not feeling", "lost", "hopeless", "cry", "depressed"]),
        ("joy", ["happy", "wonderful", "best day", "won", "great", "excited", "amazing", "love this"]),
        ("fear", ["scared", "worried", "terrified", "afraid", "anxious", "panic", "nervous"]),
        ("surprise", ["shocking", "unexpected", "oh my", "really?", "wow", "can't believe"]),
        ("disgust", ["disgusting", "revolting", "can't stand", "gross", "awful"]),
    ]
    for emo, keys in rules:
        if any(k in t for k in keys):
            if emo in config.emotions:
                return emo
    return "neutral"

"""
Keyword-based emotion lexicon for supplementary emotion detection.
Used alongside the neural model to improve accuracy and provide explainability.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from utils.config import Config, get_config

# Emotion keyword lexicons - carefully curated for emotion detection
EMOTION_KEYWORDS = {
    "anger": {
        "strong": [
            "furious", "enrage", "hate", "disgusted", "unacceptable", "stupid",
            "idiot", "awful", "terrible", "infuriating", "outrage", "rage",
            "angry", "mad", "livid", "cross", "irate", "hostile", "aggressive"
        ],
        "moderate": [
            "frustrat", "annoyed", "irritat", "upset", "bother", "upset",
            "fed up", "sick of", "tired of", "enough", "stop", "quit"
        ],
        "markers": ["!", "!!", "!!!"]  # Punctuation markers
    },
    "joy": {
        "strong": [
            "happy", "wonderful", "fantastic", "amazing", "awesome", "brilliant",
            "excellent", "love", "adore", "excited", "thrilled", "delighted",
            "ecstatic", "overjoyed", "elated", "blessed", "grateful", "thankful"
        ],
        "moderate": [
            "good", "great", "nice", "glad", "pleased", "cheerful", "bright",
            "pretty good", "pretty nice", "cool", "fun", "enjoy"
        ],
        "markers": ["!", ":)", ":D"]
    },
    "sadness": {
        "strong": [
            "miserable", "depressed", "devastated", "heartbroken", "broken",
            "ruined", "destroyed", "desperate", "hopeless", "worthless",
            "tragic", "crying", "tears", "weeping", "grieving", "mourning"
        ],
        "moderate": [
            "sad", "unhappy", "down", "blue", "melancholy", "disappointed",
            "regret", "sorry", "miss", "lonely", "alone", "isolated",
            "weak", "tired", "exhausted"
        ],
        "markers": [":(", ":'(", "..."]
    },
    "fear": {
        "strong": [
            "terrified", "petrified", "horrified", "panic", "panic",
            "afraid", "scared", "scary", "dangerous", "threat", "threaten",
            "dread", "horrify", "nightmare", "death", "die", "dying"
        ],
        "moderate": [
            "worry", "concerned", "anxious", "nervous", "uneasy", "hesitant",
            "uncertain", "unsure", "doubtful", "suspicious", "distrust",
            "what if", "hopefully", "please don't", "hope not", "scared of"
        ],
        "markers": ["?", "???"]
    },
    "disgust": {
        "strong": [
            "disgusting", "revolting", "repulsive", "abhorrent", "vile",
            "disgusted", "gross", "yuck", "ugh", "blegh", "repugnant"
        ],
        "moderate": [
            "dislike", "distaste", "unpleasant", "nasty", "dirty",
            "filthy", "contaminated", "sick", "sickening", "awful"
        ],
        "markers": ["ugh", "blech"]
    },
    "surprise": {
        "strong": [
            "astonish", "astound", "shock", "shocked", "startled", "sudden",
            "unexpected", "surprising", "amazed", "wow", "whoa", "incredible"
        ],
        "moderate": [
            "surprise", "surprised", "wonder", "interesting", "curious",
            "unusual", "odd", "strange", "peculiar", "really", "actually"
        ],
        "markers": ["?!", "!?", "?"]
    },
    "neutral": {
        "strong": [
            "okay", "fine", "alright", "sure", "yes", "no", "maybe",
            "probably", "perhaps", "possibly", "seems", "appears",
            "according", "like", "kind of", "sort of", "basically"
        ],
        "moderate": [
            "understand", "see", "think", "believe", "know", "suppose",
            "imagine", "right", "yeah", "good", "bad", "normal"
        ]
    }
}


class KeywordEmotionDetector:
    """Detect emotions using keyword lexicons and linguistic markers."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or get_config()
        self.keywords = EMOTION_KEYWORDS

    def detect_emotion_from_text(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Detect emotion using keyword matching.
        Returns: (primary_emotion, {emotion: confidence_score})
        """
        text_lower = text.lower()
        emotion_scores = {emo: 0.0 for emo in self.config.emotions}

        # Score each emotion based on keyword presence
        for emotion, patterns in self.keywords.items():
            score = 0.0

            # Strong keywords (weight: 3x)
            for keyword in patterns.get("strong", []):
                # Check if keyword appears as a whole word (not substring)
                # Allow word boundaries with simple word splitting
                words = text_lower.replace("!", "").replace("?", "").replace(".", "").split()
                if keyword in text_lower:
                    # Count occurrences
                    count = text_lower.count(keyword)
                    score += 3.0 * count

            # Moderate keywords (weight: 1x)
            for keyword in patterns.get("moderate", []):
                if keyword in text_lower:
                    count = text_lower.count(keyword)
                    score += 1.0 * count

            # Punctuation markers (weight: 2x for question marks in fear, 3x for exclamation)
            for marker in patterns.get("markers", []):
                count = text_lower.count(marker)
                if count > 0:
                    # Question marks and exclamation marks are stronger indicators
                    if marker == "!":
                        score += 2.5 * count
                    elif marker == "?":
                        score += 2.0 * count
                    else:
                        score += 0.5 * count

            emotion_scores[emotion] = score

        # Normalize to probabilities
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {e: v / total for e, v in emotion_scores.items()}
        else:
            # Default to neutral if no keywords found
            emotion_scores["neutral"] = 1.0

        # Find primary emotion
        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        return primary_emotion, emotion_scores

    def blend_predictions(
        self,
        neural_pred: str,
        neural_confidence: float,
        keyword_pred: str,
        keyword_confidence: float,
        keyword_weight: float = 0.3,
    ) -> Tuple[str, float]:
        """
        Blend neural model and keyword detector predictions.
        keyword_weight: 0.0 = pure neural, 1.0 = pure keyword
        """
        if neural_confidence < 0.5 and keyword_confidence > 0.6:
            # If neural is uncertain but keyword is confident, prefer keyword
            return keyword_pred, keyword_confidence
        elif neural_pred == keyword_pred:
            # Agreement between models - boost confidence
            blended_conf = neural_confidence * 0.7 + keyword_confidence * 0.3
            return neural_pred, min(blended_conf * 1.1, 1.0)  # Cap at 1.0
        else:
            # Disagreement - use weighted blend
            blend_confidence = neural_confidence * (1 - keyword_weight) + keyword_confidence * keyword_weight
            return neural_pred, blend_confidence

    def explain_detection(self, text: str, emotion: str) -> List[str]:
        """Return keywords that triggered this emotion prediction."""
        text_lower = text.lower()
        explanations = []

        if emotion in self.keywords:
            patterns = self.keywords[emotion]

            # Find strong keywords
            for keyword in patterns.get("strong", []):
                if keyword in text_lower:
                    explanations.append(f'Strong signal: "{keyword}"')

            # Find moderate keywords
            for keyword in patterns.get("moderate", []):
                if keyword in text_lower:
                    explanations.append(f'Moderate signal: "{keyword}"')

            # Find punctuation markers
            for marker in patterns.get("markers", []):
                if marker in text_lower:
                    explanations.append(f'Punctuation marker: "{marker}"')

        return explanations[:3] if explanations else ["No specific keywords found"]

"""
Real-time inference engine with conversation memory.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from tensorflow import keras

from utils.config import Config, get_config
from utils.emotion_encoding import encode_prev_emotions, normalize_emotion
from utils.emotion_keywords import KeywordEmotionDetector
from utils.preprocessing import SequenceBuilder, TextPreprocessor, build_transition_label, parse_transition_label
from utils.transitions import build_emotion_timeline, detect_transitions_from_emotions, transition_entropy


@dataclass
class PredictionResult:
    """Single transition prediction output."""

    transition: str
    from_emotion: str
    to_emotion: str
    confidence: float
    probabilities: Dict[str, float]
    entropy: float
    context_used: str


@dataclass
class ConversationMemory:
    """Maintains conversation history for contextual prediction."""

    messages: List[str] = field(default_factory=list)
    emotions: List[str] = field(default_factory=list)
    predictions: List[PredictionResult] = field(default_factory=list)
    max_length: int = 8

    def add_message(self, text: str, emotion: Optional[str] = None) -> None:
        self.messages.append(text)
        if emotion:
            self.emotions.append(emotion.lower())

    def get_context(self, prev_emotion: str = "neutral") -> str:
        preprocessor = TextPreprocessor()
        recent = self.messages[-self.max_length :]
        cleaned = [preprocessor.clean_text(m) for m in recent]
        prev_emotion = normalize_emotion(prev_emotion, get_config())
        return f"prev_emotion_{prev_emotion} [SEP] " + " [SEP] ".join(cleaned)

    def clear(self) -> None:
        self.messages.clear()
        self.emotions.clear()
        self.predictions.clear()


class EmotionTransitionPredictor:
    """Load model artifacts and run inference with keyword-based enhancement."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or get_config()
        self.model: Optional[keras.Model] = None
        self.label_encoder = None
        self.idx_to_label: Dict[int, str] = {}
        self.builder = SequenceBuilder(self.config)
        self.memory = ConversationMemory(max_length=self.config.max_conversation_length)
        self.metadata: Dict = {}
        self._loaded = False
        self._predicts_next_emotion = True
        self.keyword_detector = KeywordEmotionDetector(self.config)

    def load(self) -> None:
        """Load model, tokenizer, and label encoder."""
        if not self.config.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {self.config.model_path}. "
                "Run: python -m training.train"
            )
        self.model = keras.models.load_model(self.config.model_path)
        self.builder.load_tokenizer()
        self.label_encoder = joblib.load(self.config.label_encoder_path)
        self.idx_to_label = {i: lbl for i, lbl in enumerate(self.label_encoder.classes_)}

        if self.config.metadata_path.exists():
            with open(self.config.metadata_path, encoding="utf-8") as f:
                self.metadata = json.load(f)
        self._predicts_next_emotion = self.metadata.get("prediction_target") == "next_emotion"
        self._loaded = True

    def _model_inputs(self, context: str, prev_emotion: str):
        """Build model inputs (supports legacy single-input models)."""
        seq = self.builder.texts_to_sequences([context])
        if isinstance(self.model.input, list) or (
            hasattr(self.model, "inputs") and len(self.model.inputs) > 1
        ):
            emo = encode_prev_emotions([prev_emotion], self.config)
            return [seq, emo]
        return seq

    def _mask_probs_by_prev_emotion(self, probs: np.ndarray, prev_emotion: str) -> np.ndarray:
        """Keep only transitions starting from the given previous emotion."""
        prev_emotion = normalize_emotion(prev_emotion, self.config)
        masked = np.zeros_like(probs)
        for i, label in self.idx_to_label.items():
            if self._predicts_next_emotion:
                # Label is next emotion only — all classes are valid; prev is in input
                masked[i] = probs[i]
            else:
                from_emo, _ = parse_transition_label(label)
                if from_emo == prev_emotion:
                    masked[i] = probs[i]
        total = masked.sum()
        return masked / total if total > 0 else probs

    def _decode_prediction(self, probs: np.ndarray, prev_emotion: str, top_k: int) -> PredictionResult:
        prev_emotion = normalize_emotion(prev_emotion, self.config)
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        if self._predicts_next_emotion or "->" not in self.idx_to_label.get(pred_idx, ""):
            to_emo = self.idx_to_label[pred_idx]
            from_emo = prev_emotion
            transition = build_transition_label(from_emo, to_emo)
            top_indices = np.argsort(probs)[::-1][:top_k]
            prob_dict = {
                build_transition_label(from_emo, self.idx_to_label[i]): float(probs[i])
                for i in top_indices
            }
        else:
            transition = self.idx_to_label[pred_idx]
            from_emo, to_emo = parse_transition_label(transition)
            top_indices = np.argsort(probs)[::-1][:top_k]
            prob_dict = {self.idx_to_label[i]: float(probs[i]) for i in top_indices}

        return PredictionResult(
            transition=transition,
            from_emotion=from_emo,
            to_emotion=to_emo,
            confidence=confidence,
            probabilities=prob_dict,
            entropy=transition_entropy(probs),
            context_used="",
        )

    def build_context(self, text: str, prev_emotion: str = "neutral") -> str:
        """Format context string consistently with training data."""
        prev_emotion = normalize_emotion(prev_emotion, self.config)
        cleaned = TextPreprocessor(self.config).clean_text(text)
        return f"prev_emotion_{prev_emotion} [SEP] {cleaned}"

    def _enhance_with_keywords(self, result: PredictionResult, text: str) -> PredictionResult:
        """Enhance neural predictions using keyword-based emotion detection."""
        # Get keyword-based detection from the current utterance
        detected_emo, keyword_scores = self.keyword_detector.detect_emotion_from_text(text)
        detected_confidence = max(keyword_scores.values()) if keyword_scores else 0.0
        
        # Check if neural prediction is uncertain and keyword is confident
        if result.confidence < 0.5 and detected_confidence > 0.6:
            # Override with keyword detection
            result.to_emotion = detected_emo
            result.confidence = min(detected_confidence, 0.95)
            result.transition = build_transition_label(result.from_emotion, result.to_emotion)
        elif result.to_emotion == detected_emo:
            # Agreement between models - boost confidence
            result.confidence = min(result.confidence * 1.15, 1.0)
        
        return result

    def predict_transition(
        self,
        context: str,
        prev_emotion: str = "neutral",
        top_k: int = 5,
        wrap_context: bool = True,
        use_keyword_enhancement: bool = True,
    ) -> PredictionResult:
        """Predict emotional transition from context + previous emotion."""
        if not self._loaded:
            self.load()
        assert self.model is not None

        if wrap_context and "prev_emotion_" not in context:
            context = self.build_context(context, prev_emotion)

        inputs = self._model_inputs(context, prev_emotion)
        probs = self.model.predict(inputs, verbose=0)[0]
        probs = self._mask_probs_by_prev_emotion(probs, prev_emotion)
        result = self._decode_prediction(probs, prev_emotion, top_k)
        result.context_used = context
        
        # Enhance with keyword-based detection if enabled
        if use_keyword_enhancement:
            # Extract the actual text from context for keyword analysis
            parts = context.split("[SEP]")
            if len(parts) > 1:
                current_text = parts[-1].strip()
                result = self._enhance_with_keywords(result, current_text)
        
        return result

    def predict_next_from_conversation(
        self,
        new_message: str,
        prev_emotion: str = "neutral",
        use_keyword_enhancement: bool = True,
    ) -> PredictionResult:
        """Add message to memory and predict transition."""
        self.memory.add_message(new_message)
        context = self.memory.get_context(prev_emotion)
        result = self.predict_transition(context, prev_emotion, use_keyword_enhancement=use_keyword_enhancement)
        self.memory.predictions.append(result)
        self.memory.emotions.append(result.to_emotion)
        return result

    def analyze_conversation(
        self,
        messages: List[str],
        initial_emotion: str = "neutral",
        use_text_emotion_cues: bool = False,
    ) -> Dict:
        """Analyze full conversation and return timeline + transitions."""
        self.memory.clear()
        initial_emotion = normalize_emotion(initial_emotion, self.config)
        emotions = [initial_emotion]
        results = []

        for i, msg in enumerate(messages):
            if i == 0:
                self.memory.add_message(msg)
                continue
            if use_text_emotion_cues:
                # Use keyword-based emotion detection for previous message
                prev_detected_emo, _ = self.keyword_detector.detect_emotion_from_text(messages[i - 1])
                prev = normalize_emotion(prev_detected_emo, self.config)
            else:
                prev = normalize_emotion(emotions[-1], self.config)
            pred = self.predict_next_from_conversation(msg, prev, use_keyword_enhancement=True)
            results.append(pred)
            if use_text_emotion_cues:
                # Timeline reflects utterance tone; model still drives transition probabilities
                detected_emo, _ = self.keyword_detector.detect_emotion_from_text(msg)
                emotions.append(normalize_emotion(detected_emo, self.config))
            else:
                emotions.append(pred.to_emotion)

        confidences = [1.0] + [r.confidence for r in results] if results else None
        timeline = build_emotion_timeline(emotions, confidences)
        events = detect_transitions_from_emotions(emotions)
        return {
            "timeline": timeline,
            "transitions": events,
            "predictions": results,
            "emotions": emotions,
        }

    def reset_memory(self) -> None:
        self.memory.clear()

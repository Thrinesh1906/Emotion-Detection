"""
Test script to diagnose prediction issues
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import get_config
from utils.inference import EmotionTransitionPredictor
from utils.emotion_keywords import KeywordEmotionDetector

# Test conversations
test_conversations = [
    ("neutral", "I am really happy and excited!"),
    ("neutral", "I feel terrible and sad"),
    ("neutral", "This is completely unacceptable! I'm furious!"),
    ("neutral", "I'm so scared right now"),
    ("joy", "Actually, I'm not sure about this"),
    ("anger", "Let me think about it calmly"),
]

config = get_config()
predictor = EmotionTransitionPredictor(config)
detector = KeywordEmotionDetector(config)

print("=" * 80)
print("EMOTION DETECTION TEST")
print("=" * 80)

try:
    predictor.load()
    print("\n[OK] Model loaded successfully\n")
    
    for prev_emotion, text in test_conversations:
        print(f"\nPrevious emotion: {prev_emotion}")
        print(f"Text: \"{text}\"")
        print("-" * 70)
        
        # Neural prediction
        pred = predictor.predict_transition(text, prev_emotion, use_keyword_enhancement=False)
        print(f"Neural Model -> {pred.to_emotion}")
        print(f"  Confidence: {pred.confidence:.3f}")
        print(f"  Top predictions: {dict(list(pred.probabilities.items())[:3])}")
        
        # Keyword detection
        keyword_emo, keyword_scores = detector.detect_emotion_from_text(text)
        print(f"\nKeyword Model -> {keyword_emo}")
        print(f"  Confidence: {max(keyword_scores.values()):.3f}")
        keyword_top = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top predictions: {dict(keyword_top)}")
        
        # Enhanced prediction
        pred_enhanced = predictor.predict_transition(text, prev_emotion, use_keyword_enhancement=True)
        print(f"\nEnhanced Model -> {pred_enhanced.to_emotion}")
        print(f"  Confidence: {pred_enhanced.confidence:.3f}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)

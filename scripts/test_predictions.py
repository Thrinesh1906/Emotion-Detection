import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.inference import EmotionTransitionPredictor

p = EmotionTransitionPredictor()
p.load()
msgs = [
    "Hey, how are you doing today?",
    "Actually, I am not feeling great.",
    "I just found out some really bad news.",
    "Why would they do something like that?",
]
a = p.analyze_conversation(msgs, "neutral", use_text_emotion_cues=True)
print("Emotions (demo boost):", a["emotions"])
for r in a["predictions"]:
    print(" ", r.transition, f"{r.confidence:.0%}")
print("---")
tests = [
    ("I cant believe you said that!", "neutral"),
    ("I am so happy today!", "joy"),
    ("I am terrified something bad will happen", "fear"),
]
for text, prev in tests:
    r = p.predict_transition(text, prev)
    print(f"{prev} -> {r.to_emotion} ({r.transition}) {r.confidence:.0%}")

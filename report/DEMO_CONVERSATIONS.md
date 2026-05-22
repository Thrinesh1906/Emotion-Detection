# Demo Conversations — Expected Outputs (After Retrain)

> **Important:** Retrain after the model fix:
> ```bash
> python -m training.train --quick
> ```
> Set **Current Emotion** in the sidebar to match each turn’s *previous* state.

The model predicts **next emotion** from:
- conversation text (context window)
- **previous emotion** (sidebar dropdown)

Displayed transition = `previous → predicted next` (e.g. `neutral → sadness`).

---

## Demo 1 — Neutral → Sadness → Anger (escalation)

**Sidebar emotion:** start `neutral`, then follow table below.

**Conversation (paste in Simulator):**
```
Hey, how are you doing today?
Actually, I'm not feeling great.
I just found out some really bad news.
Why would they do something like that?
I can't stop thinking about it.
```

| Step | Message (summary) | Prev emotion (sidebar) | Typical prediction |
|------|-------------------|------------------------|------------------|
| 1 | Hey, how are you... | neutral | (no transition yet — sets context) |
| 2 | Not feeling great | neutral | neutral → **sadness** |
| 3 | Bad news | sadness | sadness → **sadness** or **anger** |
| 4 | Why would they... | sadness/anger | → **anger** |
| 5 | Can't stop thinking | anger | anger → **sadness** or **anger** |

**Timeline should show:** neutral → sadness → anger (not all neutral).

---

## Demo 2 — Joy → Neutral (calming down)

**Initial:** `joy`

```
We won the competition!
This is the best day ever!
Okay let me calm down now.
Thanks everyone for your support.
```

| Step | Prev | Typical next |
|------|------|--------------|
| 2 | joy | joy → **joy** |
| 3 | joy | joy → **neutral** |
| 4 | neutral | neutral → **neutral** or **joy** |

---

## Demo 3 — Fear / Stress

**Initial:** `fear`

```
Did you hear about the deadline?
I am really worried we will fail.
Everything is going wrong at once.
I cannot handle this pressure anymore.
```

| Step | Prev | Typical next |
|------|------|--------------|
| 2 | fear | fear → **fear** or **sadness** |
| 3 | fear | fear → **sadness** |
| 4 | sadness | sadness → **anger** or **sadness** |

---

## Live Prediction — Single lines

| Type this | Sidebar prev | Expect (examples) |
|-----------|--------------|-------------------|
| I am so happy today! | joy | joy → **joy** |
| I can't believe you did that! | neutral | neutral → **anger** |
| I'm terrified something bad will happen | fear | fear → **fear** |
| That's wonderful news! | neutral | neutral → **joy** |
| I feel empty and hopeless | sadness | sadness → **sadness** |

---

## All 49 possible transition *labels* (display format)

For emotions: anger, disgust, fear, joy, neutral, sadness, surprise:

```
anger → anger      anger → disgust    anger → fear       anger → joy
anger → neutral    anger → sadness    anger → surprise

disgust → (all 7)  fear → (all 7)     joy → (all 7)
neutral → (all 7)  sadness → (all 7)  surprise → (all 7)
```

**7 × 7 = 49** transition strings. The model outputs **7 next emotions**; the UI builds `prev → next`.

---

## Why you saw only `neutral → neutral`

1. Old model ignored sidebar **Current Emotion**.
2. Training used **49 imbalanced classes**; model guessed the most common class.
3. **Fix:** model now uses previous emotion + predicts next emotion (7 classes) + class weights.

**You must retrain** and **refresh** the Streamlit browser after retraining.

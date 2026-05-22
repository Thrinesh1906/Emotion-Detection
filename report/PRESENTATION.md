# PPT Presentation Content
## Context-Aware Emotional Transition Detection using LSTM Networks

*Use each `---` section as one PowerPoint slide. Suggested: 18–22 slides, dark theme template.*

---

## Slide 1: Title

**Context-Aware Emotional Transition Detection using LSTM Networks**

Final Year Project  
[Student Name] | [Roll No]  
[Department], [College Name]  
Guide: [Guide Name]  
2025–2026

---

## Slide 2: Agenda

1. Introduction & Motivation  
2. Problem Statement  
3. Objectives  
4. Literature Survey  
5. Proposed System  
6. Dataset (MELD)  
7. Methodology  
8. Model Architecture  
9. LSTM & Hidden State  
10. Implementation  
11. Results & Demo  
12. Conclusion & Future Work  

---

## Slide 3: Introduction

- Emotions in conversations **change over time**
- Traditional AI: one sentence → one emotion label
- **Our approach:** sequence of messages → **emotional transition**
- Example: `neutral` → `anger` after escalating dialogue

**Speaker notes:** Open with a relatable example—a customer support chat that starts polite then becomes angry.

---

## Slide 4: Motivation

| Limitation (Existing) | Our Solution |
|----------------------|--------------|
| No conversation memory | LSTM hidden state |
| Static labels only | Transition prediction |
| No visual analytics | Streamlit dashboard |
| Academic-only models | Full deployable pipeline |

---

## Slide 5: Problem Statement

> How can we detect **emotional transitions** in multi-turn conversations by learning **contextual dependencies** across utterance sequences using deep recurrent neural networks?

---

## Slide 6: Objectives

**Primary:**
- Sequence-based emotion flow analysis  
- LSTM/BiLSTM transition prediction  
- Contextual memory demonstration  

**Secondary:**
- Interactive dashboard  
- Training visualizations  
- REST API deployment  

---

## Slide 7: Literature Survey

- **LSTM** (Hochreiter, 1997) — Gated memory  
- **EmotionLines / MELD** (2019) — Dialogue emotions  
- **GoEmotions** (2020) — Fine-grained but non-dialogue  
- **BERT** models — Contextual but heavy  

**Gap:** Few systems predict explicit **transition classes** with full UI deployment.

---

## Slide 8: Existing vs Proposed System

```
EXISTING:  Sentence → Classifier → Emotion
PROPOSED:  Dialogue Window → BiLSTM → Transition → Timeline UI
```

---

## Slide 9: Dataset — MELD

- **M**ultimodal **E**motionLines **D**ataset  
- Source: TV series *Friends* dialogues  
- 7 emotions, 13K+ utterances  
- Fields: Utterance, Emotion, Dialogue_ID, Utterance_ID  

*[Insert sample dialogue screenshot]*

---

## Slide 10: Methodology

1. Download & load MELD  
2. Build sliding conversation windows  
3. Create transition labels (`prev → next`)  
4. Preprocess text (NLTK)  
5. Tokenize & pad sequences  
6. Train BiLSTM  
7. Evaluate & deploy  

---

## Slide 11: System Architecture

*[Insert diagram from report/DIAGRAMS.md]*

Layers: Streamlit UI → Inference Engine → BiLSTM → MELD Data

---

## Slide 12: Model Architecture

| Layer | Details |
|-------|---------|
| Input | Padded tokens (50) |
| Embedding | 128 dimensions |
| BiLSTM | 128 + 64 units |
| Dropout | 0.4 |
| Dense | 128 → 64 |
| Output | Softmax (transitions) |

---

## Slide 13: LSTM — Why & How

**Gates:** Forget, Input, Output  
**Hidden state:** Conversational memory vector  
**Cell state:** Long-term emotional context  

*"LSTM remembers that the conversation started calm before predicting anger."*

---

## Slide 14: Sequence Learning

```
Turn 1: "Hi!"           → context builds
Turn 2: "Bad news..."   → h_t updates
Turn 3: "I'm furious!"  → predicts neutral → anger
```

Bidirectional: uses full window context.

---

## Slide 15: Implementation Stack

- Python, TensorFlow/Keras  
- FastAPI, Streamlit  
- Plotly, Matplotlib, NLTK  
- Modular: `utils/`, `models/`, `training/`

**Live demo command:**
`streamlit run streamlit_app.py`

---

## Slide 16: Results

*[Insert training_accuracy_loss.png]*

- Validation accuracy: [from metadata]  
- Test accuracy: [from metadata]  
- Macro F1: [from evaluation_metrics.json]  
- Confusion matrix: [insert image]  

---

## Slide 17: Dashboard Features

- Live prediction panel  
- Conversation simulator  
- Emotion timeline (Plotly)  
- Confidence bar charts  
- Transition heatmap  
- PDF export  

*[Insert screenshots from screenshots/ folder]*

---

## Slide 18: Bonus Features

- FastAPI REST endpoints  
- Conversation memory  
- RNN vs LSTM comparison (`--model rnn`)  
- Multilingual UI labels  
- PDF analytics export  

---

## Slide 19: Conclusion

- Built complete **context-aware** emotion transition system  
- BiLSTM captures **sequence dependency**  
- MELD provides realistic conversational training data  
- Deployed **industry-style** dashboard + API  

---

## Slide 20: Future Scope

- Multimodal (audio/video)  
- Transformer/BERT comparison  
- Voice input pipeline  
- Attention interpretability  

---

## Slide 21: References

1. Poria et al., MELD, ACL 2019  
2. Hochreiter & Schmidhuber, LSTM, 1997  
3. Chatterjee et al., Emotion Survey, 2019  

---

## Slide 22: Thank You

**Questions?**

GitHub/Project folder: `RNN/`  
Demo: `streamlit run streamlit_app.py`

---

## Appendix Slides (Backup)

### Backup A: Hyperparameters Table
### Backup B: Confusion Matrix Deep Dive  
### Backup C: API Endpoint List  
### Backup D: Viva-style Q&A summary  

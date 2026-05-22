# Viva Questions & Answers

## Deep Learning & LSTM

**Q1: Why did you choose LSTM over a simple feedforward neural network?**  
**A:** Feedforward networks treat each input independently without memory. Emotions in conversations depend on **prior utterances**. LSTM maintains hidden and cell states across timesteps, enabling sequence-dependent transition prediction.

**Q2: Explain the hidden state in LSTM.**  
**A:** The hidden state \( h_t \) is a vector summarizing all relevant information from previous timesteps up to \( t \). It acts as **short-term conversational memory**, influencing how the model interprets the current utterance and predicts the next emotional transition.

**Q3: What is the cell state and how does it differ from the hidden state?**  
**A:** The cell state \( c_t \) is the long-term memory highway in LSTM. Gates add or remove information. The hidden state \( h_t \) is a filtered version of \( c_t \) used for output and next-step computation.

**Q4: Why use Bidirectional LSTM?**  
**A:** BiLSTM processes the sequence in both forward and backward directions, capturing context from utterances **before and after** within the window—useful when later words clarify emotional tone.

**Q5: What is the vanishing gradient problem?**  
**A:** In deep or long sequences, gradients shrink during backpropagation, preventing vanilla RNNs from learning long-range dependencies. LSTM gates mitigate this by providing additive memory paths.

---

## Project-Specific

**Q6: What is emotional transition detection?**  
**A:** Instead of classifying a single sentence's emotion, we predict the **change** from one emotional state to another (e.g., `neutral → anger`) based on conversational context.

**Q7: Why did you select the MELD dataset?**  
**A:** MELD provides multiparty dialogues with emotion labels per utterance and dialogue/utterance IDs—essential for constructing realistic **transition samples** with conversational structure.

**Q8: How do you create transition labels from MELD?**  
**A:** For each dialogue, utterances are sorted by ID. For consecutive pairs \( (e_{i-1}, e_i) \), the label is `e_{i-1} -> e_i`. The model input is the concatenated text of prior utterances in a sliding window.

**Q9: How do you prevent data leakage?**  
**A:** Train/validation/test splits are done at the **dialogue level**, not utterance level, so no dialogue appears in multiple splits.

**Q10: What does the embedding layer do?**  
**A:** It maps integer token indices to dense vectors in \( \mathbb{R}^{128} \), capturing semantic similarity between words for emotion-related language patterns.

---

## Model & Training

**Q11: Why use dropout?**  
**A:** Dropout randomly deactivates neurons during training to prevent overfitting—a common issue when many transition classes have limited examples.

**Q12: What loss function did you use?**  
**A:** Sparse categorical cross-entropy, suitable for integer-encoded multi-class transition labels with softmax output.

**Q13: What is softmax output?**  
**A:** Softmax converts raw logits into a probability distribution over all transition classes, summing to 1. The highest probability class is the predicted transition.

**Q14: How do you handle class imbalance?**  
**A:** We report macro F1 score, use early stopping on validation loss, and acknowledge that frequent transitions like `neutral → neutral` dominate—future work can use class weighting.

**Q15: What callbacks did you use during training?**  
**A:** EarlyStopping (stop when validation loss plateaus), ModelCheckpoint (save best model), ReduceLROnPlateau (adaptive learning rate).

---

## System Design

**Q16: Explain conversation memory in your system.**  
**A:** The `ConversationMemory` class stores recent messages in a sliding window (default 8). Each new prediction uses accumulated context, simulating real-time dialogue analysis.

**Q17: What is the difference between your API and Streamlit UI?**  
**A:** FastAPI provides REST endpoints for programmatic integration; Streamlit provides a visual dashboard for demonstration, analytics, and live testing.

**Q18: How does real-time prediction work?**  
**A:** User text is cleaned, tokenized, padded to length 50, fed through the saved BiLSTM model, and the argmax softmax output gives the predicted transition with confidence.

---

## Evaluation & Results

**Q19: What metrics do you use for evaluation?**  
**A:** Accuracy, macro F1-score, classification report, and confusion matrix visualizations on the held-out test set.

**Q20: What is a confusion matrix?**  
**A:** A table showing actual vs predicted transition classes, revealing which transitions are commonly confused (e.g., `neutral → sadness` vs `neutral → anger`).

**Q21: What is prediction entropy?**  
**A:** Shannon entropy of the output probability distribution—high entropy means the model is uncertain across many transitions; low entropy indicates confident predictions.

---

## Comparison & Advanced

**Q22: Compare RNN and LSTM for this project.**  
**A:** Vanilla RNN is simpler but struggles with long contexts due to vanishing gradients. LSTM's gated architecture better preserves emotional context across multiple dialogue turns.

**Q23: How would BERT improve this project?**  
**A:** BERT provides contextual token embeddings pre-trained on large corpora, potentially improving semantic understanding—but with higher compute and less explicit recurrent memory interpretation.

**Q24: What are future enhancements?**  
**A:** Multimodal fusion (MELD audio/video), Transformer models, voice input pipeline, multilingual support, and attention-based interpretability.

**Q25: What is the real-world application of this project?**  
**A:** Mental health monitoring chatbots, customer service escalation detection, social media conversation moderation, and empathetic AI assistants that respond appropriately to emotional shifts.

---

## Quick Definitions

| Term | Definition |
|------|------------|
| Tokenization | Converting text to integer word indices |
| Padding | Making all sequences equal length (50) |
| Sequence learning | Model learns from ordered data |
| Transition class | Label like `joy → sadness` |
| Context window | Last k utterances used as input |

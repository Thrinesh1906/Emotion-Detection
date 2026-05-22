# Problem Statement

Traditional emotion recognition systems classify **individual text segments** in isolation. They fail to capture how emotional states **evolve across multi-turn conversations**—for example, when a dialogue shifts from *neutral* small talk to *anger* during conflict escalation.

**Formal Problem:** Given an ordered sequence of conversational utterances \( U = \{u_1, u_2, \ldots, u_t\} \) and the previous emotional state \( e_{t-1} \), predict the emotional transition \( e_{t-1} \rightarrow e_t \) where \( e_i \in \mathcal{E} \) and \( \mathcal{E} \) is a finite emotion label set.

**Challenges:**
1. Contextual dependency across variable-length dialogues  
2. Class imbalance in transition types (49 classes for 7 emotions)  
3. Semantic ambiguity in short utterances without history  
4. Need for deployable real-time inference with memory  

**Proposed Solution:** A Bidirectional LSTM sequence classifier trained on the MELD conversational corpus with sliding context windows, deployed via Streamlit and FastAPI.

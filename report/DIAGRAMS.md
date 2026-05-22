# Architecture Diagrams & Flowcharts

## 1. High-Level System Architecture

```mermaid
flowchart TB
    subgraph Client["Presentation Layer"]
        ST[Streamlit Dashboard]
        API[FastAPI REST API]
    end
    subgraph Service["Inference Layer"]
        INF[EmotionTransitionPredictor]
        MEM[ConversationMemory]
    end
    subgraph ML["Machine Learning Layer"]
        PP[Preprocessor]
        TOK[Tokenizer]
        LSTM[BiLSTM Model]
    end
    subgraph Data["Data Layer"]
        MELD[MELD Dataset]
        ART[Saved Model Artifacts]
    end
    ST --> INF
    API --> INF
    INF --> MEM
    INF --> PP --> TOK --> LSTM
    LSTM --> ART
    MELD --> PP
```

## 2. Data Processing Flowchart

```mermaid
flowchart TD
    A[Download MELD CSV] --> B[Load Raw Utterances]
    B --> C[Group by Dialogue_ID]
    C --> D[Sort by Utterance_ID]
    D --> E[Create Sliding Windows]
    E --> F[Label Transitions: prev → next]
    F --> G[Text Cleaning + Lemmatization]
    G --> H[Tokenization + Padding]
    H --> I[Label Encoding]
    I --> J[Train / Val / Test Split by Dialogue]
```

## 3. LSTM Training Pipeline

```mermaid
flowchart LR
    X[Input Sequences] --> E[Embedding]
    E --> B1[BiLSTM Layer 1]
    B1 --> D1[Dropout]
    D1 --> B2[BiLSTM Layer 2]
    B2 --> D2[Dropout]
    D2 --> FC[Dense Layers]
    FC --> S[Softmax]
    S --> L[Cross-Entropy Loss]
    L --> O[Adam Optimizer]
```

## 4. Inference Flow

```mermaid
sequenceDiagram
    participant U as User
    participant UI as Streamlit
    participant P as Predictor
    participant M as BiLSTM Model
    U->>UI: Enter message
    UI->>P: add_message()
    P->>P: Build context window
    P->>M: predict(sequence)
    M-->>P: probability vector
    P-->>UI: transition + confidence
    UI-->>U: Timeline + bars
```

## 5. Emotion State Machine (Conceptual)

```mermaid
stateDiagram-v2
    [*] --> neutral
    neutral --> joy: positive news
    neutral --> anger: conflict
    joy --> sadness: disappointment
    anger --> neutral: resolution
    fear --> surprise: unexpected event
    sadness --> joy: good news
    surprise --> neutral: processing
```

## 6. Hidden State Memory Concept

```
Timestep:    t=1        t=2         t=3         t=4
Input:       x_1        x_2         x_3         x_4
Hidden:      h_1  -->   h_2   -->   h_3   -->   h_4
             │          │           │           │
Context:   "hello"   "I'm upset"  "why?"    "stop it"
                                              ↓
Prediction:                          neutral → anger
```

The hidden state **h_t** accumulates emotional context from all prior utterances in the window, enabling transition detection that depends on conversational history.

# Abstract

Emotion artificial intelligence has traditionally focused on classifying isolated sentences into static emotional categories. However, human conversations exhibit **dynamic emotional evolution**—feelings shift gradually or abruptly across dialogue turns. This project presents **Context-Aware Emotional Transition Detection using LSTM Networks**, a deep learning system that models emotional flow as a **sequence learning** problem.

The proposed system accepts a window of conversational utterances, preprocesses text using NLP normalization, and encodes tokens through an **embedding layer**. A **Bidirectional Long Short-Term Memory (BiLSTM)** network captures contextual dependencies via hidden states that function as conversational memory. The model predicts **transition classes** such as *happy → neutral*, *neutral → angry*, and *fear → surprise* using a **softmax** output layer.

Training and evaluation are conducted on the publicly available **MELD (Multimodal EmotionLines Dataset)**, comprising over 13,000 utterances from multiparty dialogues. Experimental results demonstrate that sequence-aware modeling outperforms context-agnostic baselines for transition prediction. A **Streamlit dashboard** provides real-time inference, emotion timelines, confidence visualization, and analytics export.

**Keywords:** LSTM, Emotion Recognition, Sequence Learning, MELD, Deep Learning, Conversational AI, Transition Detection

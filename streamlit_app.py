"""
Context-Aware Emotional Transition Detection — Streamlit Dashboard
Run: streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import get_config
from utils.inference import EmotionTransitionPredictor
from utils.pdf_export import export_analytics_pdf
from utils.transitions import emotion_transition_matrix, top_k_transitions
from utils.visualization import (
    plotly_confidence_bars,
    plotly_emotion_timeline,
    plotly_transition_heatmap,
)

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Transition AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%); }
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header { color: #a0aec0; font-size: 1rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stSidebar { background: #0d1117 !important; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; color: #667eea; }
    .stButton>button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    .chat-bubble {
        background: rgba(102,126,234,0.15);
        border-left: 3px solid #667eea;
        padding: 0.8rem 1rem;
        border-radius: 0 12px 12px 0;
        margin: 0.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

config = get_config()

# ─── Session state ─────────────────────────────────────────────────────────────
if "predictor" not in st.session_state:
    st.session_state.predictor = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_emotions" not in st.session_state:
    st.session_state.chat_emotions = ["neutral"]
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


@st.cache_resource
def load_predictor():
    pred = EmotionTransitionPredictor(config)
    pred.load()
    return pred


def get_predictor_safe():
    try:
        return load_predictor()
    except FileNotFoundError:
        return None


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=64)
    st.markdown("### 🧠 Emotion Transition AI")
    st.markdown("*LSTM-powered contextual analysis*")
    st.divider()
    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🔮 Live Prediction",
            "💬 Conversation Simulator",
            "📊 Analytics Dashboard",
            "📈 Training Results",
            "🔬 Model Architecture",
            "📚 Documentation",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Settings**")
    prev_emotion = st.selectbox("Current Emotion State", list(config.emotions), index=4)
    language = st.selectbox("Language (UI)", ["English", "Hindi", "Spanish", "French"])
    st.caption("Multilingual UI labels; model trained on English (MELD).")
    enable_voice = st.checkbox("Voice Input (experimental)", value=False)
    if enable_voice:
        st.info("Use browser speech-to-text or type messages in Live Prediction.")

    model_loaded = get_predictor_safe() is not None
    st.divider()
    if model_loaded:
        st.success("✅ Model loaded")
    else:
        st.error("❌ Model not trained")
        st.code("python -m training.train --quick", language="bash")


# ─── Home ────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown('<p class="main-header">Context-Aware Emotional Transition Detection</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Analyze emotional flow across conversations using Bidirectional LSTM networks</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    meta = {}
    if config.metadata_path.exists():
        with open(config.metadata_path) as f:
            meta = json.load(f)
    col1.metric("Model", "BiLSTM")
    col2.metric("Dataset", "MELD")
    col3.metric("Emotions", len(config.emotions))
    col4.metric("Val Accuracy", f"{meta.get('final_val_accuracy', 0):.1%}" if meta else "N/A")

    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("""
        ### Why LSTM for Emotion Transitions?

        Unlike single-sentence classifiers, this system models **sequences** of utterances:

        | Concept | Role |
        |---------|------|
        | **Hidden State** | Memory vector h_t encoding emotional context |
        | **Cell State (LSTM)** | Long-term memory gate — retains context over many turns |
        | **Bidirectional LSTM** | Reads context forward & backward within the window |
        | **Sequence Learning** | Learns patterns like *neutral → anger* after accusatory phrasing |
        | **Transition Classes** | Predicts `happy → neutral`, `anxious → fear`, etc. |

        **RNN memory**: At each timestep t, the network updates hidden state using previous
        state h_{t-1} and current input x_t, enabling contextual dependency across dialogue.
        """)
    with c2:
        st.markdown("### Quick Start")
        st.code("pip install -r requirements.txt\npython -m training.train --quick\nstreamlit run streamlit_app.py", language="bash")
        if st.button("🚀 Go to Live Prediction"):
            st.session_state["_nav"] = "live"

    # Architecture diagram
    st.markdown("### System Architecture")
    st.markdown("""
```mermaid
flowchart LR
    A[Conversation Messages] --> B[Text Preprocessing]
    B --> C[Tokenization & Padding]
    C --> D[Embedding Layer]
    D --> E[Bidirectional LSTM]
    E --> F[Dropout + Dense]
    F --> G[Softmax Transition Output]
    G --> H[Timeline & Analytics UI]
```
    """)


# ─── Live Prediction ───────────────────────────────────────────────────────────
elif page == "🔮 Live Prediction":
    st.markdown("## 🔮 Real-Time Transition Prediction")
    predictor = get_predictor_safe()

    if predictor is None:
        st.warning("Train the model first: `python -m training.train --quick`")
    else:
        col_in, col_out = st.columns([1, 1])
        with col_in:
            st.markdown("### Input")
            live_text = st.text_area(
                "Enter message or context",
                placeholder="I can't believe you said that to me!",
                height=120,
            )
            if enable_voice and live_text == "":
                st.caption("🎤 Voice: type after speaking (browser STT) or use Conversation Simulator.")

            if st.button("Predict Transition", type="primary"):
                context = live_text.strip()
                if context:
                    result = predictor.predict_transition(context, prev_emotion, use_keyword_enhancement=True)
                    st.session_state["last_result"] = result
                    st.session_state.prediction_history.append(result.transition)

        with col_out:
            st.markdown("### Prediction")
            if "last_result" in st.session_state:
                r = st.session_state["last_result"]
                st.success(f"**{r.transition}**")
                m1, m2, m3 = st.columns(3)
                m1.metric("From", r.from_emotion)
                m2.metric("To", r.to_emotion)
                m3.metric("Confidence", f"{r.confidence:.1%}")
                st.metric("Entropy", f"{r.entropy:.3f}")

                labels = list(r.probabilities.keys())
                probs = np.array(list(r.probabilities.values()))
                fig = plotly_confidence_bars(labels, probs / probs.sum() if probs.sum() else probs)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Enter text and click Predict.")

        # History memory
        st.markdown("### Conversation Memory")
        mem_col1, mem_col2 = st.columns(2)
        if mem_col1.button("Add to Memory"):
            if live_text.strip():
                predictor.memory.add_message(live_text.strip())
                st.session_state.chat_messages.append(live_text.strip())
        if mem_col2.button("Clear Memory"):
            predictor.reset_memory()
            st.session_state.chat_messages = []
            st.session_state.chat_emotions = ["neutral"]

        if st.session_state.chat_messages:
            for i, msg in enumerate(st.session_state.chat_messages):
                st.markdown(f'<div class="chat-bubble"><b>Turn {i+1}:</b> {msg}</div>', unsafe_allow_html=True)


# ─── Conversation Simulator ────────────────────────────────────────────────────
elif page == "💬 Conversation Simulator":
    st.markdown("## 💬 Conversation Simulator")
    predictor = get_predictor_safe()

    if predictor is None:
        st.warning("Model required. Run training first.")
    else:
        st.markdown("Simulate multi-turn dialogue and watch emotional transitions unfold.")
        default_conv = """Hey, how are you doing today?
Actually, I'm not feeling great.
I just found out some really bad news.
Why would they do something like that?
I can't stop thinking about it."""
        conv_text = st.text_area("Conversation (one message per line)", value=default_conv, height=200)
        init_emo = st.selectbox("Initial emotion", list(config.emotions), index=4)
        demo_boost = st.checkbox(
            "Demo boost (use text emotion cues per turn — clearer timeline for presentation)",
            value=True,
            help="Uses keywords in each previous line to set 'previous emotion' so transitions are not stuck on neutral → neutral.",
        )

        if st.button("Analyze Conversation", type="primary"):
            messages = [m.strip() for m in conv_text.strip().split("\n") if m.strip()]
            with st.spinner("Analyzing emotional flow..."):
                analysis = predictor.analyze_conversation(
                    messages, init_emo, use_text_emotion_cues=demo_boost
                )

            st.session_state["last_analysis"] = analysis

        if "last_analysis" in st.session_state:
            analysis = st.session_state["last_analysis"]
            timeline = analysis["timeline"]

            fig = plotly_emotion_timeline(timeline)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Transition Events")
            trans_df = pd.DataFrame(analysis["transitions"])
            st.dataframe(trans_df, use_container_width=True)

            # Chat-style display
            st.markdown("### Dialogue View")
            for i, msg in enumerate(conv_text.strip().split("\n")):
                if not msg.strip():
                    continue
                emo = analysis["emotions"][min(i, len(analysis["emotions"]) - 1)]
                color = {"joy": "🟡", "anger": "🔴", "sadness": "🔵", "fear": "🟣",
                         "surprise": "🩵", "disgust": "🟢", "neutral": "⚪"}.get(emo, "⚪")
                st.markdown(f"{color} **{emo.upper()}** — {msg.strip()}")

            if st.button("Export Analytics PDF"):
                pdf_path = export_analytics_pdf(
                    analysis["emotions"],
                    analysis["transitions"],
                )
                st.success(f"Saved: {pdf_path}")
                with open(pdf_path, "rb") as f:
                    st.download_button("Download PDF", f, file_name=pdf_path.name)


# ─── Analytics Dashboard ─────────────────────────────────────────────────────
elif page == "📊 Analytics Dashboard":
    st.markdown("## 📊 Emotion Analytics Dashboard")
    predictor = get_predictor_safe()

    proc_path = config.processed_dir / "transitions_all.csv"
    if proc_path.exists():
        df = pd.read_csv(proc_path)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Emotion Frequency (Next State)")
            freq = df["next_emotion"].value_counts().reset_index()
            freq.columns = ["emotion", "count"]
            fig = px.pie(freq, values="count", names="emotion", title="Emotion Distribution", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Top Transition Types")
            top_trans = df["transition"].value_counts().head(15).reset_index()
            top_trans.columns = ["transition", "count"]
            fig2 = px.bar(top_trans, x="count", y="transition", orientation="h", template="plotly_dark",
                          title="Most Frequent Transitions")
            st.plotly_chart(fig2, use_container_width=True)

        matrix = emotion_transition_matrix(df["transition"].tolist())
        st.markdown("### Transition Heatmap")
        st.plotly_chart(plotly_transition_heatmap(matrix), use_container_width=True)

        if st.session_state.prediction_history:
            st.markdown("### Session Predictions")
            st.write(top_k_transitions(st.session_state.prediction_history))
    else:
        st.info("Run `python -m training.train` to generate analytics data.")

    if config.outputs_dir.joinpath("emotion_distribution.png").exists():
        st.image(str(config.outputs_dir / "emotion_distribution.png"), caption="Emotion Distribution")


# ─── Training Results ──────────────────────────────────────────────────────────
elif page == "📈 Training Results":
    st.markdown("## 📈 Training Visualization")
    hist_path = config.history_path
    if hist_path.exists():
        with open(hist_path) as f:
            history = json.load(f)
        epochs = list(range(1, len(history["accuracy"]) + 1))
        hdf = pd.DataFrame({
            "epoch": epochs,
            "train_acc": history["accuracy"],
            "val_acc": history["val_accuracy"],
            "train_loss": history["loss"],
            "val_loss": history["val_loss"],
        })
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(hdf, x="epoch", y=["train_acc", "val_acc"], title="Accuracy", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.line(hdf, x="epoch", y=["train_loss", "val_loss"], title="Loss", template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No training history found.")

    for img_name, caption in [
        ("training_accuracy_loss.png", "Accuracy & Loss"),
        ("confusion_matrix.png", "Confusion Matrix"),
    ]:
        p = config.outputs_dir / img_name
        if p.exists():
            st.image(str(p), caption=caption)

    metrics_path = config.outputs_dir / "evaluation_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            st.json(json.load(f))


# ─── Model Architecture ───────────────────────────────────────────────────────
elif page == "🔬 Model Architecture":
    st.markdown("## 🔬 Model Architecture & Algorithm")
    st.markdown("""
    ### Layer-by-Layer Architecture

    | Layer | Output Shape | Purpose |
    |-------|-------------|---------|
    | Input | (batch, 50) | Padded token sequence |
    | Embedding | (batch, 50, 128) | Dense word representations |
    | BiLSTM-1 | (batch, 50, 256) | Contextual sequence encoding |
    | Dropout | — | Regularization (40%) |
    | BiLSTM-2 | (batch, 128) | Aggregated sequence representation |
    | Dense 128 + ReLU | (batch, 128) | Non-linear feature learning |
    | Dense 64 + ReLU | (batch, 64) | Transition pattern refinement |
    | Softmax | (batch, num_classes) | Transition probability distribution |

    ### LSTM Gating (Conceptual)

    ```
    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)   # Forget gate
    i_t = σ(W_i · [h_{t-1}, x_t] + b_i)   # Input gate
    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)   # Output gate
    c_t = f_t * c_{t-1} + i_t * tanh(W_c · [h_{t-1}, x_t] + b_c)
    h_t = o_t * tanh(c_t)
    ```

    ### Emotion Transition Modeling

    Each training sample: **context window** (previous utterances) → **transition label** (e.g., `neutral -> anger`).
    The model learns that emotional state shifts depend on *conversational history*, not isolated sentences.
    """)

    if config.metadata_path.exists():
        with open(config.metadata_path) as f:
            st.json(json.load(f))

    st.markdown("### RNN vs LSTM Comparison")
    st.markdown("Run `python -m training.train --model rnn` to train vanilla RNN for comparison.")


# ─── Documentation ───────────────────────────────────────────────────────────
elif page == "📚 Documentation":
    st.markdown("## 📚 Project Documentation")
    report_path = config.project_root / "report" / "PROJECT_REPORT.md"
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8")[:8000] + "\n\n*[Report truncated in UI — see report/PROJECT_REPORT.md for full document]*")
    else:
        st.info("See report/PROJECT_REPORT.md")
    viva_path = config.project_root / "report" / "VIVA_QUESTIONS.md"
    if viva_path.exists():
        with st.expander("Viva Questions & Answers"):
            st.markdown(viva_path.read_text(encoding="utf-8"))

"""
FastAPI REST API for emotional transition detection.
Run: uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.inference import EmotionTransitionPredictor

app = FastAPI(
    title="Emotional Transition Detection API",
    description="Context-aware LSTM API for conversational emotion transition analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor: Optional[EmotionTransitionPredictor] = None


def get_predictor() -> EmotionTransitionPredictor:
    global _predictor
    if _predictor is None:
        _predictor = EmotionTransitionPredictor()
        try:
            _predictor.load()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _predictor


class PredictRequest(BaseModel):
    context: str = Field(..., description="Conversation context (utterances joined)")
    prev_emotion: str = Field(default="neutral", description="Previous emotion state")


class MessageRequest(BaseModel):
    message: str
    prev_emotion: str = "neutral"


class ConversationRequest(BaseModel):
    messages: List[str]
    initial_emotion: str = "neutral"


class PredictResponse(BaseModel):
    transition: str
    from_emotion: str
    to_emotion: str
    confidence: float
    probabilities: dict
    entropy: float


@app.get("/")
def root():
    return {
        "project": "Context-Aware Emotional Transition Detection",
        "model": "Bidirectional LSTM",
        "endpoints": ["/predict", "/predict/message", "/predict/conversation", "/health"],
    }


@app.get("/health")
def health():
    try:
        get_predictor()
        return {"status": "healthy", "model_loaded": True}
    except HTTPException:
        return {"status": "degraded", "model_loaded": False}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    pred = get_predictor()
    result = pred.predict_transition(req.context, req.prev_emotion)
    return PredictResponse(
        transition=result.transition,
        from_emotion=result.from_emotion,
        to_emotion=result.to_emotion,
        confidence=result.confidence,
        probabilities=result.probabilities,
        entropy=result.entropy,
    )


@app.post("/predict/message", response_model=PredictResponse)
def predict_message(req: MessageRequest):
    pred = get_predictor()
    result = pred.predict_next_from_conversation(req.message, req.prev_emotion)
    return PredictResponse(
        transition=result.transition,
        from_emotion=result.from_emotion,
        to_emotion=result.to_emotion,
        confidence=result.confidence,
        probabilities=result.probabilities,
        entropy=result.entropy,
    )


@app.post("/predict/conversation")
def predict_conversation(req: ConversationRequest):
    pred = get_predictor()
    pred.reset_memory()
    analysis = pred.analyze_conversation(req.messages, req.initial_emotion)
    return {
        "emotions": analysis["emotions"],
        "timeline": analysis["timeline"].to_dict(orient="records"),
        "transitions": analysis["transitions"],
        "predictions": [
            {
                "transition": p.transition,
                "confidence": p.confidence,
                "to_emotion": p.to_emotion,
            }
            for p in analysis["predictions"]
        ],
    }


@app.post("/reset")
def reset_memory():
    pred = get_predictor()
    pred.reset_memory()
    return {"status": "memory cleared"}

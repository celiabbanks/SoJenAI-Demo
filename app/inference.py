# SoJenAI-Demo/app/inference.py
"""
Inference adapter for SoJen.AI Bias API.

This wraps the model pipeline defined in core.models and exposes
a simple `predict(texts)` function that predictor_smoke.py and
the FastAPI layer can call.

We DO NOT reimplement the model logic here; instead, we delegate to:

    core.models.predict_bias_type(text: str) -> Dict[str, float]
    core.models.assess_bias_severity(text: str, scores: Dict[str,float])
    core.models.predict_sentiment(text: str)

and add a consistent result shape.
"""
from __future__ import annotations

from typing import List, Dict, Any

from core.models import (
    predict_bias_type,
    assess_bias_severity,
    predict_sentiment,
    DEVICE,
    model_debug_summary,
)


import torch

def predict(texts: List[str]) -> List[Dict[str, Any]]:
    if not isinstance(texts, list):
        raise TypeError("predict(texts) expects a list of strings")

    dbg = model_debug_summary()
    head_size = None
    if isinstance(dbg, dict) and "bias" in dbg and isinstance(dbg["bias"], dict):
        head_size = dbg["bias"].get("num_labels")

    results: List[Dict[str, Any]] = []

    for text in texts:
        scores = predict_bias_type(text) or {}

        # Let severity logic (with lexicon override) decide label
        severity, reason, sev_meta = assess_bias_severity(text, scores)
        top_label = sev_meta.get("top_label")
        max_prob = sev_meta.get("max_prob", 0.0)

        sentiment = predict_sentiment(text)

        item: Dict[str, Any] = {
            "text": text,
            "scores": scores,
            "top_label": top_label,
            "sentiment": sentiment,
            "severity": severity,
            "meta": {
                "backend": "core.models.predict_bias_type",
                "device": DEVICE,
                "head_size": head_size,
                "severity_reason": reason,
                "severity_meta": sev_meta,
            },
        }
        results.append(item)

    return results




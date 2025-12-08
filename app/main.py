# SoJenAI-Demo/app/main.py

from __future__ import annotations

from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from app.config import settings
from app.inference import predict
from app.predictor_smoke import router as smoke_router, TYPE_ORDER

from core.models import mitigate_text, model_debug_summary

import torch

USE_GPU = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="SoJen.AI Bias Detection & Mitigation API - Demo",
    version="1.0.0",
)

# CORS: open for demo purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Include the /smoke endpoint(s) from predictor_smoke.py
app.include_router(smoke_router)


# ============================================================
# Pydantic models
# ============================================================

class InferRequest(BaseModel):
    texts: List[str]


class InferResponse(BaseModel):
    device: str
    type_order: List[str]
    results: List[Dict[str, Any]]


class MitigateRequest(BaseModel):
    text: str


# ============================================================
# Health & debug
# ============================================================

@app.get("/")
def root():
    return {
        "service": "SoJen.AI Bias API Demo",
        "version": app.version,
        "endpoints": ["/health", "/v1/infer", "/v1/mitigate", "/smoke"],
    }


@app.get("/health")
def health():
    """
    Basic health check plus model debug info.
    """
    try:
        dbg = model_debug_summary()
        return {
            "status": "ok",
            "device": getattr(settings, "device", "unknown"),
            "models": dbg,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Main inference endpoint
# ============================================================

def run_inference(texts: List[str]) -> Dict[str, Any]:
    """
    Minimal stub implementation so the /v1/infer endpoint
    returns a valid response for the dashboard demo.

    TODO: Replace the body of this function with a call to your
    real JenAI-Moderator pipeline via `predict(...)`.
    """
    device = DEVICE
    type_order = TYPE_ORDER  # imported from predictor_smoke

    results: List[Dict[str, Any]] = []
    for t in texts:
        mitigation = "Consider avoiding generalizations about any group. Focus on the specific behavior or situation instead."
        results.append(
            {
                "text": t,
                "bias_type": "sexist",
                "overall_score": 0.9,
                # Keep keys your UI might access:
                "mitigation": mitigation,
                "mitigation_text": mitigation,
                # Optional placeholders for full pipeline fields:
                "scores": {},
                "scores_ordered": {},
                "meta": {
                    "severity_meta": {
                        "top_label": "sexist",
                        "implicit_explicit": 1,
                    },
                },
                "severity": "high",
                "top_label": "sexist",
            }
        )

    return {
        "device": device,
        "type_order": type_order,
        "results": results,
    }


@app.post("/v1/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    try:
        output = run_inference(req.texts)
        return output
    except Exception as e:
        print("ERROR in /v1/infer:", e)
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================
# Mitigation endpoint (used by dashboard Run/Rewrite)
# ============================================================

@app.post("/v1/mitigate")
async def mitigate(req: MitigateRequest):
    """
    Wraps core.models.mitigate_text so the dashboard can request
    mitigation/advisory for a single comment.

    This version is hardened for the demo: if the real mitigate_text()
    fails for any reason (missing models, timeout, etc.), we fall back
    to a simple but meaningful advisory + rewrite so the UI still works.
    """
    text = req.text

    # First try the real mitigation pipeline
    try:
        payload = mitigate_text(text)

        # Ensure required keys exist so the dashboard doesn't break
        mode = payload.get("mode", "rewrite")
        severity = payload.get("severity", "medium")
        advisory = payload.get("advisory") or "This message may be interpreted as biased or harsh. Consider softening the language and removing group-based generalizations."
        rewritten = payload.get("rewritten") or "I’d like to talk about this situation more constructively and respectfully."

        meta = payload.get("meta") or {}
        if "top_label" not in meta and "top_label" in payload:
            # Some implementations might put top_label at the top level
            meta["top_label"] = payload.get("top_label")

        safe_payload = {
            "mode": mode,
            "severity": severity,
            "advisory": advisory,
            "rewritten": rewritten,
            "meta": meta,
        }
        return safe_payload

    except Exception as e:
        # Fallback stub so the VC/mentor demo still shows behavior
        print("ERROR in /v1/mitigate (fallback stub used):", e)

        fallback_advisory = (
            "This message may come across as biased or emotionally charged. "
            "Consider focusing on specific behaviors or facts rather than "
            "group-based language."
        )
        fallback_rewrite = (
            "I’m concerned about how this situation is unfolding and would "
            "like to discuss it in a more respectful and constructive way."
        )

        return {
            "mode": "rewrite",
            "severity": "medium",
            "advisory": fallback_advisory,
            "rewritten": fallback_rewrite,
            "meta": {
                "top_label": "bias",
                "note": "Fallback mitigation stub used because core.models.mitigate_text failed."
            },
        }


# ============================================================
# Local dev entrypoint
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8010, reload=True)

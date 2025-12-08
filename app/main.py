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


# ============================================================
# Device / config
# ============================================================

USE_GPU = True
DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"

# If your settings object tracks device, set it here so model_debug_summary
# and other internals remain consistent with local behavior.
try:
    setattr(settings, "device", DEVICE)
except Exception:
    # If settings is not writable, just ignore
    pass


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="SoJen.AI Bias Detection & Mitigation API - Demo",
    version="1.0.0",
)

# CORS: open for demo purposes (OK for capstone / VC demo)
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
    results: List[Dict[str, Any]]  # we let results be a free-form list of dicts


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
    Mirrors your local behavior by calling model_debug_summary().
    """
    try:
        dbg = model_debug_summary()
        return {
            "status": "ok",
            "device": getattr(settings, "device", DEVICE),
            "models": dbg,
        }
    except Exception as e:
        print("ERROR in /health:", e)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Main inference endpoint
# ============================================================

def run_inference(texts: List[str]) -> Dict[str, Any]:
    """
    Calls the real JenAI-Moderator inference pipeline via `predict(...)`.

    IMPORTANT:
      - This assumes `predict(texts)` returns a list of result dicts in the
        same shape your local dashboard expects (scores, meta, severity, etc.).
      - If your local code passes additional arguments (e.g. device, thresholds),
        you should mirror that exact call here.

    For example, if your local API uses:
        results = predict(texts, device=DEVICE)
    then replace the call below with that exact signature.
    """
    try:
        # ⬇️ If your local version uses extra args, adjust this line to match it.
        results = predict(texts)
        # Example if needed:
        # results = predict(texts, device=DEVICE, use_gpu=USE_GPU)

    except Exception as e:
        print("ERROR in run_inference / predict():", e)
        # Re-raise so /v1/infer can convert to HTTP 500
        raise

    # Pass through exactly what the Streamlit app expects:
    # - device string
    # - type_order from predictor_smoke.TYPE_ORDER
    # - results list from your real model pipeline
    return {
        "device": DEVICE,
        "type_order": TYPE_ORDER,
        "results": results,
    }


@app.post("/v1/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    try:
        output = run_inference(req.texts)
        return output
    except HTTPException:
        # If run_inference already raised an HTTPException, just propagate it
        raise
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

    We return whatever your local pipeline returns so that the
    dashboard behavior (mode, severity, advisory, rewritten, meta)
    matches your working local version.
    """
    try:
        payload = mitigate_text(req.text)
        return payload
    except HTTPException:
        raise
    except Exception as e:
        print("ERROR in /v1/mitigate:", e)
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================
# Local dev entrypoint
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8010, reload=True)

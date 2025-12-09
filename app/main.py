# SoJenAI-Demo/app/main.py
# SoJenAI-Demo/app/main.py

from __future__ import annotations

from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from app.config import settings
from app.predictor_smoke import router as smoke_router, TYPE_ORDER, predict as smoke_predict

from core.models import mitigate_text, model_debug_summary

import os
import torch

import traceback


# ============================================================
# Device / config
# ============================================================

USE_GPU = True
DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"

# Try to keep settings.device in sync for /health and internals
try:
    setattr(settings, "device", DEVICE)
except Exception:
    pass


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="SoJen.AI Bias Detection & Mitigation API - Demo",
    version="1.0.0",
)

# CORS: open for demo / capstone purposes
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
# Secure website access to FastAPI app via Railway
# ============================================================


def require_api_key(x_api_key: str = Header(None)):
    """
    Validates the caller has the correct API key.
    """
    expected = os.getenv("SOJEN_API_KEY")
    if expected is None:
        # If API key not configured, allow all (for debugging).
        return

    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ============================================================
# Pydantic request models
# (we do NOT constrain the response model here; we return dicts)
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
# Health & root
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
# /v1/infer — main inference endpoint
# ============================================================

@app.post("/v1/infer", response_model=InferResponse, dependencies=[Depends(require_api_key)])
async def infer(req: InferRequest):

# @app.post("/v1/infer")
# async def infer(req: InferRequest) -> Dict[str, Any]:
    """
    Calls the canonical predictor from predictor_smoke.predict(),
    which in turn uses app.inference (or inference) and normalizes
    scores into scores_ordered following TYPE_ORDER.

    Returns the same structure that your local Streamlit app expects:
      {
        "device": "cuda" | "cpu",
        "type_order": [...],
        "results": [ { ... per-text result dict ... }, ... ]
      }
    """
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    try:
        # Use the canonical wrapper so behavior matches /smoke and local
        results = smoke_predict(req.texts)

        payload: Dict[str, Any] = {
            "device": DEVICE,
            "type_order": TYPE_ORDER,
            "results": results,
        }
        return payload

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print("ERROR in /v1/infer:", e)
        print(tb)
        # Return the error + traceback so we can see it from your laptop
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "traceback": tb,
            },
        )



# ============================================================
# /v1/mitigate — advisory / rewrite endpoint
# ============================================================

@app.post("/v1/mitigate", dependencies=[Depends(require_api_key)])
async def mitigate(req: MitigateRequest):

# @app.post("/v1/mitigate")
# async def mitigate(req: MitigateRequest) -> Dict[str, Any]:
    """
    Wraps core.models.mitigate_text so the dashboard can request
    mitigation/advisory for a single comment.

    Returns whatever your local pipeline returns (mode, severity,
    advisory, rewritten, meta, etc.) so the dashboard behavior
    matches your working local version.
    """
    try:
        payload = mitigate_text(req.text)
        # We trust this to already be a serializable dict in the shape your
        # local Streamlit app is coded against.
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

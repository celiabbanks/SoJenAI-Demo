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
    device = DEVICE
    type_order = TYPE_ORDER  # imported from predictor_smoke

    results: List[Dict[str, Any]] = []
    for t in texts:
        mitigation = "Consider avoiding gender-based generalizations."
        results.append(
            {
                "text": t,
                "bias_type": "sexist",
                "overall_score": 0.9,
                # 👇 provide BOTH keys so whatever the UI expects will work
                "mitigation": mitigation,
                "mitigation_text": mitigation,
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
    """
    try:
        payload = mitigate_text(req.text)
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Local dev entrypoint
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8010, reload=True)

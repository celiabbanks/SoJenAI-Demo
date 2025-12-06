# app/predictor_smoke.py
# predictor_smoke.py
# Second of 3 files for
# scaled-down investor preview

# predictor_smoke.py

import os
import sys
import importlib
import inspect
import traceback
import json
from pathlib import Path
from typing import List, Dict, Any

from fastapi import APIRouter

# ------------------------------------------
# FastAPI router for /smoke endpoint
# ------------------------------------------
router = APIRouter()

# ------------------------------------------
# Label order used across UI + API
# ------------------------------------------
TYPE_ORDER = [
    "political",
    "racial",
    "sexist",
    "classist",
    "ageism",
    "antisemitic",
    "bullying",
    "brand",
]

# ------------------------------------------
# Add this folder to sys.path (for CLI use)
# ------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print(f"[DEBUG] predictor_smoke.py CWD={os.getcwd()}")
print("[DEBUG] predictor_smoke.py sys.path[0:3] =", sys.path[0:3])

# This is the order of modules we will try to import for inference
# Prefer the fully qualified app.inference when running under uvicorn.
_DEF_MODULES = [
    "app.inference",   # app/inference.py as a module
    "inference",       # fallback for direct CLI usage if run from app/
]

_predict_fn = None
_device = "cpu"


# ------------------------------------------
# Utility: try importing inference modules
# ------------------------------------------
def _try_import(name: str):
    try:
        print(f"[DEBUG] trying import: {name}")
        return importlib.import_module(name), None
    except Exception as e:
        return None, f"{name}: {e}\n{traceback.format_exc()}"


# ------------------------------------------
# Load predictor (called once at startup)
# ------------------------------------------
def load_predictor(preferred_modules: List[str] = None, device: str = "auto"):
    """
    Loads an inference module (either a `predict()` function or
    a Predictor class exposing .predict()).
    Also resolves device selection.
    """
    global _predict_fn, _device

    # Device resolution
    if device == "auto":
        try:
            import torch

            _device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            _device = "cpu"
    else:
        _device = device

    modules = preferred_modules or _DEF_MODULES
    errors = []
    impl = None

    # Try import modules in order
    for m in modules:
        mod, err = _try_import(m)

        if mod is None:
            errors.append(err)
            continue

        # Case 1: module-level predict()
        if hasattr(mod, "predict") and inspect.isfunction(mod.predict):
            impl = mod.predict
            print(f"[INFO] Using predict() from module '{m}'")
            break

        # Case 2: A Predictor class with .predict()
        if hasattr(mod, "Predictor"):
            cls = getattr(mod, "Predictor")
            sig = inspect.signature(cls)

            if "device" in sig.parameters:
                inst = cls(device=_device)
            else:
                inst = cls()

            if hasattr(inst, "predict") and callable(inst.predict):
                impl = inst.predict
                print(f"[INFO] Using Predictor.predict() from module '{m}'")
                break

        # If import succeeded but lacked predict()
        errors.append(
            f"Module '{m}' imported but no usable predict() or Predictor.predict()."
        )

    # If no working implementation was found
    if impl is None:
        joined = "\n\n".join(e for e in errors if e)
        raise RuntimeError(
            "Could not load inference implementation.\n"
            f"Tried modules: {modules}\n\nErrors:\n{joined}"
        )

    _predict_fn = impl
    print(f"[INFO] Predictor loaded on device={_device}")


# ------------------------------------------
# Public: predict wrapper (normalizes output)
# ------------------------------------------
def predict(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Calls the underlying predictor.
    Ensures results include `scores_ordered` following TYPE_ORDER.
    """
    if _predict_fn is None:
        load_predictor()

    results = _predict_fn(texts)

    normalized = []
    for r in results:
        scores = r.get("scores", {}) or {}
        ordered = {label: scores.get(label, 0.0) for label in TYPE_ORDER}

        r["scores_ordered"] = ordered
        normalized.append(r)

    return normalized


# ------------------------------------------
# FastAPI /smoke endpoint
# ------------------------------------------
@router.get("/smoke")
def smoke_endpoint():
    """
    Basic smoke test endpoint:
    Runs two hard-coded example comments through the pipeline.
    """
    examples = [
        "I won't get in an elevator alone with a black person.",
        "Our brand is being dragged on Twitter.",
    ]

    try:
        out = predict(examples)
        return {
            "ok": True,
            "device": _device,
            "type_order": TYPE_ORDER,
            "results": out,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
        }


# ------------------------------------------
# CLI Smoke test (if executed directly)
# ------------------------------------------
def run_smoke():
    print("[SMOKE] Running demo predictions...\n")

    examples = [
        "I won't get in an elevator alone with a black person.",
        "Our brand is being dragged on Twitter.",
    ]

    out = predict(examples)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    run_smoke()

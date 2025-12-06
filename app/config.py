# app/config.py

from dataclasses import dataclass
import os
import torch


@dataclass
class Settings:
    """
    Minimal demo configuration. This file only controls API behavior,
    not model loading — core/models.py handles that independently.
    """

    # Device selection for display purposes (health endpoint)
    device: str = os.getenv("SOJEN_DEVICE", "auto")

    # Inference adapter module name (in case you swap backends later)
    inference_module: str = os.getenv("SOJEN_INFERENCE_MODULE", "inference")

    # Batch / text constraints for API safety
    max_batch_size: int = int(os.getenv("SOJEN_MAX_BATCH", 32))
    max_text_len: int = int(os.getenv("SOJEN_MAX_TEXT_LEN", 4096))


settings = Settings()

# Resolve automatic device selection to real hardware
if settings.device == "auto":
    settings.device = "cuda" if torch.cuda.is_available() else "cpu"

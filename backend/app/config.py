import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Central place for every tunable setting. Everything can be overridden
    via environment variables / a .env file, so no secrets live in code."""

    # --- Natural language instruction parsing ---
    # "openai" | "gemini" | "none". If "none" (or the key is missing), a
    # regex-based fallback parser is used instead so the app still works
    # without any API key configured.
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    # --- Pipeline mode ---
    # True  -> lightweight, dependency-free stub pipeline (no torch/transformers
    #          needed). Great for local development and for recording a
    #          reliable demo video regardless of GPU/internet availability.
    # False -> "real" AI pipeline: OWL-ViT (open-vocabulary detection) +
    #          Segment Anything (SAM) for segmentation. Requires the packages
    #          in requirements-ai.txt and a downloaded SAM checkpoint.
    DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"

    # --- Storage ---
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "storage/uploads")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "storage/outputs")
    FRAMES_DIR = os.getenv("FRAMES_DIR", "storage/frames")

    # --- Real-AI pipeline model settings (only used when DEMO_MODE=false) ---
    OWLVIT_MODEL = os.getenv("OWLVIT_MODEL", "google/owlvit-base-patch32")
    SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT", "models/sam_vit_b_01ec64.pth")
    SAM_MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_b")

    # --- CORS ---
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")


settings = Settings()

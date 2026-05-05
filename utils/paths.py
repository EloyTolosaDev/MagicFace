from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / ".models"
HUGGINGFACE_CACHE_DIR = MODEL_ROOT / "huggingface"
INSIGHTFACE_ROOT = MODEL_ROOT / "insightface"
MAGICFACE_ASSET_DIR = MODEL_ROOT / "magicface"

import os
from pathlib import Path

from huggingface_hub import snapshot_download

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

try:
    from .paths import HUGGINGFACE_CACHE_DIR, INSIGHTFACE_ROOT, MAGICFACE_ASSET_DIR, MODEL_ROOT
except ImportError:
    from paths import HUGGINGFACE_CACHE_DIR, INSIGHTFACE_ROOT, MAGICFACE_ASSET_DIR, MODEL_ROOT


MAGICFACE_REPO_ID = os.environ.get("MAGICFACE_REPO_ID", "mengtingwei/magicface")
INSIGHTFACE_MODEL_NAME = "antelopev2"

MAGICFACE_REQUIRED_PATHS = [
    "ID_enc",
    "denoising_unet",
    "utils/79999_iter.pth",
    "utils/checkpoints/third_party/d3dfr_res50_nofc.pth",
    "utils/checkpoints/third_party/BFM_model_front.mat",
    "utils/third_party",
]

def ensure_model_dirs():
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    HUGGINGFACE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MAGICFACE_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    INSIGHTFACE_ROOT.mkdir(parents=True, exist_ok=True)


def _has_magicface_assets():
    return all((MAGICFACE_ASSET_DIR / path).exists() for path in MAGICFACE_REQUIRED_PATHS)


def ensure_magicface_assets() -> Path:
    ensure_model_dirs()
    if _has_magicface_assets():
        return MAGICFACE_ASSET_DIR

    try:
        snapshot_download(
            repo_id=MAGICFACE_REPO_ID,
            cache_dir=HUGGINGFACE_CACHE_DIR,
            local_dir=MAGICFACE_ASSET_DIR,
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download MagicFace model assets from '{MAGICFACE_REPO_ID}' "
            f"into '{MAGICFACE_ASSET_DIR}'. Check network access or pre-populate that directory."
        ) from exc

    if not _has_magicface_assets():
        missing = [path for path in MAGICFACE_REQUIRED_PATHS if not (MAGICFACE_ASSET_DIR / path).exists()]
        raise FileNotFoundError(
            f"MagicFace assets in '{MAGICFACE_ASSET_DIR}' are incomplete. Missing: {', '.join(missing)}"
        )

    return MAGICFACE_ASSET_DIR


def ensure_insightface_model() -> Path:
    ensure_model_dirs()
    model_dir = INSIGHTFACE_ROOT / "models" / INSIGHTFACE_MODEL_NAME
    if model_dir.exists() and any(model_dir.glob("*.onnx")):
        return model_dir

    try:
        from insightface.utils.storage import download

        model_dir = Path(download("models", INSIGHTFACE_MODEL_NAME, force=True, root=str(INSIGHTFACE_ROOT)))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download InsightFace '{INSIGHTFACE_MODEL_NAME}' model into '{INSIGHTFACE_ROOT}'. "
            "Check network access or pre-populate "
            f"'{INSIGHTFACE_ROOT / 'models' / INSIGHTFACE_MODEL_NAME}'."
        ) from exc

    if not any(model_dir.glob("*.onnx")):
        raise FileNotFoundError(
            f"InsightFace model directory '{model_dir}' is incomplete. Expected at least one .onnx file."
        )

    return model_dir


def ensure_all_models():
    ensure_magicface_assets()
    ensure_insightface_model()


if __name__ == "__main__":
    ensure_all_models()
    print(f"Models are available under {MODEL_ROOT}")

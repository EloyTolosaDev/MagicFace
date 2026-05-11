import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "magicface_matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import onnxruntime as ort
from insightface.app import FaceAnalysis

try:
    from .model_assets import ensure_insightface_model
    from .paths import INSIGHTFACE_ROOT
except ImportError:
    from model_assets import ensure_insightface_model
    from paths import INSIGHTFACE_ROOT


app = None


def requires_onnx_cuda():
    return os.environ.get("MAGICFACE_REQUIRE_ONNX_CUDA", "").lower() in {"1", "true", "yes"}


def get_onnx_providers():
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
    if requires_onnx_cuda():
        raise RuntimeError(
            "CUDAExecutionProvider is required but is not available in ONNX Runtime. "
            f"Available providers: {available_providers}. Install a matching onnxruntime-gpu build "
            "and verify NVIDIA drivers/CUDA libraries are visible in this environment."
        )
    return ["CPUExecutionProvider"], -1


def validate_cuda_provider(face_analysis_app):
    if not requires_onnx_cuda():
        return

    cpu_sessions = []
    for model_name, model in getattr(face_analysis_app, "models", {}).items():
        session = getattr(model, "session", None)
        if session is None or not hasattr(session, "get_providers"):
            continue
        session_providers = session.get_providers()
        if "CUDAExecutionProvider" not in session_providers:
            cpu_sessions.append(f"{model_name}: {session_providers}")

    if cpu_sessions:
        raise RuntimeError(
            "CUDAExecutionProvider was requested, but these InsightFace ONNX sessions are not using it: "
            + "; ".join(cpu_sessions)
        )


def get_face_analysis_app():
    global app

    if app is not None:
        return app

    providers, ctx_id = get_onnx_providers()
    ensure_insightface_model()

    try:
        app = FaceAnalysis(name="antelopev2", root=str(INSIGHTFACE_ROOT), providers=providers)
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize InsightFace models from '{INSIGHTFACE_ROOT}'. "
            "Expected files under 'models/antelopev2' inside that directory."
        ) from exc

    validate_cuda_provider(app)
    return app

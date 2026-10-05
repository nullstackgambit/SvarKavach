"""Verify the local environment can run SvarKavach.

    python -m scripts.check_environment
"""

from __future__ import annotations

import importlib
import platform
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

REQUIRED = ["fastapi", "uvicorn", "pydantic", "pydantic_settings",
            "numpy", "scipy", "soundfile", "onnxruntime"]
OPTIONAL = ["torch", "transformers", "safetensors", "pytest", "httpx"]

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "


def line(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label:<34} {detail}")


def main() -> int:
    failures = 0
    print(f"SvarKavach environment check\n{'-' * 68}")
    line(OK, "Python", f"{platform.python_version()} ({sys.executable})")
    if sys.version_info < (3, 11):
        line(BAD, "Python version", "3.11+ required")
        failures += 1

    print("\nRequired packages")
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            line(OK, name, getattr(module, "__version__", ""))
        except ImportError:
            line(BAD, name, "missing — pip install -r requirements.txt")
            failures += 1

    print("\nOptional packages")
    for name in OPTIONAL:
        try:
            module = importlib.import_module(name)
            line(OK, name, getattr(module, "__version__", ""))
        except ImportError:
            line(WARN, name, "not installed")

    print("\nSystem tools")
    ffmpeg = shutil.which("ffmpeg")
    line(OK if ffmpeg else WARN, "ffmpeg",
         ffmpeg or "not found (only needed for offline evaluation of exotic formats)")

    print("\nCompute")
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        cuda = "CUDAExecutionProvider" in providers
        line(OK if cuda else WARN, "ONNX Runtime providers", ", ".join(providers))
        if not cuda:
            line(WARN, "CUDA (ONNX)",
                 "CPU only — install onnxruntime-gpu on a CUDA machine")
    except ImportError:
        line(BAD, "onnxruntime", "missing")
        failures += 1
    try:
        import torch
        line(OK if torch.cuda.is_available() else WARN, "torch CUDA",
             torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable")
    except Exception:
        pass

    print("\nModels")
    vad = REPO_ROOT / "models" / "silero_vad" / "silero_vad.onnx"
    onnx = REPO_ROOT / "models" / "spectra_aasist" / "spectra-aasist.onnx"
    line(OK if vad.is_file() else WARN, "Silero VAD",
         f"{vad.stat().st_size/1e6:.1f} MB" if vad.is_file()
         else "missing — python -m scripts.download_models --vad-only")
    line(OK if onnx.is_file() else WARN, "XLS-R+AASIST (ONNX)",
         f"{onnx.stat().st_size/1e6:.1f} MB" if onnx.is_file()
         else "missing — python -m scripts.download_models  (mock mode still works)")

    print("\nConfiguration")
    try:
        from backend.app.config import get_settings
        s = get_settings()
        line(OK, "settings validated",
             f"detector={s.detector_provider} device={s.model_device} "
             f"window={s.window_seconds}s hop={s.hop_seconds}s")
    except Exception as exc:
        line(BAD, "settings", str(exc))
        failures += 1

    print("-" * 68)
    print("Environment OK." if not failures else f"{failures} blocking problem(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

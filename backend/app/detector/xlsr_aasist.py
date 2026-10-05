"""XLS-R-300M + AASIST synthetic-speech detector.

Model: `lab260/Spectra-AASIST` — a wav2vec2-XLS-R-300M SSL front-end feeding an
AASIST spectro-temporal graph-attention back-end, as specified in plan §12.2.
Licence: Apache-2.0 (per the model repository metadata; the README text says
MIT — both are permissive, and the metadata value is treated as authoritative).

Two providers share one preprocessing path:

* `OnnxXlsrAasistDetector`  — ONNX Runtime. Default: lighter and faster on CPU,
  and picks up CUDA automatically when `onnxruntime-gpu` is installed.
* `TorchXlsrAasistDetector` — PyTorch + transformers, using the upstream
  `model.py`. Heavier, but the reference implementation.

Contract, verified against the upstream source rather than assumed:
  input   `wav`    float32 [batch, 64600]  (mono, 16 kHz)
  output  `logits` float32 [batch, 2]      index 0 = spoof, index 1 = bona fide
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..config import REPO_ROOT
from ..models.risk import DetectorResult
from ..utils.logging import get_logger
from .base import DetectorError, DetectorProvider, fit_to_length

log = get_logger(__name__)

MODEL_DIR = REPO_ROOT / "models" / "spectra_aasist"
ONNX_FILENAME = "spectra-aasist.onnx"
HF_REPO_ID = "lab260/Spectra-AASIST"

#: Native input length of the model, in samples at 16 kHz (~4.0375 s).
NATIVE_SAMPLES = 64600
#: Pre-emphasis coefficient. Matches torchaudio.functional.preemphasis' default,
#: which the upstream model card specifies as the preprocessing step.
PREEMPHASIS_COEFF = 0.97

SPOOF_INDEX = 0
BONAFIDE_INDEX = 1


def preemphasis(audio: np.ndarray, coeff: float = PREEMPHASIS_COEFF) -> np.ndarray:
    """y[0] = x[0]; y[n] = x[n] - coeff * x[n-1]  (torchaudio's formulation)."""
    if audio.size == 0:
        return audio
    out = np.empty_like(audio, dtype=np.float32)
    out[0] = audio[0]
    out[1:] = audio[1:] - coeff * audio[:-1]
    return out


def prepare_input(audio: np.ndarray) -> np.ndarray:
    """Full preprocessing: pre-emphasis then fit to the native length."""
    audio = np.asarray(audio, dtype=np.float32)
    return fit_to_length(preemphasis(audio), NATIVE_SAMPLES)


def logits_to_scores(logits: np.ndarray, temperature: float = 1.0) -> tuple[float, float]:
    """Temperature-scaled softmax -> (p_spoof, p_bonafide).

    Temperature scaling is the calibration layer of plan §28. At T=1 this is the
    plain softmax of the raw logits, which is NOT a validated probability.
    """
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    if z.size != 2:
        raise DetectorError(f"expected 2 logits, got shape {np.shape(logits)}")
    t = max(float(temperature), 1e-6)
    z = z / t
    z = z - z.max()
    e = np.exp(z)
    p = e / e.sum()
    return float(p[SPOOF_INDEX]), float(p[BONAFIDE_INDEX])


def _build_result(
    logits: np.ndarray, model_name: str, device: str, elapsed_ms: float, temperature: float
) -> DetectorResult:
    p_spoof, p_bonafide = logits_to_scores(logits, temperature)
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    return DetectorResult(
        # Raw, uncalibrated model output (the bona fide-vs-spoof logit margin
        # mapped through softmax). Named per plan §28.
        detector_score=p_spoof,
        synthetic_probability=p_spoof,
        human_probability=p_bonafide,
        confidence=float(np.clip(abs(p_spoof - 0.5) * 2.0, 0.0, 1.0)),
        model=model_name,
        inference_ms=elapsed_ms,
        is_mock=False,
        raw_logits=(float(z[0]), float(z[1])),
    )


# --------------------------------------------------------------------------
class OnnxXlsrAasistDetector(DetectorProvider):
    """ONNX Runtime provider. Default real detector."""

    name = "xlsr_aasist_onnx"
    is_mock = False
    expected_samples = NATIVE_SAMPLES

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "auto",
        threads: int = 0,
        temperature: float = 1.0,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise DetectorError("onnxruntime is not installed") from exc

        path = Path(model_path) if model_path else (MODEL_DIR / ONNX_FILENAME)
        if not path.is_file():
            raise DetectorError(
                f"XLS-R+AASIST ONNX model not found at {path}. "
                f"Run: python -m scripts.download_models"
            )

        available = ort.get_available_providers()
        if device in ("auto", "cuda") and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._device = "cuda"
        else:
            if device == "cuda":
                raise DetectorError(
                    "MODEL_DEVICE=cuda but CUDAExecutionProvider is unavailable. "
                    "Install onnxruntime-gpu, or set MODEL_DEVICE=auto."
                )
            providers = ["CPUExecutionProvider"]
            self._device = "cpu"

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads > 0:
            opts.intra_op_num_threads = threads
        opts.log_severity_level = 3

        t0 = time.perf_counter()
        self._session = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
        self._temperature = temperature
        self._input_name = self._session.get_inputs()[0].name

        log.info(
            "XLS-R+AASIST (ONNX) loaded",
            extra={
                "model": self.name,
                "device": self._device,
                "load_seconds": round(time.perf_counter() - t0, 2),
                "providers": self._session.get_providers(),
                "path": str(path),
            },
        )

    @property
    def device(self) -> str:
        return self._device

    def predict(self, audio: np.ndarray, sample_rate: int) -> DetectorResult:
        prepared = prepare_input(audio).reshape(1, NATIVE_SAMPLES)
        t0 = time.perf_counter()
        try:
            logits = self._session.run(None, {self._input_name: prepared})[0]
        except Exception as exc:
            raise DetectorError(f"ONNX inference failed: {exc}") from exc
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return _build_result(logits[0], self.name, self._device, elapsed_ms, self._temperature)

    def warmup(self) -> None:
        self.predict(np.zeros(NATIVE_SAMPLES, dtype=np.float32), 16000)


# --------------------------------------------------------------------------
class TorchXlsrAasistDetector(DetectorProvider):
    """PyTorch provider using the upstream `model.py`. Optional extra."""

    name = "xlsr_aasist_torch"
    is_mock = False
    expected_samples = NATIVE_SAMPLES

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "auto",
        threads: int = 0,
        temperature: float = 1.0,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise DetectorError(
                "PyTorch is not installed. Install the optional extras: "
                "pip install -r requirements-torch.txt"
            ) from exc

        if threads > 0:
            torch.set_num_threads(threads)

        if device == "cuda" and not torch.cuda.is_available():
            raise DetectorError("MODEL_DEVICE=cuda but torch.cuda.is_available() is False")
        self._device = "cuda" if (device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"

        source = Path(model_path) if model_path else MODEL_DIR
        model_py = source / "model.py"
        if not model_py.is_file():
            raise DetectorError(
                f"upstream model.py not found at {model_py}. "
                f"Run: python -m scripts.download_models --torch"
            )

        # Load the upstream module by path so we do not vendor a copy of it.
        import importlib.util

        spec = importlib.util.spec_from_file_location("spectra_aasist_upstream", model_py)
        if spec is None or spec.loader is None:
            raise DetectorError(f"could not import {model_py}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise DetectorError(f"failed to import upstream model.py: {exc}") from exc

        t0 = time.perf_counter()
        try:
            self._model = module.SpectraAASIST.from_pretrained(str(source))
        except Exception as exc:
            raise DetectorError(f"failed to load Spectra-AASIST weights: {exc}") from exc
        self._model.eval().to(self._device)
        self._torch = torch
        self._temperature = temperature

        log.info(
            "XLS-R+AASIST (PyTorch) loaded",
            extra={
                "model": self.name,
                "device": self._device,
                "load_seconds": round(time.perf_counter() - t0, 2),
                "path": str(source),
            },
        )

    @property
    def device(self) -> str:
        return self._device

    def predict(self, audio: np.ndarray, sample_rate: int) -> DetectorResult:
        torch = self._torch
        prepared = prepare_input(audio).reshape(1, NATIVE_SAMPLES)
        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                tensor = torch.from_numpy(prepared).to(self._device)
                logits = self._model(tensor).detach().float().cpu().numpy()
        except Exception as exc:
            raise DetectorError(f"torch inference failed: {exc}") from exc
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return _build_result(logits[0], self.name, self._device, elapsed_ms, self._temperature)

    def warmup(self) -> None:
        self.predict(np.zeros(NATIVE_SAMPLES, dtype=np.float32), 16000)

"""Voice activity detection.

Silero VAD (MIT) is used when its ONNX model is present. If it is missing or
fails to load we fall back to an RMS energy gate, and the fallback is reported
explicitly through `name` and `is_fallback` so the UI and logs can label it
(plan §10). VAD is never silently degraded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..config import REPO_ROOT
from ..utils.logging import get_logger

log = get_logger(__name__)

#: Silero requires exactly this many new samples per call at 16 kHz.
SILERO_FRAME_SAMPLES = 512
#: Silero v5+ prepends this many samples of previous-frame context to the
#: model input, so the tensor actually fed to ONNX is 512 + 64 = 576 wide.
SILERO_CONTEXT_SAMPLES = 64
DEFAULT_VAD_PATH = REPO_ROOT / "models" / "silero_vad" / "silero_vad.onnx"


@runtime_checkable
class VoiceActivityDetector(Protocol):
    name: str
    is_fallback: bool
    frame_samples: int

    def speech_probability(self, frame: np.ndarray) -> float: ...
    def reset(self) -> None: ...


class EnergyVad:
    """RMS energy gate. Crude but dependency-free — the documented fallback."""

    name = "energy_rms_fallback"
    is_fallback = True
    frame_samples = SILERO_FRAME_SAMPLES

    def __init__(self, floor_dbfs: float = -45.0, ceiling_dbfs: float = -20.0) -> None:
        self._floor = floor_dbfs
        self._ceiling = ceiling_dbfs

    def speech_probability(self, frame: np.ndarray) -> float:
        if frame.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        if rms <= 0.0:
            return 0.0
        dbfs = 20.0 * np.log10(rms)
        # Linear ramp between floor and ceiling -> a pseudo-probability.
        return float(np.clip((dbfs - self._floor) / (self._ceiling - self._floor), 0.0, 1.0))

    def reset(self) -> None:  # stateless
        return


class SileroVad:
    """Streaming Silero VAD over ONNX Runtime. Carries LSTM state across frames."""

    name = "silero_vad"
    is_fallback = False
    frame_samples = SILERO_FRAME_SAMPLES

    def __init__(self, model_path: Path, sample_rate: int = 16000) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # tiny model; threading costs more than it saves
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        # ONNX wants a 0-d ndarray here, not a numpy scalar.
        self._sample_rate = np.array(sample_rate, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, SILERO_CONTEXT_SAMPLES), dtype=np.float32)

    def speech_probability(self, frame: np.ndarray) -> float:
        if frame.size != self.frame_samples:
            raise ValueError(
                f"SileroVad expects exactly {self.frame_samples} samples, got {frame.size}"
            )
        frame2d = frame.reshape(1, -1).astype(np.float32, copy=False)
        model_input = np.concatenate([self._context, frame2d], axis=1)
        out, self._state = self._session.run(
            None,
            {"input": model_input, "state": self._state, "sr": self._sample_rate},
        )
        self._context = model_input[:, -SILERO_CONTEXT_SAMPLES:]
        return float(out[0][0])


def build_vad(
    enabled: bool, model_path: str | None, sample_rate: int = 16000
) -> VoiceActivityDetector | None:
    """Construct the best available VAD, or None when VAD is disabled.

    Every caller needs its *own* instance — see `VadFactory`. Sharing one across
    concurrent streams silently corrupts all of them.
    """
    if not enabled:
        log.warning("VAD disabled by configuration; every window will be scored")
        return None

    path = Path(model_path) if model_path else DEFAULT_VAD_PATH
    if path.is_file():
        try:
            vad = SileroVad(path, sample_rate)
            log.info("VAD ready", extra={"vad": vad.name, "path": str(path)})
            return vad
        except Exception:
            log.exception("Silero VAD failed to load; using energy fallback")
    else:
        log.warning(
            "Silero VAD model not found; using energy fallback",
            extra={"expected_path": str(path)},
        )
    return EnergyVad()


class VadFactory:
    """Resolves the VAD implementation once, then builds one per session.

    Per-session instances are a correctness requirement, not an optimisation.
    `SileroVad` carries LSTM state and a context window across calls, so one
    shared instance interleaves concurrent sessions' frames into a single
    recurrent state: measured on real speech, sharing with just one other
    session cut detected speech from 1.02 s to 0.42 s over the same clip, which
    drops windows under MIN_SPEECH_SECONDS and forces spurious abstention.
    A shared instance is also never `reset()`, so each session would inherit the
    previous one's state.

    Resolution (Silero vs the energy fallback) happens once at startup so the
    decision — and its log line — is not repeated per connection.
    """

    def __init__(
        self, enabled: bool, model_path: str | None, sample_rate: int = 16000
    ) -> None:
        self._sample_rate = sample_rate
        probe = build_vad(enabled, model_path, sample_rate)
        self._path: Path | None = None
        if probe is None:
            self._kind = "disabled"
        elif isinstance(probe, SileroVad):
            self._kind = "silero"
            self._path = Path(model_path) if model_path else DEFAULT_VAD_PATH
        else:
            self._kind = "energy"
        self.name: str = probe.name if probe else "disabled"
        self.is_fallback: bool = bool(probe.is_fallback) if probe else False

    def __call__(self) -> VoiceActivityDetector | None:
        """A fresh, independent VAD for one session."""
        if self._kind == "silero":
            try:
                return SileroVad(self._path, self._sample_rate)
            except Exception:
                # The model loaded once at startup; if it fails now, degrade to
                # the documented fallback rather than failing the session.
                log.exception("Silero VAD failed to load for a session; using fallback")
                return EnergyVad()
        if self._kind == "energy":
            return EnergyVad()
        return None

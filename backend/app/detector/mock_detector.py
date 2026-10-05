"""Deterministic stand-in detector for development and demos.

This is NOT a synthetic-speech detector. It derives a stable pseudo-score from
signal statistics so the pipeline can be exercised end to end before the real
model is available. It always reports `is_mock=True`, and the UI is required to
display that prominently (plan §12.3, §40).
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

from ..models.risk import DetectorResult
from .base import DetectorProvider, fit_to_length


def _spectral_flatness(audio: np.ndarray) -> float:
    """Geometric-mean / arithmetic-mean of the power spectrum, in [0, 1]."""
    if audio.size < 256:
        return 0.5
    spec = np.abs(np.fft.rfft(audio * np.hanning(audio.size))) ** 2
    spec = spec[1:]  # drop DC
    spec = spec[spec > 0]
    if spec.size == 0:
        return 0.5
    log_mean = np.mean(np.log(spec))
    return float(np.clip(np.exp(log_mean) / np.mean(spec), 0.0, 1.0))


class MockDetector(DetectorProvider):
    name = "mock"
    is_mock = True

    def __init__(
        self,
        expected_samples: int = 64600,
        base_score: float = 0.5,
        spread: float = 0.35,
        latency_ms: float = 90.0,
    ) -> None:
        self.expected_samples = expected_samples
        self._base = float(np.clip(base_score, 0.0, 1.0))
        self._spread = float(np.clip(spread, 0.0, 1.0))
        self._latency_s = max(latency_ms, 0.0) / 1000.0

    def predict(self, audio: np.ndarray, sample_rate: int) -> DetectorResult:
        start = time.perf_counter()
        audio = fit_to_length(np.asarray(audio, dtype=np.float32), self.expected_samples)

        # Content-derived but fully deterministic: the same audio always
        # produces the same score, so demos are reproducible.
        digest = hashlib.blake2b(
            np.round(audio[::97] * 4096).astype(np.int32).tobytes(), digest_size=8
        ).digest()
        jitter = int.from_bytes(digest, "big") / float(1 << 64)   # [0, 1)
        flatness = _spectral_flatness(audio)

        # Weight content more than the hash so different files differ audibly.
        raw = self._base + self._spread * (0.65 * (flatness - 0.5) * 2.0
                                           + 0.35 * (jitter - 0.5) * 2.0)
        score = float(np.clip(raw, 0.005, 0.995))

        if self._latency_s:
            time.sleep(self._latency_s)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return DetectorResult(
            detector_score=score,
            synthetic_probability=score,
            human_probability=1.0 - score,
            # Confidence peaks when the score is decisive, not when it is 0.5.
            confidence=float(np.clip(abs(score - 0.5) * 2.0, 0.05, 0.99)),
            model=self.name,
            inference_ms=elapsed_ms,
            is_mock=True,
        )

    def warmup(self) -> None:
        self.predict(np.zeros(self.expected_samples, dtype=np.float32), 16000)

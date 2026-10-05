"""The detector abstraction — the most important interface in the system.

Everything downstream depends only on this. Swapping the mock for XLS-R+AASIST,
or later for a commercial provider, is a configuration change, not a rewrite
(plan §12.1, §43).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..models.risk import DetectorResult


class DetectorError(RuntimeError):
    """Raised when inference fails. Must not kill the WebSocket session."""


class DetectorProvider(ABC):
    """A synthetic-speech detector."""

    #: Human-readable model identifier surfaced to the UI and logs.
    name: str = "base"
    #: True only for simulated inference. Drives the mandatory UI warning (§40).
    is_mock: bool = False
    #: Native input length in samples, or None if the model is length-agnostic.
    expected_samples: int | None = None

    @abstractmethod
    def predict(self, audio: np.ndarray, sample_rate: int) -> DetectorResult:
        """Score one analysis window of mono float32 audio in [-1, 1]."""
        raise NotImplementedError

    def warmup(self) -> None:
        """Optional: run a dummy input so the first real window is not slow."""
        return

    def close(self) -> None:
        """Optional: release model resources."""
        return

    @property
    def device(self) -> str:
        return "cpu"

    def describe(self) -> dict:
        return {
            "model": self.name,
            "is_mock": self.is_mock,
            "device": self.device,
            "expected_samples": self.expected_samples,
        }


def fit_to_length(audio: np.ndarray, target: int) -> np.ndarray:
    """Match the model's native input length.

    Longer input is truncated; shorter input is tiled, which is the convention
    the AASIST-family reference implementations use.
    """
    n = audio.size
    if n == target:
        return audio.astype(np.float32, copy=False)
    if n > target:
        return audio[:target].astype(np.float32, copy=False)
    if n == 0:
        return np.zeros(target, dtype=np.float32)
    reps = int(np.ceil(target / n))
    return np.tile(audio, reps)[:target].astype(np.float32, copy=False)

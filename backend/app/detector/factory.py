"""Detector construction and the startup fallback policy.

A requested real detector that fails to load is a hard error unless fallback is
explicitly permitted. The system never silently substitutes mock inference for
requested real inference (plan §36, §47.5).
"""

from __future__ import annotations

from ..config import Settings
from ..utils.logging import get_logger
from .base import DetectorError, DetectorProvider
from .mock_detector import MockDetector

log = get_logger(__name__)


def build_detector(settings: Settings) -> DetectorProvider:
    provider = settings.detector_provider

    if provider == "mock":
        log.warning(
            "Using MOCK detector — simulated inference, not a real model",
            extra={"detector": "mock"},
        )
        return MockDetector(
            expected_samples=settings.window_samples,
            base_score=settings.mock_base_score,
            spread=settings.mock_spread,
            latency_ms=settings.mock_latency_ms,
        )

    try:
        from .xlsr_aasist import OnnxXlsrAasistDetector, TorchXlsrAasistDetector

        cls = {
            "xlsr_aasist_onnx": OnnxXlsrAasistDetector,
            "xlsr_aasist_torch": TorchXlsrAasistDetector,
        }[provider]
        detector = cls(
            model_path=settings.model_path or None,
            device=settings.model_device,
            threads=settings.inference_threads,
            temperature=settings.calibration_temperature,
        )
    except (DetectorError, ImportError, KeyError) as exc:
        if settings.allow_detector_fallback:
            log.error(
                "Real detector failed to load; ALLOW_DETECTOR_FALLBACK=true so "
                "falling back to MOCK inference",
                extra={"requested": provider, "error": str(exc)},
            )
            return MockDetector(
                expected_samples=settings.window_samples,
                base_score=settings.mock_base_score,
                spread=settings.mock_spread,
                latency_ms=settings.mock_latency_ms,
            )
        raise DetectorError(
            f"detector '{provider}' could not be loaded: {exc}. "
            f"Set ALLOW_DETECTOR_FALLBACK=true to permit mock fallback, "
            f"or DETECTOR_PROVIDER=mock to request mock explicitly."
        ) from exc

    return detector

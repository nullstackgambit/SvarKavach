"""Quality gating.

Abstention is a first-class output. When the audio is too poor to judge we
return INSUFFICIENT_EVIDENCE rather than letting a low-quality sample become a
confident synthetic verdict (plan §15).
"""

from __future__ import annotations

import numpy as np

from ..ingestion.normalizer import compute_stats
from ..models.risk import QualityFlag
from .windowing import AnalysisWindow


def assess_window(
    window: AnalysisWindow,
    *,
    min_speech_seconds: float,
    min_rms_dbfs: float,
    max_clipping_ratio: float,
    vad_active: bool,
) -> tuple[list[QualityFlag], dict]:
    """Return quality flags plus the metrics that produced them."""
    flags: list[QualityFlag] = []
    # Judge the real audio, not the zero-padding `flush()` adds to reach the
    # model's native length. Padding drags RMS down (6.1 dB for a 1 s tail in a
    # 4.04 s window) and dilutes the clipping ratio, so measuring the padded
    # array flags healthy tails as LOW_ENERGY and hides genuinely clipped ones.
    samples = window.analysable

    if samples.size == 0 or not np.all(np.isfinite(samples)):
        return [QualityFlag.INVALID_AUDIO], {"reason": "empty or non-finite samples"}

    stats = compute_stats(samples)

    # Only meaningful when a VAD is actually running; without one we cannot
    # distinguish speech from silence and must not claim to.
    if vad_active and window.speech_seconds < min_speech_seconds:
        flags.append(QualityFlag.INSUFFICIENT_SPEECH)

    if stats.rms_dbfs < min_rms_dbfs:
        flags.append(QualityFlag.LOW_ENERGY)

    if stats.clipping_ratio > max_clipping_ratio:
        flags.append(QualityFlag.CLIPPED)

    metrics = {
        **stats.as_dict(),
        "speech_seconds": round(window.speech_seconds, 3),
        "speech_ratio": round(window.speech_ratio, 3),
        "mean_speech_prob": round(window.mean_speech_prob, 3),
    }
    return (flags or [QualityFlag.OK]), metrics


def is_scoreable(flags: list[QualityFlag]) -> bool:
    """True when the window is good enough to send to the detector."""
    return flags == [QualityFlag.OK] or not flags

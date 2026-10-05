"""Risk states, detector results, and the deterministic recommendation text."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RiskState(str, Enum):
    """Actionable security states. Advisory only — the MVP never blocks."""

    NONE = "NONE"
    WATCH = "WATCH"
    VERIFY = "VERIFY"
    ESCALATE = "ESCALATE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


#: Deterministic operator guidance (plan §17). Never model-generated.
RECOMMENDATIONS: dict[RiskState, str] = {
    RiskState.NONE: (
        "No significant synthetic-voice signal detected. "
        "Continue normal verification procedures."
    ),
    RiskState.WATCH: (
        "Potential synthetic-voice indicators detected. "
        "Maintain heightened verification awareness."
    ),
    RiskState.VERIFY: (
        "Elevated synthetic-voice risk detected. Verify the caller using an "
        "independent trusted channel before authorizing sensitive actions."
    ),
    RiskState.ESCALATE: (
        "High synthetic-voice impersonation risk detected. Do not rely on the "
        "current voice channel alone. Escalate and require an independent "
        "verification step."
    ),
    RiskState.INSUFFICIENT_EVIDENCE: (
        "Audio quality is insufficient for a reliable judgement. No synthetic-voice "
        "determination has been made. Rely on standard verification procedures."
    ),
}


class QualityFlag(str, Enum):
    """Reasons a window may be unscoreable. Presence forces abstention."""

    OK = "OK"
    INSUFFICIENT_SPEECH = "INSUFFICIENT_SPEECH"
    LOW_ENERGY = "LOW_ENERGY"
    CLIPPED = "CLIPPED"
    INVALID_AUDIO = "INVALID_AUDIO"


@dataclass(slots=True)
class DetectorResult:
    """Output of a `DetectorProvider`. Terminology follows plan §28.

    `detector_score` is the raw, uncalibrated model output. Only call it a
    `synthetic_probability` once it has been through the calibration layer.
    """

    detector_score: float
    synthetic_probability: float
    human_probability: float
    confidence: float
    model: str
    inference_ms: float
    is_mock: bool = False
    raw_logits: tuple[float, float] | None = None

    def as_dict(self) -> dict:
        return {
            "detector_score": round(self.detector_score, 6),
            "synthetic_probability": round(self.synthetic_probability, 6),
            "human_probability": round(self.human_probability, 6),
            "confidence": round(self.confidence, 6),
            "model": self.model,
            "inference_ms": round(self.inference_ms, 2),
            "is_mock": self.is_mock,
        }


@dataclass(slots=True)
class RiskAssessment:
    """Final system-level judgement for one analysis window."""

    risk_state: RiskState
    risk_score: float
    smoothed_score: float
    context_multiplier: float
    recommendation: str
    sustained: bool = False
    quality_flags: list[QualityFlag] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "risk_state": self.risk_state.value,
            "risk_score": round(self.risk_score, 6),
            "smoothed_score": round(self.smoothed_score, 6),
            "context_multiplier": round(self.context_multiplier, 4),
            "recommendation": self.recommendation,
            "sustained": self.sustained,
            "quality_flags": [f.value for f in self.quality_flags],
        }

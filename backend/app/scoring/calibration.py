"""Calibration layer.

A raw neural-network output is not a probability. This layer is where a fitted
mapping belongs once representative data exists. Until then it is an explicit,
documented identity pass-through — the honest option (plan §28).

Temperature scaling is applied at the logit level inside the detector, which is
the correct place for it; this module handles any additional score-domain
mapping and is the hook for isotonic/piecewise calibration later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Calibrator:
    """Piecewise-linear score mapping. Identity unless control points are given.

    `points` is a list of (raw, calibrated) pairs with raw values strictly
    increasing. Values outside the range are clamped to the end points.
    """

    points: list[tuple[float, float]] | None = None
    fitted: bool = False

    def __post_init__(self) -> None:
        if self.points:
            xs = [p[0] for p in self.points]
            if len(xs) < 2:
                raise ValueError("calibration needs at least two control points")
            if any(b <= a for a, b in zip(xs, xs[1:])):
                raise ValueError("calibration raw values must strictly increase")
            self.fitted = True

    def calibrate(self, raw_score: float) -> float:
        score = float(min(max(raw_score, 0.0), 1.0))
        if not self.points:
            return score
        xs = np.array([p[0] for p in self.points], dtype=np.float64)
        ys = np.array([p[1] for p in self.points], dtype=np.float64)
        return float(np.clip(np.interp(score, xs, ys), 0.0, 1.0))

    def describe(self) -> dict:
        return {
            "fitted": self.fitted,
            "method": "piecewise_linear" if self.points else "identity",
            "note": (
                "Identity pass-through. Thresholds are demonstration defaults and "
                "have not been validated on representative data."
                if not self.points
                else "Piecewise-linear mapping from configured control points."
            ),
        }

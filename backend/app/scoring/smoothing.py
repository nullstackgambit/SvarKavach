"""Temporal fusion: EWMA plus a k-of-m sustained-evidence rule.

A single high window is not evidence. Requiring k of the last m windows above
threshold removes the dominant false-positive source — one bad window from
packet loss, cross-talk, or a clipped syllable (plan §14).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(slots=True)
class SmoothedScore:
    current: float
    smoothed: float
    session_mean: float
    windows_analyzed: int
    fraction_above_threshold: float
    sustained: bool


class ScoreSmoother:
    def __init__(
        self,
        alpha: float = 0.35,
        sustain_k: int = 3,
        sustain_m: int = 5,
        threshold: float = 0.65,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if sustain_k > sustain_m:
            raise ValueError("sustain_k must not exceed sustain_m")
        self._alpha = alpha
        self._k = sustain_k
        self._threshold = threshold
        self._recent: deque[float] = deque(maxlen=sustain_m)
        self._smoothed: float | None = None
        self._sum = 0.0
        self._count = 0
        self._above = 0

    def update(self, score: float) -> SmoothedScore:
        score = float(min(max(score, 0.0), 1.0))
        # First observation seeds the EWMA rather than being dragged toward zero.
        self._smoothed = score if self._smoothed is None else (
            self._alpha * score + (1.0 - self._alpha) * self._smoothed
        )
        self._sum += score
        self._count += 1
        if score >= self._threshold:
            self._above += 1
        self._recent.append(score)

        return SmoothedScore(
            current=score,
            smoothed=self._smoothed,
            session_mean=self._sum / self._count,
            windows_analyzed=self._count,
            fraction_above_threshold=self._above / self._count,
            sustained=sum(1 for s in self._recent if s >= self._threshold) >= self._k,
        )

    @property
    def windows_analyzed(self) -> int:
        return self._count

    @property
    def smoothed(self) -> float:
        return self._smoothed if self._smoothed is not None else 0.0

    @property
    def session_mean(self) -> float:
        return self._sum / self._count if self._count else 0.0

    @property
    def sustained(self) -> bool:
        """Whether k of the last m windows are above the threshold.

        This is the evidence rule itself. Deriving 'sustained' from the smoothed
        score instead would make the rule a no-op, because a single spike can
        drag the EWMA above the threshold on its own.
        """
        return sum(1 for score in self._recent if score >= self._threshold) >= self._k

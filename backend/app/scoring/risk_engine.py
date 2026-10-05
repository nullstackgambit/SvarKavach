"""Risk engine: detector output + context -> an actionable security state.

This is policy, not modelling. No threshold lives inside the detector and no
model-specific logic lives here (plan §43). The engine is advisory: there is
deliberately no BLOCK action.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models.messages import SessionContext
from ..models.risk import (
    RECOMMENDATIONS,
    QualityFlag,
    RiskAssessment,
    RiskState,
)


@dataclass(slots=True)
class RiskThresholds:
    watch: float = 0.40
    verify: float = 0.65
    escalate: float = 0.85

    def __post_init__(self) -> None:
        if not self.watch < self.verify < self.escalate:
            raise ValueError("thresholds must be strictly increasing")


class RiskEngine:
    def __init__(
        self,
        thresholds: RiskThresholds,
        high_value_transaction: float = 1_000_000.0,
        high_value_multiplier: float = 0.10,
        unknown_contact_multiplier: float = 0.05,
        require_sustained: bool = True,
    ) -> None:
        self._t = thresholds
        self._high_value = high_value_transaction
        self._high_value_mult = high_value_multiplier
        self._unknown_mult = unknown_contact_multiplier
        self._require_sustained = require_sustained

    # -- context -------------------------------------------------------
    def context_multiplier(self, context: SessionContext) -> float:
        """Demonstration mechanism, NOT a validated fraud-risk model (plan §16)."""
        multiplier = 1.0
        if context.transaction_value >= self._high_value:
            multiplier += self._high_value_mult
        if not context.contact_known:
            multiplier += self._unknown_mult
        return multiplier

    # -- state ---------------------------------------------------------
    def _state_for(self, score: float) -> RiskState:
        if score >= self._t.escalate:
            return RiskState.ESCALATE
        if score >= self._t.verify:
            return RiskState.VERIFY
        if score >= self._t.watch:
            return RiskState.WATCH
        return RiskState.NONE

    def evaluate(
        self,
        smoothed_score: float,
        context: SessionContext,
        *,
        quality_flags: list[QualityFlag] | None = None,
        sustained: bool = False,
    ) -> RiskAssessment:
        flags = [f for f in (quality_flags or []) if f is not QualityFlag.OK]

        # Bad audio never becomes a confident verdict.
        if flags:
            return RiskAssessment(
                risk_state=RiskState.INSUFFICIENT_EVIDENCE,
                risk_score=0.0,
                smoothed_score=float(min(max(smoothed_score, 0.0), 1.0)),
                context_multiplier=1.0,
                recommendation=RECOMMENDATIONS[RiskState.INSUFFICIENT_EVIDENCE],
                sustained=False,
                quality_flags=flags,
            )

        multiplier = self.context_multiplier(context)
        risk_score = min(max(smoothed_score * multiplier, 0.0), 1.0)
        state = self._state_for(risk_score)

        # Sustained-evidence gate: a transient spike may raise awareness but
        # must not trigger an operator action on its own.
        if (
            self._require_sustained
            and not sustained
            and state in (RiskState.VERIFY, RiskState.ESCALATE)
        ):
            state = RiskState.WATCH

        return RiskAssessment(
            risk_state=state,
            risk_score=risk_score,
            smoothed_score=float(min(max(smoothed_score, 0.0), 1.0)),
            context_multiplier=multiplier,
            recommendation=RECOMMENDATIONS[state],
            sustained=sustained,
            quality_flags=[],
        )

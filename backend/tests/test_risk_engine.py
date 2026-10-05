import pytest

from backend.app.models.messages import SessionContext
from backend.app.models.risk import RECOMMENDATIONS, QualityFlag, RiskState
from backend.app.scoring.risk_engine import RiskEngine, RiskThresholds


@pytest.fixture()
def engine():
    return RiskEngine(RiskThresholds(0.40, 0.65, 0.85))


PLAIN = SessionContext(contact_known=True, transaction_value=0)


@pytest.mark.parametrize("score,expected", [
    (0.00, RiskState.NONE), (0.399, RiskState.NONE),
    (0.40, RiskState.WATCH), (0.649, RiskState.WATCH),
    (0.65, RiskState.VERIFY), (0.849, RiskState.VERIFY),
    (0.85, RiskState.ESCALATE), (1.00, RiskState.ESCALATE),
])
def test_threshold_boundaries(engine, score, expected):
    assert engine.evaluate(score, PLAIN, sustained=True).risk_state is expected


def test_context_multiplier_components(engine):
    assert engine.context_multiplier(PLAIN) == pytest.approx(1.0)
    assert engine.context_multiplier(
        SessionContext(contact_known=True, transaction_value=2_000_000)) == pytest.approx(1.10)
    assert engine.context_multiplier(
        SessionContext(contact_known=False, transaction_value=0)) == pytest.approx(1.05)
    assert engine.context_multiplier(
        SessionContext(contact_known=False, transaction_value=2_000_000)) == pytest.approx(1.15)


def test_context_can_push_score_across_a_threshold(engine):
    ctx = SessionContext(contact_known=False, transaction_value=2_500_000)
    assert engine.evaluate(0.60, PLAIN, sustained=True).risk_state is RiskState.WATCH
    assert engine.evaluate(0.60, ctx, sustained=True).risk_state is RiskState.VERIFY


def test_risk_score_is_clamped_to_one(engine):
    ctx = SessionContext(contact_known=False, transaction_value=9_000_000)
    assert engine.evaluate(0.99, ctx, sustained=True).risk_score == 1.0


def test_score_is_clamped_at_zero(engine):
    assert engine.evaluate(-5.0, PLAIN, sustained=True).risk_score == 0.0


def test_unsustained_evidence_cannot_exceed_watch(engine):
    assert engine.evaluate(0.95, PLAIN, sustained=False).risk_state is RiskState.WATCH
    assert engine.evaluate(0.95, PLAIN, sustained=True).risk_state is RiskState.ESCALATE


def test_quality_flags_force_abstention(engine):
    result = engine.evaluate(0.99, PLAIN,
                             quality_flags=[QualityFlag.LOW_ENERGY], sustained=True)
    assert result.risk_state is RiskState.INSUFFICIENT_EVIDENCE
    assert result.risk_score == 0.0
    assert QualityFlag.LOW_ENERGY in result.quality_flags


def test_ok_flag_does_not_trigger_abstention(engine):
    result = engine.evaluate(0.90, PLAIN, quality_flags=[QualityFlag.OK], sustained=True)
    assert result.risk_state is RiskState.ESCALATE


def test_every_state_has_recommendation_text(engine):
    for state in RiskState:
        assert RECOMMENDATIONS[state].strip()


def test_recommendation_matches_state(engine):
    result = engine.evaluate(0.70, PLAIN, sustained=True)
    assert result.recommendation == RECOMMENDATIONS[RiskState.VERIFY]


def test_thresholds_must_increase():
    with pytest.raises(ValueError):
        RiskThresholds(0.8, 0.5, 0.9)


def test_engine_never_emits_a_block_state(engine):
    states = {engine.evaluate(s / 20, PLAIN, sustained=True).risk_state
              for s in range(21)}
    assert all(s in set(RiskState) for s in states)
    assert not any("BLOCK" in s.value for s in states)

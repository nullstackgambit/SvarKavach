import pytest

from backend.app.scoring.smoothing import ScoreSmoother


def test_first_prediction_seeds_the_average():
    assert ScoreSmoother(alpha=0.35).update(0.8).smoothed == pytest.approx(0.8)


def test_repeated_identical_scores_converge_to_that_score():
    s = ScoreSmoother(alpha=0.35)
    for _ in range(10):
        result = s.update(0.9)
    assert result.smoothed == pytest.approx(0.9, abs=1e-6)


def test_ewma_follows_the_documented_formula():
    s = ScoreSmoother(alpha=0.35)
    s.update(1.0)
    assert s.update(0.0).smoothed == pytest.approx(0.65)


def test_scores_are_clamped_to_unit_range():
    s = ScoreSmoother()
    assert s.update(5.0).current == 1.0
    assert ScoreSmoother().update(-5.0).current == 0.0


def test_single_spike_is_not_sustained():
    s = ScoreSmoother(alpha=0.35, sustain_k=3, sustain_m=5, threshold=0.65)
    for v in (0.1, 0.1, 0.95, 0.1, 0.1):
        result = s.update(v)
    assert result.sustained is False


def test_k_of_m_triggers_when_sustained():
    s = ScoreSmoother(alpha=0.35, sustain_k=3, sustain_m=5, threshold=0.65)
    for v in (0.1, 0.9, 0.9, 0.2, 0.9):
        result = s.update(v)
    assert result.sustained is True


def test_sustained_window_expires_old_scores():
    s = ScoreSmoother(alpha=0.35, sustain_k=3, sustain_m=5, threshold=0.65)
    for v in (0.9, 0.9, 0.9):
        assert s.update(v).sustained in (False, True)
    for v in (0.1, 0.1, 0.1, 0.1, 0.1):
        result = s.update(v)
    assert result.sustained is False


def test_session_statistics_are_tracked():
    s = ScoreSmoother(alpha=0.5, threshold=0.65)
    for v in (0.8, 0.4, 0.9, 0.2):
        result = s.update(v)
    assert result.windows_analyzed == 4
    assert result.session_mean == pytest.approx(0.575)
    assert result.fraction_above_threshold == pytest.approx(0.5)


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        ScoreSmoother(alpha=0.0)
    with pytest.raises(ValueError):
        ScoreSmoother(alpha=1.5)
    with pytest.raises(ValueError):
        ScoreSmoother(sustain_k=6, sustain_m=5)


def test_sustained_property_matches_last_update():
    """The exposed rule must agree with what update() computed."""
    s = ScoreSmoother(alpha=0.35, sustain_k=3, sustain_m=5, threshold=0.65)
    for v in (0.1, 0.9, 0.9, 0.2, 0.9):
        result = s.update(v)
    assert s.sustained is result.sustained is True

    s2 = ScoreSmoother(alpha=0.35, sustain_k=3, sustain_m=5, threshold=0.65)
    for v in (0.1, 0.1, 0.99, 0.1, 0.1):
        result2 = s2.update(v)
    assert s2.sustained is result2.sustained is False


def test_a_single_spike_never_counts_as_sustained_even_if_it_lifts_the_ewma():
    """Regression: one 0.99 window must not produce a sustained verdict.

    A high spike can drag the EWMA above the threshold on its own; the evidence
    rule must still say 'not sustained'.
    """
    s = ScoreSmoother(alpha=0.9, sustain_k=3, sustain_m=5, threshold=0.65)
    for v in (0.01, 0.01, 0.999):
        s.update(v)
    assert s.smoothed > 0.65, "the spike should have lifted the EWMA"
    assert s.sustained is False, "but one window is not sustained evidence"

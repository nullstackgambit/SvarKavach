import numpy as np
import pytest

from backend.app.detector.base import DetectorProvider, fit_to_length
from backend.app.detector.factory import build_detector
from backend.app.detector.mock_detector import MockDetector
from backend.app.detector.xlsr_aasist import logits_to_scores, preemphasis, prepare_input


def test_fit_to_length_truncates_and_tiles():
    assert fit_to_length(np.arange(100_000, dtype=np.float32), 64600).size == 64600
    assert fit_to_length(np.arange(10, dtype=np.float32), 64600).size == 64600
    assert np.allclose(fit_to_length(np.array([1, 2, 3], dtype=np.float32), 8),
                       [1, 2, 3, 1, 2, 3, 1, 2])


def test_fit_to_length_handles_empty_input():
    assert np.all(fit_to_length(np.zeros(0, dtype=np.float32), 16) == 0)


def test_mock_detector_is_deterministic():
    d = MockDetector(latency_ms=0.0)
    audio = (np.random.default_rng(1).standard_normal(64600) * 0.1).astype(np.float32)
    assert d.predict(audio, 16000).synthetic_probability == \
           d.predict(audio, 16000).synthetic_probability


def test_mock_detector_labels_itself():
    result = MockDetector(latency_ms=0.0).predict(np.zeros(64600, dtype=np.float32), 16000)
    assert result.is_mock is True and result.model == "mock"
    assert 0.0 <= result.synthetic_probability <= 1.0
    assert result.human_probability == pytest.approx(1 - result.synthetic_probability)
    assert result.inference_ms >= 0.0


def test_mock_implements_the_interface():
    assert isinstance(MockDetector(), DetectorProvider)


def test_preemphasis_matches_the_reference_formula():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert np.allclose(preemphasis(x), [1.0, 2 - 0.97, 3 - 0.97 * 2, 4 - 0.97 * 3])


def test_preemphasis_handles_empty():
    assert preemphasis(np.zeros(0, dtype=np.float32)).size == 0


def test_prepare_input_produces_native_length():
    assert prepare_input(np.ones(1000, dtype=np.float32)).size == 64600


def test_logits_to_scores_index_convention():
    """Index 0 is spoof, index 1 is bona fide — verified against the upstream model."""
    p_spoof, p_bona = logits_to_scores(np.array([2.0, -2.0]))
    assert p_spoof > 0.9 and p_bona < 0.1
    p_spoof, p_bona = logits_to_scores(np.array([-2.0, 2.0]))
    assert p_spoof < 0.1 and p_bona > 0.9
    assert p_spoof + p_bona == pytest.approx(1.0)


def test_temperature_softens_confidence():
    sharp, _ = logits_to_scores(np.array([3.0, -3.0]), temperature=1.0)
    soft, _ = logits_to_scores(np.array([3.0, -3.0]), temperature=5.0)
    assert sharp > soft > 0.5


def test_logits_to_scores_rejects_wrong_shape():
    from backend.app.detector.base import DetectorError
    with pytest.raises(DetectorError):
        logits_to_scores(np.array([1.0, 2.0, 3.0]))


def test_factory_returns_mock_when_requested(settings):
    assert build_detector(settings).is_mock is True


def test_factory_refuses_silent_fallback():
    """A requested real detector must fail loudly, not degrade to mock (§47.5)."""
    from backend.app.config import Settings
    from backend.app.detector.base import DetectorError
    s = Settings(detector_provider="xlsr_aasist_onnx",
                 model_path="/nonexistent/model.onnx",
                 allow_detector_fallback=False)
    with pytest.raises(DetectorError):
        build_detector(s)


def test_factory_allows_fallback_only_when_opted_in():
    from backend.app.config import Settings
    s = Settings(detector_provider="xlsr_aasist_onnx",
                 model_path="/nonexistent/model.onnx",
                 allow_detector_fallback=True)
    assert build_detector(s).is_mock is True

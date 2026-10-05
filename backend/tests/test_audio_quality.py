import numpy as np
import pytest

from backend.app.models.risk import QualityFlag
from backend.app.preprocessing.audio_quality import assess_window, is_scoreable
from backend.app.preprocessing.windowing import AnalysisWindow

GATES = dict(min_speech_seconds=1.5, min_rms_dbfs=-55.0, max_clipping_ratio=0.02)


def window(samples, speech_seconds=3.0):
    return AnalysisWindow(
        samples=samples, start_sample=0, end_sample=samples.size,
        sample_rate=16000, speech_seconds=speech_seconds,
        speech_ratio=speech_seconds / 4.0, mean_speech_prob=0.9,
    )


def test_healthy_window_passes():
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(64600) * 0.1).astype(np.float32)
    flags, metrics = assess_window(window(audio), vad_active=True, **GATES)
    assert flags == [QualityFlag.OK] and is_scoreable(flags)
    assert "rms_dbfs" in metrics


def test_short_speech_is_flagged():
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(64600) * 0.1).astype(np.float32)
    flags, _ = assess_window(window(audio, speech_seconds=0.4), vad_active=True, **GATES)
    assert QualityFlag.INSUFFICIENT_SPEECH in flags and not is_scoreable(flags)


def test_speech_gate_is_skipped_when_vad_is_off():
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(64600) * 0.1).astype(np.float32)
    flags, _ = assess_window(window(audio, speech_seconds=0.0), vad_active=False, **GATES)
    assert QualityFlag.INSUFFICIENT_SPEECH not in flags


def test_low_energy_is_flagged():
    audio = (np.ones(64600, dtype=np.float32) * 1e-5)
    flags, _ = assess_window(window(audio), vad_active=True, **GATES)
    assert QualityFlag.LOW_ENERGY in flags


def test_clipping_is_flagged():
    audio = np.ones(64600, dtype=np.float32)
    flags, _ = assess_window(window(audio), vad_active=True, **GATES)
    assert QualityFlag.CLIPPED in flags


def test_non_finite_audio_is_invalid():
    audio = np.full(64600, np.nan, dtype=np.float32)
    flags, _ = assess_window(window(audio), vad_active=True, **GATES)
    assert flags == [QualityFlag.INVALID_AUDIO] and not is_scoreable(flags)


def test_empty_window_is_invalid():
    flags, _ = assess_window(window(np.zeros(0, dtype=np.float32)), vad_active=True, **GATES)
    assert flags == [QualityFlag.INVALID_AUDIO]


# ---- padded tail windows (audit M4) ----------------------------------
# `SlidingWindowBuffer.flush()` right-aligns a short tail into a full-length
# window and zero-pads the left, so the detector always gets its native input
# size. The quality gate must judge the audio, not the padding: measuring the
# padded array costs 6.1 dB for a 1 s tail in a 4.04 s window, which flagged
# healthy tails as LOW_ENERGY and diluted the clipping ratio of clipped ones.
def padded_tail(tail: np.ndarray, window_samples: int = 64600) -> AnalysisWindow:
    """Mirror what flush() builds, without needing the buffer."""
    samples = np.zeros(window_samples, dtype=np.float32)
    samples[-tail.size:] = tail
    return AnalysisWindow(
        samples=samples, start_sample=0, end_sample=tail.size,
        sample_rate=16000, speech_seconds=3.0, speech_ratio=0.75,
        mean_speech_prob=0.9, is_final=True, valid_samples=tail.size,
    )


def test_quiet_but_valid_tail_is_not_gated_by_its_own_padding():
    # -52 dBFS clears the -55 gate on its own; padding to full length used to
    # drag it to -58 dBFS and abstain on perfectly usable audio.
    tail = np.full(16000, 10 ** (-52 / 20), dtype=np.float32)
    flags, metrics = assess_window(padded_tail(tail), vad_active=True, **GATES)
    assert metrics["rms_dbfs"] == pytest.approx(-52.0, abs=0.1)
    assert flags == [QualityFlag.OK] and is_scoreable(flags)


def test_clipped_tail_is_still_flagged_despite_padding():
    # The other direction: padding must not dilute a clipped tail under the gate.
    tail = np.ones(16000, dtype=np.float32)
    flags, metrics = assess_window(padded_tail(tail), vad_active=True, **GATES)
    assert metrics["clipping_ratio"] == pytest.approx(1.0)
    assert QualityFlag.CLIPPED in flags and not is_scoreable(flags)


def test_padding_does_not_change_what_the_detector_receives():
    tail = np.full(16000, 0.1, dtype=np.float32)
    window = padded_tail(tail)
    assert window.samples.size == 64600, "detector must still get its native length"
    assert window.analysable.size == 16000, "the gate must see only the real audio"

"""VAD behaviour. Needs genuine recorded speech, so it skips where none exists."""

import numpy as np
import pytest

from backend.app.preprocessing.vad import (
    SILERO_CONTEXT_SAMPLES, SILERO_FRAME_SAMPLES, EnergyVad, VadFactory, build_vad,
)


def frame_probs(vad, audio):
    vad.reset()
    return np.array([
        vad.speech_probability(audio[i:i + SILERO_FRAME_SAMPLES])
        for i in range(0, len(audio) - SILERO_FRAME_SAMPLES + 1, SILERO_FRAME_SAMPLES)
    ])


def test_silero_detects_real_speech(real_speech_16k):
    vad = build_vad(True, None)
    if vad.is_fallback:
        pytest.skip("Silero model not downloaded")
    probs = frame_probs(vad, real_speech_16k)
    assert probs.max() > 0.8
    assert (probs >= 0.5).mean() > 0.3, "Silero should fire on a majority of speech"


def test_silero_rejects_silence():
    vad = build_vad(True, None)
    if vad.is_fallback:
        pytest.skip("Silero model not downloaded")
    probs = frame_probs(vad, np.zeros(16000 * 3, dtype=np.float32))
    assert probs.max() < 0.2


def test_silero_uses_context_window():
    """v5+ needs 64 samples of previous-frame context; without it every
    probability collapses to ~0.001, including on clear speech."""
    assert SILERO_CONTEXT_SAMPLES == 64
    assert SILERO_FRAME_SAMPLES == 512


def test_silero_rejects_wrong_frame_size():
    vad = build_vad(True, None)
    if vad.is_fallback:
        pytest.skip("Silero model not downloaded")
    with pytest.raises(ValueError):
        vad.speech_probability(np.zeros(100, dtype=np.float32))


def test_energy_fallback_discriminates_by_level():
    vad = EnergyVad()
    assert vad.speech_probability(np.zeros(512, dtype=np.float32)) == 0.0
    loud = np.ones(512, dtype=np.float32) * 0.3
    assert vad.speech_probability(loud) > 0.5


def test_energy_fallback_is_labelled_as_fallback():
    assert EnergyVad().is_fallback is True
    assert build_vad(False, None) is None


# ---- per-session isolation (audit H1) --------------------------------
def test_factory_hands_out_independent_instances():
    factory = VadFactory(True, None)
    first, second = factory(), factory()
    assert first is not second
    assert type(first) is type(second)
    assert factory.name == first.name
    assert factory.is_fallback == first.is_fallback


def test_factory_reports_disabled_without_building_anything():
    factory = VadFactory(False, None)
    assert factory() is None
    assert factory.name == "disabled" and factory.is_fallback is False


def test_one_vad_per_session_or_concurrent_streams_corrupt_each_other(real_speech_16k):
    """Silero carries LSTM state, so two streams must never share an instance.

    A single app-wide VAD was handed to every session. Frames from concurrent
    sessions then interleaved into one recurrent state: measured on this clip,
    sharing with one other session cut detected speech from 1.02 s to 0.42 s,
    which drops windows under MIN_SPEECH_SECONDS and forces spurious abstention.
    """
    factory = VadFactory(True, None)
    if factory.is_fallback:
        pytest.skip("Silero model not downloaded")

    speech = real_speech_16k[:SILERO_FRAME_SAMPLES * 40]
    silence = np.zeros_like(speech)

    def voiced(probs):
        return int((probs >= 0.5).sum())

    alone = voiced(frame_probs(factory(), speech))
    assert alone > 0, "fixture carries no detectable speech"

    # Two sessions running at once, each with its own VAD (the fixed design).
    session_a, session_b = factory(), factory()
    session_a.reset(); session_b.reset()
    interleaved = []
    for i in range(0, speech.size - SILERO_FRAME_SAMPLES + 1, SILERO_FRAME_SAMPLES):
        session_b.speech_probability(silence[i:i + SILERO_FRAME_SAMPLES])
        interleaved.append(session_a.speech_probability(speech[i:i + SILERO_FRAME_SAMPLES]))

    assert voiced(np.array(interleaved)) == alone, (
        "session A's speech accounting changed because another session was live"
    )

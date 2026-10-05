"""The final verdict must obey the same evidence rule as the live events.

Regression: `finish()` once derived `sustained` from the smoothed score, which
made the k-of-m rule a no-op for the final result — a single spiking window
could drag the EWMA up and produce ESCALATE on its own. Live capture surfaced
this, because long sessions contain quiet windows that score high.
"""

import asyncio

import numpy as np
import pytest

from backend.app.config import Settings
from backend.app.detector.base import DetectorProvider
from backend.app.ingestion.session_manager import SessionManager
from backend.app.models.messages import AudioChunk, SessionContext
from backend.app.models.risk import DetectorResult, RiskState
from backend.app.pipeline.stream_processor import StreamProcessor
from backend.app.ingestion.normalizer import float_to_pcm16


class ScriptedDetector(DetectorProvider):
    """Returns a fixed sequence of scores, repeating the last one."""

    name = "scripted"
    is_mock = True

    def __init__(self, scores):
        self.scores = list(scores)
        self.expected_samples = 64600
        self._i = 0

    def predict(self, audio, sample_rate):
        score = self.scores[min(self._i, len(self.scores) - 1)]
        self._i += 1
        return DetectorResult(
            detector_score=score, synthetic_probability=score,
            human_probability=1 - score, confidence=abs(score - 0.5) * 2,
            model=self.name, inference_ms=0.0, is_mock=True,
        )


async def run_session(scores, seconds=20):
    settings = Settings(detector_provider="mock", vad_enabled=False,
                        mock_latency_ms=0.0, smoothing_alpha=0.9)
    sessions = SessionManager()
    session = sessions.create("verdict", "demo_web", SessionContext(
        contact_known=True, transaction_value=0), "scripted")
    events = []

    async def emit(payload):
        events.append(payload)

    processor = StreamProcessor(settings, ScriptedDetector(scores), None, session, emit)
    await processor.start()

    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(16000 * seconds) * 0.2).astype(np.float32)
    for i in range(0, audio.size, 4000):
        await processor.push_audio(AudioChunk(
            "verdict", f"verdict:{i}", 0, float_to_pcm16(audio[i:i + 4000]), 16000, 1))
        # Yield so the inference worker actually drains windows. Pushing in a
        # tight loop would leave everything to latest-window-wins and score only
        # one or two windows, which cannot exercise a k-of-m rule.
        await asyncio.sleep(0.002)
    final = await processor.finish()
    await processor.close()
    return final, events


@pytest.mark.asyncio
async def test_single_spike_does_not_escalate_the_final_verdict():
    # Exactly one high window among clean ones. alpha=0.9 makes the spike
    # dominate the EWMA briefly, so only the evidence rule can hold the line.
    # The trailing lows matter: ScriptedDetector repeats its final value.
    final, _ = await run_session([0.01] * 8 + [0.999] + [0.01] * 40)
    assert final["risk_state"] not in (RiskState.ESCALATE.value, RiskState.VERIFY.value), \
        f"a lone spike must not reach {final['risk_state']} without sustained evidence"


@pytest.mark.asyncio
async def test_sustained_high_scores_do_escalate():
    """The rule must not be so strict that a genuine attack is missed."""
    final, _ = await run_session([0.95] * 12)
    assert final["risk_state"] == RiskState.ESCALATE.value


@pytest.mark.asyncio
async def test_final_state_is_consistent_with_the_last_live_event():
    """The final card must not contradict what the operator just watched."""
    final, events = await run_session([0.01] * 8 + [0.999] + [0.01] * 40)
    live = [e for e in events if e["type"] == "analysis"]
    assert live, "expected live analysis events"
    ranking = {RiskState.NONE.value: 0, RiskState.WATCH.value: 1,
               RiskState.VERIFY.value: 2, RiskState.ESCALATE.value: 3}
    worst_live = max(ranking.get(e["risk_state"], 0) for e in live)
    assert ranking.get(final["risk_state"], 0) <= worst_live, (
        f"final {final['risk_state']} exceeds the worst live state seen")

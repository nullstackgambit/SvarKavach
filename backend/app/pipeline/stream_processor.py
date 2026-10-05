"""The detection pipeline: audio in, risk events out.

    AudioChunk -> normalise -> buffer/VAD -> window -> quality gate
               -> detector -> calibrate -> smooth -> risk -> events

Inference never runs on the event loop. Windows are handed to a worker task
which executes the (blocking) model call in a thread (plan §13).

Backpressure: completed windows wait in a bounded backlog, oldest first. The
backlog is sized so one maximum-size chunk can never overflow it, so how a
client batches its bytes never decides what gets scored — only a genuinely slow
detector causes drops. When inference does fall far enough behind to fill the
backlog, the oldest window is dropped so the analysis stays live instead of
drifting further behind. Skipped windows are counted and reported, never hidden.
On a GPU this never triggers.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from typing import Awaitable, Callable

import numpy as np

from ..config import Settings
from ..detector.base import DetectorError, DetectorProvider
from ..ingestion.normalizer import InvalidAudioError, normalize_chunk
from ..ingestion.session_manager import Session
from ..models.messages import AudioChunk, SessionState
from ..models.risk import QualityFlag, RiskState
from ..preprocessing.audio_quality import assess_window, is_scoreable
from ..preprocessing.vad import VoiceActivityDetector
from ..preprocessing.windowing import AnalysisWindow, SlidingWindowBuffer
from ..scoring.calibration import Calibrator
from ..scoring.risk_engine import RiskEngine, RiskThresholds
from ..scoring.smoothing import ScoreSmoother
from ..utils.logging import get_logger

log = get_logger(__name__)

EmitFn = Callable[[dict], Awaitable[None]]


def json_safe(value):
    """Replace non-finite floats so the payload is valid JSON.

    `json.dumps` happily emits `NaN` / `Infinity` / `-Infinity`, none of which
    are legal JSON — a browser's `JSON.parse` throws on them and the event is
    lost. Individual producers already avoid non-finite values; this is the
    backstop so a future metric cannot silently reintroduce the problem.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


class StreamProcessor:
    """Owns all per-session pipeline state."""

    def __init__(
        self,
        settings: Settings,
        detector: DetectorProvider,
        vad: VoiceActivityDetector | None,
        session: Session,
        emit: EmitFn,
    ) -> None:
        self._s = settings
        self._detector = detector
        self._vad = vad
        self._session = session
        self._emit = emit

        self._buffer = SlidingWindowBuffer(
            window_samples=settings.window_samples,
            hop_samples=settings.hop_samples,
            sample_rate=settings.target_sample_rate,
            vad=vad,
            vad_threshold=settings.vad_threshold,
        )
        self._smoother = ScoreSmoother(
            alpha=settings.smoothing_alpha,
            sustain_k=settings.sustained_k,
            sustain_m=settings.sustained_m,
            threshold=settings.verify_threshold,
        )
        self._risk = RiskEngine(
            RiskThresholds(
                watch=settings.watch_threshold,
                verify=settings.verify_threshold,
                escalate=settings.escalate_threshold,
            ),
            high_value_transaction=settings.high_value_transaction,
            high_value_multiplier=settings.high_value_multiplier,
            unknown_contact_multiplier=settings.unknown_contact_multiplier,
        )
        self._calibrator = Calibrator()

        # A bounded backlog, not a single slot. With one slot the drop policy
        # depended on how the *client* batched its bytes: a single push_audio()
        # completing several windows discarded all but the last even with an
        # instantaneous detector, losing 93% of the audio at the legal 8 s
        # maximum chunk size. Dropping is now driven by the worker genuinely
        # falling behind, which is what "latest-window-wins" was meant to mean.
        self._pending: deque[AnalysisWindow] = deque(
            maxlen=self._backlog_capacity(settings)
        )
        self._wakeup = asyncio.Event()
        self._worker: asyncio.Task | None = None
        self._busy = asyncio.Lock()
        self._closing = False
        self._last_state: RiskState | None = None
        self._last_inference_ms: float = 0.0

    # -- lifecycle -----------------------------------------------------
    async def start(self) -> None:
        self._worker = asyncio.create_task(self._run_worker())

    async def close(self) -> None:
        self._closing = True
        self._wakeup.set()
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    # -- ingest --------------------------------------------------------
    async def push_audio(self, chunk: AudioChunk) -> None:
        """Normalise a chunk and queue any windows it completes."""
        try:
            samples, _stats = normalize_chunk(chunk, self._s.target_sample_rate)
        except InvalidAudioError:
            raise

        self._session.bytes_received += len(chunk.audio_pcm16)
        self._session.samples_received += int(samples.size)
        self._session.last_audio_at = time.time()

        self._buffer.push(samples)
        for window in self._buffer.pop_windows():
            self._offer(window)

    @staticmethod
    def _backlog_capacity(settings: Settings) -> int:
        """How many completed windows may wait for the worker.

        Sized so one maximum-size chunk can never overflow it: MAX_CHUNK_BYTES
        is 8.19 s at 16 kHz, or 17 windows at a 0.5 s hop. Beyond that the
        oldest window is dropped and counted, so a genuinely slow detector still
        degrades to newest-first rather than growing memory without bound.
        """
        max_samples = settings.max_chunk_bytes // 2          # PCM16, mono
        per_chunk = -(-max_samples // settings.hop_samples)  # ceil division
        return max(2, min(per_chunk + 1, 64))

    def _offer(self, window: AnalysisWindow) -> None:
        """Queue a window, dropping the oldest once the backlog is full."""
        if len(self._pending) == self._pending.maxlen:
            # deque(maxlen=) evicts the oldest on append; count what we lose.
            self._session.windows_skipped += 1
        self._pending.append(window)
        self._wakeup.set()

    # -- worker --------------------------------------------------------
    async def _run_worker(self) -> None:
        while not self._closing:
            await self._wakeup.wait()
            self._wakeup.clear()
            while (window := self._take_pending()) is not None:
                async with self._busy:
                    await self._process(window)

    def _take_pending(self) -> AnalysisWindow | None:
        return self._pending.popleft() if self._pending else None

    async def _process(self, window: AnalysisWindow) -> None:
        flags, metrics = assess_window(
            window,
            min_speech_seconds=self._s.min_speech_seconds,
            min_rms_dbfs=self._s.min_rms_dbfs,
            max_clipping_ratio=self._s.max_clipping_ratio,
            vad_active=self._vad is not None,
        )

        if not is_scoreable(flags):
            self._session.windows_abstained += 1
            assessment = self._risk.evaluate(
                self._smoother.smoothed,
                self._session.context,
                quality_flags=flags,
            )
            await self._emit_analysis(window, None, assessment, metrics)
            return

        try:
            # The blocking model call goes to a thread so the event loop — and
            # therefore every other session — stays responsive.
            result = await asyncio.to_thread(
                self._detector.predict, window.samples, window.sample_rate
            )
        except DetectorError as exc:
            log.error(
                "detector failed for window",
                extra={
                    "session_id": self._session.session_id,
                    "window_start": round(window.start_seconds, 3),
                    "error": str(exc),
                },
            )
            await self._emit(
                {
                    "type": "error",
                    "code": "DETECTOR_ERROR",
                    "message": f"Inference failed for window at "
                               f"{window.start_seconds:.2f}s; analysis continues.",
                }
            )
            return

        self._last_inference_ms = result.inference_ms
        calibrated = self._calibrator.calibrate(result.synthetic_probability)
        smoothed = self._smoother.update(calibrated)
        assessment = self._risk.evaluate(
            smoothed.smoothed,
            self._session.context,
            quality_flags=[],
            sustained=smoothed.sustained,
        )
        self._session.windows_processed += 1

        log.info(
            "window analysed",
            extra={
                "session_id": self._session.session_id,
                "window_start": round(window.start_seconds, 3),
                "window_end": round(window.end_seconds, 3),
                "speech_duration": metrics.get("speech_seconds"),
                "synthetic_probability": round(calibrated, 4),
                "smoothed_score": round(smoothed.smoothed, 4),
                "risk_state": assessment.risk_state.value,
                "model": result.model,
                "inference_ms": round(result.inference_ms, 1),
            },
        )
        await self._emit_analysis(window, result, assessment, metrics, smoothed)

    # -- events --------------------------------------------------------
    async def _emit_analysis(self, window, result, assessment, metrics, smoothed=None) -> None:
        event = {
            "type": "analysis",
            "session_id": self._session.session_id,
            "window_start": round(window.start_seconds, 3),
            "window_end": round(window.end_seconds, 3),
            "risk_state": assessment.risk_state.value,
            "smoothed_score": round(assessment.smoothed_score, 4),
            "risk_score": round(assessment.risk_score, 4),
            "context_multiplier": round(assessment.context_multiplier, 4),
            "sustained": assessment.sustained,
            "quality_flags": [f.value for f in assessment.quality_flags],
            "metrics": metrics,
            "windows_analyzed": self._smoother.windows_analyzed,
            "windows_skipped": self._session.windows_skipped,
            "windows_abstained": self._session.windows_abstained,
        }
        if result is not None:
            event.update(
                {
                    "synthetic_probability": round(result.synthetic_probability, 4),
                    "detector_score": round(result.detector_score, 4),
                    "confidence": round(result.confidence, 4),
                    "model": result.model,
                    "is_mock": result.is_mock,
                    "inference_ms": round(result.inference_ms, 1),
                    "device": self._detector.device,
                }
            )
            if smoothed is not None:
                event["session_mean"] = round(smoothed.session_mean, 4)
                event["fraction_above_threshold"] = round(
                    smoothed.fraction_above_threshold, 4
                )
        else:
            event.update({"model": self._detector.name, "is_mock": self._detector.is_mock})

        event = json_safe(event)
        self._session.last_result = event
        await self._emit(event)

        if assessment.risk_state is not self._last_state:
            self._last_state = assessment.risk_state
            await self._emit(
                {
                    "type": "recommendation",
                    "risk_state": assessment.risk_state.value,
                    "message": assessment.recommendation,
                }
            )

    # -- completion ----------------------------------------------------
    async def finish(self) -> dict:
        """Drain, score the tail if it is usable, and build the final result."""
        self._session.state = SessionState.ANALYZING

        # Process anything already queued, then the flushed tail.
        while (window := self._take_pending()) is not None:
            async with self._busy:
                await self._process(window)

        tail = self._buffer.flush()
        if tail is not None and tail.speech_seconds >= self._s.min_speech_seconds:
            async with self._busy:
                await self._process(tail)

        # Wait for any in-flight window to land.
        async with self._busy:
            pass

        analyzed = self._smoother.windows_analyzed
        if analyzed == 0:
            final = {
                "type": "final",
                "session_id": self._session.session_id,
                "final_score": 0.0,
                "risk_state": RiskState.INSUFFICIENT_EVIDENCE.value,
                "recommendation": (
                    "No analysable speech was found in the supplied audio. "
                    "No synthetic-voice determination has been made."
                ),
                "windows_analyzed": 0,
                "windows_skipped": self._session.windows_skipped,
                "windows_abstained": self._session.windows_abstained,
                "model": self._detector.name,
                "is_mock": self._detector.is_mock,
                "device": self._detector.device,
                # Report the audio we did process. Omitting this made the UI
                # show "0 s" after a long capture in which every window
                # abstained, which reads as "nothing was captured" rather than
                # "captured, but not scoreable".
                "audio_seconds": round(
                    self._session.audio_seconds(self._s.target_sample_rate), 2
                ),
            }
        else:
            assessment = self._risk.evaluate(
                self._smoother.smoothed,
                self._session.context,
                quality_flags=[],
                # The real k-of-m evidence, not a value derived from the
                # smoothed score — otherwise a single spike that lifts the EWMA
                # would satisfy the rule by itself.
                sustained=self._smoother.sustained,
            )
            final = {
                "type": "final",
                "session_id": self._session.session_id,
                "final_score": round(assessment.risk_score, 4),
                "smoothed_score": round(self._smoother.smoothed, 4),
                "session_mean": round(self._smoother.session_mean, 4),
                "risk_state": assessment.risk_state.value,
                "recommendation": assessment.recommendation,
                "context_multiplier": round(assessment.context_multiplier, 4),
                "windows_analyzed": analyzed,
                "windows_skipped": self._session.windows_skipped,
                "windows_abstained": self._session.windows_abstained,
                "model": self._detector.name,
                "is_mock": self._detector.is_mock,
                "device": self._detector.device,
                "last_inference_ms": round(self._last_inference_ms, 1),
                "audio_seconds": round(
                    self._session.audio_seconds(self._s.target_sample_rate), 2
                ),
            }

        final = json_safe(final)
        self._session.final_result = final
        self._session.state = SessionState.COMPLETED
        return final

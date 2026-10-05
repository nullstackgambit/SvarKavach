"""Sliding-window buffering with speech accounting.

Accumulates normalised mono audio and emits fixed-length analysis windows at a
fixed hop. VAD runs once per frame as audio arrives, never per window, so the
overlapping windows share speech statistics for free.

Memory is bounded: consumed samples ahead of the next window start are dropped,
so a long session costs the same as a short one (plan §35).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .vad import SILERO_FRAME_SAMPLES, VoiceActivityDetector


@dataclass(slots=True)
class AnalysisWindow:
    """One fixed-length window handed to the detector."""

    samples: np.ndarray
    start_sample: int
    end_sample: int
    sample_rate: int
    speech_seconds: float
    speech_ratio: float
    mean_speech_prob: float
    is_final: bool = False
    #: How many trailing samples are real audio. `flush()` left-pads a short
    #: tail with zeros to the model's native length, and the quality gate must
    #: judge the audio rather than the padding. 0 means the whole window.
    valid_samples: int = 0

    @property
    def analysable(self) -> np.ndarray:
        """The real audio in this window, excluding any `flush()` padding."""
        if self.valid_samples <= 0 or self.valid_samples >= self.samples.size:
            return self.samples
        return self.samples[-self.valid_samples:]

    @property
    def start_seconds(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end_seconds(self) -> float:
        return self.end_sample / self.sample_rate


class SlidingWindowBuffer:
    """Turns a stream of samples into overlapping fixed-length windows."""

    def __init__(
        self,
        window_samples: int,
        hop_samples: int,
        sample_rate: int,
        vad: VoiceActivityDetector | None = None,
        vad_threshold: float = 0.5,
    ) -> None:
        if hop_samples <= 0 or window_samples <= 0:
            raise ValueError("window_samples and hop_samples must be positive")
        if hop_samples > window_samples:
            raise ValueError("hop_samples must not exceed window_samples")

        self.window_samples = window_samples
        self.hop_samples = hop_samples
        self.sample_rate = sample_rate
        self._vad = vad
        self._vad_threshold = vad_threshold
        self._frame_samples = vad.frame_samples if vad else SILERO_FRAME_SAMPLES

        self._buf: deque[np.ndarray] = deque()
        self._buf_len = 0
        self._origin = 0          # absolute index of the first buffered sample
        self._next_start = 0      # absolute index where the next window begins
        self._total_samples = 0

        self._probs: deque[float] = deque()
        self._frame_origin = 0    # absolute frame index of _probs[0]
        self._frames_done = 0     # absolute count of frames scored so far

    # -- ingest --------------------------------------------------------
    def push(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        self._buf.append(np.asarray(samples, dtype=np.float32))
        self._buf_len += samples.size
        self._total_samples += samples.size
        self._run_vad()

    def _contiguous(self) -> np.ndarray:
        """Collapse the deque into one array (cheap: it is trimmed aggressively)."""
        if len(self._buf) > 1:
            merged = np.concatenate(self._buf)
            self._buf.clear()
            self._buf.append(merged)
        return self._buf[0] if self._buf else np.zeros(0, dtype=np.float32)

    def _run_vad(self) -> None:
        """Score every newly-complete VAD frame exactly once."""
        buf = self._contiguous()
        end_abs = self._origin + buf.size
        while (self._frames_done + 1) * self._frame_samples <= end_abs:
            start_abs = self._frames_done * self._frame_samples
            if start_abs < self._origin:  # already trimmed away; cannot score
                self._probs.append(0.0)
                self._frames_done += 1
                continue
            lo = start_abs - self._origin
            frame = buf[lo: lo + self._frame_samples]
            prob = self._vad.speech_probability(frame) if self._vad else 1.0
            self._probs.append(float(prob))
            self._frames_done += 1

    # -- speech accounting ---------------------------------------------
    def _speech_stats(self, start_abs: int, end_abs: int) -> tuple[float, float, float]:
        """Speech seconds, speech ratio, and mean probability inside a window."""
        if self._vad is None:
            span = (end_abs - start_abs) / self.sample_rate
            return span, 1.0, 1.0
        first = max(start_abs // self._frame_samples, self._frame_origin)
        last = end_abs // self._frame_samples
        probs = [
            self._probs[i - self._frame_origin]
            for i in range(first, last)
            if 0 <= i - self._frame_origin < len(self._probs)
        ]
        if not probs:
            return 0.0, 0.0, 0.0
        arr = np.asarray(probs, dtype=np.float32)
        voiced = int(np.count_nonzero(arr >= self._vad_threshold))
        return (
            voiced * self._frame_samples / self.sample_rate,
            voiced / arr.size,
            float(arr.mean()),
        )

    # -- emit ----------------------------------------------------------
    def pop_windows(self) -> Iterator[AnalysisWindow]:
        """Yield every window that has become complete since the last call."""
        while self._origin + self._buf_len >= self._next_start + self.window_samples:
            buf = self._contiguous()
            lo = self._next_start - self._origin
            hi = lo + self.window_samples
            start_abs, end_abs = self._next_start, self._next_start + self.window_samples
            speech_s, ratio, mean_p = self._speech_stats(start_abs, end_abs)
            yield AnalysisWindow(
                samples=buf[lo:hi].copy(),
                start_sample=start_abs,
                end_sample=end_abs,
                sample_rate=self.sample_rate,
                speech_seconds=speech_s,
                speech_ratio=ratio,
                mean_speech_prob=mean_p,
            )
            self._next_start += self.hop_samples
            self._trim()

    def _trim(self) -> None:
        """Drop samples and frame scores that no future window can reference."""
        keep_from = self._next_start
        drop = keep_from - self._origin
        if drop <= 0:
            return
        buf = self._contiguous()
        self._buf.clear()
        remaining = buf[drop:] if drop < buf.size else np.zeros(0, dtype=np.float32)
        if remaining.size:
            self._buf.append(remaining)
        self._buf_len = remaining.size
        self._origin = keep_from

        first_frame = keep_from // self._frame_samples
        while self._frame_origin < first_frame and self._probs:
            self._probs.popleft()
            self._frame_origin += 1

    def flush(self) -> AnalysisWindow | None:
        """Emit a final padded window from the tail, if any audio remains.

        The tail is right-aligned into a full-length window and zero-padded on
        the left, so the detector always receives its native input length.
        """
        available = self._origin + self._buf_len - self._next_start
        if available <= 0:
            return None
        buf = self._contiguous()
        lo = max(self._next_start - self._origin, 0)
        tail = buf[lo:]
        if tail.size == 0:
            return None
        start_abs = self._next_start
        end_abs = start_abs + tail.size
        speech_s, ratio, mean_p = self._speech_stats(start_abs, end_abs)
        padded = np.zeros(self.window_samples, dtype=np.float32)
        take = min(tail.size, self.window_samples)
        padded[-take:] = tail[-take:]
        self._next_start = end_abs
        return AnalysisWindow(
            samples=padded,
            start_sample=start_abs,
            end_sample=end_abs,
            sample_rate=self.sample_rate,
            speech_seconds=speech_s,
            speech_ratio=ratio,
            mean_speech_prob=mean_p,
            is_final=True,
            valid_samples=take,
        )

    # -- introspection --------------------------------------------------
    @property
    def total_samples(self) -> int:
        return self._total_samples

    @property
    def buffered_samples(self) -> int:
        return self._buf_len

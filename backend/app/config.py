"""Typed application configuration.

Every tunable in SvarKavach lives here and is environment-driven. No threshold,
window size, or model path is hardcoded anywhere else in the codebase.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

DetectorProviderName = Literal["mock", "xlsr_aasist_onnx", "xlsr_aasist_torch"]
DeviceName = Literal["auto", "cpu", "cuda"]


class Settings(BaseSettings):
    """Validated runtime settings, loaded from the environment and `.env`."""

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # ---- application -------------------------------------------------
    app_name: str = "SvarKavach"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    log_json: bool = False

    # ---- detector ----------------------------------------------------
    detector_provider: DetectorProviderName = "mock"
    model_device: DeviceName = "auto"
    model_path: str = ""
    # Number of intra-op threads for ONNX Runtime / torch. 0 = library default.
    inference_threads: int = 0
    # Logit threshold on the bona fide logit, from the upstream model card.
    # Used only to report where the operating point sits; scoring uses the
    # calibrated probability, not this value.
    detector_logit_threshold: float = -1.0625009
    # Temperature for the softmax that turns raw logits into a probability.
    calibration_temperature: float = 1.0
    # Allow falling back to the mock detector if the real one fails to load.
    # Deliberately defaults to False: a requested real detector must fail loudly.
    allow_detector_fallback: bool = False

    # ---- mock detector (demo only) -----------------------------------
    # Centre of the simulated score range and how far content may move it.
    mock_base_score: float = 0.5
    mock_spread: float = 0.35
    # Simulated inference cost, so the demo exercises the same async path
    # the real model does. Reported latency is always measured, never faked.
    mock_latency_ms: float = 90.0

    # ---- audio / preprocessing ---------------------------------------
    target_sample_rate: int = 16000
    window_seconds: float = 4.0375  # 64,600 samples — the model's native input
    hop_seconds: float = 0.5
    vad_enabled: bool = True
    vad_model_path: str = ""
    vad_threshold: float = 0.5
    # Minimum speech inside a window before it is worth scoring.
    min_speech_seconds: float = 1.5

    # ---- quality gates -----------------------------------------------
    min_rms_dbfs: float = -55.0        # below this, the window is too quiet
    max_clipping_ratio: float = 0.02   # fraction of samples at full scale

    # ---- scoring -----------------------------------------------------
    smoothing_alpha: float = 0.35
    # Sustained-evidence rule: k of the last m windows must exceed the
    # threshold before the state is allowed to escalate past WATCH.
    sustained_k: int = 3
    sustained_m: int = 5

    # ---- risk thresholds ---------------------------------------------
    watch_threshold: float = 0.40
    verify_threshold: float = 0.65
    escalate_threshold: float = 0.85

    # ---- context weighting -------------------------------------------
    high_value_transaction: float = 1_000_000.0
    high_value_multiplier: float = 0.10
    unknown_contact_multiplier: float = 0.05

    # ---- limits / safety ---------------------------------------------
    max_session_seconds: int = 900
    max_chunk_bytes: int = 262_144
    max_concurrent_sessions: int = 16

    # ---- system audio capture (loopback) ------------------------------
    # Opt-in. When enabled, a client that can reach this server can start
    # recording the host machine's audio output. Keep it false unless you are
    # deliberately demonstrating live capture, and see README § Privacy.
    system_audio_enabled: bool = False
    # Empty = the default loopback device. Otherwise the backend device id.
    system_audio_device: str = ""
    # Capture block size. 100 ms keeps the WebSocket responsive without
    # producing an excessive number of tiny reads.
    system_audio_block_ms: int = 100
    # Bounded backlog. Beyond this the oldest block is dropped rather than
    # letting a slow consumer grow memory without limit.
    system_audio_queue_blocks: int = 50

    # ---- privacy -----------------------------------------------------
    # Opt-in only. When False (default) no audio ever touches the disk.
    persist_audio_debug: bool = False

    # ------------------------------------------------------------------
    @field_validator("smoothing_alpha")
    @classmethod
    def _check_alpha(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        return v

    @field_validator("target_sample_rate")
    @classmethod
    def _check_sr(cls, v: int) -> int:
        if v not in (8000, 16000):
            raise ValueError("target_sample_rate must be 8000 or 16000")
        return v

    @field_validator("system_audio_block_ms")
    @classmethod
    def _check_block_ms(cls, v: int) -> int:
        if not 10 <= v <= 1000:
            raise ValueError("system_audio_block_ms must be between 10 and 1000")
        return v

    @field_validator("system_audio_queue_blocks")
    @classmethod
    def _check_queue_blocks(cls, v: int) -> int:
        if not 1 <= v <= 1000:
            raise ValueError("system_audio_queue_blocks must be between 1 and 1000")
        return v

    @model_validator(mode="after")
    def _check_ordering(self) -> "Settings":
        t = (self.watch_threshold, self.verify_threshold, self.escalate_threshold)
        if not all(0.0 <= x <= 1.0 for x in t):
            raise ValueError("risk thresholds must lie in [0, 1]")
        if not t[0] < t[1] < t[2]:
            raise ValueError(
                "risk thresholds must be strictly increasing: "
                "watch < verify < escalate"
            )
        if self.hop_seconds <= 0 or self.window_seconds <= 0:
            raise ValueError("window_seconds and hop_seconds must be positive")
        if self.hop_seconds > self.window_seconds:
            raise ValueError("hop_seconds must not exceed window_seconds")
        if self.sustained_k > self.sustained_m:
            raise ValueError("sustained_k must not exceed sustained_m")
        if self.min_speech_seconds > self.window_seconds:
            raise ValueError("min_speech_seconds must not exceed window_seconds")
        return self

    # ---- derived -----------------------------------------------------
    @property
    def window_samples(self) -> int:
        return int(round(self.window_seconds * self.target_sample_rate))

    @property
    def hop_samples(self) -> int:
        return int(round(self.hop_seconds * self.target_sample_rate))

    @property
    def uses_real_detector(self) -> bool:
        return self.detector_provider != "mock"

    def public_config(self) -> dict:
        """Non-secret settings, safe to expose over HTTP (see plan §18)."""
        return {
            "app_name": self.app_name,
            "detector_provider": self.detector_provider,
            "model_device": self.model_device,
            "target_sample_rate": self.target_sample_rate,
            "window_seconds": self.window_seconds,
            "hop_seconds": self.hop_seconds,
            "window_samples": self.window_samples,
            "hop_samples": self.hop_samples,
            "vad_enabled": self.vad_enabled,
            "smoothing_alpha": self.smoothing_alpha,
            "sustained_k": self.sustained_k,
            "sustained_m": self.sustained_m,
            "thresholds": {
                "watch": self.watch_threshold,
                "verify": self.verify_threshold,
                "escalate": self.escalate_threshold,
            },
            "min_speech_seconds": self.min_speech_seconds,
            "max_session_seconds": self.max_session_seconds,
            "max_chunk_bytes": self.max_chunk_bytes,
            "system_audio_enabled": self.system_audio_enabled,
        }

    @property
    def binds_publicly(self) -> bool:
        """True when the server listens beyond loopback.

        Relevant because system-audio capture is a privileged capability and
        this application has no authentication.
        """
        return self.host not in ("127.0.0.1", "localhost", "::1")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

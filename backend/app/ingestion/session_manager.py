"""In-memory session registry and per-session metrics (plan §23).

No database. Sessions live only as long as the WebSocket connection, and
nothing here retains audio.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..models.messages import SessionContext, SessionState
from ..utils.logging import get_logger

log = get_logger(__name__)


class SessionLimitError(RuntimeError):
    """Raised when the server is already at its concurrent-session cap."""


class SessionExistsError(RuntimeError):
    """Raised when a session id is already in use by a live connection."""


@dataclass(slots=True)
class Session:
    session_id: str
    source: str
    context: SessionContext
    detector_name: str
    state: SessionState = SessionState.CREATED
    started_at: float = field(default_factory=time.time)
    last_audio_at: float | None = None
    bytes_received: int = 0
    samples_received: int = 0
    windows_processed: int = 0
    windows_skipped: int = 0
    windows_abstained: int = 0
    last_result: dict | None = None
    final_result: dict | None = None
    error: str | None = None

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.started_at

    def audio_seconds(self, sample_rate: int) -> float:
        return self.samples_received / sample_rate if sample_rate else 0.0

    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "source": self.source,
            "state": self.state.value,
            "detector": self.detector_name,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "bytes_received": self.bytes_received,
            "samples_received": self.samples_received,
            "windows_processed": self.windows_processed,
            "windows_skipped": self.windows_skipped,
            "windows_abstained": self.windows_abstained,
            "context": self.context.model_dump(),
        }


class SessionManager:
    """Tracks live sessions and enforces the concurrency cap."""

    def __init__(self, max_concurrent: int = 16) -> None:
        self._sessions: dict[str, Session] = {}
        self._max = max_concurrent

    def create(
        self,
        session_id: str,
        source: str,
        context: SessionContext,
        detector_name: str,
    ) -> Session:
        if session_id in self._sessions:
            # "Newest connection wins" cannot work here: session_id is
            # client-supplied, and the first connection's teardown removes the
            # id unconditionally — evicting the *second* connection's entry
            # while its socket is still live. That silently under-counts active
            # sessions and lets a client walk past max_concurrent entirely.
            raise SessionExistsError(
                f"session id {session_id!r} is already in use by a live connection"
            )
        if len(self._sessions) >= self._max:
            raise SessionLimitError(
                f"server at capacity ({self._max} concurrent sessions)"
            )
        session = Session(
            session_id=session_id,
            source=source,
            context=context,
            detector_name=detector_name,
            state=SessionState.READY,
        )
        self._sessions[session_id] = session
        log.info("session created", extra={"session_id": session_id, "source": source})
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def remove(self, session: Session) -> None:
        """Remove a session, but only if it is still the registered one.

        Identity-checked so a stale handler tearing down cannot evict a newer
        session that happens to share its id.
        """
        if self._sessions.get(session.session_id) is session:
            del self._sessions[session.session_id]
            log.info("session removed", extra={"session_id": session.session_id})

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    def snapshot(self) -> list[dict]:
        return [s.as_dict() for s in self._sessions.values()]

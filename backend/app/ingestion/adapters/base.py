"""Audio source adapters.

An adapter turns some external audio source into the one internal
representation the pipeline understands: a stream of `AudioChunk`. Nothing
downstream of this boundary knows or cares where the audio came from — that is
what makes the demo webpage swappable for Zoom, SIP, WebRTC or a contact-centre
media feed later (plan §3, §41).

The browser upload path is one adapter (implemented inline in the WebSocket
handler); `SystemAudioAdapter` is the second.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from ...models.messages import AudioChunk


class CaptureError(RuntimeError):
    """Raised when a source cannot be opened or fails mid-capture."""


class AudioSourceAdapter(ABC):
    """A source of normalised-on-arrival audio chunks."""

    #: Identifier reported to the client and written to logs.
    name: str = "base"

    @abstractmethod
    def stream(self) -> AsyncIterator[AudioChunk]:
        """Yield chunks until the source is exhausted or `close()` is called.

        Implementations must not block the event loop. Anything that reads from
        a blocking device belongs on a worker thread.
        """
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """Stop capture and release the device. Must be idempotent."""
        raise NotImplementedError

    def describe(self) -> dict:
        return {"source": self.name}

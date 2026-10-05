"""Capture device enumeration for live system-audio analysis.

Only opaque, server-assigned device ids reach the browser; the underlying
backend device name never leaves the server (plan §35).
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..ingestion.adapters.system_audio import list_capture_devices

router = APIRouter(tags=["audio"])


@router.get("/audio/devices")
async def audio_devices(request: Request) -> dict:
    state = request.app.state
    settings = state.settings

    if not settings.system_audio_enabled:
        return {
            "enabled": False,
            "backend": None,
            "reason": (
                "System audio capture is disabled. Set SYSTEM_AUDIO_ENABLED=true "
                "in .env to enable it."
            ),
            "devices": [],
        }

    backend = getattr(state, "capture_backend", None)
    if backend is None:
        return {
            "enabled": False,
            "backend": None,
            "reason": getattr(state, "capture_unavailable_reason", "audio backend unavailable"),
            "devices": [],
        }

    try:
        devices = list_capture_devices(backend, loopback_only=True)
    except Exception as exc:
        return {
            "enabled": False,
            "backend": backend.name,
            "reason": f"could not enumerate capture devices: {exc}",
            "devices": [],
        }

    return {
        "enabled": True,
        "backend": backend.name,
        "reason": None,
        "devices": [d.public() for d in devices],
    }

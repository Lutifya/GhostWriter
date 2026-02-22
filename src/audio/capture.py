"""
Cross-platform system-audio (loopback) capture.

Platform dispatch:
  macOS   → BlackHole virtual device (sounddevice, input device matching "BlackHole")
  Windows → WASAPI loopback (sounddevice, wasapi_loopback=True)
  Linux   → PulseAudio/PipeWire monitor source (sounddevice, device matching ".monitor")

Usage
-----
    from src.audio.capture import create_capture

    cap = create_capture(samplerate=16000, blocksize=1024)
    cap.start(callback)   # callback(pcm_block: np.ndarray) called in audio thread
    ...
    cap.stop()
"""

from __future__ import annotations

import platform
import queue
import threading
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import sounddevice as sd

AudioCallback = Callable[[np.ndarray], None]


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class AudioCapture(ABC):
    """Abstract loopback audio source."""

    def __init__(self, samplerate: int = 16_000, blocksize: int = 1024, channels: int = 1) -> None:
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels
        self._stream: sd.InputStream | None = None
        self._callback: AudioCallback | None = None

    def start(self, callback: AudioCallback) -> None:
        """Open the audio stream and start calling *callback* with PCM blocks."""
        self._callback = callback
        self._stream = self._build_stream()
        self._stream.start()

    def stop(self) -> None:
        """Stop and close the audio stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _sd_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            import sys
            print(f"[audio] sounddevice status: {status}", file=sys.stderr)
        if self._callback is not None:
            # Always pass a mono float32 copy
            mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
            self._callback(mono)

    @abstractmethod
    def _build_stream(self) -> sd.InputStream:
        """Return a configured but not-yet-started InputStream."""

    @staticmethod
    def list_devices() -> None:
        """Print available audio devices (helper for debugging)."""
        print(sd.query_devices())


# ─────────────────────────────────────────────────────────────────────────────
# macOS – BlackHole
# ─────────────────────────────────────────────────────────────────────────────

class BlackHoleCapture(AudioCapture):
    """Capture via BlackHole virtual audio driver (macOS)."""

    DEVICE_KEYWORD = "BlackHole"

    def __init__(self, *, device_name: str | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._device_name = device_name or self.DEVICE_KEYWORD

    def _find_device_index(self) -> int:
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if self._device_name.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
                return idx
        available = [d["name"] for d in devices if d["max_input_channels"] > 0]
        raise RuntimeError(
            f"BlackHole device '{self._device_name}' not found.\n"
            f"Available input devices: {available}\n"
            "Install BlackHole from https://existential.audio/blackhole/"
        )

    def _build_stream(self) -> sd.InputStream:
        device_idx = self._find_device_index()
        return sd.InputStream(
            device=device_idx,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            channels=self.channels,
            dtype="float32",
            callback=self._sd_callback,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Windows – WASAPI loopback
# ─────────────────────────────────────────────────────────────────────────────

class WASAPILoopbackCapture(AudioCapture):
    """Capture system audio via WASAPI loopback (Windows)."""

    def _find_loopback_device(self) -> int:
        """Return the device index of the default loopback device."""
        devices = sd.query_devices()
        # sounddevice exposes WASAPI loopback devices whose names end with " [Loopback]"
        for idx, dev in enumerate(devices):
            name: str = dev["name"]
            if "[Loopback]" in name and dev["max_input_channels"] > 0:
                return idx
        # Fallback: try the default output device as loopback
        try:
            default_out = sd.query_devices(kind="output")
            for idx, dev in enumerate(devices):
                if dev["name"] == default_out["name"] and dev["max_input_channels"] > 0:
                    return idx
        except Exception:
            pass
        raise RuntimeError(
            "No WASAPI loopback device found. "
            "Ensure your sounddevice build supports WASAPI (Windows only)."
        )

    def _build_stream(self) -> sd.InputStream:
        device_idx = self._find_loopback_device()
        # extra_settings for WASAPI loopback flag
        try:
            wasapi_settings = sd.WasapiSettings(loopback=True)
            return sd.InputStream(
                device=device_idx,
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                channels=self.channels,
                dtype="float32",
                extra_settings=wasapi_settings,
                callback=self._sd_callback,
            )
        except AttributeError:
            # Older sounddevice without WasapiSettings – fall back to plain stream
            return sd.InputStream(
                device=device_idx,
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                channels=self.channels,
                dtype="float32",
                callback=self._sd_callback,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Linux – PulseAudio/PipeWire monitor
# ─────────────────────────────────────────────────────────────────────────────

class PulseAudioMonitorCapture(AudioCapture):
    """Capture system audio via a PulseAudio/PipeWire monitor source (Linux)."""

    MONITOR_KEYWORD = ".monitor"

    def _find_monitor_device(self) -> int:
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            name: str = dev["name"]
            if self.MONITOR_KEYWORD in name and dev["max_input_channels"] > 0:
                return idx
        available = [d["name"] for d in devices if d["max_input_channels"] > 0]
        raise RuntimeError(
            f"No PulseAudio/PipeWire monitor source found (keyword='{self.MONITOR_KEYWORD}').\n"
            f"Available input devices: {available}\n"
            "Try: pactl list sources | grep monitor"
        )

    def _build_stream(self) -> sd.InputStream:
        device_idx = self._find_monitor_device()
        return sd.InputStream(
            device=device_idx,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            channels=self.channels,
            dtype="float32",
            callback=self._sd_callback,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_capture(
    samplerate: int = 16_000,
    blocksize: int = 1024,
    channels: int = 1,
    **kwargs: object,
) -> AudioCapture:
    """
    Instantiate the correct AudioCapture implementation for the current OS.

    Parameters
    ----------
    samplerate : int
        Sample rate in Hz (default 16 kHz, Whisper's native rate).
    blocksize : int
        Number of frames per callback invocation.
    channels : int
        Number of audio channels (1 = mono).
    **kwargs
        Forwarded to the platform-specific constructor.
    """
    os_name = platform.system()
    if os_name == "Darwin":
        return BlackHoleCapture(samplerate=samplerate, blocksize=blocksize, channels=channels, **kwargs)
    elif os_name == "Windows":
        return WASAPILoopbackCapture(samplerate=samplerate, blocksize=blocksize, channels=channels, **kwargs)
    elif os_name == "Linux":
        return PulseAudioMonitorCapture(samplerate=samplerate, blocksize=blocksize, channels=channels, **kwargs)
    else:
        raise NotImplementedError(f"Unsupported operating system: {os_name}")

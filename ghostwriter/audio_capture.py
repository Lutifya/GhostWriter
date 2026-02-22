"""System audio capture using sounddevice.

On Linux the daemon auto-detects a PulseAudio/PipeWire *monitor* device so
that the system's speaker output is captured rather than the microphone.
On Windows, WASAPI loopback devices appear as regular input devices; pass
the appropriate device index via ``--device``.
"""

import queue
from typing import Optional

import numpy as np
import sounddevice as sd


class AudioCapture:
    """Captures audio from a sounddevice input stream.

    The captured audio is made available through :py:meth:`read` which
    returns one block of ``float32`` samples at a time.

    Args:
        device: sounddevice device index (or ``None`` for the default input).
        sample_rate: Sample rate in Hz.  16 000 Hz matches Whisper's
            expected input rate.
        channels: Number of input channels.  Whisper expects mono (``1``).
        chunk_duration: Duration in seconds of each audio block pushed to
            the internal queue.
    """

    DEFAULT_SAMPLE_RATE: int = 16_000
    DEFAULT_CHANNELS: int = 1
    DEFAULT_CHUNK_DURATION: float = 0.5

    def __init__(
        self,
        device: Optional[int] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        chunk_duration: float = DEFAULT_CHUNK_DURATION,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.blocksize: int = int(sample_rate * chunk_duration)

        self._queue: queue.Queue = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Stream lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open and start the audio input stream."""
        self._running = True
        self._stream = sd.InputStream(
            device=self.device,
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            callback=self._callback,
            dtype="float32",
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop and close the audio input stream."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def read(self, timeout: float = 1.0) -> np.ndarray:
        """Return the next audio chunk from the internal queue.

        Args:
            timeout: Seconds to wait for the next chunk before raising
                :py:exc:`queue.Empty`.

        Returns:
            A 1-D ``float32`` NumPy array of ``blocksize`` samples.
        """
        return self._queue.get(timeout=timeout)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """``True`` while the audio stream is active."""
        return self._running

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def list_devices():
        """Return all audio devices visible to sounddevice."""
        return sd.query_devices()

    @staticmethod
    def find_monitor_device() -> Optional[int]:
        """Return the index of a system-audio monitor / loopback device.

        On Linux (PulseAudio / PipeWire) output devices expose a companion
        *monitor* input.  This method returns the first such device index.
        Returns ``None`` when none is found so the caller can fall back to
        the system default input.
        """
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            name: str = device["name"].lower()
            if "monitor" in name and device["max_input_channels"] > 0:
                return i
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,  # noqa: ARG002
        time,  # noqa: ARG002
        status,
    ) -> None:
        if status:
            print(f"[AudioCapture] {status}")
        if self._running:
            # Always store mono float32 samples
            mono = indata[:, 0] if indata.ndim > 1 else indata
            self._queue.put(mono.copy())

"""
Thread-safe circular audio buffer.

Continuously receives PCM blocks from the audio callback thread and allows
the VAD/pipeline thread to read contiguous windows of samples.
"""

from __future__ import annotations

import threading

import numpy as np


class CircularAudioBuffer:
    """
    Lock-protected ring buffer for float32 mono PCM samples.

    Parameters
    ----------
    capacity : int
        Maximum number of samples the buffer can hold.
        Oldest samples are overwritten when full (ring behaviour).
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._write_pos: int = 0
        self._size: int = 0          # current number of valid samples
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write

    def write(self, data: np.ndarray) -> None:
        """Append *data* to the buffer (overwrites oldest if full)."""
        data = np.asarray(data, dtype=np.float32).ravel()
        n = len(data)
        if n == 0:
            return

        with self._lock:
            if n >= self._capacity:
                # Data is larger than the whole buffer; keep only the tail
                self._buf[:] = data[-self._capacity:]
                self._write_pos = 0
                self._size = self._capacity
                return

            end = self._write_pos + n
            if end <= self._capacity:
                self._buf[self._write_pos:end] = data
            else:
                first = self._capacity - self._write_pos
                self._buf[self._write_pos:] = data[:first]
                self._buf[: n - first] = data[first:]

            self._write_pos = end % self._capacity
            self._size = min(self._size + n, self._capacity)

    # ------------------------------------------------------------------
    # Read

    def read_last(self, n_samples: int) -> np.ndarray:
        """
        Return the *n_samples* most recently written samples (copy).
        If fewer samples are available, returns all available samples.
        """
        with self._lock:
            available = min(n_samples, self._size)
            if available == 0:
                return np.zeros(0, dtype=np.float32)

            end = self._write_pos
            start = (end - available) % self._capacity
            if start < end:
                return self._buf[start:end].copy()
            else:
                return np.concatenate([self._buf[start:], self._buf[:end]])

    def drain(self) -> np.ndarray:
        """Return all buffered samples and reset the buffer."""
        with self._lock:
            if self._size == 0:
                return np.zeros(0, dtype=np.float32)

            end = self._write_pos
            start = (end - self._size) % self._capacity
            if start < end:
                result = self._buf[start:end].copy()
            else:
                result = np.concatenate([self._buf[start:], self._buf[:end]])
            self._write_pos = 0
            self._size = 0
            self._buf[:] = 0.0
            return result

    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Current number of samples in the buffer."""
        with self._lock:
            return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

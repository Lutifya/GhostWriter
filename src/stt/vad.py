"""
Voice Activity Detection (VAD) filter using silero-vad.

Wraps the silero-vad model to split a raw PCM stream into speech segments,
discarding silence before passing audio to the Whisper engine.

Silero-vad operates at 16 kHz or 8 kHz. We always use 16 kHz to match Whisper.

Reference: https://github.com/snakers4/silero-vad
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Generator

import numpy as np

try:
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps
except ImportError as exc:
    raise ImportError(
        "silero-vad and torch are required: pip install silero-vad torch"
    ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpeechChunk:
    audio: np.ndarray   # float32 mono PCM at 16 kHz
    start_sample: int   # index in the original stream
    end_sample: int


# ─────────────────────────────────────────────────────────────────────────────
# VADFilter
# ─────────────────────────────────────────────────────────────────────────────

class VADFilter:
    """
    Runs silero-vad on an audio buffer and yields SpeechChunk objects.

    Parameters
    ----------
    samplerate : int
        Must be 16 000 Hz.
    threshold : float
        VAD probability threshold (0.0–1.0). Higher = more conservative.
    min_speech_duration_ms : int
        Discard speech segments shorter than this (ms).
    min_silence_duration_ms : int
        Minimum silence between two speech segments before splitting (ms).
    padding_ms : int
        Milliseconds of audio to add before and after each detected segment.
    """

    SUPPORTED_RATES = {8_000, 16_000}

    def __init__(
        self,
        samplerate: int = 16_000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 300,
        padding_ms: int = 30,
    ) -> None:
        if samplerate not in self.SUPPORTED_RATES:
            raise ValueError(f"silero-vad only supports {self.SUPPORTED_RATES} Hz, got {samplerate}")

        self.samplerate = samplerate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.padding_ms = padding_ms

        print("[VADFilter] Loading silero-vad model …")
        self._model = load_silero_vad()
        print("[VADFilter] Model loaded.")

    # ------------------------------------------------------------------

    def get_speech_chunks(self, audio: np.ndarray) -> list[SpeechChunk]:
        """
        Run VAD on *audio* and return a list of SpeechChunk.

        Parameters
        ----------
        audio : np.ndarray
            float32 mono PCM at self.samplerate.
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        tensor = torch.from_numpy(audio)

        timestamps = get_speech_timestamps(
            tensor,
            self._model,
            threshold=self.threshold,
            sampling_rate=self.samplerate,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.padding_ms,
            return_seconds=False,
        )

        chunks: list[SpeechChunk] = []
        for ts in timestamps:
            start: int = int(ts["start"])
            end: int = int(ts["end"])
            chunks.append(
                SpeechChunk(
                    audio=audio[start:end].copy(),
                    start_sample=start,
                    end_sample=end,
                )
            )
        return chunks

    # ------------------------------------------------------------------

    def is_speech(self, block: np.ndarray) -> float:
        """
        Return the VAD probability for a single audio block.
        Useful for real-time gating before buffering.
        """
        if block.dtype != np.float32:
            block = block.astype(np.float32)
        tensor = torch.from_numpy(block)
        with torch.no_grad():
            prob: float = self._model(tensor, self.samplerate).item()
        return prob

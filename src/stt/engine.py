"""
Distil-Whisper inference engine backed by faster-whisper.

Supported models (distil-* family recommended):
  - "distil-large-v3"   – multilingual, best quality  (~1.5 GB)
  - "distil-medium.en"  – English-only, fastest        (~750 MB)
  - any HuggingFace model id accepted by faster-whisper

Device selection (automatic unless overridden):
  - CUDA available  → "cuda"
  - macOS ARM64     → "cpu"  (CTranslate2 NEON-optimised; Metal not yet stable)
  - Fallback        → "cpu"  (AVX2/AVX512 via CTranslate2)
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError as exc:
    raise ImportError(
        "faster-whisper is required: pip install faster-whisper"
    ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

DeviceType = Literal["cuda", "cpu", "auto"]
ComputeType = Literal["float16", "float32", "int8", "int8_float16", "auto"]


@dataclass
class TranscriptSegment:
    text: str
    start: float          # seconds from audio start
    end: float
    language: str | None  # detected language (None if not detected)
    latency_ms: float     # wall-clock ms from inference start to result


# ─────────────────────────────────────────────────────────────────────────────
# Device helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_device() -> tuple[str, str]:
    """Return (device, compute_type) for the current hardware."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass

    # macOS Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "cpu", "int8"  # CTranslate2 int8 is fastest on ARM NEON

    return "cpu", "int8"


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class WhisperEngine:
    """
    Wrapper around a faster-whisper WhisperModel for real-time transcription.

    Parameters
    ----------
    model_id : str
        HuggingFace model id or local path.
    device : DeviceType
        "auto" selects cuda > cpu automatically.
    compute_type : ComputeType
        "auto" picks the best type for the selected device.
    language : str | None
        Force a language code (e.g. "en") or None for auto-detection.
    beam_size : int
        Beam size for decoding (1 = greedy, fastest).
    """

    def __init__(
        self,
        model_id: str = "distil-large-v3",
        device: DeviceType = "auto",
        compute_type: ComputeType = "auto",
        language: str | None = None,
        beam_size: int = 1,
    ) -> None:
        self.model_id = model_id
        self.language = language
        self.beam_size = beam_size

        resolved_device, resolved_compute = _detect_device()
        self._device = resolved_device if device == "auto" else device
        self._compute_type = resolved_compute if compute_type == "auto" else compute_type

        print(
            f"[WhisperEngine] Loading '{model_id}' on {self._device} ({self._compute_type}) …"
        )
        t0 = time.monotonic()
        self._model = WhisperModel(
            model_id,
            device=self._device,
            compute_type=self._compute_type,
        )
        elapsed = (time.monotonic() - t0) * 1000
        print(f"[WhisperEngine] Model loaded in {elapsed:.0f} ms")

    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        samplerate: int = 16_000,
    ) -> list[TranscriptSegment]:
        """
        Transcribe a NumPy PCM array and return a list of TranscriptSegment.

        Parameters
        ----------
        audio : np.ndarray
            float32 mono PCM, any length.
        samplerate : int
            Must be 16 000 Hz (Whisper native rate).
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.ndim != 1:
            audio = audio[:, 0]

        t_start = time.monotonic()

        segments_gen, info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,  # VAD handled separately by VADFilter
            without_timestamps=False,
        )

        results: list[TranscriptSegment] = []
        for seg in segments_gen:
            latency_ms = (time.monotonic() - t_start) * 1000
            results.append(
                TranscriptSegment(
                    text=seg.text.strip(),
                    start=seg.start,
                    end=seg.end,
                    language=info.language,
                    latency_ms=latency_ms,
                )
            )

        return results

    # ------------------------------------------------------------------

    @property
    def device(self) -> str:
        return self._device

    @property
    def compute_type(self) -> str:
        return self._compute_type

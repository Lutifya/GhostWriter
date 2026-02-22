"""Real-time speech transcription powered by faster-whisper.

Audio chunks are fed via :py:meth:`Transcriber.feed_audio`.  Internally the
transcriber accumulates audio until ``buffer_duration`` seconds have been
collected, then runs a Whisper inference pass and fires the
``on_transcription`` callback with the resulting text.
"""

import queue
import threading
from typing import Callable, Optional

import numpy as np
from faster_whisper import WhisperModel


class Transcriber:
    """Converts audio chunks to text using a local Whisper model.

    Args:
        model_size: faster-whisper model variant (e.g. ``"base"``).
        device: Compute device – ``"cpu"`` or ``"cuda"``.
        compute_type: Quantisation type forwarded to CTranslate2
            (e.g. ``"int8"`` for CPU).
        language: Optional BCP-47 language code.  ``None`` enables Whisper's
            automatic language detection.
        on_transcription: Callback invoked with the recognised text string
            each time a buffer is transcribed.  Called from a background
            thread.
        buffer_duration: Seconds of audio to accumulate before each
            inference call.
        sample_rate: Sample rate of incoming audio (must match the audio
            capture rate).
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None,
        on_transcription: Optional[Callable[[str], None]] = None,
        buffer_duration: float = 3.0,
        sample_rate: int = 16_000,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.on_transcription = on_transcription
        self.buffer_duration = buffer_duration
        self.sample_rate = sample_rate

        self._model: Optional[WhisperModel] = None
        self._audio_queue: queue.Queue = queue.Queue()
        self._buffer: np.ndarray = np.array([], dtype=np.float32)
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the Whisper model into memory."""
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    def start(self) -> None:
        """Load the model (if needed) and start the transcription thread."""
        if self._model is None:
            self.load_model()
        self._running = True
        self._thread = threading.Thread(
            target=self._transcription_loop, daemon=True, name="ghostwriter-transcriber"
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the transcription thread to stop and wait for it."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def feed_audio(self, chunk: np.ndarray) -> None:
        """Enqueue a mono float32 audio chunk for transcription.

        Args:
            chunk: 1-D ``float32`` NumPy array.
        """
        self._audio_queue.put(chunk)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _transcription_loop(self) -> None:
        """Background loop: accumulate audio and transcribe in batches."""
        min_samples = int(self.sample_rate * self.buffer_duration)

        while self._running:
            try:
                chunk = self._audio_queue.get(timeout=0.5)
                self._buffer = np.concatenate([self._buffer, chunk])

                if len(self._buffer) >= min_samples:
                    audio, self._buffer = self._buffer.copy(), np.array([], dtype=np.float32)
                    self._transcribe(audio)
            except queue.Empty:
                # Drain leftover audio when the queue goes quiet
                if len(self._buffer) > 0:
                    audio, self._buffer = self._buffer.copy(), np.array([], dtype=np.float32)
                    self._transcribe(audio)

    def _transcribe(self, audio: np.ndarray) -> None:
        """Run Whisper on *audio* and dispatch the result."""
        if self._model is None:
            return

        segments, _ = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=1,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        if text and self.on_transcription is not None:
            self.on_transcription(text)

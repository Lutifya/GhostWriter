"""
Real-time transcription pipeline.

Thread model
────────────
  AudioThread  ──[pcm_queue]──▶  VADThread  ──[speech_queue]──▶  InferenceThread
                                                                          │
                                                                    on_transcript callback
                                                                    (called in InferenceThread)

Each stage communicates via bounded queues to prevent unbounded memory growth.
Threads are daemon threads so they die with the main process.

Usage
-----
    from src.pipeline.pipeline import Pipeline

    def on_text(text: str, latency_ms: float) -> None:
        print(f"[{latency_ms:.0f}ms] {text}")

    pipeline = Pipeline(on_transcript=on_text)
    pipeline.start()
    # … keep running …
    pipeline.stop()
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable

import numpy as np

from src.audio.capture import AudioCapture, create_capture
from src.pipeline.buffer import CircularAudioBuffer
from src.stt.engine import WhisperEngine
from src.stt.vad import VADFilter

TranscriptCallback = Callable[[str, float], None]   # (text, latency_ms)

_SENTINEL = object()   # unique poison pill for queue shutdown – never use None


class Pipeline:
    """
    Ties together audio capture, VAD, and Whisper inference.

    Parameters
    ----------
    on_transcript : TranscriptCallback
        Called from the inference thread whenever text is produced.
        Signature: ``callback(text: str, latency_ms: float)``.
    model_id : str
        faster-whisper model id (distil-* recommended).
    language : str | None
        Force language or None for auto-detection.
    samplerate : int
        Audio sample rate (must be 16 000 for Whisper).
    blocksize : int
        Audio callback block size in frames.
    vad_window_s : float
        Seconds of audio accumulated before each VAD pass.
    max_segment_s : float
        Hard-cap: flush audio to Whisper at least every N seconds,
        even if VAD hasn't found a pause.
    pcm_queue_size : int
        Max PCM blocks in the audio→VAD queue before dropping.
    speech_queue_size : int
        Max speech arrays in the VAD→inference queue.
    """

    def __init__(
        self,
        on_transcript: TranscriptCallback,
        model_id: str = "distil-large-v3",
        language: str | None = None,
        samplerate: int = 16_000,
        blocksize: int = 1024,
        vad_window_s: float = 1.0,
        max_segment_s: float = 30.0,
        pcm_queue_size: int = 512,
        speech_queue_size: int = 32,
    ) -> None:
        self._on_transcript = on_transcript
        self._samplerate = samplerate
        self._blocksize = blocksize
        self._vad_window_samples = int(vad_window_s * samplerate)
        self._max_segment_samples = int(max_segment_s * samplerate)

        # Queues
        self._pcm_queue: queue.Queue[np.ndarray | object] = queue.Queue(maxsize=pcm_queue_size)
        self._speech_queue: queue.Queue[np.ndarray | object] = queue.Queue(maxsize=speech_queue_size)

        # Audio capture (OS-specific)
        self._capture: AudioCapture = create_capture(samplerate=samplerate, blocksize=blocksize)

        # Lazy-load heavy models (done in their threads to keep start() fast)
        self._model_id = model_id
        self._language = language

        # Thread handles
        self._vad_thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None
        self._running = False

    # ──────────────────────────────────────────────────────────────────
    # Public API

    def start(self) -> None:
        """Start all pipeline threads and the audio stream."""
        if self._running:
            return
        self._running = True

        self._vad_thread = threading.Thread(
            target=self._vad_worker, name="vad-thread", daemon=True
        )
        self._inference_thread = threading.Thread(
            target=self._inference_worker, name="inference-thread", daemon=True
        )

        self._vad_thread.start()
        self._inference_thread.start()
        self._capture.start(self._audio_callback)

    def stop(self, timeout: float = 5.0) -> None:
        """Gracefully stop the pipeline."""
        if not self._running:
            return
        self._running = False
        self._capture.stop()

        # Send poison pills
        try:
            self._pcm_queue.put(_SENTINEL, timeout=1.0)
        except queue.Full:
            pass
        try:
            self._speech_queue.put(_SENTINEL, timeout=1.0)
        except queue.Full:
            pass

        if self._vad_thread:
            self._vad_thread.join(timeout=timeout)
        if self._inference_thread:
            self._inference_thread.join(timeout=timeout)

    # ──────────────────────────────────────────────────────────────────
    # Stage 1 – audio callback (runs in sounddevice thread)

    def _audio_callback(self, block: np.ndarray) -> None:
        try:
            self._pcm_queue.put_nowait(block.copy())
        except queue.Full:
            pass  # drop block rather than block the audio thread

    # ──────────────────────────────────────────────────────────────────
    # Stage 2 – VAD worker

    def _vad_worker(self) -> None:
        vad = VADFilter(samplerate=self._samplerate)
        ring = CircularAudioBuffer(capacity=self._max_segment_samples * 2)
        last_flush = time.monotonic()

        while True:
            try:
                block = self._pcm_queue.get(timeout=0.2)
            except queue.Empty:
                block = None   # timeout – check flush condition

            if block is _SENTINEL:
                # Final flush
                self._flush_speech(vad, ring)
                self._speech_queue.put(_SENTINEL)
                break

            if block is not None:
                ring.write(block)

            # Flush if we've accumulated enough for a VAD pass
            if ring.size >= self._vad_window_samples or (
                time.monotonic() - last_flush >= self._max_segment_samples / self._samplerate
            ):
                self._flush_speech(vad, ring)
                last_flush = time.monotonic()

    def _flush_speech(self, vad: VADFilter, ring: CircularAudioBuffer) -> None:
        if ring.size == 0:
            return
        audio = ring.drain()
        chunks = vad.get_speech_chunks(audio)
        for chunk in chunks:
            try:
                self._speech_queue.put(chunk.audio, timeout=1.0)
            except queue.Full:
                pass  # drop if inference can't keep up

    # ──────────────────────────────────────────────────────────────────
    # Stage 3 – inference worker

    def _inference_worker(self) -> None:
        engine = WhisperEngine(model_id=self._model_id, language=self._language)

        while True:
            try:
                speech = self._speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if speech is _SENTINEL:
                break

            segments = engine.transcribe(speech, samplerate=self._samplerate)
            for seg in segments:
                if seg.text:
                    self._on_transcript(seg.text, seg.latency_ms)

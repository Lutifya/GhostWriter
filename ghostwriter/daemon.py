"""GhostWriter daemon – orchestrates audio capture, transcription, and display.

The daemon wires three components together:

1. :py:class:`~ghostwriter.audio_capture.AudioCapture` – reads audio blocks
   from the system-audio monitor device.
2. :py:class:`~ghostwriter.transcriber.Transcriber` – accumulates audio
   blocks and runs Whisper inference on each buffer.
3. :py:class:`~ghostwriter.overlay.SubtitleOverlay` – displays the
   transcribed text in a transparent, always-on-top Tkinter window.

The audio-capture loop runs in a daemon thread; the Whisper inference loop
runs in a second daemon thread; the Tkinter event loop runs on the main
thread (required by Tk on most platforms).
"""

import queue
import signal
import sys
import threading
from typing import Optional

from .audio_capture import AudioCapture
from .config import Config
from .overlay import SubtitleOverlay
from .transcriber import Transcriber


class GhostWriterDaemon:
    """Lightweight daemon that produces real-time subtitles for system audio.

    Args:
        config: :py:class:`~ghostwriter.config.Config` instance.  A default
            configuration is used when ``None`` is passed.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config: Config = config or Config()

        self._audio_capture: Optional[AudioCapture] = None
        self._transcriber: Optional[Transcriber] = None
        self._overlay: Optional[SubtitleOverlay] = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start all components and block until the overlay is closed."""
        self._running = True

        # Build overlay
        self._overlay = SubtitleOverlay(
            font=(self.config.font_family, self.config.font_size, "bold"),
            fg_color=self.config.text_color,
            bg_color=self.config.background_color,
            alpha=self.config.opacity,
            max_lines=self.config.max_lines,
            display_duration=self.config.display_duration,
            position=self.config.position,
        )

        # Build transcriber
        self._transcriber = Transcriber(
            model_size=self.config.model_size,
            language=self.config.language,
            on_transcription=self._on_transcription,
            buffer_duration=self.config.buffer_duration,
        )
        self._transcriber.start()

        # Resolve audio device
        device = self.config.audio_device
        if device is None:
            device = AudioCapture.find_monitor_device()

        # Build audio capture
        self._audio_capture = AudioCapture(
            device=device,
            sample_rate=16_000,
            channels=1,
        )

        # Start the audio-capture loop in a daemon thread
        capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="ghostwriter-capture"
        )
        capture_thread.start()

        # Register SIGINT / SIGTERM handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Run the Tk overlay on the main thread (blocks until closed)
        self._overlay.run()

    def stop(self) -> None:
        """Stop all components gracefully."""
        self._running = False
        if self._audio_capture is not None:
            self._audio_capture.stop()
        if self._transcriber is not None:
            self._transcriber.stop()
        if self._overlay is not None:
            self._overlay.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Read audio blocks and feed them to the transcriber."""
        self._audio_capture.start()
        try:
            while self._running:
                try:
                    chunk = self._audio_capture.read(timeout=1.0)
                    self._transcriber.feed_audio(chunk)
                except queue.Empty:
                    pass
        finally:
            self._audio_capture.stop()

    def _on_transcription(self, text: str) -> None:
        """Receive a transcription result and push it to the overlay."""
        if self._overlay is not None:
            self._overlay.show_text(text)

    def _signal_handler(self, _signum, _frame) -> None:
        """Handle SIGINT / SIGTERM for clean shutdown."""
        self.stop()
        sys.exit(0)

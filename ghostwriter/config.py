"""Configuration dataclass for GhostWriter."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """All tuneable parameters for GhostWriter.

    Attributes:
        audio_device: Index of the sounddevice input device to capture.
            ``None`` triggers automatic detection of a monitor / loopback
            device so that system *output* is captured rather than the
            microphone.
        model_size: faster-whisper model size.  Smaller models are faster
            but less accurate.  Choices: ``tiny``, ``base``, ``small``,
            ``medium``, ``large``.
        language: BCP-47 language code passed to Whisper (e.g. ``"en"``,
            ``"fr"``).  ``None`` enables automatic language detection.
        font_family: Font family used for the subtitle label.
        font_size: Font size (points) for subtitles.
        text_color: Foreground colour of the subtitle text (Tk colour name
            or ``#RRGGBB`` hex string).
        background_color: Background colour of the subtitle box.
        opacity: Overall window opacity in the range ``[0.0, 1.0]``.
        display_duration: Milliseconds to keep the last subtitle visible
            before the overlay auto-hides.
        position: Vertical placement of the overlay – ``"bottom"``,
            ``"top"``, or ``"center"``.
        max_lines: Maximum number of subtitle lines kept on screen at once.
        buffer_duration: Seconds of audio buffered before each Whisper
            inference call.
    """

    # Audio
    audio_device: Optional[int] = None

    # Transcription
    model_size: str = "base"
    language: Optional[str] = None

    # Display
    font_family: str = "Arial"
    font_size: int = 24
    text_color: str = "white"
    background_color: str = "black"
    opacity: float = 0.85
    display_duration: int = 5000
    position: str = "bottom"
    max_lines: int = 2

    # Pipeline tuning
    buffer_duration: float = 3.0

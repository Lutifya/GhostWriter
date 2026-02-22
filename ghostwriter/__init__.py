"""GhostWriter – System-Wide Live Captions.

A lightweight background daemon that listens to your system's audio output
and generates real-time subtitles in a transparent, always-on-top overlay.
"""

__version__ = "0.1.0"
__author__ = "GhostWriter Contributors"

from .config import Config
from .daemon import GhostWriterDaemon

__all__ = ["GhostWriterDaemon", "Config"]

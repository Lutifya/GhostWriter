"""Tests for ghostwriter.daemon."""

from unittest.mock import MagicMock, patch

import pytest

from ghostwriter.config import Config
from ghostwriter.daemon import GhostWriterDaemon


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_default_init():
    d = GhostWriterDaemon()
    assert isinstance(d.config, Config)
    assert not d._running
    assert d._audio_capture is None
    assert d._transcriber is None
    assert d._overlay is None


def test_custom_config():
    cfg = Config(model_size="tiny", language="en", position="top")
    d = GhostWriterDaemon(config=cfg)
    assert d.config.model_size == "tiny"
    assert d.config.language == "en"
    assert d.config.position == "top"


# ---------------------------------------------------------------------------
# _on_transcription
# ---------------------------------------------------------------------------


def test_on_transcription_calls_show_text():
    d = GhostWriterDaemon()
    d._overlay = MagicMock()
    d._on_transcription("Hello world")
    d._overlay.show_text.assert_called_once_with("Hello world")


def test_on_transcription_without_overlay_is_noop():
    d = GhostWriterDaemon()
    d._overlay = None
    d._on_transcription("Hello")  # Must not raise


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


def test_stop_sets_running_false():
    d = GhostWriterDaemon()
    d._running = True
    d._audio_capture = MagicMock()
    d._transcriber = MagicMock()
    d._overlay = MagicMock()

    d.stop()

    assert not d._running


def test_stop_calls_component_stops():
    d = GhostWriterDaemon()
    d._running = True
    d._audio_capture = MagicMock()
    d._transcriber = MagicMock()
    d._overlay = MagicMock()

    d.stop()

    d._audio_capture.stop.assert_called_once()
    d._transcriber.stop.assert_called_once()
    d._overlay.close.assert_called_once()


def test_stop_partial_init():
    """stop() must not raise when some components were never initialised."""
    d = GhostWriterDaemon()
    d._running = True
    d._audio_capture = MagicMock()
    # _transcriber and _overlay remain None
    d.stop()
    d._audio_capture.stop.assert_called_once()


# ---------------------------------------------------------------------------
# Config propagation
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = Config()
    assert cfg.model_size == "base"
    assert cfg.language is None
    assert cfg.audio_device is None
    assert cfg.font_family == "Arial"
    assert cfg.font_size == 24
    assert cfg.text_color == "white"
    assert cfg.background_color == "black"
    assert cfg.opacity == 0.85
    assert cfg.display_duration == 5_000
    assert cfg.position == "bottom"
    assert cfg.max_lines == 2
    assert cfg.buffer_duration == 3.0

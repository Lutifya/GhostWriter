"""Tests for ghostwriter.transcriber."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ghostwriter.transcriber import Transcriber


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_default_init():
    t = Transcriber()
    assert t.model_size == "base"
    assert t.device == "cpu"
    assert t.compute_type == "int8"
    assert t.language is None
    assert t.on_transcription is None
    assert t._model is None
    assert not t._running


def test_custom_init():
    cb = MagicMock()
    t = Transcriber(model_size="small", language="en", on_transcription=cb)
    assert t.model_size == "small"
    assert t.language == "en"
    assert t.on_transcription is cb


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


@patch("ghostwriter.transcriber.WhisperModel")
def test_load_model(mock_cls):
    t = Transcriber(model_size="tiny")
    t.load_model()
    mock_cls.assert_called_once_with("tiny", device="cpu", compute_type="int8")
    assert t._model is not None


# ---------------------------------------------------------------------------
# feed_audio
# ---------------------------------------------------------------------------


def test_feed_audio_enqueues_chunk():
    t = Transcriber()
    chunk = np.zeros(8_000, dtype=np.float32)
    t.feed_audio(chunk)
    assert not t._audio_queue.empty()
    result = t._audio_queue.get_nowait()
    np.testing.assert_array_equal(result, chunk)


# ---------------------------------------------------------------------------
# _transcribe
# ---------------------------------------------------------------------------


def test_transcribe_fires_callback():
    cb = MagicMock()
    t = Transcriber(on_transcription=cb)

    seg = MagicMock()
    seg.text = "  Hello world  "
    t._model = MagicMock()
    t._model.transcribe.return_value = ([seg], MagicMock())

    t._transcribe(np.zeros(48_000, dtype=np.float32))
    cb.assert_called_once_with("Hello world")


def test_transcribe_multiple_segments():
    cb = MagicMock()
    t = Transcriber(on_transcription=cb)

    segs = [MagicMock(text="Hello"), MagicMock(text="world")]
    t._model = MagicMock()
    t._model.transcribe.return_value = (segs, MagicMock())

    t._transcribe(np.zeros(48_000, dtype=np.float32))
    cb.assert_called_once_with("Hello world")


def test_transcribe_empty_result_skips_callback():
    cb = MagicMock()
    t = Transcriber(on_transcription=cb)
    t._model = MagicMock()
    t._model.transcribe.return_value = ([], MagicMock())

    t._transcribe(np.zeros(48_000, dtype=np.float32))
    cb.assert_not_called()


def test_transcribe_whitespace_only_skips_callback():
    cb = MagicMock()
    t = Transcriber(on_transcription=cb)

    seg = MagicMock(text="   ")
    t._model = MagicMock()
    t._model.transcribe.return_value = ([seg], MagicMock())

    t._transcribe(np.zeros(48_000, dtype=np.float32))
    cb.assert_not_called()


def test_transcribe_no_model_does_nothing():
    """_transcribe should be a no-op when the model hasn't been loaded."""
    cb = MagicMock()
    t = Transcriber(on_transcription=cb)
    t._model = None
    t._transcribe(np.zeros(8_000, dtype=np.float32))
    cb.assert_not_called()


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------


@patch("ghostwriter.transcriber.WhisperModel")
def test_start_launches_thread(mock_cls):
    t = Transcriber(model_size="tiny")
    t.start()
    assert t._running
    assert t._thread is not None
    assert t._thread.is_alive()
    t.stop()


@patch("ghostwriter.transcriber.WhisperModel")
def test_stop_joins_thread(mock_cls):
    t = Transcriber(model_size="tiny")
    t.start()
    t.stop()
    assert not t._running
    assert t._thread is None

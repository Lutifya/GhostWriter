"""Tests for ghostwriter.audio_capture."""

import queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ghostwriter.audio_capture import AudioCapture


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_default_init():
    capture = AudioCapture()
    assert capture.sample_rate == 16_000
    assert capture.channels == 1
    assert capture.chunk_duration == 0.5
    assert capture.blocksize == 8_000
    assert capture.device is None
    assert not capture.is_running


def test_custom_params():
    capture = AudioCapture(sample_rate=44_100, channels=2, chunk_duration=1.0)
    assert capture.sample_rate == 44_100
    assert capture.channels == 2
    assert capture.blocksize == 44_100


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


def test_callback_enqueues_mono_chunk():
    capture = AudioCapture()
    capture._running = True

    indata = np.ones((800, 1), dtype=np.float32)
    capture._callback(indata, 800, None, None)

    chunk = capture._queue.get_nowait()
    assert chunk.shape == (800,)
    assert chunk.dtype == np.float32
    np.testing.assert_array_equal(chunk, np.ones(800, dtype=np.float32))


def test_callback_ignores_when_stopped():
    capture = AudioCapture()
    capture._running = False

    indata = np.zeros((800, 1), dtype=np.float32)
    capture._callback(indata, 800, None, None)

    assert capture._queue.empty()


def test_callback_prints_status(capsys):
    capture = AudioCapture()
    capture._running = True

    indata = np.zeros((800, 1), dtype=np.float32)
    capture._callback(indata, 800, None, "input overflow")

    captured = capsys.readouterr()
    assert "input overflow" in captured.out


# ---------------------------------------------------------------------------
# find_monitor_device
# ---------------------------------------------------------------------------


def test_find_monitor_device_returns_index():
    mock_devices = [
        {"name": "Built-in Microphone", "max_input_channels": 1, "max_output_channels": 0},
        {"name": "pulse monitor", "max_input_channels": 2, "max_output_channels": 0},
        {"name": "HDMI Output", "max_input_channels": 0, "max_output_channels": 2},
    ]
    with patch("sounddevice.query_devices", return_value=mock_devices):
        assert AudioCapture.find_monitor_device() == 1


def test_find_monitor_device_returns_none_when_absent():
    mock_devices = [
        {"name": "Built-in Microphone", "max_input_channels": 1, "max_output_channels": 0},
    ]
    with patch("sounddevice.query_devices", return_value=mock_devices):
        assert AudioCapture.find_monitor_device() is None


def test_find_monitor_device_case_insensitive():
    mock_devices = [
        {"name": "Pulse Monitor Source", "max_input_channels": 2, "max_output_channels": 0},
    ]
    with patch("sounddevice.query_devices", return_value=mock_devices):
        assert AudioCapture.find_monitor_device() == 0


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------


@patch("sounddevice.InputStream")
def test_start_creates_stream(mock_cls):
    mock_stream = MagicMock()
    mock_cls.return_value = mock_stream

    capture = AudioCapture(device=1, sample_rate=16_000, channels=1)
    capture.start()

    assert capture.is_running
    mock_cls.assert_called_once()
    mock_stream.start.assert_called_once()


@patch("sounddevice.InputStream")
def test_stop_closes_stream(mock_cls):
    mock_stream = MagicMock()
    mock_cls.return_value = mock_stream

    capture = AudioCapture()
    capture.start()
    capture.stop()

    assert not capture.is_running
    mock_stream.stop.assert_called_once()
    mock_stream.close.assert_called_once()
    assert capture._stream is None


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


def test_read_returns_queued_chunk():
    capture = AudioCapture()
    expected = np.ones(800, dtype=np.float32)
    capture._queue.put(expected)

    result = capture.read(timeout=1.0)
    np.testing.assert_array_equal(result, expected)


def test_read_raises_on_timeout():
    capture = AudioCapture()
    with pytest.raises(queue.Empty):
        capture.read(timeout=0.01)

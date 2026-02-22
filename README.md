# GhostWriter

GhostWriter is a lightweight background daemon that listens to your system's
audio output and generates real-time subtitles directly on your screen.

Whether you are in a Zoom call with friends, watching a movie in a foreign
language, or viewing a video without captions, GhostWriter provides a sleek,
transparent overlay that stays on top of everything, making any audio content
accessible and easy to follow.

---

## Features

- **System-audio capture** – records what your speakers play, not the
  microphone, using PulseAudio/PipeWire monitor devices (Linux) or WASAPI
  loopback (Windows).
- **Real-time transcription** – powered by
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper), a highly
  efficient CTranslate2-based Whisper implementation that runs fully offline.
- **Transparent overlay** – a borderless, always-on-top Tkinter window that
  auto-hides when audio is silent and auto-positions itself at the bottom
  (or top / centre) of the screen.
- **Configurable** – model size, language, font, opacity, position and more
  can all be tuned via CLI flags.

---

## Requirements

- Python ≥ 3.8
- [sounddevice](https://python-sounddevice.readthedocs.io/) and PortAudio
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- NumPy
- Tkinter (ships with most Python distributions; on Debian/Ubuntu install
  `python3-tk`)

### Linux – capturing system audio

GhostWriter automatically looks for a PulseAudio / PipeWire *monitor*
source.  To list all detected devices and confirm which one will be used, run:

```bash
python -m ghostwriter --list-devices
```

---

## Installation

```bash
pip install -r requirements.txt
# or install as a package
pip install .
```

---

## Usage

```bash
# Auto-detect system monitor device, use the "base" Whisper model
python -m ghostwriter

# Pick a specific audio device and a smaller/faster model
python -m ghostwriter --device 2 --model tiny

# Specify language explicitly (skip auto-detection overhead)
python -m ghostwriter --language en

# Show the overlay at the top of the screen with larger text
python -m ghostwriter --position top --font-size 32

# List all available audio devices
python -m ghostwriter --list-devices
```

### All CLI options

| Flag | Default | Description |
|---|---|---|
| `--device INDEX` | auto | sounddevice input device index |
| `--model {tiny,base,small,medium,large}` | `base` | Whisper model size |
| `--language CODE` | auto | BCP-47 language code (e.g. `en`, `fr`) |
| `--position {bottom,top,center}` | `bottom` | Overlay placement |
| `--font-size N` | `24` | Subtitle font size in points |
| `--opacity 0–1` | `0.85` | Window opacity |
| `--buffer-duration SECONDS` | `3.0` | Audio buffer before each inference |
| `--list-devices` | – | Print devices and exit |

---

## Architecture

```
ghostwriter/
├── __init__.py        # Public API and version
├── __main__.py        # CLI entry point (argparse)
├── config.py          # Dataclass-based configuration
├── audio_capture.py   # sounddevice InputStream wrapper
├── transcriber.py     # faster-whisper inference loop (daemon thread)
├── overlay.py         # Tkinter transparent subtitle window (main thread)
└── daemon.py          # Orchestrator that wires the three components
```

The capture and transcription loops run in daemon threads; the Tkinter event
loop runs on the main thread (required by Tk on most platforms).

---

## Running the tests

```bash
pip install pytest
pytest tests/
```

---

## License

MIT

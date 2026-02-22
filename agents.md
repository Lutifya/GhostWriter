# GhostWriter – Agents Knowledge Base

> This file is maintained by the AI assistant and updated after every phase.  
> It serves as a living reference of architectural decisions, module responsibilities, known pitfalls, and implementation notes.

---

## Project Overview

**GhostWriter** is a cross-platform (macOS / Windows / Linux) live-caption overlay.  
Stack: Python 3.12+, faster-whisper (Distil-Whisper), silero-vad, PyQt6, sounddevice.

---

## Module Map

| Module | File | Status |
|--------|------|--------|
| Audio Capture | `src/audio/capture.py` | ✅ Phase 1 – done |
| STT Engine | `src/stt/engine.py` | ✅ Phase 2 – done |
| VAD Filter | `src/stt/vad.py` | ✅ Phase 2 – done |
| Pipeline | `src/pipeline/pipeline.py` | ✅ Phase 3 – done |
| Circular Buffer | `src/pipeline/buffer.py` | ✅ Phase 3 – done |
| Overlay UI | `src/ui/overlay.py` | ✅ Phase 4 – done |

---

## Phase 1 – Audio Capture (`src/audio/capture.py`)

### Architecture
- `AudioCapture` – abstract base class with `start(callback)` / `stop()`.
- Three concrete subclasses, selected by `platform.system()` via the `create_capture()` factory:
  - `BlackHoleCapture` (macOS) – looks for a device containing "BlackHole" in its name.
  - `WASAPILoopbackCapture` (Windows) – searches for a `[Loopback]` device; falls back to `WasapiSettings(loopback=True)`.
  - `PulseAudioMonitorCapture` (Linux) – matches any device whose name contains `.monitor`.
- All streams are `float32`, mono (1 channel), 16 kHz by default (Whisper's native rate).

### Key Decisions
- **16 kHz default sample rate**: Whisper was trained at 16 kHz; resampling would add latency.
- **mono**: Whisper processes mono audio; stereo adds no benefit and doubles memory.
- **blocksize=1024**: ~64 ms at 16 kHz – balances latency vs. callback overhead.
- **Non-blocking callback**: The `_sd_callback` never blocks; it hands data to the consumer via a `queue.Queue`.

### Known Pitfalls / Prerequisites
- **macOS**: BlackHole must be installed and set as the system output device (or used via a Multi-Output Device). Download: https://existential.audio/blackhole/
- **Windows**: sounddevice must be built with WASAPI support (default on Windows wheels). Older versions may lack `WasapiSettings`; handled with try/except.
- **Linux**: A PipeWire/PulseAudio monitor source must exist. Run `pactl list sources | grep monitor` to verify.  On headless systems the monitor may not be created; use `pactl load-module module-null-sink` to create a virtual sink.

### Debug Entry-Point
`python main_phase1.py` – prints RMS level bar to console.  
`python main_phase1.py --list-devices` – dumps all sounddevice devices.

---

## Phase 2 – STT Engine (`src/stt/engine.py`) & VAD (`src/stt/vad.py`)

### WhisperEngine
- Wraps `faster_whisper.WhisperModel`.
- Device auto-detection: CUDA (if torch available + `torch.cuda.is_available()`) → int8 on Apple Silicon ARM64 → int8 CPU fallback.
- `compute_type`: `float16` on CUDA, `int8` on CPU (fastest via CTranslate2 AVX/NEON).
- `beam_size=1` (greedy) for real-time use; increase for better accuracy at the cost of latency.
- `vad_filter=False` in `transcribe()` – VAD is handled separately.
- Returns `list[TranscriptSegment]` with `text`, `start`, `end`, `language`, `latency_ms`.

### VADFilter
- Wraps `silero_vad.load_silero_vad()` + `get_speech_timestamps()`.
- Operates at **16 kHz** only (8 kHz also supported but not used).
- Key params: `threshold=0.5`, `min_speech_duration_ms=250`, `min_silence_duration_ms=300`, `padding_ms=30`.
- `get_speech_chunks(audio)` → `list[SpeechChunk]` (each has `.audio`, `.start_sample`, `.end_sample`).
- `is_speech(block)` → float probability (for lightweight real-time gating).

### Debug Entry-Point
`python main_phase2.py --wav file.wav` or `python main_phase2.py --record 5`

---

## Phase 3 – Real-Time Pipeline (`src/pipeline/`)

### CircularAudioBuffer (`buffer.py`)
- Lock-protected ring buffer for float32 mono PCM.
- `write(data)` – appends, overwrites oldest if full.
- `read_last(n)` – returns the N most recent samples (non-destructive).
- `drain()` – returns all samples and resets the buffer.

### Pipeline (`pipeline.py`)
Thread model:
```
AudioThread  ──[pcm_queue]──▶  VADThread  ──[speech_queue]──▶  InferenceThread → on_transcript callback
```
- `_audio_callback` (sounddevice thread): puts PCM blocks into `pcm_queue` (non-blocking, drops if full).
- `_vad_worker`: accumulates into `CircularAudioBuffer`; drains and runs VAD every `vad_window_s` (default 1 s) or when the buffer exceeds `max_segment_s` (default 30 s); puts speech chunks into `speech_queue`.
- `_inference_worker`: calls `WhisperEngine.transcribe()` for each speech chunk; invokes `on_transcript(text, latency_ms)`.
- Shutdown via poison-pill `None` sentinel through both queues.
- All threads are **daemon threads** (die with the process).

### Key Decisions
- Queue bounds (`pcm_queue_size=512`, `speech_queue_size=32`) prevent unbounded memory growth under load.
- VAD and Whisper models are loaded inside their worker threads (not in `start()`) to avoid blocking the caller.

### Debug Entry-Point
`python main_phase3.py` – live transcription printed to console.

---

## Phase 4 – Overlay UI (`src/ui/overlay.py`)

### OverlayWindow (PyQt6)
- `FramelessWindowHint | WindowStaysOnTopHint | Tool` – frameless, always on top, no taskbar.
- `WA_TranslucentBackground` – true transparency (composited by the OS).
- `WA_TransparentForMouseEvents` – all clicks fall through to the window below.
- `X11BypassWindowManagerHint` added on Linux to stay above fullscreen/borderless games.
- Custom `paintEvent` draws a semi-transparent rounded-rectangle background.
- Positioned horizontally centred, 80 px from the bottom of the primary screen.
- Auto-resizes with text content.

### Thread Safety
`show_text(text)` is callable from any thread via `QMetaObject.invokeMethod(..., QueuedConnection)` → `_set_text(str)` slot runs in the Qt main thread.

### Auto-Clear
`QTimer` (single-shot, default 4 s) calls `_clear_text()` after each caption, hiding the window.

### Full App Entry-Point
`python main.py` – starts pipeline + shows overlay.

---

## Phase 2 – STT Engine & VAD

---

## Cross-Cutting Notes

### Thread Model (planned)
```
AudioThread  ──[pcm queue]──▶  VADThread  ──[speech queue]──▶  InferenceThread  ──[text signal]──▶  UIThread
```

### Hardware Acceleration
| Platform | Device | Notes |
|----------|--------|-------|
| macOS (Apple Silicon) | `cpu` | CTranslate2 ARM64 NEON; Metal not yet stable in faster-whisper |
| Windows/Linux + NVIDIA | `cuda` | requires torch+cu121 wheels |
| Windows/Linux CPU | `cpu` | AVX2/AVX512 path via CTranslate2 |

### Model Choice
- **`distil-large-v3`** – best multilingual accuracy, ~1.5 GB VRAM.
- **`distil-medium.en`** – English only, fastest, ~750 MB VRAM.
- Set via `--model` CLI flag (available in `main_phase2.py`, `main_phase3.py`, and `main.py`).

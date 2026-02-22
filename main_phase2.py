#!/usr/bin/env python3
"""
Phase 2 debug entry-point.

Loads a Distil-Whisper model and a silero-vad filter, then transcribes
a WAV file (or a short recorded clip) and prints the result + latency.

Usage
-----
    # Transcribe a WAV file:
    python main_phase2.py --wav path/to/file.wav

    # Record 5 seconds from the loopback device then transcribe:
    python main_phase2.py --record 5

    # Choose a different Distil-Whisper model:
    python main_phase2.py --wav file.wav --model distil-medium.en
"""

from __future__ import annotations

import argparse
import queue
import time
import wave
from pathlib import Path

import numpy as np

from src.stt.engine import WhisperEngine
from src.stt.vad import VADFilter

SAMPLE_RATE = 16_000


def load_wav(path: Path) -> np.ndarray:
    """Load a 16 kHz mono WAV file as a float32 ndarray."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)[:, 0]

    if framerate != SAMPLE_RATE:
        raise ValueError(
            f"WAV file must be {SAMPLE_RATE} Hz (got {framerate} Hz). "
            "Resample first: ffmpeg -i input.wav -ar 16000 output.wav"
        )

    return audio


def record_clip(seconds: float) -> np.ndarray:
    """Record *seconds* seconds from the loopback device."""
    from src.audio.capture import create_capture

    buf: list[np.ndarray] = []
    done = queue.Event() if False else __import__("threading").Event()

    def on_audio(block: np.ndarray) -> None:
        buf.append(block.copy())

    capture = create_capture(samplerate=SAMPLE_RATE, blocksize=1024)
    print(f"Recording {seconds:.1f}s via {capture.__class__.__name__} …")
    capture.start(on_audio)
    time.sleep(seconds)
    capture.stop()

    return np.concatenate(buf) if buf else np.zeros(0, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="GhostWriter Phase 2 – STT + VAD test")
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--wav", type=Path, help="Path to a 16 kHz mono WAV file")
    src_group.add_argument("--record", type=float, metavar="SECONDS", help="Record N seconds from loopback")
    parser.add_argument("--model", default="distil-large-v3", help="Distil-Whisper model id")
    parser.add_argument("--language", default=None, help="Force language code (e.g. 'en')")
    parser.add_argument("--no-vad", action="store_true", help="Skip VAD and transcribe raw audio")
    args = parser.parse_args()

    # ── Load audio ────────────────────────────────────────────────────────────
    if args.wav:
        audio = load_wav(args.wav)
        print(f"Loaded {len(audio)/SAMPLE_RATE:.2f}s of audio from {args.wav}")
    else:
        audio = record_clip(args.record)
        print(f"Recorded {len(audio)/SAMPLE_RATE:.2f}s of audio")

    # ── VAD ───────────────────────────────────────────────────────────────────
    if not args.no_vad:
        vad = VADFilter(samplerate=SAMPLE_RATE)
        t_vad = time.monotonic()
        chunks = vad.get_speech_chunks(audio)
        vad_ms = (time.monotonic() - t_vad) * 1000
        print(f"\nVAD detected {len(chunks)} speech segment(s) in {vad_ms:.0f} ms")

        if not chunks:
            print("No speech detected – try --no-vad to skip VAD.")
            return

        # Concatenate all speech chunks for transcription
        speech_audio = np.concatenate([c.audio for c in chunks])
    else:
        speech_audio = audio

    # ── Transcribe ────────────────────────────────────────────────────────────
    engine = WhisperEngine(model_id=args.model, language=args.language)
    print(f"\nTranscribing {len(speech_audio)/SAMPLE_RATE:.2f}s of speech …")

    t0 = time.monotonic()
    segments = engine.transcribe(speech_audio, samplerate=SAMPLE_RATE)
    total_ms = (time.monotonic() - t0) * 1000

    print(f"\n{'─'*60}")
    print(f"  Segments : {len(segments)}")
    print(f"  Total latency : {total_ms:.0f} ms")
    print(f"{'─'*60}")
    for seg in segments:
        print(f"  [{seg.start:.2f}s → {seg.end:.2f}s] ({seg.latency_ms:.0f} ms)  {seg.text}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()

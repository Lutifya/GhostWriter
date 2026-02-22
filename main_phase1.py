#!/usr/bin/env python3
"""
Phase 1 debug entry-point.

Captures system audio and prints the RMS level to the console once per second.
Press Ctrl+C to stop.

Usage
-----
    python main_phase1.py [--list-devices]
"""

from __future__ import annotations

import argparse
import queue
import sys
import time

import numpy as np

from src.audio.capture import AudioCapture, create_capture


def compute_rms(block: np.ndarray) -> float:
    """Return the Root Mean Square amplitude of *block* (range 0.0–1.0)."""
    return float(np.sqrt(np.mean(block ** 2)))


def rms_to_bar(rms: float, width: int = 40) -> str:
    """Convert an RMS value (0–1) to an ASCII bar string."""
    filled = int(min(rms * width * 10, width))  # scale up for typical audio levels
    return f"[{'█' * filled}{' ' * (width - filled)}] {rms:.5f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="GhostWriter Phase 1 – Audio Level Monitor")
    parser.add_argument("--list-devices", action="store_true", help="Print available audio devices and exit")
    parser.add_argument("--samplerate", type=int, default=16_000, help="Sample rate (Hz)")
    parser.add_argument("--blocksize", type=int, default=1024, help="Block size (frames per callback)")
    args = parser.parse_args()

    if args.list_devices:
        AudioCapture.list_devices()
        return

    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)

    def on_audio(block: np.ndarray) -> None:
        try:
            audio_queue.put_nowait(block)
        except queue.Full:
            pass  # drop oldest – non-blocking

    capture = create_capture(samplerate=args.samplerate, blocksize=args.blocksize)
    print(f"Starting capture on {capture.__class__.__name__} (sr={args.samplerate}, bs={args.blocksize})")
    print("Press Ctrl+C to stop.\n")

    capture.start(on_audio)

    # Accumulate one second of blocks, then print the average RMS
    accumulated: list[np.ndarray] = []
    last_print = time.monotonic()

    try:
        while True:
            try:
                block = audio_queue.get(timeout=0.1)
                accumulated.append(block)
            except queue.Empty:
                pass

            now = time.monotonic()
            if now - last_print >= 1.0 and accumulated:
                combined = np.concatenate(accumulated)
                rms = compute_rms(combined)
                bar = rms_to_bar(rms)
                sys.stdout.write(f"\rRMS {bar}  ")
                sys.stdout.flush()
                accumulated.clear()
                last_print = now
    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        capture.stop()


if __name__ == "__main__":
    main()

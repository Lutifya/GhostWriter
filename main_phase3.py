#!/usr/bin/env python3
"""
Phase 3 debug entry-point.

Runs the full audio → VAD → Whisper pipeline and prints live transcriptions.
Press Ctrl+C to stop.

Usage
-----
    python main_phase3.py
    python main_phase3.py --model distil-medium.en --language en
"""

from __future__ import annotations

import argparse
import sys
import time

from src.pipeline.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="GhostWriter Phase 3 – Live Transcription")
    parser.add_argument("--model", default="distil-large-v3", help="Distil-Whisper model id")
    parser.add_argument("--language", default=None, help="Force language code (e.g. 'en')")
    parser.add_argument("--vad-window", type=float, default=1.0, metavar="S", help="VAD analysis window (seconds)")
    args = parser.parse_args()

    def on_transcript(text: str, latency_ms: float) -> None:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] ({latency_ms:.0f}ms)  {text}")
        sys.stdout.flush()

    pipeline = Pipeline(
        on_transcript=on_transcript,
        model_id=args.model,
        language=args.language,
        vad_window_s=args.vad_window,
    )

    print("Starting GhostWriter pipeline. Press Ctrl+C to stop.\n")
    pipeline.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping …")
    finally:
        pipeline.stop()
        print("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
GhostWriter – full application entry-point.

Starts the real-time pipeline and displays transcriptions in the overlay.

Usage
-----
    python main.py
    python main.py --model distil-medium.en --language en --font-size 24
"""

from __future__ import annotations

import argparse
import sys

from PyQt6.QtWidgets import QApplication

from src.pipeline.pipeline import Pipeline
from src.ui.overlay import OverlayWindow


def main() -> None:
    parser = argparse.ArgumentParser(description="GhostWriter – Live Caption Overlay")
    parser.add_argument("--model", default="distil-large-v3", help="Distil-Whisper model id")
    parser.add_argument("--language", default=None, help="Force language code (e.g. 'en')")
    parser.add_argument("--font-size", type=int, default=22, help="Caption font size (pt)")
    parser.add_argument("--display-ms", type=int, default=4000, help="Caption display duration (ms)")
    parser.add_argument("--vad-window", type=float, default=1.0, help="VAD analysis window (s)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # pipeline threads keep the loop alive

    overlay = OverlayWindow(
        font_size=args.font_size,
        display_duration_ms=args.display_ms,
    )

    def on_transcript(text: str, latency_ms: float) -> None:
        overlay.show_text(text)

    pipeline = Pipeline(
        on_transcript=on_transcript,
        model_id=args.model,
        language=args.language,
        vad_window_s=args.vad_window,
    )

    print("GhostWriter starting … (Ctrl+C or close overlay to quit)")
    pipeline.start()

    try:
        exit_code = app.exec()
    finally:
        pipeline.stop()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

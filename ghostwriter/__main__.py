"""CLI entry point for GhostWriter.

Usage examples::

    # Auto-detect system audio monitor device
    python -m ghostwriter

    # List audio devices and exit
    python -m ghostwriter --list-devices

    # Specify device, model size, and language explicitly
    python -m ghostwriter --device 2 --model small --language en
"""

import argparse
import sys

from .audio_capture import AudioCapture
from .config import Config
from .daemon import GhostWriterDaemon


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ghostwriter",
        description=(
            "GhostWriter – Real-time subtitle overlay for system audio.\n\n"
            "Listens to the system's audio output and renders live subtitles "
            "in a transparent, always-on-top window."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--device",
        type=int,
        default=None,
        metavar="INDEX",
        help="Audio device index to capture from (default: auto-detect monitor device).",
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        dest="model_size",
        help="Whisper model size.  Smaller = faster; larger = more accurate.",
    )
    parser.add_argument(
        "--language",
        default=None,
        metavar="CODE",
        help="BCP-47 language code (e.g. 'en', 'fr').  Omit for auto-detection.",
    )
    parser.add_argument(
        "--position",
        default="bottom",
        choices=["bottom", "top", "center"],
        help="Vertical placement of the subtitle overlay.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=24,
        dest="font_size",
        help="Subtitle font size in points.",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.85,
        help="Overlay window opacity between 0.0 (invisible) and 1.0 (opaque).",
    )
    parser.add_argument(
        "--buffer-duration",
        type=float,
        default=3.0,
        dest="buffer_duration",
        metavar="SECONDS",
        help="Seconds of audio buffered before each transcription call.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print all available audio devices and exit.",
    )
    return parser


def main(argv=None) -> None:
    """Parse arguments and launch GhostWriter."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        devices = AudioCapture.list_devices()
        print("Available audio devices:")
        for i, dev in enumerate(devices):
            in_ch = dev["max_input_channels"]
            out_ch = dev["max_output_channels"]
            print(f"  [{i:2d}] {dev['name']}  (in: {in_ch}, out: {out_ch})")
        sys.exit(0)

    config = Config(
        audio_device=args.device,
        model_size=args.model_size,
        language=args.language,
        position=args.position,
        font_size=args.font_size,
        opacity=args.opacity,
        buffer_duration=args.buffer_duration,
    )

    daemon = GhostWriterDaemon(config=config)
    daemon.start()


if __name__ == "__main__":
    main()

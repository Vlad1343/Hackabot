"""HACKABOT Unified Entry Point."""
from __future__ import annotations

import argparse
import asyncio

import indoor_demo
import outdoor_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HACKABOT unified entry point")
    parser.add_argument("--mode", choices=["indoor", "outdoor"], default="indoor", help="Indoor or outdoor demo")
    # Indoor embedded loop options
    parser.add_argument("--role", choices=["sensor", "receiver", "both"], default="both", help="Indoor only: which node role to run")
    parser.add_argument("--sensor-count", type=int, default=1, help="Indoor only: number of ultrasonic sensors (1-3)")
    parser.add_argument("--tx-rate-hz", type=float, default=10.0, help="Indoor only: wireless packet rate")
    parser.add_argument("--danger-cm", type=float, default=30.0, help="Indoor only: DANGER threshold")
    parser.add_argument("--safe-cm", type=float, default=100.0, help="Indoor only: SAFE threshold")
    parser.add_argument("--max-range-cm", type=float, default=400.0, help="Indoor only: max valid sensor range")
    parser.add_argument("--filter-mode", choices=["median", "average"], default="median", help="Indoor only: filter method over last 3 reads")
    parser.add_argument("--use-hardware-radio", action="store_true", help="Indoor only: use nRF24 hardware transport placeholder")
    # Outdoor/computer-vision options (ignored by indoor loop)
    parser.add_argument("--stream", type=str, default="http://10.205.48.81:4747/video", help="DroidCam stream URL")
    parser.add_argument("--stream-fallback", type=str, default="http://10.205.48.81:4747/mjpegfeed", help="DroidCam fallback stream URL")
    parser.add_argument("--video", type=str, default="", help="Pre-recorded video path (optional)")
    parser.add_argument("--show-overlay", action="store_true", help="Show bounding box overlay")
    parser.add_argument("--no-flip", action="store_true", help="Disable mirrored USER VIEW window")
    parser.add_argument("--mirror-input", dest="mirror_input", action="store_true", default=None, help="Mirror USER VIEW window (inference stays non-mirrored)")
    parser.add_argument("--no-mirror-input", dest="mirror_input", action="store_false", help="Do not mirror USER VIEW window")
    parser.add_argument("--keyboard-sensor", action="store_true", help="Enable keyboard distance mock (O/C)")
    parser.add_argument("--simulate-obstacles", action="store_true", help="Enable obstacle simulator")
    parser.add_argument("--simulate-navigation", action="store_true", help="Enable navigation simulator")
    parser.add_argument("--nav-interval", type=float, default=6.0, help="Navigation message interval (sec)")
    parser.add_argument("--detection-interval", type=float, default=0.0, help="Detection interval in sec (0 = use config)")
    return parser


async def run() -> None:
    args = build_parser().parse_args()
    if args.mode == "indoor":
        await indoor_demo.run(args)
    else:
        await outdoor_demo.run(args)


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("[Main] Stopped by user")

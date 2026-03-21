#!/usr/bin/env python3
"""
Laptop monitor for Pico B relay.

Reads USB serial packets like:
  L:0,C:2,R:1
  C:1
  NO_SIGNAL

Outputs:
- Clear terminal status
- Voice alerts (pyttsx3) if enabled
"""

from __future__ import annotations

import argparse
import re
import sys
import time

import serial

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

PATTERN_3 = re.compile(r"L:(\d),C:(\d),R:(\d)")
PATTERN_1 = re.compile(r"C:(\d)")


def parse_packet(line: str):
    line = line.strip()
    if line == "NO_SIGNAL":
        return {"type": "no_signal"}
    m3 = PATTERN_3.fullmatch(line)
    if m3:
        return {"type": "state3", "L": int(m3.group(1)), "C": int(m3.group(2)), "R": int(m3.group(3))}
    m1 = PATTERN_1.fullmatch(line)
    if m1:
        return {"type": "state1", "C": int(m1.group(1))}
    return None


def state_to_text(state: int) -> str:
    if state == 2:
        return "DANGER"
    if state == 1:
        return "WARNING"
    return "SAFE"


def choose_message(pkt: dict) -> str:
    if pkt["type"] == "no_signal":
        return "No signal from sensor node"
    if pkt["type"] == "state1":
        c = pkt["C"]
        if c == 2:
            return "Danger ahead"
        if c == 1:
            return "Obstacle ahead"
        return "Clear path"

    l, c, r = pkt["L"], pkt["C"], pkt["R"]
    if c == 2:
        return "Danger ahead"
    if l == 2:
        return "Danger on your left"
    if r == 2:
        return "Danger on your right"
    if c == 1:
        return "Obstacle ahead"
    if l == 1:
        return "Obstacle on your left"
    if r == 1:
        return "Obstacle on your right"
    return "Clear path"


def main() -> int:
    p = argparse.ArgumentParser(description="Pico B relay monitor")
    p.add_argument("--port", required=True, help="Serial port, e.g. /dev/tty.usbmodemXXXX")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--voice", action="store_true", help="Enable speech output")
    p.add_argument("--min-repeat-sec", type=float, default=1.0, help="Minimum repeat interval for same message")
    args = p.parse_args()

    tts = None
    if args.voice and pyttsx3 is not None:
        tts = pyttsx3.init()
        tts.setProperty("rate", 170)
    elif args.voice:
        print("[Warn] pyttsx3 unavailable; running text-only")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.2)
    except Exception as exc:
        print(f"[Error] Cannot open serial port {args.port}: {exc}")
        return 1

    print(f"[RelayMonitor] Listening on {args.port} @ {args.baud}")
    last_msg = ""
    last_t = 0.0

    try:
        while True:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", "ignore").strip()
            if not line:
                continue

            pkt = parse_packet(line)
            if pkt is None:
                print(f"[RAW] {line}")
                continue

            msg = choose_message(pkt)
            now = time.monotonic()
            print(f"[STATE] {line} -> {msg}")

            if msg != last_msg or (now - last_t) >= max(0.2, args.min_repeat_sec):
                last_msg = msg
                last_t = now
                if tts is not None:
                    tts.say(msg)
                    tts.runAndWait()
    except KeyboardInterrupt:
        print("\n[RelayMonitor] Stopped")
    finally:
        ser.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())


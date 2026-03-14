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
from phase_audio import PhaseToneEngine

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

PATTERN_3 = re.compile(r"L:(\d),C:(\d),R:(\d)")
PATTERN_1 = re.compile(r"C:(\d)")
PATTERN_DISTANCE = re.compile(r"Distance:\s*([0-9]+(?:\.[0-9]+)?)\s*cm", re.IGNORECASE)
PATTERN_NO_READING = re.compile(r"No reading", re.IGNORECASE)


def distance_to_state(distance_cm: float | None, warning_cm: float, danger_cm: float) -> int:
    if distance_cm is None:
        return 0
    if distance_cm <= danger_cm:
        return 2
    if distance_cm <= warning_cm:
        return 1
    return 0


def parse_packet(line: str, warning_cm: float, danger_cm: float):
    line = line.strip()
    if line == "NO_SIGNAL":
        return {"type": "no_signal"}
    m3 = PATTERN_3.fullmatch(line)
    if m3:
        return {"type": "state3", "L": int(m3.group(1)), "C": int(m3.group(2)), "R": int(m3.group(3))}
    m1 = PATTERN_1.fullmatch(line)
    if m1:
        return {"type": "state1", "C": int(m1.group(1))}
    md = PATTERN_DISTANCE.search(line)
    if md:
        d = float(md.group(1))
        return {
            "type": "state1",
            "C": distance_to_state(d, warning_cm=warning_cm, danger_cm=danger_cm),
            "distance_cm": d,
            "source": "arduino_distance",
        }
    if PATTERN_NO_READING.search(line):
        return {"type": "state1", "C": 0, "source": "arduino_distance"}
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


def choose_phase(pkt: dict) -> str:
    if pkt["type"] == "no_signal":
        return "SAFE"
    if pkt["type"] == "state1":
        c = int(pkt.get("C", 0))
        if c == 2:
            return "DANGER"
        if c == 1:
            return "WARNING"
        return "SAFE"
    l, c, r = int(pkt.get("L", 0)), int(pkt.get("C", 0)), int(pkt.get("R", 0))
    if 2 in (l, c, r):
        return "DANGER"
    if 1 in (l, c, r):
        return "WARNING"
    return "SAFE"


def main() -> int:
    p = argparse.ArgumentParser(description="Pico B relay monitor")
    p.add_argument("--port", required=True, help="Serial port, e.g. /dev/tty.usbmodemXXXX")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--voice", action="store_true", help="Enable speech output")
    p.add_argument("--phase-tones", action="store_true", help="Enable 3-phase proximity tones (louder+higher when closer)")
    p.add_argument("--min-repeat-sec", type=float, default=1.0, help="Minimum repeat interval for same message")
    p.add_argument("--warning-cm", type=float, default=100.0, help="Distance threshold (cm) for WARNING when parsing 'Distance: X cm'")
    p.add_argument("--danger-cm", type=float, default=30.0, help="Distance threshold (cm) for DANGER when parsing 'Distance: X cm'")
    args = p.parse_args()

    tts = None
    if args.voice and pyttsx3 is not None:
        tts = pyttsx3.init()
        tts.setProperty("rate", 170)
    elif args.voice:
        print("[Warn] pyttsx3 unavailable; running text-only")
    tone = PhaseToneEngine() if args.phase_tones else None

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

            pkt = parse_packet(line, warning_cm=args.warning_cm, danger_cm=args.danger_cm)
            if pkt is None:
                print(f"[RAW] {line}")
                continue

            msg = choose_message(pkt)
            phase = choose_phase(pkt)
            now = time.monotonic()
            extra = ""
            if "distance_cm" in pkt:
                extra = f" | distance={pkt['distance_cm']:.1f}cm"
            print(f"[STATE] {line} -> {msg} | phase={phase}{extra}")

            if tone is not None:
                tone.play_if_due(phase)

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

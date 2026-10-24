#!/usr/bin/env python3
"""
Single-sensor runtime with two Picos connected.

Goal:
- Auto-detect which Pico is the working sensor node.
- Read distance/state from that one Pico only.
- Ignore the second Pico even if connected.
- Keep running with reconnect logic when ports rename/reset.
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass

import serial
from serial.tools import list_ports

from phase_audio import PhaseToneEngine

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

RE_DISTANCE = re.compile(r"Distance:\s*([0-9]+(?:\.[0-9]+)?)\s*cm", re.IGNORECASE)
RE_DISTANCE_EQ = re.compile(r"distance\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*cm", re.IGNORECASE)
RE_DIST_C = re.compile(r"dist\s*C:\s*(None|[0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
RE_STATE_C = re.compile(r"\bC:([0-2])\b")
RE_NO_READING = re.compile(r"No reading", re.IGNORECASE)
RE_NO_SIGNAL = re.compile(r"NO_SIGNAL", re.IGNORECASE)


@dataclass
class Parsed:
    distance_cm: float | None
    phase: str | None


def list_usbmodem() -> list[str]:
    return sorted([p.device.strip() for p in list_ports.comports() if "usbmodem" in p.device])


def distance_to_phase(distance_cm: float | None, warning_cm: float, danger_cm: float) -> str:
    if distance_cm is None:
        return "SAFE"
    if distance_cm <= danger_cm:
        return "DANGER"
    if distance_cm <= warning_cm:
        return "WARNING"
    return "SAFE"


def parse_line(line: str, warning_cm: float, danger_cm: float) -> Parsed:
    s = (line or "").strip()
    if not s:
        return Parsed(distance_cm=None, phase=None)

    m = RE_DISTANCE.search(s)
    if m:
        d = float(m.group(1))
        return Parsed(distance_cm=d, phase=distance_to_phase(d, warning_cm, danger_cm))

    m = RE_DISTANCE_EQ.search(s)
    if m:
        d = float(m.group(1))
        return Parsed(distance_cm=d, phase=distance_to_phase(d, warning_cm, danger_cm))

    m = RE_DIST_C.search(s)
    if m:
        tok = m.group(1).strip().lower()
        if tok == "none":
            return Parsed(distance_cm=None, phase="SAFE")
        d = float(tok)
        return Parsed(distance_cm=d, phase=distance_to_phase(d, warning_cm, danger_cm))

    m = RE_STATE_C.search(s)
    if m:
        c = int(m.group(1))
        if c == 2:
            return Parsed(distance_cm=None, phase="DANGER")
        if c == 1:
            return Parsed(distance_cm=None, phase="WARNING")
        return Parsed(distance_cm=None, phase="SAFE")

    if RE_NO_READING.search(s):
        return Parsed(distance_cm=None, phase="SAFE")
    if RE_NO_SIGNAL.search(s):
        return Parsed(distance_cm=None, phase="SAFE")

    return Parsed(distance_cm=None, phase=None)


def probe_sensor_port(ports: list[str], baud: int, warning_cm: float, danger_cm: float, probe_sec: float = 1.2) -> tuple[str | None, dict[str, tuple[int, int]]]:
    """
    Returns (best_port, score_map)
    score_map[port] = (sensor_hits, total_lines)
    """
    score_map: dict[str, tuple[int, int]] = {}
    for port in ports:
        hits = 0
        total = 0
        try:
            ser = serial.Serial(port, baud, timeout=0.2)
        except Exception:
            score_map[port] = (0, 0)
            continue

        t_end = time.monotonic() + max(0.5, probe_sec)
        try:
            while time.monotonic() < t_end:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", "ignore").strip()
                if not line:
                    continue
                total += 1
                parsed = parse_line(line, warning_cm, danger_cm)
                if parsed.phase is not None:
                    hits += 1
        except Exception:
            pass
        finally:
            try:
                ser.close()
            except Exception:
                pass

        score_map[port] = (hits, total)

    if not score_map:
        return None, score_map

    ranked = sorted([(v[0], v[1], k) for k, v in score_map.items()], reverse=True)
    best_hits, _, best_port = ranked[0]
    if best_hits <= 0:
        return None, score_map
    return best_port, score_map


def choose_message(phase: str, distance_cm: float | None) -> str:
    if phase == "DANGER":
        return "Danger ahead"
    if phase == "WARNING":
        return "Obstacle ahead"
    return "Clear path"


def main() -> int:
    p = argparse.ArgumentParser(description="Run one working Pico sensor while two Picos are connected")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--warning-cm", type=float, default=120.0)
    p.add_argument("--danger-cm", type=float, default=50.0)
    p.add_argument("--voice", action="store_true")
    p.add_argument("--tones", action="store_true")
    p.add_argument("--voice-min-repeat-sec", type=float, default=1.0)
    p.add_argument("--reprobe-sec", type=float, default=3.0, help="Re-probe interval when sensor stream is lost")
    args = p.parse_args()

    tone = PhaseToneEngine() if args.tones else None
    tts = None
    if args.voice and pyttsx3 is not None:
        tts = pyttsx3.init()
        tts.setProperty("rate", 170)
    elif args.voice:
        print("[Warn] pyttsx3 unavailable; running without voice")

    active_port: str | None = None
    ser = None
    last_probe_t = 0.0
    last_sensor_t = 0.0
    last_msg = ""
    last_msg_t = 0.0
    last_phase = "SAFE"
    last_dist: float | None = None

    print("[Runtime] starting single-Pico sensor runtime")
    try:
        while True:
            now = time.monotonic()
            needs_probe = (
                ser is None
                or active_port is None
                or (now - last_sensor_t) > max(1.0, args.reprobe_sec)
            )
            if needs_probe and (now - last_probe_t) >= max(0.8, args.reprobe_sec):
                last_probe_t = now
                ports = list_usbmodem()
                if ports:
                    best, scores = probe_sensor_port(
                        ports=ports,
                        baud=args.baud,
                        warning_cm=args.warning_cm,
                        danger_cm=args.danger_cm,
                    )
                    if best is not None and best != active_port:
                        if ser is not None:
                            try:
                                ser.close()
                            except Exception:
                                pass
                            ser = None
                        active_port = best
                        print(f"[Runtime] selected sensor Pico: {active_port}")
                        ignored = [x for x in ports if x != active_port]
                        if ignored:
                            print(f"[Runtime] ignoring other Pico port(s): {ignored}")
                    if active_port is None:
                        print(f"[Runtime] no sensor stream yet; port scores={scores}")
                else:
                    print("[Runtime] no usbmodem ports found")

            if active_port and ser is None:
                try:
                    ser = serial.Serial(active_port, args.baud, timeout=0.2)
                    print(f"[Runtime] listening on {active_port} @ {args.baud}")
                except Exception as exc:
                    print(f"[Runtime] open failed on {active_port}: {exc}")
                    ser = None
                    time.sleep(0.25)
                    continue

            if ser is None:
                time.sleep(0.1)
                continue

            try:
                raw = ser.readline()
            except Exception as exc:
                print(f"[Runtime] serial read error on {active_port}: {exc}")
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None
                time.sleep(0.2)
                continue

            if not raw:
                if tone is not None:
                    tone.play_if_due(last_phase)
                continue

            line = raw.decode("utf-8", "ignore").strip()
            parsed = parse_line(line, args.warning_cm, args.danger_cm)
            if parsed.phase is None:
                continue

            last_sensor_t = time.monotonic()
            last_phase = parsed.phase
            last_dist = parsed.distance_cm
            msg = choose_message(last_phase, last_dist)
            extra = "-" if last_dist is None else f"{last_dist:.1f}cm"
            print(f"[STATE] {line} -> {msg} | phase={last_phase} | distance={extra}")

            if tone is not None:
                tone.play_if_due(last_phase)

            now = time.monotonic()
            if tts is not None and (msg != last_msg or (now - last_msg_t) >= max(0.2, args.voice_min_repeat_sec)):
                last_msg = msg
                last_msg_t = now
                tts.say(msg)
                tts.runAndWait()

    except KeyboardInterrupt:
        print("\n[Runtime] stopped")
    finally:
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


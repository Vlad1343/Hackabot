#!/usr/bin/env python3
"""
Pico wall-alert beeper (sensor-first, no camera depth).

Reads serial packets from Pico B, e.g.:
  L:0,C:2,R:1
  C:1
  NO_SIGNAL

Outputs on laptop:
- 3-phase beeps (SAFE/WARNING/DANGER)
- Optional directional speech
- Stabilized state with debounce to reduce flicker
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import deque

import serial
from serial.tools import list_ports

from phase_audio import PhaseToneEngine

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

PATTERN_3 = re.compile(r"L:(\d),C:(\d),R:(\d)")
PATTERN_1 = re.compile(r"C:(\d)")
PATTERN_DISTANCE = re.compile(r"Distance:\s*([0-9]+(?:\.[0-9]+)?)\s*cm", re.IGNORECASE)
PATTERN_NO_READING = re.compile(r"No reading", re.IGNORECASE)

SAFE = 0
WARNING = 1
DANGER = 2


def distance_to_state(distance_cm: float | None, warning_cm: float, danger_cm: float) -> int:
    if distance_cm is None:
        return SAFE
    if distance_cm <= danger_cm:
        return DANGER
    if distance_cm <= warning_cm:
        return WARNING
    return SAFE


def parse_packet(line: str, warning_cm: float, danger_cm: float):
    s = line.strip()
    if s == "NO_SIGNAL":
        return {"type": "no_signal"}
    m3 = PATTERN_3.fullmatch(s)
    if m3:
        return {"type": "state3", "L": int(m3.group(1)), "C": int(m3.group(2)), "R": int(m3.group(3))}
    m1 = PATTERN_1.fullmatch(s)
    if m1:
        return {"type": "state1", "C": int(m1.group(1))}
    md = PATTERN_DISTANCE.search(s)
    if md:
        d = float(md.group(1))
        return {
            "type": "state1",
            "C": distance_to_state(d, warning_cm=warning_cm, danger_cm=danger_cm),
            "distance_cm": d,
            "source": "arduino_distance",
        }
    if PATTERN_NO_READING.search(s):
        return {"type": "state1", "C": SAFE, "source": "arduino_distance"}
    return None


def choose_phase(pkt: dict) -> str:
    if pkt["type"] == "no_signal":
        return "SAFE"
    if pkt["type"] == "state1":
        c = int(pkt.get("C", 0))
        if c == DANGER:
            return "DANGER"
        if c == WARNING:
            return "WARNING"
        return "SAFE"
    l, c, r = int(pkt.get("L", 0)), int(pkt.get("C", 0)), int(pkt.get("R", 0))
    if DANGER in (l, c, r):
        return "DANGER"
    if WARNING in (l, c, r):
        return "WARNING"
    return "SAFE"


def choose_message(pkt: dict) -> str:
    if pkt["type"] == "no_signal":
        return "No signal from sensor node"
    if pkt["type"] == "state1":
        c = int(pkt.get("C", 0))
        if c == DANGER:
            return "Danger ahead"
        if c == WARNING:
            return "Obstacle ahead"
        return "Clear path"

    l, c, r = int(pkt.get("L", 0)), int(pkt.get("C", 0)), int(pkt.get("R", 0))
    # Priority: front danger > side danger > front warning > side warning
    if c == DANGER:
        return "Danger ahead"
    if l == DANGER:
        return "Danger on your left"
    if r == DANGER:
        return "Danger on your right"
    if c == WARNING:
        return "Obstacle ahead"
    if l == WARNING:
        return "Obstacle on your left"
    if r == WARNING:
        return "Obstacle on your right"
    return "Clear path"


def stable_vote(values: deque[str]) -> str:
    if not values:
        return "SAFE"
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    # deterministic tie-break SAFE < WARNING < DANGER priority by risk
    order = {"SAFE": 0, "WARNING": 1, "DANGER": 2}
    best = max(counts.items(), key=lambda kv: (kv[1], order.get(kv[0], 0)))
    return best[0]


def _is_sensor_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if PATTERN_DISTANCE.search(s):
        return True
    if PATTERN_NO_READING.search(s):
        return True
    if PATTERN_3.fullmatch(s) or PATTERN_1.fullmatch(s):
        return True
    return False


def _probe_port_for_sensor(port: str, baud: int, probe_sec: float = 1.1) -> tuple[int, int]:
    """
    Returns (sensor_hits, total_lines).
    Higher sensor_hits means this port looks like Pico sensor output.
    """
    sensor_hits = 0
    total_lines = 0
    try:
        ser = serial.Serial(port, baud, timeout=0.2)
    except Exception:
        return (0, 0)
    t_end = time.monotonic() + max(0.4, probe_sec)
    try:
        while time.monotonic() < t_end:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", "ignore").strip()
            if not line:
                continue
            total_lines += 1
            if _is_sensor_line(line):
                sensor_hits += 1
    except Exception:
        pass
    finally:
        try:
            ser.close()
        except Exception:
            pass
    return (sensor_hits, total_lines)


def resolve_port(port_arg: str, baud: int) -> str | None:
    if port_arg != "auto":
        return port_arg
    ports = sorted(p.device for p in list_ports.comports() if "usbmodem" in p.device)
    if not ports:
        return None
    # Pick port that actually emits sensor/state lines.
    scored: list[tuple[int, int, str]] = []
    for p in ports:
        hits, lines = _probe_port_for_sensor(p, baud=baud)
        scored.append((hits, lines, p))
    scored.sort(reverse=True)
    best_hits, best_lines, best_port = scored[0]
    if best_hits > 0:
        print(f"[PicoWall] auto-selected sensor port: {best_port} (hits={best_hits}, lines={best_lines})")
        return best_port
    # Fallback if no hits during probe: pick last enumerated
    fallback = ports[-1]
    print(f"[PicoWall] auto-selected fallback port: {fallback} (no sensor hits yet)")
    return fallback


def main() -> int:
    p = argparse.ArgumentParser(description="Pico wall beeper (ultrasonic -> serial -> beeps)")
    p.add_argument("--port", required=True, help="Serial port, e.g. /dev/cu.usbmodemXXXX, or auto")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--voice", action="store_true", help="Enable speech output")
    p.add_argument("--tones", action="store_true", help="Enable phase beeps")
    p.add_argument("--voice-min-repeat-sec", type=float, default=1.0, help="Minimum time before repeating same voice")
    p.add_argument("--phase-window", type=int, default=3, help="Window size for phase majority smoothing")
    p.add_argument("--warning-cm", type=float, default=100.0, help="Distance threshold (cm) for WARNING when parsing 'Distance: X cm'")
    p.add_argument("--danger-cm", type=float, default=30.0, help="Distance threshold (cm) for DANGER when parsing 'Distance: X cm'")
    p.add_argument("--signal-hold-sec", type=float, default=1.2, help="How long to keep last phase without fresh packets")
    p.add_argument("--beep-safe", action="store_true", help="Also play SAFE tone (default: off)")
    p.add_argument("--danger-burst-count", type=int, default=3, help="Extra immediate beeps for each DANGER packet")
    p.add_argument("--danger-burst-gap-sec", type=float, default=0.08, help="Gap between extra DANGER beeps")
    args = p.parse_args()

    tts = None
    if args.voice and pyttsx3 is not None:
        tts = pyttsx3.init()
        tts.setProperty("rate", 170)
    elif args.voice:
        print("[Warn] pyttsx3 unavailable; running without voice")

    tone = PhaseToneEngine() if args.tones else None

    phase_hist: deque[str] = deque(maxlen=max(3, args.phase_window))
    last_voice_msg = ""
    last_voice_t = 0.0
    last_rx_t = time.monotonic()
    current_phase = "SAFE"
    burst_gap = max(0.02, args.danger_burst_gap_sec)
    ser = None
    connected_port = ""
    next_reconnect_t = 0.0

    print(f"[PicoWall] Starting serial reader on {args.port} @ {args.baud}")
    try:
        while True:
            if ser is None:
                now = time.monotonic()
                if now < next_reconnect_t:
                    time.sleep(0.05)
                    continue
                port = resolve_port(args.port, baud=args.baud)
                if not port:
                    print("[PicoWall] No usbmodem port found, retrying...")
                    next_reconnect_t = now + 1.0
                    continue
                try:
                    ser = serial.Serial(port, args.baud, timeout=0.05)
                    connected_port = port
                    print(f"[PicoWall] Listening on {connected_port} @ {args.baud}")
                except Exception as exc:
                    print(f"[PicoWall] Open failed on {port}: {exc}")
                    ser = None
                    next_reconnect_t = now + 1.0
                    continue

            try:
                raw = ser.readline()
            except Exception as exc:
                print(f"[PicoWall] Serial read error on {connected_port}: {exc}. Reconnecting...")
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None
                connected_port = ""
                next_reconnect_t = time.monotonic() + 0.8
                continue

            if not raw:
                # If no data for too long, degrade to safe tone.
                age = time.monotonic() - last_rx_t
                if age > max(0.1, args.signal_hold_sec):
                    phase_hist.append("SAFE")
                    current_phase = stable_vote(phase_hist)
                # Keep beeping based on current stable phase even between packets.
                if tone is not None and (current_phase != "SAFE" or args.beep_safe):
                    tone.play_if_due(current_phase)
                continue

            line = raw.decode("utf-8", "ignore").strip()
            if not line:
                continue
            pkt = parse_packet(line, warning_cm=args.warning_cm, danger_cm=args.danger_cm)
            if pkt is None:
                print(f"[RAW] {line}")
                continue

            last_rx_t = time.monotonic()
            phase_hist.append(choose_phase(pkt))
            current_phase = stable_vote(phase_hist)
            msg = choose_message(pkt)
            extra = ""
            if "distance_cm" in pkt:
                extra = f" | distance={pkt['distance_cm']:.1f}cm"
            print(f"[STATE] {line} -> {msg} | phase={current_phase}{extra}")

            if tone is not None and (current_phase != "SAFE" or args.beep_safe):
                # Manual burst on each danger packet (requested behavior):
                # one packet can produce multiple quick beeps.
                if current_phase == "DANGER" and args.danger_burst_count > 0:
                    tone.play_force("DANGER")
                    for _ in range(max(0, args.danger_burst_count - 1)):
                        time.sleep(burst_gap)
                        tone.play_force("DANGER")
                else:
                    tone.play_if_due(current_phase)

            now = time.monotonic()
            if tts is not None and (msg != last_voice_msg or (now - last_voice_t) >= max(0.2, args.voice_min_repeat_sec)):
                last_voice_msg = msg
                last_voice_t = now
                tts.say(msg)
                tts.runAndWait()
    except KeyboardInterrupt:
        print("\n[PicoWall] Stopped")
    finally:
        if ser is not None:
            ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

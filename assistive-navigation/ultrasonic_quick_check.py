#!/usr/bin/env python3
"""
Quick HC-SR04 distance checker.

Usage examples:
1) Raspberry Pi (real sensor):
   python3 ultrasonic_quick_check.py --sensor front:23:24 --sensor left:17:27

2) Mock mode on laptop:
   python3 ultrasonic_quick_check.py --mock --sensor front:23:24

Format for --sensor:
  name:TRIG_PIN:ECHO_PIN
"""
from __future__ import annotations

import argparse
import random
import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class SensorPins:
    name: str
    trig: int
    echo: int


def parse_sensor_spec(spec: str) -> SensorPins:
    try:
        name, trig_s, echo_s = spec.split(":")
        return SensorPins(name=name.strip().lower(), trig=int(trig_s), echo=int(echo_s))
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --sensor '{spec}'. Expected format name:TRIG:ECHO (example: front:23:24)"
        ) from exc


class MockHCSR04:
    def __init__(self, names: list[str]) -> None:
        self.values = {n: random.uniform(40, 160) for n in names}

    def read_cm(self, name: str, timeout_s: float = 0.03) -> Optional[float]:
        _ = timeout_s
        v = self.values[name] + random.uniform(-7.0, 7.0)
        self.values[name] = max(8.0, min(250.0, v))
        if random.random() < 0.04:
            return None
        return self.values[name]

    def cleanup(self) -> None:
        return


class RaspberryPiHCSR04:
    def __init__(self, sensors: list[SensorPins]) -> None:
        try:
            import RPi.GPIO as GPIO  # type: ignore
        except Exception as exc:
            raise RuntimeError("RPi.GPIO is not available. Use --mock on non-Pi machines.") from exc

        self.GPIO = GPIO
        self.GPIO.setmode(GPIO.BCM)
        self.GPIO.setwarnings(False)
        self.sensors: Dict[str, SensorPins] = {s.name: s for s in sensors}

        for s in sensors:
            GPIO.setup(s.trig, GPIO.OUT)
            GPIO.setup(s.echo, GPIO.IN)
            GPIO.output(s.trig, GPIO.LOW)
        time.sleep(0.05)

    def read_cm(self, name: str, timeout_s: float = 0.03) -> Optional[float]:
        s = self.sensors[name]
        GPIO = self.GPIO

        GPIO.output(s.trig, GPIO.LOW)
        time.sleep(0.000002)
        GPIO.output(s.trig, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(s.trig, GPIO.LOW)

        start_wait = time.perf_counter()
        while GPIO.input(s.echo) == 0:
            if time.perf_counter() - start_wait > timeout_s:
                return None
        pulse_start = time.perf_counter()

        while GPIO.input(s.echo) == 1:
            if time.perf_counter() - pulse_start > timeout_s:
                return None
        pulse_end = time.perf_counter()

        pulse_s = max(0.0, pulse_end - pulse_start)
        distance_cm = (pulse_s * 34300.0) / 2.0
        if distance_cm <= 0.0 or distance_cm > 500.0:
            return None
        return distance_cm

    def cleanup(self) -> None:
        self.GPIO.cleanup()


def classify(distance_cm: Optional[float], danger_cm: float, safe_cm: float) -> str:
    if distance_cm is None:
        return "NO_READING"
    if distance_cm < danger_cm:
        return "DANGER"
    if distance_cm < safe_cm:
        return "WARNING"
    return "SAFE"


def main() -> None:
    p = argparse.ArgumentParser(description="Quick HC-SR04 checker")
    p.add_argument(
        "--sensor",
        type=parse_sensor_spec,
        action="append",
        required=True,
        help="Sensor config: name:TRIG:ECHO (can be repeated)",
    )
    p.add_argument("--interval", type=float, default=0.2, help="Read interval seconds")
    p.add_argument("--samples", type=int, default=3, help="Samples per sensor per tick (median)")
    p.add_argument("--danger-cm", type=float, default=30.0, help="DANGER threshold")
    p.add_argument("--safe-cm", type=float, default=100.0, help="SAFE threshold")
    p.add_argument("--timeout-ms", type=float, default=30.0, help="Echo timeout per sample")
    p.add_argument("--mock", action="store_true", help="Use fake readings (for laptop testing)")
    args = p.parse_args()

    sensors: list[SensorPins] = args.sensor
    names = [s.name for s in sensors]
    timeout_s = max(0.005, args.timeout_ms / 1000.0)

    if args.mock:
        driver = MockHCSR04(names)
        print("[UltrasonicCheck] Running in MOCK mode")
    else:
        driver = RaspberryPiHCSR04(sensors)
        print("[UltrasonicCheck] Running with RPi.GPIO backend")

    histories: Dict[str, deque] = {n: deque(maxlen=3) for n in names}
    print("[UltrasonicCheck] Ctrl+C to stop")

    try:
        while True:
            row = []
            for name in names:
                readings = []
                for _ in range(max(1, args.samples)):
                    v = driver.read_cm(name, timeout_s=timeout_s)
                    if v is not None:
                        readings.append(v)
                filtered = statistics.median(readings) if readings else None
                if filtered is not None:
                    histories[name].append(filtered)
                smoothed = statistics.median(histories[name]) if histories[name] else None
                state = classify(smoothed, danger_cm=args.danger_cm, safe_cm=args.safe_cm)
                d_text = f"{smoothed:6.1f}cm" if smoothed is not None else "  None "
                row.append(f"{name.upper():>6}: {d_text}  {state}")

            print(" | ".join(row))
            time.sleep(max(0.02, args.interval))
    except KeyboardInterrupt:
        print("\n[UltrasonicCheck] Stopped")
    finally:
        driver.cleanup()


if __name__ == "__main__":
    main()


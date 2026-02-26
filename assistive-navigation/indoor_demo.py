#!/usr/bin/env python3
"""Indoor embedded obstacle system (hardened deterministic runtime)."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import random
import shutil
import socket
import statistics
import time
import traceback
from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

from hardware_config import HARDWARE_CONFIG

SAFE = "SAFE"
WARNING = "WARNING"
DANGER = "DANGER"
VALID_STATES = {SAFE, WARNING, DANGER}


def get_hardware_config() -> Dict[str, Any]:
    return deepcopy(HARDWARE_CONFIG)


def _is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.upper().startswith(("GPIO_", "SPI_", "I2C_"))


def _require_int_pin(name: str, value: Any) -> int:
    if _is_placeholder(value):
        raise ValueError(f"{name} is placeholder in hardware_config.py: {value}")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise ValueError(f"{name} must be integer pin in hardware_config.py, got: {value}")


def classify_distance(distance_cm: float, danger_cm: float, safe_cm: float) -> str:
    if distance_cm < danger_cm:
        return DANGER
    if distance_cm < safe_cm:
        return WARNING
    return SAFE


def fuse_sensor_states(sensor_states: Dict[str, str]) -> str:
    values = set(sensor_states.values())
    if DANGER in values:
        return DANGER
    if WARNING in values:
        return WARNING
    return SAFE


def nearest_direction(distances: Dict[str, float]) -> str:
    if not distances:
        return "FRONT"
    nearest_sensor = min(distances, key=distances.get)
    mapping = {"front": "FRONT", "left": "LEFT", "right": "RIGHT"}
    return mapping.get(nearest_sensor, "FRONT")


def zone_from_x_norm(x_norm: float, left_threshold: float, right_threshold: float) -> str:
    if x_norm < left_threshold:
        return "LEFT"
    if x_norm > right_threshold:
        return "RIGHT"
    return "FRONT"


def zone_from_centroid_score(x_norm: float, confidence: float, margin: float) -> str:
    x = max(0.0, min(1.0, float(x_norm)))
    conf = max(0.0, float(confidence))
    m = max(0.0, float(margin))
    left_score = conf * (1.0 - x)
    right_score = conf * x
    if left_score > (right_score + m):
        return "LEFT"
    if right_score > (left_score + m):
        return "RIGHT"
    return "FRONT"


class PeriodicScheduler:
    def __init__(self, hz: float) -> None:
        self.interval = 1.0 / max(0.1, hz)
        now = time.monotonic()
        self.next_ts = now

    def due(self, now: float) -> bool:
        return now >= self.next_ts

    def advance(self, now: float) -> None:
        self.next_ts += self.interval
        if self.next_ts < now:
            missed = int((now - self.next_ts) / self.interval) + 1
            self.next_ts += missed * self.interval


def _sleep_to_next(now: float, schedulers: List[PeriodicScheduler], max_sleep: float = 0.02) -> float:
    soonest = min(s.next_ts for s in schedulers)
    dt = soonest - now
    if dt <= 0:
        return 0.0
    return min(max_sleep, dt)


class StateStabilizer:
    """Hysteresis + debounce for SAFE/WARNING/DANGER stability."""

    def __init__(self, hysteresis_cm: float = 5.0, debounce_ticks: int = 2) -> None:
        self.hysteresis_cm = max(0.0, hysteresis_cm)
        self.debounce_ticks = max(1, debounce_ticks)
        self.state = SAFE
        self._pending = SAFE
        self._pending_count = 0

    def update(self, distance_cm: Optional[float], danger_cm: float, safe_cm: float) -> str:
        if distance_cm is None:
            candidate = SAFE
        else:
            d = float(distance_cm)
            if self.state == DANGER:
                if d < (danger_cm + self.hysteresis_cm):
                    candidate = DANGER
                elif d < safe_cm:
                    candidate = WARNING
                else:
                    candidate = SAFE
            elif self.state == WARNING:
                if d < danger_cm:
                    candidate = DANGER
                elif d >= (safe_cm + self.hysteresis_cm):
                    candidate = SAFE
                else:
                    candidate = WARNING
            else:
                if d < danger_cm:
                    candidate = DANGER
                elif d < safe_cm:
                    candidate = WARNING
                else:
                    candidate = SAFE

        if candidate != self._pending:
            self._pending = candidate
            self._pending_count = 1
        else:
            self._pending_count += 1

        if self._pending != self.state and self._pending_count >= self.debounce_ticks:
            self.state = self._pending

        return self.state


@dataclass
class IndoorPacket:
    state: str
    distance: Optional[float] = None
    distances: Optional[Dict[str, float]] = None

    def to_wire(self) -> Dict[str, object]:
        payload: Dict[str, object] = {"state": self.state}
        if self.distances is not None:
            payload["distances"] = self.distances
        else:
            payload["distance"] = self.distance
        return payload


class DistanceFilter:
    """Median/average of last 3 valid samples, rejects invalid spikes."""

    def __init__(self, max_range_cm: float = 400.0, window: int = 3, mode: str = "median") -> None:
        self.max_range_cm = max_range_cm
        self.window = max(3, window)
        self.mode = mode
        self.buffers: Dict[str, Deque[float]] = {}

    def update(self, sensor_name: str, raw: Optional[float]) -> Optional[float]:
        if raw is None:
            return self.current(sensor_name)

        value = float(raw)
        if value <= 0 or value > self.max_range_cm:
            return self.current(sensor_name)

        buf = self.buffers.setdefault(sensor_name, deque(maxlen=self.window))
        buf.append(value)
        return self.current(sensor_name)

    def current(self, sensor_name: str) -> Optional[float]:
        buf = self.buffers.get(sensor_name)
        if not buf:
            return None
        values = list(buf)
        if self.mode == "average":
            return float(sum(values) / len(values))
        return float(statistics.median(values))


class UltrasonicSensorProvider:
    async def read_distance_cm(self, sensor_name: str) -> Optional[float]:
        raise NotImplementedError


class MockUltrasonicSensorProvider(UltrasonicSensorProvider):
    def __init__(self) -> None:
        self.base = {"front": 140.0, "left": 160.0, "right": 155.0}

    async def read_distance_cm(self, sensor_name: str) -> Optional[float]:
        drift = random.uniform(-12, 12)
        noise = random.uniform(-4, 4)
        value = self.base.get(sensor_name, 150.0) + drift + noise
        if random.random() < 0.03:
            return 0.0
        self.base[sensor_name] = max(15.0, min(220.0, value))
        return self.base[sensor_name]


class HCSR04SensorProvider(UltrasonicSensorProvider):
    def __init__(self, sensor_cfg: Dict[str, Any]) -> None:
        self.sensor_cfg = sensor_cfg
        self.timeout_us = int(sensor_cfg.get("echo_timeout_us", 30000))
        self.read_timeout_sec = float(sensor_cfg.get("read_timeout_sec", 0.04))
        self.max_retries = int(sensor_cfg.get("read_retries", 2))
        self.backend = None
        self._pins: Dict[str, Dict[str, Any]] = {}
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            import machine  # type: ignore

            self.backend = "machine"
            for name in ("front", "left", "right"):
                if name not in self.sensor_cfg:
                    continue
                trig_pin = _require_int_pin(f"sensors.{name}.trig", self.sensor_cfg[name].get("trig"))
                echo_pin = _require_int_pin(f"sensors.{name}.echo", self.sensor_cfg[name].get("echo"))
                trig = machine.Pin(trig_pin, machine.Pin.OUT)
                echo = machine.Pin(echo_pin, machine.Pin.IN)
                trig.value(0)
                self._pins[name] = {"trig": trig, "echo": echo, "machine": machine}
            if not self._pins:
                raise ValueError("No HC-SR04 pins configured")
            return
        except Exception:
            pass

        try:
            import RPi.GPIO as GPIO  # type: ignore

            self.backend = "rpi_gpio"
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            for name in ("front", "left", "right"):
                if name not in self.sensor_cfg:
                    continue
                trig_pin = _require_int_pin(f"sensors.{name}.trig", self.sensor_cfg[name].get("trig"))
                echo_pin = _require_int_pin(f"sensors.{name}.echo", self.sensor_cfg[name].get("echo"))
                GPIO.setup(trig_pin, GPIO.OUT)
                GPIO.setup(echo_pin, GPIO.IN)
                GPIO.output(trig_pin, False)
                self._pins[name] = {"trig": trig_pin, "echo": echo_pin, "GPIO": GPIO}
            if not self._pins:
                raise ValueError("No HC-SR04 pins configured")
            return
        except Exception as exc:
            raise RuntimeError("HC-SR04 backend selected but GPIO runtime unavailable") from exc

    def _read_machine_once(self, sensor_name: str) -> Optional[float]:
        p = self._pins.get(sensor_name)
        if not p:
            return None
        trig = p["trig"]
        echo = p["echo"]
        machine = p["machine"]

        trig.value(0)
        time.sleep(0.000002)
        trig.value(1)
        time.sleep(0.00001)
        trig.value(0)

        try:
            pulse_us = machine.time_pulse_us(echo, 1, self.timeout_us)
        except Exception:
            return None
        if pulse_us <= 0:
            return None
        return (pulse_us * 0.0343) / 2.0

    def _read_rpi_once(self, sensor_name: str) -> Optional[float]:
        p = self._pins.get(sensor_name)
        if not p:
            return None

        GPIO = p["GPIO"]
        trig = p["trig"]
        echo = p["echo"]

        GPIO.output(trig, True)
        time.sleep(0.00001)
        GPIO.output(trig, False)

        t0 = time.perf_counter()
        timeout_s = self.timeout_us / 1_000_000.0

        while GPIO.input(echo) == 0:
            if time.perf_counter() - t0 > timeout_s:
                return None

        start = time.perf_counter()
        while GPIO.input(echo) == 1:
            if time.perf_counter() - start > timeout_s:
                return None

        duration = time.perf_counter() - start
        return (duration * 34300.0) / 2.0

    async def _read_once(self, sensor_name: str) -> Optional[float]:
        if self.backend == "machine":
            return await asyncio.wait_for(asyncio.to_thread(self._read_machine_once, sensor_name), timeout=self.read_timeout_sec)
        if self.backend == "rpi_gpio":
            return await asyncio.wait_for(asyncio.to_thread(self._read_rpi_once, sensor_name), timeout=self.read_timeout_sec)
        return None

    async def read_distance_cm(self, sensor_name: str) -> Optional[float]:
        for _ in range(max(1, self.max_retries)):
            try:
                v = await self._read_once(sensor_name)
            except Exception:
                v = None
            if v is not None and v > 0:
                return v
        return None


class RadioTX:
    async def send(self, packet: Dict[str, object]) -> None:
        raise NotImplementedError


class RadioRX:
    async def recv(self) -> Optional[Dict[str, object]]:
        raise NotImplementedError


class InMemoryRadioTX(RadioTX):
    def __init__(self, queue: asyncio.Queue) -> None:
        self.queue = queue

    async def send(self, packet: Dict[str, object]) -> None:
        try:
            self.queue.put_nowait(packet)
        except asyncio.QueueFull:
            pass


class InMemoryRadioRX(RadioRX):
    def __init__(self, queue: asyncio.Queue) -> None:
        self.queue = queue

    async def recv(self) -> Optional[Dict[str, object]]:
        try:
            return self.queue.get_nowait()
        except asyncio.QueueEmpty:
            return None


class _NRF24Adapter:
    def __init__(self, radio_cfg: Dict[str, Any], role: str) -> None:
        self.role = role
        self.cfg = radio_cfg
        self.backend = None
        self.radio = None
        self._init_backend()

    @staticmethod
    def _pipe_bytes(value: Any) -> bytes:
        if isinstance(value, bytes):
            return value
        s = str(value)
        if s.startswith("0x"):
            s = s[2:]
        s = s.zfill(10)
        return bytes.fromhex(s)

    def _init_micropython(self) -> bool:
        try:
            import machine  # type: ignore
            import nrf24l01  # type: ignore

            spi_id = int(self.cfg.get("spi_bus", 0))
            sck = _require_int_pin("radio.sck_pin", self.cfg.get("sck_pin"))
            mosi = _require_int_pin("radio.mosi_pin", self.cfg.get("mosi_pin"))
            miso = _require_int_pin("radio.miso_pin", self.cfg.get("miso_pin"))
            ce = _require_int_pin("radio.ce_pin", self.cfg.get("ce_pin"))
            csn = _require_int_pin("radio.csn_pin", self.cfg.get("csn_pin"))

            spi = machine.SPI(
                spi_id,
                baudrate=int(self.cfg.get("spi_baudrate", 1_000_000)),
                polarity=0,
                phase=0,
                sck=machine.Pin(sck),
                mosi=machine.Pin(mosi),
                miso=machine.Pin(miso),
            )
            nrf = nrf24l01.NRF24L01(
                spi,
                machine.Pin(csn, machine.Pin.OUT),
                machine.Pin(ce, machine.Pin.OUT),
                payload_size=int(self.cfg.get("payload_size", 32)),
            )

            if hasattr(nrf, "set_channel"):
                nrf.set_channel(int(self.cfg.get("channel", 76)))

            tx_pipe = self._pipe_bytes(self.cfg.get("tx_pipe", "0xE8E8F0F0E1"))
            rx_pipe = self._pipe_bytes(self.cfg.get("rx_pipe", "0xE8E8F0F0D2"))
            if self.role == "tx":
                nrf.open_tx_pipe(tx_pipe)
            else:
                nrf.open_rx_pipe(1, rx_pipe)
                nrf.start_listening()

            self.backend = "micropython"
            self.radio = nrf
            return True
        except Exception:
            return False

    def _init_rf24_linux(self) -> bool:
        try:
            from RF24 import RF24  # type: ignore

            ce = _require_int_pin("radio.ce_pin", self.cfg.get("ce_pin"))
            csn = _require_int_pin("radio.csn_pin", self.cfg.get("csn_pin"))
            radio = RF24(ce, csn)
            if not radio.begin():
                return False
            radio.setChannel(int(self.cfg.get("channel", 76)))
            radio.setPALevel(int(self.cfg.get("pa_level", 1)))

            tx_pipe_i = int.from_bytes(self._pipe_bytes(self.cfg.get("tx_pipe", "0xE8E8F0F0E1")), "little")
            rx_pipe_i = int.from_bytes(self._pipe_bytes(self.cfg.get("rx_pipe", "0xE8E8F0F0D2")), "little")
            if self.role == "tx":
                radio.openWritingPipe(tx_pipe_i)
                radio.stopListening()
            else:
                radio.openReadingPipe(1, rx_pipe_i)
                radio.startListening()

            self.backend = "rf24"
            self.radio = radio
            return True
        except Exception:
            return False

    def _init_backend(self) -> None:
        if self._init_micropython() or self._init_rf24_linux():
            return
        raise RuntimeError("nRF24 backend selected but no supported driver found")

    def send_json(self, packet: Dict[str, object]) -> bool:
        payload = json.dumps(packet, separators=(",", ":")).encode("utf-8")
        max_payload = int(self.cfg.get("payload_size", 32))
        if len(payload) > max_payload:
            return False

        if self.backend == "micropython":
            if hasattr(self.radio, "stop_listening"):
                self.radio.stop_listening()
            self.radio.send(payload)
            return True

        if self.backend == "rf24":
            self.radio.stopListening()
            return bool(self.radio.write(payload))

        return False

    def recv_json(self) -> Optional[Dict[str, object]]:
        raw: Optional[bytes] = None

        if self.backend == "micropython":
            if hasattr(self.radio, "any") and self.radio.any():
                raw = self.radio.recv()
        elif self.backend == "rf24":
            if self.radio.available():
                size = self.radio.getDynamicPayloadSize() if hasattr(self.radio, "getDynamicPayloadSize") else 32
                raw = self.radio.read(size)

        if not raw:
            return None
        try:
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else None
        except Exception:
            return None


class NRF24TX(RadioTX):
    def __init__(self, radio_cfg: Dict[str, Any]) -> None:
        self.adapter = _NRF24Adapter(radio_cfg=radio_cfg, role="tx")
        self.timeout_sec = float(radio_cfg.get("send_timeout_sec", 0.05))
        self.retries = int(radio_cfg.get("send_retries", 2))

    async def send(self, packet: Dict[str, object]) -> None:
        for _ in range(max(1, self.retries)):
            try:
                ok = await asyncio.wait_for(asyncio.to_thread(self.adapter.send_json, packet), timeout=self.timeout_sec)
                if ok:
                    return
            except Exception:
                pass
        raise RuntimeError("nRF24 send failed after retries")


class NRF24RX(RadioRX):
    def __init__(self, radio_cfg: Dict[str, Any]) -> None:
        self.adapter = _NRF24Adapter(radio_cfg=radio_cfg, role="rx")
        self.timeout_sec = float(radio_cfg.get("recv_timeout_sec", 0.02))

    async def recv(self) -> Optional[Dict[str, object]]:
        try:
            return await asyncio.wait_for(asyncio.to_thread(self.adapter.recv_json), timeout=self.timeout_sec)
        except Exception:
            return None


class SemanticEventProvider:
    async def get_events(self) -> List[Dict[str, Any]]:
        return []


class TemporalEventSmoother:
    def __init__(self, frames: int, confidence_threshold: float) -> None:
        self.frames = max(1, frames)
        self.confidence_threshold = confidence_threshold
        self.history: Deque[List[Dict[str, Any]]] = deque(maxlen=self.frames)

    def update(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        accepted = [e for e in events if float(e.get("confidence", 0.0)) >= self.confidence_threshold]
        self.history.append(accepted)

        counts: Counter[tuple] = Counter()
        best: Dict[tuple, Dict[str, Any]] = {}
        for frame_events in self.history:
            frame_seen = set()
            for e in frame_events:
                key = (str(e.get("label", "UNKNOWN")), str(e.get("zone", "FRONT")))
                if key in frame_seen:
                    continue
                frame_seen.add(key)
                counts[key] += 1
                prev = best.get(key)
                if prev is None or float(e.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
                    best[key] = dict(e)

        min_votes = max(1, (self.frames + 1) // 2)
        out = [best[k] for k, c in counts.items() if c >= min_votes]
        out.sort(key=lambda e: float(e.get("confidence", 0.0)), reverse=True)
        return out


class MockSemanticEventProvider(SemanticEventProvider):
    """Mock scenarios:
    - person left -> front -> right progression
    - flicker between zones
    - occasional disconnect (empty events)
    """

    def __init__(self, rate_hz: float = 2.0, confidence_threshold: float = 0.5, smoothing_frames: int = 3) -> None:
        self.tick = PeriodicScheduler(rate_hz)
        self.step = 0
        self.frame_id = 0
        self.latest: List[Dict[str, Any]] = []
        self.smoother = TemporalEventSmoother(smoothing_frames, confidence_threshold)

    async def get_events(self) -> List[Dict[str, Any]]:
        now = time.monotonic()
        if self.tick.due(now):
            self.tick.advance(now)
            phase = self.step % 12
            self.step += 1
            self.frame_id += 1
            ts_ms = int(time.time() * 1000)

            if phase in {0, 1, 2}:
                events = [{"label": "PERSON", "x_center_norm": 0.18, "y_center_norm": 0.55, "zone": "LEFT", "confidence": 0.82, "timestamp_ms": ts_ms, "frame_id": self.frame_id}]
            elif phase in {3, 4, 5}:
                events = [{"label": "PERSON", "x_center_norm": 0.50, "y_center_norm": 0.55, "zone": "FRONT", "confidence": 0.84, "timestamp_ms": ts_ms, "frame_id": self.frame_id}]
            elif phase in {6, 7, 8}:
                events = [{"label": "PERSON", "x_center_norm": 0.83, "y_center_norm": 0.55, "zone": "RIGHT", "confidence": 0.81, "timestamp_ms": ts_ms, "frame_id": self.frame_id}]
            elif phase == 9:
                # flicker
                x_norm = random.choice([0.31, 0.34, 0.66, 0.69])
                zone = zone_from_centroid_score(x_norm, 0.52, 0.15)
                events = [{"label": "OBSTACLE", "x_center_norm": x_norm, "y_center_norm": 0.60, "zone": zone, "confidence": 0.52, "timestamp_ms": ts_ms, "frame_id": self.frame_id}]
            elif phase == 10:
                # low confidence noise
                events = [{"label": "CHAIR", "x_center_norm": 0.48, "y_center_norm": 0.52, "zone": "FRONT", "confidence": 0.41, "timestamp_ms": ts_ms, "frame_id": self.frame_id}]
            else:
                # disconnect-like empty cycle
                events = []

            self.latest = self.smoother.update(events)

        return self.latest


class UDPYoloEventProvider(SemanticEventProvider):
    """Receives JSON packets with schema:
    {"objects":[{"label":"PERSON","x_center_norm":0.42,"y_center_norm":0.55,"zone":"LEFT|FRONT|RIGHT","confidence":0.78,"timestamp_ms":123,"frame_id":1}]}
    """

    def __init__(
        self,
        host: str,
        port: int,
        confidence_threshold: float,
        smoothing_frames: int,
        left_threshold: float,
        right_threshold: float,
        direction_margin: float,
        processing_flip: bool,
        multi_object_threshold: int,
        latest_frame_only: bool,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.left_threshold = left_threshold
        self.right_threshold = right_threshold
        self.direction_margin = direction_margin
        self.processing_flip = processing_flip
        self.multi_object_threshold = max(2, int(multi_object_threshold))
        self.latest_frame_only = latest_frame_only
        self._last_frame_id = -1
        self._stable_zone = "FRONT"
        self._candidate_zone = "FRONT"
        self._candidate_count = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.bind((self.host, self.port))
        self.latest: List[Dict[str, Any]] = []
        self.smoother = TemporalEventSmoother(smoothing_frames, confidence_threshold)

    def _normalize_obj(self, obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cls = str(obj.get("label", obj.get("class", "UNKNOWN"))).upper()
        conf = float(obj.get("confidence", 0.0))

        x_norm = obj.get("x_center_norm")
        y_norm = obj.get("y_center_norm")
        zone = str(obj.get("zone", "")).upper()
        frame_id = int(obj.get("frame_id", 0))
        timestamp_ms = int(obj.get("timestamp_ms", int(time.time() * 1000)))

        if x_norm is None:
            pos = str(obj.get("position", "front")).lower()
            x_norm = {"left": 0.16, "front": 0.5, "right": 0.84}.get(pos, 0.5)

        try:
            x_norm = float(x_norm)
        except Exception:
            x_norm = 0.5

        try:
            y_norm = float(y_norm) if y_norm is not None else 0.5
        except Exception:
            y_norm = 0.5

        # Processing frame must remain non-mirrored (geometry-safe).
        if self.processing_flip:
            # still allow config but undo any accidental mirrored input by flipping x back.
            x_norm = 1.0 - x_norm

        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))

        computed_zone = zone_from_centroid_score(x_norm, conf, self.direction_margin)
        if zone not in {"LEFT", "FRONT", "RIGHT"}:
            zone = computed_zone
        else:
            zone = computed_zone

        # Strict unified schema for vision events.
        return {
            "label": cls,
            "x_center_norm": round(x_norm, 4),
            "y_center_norm": round(y_norm, 4),
            "zone": zone,
            "confidence": round(conf, 3),
            "timestamp_ms": timestamp_ms,
            "frame_id": frame_id,
        }

    def _bbox_area(self, obj: Dict[str, Any]) -> float:
        bbox = obj.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox]
                return max(0.0, x2 - x1) * max(0.0, y2 - y1)
            except Exception:
                return 0.0
        return float(obj.get("bbox_area", 0.0) or 0.0)

    def _select_primary(self, objs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not objs:
            return None
        ranked = sorted(
            objs,
            key=lambda o: (
                -float(o.get("confidence", 0.0)),
                -self._bbox_area(o),
                abs(float(o.get("x_center_norm", 0.5)) - 0.5),
            ),
        )
        primary = dict(ranked[0])
        if len(objs) >= self.multi_object_threshold:
            primary["label"] = "MULTIPLE_OBJECTS"
        return primary

    def _stabilize_zone(self, zone: str) -> str:
        zone = zone if zone in {"LEFT", "FRONT", "RIGHT"} else "FRONT"
        if zone == self._stable_zone:
            self._candidate_zone = zone
            self._candidate_count = 0
            return self._stable_zone
        if zone == self._candidate_zone:
            self._candidate_count += 1
        else:
            self._candidate_zone = zone
            self._candidate_count = 1
        if self._candidate_count >= 2:
            self._stable_zone = zone
            self._candidate_count = 0
        return self._stable_zone

    async def get_events(self) -> List[Dict[str, Any]]:
        latest_data: Optional[bytes] = None
        try:
            while True:
                data, _ = self.sock.recvfrom(4096)
                latest_data = data
                if not self.latest_frame_only:
                    break
        except BlockingIOError:
            if latest_data is None:
                return self.latest
        except Exception:
            return self.latest

        try:
            payload = json.loads((latest_data or b"{}").decode("utf-8"))
            frame_id = int(payload.get("frame_id", 0))
            if self.latest_frame_only and frame_id <= self._last_frame_id:
                return self.latest
            if frame_id > 0:
                self._last_frame_id = frame_id
            objects = payload.get("objects", [])
            if isinstance(objects, list):
                normalized = []
                for obj in objects:
                    if not isinstance(obj, dict):
                        continue
                    n = self._normalize_obj(obj)
                    if n is not None:
                        normalized.append(n)
                primary = self._select_primary(normalized)
                if primary is None:
                    self.latest = []
                else:
                    primary["zone"] = self._stabilize_zone(str(primary.get("zone", "FRONT")))
                    self.latest = self.smoother.update([primary])
        except Exception:
            pass
        return self.latest


class IndoorAudioAnnouncer:
    def __init__(
        self,
        enabled: bool,
        folder: str,
        cooldown_ms: int,
        confidence_threshold: float,
    ) -> None:
        self.enabled = enabled
        self.folder = folder
        self.repeat_ms = max(1000, int(cooldown_ms))
        self.active_timeout_ms = 1500
        self.confidence_threshold = float(confidence_threshold)
        self.current_object: Optional[str] = None
        self.last_spoken_time = 0
        self.last_frame_id = -1
        self.active_object_timer = 0
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._file_map = {
            "PERSON": "person.mp3",
            "CHAIR": "chair.mp3",
            "TABLE": "table.mp3",
            "CAR": "car.mp3",
            "MULTIPLE_OBJECTS": "multiple_objects.mp3",
        }
        self._priority = {
            "PERSON": 4,
            "CHAIR": 3,
            "TABLE": 2,
            "MULTIPLE_OBJECTS": 1,
        }

    def _resolve_audio_path(self, key: str) -> Optional[str]:
        fn = self._file_map.get(key)
        if not fn:
            return None
        path = os.path.join(self.folder, fn)
        return path if os.path.exists(path) else None

    async def start(self) -> None:
        return

    async def stop(self) -> None:
        await self.interrupt()

    async def interrupt(self) -> None:
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=0.2)
            except Exception:
                self._proc.kill()
                with contextlib.suppress(Exception):
                    await self._proc.wait()
        self._proc = None

    async def _speak(self, key: str) -> None:
        path = self._resolve_audio_path(key)
        if path is None:
            print(f"[Audio] Missing file for {key} in {self.folder}")
            return

        await self.interrupt()
        player = None
        if shutil.which("afplay"):
            player = ["afplay", path]
        elif shutil.which("ffplay"):
            player = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path]
        elif shutil.which("mpg123"):
            player = ["mpg123", "-q", path]
        try:
            if player is None:
                print(f"[Audio] No player found; simulated: {key}")
                await asyncio.sleep(0.15)
            else:
                self._proc = await asyncio.create_subprocess_exec(
                    *player,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
        except Exception as exc:
            print(f"[Audio] playback error: {exc}")
            self._proc = None

    def _is_higher_priority(self, new_key: str, old_key: Optional[str]) -> bool:
        if old_key is None:
            return True
        return self._priority.get(new_key, 0) >= self._priority.get(old_key, 0)

    async def process(
        self,
        *,
        detected_key: Optional[str],
        frame_id: int,
        confidence: float,
        suppress: bool,
    ) -> None:
        if not self.enabled:
            return
        now_ms = int(time.time() * 1000)

        if suppress:
            await self.interrupt()
            self.current_object = None
            self.active_object_timer = 0
            return

        # Reset active state when object vanished.
        if self.current_object and (now_ms - self.active_object_timer) > self.active_timeout_ms:
            self.current_object = None

        key = detected_key.upper() if detected_key else None
        has_valid_detection = bool(key) and float(confidence) >= self.confidence_threshold
        has_new_frame = frame_id > self.last_frame_id

        if has_valid_detection and has_new_frame:
            self.last_frame_id = frame_id
            self.active_object_timer = now_ms
            if key != self.current_object:
                # New event interrupts immediately (single active channel rule).
                if key:
                    self.current_object = key
                    await self._speak(key)
                    self.last_spoken_time = now_ms
                    return
            else:
                # Same object, repeat every 1 second while persistent.
                if (now_ms - self.last_spoken_time) >= self.repeat_ms:
                    await self._speak(key)
                    self.last_spoken_time = now_ms
                    return

        # Continue repeating for persistent object while detections keep it active.
        if self.current_object and (now_ms - self.active_object_timer) <= self.active_timeout_ms:
            if (now_ms - self.last_spoken_time) >= self.repeat_ms:
                await self._speak(self.current_object)
                self.last_spoken_time = now_ms


class _SSD1306Adapter:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.display = None
        self._init_display()

    def _init_display(self) -> None:
        import machine  # type: ignore
        import ssd1306  # type: ignore

        i2c_bus = int(self.cfg.get("i2c_bus", 1))
        scl_pin = _require_int_pin("display.scl_pin", self.cfg.get("scl_pin"))
        sda_pin = _require_int_pin("display.sda_pin", self.cfg.get("sda_pin"))

        i2c = machine.I2C(
            i2c_bus,
            scl=machine.Pin(scl_pin),
            sda=machine.Pin(sda_pin),
            freq=int(self.cfg.get("i2c_freq", 400000)),
        )
        self.display = ssd1306.SSD1306_I2C(
            int(self.cfg.get("width", 128)),
            int(self.cfg.get("height", 64)),
            i2c,
            addr=int(str(self.cfg.get("i2c_address", "0x3C")), 16),
        )

    def render(self, line1: str, line2: str = "") -> None:
        d = self.display
        d.fill(0)
        d.text(line1[:16], 0, 0)
        if line2:
            d.text(line2[:16], 0, 16)
        d.show()


class ReceiverUI:
    def __init__(self, backend: str = "console", display_cfg: Optional[Dict[str, Any]] = None) -> None:
        self.backend = backend
        self.display_cfg = display_cfg or {}
        self._last_text: Optional[str] = None
        self._oled = None
        if self.backend == "ssd1306":
            try:
                self._oled = _SSD1306Adapter(self.display_cfg)
            except Exception as exc:
                print(f"[OLED] init failed, fallback console: {exc}")
                self.backend = "console"

    async def render(self, text: str, sub: str = "") -> None:
        if text == self._last_text:
            return
        if self.backend == "console":
            print(f"[OLED] {text}" + (f" | {sub}" if sub else ""))
        else:
            try:
                await asyncio.wait_for(asyncio.to_thread(self._oled.render, text, sub), timeout=0.03)
            except Exception as exc:
                print(f"[OLED] render failed: {exc}")
        self._last_text = text


class BuzzerController:
    def __init__(self, backend: str = "console", buzzer_cfg: Optional[Dict[str, Any]] = None) -> None:
        self.backend = backend
        self.cfg = buzzer_cfg or {}
        self.state = SAFE
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._pin_obj = None
        self._active_high = bool(self.cfg.get("active_high", True))
        self._init_backend()

    def _init_backend(self) -> None:
        if self.backend == "console":
            return
        if self.backend != "gpio":
            self.backend = "console"
            return

        try:
            import machine  # type: ignore

            pin = _require_int_pin("buzzer.pin", self.cfg.get("pin"))
            self._pin_obj = machine.Pin(pin, machine.Pin.OUT)
            self._write(False)
            return
        except Exception:
            pass

        try:
            import RPi.GPIO as GPIO  # type: ignore

            pin = _require_int_pin("buzzer.pin", self.cfg.get("pin"))
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin, GPIO.OUT)
            self._pin_obj = (GPIO, pin)
            self._write(False)
            return
        except Exception as exc:
            print(f"[BUZZER] init failed, fallback console: {exc}")
            self.backend = "console"
            self._pin_obj = None

    def _write(self, on: bool) -> None:
        if self.backend == "console":
            return
        level = 1 if (on == self._active_high) else 0
        if hasattr(self._pin_obj, "value"):
            self._pin_obj.value(level)
        elif isinstance(self._pin_obj, tuple):
            GPIO, pin = self._pin_obj
            GPIO.output(pin, GPIO.HIGH if level else GPIO.LOW)

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="buzzer-loop")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._write(False)

    def set_state(self, state: str) -> None:
        self.state = state if state in VALID_STATES else SAFE

    async def _loop(self) -> None:
        while self._running:
            if self.state == SAFE:
                self._write(False)
                await asyncio.sleep(0.05)
            elif self.state == WARNING:
                if self.backend == "console":
                    print("[BUZZER] BEEP")
                self._write(True)
                await asyncio.sleep(0.10)
                self._write(False)
                await asyncio.sleep(0.60)
            else:
                if self.backend == "console":
                    print("[BUZZER] BEEP BEEP")
                self._write(True)
                await asyncio.sleep(0.08)
                self._write(False)
                await asyncio.sleep(0.08)


def _validate_packet(packet: Dict[str, object]) -> bool:
    if not isinstance(packet, dict):
        return False
    state = packet.get("state")
    if state not in VALID_STATES:
        return False

    if "distance" in packet:
        try:
            float(packet["distance"])
            return True
        except Exception:
            return False

    if "distances" in packet and isinstance(packet["distances"], dict):
        try:
            for v in packet["distances"].values():
                float(v)
            return True
        except Exception:
            return False

    return False


def _debug(enabled: bool, event: str, **fields: Any) -> None:
    if not enabled:
        return
    payload = {"ts": round(time.monotonic(), 4), "event": event, **fields}
    print(f"[DEBUG] {json.dumps(payload, separators=(',', ':'))}")


def select_audio_event(
    events: List[Dict[str, Any]],
    multi_object_threshold: int,
    confidence_threshold: float,
) -> tuple[Optional[str], int, float]:
    valid = [e for e in events if float(e.get("confidence", 0.0)) >= confidence_threshold]
    if not valid:
        return None, -1, 0.0
    frame_id = max(int(e.get("frame_id", -1)) for e in valid)
    if len(valid) >= max(2, multi_object_threshold):
        conf = max(float(e.get("confidence", 0.0)) for e in valid)
        return "MULTIPLE_OBJECTS", frame_id, conf

    priority = {"PERSON": 4, "CHAIR": 3, "TABLE": 2, "MULTIPLE_OBJECTS": 1}
    best = sorted(
        valid,
        key=lambda e: (
            -priority.get(str(e.get("label", "")).upper(), 0),
            -float(e.get("confidence", 0.0)),
        ),
    )[0]
    return str(best.get("label", "")).upper(), int(best.get("frame_id", frame_id)), float(best.get("confidence", 0.0))


async def sensor_node_loop(
    sensor_provider: UltrasonicSensorProvider,
    tx: RadioTX,
    sensor_names: List[str],
    sensor_hz: float,
    radio_hz: float,
    danger_cm: float,
    safe_cm: float,
    filter_mode: str,
    max_range_cm: float,
    hysteresis_cm: float,
    debounce_ticks: int,
    raw_samples_per_tick: int,
    ema_alpha: float,
    debug_enabled: bool,
) -> None:
    filt = DistanceFilter(max_range_cm=max_range_cm, window=3, mode=filter_mode)
    stabilizers = {name: StateStabilizer(hysteresis_cm=hysteresis_cm, debounce_ticks=debounce_ticks) for name in sensor_names}
    ema_prev: Dict[str, float] = {}
    sensor_tick = PeriodicScheduler(sensor_hz)
    radio_tick = PeriodicScheduler(radio_hz)

    latest_packet = IndoorPacket(state=SAFE, distance=round(safe_cm, 1)).to_wire()

    while True:
        now = time.monotonic()

        if sensor_tick.due(now):
            filtered: Dict[str, float] = {}
            for name in sensor_names:
                for _ in range(max(1, raw_samples_per_tick)):
                    try:
                        raw = await sensor_provider.read_distance_cm(name)
                    except Exception:
                        raw = None
                    filt.update(name, raw)
                v = filt.current(name)
                if v is not None:
                    prev = ema_prev.get(name, v)
                    ema = (ema_alpha * v) + ((1.0 - ema_alpha) * prev)
                    ema_prev[name] = ema
                    filtered[name] = round(ema, 1)

            sensor_states: Dict[str, str] = {}
            for name in sensor_names:
                sensor_states[name] = stabilizers[name].update(filtered.get(name), danger_cm=danger_cm, safe_cm=safe_cm)
            state = fuse_sensor_states(sensor_states)

            if len(sensor_names) == 1:
                nearest = filtered.get(sensor_names[0])
                dist = round(nearest, 1) if nearest is not None else round(safe_cm, 1)
                latest_packet = IndoorPacket(state=state, distance=dist).to_wire()
            else:
                dist_map = filtered if filtered else {name: round(safe_cm, 1) for name in sensor_names}
                latest_packet = IndoorPacket(state=state, distances=dist_map).to_wire()

            _debug(debug_enabled, "sensor_tick", distances=filtered, sensor_states=sensor_states, state=state, packet=latest_packet)
            sensor_tick.advance(now)

        now = time.monotonic()
        if radio_tick.due(now):
            status = "ok"
            try:
                await tx.send(latest_packet)
            except Exception:
                status = "failed"
            _debug(debug_enabled, "radio_tx_tick", status=status, state=latest_packet.get("state"))
            radio_tick.advance(now)

        sleep_for = _sleep_to_next(time.monotonic(), [sensor_tick, radio_tick])
        await asyncio.sleep(sleep_for)


async def receiver_node_loop(
    rx: RadioRX,
    semantic_provider: Optional[SemanticEventProvider],
    display_backend: str,
    display_cfg: Dict[str, Any],
    buzzer_backend: str,
    buzzer_cfg: Dict[str, Any],
    radio_hz: float,
    ui_hz: float,
    no_signal_timeout_sec: float,
    max_event_age_ms: int,
    audio_enabled: bool,
    audio_folder: str,
    audio_cooldown_ms: int,
    multi_object_threshold: int,
    audio_confidence_threshold: float,
    debug_enabled: bool,
) -> None:
    ui = ReceiverUI(backend=display_backend, display_cfg=display_cfg)
    buzzer = BuzzerController(backend=buzzer_backend, buzzer_cfg=buzzer_cfg)
    announcer = IndoorAudioAnnouncer(
        enabled=audio_enabled,
        folder=audio_folder,
        cooldown_ms=audio_cooldown_ms,
        confidence_threshold=audio_confidence_threshold,
    )
    await buzzer.start()
    await announcer.start()

    radio_tick = PeriodicScheduler(radio_hz)
    ui_tick = PeriodicScheduler(ui_hz)

    last_packet_at = time.monotonic()
    latest_packet: Optional[Dict[str, object]] = None
    latest_state = SAFE
    latest_events: List[Dict[str, Any]] = []

    try:
        while True:
            now = time.monotonic()
            if radio_tick.due(now):
                packet = None
                try:
                    packet = await rx.recv()
                except Exception:
                    packet = None

                if packet is not None and _validate_packet(packet):
                    latest_packet = packet
                    latest_state = str(packet["state"])
                    last_packet_at = now
                    buzzer.set_state(latest_state)
                    if latest_state == DANGER:
                        await announcer.interrupt()
                        announcer.set_visible_key(None)
                    _debug(debug_enabled, "radio_rx_tick", status="ok", state=latest_state)
                elif packet is not None:
                    _debug(debug_enabled, "radio_rx_tick", status="invalid_packet")
                else:
                    _debug(debug_enabled, "radio_rx_tick", status="no_packet")

                radio_tick.advance(now)

            now = time.monotonic()
            if ui_tick.due(now):
                if semantic_provider is not None:
                    try:
                        latest_events = await semantic_provider.get_events()
                    except Exception:
                        latest_events = []
                now_ms = int(time.time() * 1000)
                latest_events = [
                    e
                    for e in latest_events
                    if isinstance(e, dict) and (now_ms - int(e.get("timestamp_ms", now_ms))) <= max_event_age_ms
                ]

                if now - last_packet_at >= no_signal_timeout_sec:
                    await announcer.process(detected_key=None, frame_id=-1, confidence=0.0, suppress=False)
                    await ui.render("NO SIGNAL")
                    buzzer.set_state(SAFE)
                    _debug(debug_enabled, "ui_tick", view="NO SIGNAL")
                else:
                    if latest_packet is None:
                        await ui.render("SAFE")
                    else:
                        # Ultrasonic direction always authoritative.
                        us_dir = nearest_direction(latest_packet.get("distances", {})) if "distances" in latest_packet else "FRONT"

                        if latest_state == DANGER:
                            await announcer.interrupt()
                            await announcer.process(detected_key=None, frame_id=-1, confidence=0.0, suppress=True)
                            text = f"DANGER {us_dir}"
                        elif latest_state == WARNING:
                            text = f"WARNING {us_dir}"
                            if latest_events:
                                # YOLO only adds semantic label.
                                top = max(latest_events, key=lambda x: float(x.get("confidence", 0.0)))
                                text = f"WARNING {str(top.get('label', 'OBJECT')).upper()} {us_dir}"
                        else:
                            text = "SAFE"

                        if "distance" in latest_packet:
                            sub = f"{latest_packet['distance']} cm"
                        else:
                            sub = str(latest_packet.get("distances", {}))
                        await ui.render(text, sub)
                        announce_key, announce_frame_id, announce_conf = select_audio_event(
                            latest_events,
                            multi_object_threshold=multi_object_threshold,
                            confidence_threshold=audio_confidence_threshold,
                        )
                        await announcer.process(
                            detected_key=announce_key if latest_state != DANGER else None,
                            frame_id=announce_frame_id,
                            confidence=announce_conf,
                            suppress=(latest_state == DANGER),
                        )

                        top_frame = int(latest_events[0].get("frame_id", -1)) if latest_events else -1
                        _debug(
                            debug_enabled,
                            "fusion_decision",
                            frame_id=top_frame,
                            timestamp_ms=int(time.time() * 1000),
                            state=latest_state,
                            view=text,
                            reason="ultrasonic_override" if latest_state in {WARNING, DANGER} else "vision_or_clear",
                            yolo_events=latest_events,
                            audio_key=announce_key,
                        )
                ui_tick.advance(now)

            sleep_for = _sleep_to_next(time.monotonic(), [radio_tick, ui_tick])
            await asyncio.sleep(sleep_for)
    finally:
        await announcer.stop()
        await buzzer.stop()


def get_arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


async def run(args: argparse.Namespace) -> None:
    hw = get_hardware_config()
    backend_cfg = hw.get("backend", {})
    sensor_cfg = hw.get("sensors", {})
    radio_cfg = hw.get("radio", {})
    display_cfg = hw.get("display", {})
    buzzer_cfg = hw.get("buzzer", {})
    thresholds = hw.get("thresholds_cm", {})
    runtime_cfg = hw.get("runtime", {})
    yolo_cfg = hw.get("yolo", {})
    direction_cfg = hw.get("direction", {})
    camera_cfg = hw.get("camera", {})
    vision_cfg = hw.get("vision", {})
    audio_cfg = hw.get("audio", {})

    role = str(get_arg(args, "role", "both")).lower()
    sensor_count = int(get_arg(args, "sensor_count", sensor_cfg.get("count", 1)))

    sensor_hz = float(runtime_cfg.get("sensor_hz", 10.0))
    radio_hz = float(get_arg(args, "tx_rate_hz", runtime_cfg.get("radio_hz", 10.0)))
    ui_hz = float(runtime_cfg.get("ui_hz", 10.0))

    danger_cm = float(get_arg(args, "danger_cm", thresholds.get("danger", 30.0)))
    safe_cm = float(get_arg(args, "safe_cm", thresholds.get("safe", 100.0)))
    filter_mode = str(get_arg(args, "filter_mode", "median")).lower()
    max_range_cm = float(get_arg(args, "max_range_cm", sensor_cfg.get("max_range_cm", 400.0)))
    use_hardware_radio = bool(get_arg(args, "use_hardware_radio", False))

    hysteresis_cm = float(runtime_cfg.get("state_hysteresis_cm", 5.0))
    debounce_ticks = int(runtime_cfg.get("state_debounce_ticks", 2))
    no_signal_timeout_sec = float(runtime_cfg.get("rx_no_signal_timeout_sec", 1.0))
    debug_enabled = bool(runtime_cfg.get("debug_enabled", False))
    max_event_age_ms = int(vision_cfg.get("max_event_age_ms", 1500))
    latest_frame_only = bool(vision_cfg.get("latest_frame_only", True))

    audio_enabled = bool(audio_cfg.get("enabled", True))
    audio_folder = str(audio_cfg.get("folder", "indoor_audio"))
    if not os.path.isabs(audio_folder):
        audio_folder = os.path.join(os.path.dirname(__file__), audio_folder)
    audio_cooldown_ms = int(audio_cfg.get("cooldown_ms", 1000))
    multi_object_threshold = int(audio_cfg.get("multi_object_threshold", 2))
    audio_confidence_threshold = float(audio_cfg.get("confidence_threshold", yolo_cfg.get("confidence_threshold", 0.5)))

    raw_samples_per_tick = int(sensor_cfg.get("raw_samples_per_tick", 3))
    ema_alpha = float(sensor_cfg.get("ema_alpha", 0.4))
    ema_alpha = max(0.0, min(1.0, ema_alpha))

    names = ["front", "left", "right"][: max(1, min(3, sensor_count))]

    sensor_backend = str(backend_cfg.get("sensor", "mock")).lower()
    if sensor_backend == "mock":
        sensor_provider: UltrasonicSensorProvider = MockUltrasonicSensorProvider()
    elif sensor_backend == "hc_sr04":
        sensor_provider = HCSR04SensorProvider(sensor_cfg=sensor_cfg)
    else:
        raise ValueError(f"Unknown sensor backend in hardware_config.py: {sensor_backend}")

    radio_backend = str(backend_cfg.get("radio", "mock")).lower()
    if use_hardware_radio or radio_backend == "nrf24":
        tx: RadioTX = NRF24TX(radio_cfg=radio_cfg)
        rx: RadioRX = NRF24RX(radio_cfg=radio_cfg)
    elif radio_backend == "mock":
        q: asyncio.Queue = asyncio.Queue(maxsize=32)
        tx = InMemoryRadioTX(q)
        rx = InMemoryRadioRX(q)
    else:
        raise ValueError(f"Unknown radio backend in hardware_config.py: {radio_backend}")

    semantic_provider: Optional[SemanticEventProvider] = None
    if bool(yolo_cfg.get("enabled", False)):
        yolo_backend = str(yolo_cfg.get("backend", "mock")).lower()
        confidence_threshold = float(yolo_cfg.get("confidence_threshold", 0.5))
        smoothing_frames = int(yolo_cfg.get("temporal_smoothing_frames", vision_cfg.get("smoothing_window", 3)))
        left_threshold = float(direction_cfg.get("left_threshold", 0.33))
        right_threshold = float(direction_cfg.get("right_threshold", 0.66))
        direction_margin = float(vision_cfg.get("direction_margin", 0.15))
        processing_flip = bool(camera_cfg.get("flip_for_processing", False))
        if processing_flip:
            print("[Vision] FRAME MIRRORING VIOLATION: inference pipeline configured as mirrored")

        if yolo_backend == "mock":
            semantic_provider = MockSemanticEventProvider(
                rate_hz=float(yolo_cfg.get("mock_rate_hz", 2.0)),
                confidence_threshold=confidence_threshold,
                smoothing_frames=smoothing_frames,
            )
        elif yolo_backend == "udp":
            semantic_provider = UDPYoloEventProvider(
                host=str(yolo_cfg.get("udp_host", "0.0.0.0")),
                port=int(yolo_cfg.get("udp_port", 5005)),
                confidence_threshold=confidence_threshold,
                smoothing_frames=smoothing_frames,
                left_threshold=left_threshold,
                right_threshold=right_threshold,
                direction_margin=direction_margin,
                processing_flip=processing_flip,
                multi_object_threshold=multi_object_threshold,
                latest_frame_only=latest_frame_only,
            )
        else:
            print(f"[YOLO] Unknown backend '{yolo_backend}', disabled")

    display_backend = str(backend_cfg.get("display", "console")).lower()
    buzzer_backend = str(backend_cfg.get("buzzer", "console")).lower()

    tasks: List[asyncio.Task] = []
    try:
        if role in {"sensor", "both"}:
            tasks.append(
                asyncio.create_task(
                    sensor_node_loop(
                        sensor_provider=sensor_provider,
                        tx=tx,
                        sensor_names=names,
                        sensor_hz=sensor_hz,
                        radio_hz=radio_hz,
                        danger_cm=danger_cm,
                        safe_cm=safe_cm,
                        filter_mode=filter_mode,
                        max_range_cm=max_range_cm,
                        hysteresis_cm=hysteresis_cm,
                        debounce_ticks=debounce_ticks,
                        raw_samples_per_tick=raw_samples_per_tick,
                        ema_alpha=ema_alpha,
                        debug_enabled=debug_enabled,
                    ),
                    name="indoor-sensor-node",
                )
            )

        if role in {"receiver", "both"}:
            tasks.append(
                asyncio.create_task(
                    receiver_node_loop(
                        rx=rx,
                        semantic_provider=semantic_provider,
                        display_backend=display_backend,
                        display_cfg=display_cfg,
                        buzzer_backend=buzzer_backend,
                        buzzer_cfg=buzzer_cfg,
                        radio_hz=radio_hz,
                        ui_hz=ui_hz,
                        no_signal_timeout_sec=no_signal_timeout_sec,
                        max_event_age_ms=max_event_age_ms,
                        audio_enabled=audio_enabled,
                        audio_folder=audio_folder,
                        audio_cooldown_ms=audio_cooldown_ms,
                        multi_object_threshold=multi_object_threshold,
                        audio_confidence_threshold=audio_confidence_threshold,
                        debug_enabled=debug_enabled,
                    ),
                    name="indoor-receiver-node",
                )
            )

        if not tasks:
            raise ValueError("role must be one of: sensor, receiver, both")

        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        raise
    except KeyboardInterrupt:
        print("[Indoor] Stopped by user")
    except Exception as exc:
        print(f"[Indoor] Runtime error: {exc}")
        traceback.print_exc()
    finally:
        for t in tasks:
            t.cancel()
        for t in tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Indoor embedded obstacle loop (sensor -> tx -> receiver)")
    p.add_argument("--role", choices=["sensor", "receiver", "both"], default="both", help="Run sensor node, receiver node, or both")
    p.add_argument("--sensor-count", type=int, default=1, help="1 to 3 ultrasonic sensors")
    p.add_argument("--tx-rate-hz", type=float, default=10.0, help="Radio TX/RX scheduler rate")
    p.add_argument("--danger-cm", type=float, default=30.0, help="Distance threshold for DANGER")
    p.add_argument("--safe-cm", type=float, default=100.0, help="Distance threshold for SAFE")
    p.add_argument("--max-range-cm", type=float, default=400.0, help="Reject readings above this as invalid")
    p.add_argument("--filter-mode", choices=["median", "average"], default="median", help="Use median/average over last 3 reads")
    p.add_argument("--use-hardware-radio", action="store_true", help="Use nRF24 hardware transport")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run(args))

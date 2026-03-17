#!/usr/bin/env python3
"""Indoor embedded obstacle system (hardened deterministic runtime)."""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
import traceback
from collections import deque
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
    debug_enabled: bool,
) -> None:
    filt = DistanceFilter(max_range_cm=max_range_cm, window=3, mode=filter_mode)
    stabilizer = StateStabilizer(hysteresis_cm=hysteresis_cm, debounce_ticks=debounce_ticks)
    sensor_tick = PeriodicScheduler(sensor_hz)
    radio_tick = PeriodicScheduler(radio_hz)

    latest_packet = IndoorPacket(state=SAFE, distance=round(safe_cm, 1)).to_wire()

    while True:
        now = time.monotonic()

        if sensor_tick.due(now):
            filtered: Dict[str, float] = {}
            for name in sensor_names:
                try:
                    raw = await sensor_provider.read_distance_cm(name)
                except Exception:
                    raw = None
                v = filt.update(name, raw)
                if v is not None:
                    filtered[name] = round(v, 1)

            nearest = min(filtered.values()) if filtered else None
            state = stabilizer.update(nearest, danger_cm=danger_cm, safe_cm=safe_cm)

            if len(sensor_names) == 1:
                dist = round(nearest, 1) if nearest is not None else round(safe_cm, 1)
                latest_packet = IndoorPacket(state=state, distance=dist).to_wire()
            else:
                dist_map = filtered if filtered else {name: round(safe_cm, 1) for name in sensor_names}
                latest_packet = IndoorPacket(state=state, distances=dist_map).to_wire()

            _debug(debug_enabled, "sensor_tick", nearest=nearest, state=state, packet=latest_packet)
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
    display_backend: str,
    display_cfg: Dict[str, Any],
    buzzer_backend: str,
    buzzer_cfg: Dict[str, Any],
    radio_hz: float,
    ui_hz: float,
    no_signal_timeout_sec: float,
    debug_enabled: bool,
) -> None:
    ui = ReceiverUI(backend=display_backend, display_cfg=display_cfg)
    buzzer = BuzzerController(backend=buzzer_backend, buzzer_cfg=buzzer_cfg)
    await buzzer.start()

    radio_tick = PeriodicScheduler(radio_hz)
    ui_tick = PeriodicScheduler(ui_hz)

    last_packet_at = time.monotonic()
    latest_packet: Optional[Dict[str, object]] = None
    latest_state = SAFE

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
                    _debug(debug_enabled, "radio_rx_tick", status="ok", state=latest_state)
                elif packet is not None:
                    _debug(debug_enabled, "radio_rx_tick", status="invalid_packet")
                else:
                    _debug(debug_enabled, "radio_rx_tick", status="no_packet")

                radio_tick.advance(now)

            now = time.monotonic()
            if ui_tick.due(now):
                if now - last_packet_at >= no_signal_timeout_sec:
                    await ui.render("NO SIGNAL")
                    buzzer.set_state(SAFE)
                    _debug(debug_enabled, "ui_tick", view="NO SIGNAL")
                else:
                    if latest_packet is None:
                        await ui.render("CLEAR")
                    else:
                        text = "CLEAR" if latest_state == SAFE else ("OBJECT NEAR" if latest_state == WARNING else "STOP")
                        if "distance" in latest_packet:
                            sub = f"{latest_packet['distance']} cm"
                        else:
                            sub = str(latest_packet.get("distances", {}))
                        await ui.render(text, sub)
                        _debug(debug_enabled, "ui_tick", view=text, state=latest_state)
                ui_tick.advance(now)

            sleep_for = _sleep_to_next(time.monotonic(), [radio_tick, ui_tick])
            await asyncio.sleep(sleep_for)
    finally:
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
                        display_backend=display_backend,
                        display_cfg=display_cfg,
                        buzzer_backend=buzzer_backend,
                        buzzer_cfg=buzzer_cfg,
                        radio_hz=radio_hz,
                        ui_hz=ui_hz,
                        no_signal_timeout_sec=no_signal_timeout_sec,
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

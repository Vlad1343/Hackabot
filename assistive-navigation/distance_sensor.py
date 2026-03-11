"""
HACKABOT Distance Sensor.
Interface for ultrasonic/LiDAR; mocked via keyboard (O = obstacle, C = very close).
Events sent to decision engine for immediate audio feedback.
"""
from __future__ import annotations

import asyncio
import threading
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class SensorEvent:
    level: str  # "approaching" | "close"
    source: str = "sensor"


class DistanceSensor(ABC):
    @abstractmethod
    async def start(self) -> None:
        ...

    @abstractmethod
    async def read(self) -> Optional[SensorEvent]:
        ...

    @abstractmethod
    async def stop(self) -> None:
        ...


class KeyboardDistanceSensor(DistanceSensor):
    """
    Keyboard mock for demo:
    - O: obstacle approaching
    - C: obstacle very close
    - Q: stop sensor thread
    """

    def __init__(self) -> None:
        self.queue: asyncio.Queue[SensorEvent] = asyncio.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self) -> None:
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        print("[Sensor] Keyboard sensor started (O=approaching, C=close, Q=quit sensor)")

    def _reader_loop(self) -> None:
        while self._running:
            try:
                key = input().strip().lower()
                if key == "q":
                    self._running = False
                    break
                if key == "o" and self._loop:
                    asyncio.run_coroutine_threadsafe(self.queue.put(SensorEvent(level="approaching")), self._loop)
                elif key == "c" and self._loop:
                    asyncio.run_coroutine_threadsafe(self.queue.put(SensorEvent(level="close")), self._loop)
            except EOFError:
                self._running = False
                break
            except Exception as exc:
                print(f"[Sensor] Keyboard sensor error: {exc}")
                traceback.print_exc()
                self._running = False
                break

    async def read(self) -> Optional[SensorEvent]:
        try:
            return self.queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def stop(self) -> None:
        self._running = False


class MockDistanceSensor(DistanceSensor):
    """Scripted events for automated demo."""

    def __init__(self, script: list[tuple[float, str]] | None = None) -> None:
        self.script = script or []
        self.queue: asyncio.Queue[SensorEvent] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="mock-distance-sensor")

    async def _run(self) -> None:
        for delay, level in self.script:
            await asyncio.sleep(delay)
            await self.queue.put(SensorEvent(level=level, source="mock"))

    async def read(self) -> Optional[SensorEvent]:
        try:
            return self.queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

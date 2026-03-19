"""
HACKABOT Obstacle Simulator.
Simulates obstacle events (approaching / close) for demo mode.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class ObstacleEvent:
    level: str  # "approaching" | "close"


class ObstacleSimulator:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._running = False
        self.queue: asyncio.Queue[ObstacleEvent] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if not self.enabled:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="obstacle-simulator")

    async def _run(self) -> None:
        while self._running:
            await asyncio.sleep(10.0)
            await self.queue.put(ObstacleEvent(level="approaching"))
            await asyncio.sleep(3.0)
            await self.queue.put(ObstacleEvent(level="close"))

    async def read(self) -> ObstacleEvent | None:
        try:
            return self.queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

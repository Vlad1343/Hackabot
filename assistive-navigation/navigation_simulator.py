"""
HACKABOT Navigation Simulator.
Simulates navigation instructions every few seconds:
turn_slightly_left, turn_slightly_right, path_clear.
Uses pre-recorded audio keys from config.
"""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass


@dataclass
class NavigationEvent:
    key: str


class NavigationSimulator:
    def __init__(self, interval_sec: float = 5.0, enabled: bool = True) -> None:
        self.interval_sec = interval_sec
        self.enabled = enabled
        self._running = False
        self.queue: asyncio.Queue[NavigationEvent] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if not self.enabled:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="navigation-simulator")

    async def _run(self) -> None:
        choices = ["turn_slightly_left", "turn_slightly_right", "path_clear"]
        while self._running:
            await asyncio.sleep(self.interval_sec)
            await self.queue.put(NavigationEvent(key=random.choice(choices)))

    async def read(self) -> NavigationEvent | None:
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

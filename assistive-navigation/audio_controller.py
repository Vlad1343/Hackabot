"""
HACKABOT Audio Controller.
Handles playback, prioritization (warnings first), cooldowns, and queue.
Logs all audio events with timestamp to CSV for debugging.
"""
from __future__ import annotations

import asyncio
import csv
import shutil
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import simpleaudio as sa
except Exception:
    sa = None


@dataclass(order=True)
class QueueItem:
    """Lower priority value = played first (0 = urgent, 3 = navigation)."""
    priority: int
    created_at: float
    event: "AudioEvent" = field(compare=False)


@dataclass
class AudioEvent:
    key: str
    mode: str
    priority: int
    cooldown_key: str
    cooldown_sec: float
    reason: str


class AudioController:
    """
    One message per object per direction per cooldown.
    Priority: 0 = very close, 1 = approaching, 2 = multiple/normal, 3 = navigation.
    Queue plays in order without overlapping; cooldown checked at enqueue time.
    """

    def __init__(self, config: Dict[str, Any], log_file: Path | str | None = None) -> None:
        self.config = config
        self.queue: asyncio.PriorityQueue[QueueItem] = asyncio.PriorityQueue()
        self.last_played: dict[str, float] = {}
        self.last_global_played: float = 0.0
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        self._warned_simulation = False
        self._warned_playback_failure = False

        logs_dir = Path(__file__).resolve().parent / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = Path(log_file) if log_file else logs_dir / "audio_events.csv"
        self._ensure_csv_header()

    async def _play_with_os_player(self, path: Path) -> bool:
        player = shutil.which("afplay") or shutil.which("aplay")
        if not player:
            return False
        try:
            proc = await asyncio.create_subprocess_exec(
                player,
                str(path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            return proc.returncode == 0
        except Exception as exc:
            print(f"[Audio] OS player failed ({player}): {exc}")
            traceback.print_exc()
            return False

    def _ensure_csv_header(self) -> None:
        if self.log_file.exists():
            return
        with self.log_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "mode", "key", "priority", "status", "reason", "file_path"])

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker(), name="audio-worker")

    async def stop(self) -> None:
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    def enqueue(
        self,
        *,
        key: str,
        mode: str,
        priority: int,
        cooldown_key: str | None = None,
        cooldown_sec: float | None = None,
        reason: str = "",
    ) -> bool:
        try:
            now = asyncio.get_running_loop().time()
        except RuntimeError:
            now = asyncio.get_event_loop().time()
        cooldown_key = cooldown_key or key
        cooldown_sec = float(cooldown_sec if cooldown_sec is not None else self.config["cooldowns"]["default_sec"])

        last = self.last_played.get(cooldown_key, 0.0)
        if now - last < cooldown_sec:
            self._log_event(mode=mode, key=key, priority=priority, status="suppressed", reason=f"cooldown:{cooldown_key}", file_path="")
            return False

        event = AudioEvent(
            key=key,
            mode=mode,
            priority=priority,
            cooldown_key=cooldown_key,
            cooldown_sec=cooldown_sec,
            reason=reason,
        )
        self.queue.put_nowait(QueueItem(priority=priority, created_at=now, event=event))
        return True

    def _resolve_audio_path(self, mode: str, key: str) -> Path:
        audio_cfg = self.config["audio"]
        root = Path(__file__).resolve().parent / audio_cfg["base_dir"]
        mode_dir = audio_cfg["indoors_dir"] if mode == "indoor" else audio_cfg["outdoors_dir"]

        override_name = audio_cfg.get("file_overrides", {}).get(key)
        file_name = override_name or f"{key}{audio_cfg.get('extension', '.wav')}"
        return root / mode_dir / file_name

    def _log_event(self, *, mode: str, key: str, priority: int, status: str, reason: str, file_path: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        try:
            with self.log_file.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([ts, mode, key, priority, status, reason, file_path])
        except Exception as exc:
            print(f"[Audio] Log write error: {exc}")

    def _is_riff_wav(self, path: Path) -> bool:
        try:
            with path.open("rb") as f:
                header = f.read(12)
            return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE"
        except Exception:
            return False

    async def _play_file(self, path: Path) -> None:
        if sa is None:
            if await self._play_with_os_player(path):
                return
            if not self._warned_simulation:
                print("[Audio] No playback backend found (simpleaudio/afplay/aplay); simulating playback")
                self._warned_simulation = True
            await asyncio.sleep(0.65)
            return

        if not self._is_riff_wav(path):
            if await self._play_with_os_player(path):
                return
            if not self._warned_playback_failure:
                print(f"[Audio] Playback backend unavailable; simulating audio timing ({path.name})")
                self._warned_playback_failure = True
            await asyncio.sleep(0.65)
            return

        try:
            wave_obj = await asyncio.to_thread(sa.WaveObject.from_wave_file, str(path))
            play_obj = wave_obj.play()
            await asyncio.to_thread(play_obj.wait_done)
        except FileNotFoundError:
            print(f"[Audio] Missing file: {path}")
            raise
        except Exception as exc:
            print(f"[Audio] simpleaudio failed for {path.name}, trying OS player: {exc}")
            if await self._play_with_os_player(path):
                return
            if not self._warned_playback_failure:
                print(f"[Audio] Playback backend unavailable; simulating audio timing ({path.name})")
                self._warned_playback_failure = True
            await asyncio.sleep(0.65)

    async def _worker(self) -> None:
        while self._running:
            got_item = False
            try:
                item = await self.queue.get()
                got_item = True
                event = item.event
                path = self._resolve_audio_path(event.mode, event.key)

                if not path.exists():
                    print(f"[Audio] Missing file for key '{event.key}': {path}")
                    self._log_event(
                        mode=event.mode,
                        key=event.key,
                        priority=event.priority,
                        status="missing_file",
                        reason=event.reason,
                        file_path=str(path),
                    )
                    continue

                min_gap = float(self.config.get("cooldowns", {}).get("min_gap_sec", 0.0))
                if min_gap > 0:
                    try:
                        now = asyncio.get_running_loop().time()
                    except RuntimeError:
                        now = asyncio.get_event_loop().time()
                    wait_needed = (self.last_global_played + min_gap) - now
                    if wait_needed > 0:
                        await asyncio.sleep(wait_needed)

                await self._play_file(path)
                try:
                    now = asyncio.get_running_loop().time()
                except RuntimeError:
                    now = asyncio.get_event_loop().time()
                self.last_played[event.cooldown_key] = now
                self.last_global_played = now
                self._log_event(
                    mode=event.mode,
                    key=event.key,
                    priority=event.priority,
                    status="played",
                    reason=event.reason,
                    file_path=str(path),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[Audio] Worker error: {exc}")
                traceback.print_exc()
            finally:
                if got_item:
                    self.queue.task_done()

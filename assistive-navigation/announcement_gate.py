from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple

from detection import DetectionEvent


@dataclass
class Observation:
    direction: str
    confidence: float
    is_multiple: bool
    timestamp: float


class StableAnnouncementGate:
    """Require consistent direction over multiple detection cycles before announcing."""

    def __init__(
        self,
        *,
        min_consistent_frames: int = 3,
        history_size: int = 5,
        min_confidence: float = 0.40,
        repeat_same_direction_sec: float = 12.0,
        track_lost_reset_sec: float = 4.0,
        key_mode: str = "label",
    ) -> None:
        self.min_consistent_frames = min_consistent_frames
        self.history_size = history_size
        self.min_confidence = min_confidence
        self.repeat_same_direction_sec = repeat_same_direction_sec
        self.track_lost_reset_sec = track_lost_reset_sec
        self.key_mode = key_mode

        self.history: Dict[str, Deque[Observation]] = {}
        self.last_announced: Dict[str, Tuple[str, bool, float]] = {}

    def _state_key(self, ev: DetectionEvent) -> str:
        if self.key_mode == "label_direction":
            return f"{ev.label}:{ev.direction}"
        return ev.label

    def allow(self, ev: DetectionEvent, now: float) -> bool:
        label = self._state_key(ev)
        hist = self.history.setdefault(label, deque(maxlen=self.history_size))

        if hist and (now - hist[-1].timestamp) > self.track_lost_reset_sec:
            hist.clear()

        hist.append(
            Observation(
                direction=ev.direction,
                confidence=ev.confidence,
                is_multiple=ev.is_multiple,
                timestamp=now,
            )
        )

        if len(hist) < self.min_consistent_frames:
            return False

        directions = [o.direction for o in hist]
        counts = Counter(directions)
        dominant_direction, dominant_count = counts.most_common(1)[0]
        if ev.direction != dominant_direction:
            return False
        if dominant_count < self.min_consistent_frames:
            return False

        confs = [o.confidence for o in hist if o.direction == ev.direction]
        if not confs or (sum(confs) / len(confs)) < self.min_confidence:
            return False

        last = self.last_announced.get(label)
        if last is None:
            self.last_announced[label] = (ev.direction, ev.is_multiple, now)
            return True

        last_direction, last_multiple, last_time = last
        same_state = (last_direction == ev.direction) and (last_multiple == ev.is_multiple)
        if same_state and (now - last_time) < self.repeat_same_direction_sec:
            return False

        self.last_announced[label] = (ev.direction, ev.is_multiple, now)
        return True

"""
Proximity-first fusion of ultrasonic distance and vision risk.

The assistive navigation stack pulls two independent obstacle signals into a
single stream of spoken alerts:

* an HC-SR04 ultrasonic sensor on the Pico A sensor node, relayed over
  nRF24L01 to Pico B and pushed to the laptop as serial packets containing
  a distance in centimetres;
* a YOLOv8 vision pipeline running on the laptop that emits a
  ``DetectionEvent`` per detected object with a coarse risk level of
  ``normal`` / ``approaching`` / ``close``.

Each source on its own produces unsafe alert behaviour:

* Vision alone misses transparent or low-contrast obstacles (glass doors,
  thin chair legs) and is unreliable at night, but it carries *direction*
  (left / ahead / right) and *semantics* ("person", "car").
* Ultrasonic alone is robust at short range and unaffected by lighting,
  but it sees a single forward cone, has no direction, and cannot tell a
  wall from a friend.

This module fuses them with one rule: **safety-biased max**. Each source
maps to a severity (SAFE / WARNING / DANGER), and the fused severity is the
higher of the two. Proximity gets to *escalate* a calm vision read into a
danger alert; vision gets to *escalate* a calm ultrasonic read when it sees
something close on the side. Neither source can ever drag the other one
*down*. Hysteresis prevents flicker between severities, and a staleness
check makes sure neither source can keep speaking after its underlying
signal has dropped out.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class Severity(IntEnum):
    """Ordered severity levels. Higher value means more urgent."""

    SAFE = 0
    WARNING = 1
    DANGER = 2


@dataclass(frozen=True)
class UltrasonicReading:
    """A single distance reading from the Pico sensor node.

    Attributes:
        distance_cm: Measured distance in centimetres. ``None`` represents
            "sensor responded but had no reading" (e.g. an echo timeout);
            entirely missing readings should not call ``fuse`` at all.
        timestamp_s: Monotonic timestamp at which the reading was captured.
    """

    distance_cm: Optional[float]
    timestamp_s: float


@dataclass(frozen=True)
class VisionEvent:
    """A vision detection summarised to what the fusion layer needs.

    The full ``DetectionEvent`` carries bounding boxes and class names that
    are useful for diagnostics but irrelevant for picking a severity. We
    deliberately accept only the fields that drive the rule so the contract
    between vision and fusion stays narrow.
    """

    direction: str  # "left" | "ahead" | "right"
    risk_level: str  # "normal" | "approaching" | "close"
    timestamp_s: float


@dataclass(frozen=True)
class FusedAlert:
    """The single alert downstream code is supposed to speak.

    ``source`` records which sensor drove the decision so the audio layer
    can pick an appropriate phrase (e.g. "Danger ahead" for ultrasonic
    vs "Person on your left" when vision wins).
    """

    severity: Severity
    direction: Optional[str]
    source: str  # "ultrasonic" | "vision" | "none"


@dataclass
class ObstacleFusion:
    """Combine an ultrasonic distance and a vision event into one alert.

    Args:
        danger_cm: Distance at or below which ultrasonic alone forces DANGER.
        warning_cm: Distance at or below which ultrasonic alone forces at
            least WARNING.
        stale_after_s: Maximum age of a reading before it is treated as
            missing. Both sensors share this budget; the laptop sees them
            at roughly the same rate.
        downgrade_frames: Number of consecutive lower-severity readings
            required before the fused output drops below the previous
            high. Hysteresis here prevents the user from hearing
            "Danger / Safe / Danger" flicker around the threshold.

    The class is stateful: instances must not be shared between independent
    navigation sessions because the hysteresis counter would leak between
    them. Construct one ``ObstacleFusion`` per session.
    """

    danger_cm: float = 30.0
    warning_cm: float = 100.0
    stale_after_s: float = 1.5
    downgrade_frames: int = 3

    def __post_init__(self) -> None:
        if self.danger_cm <= 0.0:
            raise ValueError("danger_cm must be positive")
        if self.warning_cm <= self.danger_cm:
            raise ValueError("warning_cm must exceed danger_cm")
        if self.stale_after_s <= 0.0:
            raise ValueError("stale_after_s must be positive")
        if self.downgrade_frames < 1:
            raise ValueError("downgrade_frames must be >= 1")
        self._last_severity: Severity = Severity.SAFE
        self._downgrade_streak: int = 0

    def fuse(
        self,
        now_s: float,
        ultrasonic: Optional[UltrasonicReading],
        vision: Optional[VisionEvent],
    ) -> FusedAlert:
        """Return the alert that best describes the current obstacle state.

        ``now_s`` is supplied by the caller (rather than read from
        ``time.monotonic`` here) so that replay tools and tests can drive
        the fusion deterministically. The two sensor arguments may be
        ``None``, indicating that no fresh reading was available this tick.
        """
        ultrasonic = self._reading_if_fresh(ultrasonic, now_s)
        vision = self._vision_if_fresh(vision, now_s)

        proposed = self._propose_severity(ultrasonic, vision)
        smoothed = self._apply_hysteresis(proposed.severity)
        # we keep the freshest direction and source even when hysteresis
        # holds the severity up: a stale "danger ahead" is more useful than
        # a stale "danger from no direction", and the freshest reading is
        # always the best guess at where the obstacle actually is
        return FusedAlert(severity=smoothed, direction=proposed.direction, source=proposed.source)

    # --------------------------------------------------------------- helpers
    def _reading_if_fresh(
        self, reading: Optional[UltrasonicReading], now_s: float
    ) -> Optional[UltrasonicReading]:
        if reading is None:
            return None
        if (now_s - reading.timestamp_s) > self.stale_after_s:
            return None
        return reading

    def _vision_if_fresh(
        self, vision: Optional[VisionEvent], now_s: float
    ) -> Optional[VisionEvent]:
        if vision is None:
            return None
        if (now_s - vision.timestamp_s) > self.stale_after_s:
            return None
        return vision

    def _propose_severity(
        self,
        ultrasonic: Optional[UltrasonicReading],
        vision: Optional[VisionEvent],
    ) -> FusedAlert:
        ultra_severity = self._ultrasonic_severity(ultrasonic)
        vision_severity = self._vision_severity(vision)

        if ultra_severity == Severity.SAFE and vision_severity == Severity.SAFE:
            # nothing urgent from either sensor. if vision still saw a
            # benign object we attach its direction so the UI can show
            # where the system is currently looking; otherwise empty
            direction = vision.direction if vision is not None else None
            source = "vision" if vision is not None else ("ultrasonic" if ultrasonic is not None else "none")
            return FusedAlert(Severity.SAFE, direction, source)

        # safety-biased max: take the more urgent of the two readings.
        # attribute the alert to whichever sensor produced that severity,
        # and break ties in favour of vision because it carries direction
        # and semantics that the audio layer can read out
        if vision_severity >= ultra_severity:
            return FusedAlert(vision_severity, vision.direction, "vision")  # type: ignore[union-attr]

        # ultrasonic wins. it has no direction of its own, so borrow one
        # from vision if any object is currently being tracked
        direction = vision.direction if vision is not None else None
        return FusedAlert(ultra_severity, direction, "ultrasonic")

    def _ultrasonic_severity(self, reading: Optional[UltrasonicReading]) -> Severity:
        if reading is None or reading.distance_cm is None:
            return Severity.SAFE
        if reading.distance_cm <= self.danger_cm:
            return Severity.DANGER
        if reading.distance_cm <= self.warning_cm:
            return Severity.WARNING
        return Severity.SAFE

    @staticmethod
    def _vision_severity(vision: Optional[VisionEvent]) -> Severity:
        if vision is None:
            return Severity.SAFE
        # unknown labels fall through to SAFE so a vision bug cannot
        # escalate a clear path into a danger alert
        return {
            "close": Severity.DANGER,
            "approaching": Severity.WARNING,
            "normal": Severity.SAFE,
        }.get(vision.risk_level, Severity.SAFE)

    def _apply_hysteresis(self, proposed: Severity) -> Severity:
        """Suppress short-lived drops in severity.

        Going *up* in severity always happens immediately: the cost of
        missing a danger alert dwarfs the cost of one false alarm. Going
        *down* requires ``downgrade_frames`` consecutive lower readings.
        """
        if proposed >= self._last_severity:
            self._last_severity = proposed
            self._downgrade_streak = 0
            return proposed

        self._downgrade_streak += 1
        if self._downgrade_streak >= self.downgrade_frames:
            self._last_severity = proposed
            self._downgrade_streak = 0
            return proposed
        return self._last_severity

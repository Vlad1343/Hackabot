"""
Tests for ``ObstacleFusion``.

We drive the fusion with synthetic ``UltrasonicReading`` and ``VisionEvent``
values and an explicit ``now_s`` clock; no real sensors are involved. The
tests are written so that each one pins exactly one rule from the module
docstring: failing one points at the rule that broke.
"""
from __future__ import annotations

import pytest

from obstacle_fusion import (
    FusedAlert,
    ObstacleFusion,
    Severity,
    UltrasonicReading,
    VisionEvent,
)


def ultra(distance_cm, at=0.0) -> UltrasonicReading:
    return UltrasonicReading(distance_cm=distance_cm, timestamp_s=at)


def vision(risk="normal", direction="ahead", at=0.0) -> VisionEvent:
    return VisionEvent(direction=direction, risk_level=risk, timestamp_s=at)


# ------------------------------------------------------------ proximity wins
def test_ultrasonic_danger_overrides_vision_normal() -> None:
    fusion = ObstacleFusion(danger_cm=30.0, warning_cm=100.0)
    alert = fusion.fuse(
        now_s=0.0,
        ultrasonic=ultra(distance_cm=20.0),
        vision=vision(risk="normal", direction="left"),
    )
    assert alert.severity == Severity.DANGER
    assert alert.source == "ultrasonic"
    # direction is borrowed from vision because ultrasonic has none
    assert alert.direction == "left"


def test_ultrasonic_warning_does_not_override_vision_danger() -> None:
    # vision says CLOSE (which maps to DANGER), ultrasonic only says WARNING.
    # we expect DANGER to win because severity is monotonic and proximity
    # only forces *upgrades*; it must not downgrade vision when vision is
    # already more urgent than what ultrasonic alone would say
    fusion = ObstacleFusion(danger_cm=30.0, warning_cm=100.0)
    alert = fusion.fuse(
        now_s=0.0,
        ultrasonic=ultra(distance_cm=50.0),
        vision=vision(risk="close", direction="ahead"),
    )
    assert alert.severity == Severity.DANGER


# ---------------------------------------------------------------- staleness
def test_stale_ultrasonic_is_ignored() -> None:
    fusion = ObstacleFusion(stale_after_s=1.0)
    # ultrasonic was captured 5 seconds ago: should be discarded so vision
    # drives the alert despite the dangerously low distance
    alert = fusion.fuse(
        now_s=5.0,
        ultrasonic=ultra(distance_cm=10.0, at=0.0),
        vision=vision(risk="normal", direction="right", at=5.0),
    )
    assert alert.severity == Severity.SAFE
    assert alert.source == "vision"


def test_stale_vision_is_ignored() -> None:
    fusion = ObstacleFusion(stale_after_s=1.0)
    alert = fusion.fuse(
        now_s=5.0,
        ultrasonic=ultra(distance_cm=200.0, at=5.0),
        vision=vision(risk="close", direction="ahead", at=0.0),
    )
    # ultrasonic reports clear, stale vision must not drag us back into danger
    assert alert.severity == Severity.SAFE


# ------------------------------------------------------------ rising vs falling
def test_severity_can_rise_immediately() -> None:
    # any single danger frame must produce a danger alert without waiting
    fusion = ObstacleFusion()
    alert = fusion.fuse(
        now_s=0.0,
        ultrasonic=ultra(distance_cm=20.0),
        vision=vision(risk="normal", direction="ahead"),
    )
    assert alert.severity == Severity.DANGER


def test_severity_drop_requires_hysteresis_streak() -> None:
    fusion = ObstacleFusion(downgrade_frames=3)

    # start at DANGER
    first = fusion.fuse(0.0, ultra(distance_cm=20.0, at=0.0), None)
    assert first.severity == Severity.DANGER

    # two clear frames are not yet enough to drop to SAFE
    clear_a = fusion.fuse(0.1, ultra(distance_cm=300.0, at=0.1), None)
    clear_b = fusion.fuse(0.2, ultra(distance_cm=300.0, at=0.2), None)
    assert clear_a.severity == Severity.DANGER
    assert clear_b.severity == Severity.DANGER

    # third consecutive clear frame finally drops the alert
    clear_c = fusion.fuse(0.3, ultra(distance_cm=300.0, at=0.3), None)
    assert clear_c.severity == Severity.SAFE


def test_intermittent_danger_resets_downgrade_streak() -> None:
    # the streak counter should not "remember" old clear frames if a danger
    # frame interrupts: one clear, one danger, one clear must not collapse
    # to "two clears" and downgrade prematurely
    fusion = ObstacleFusion(downgrade_frames=2)

    fusion.fuse(0.0, ultra(distance_cm=20.0, at=0.0), None)
    fusion.fuse(0.1, ultra(distance_cm=300.0, at=0.1), None)
    fusion.fuse(0.2, ultra(distance_cm=20.0, at=0.2), None)  # back to danger
    after_one_clear = fusion.fuse(0.3, ultra(distance_cm=300.0, at=0.3), None)
    assert after_one_clear.severity == Severity.DANGER


# ---------------------------------------------------------------- vision-only
@pytest.mark.parametrize(
    "risk,expected",
    [
        ("close", Severity.DANGER),
        ("approaching", Severity.WARNING),
        ("normal", Severity.SAFE),
        ("unknown_label", Severity.SAFE),  # unknown maps to safe, not danger
    ],
)
def test_vision_risk_levels_map_to_severity(risk, expected) -> None:
    fusion = ObstacleFusion()
    alert = fusion.fuse(now_s=0.0, ultrasonic=None, vision=vision(risk=risk))
    assert alert.severity == expected


def test_no_sensors_produces_safe_alert() -> None:
    fusion = ObstacleFusion()
    alert = fusion.fuse(now_s=0.0, ultrasonic=None, vision=None)
    assert alert == FusedAlert(severity=Severity.SAFE, direction=None, source="none")


def test_ultrasonic_reading_with_no_distance_is_ignored() -> None:
    # the pico sometimes relays "no reading" lines that arrive as a packet
    # with distance_cm=None; this must not crash and must not be treated as
    # zero centimetres
    fusion = ObstacleFusion()
    alert = fusion.fuse(
        now_s=0.0,
        ultrasonic=UltrasonicReading(distance_cm=None, timestamp_s=0.0),
        vision=vision(risk="approaching", direction="left"),
    )
    assert alert.severity == Severity.WARNING
    assert alert.source == "vision"
    assert alert.direction == "left"


# ---------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(danger_cm=0.0),
        dict(danger_cm=-5.0),
        dict(danger_cm=30.0, warning_cm=20.0),
        dict(stale_after_s=0.0),
        dict(downgrade_frames=0),
    ],
)
def test_constructor_rejects_invalid_arguments(kwargs) -> None:
    with pytest.raises(ValueError):
        ObstacleFusion(**kwargs)

#!/usr/bin/env python3
"""Deterministic tests for indoor audio state machine semantics."""
from __future__ import annotations

import unittest

import indoor_demo


class IndoorAudioStateMachineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._orig_now_ms = indoor_demo.now_ms
        self._t = 0

        def fake_now_ms() -> int:
            return self._t

        indoor_demo.now_ms = fake_now_ms

    async def asyncTearDown(self) -> None:
        indoor_demo.now_ms = self._orig_now_ms

    async def test_repeat_is_1hz_and_duplicate_frame_ignored(self) -> None:
        a = indoor_demo.IndoorAudioAnnouncer(
            enabled=True,
            folder=".",
            cooldown_ms=1000,
            confidence_threshold=0.5,
        )
        spoken: list[str] = []

        async def fake_speak(key: str) -> None:
            spoken.append(key)

        a._speak = fake_speak  # type: ignore[assignment]

        event = {"label": "PERSON", "confidence": 0.9, "frame_id": 1}
        await a.process(detected_event=event, frame_id=1, suppress=False)
        self.assertEqual(spoken, ["PERSON"])

        # Same frame_id should not retrigger immediately.
        await a.process(detected_event=event, frame_id=1, suppress=False)
        self.assertEqual(spoken, ["PERSON"])

        # Before 1s: no repeat.
        self._t = 999
        await a.process(detected_event=event, frame_id=2, suppress=False)
        self.assertEqual(spoken, ["PERSON"])

        # At/after 1s: repeat exactly once.
        self._t = 1000
        await a.process(detected_event=event, frame_id=3, suppress=False)
        self.assertEqual(spoken, ["PERSON", "PERSON"])

    async def test_priority_interrupt_and_lower_priority_ignored(self) -> None:
        a = indoor_demo.IndoorAudioAnnouncer(
            enabled=True,
            folder=".",
            cooldown_ms=1000,
            confidence_threshold=0.5,
        )
        spoken: list[str] = []

        async def fake_speak(key: str) -> None:
            spoken.append(key)

        a._speak = fake_speak  # type: ignore[assignment]

        # Lower priority active first.
        await a.process(detected_event={"label": "TABLE", "confidence": 0.9, "frame_id": 1}, frame_id=1, suppress=False)
        self.assertEqual(spoken, ["TABLE"])

        # Higher priority interrupts.
        await a.process(detected_event={"label": "PERSON", "confidence": 0.9, "frame_id": 2}, frame_id=2, suppress=False)
        self.assertEqual(spoken, ["TABLE", "PERSON"])

        # Lower priority must be ignored while PERSON active.
        await a.process(detected_event={"label": "CHAIR", "confidence": 0.9, "frame_id": 3}, frame_id=3, suppress=False)
        self.assertEqual(spoken, ["TABLE", "PERSON"])

    async def test_danger_suppresses_and_clears(self) -> None:
        a = indoor_demo.IndoorAudioAnnouncer(
            enabled=True,
            folder=".",
            cooldown_ms=1000,
            confidence_threshold=0.5,
        )
        spoken: list[str] = []

        async def fake_speak(key: str) -> None:
            spoken.append(key)

        a._speak = fake_speak  # type: ignore[assignment]

        await a.process(detected_event={"label": "PERSON", "confidence": 0.9, "frame_id": 1}, frame_id=1, suppress=False)
        self.assertEqual(a.state.current_object, "PERSON")

        await a.process(detected_event=None, frame_id=-1, suppress=True)
        self.assertIsNone(a.state.current_object)

        # While suppressed event call does not speak unless new detection arrives.
        self._t = 1000
        await a.process(detected_event=None, frame_id=-1, suppress=False)
        self.assertEqual(spoken, ["PERSON"])


if __name__ == "__main__":
    unittest.main()

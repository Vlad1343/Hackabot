#!/usr/bin/env python3
"""Direction integrity tests for centroid-score zoning."""
from __future__ import annotations

import unittest

from detection import DetectionEngine


class DirectionMappingTests(unittest.TestCase):
    def test_left_center_right(self) -> None:
        margin = 0.15
        confidence = 0.9

        left_zone = DetectionEngine.zone_from_centroid_score(0.1, confidence, margin)
        center_zone = DetectionEngine.zone_from_centroid_score(0.5, confidence, margin)
        right_zone = DetectionEngine.zone_from_centroid_score(0.9, confidence, margin)

        self.assertEqual(left_zone, "LEFT")
        self.assertEqual(center_zone, "FRONT")
        self.assertEqual(right_zone, "RIGHT")


if __name__ == "__main__":
    unittest.main()

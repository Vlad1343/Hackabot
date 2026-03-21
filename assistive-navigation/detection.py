"""
HACKABOT Object Detection.
YOLOv8 nano; filters by indoor/outdoor classes; computes direction (left/ahead/right)
and risk level (normal, approaching, close). Multiple instances → is_multiple + count.
"""
from __future__ import annotations

import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


@dataclass
class DetectionEvent:
    label: str
    direction: str  # left | ahead | right
    confidence: float
    bbox: tuple[int, int, int, int]
    count: int
    is_multiple: bool
    risk_level: str  # "normal" | "approaching" | "close"


class DetectionEngine:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model = None
        self._frame_id = 0
        self._load_model()

    def _load_model(self) -> None:
        weights = self.config["model"].get("weights_path", "yolov8n.pt")
        weights_path = Path(weights)
        if not weights_path.is_absolute():
            weights_path = Path(__file__).resolve().parent / weights

        if YOLO is None:
            print("[Detection] ultralytics is not installed; detection disabled")
            return
        if not weights_path.exists():
            print(f"[Detection] Model weights missing: {weights_path}")
            return

        try:
            self.model = YOLO(str(weights_path))
        except Exception as exc:
            print(f"[Detection] Failed to load model {weights_path}: {exc}")
            traceback.print_exc()
            self.model = None

    def _normalize_label(self, raw_label: str, mode: str) -> str:
        aliases = self.config.get("class_aliases", {})
        label = aliases.get(raw_label, raw_label)
        if mode == "outdoor" and label == "person":
            return "pedestrian"
        return label.replace(" ", "_")

    @staticmethod
    def zone_from_centroid_score(x_center_norm: float, confidence: float, margin: float) -> str:
        x_norm = max(0.0, min(1.0, float(x_center_norm)))
        conf = max(0.0, float(confidence))
        m = max(0.0, float(margin))
        left_score = conf * (1.0 - x_norm)
        right_score = conf * x_norm
        if left_score > (right_score + m):
            return "LEFT"
        if right_score > (left_score + m):
            return "RIGHT"
        return "FRONT"

    @staticmethod
    def direction_from_zone(zone: str) -> str:
        return {"LEFT": "left", "RIGHT": "right", "FRONT": "ahead"}.get(zone, "ahead")

    def _risk_from_bbox(self, bbox: tuple[int, int, int, int], frame_area: int) -> str:
        x1, y1, x2, y2 = bbox
        box_area = max(0, x2 - x1) * max(0, y2 - y1)
        ratio = box_area / float(max(1, frame_area))

        if ratio >= self.config["detection"]["close_area_ratio"]:
            return "close"
        if ratio >= self.config["detection"]["approaching_area_ratio"]:
            return "approaching"
        return "normal"

    def _crop_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        top = int(h * self.config["app"]["crop_top_ratio"])
        bottom = int(h * (1.0 - self.config["app"]["crop_bottom_ratio"]))
        top = max(0, min(top, h - 1))
        bottom = max(top + 1, min(bottom, h))
        return frame[top:bottom, 0:w]

    def detect(self, frame: np.ndarray, mode: str, *, is_mirrored: bool = False) -> tuple[list[DetectionEvent], list[dict]]:
        if self.model is None:
            return [], []

        try:
            self._frame_id += 1
            frame_id = self._frame_id
            timestamp_ms = int(time.time() * 1000)
            if is_mirrored:
                print("[Vision] FRAME MIRRORING VIOLATION: inference frame must be non-mirrored")
            cropped = self._crop_frame(frame)
            imgsz = int(self.config["app"].get("input_size", 416))
            conf_th = float(self.config["detection"].get("confidence_threshold", 0.35))
            direction_margin = float(self.config.get("vision", {}).get("direction_margin", 0.15))
            results = self.model.predict(source=cropped, conf=conf_th, imgsz=imgsz, verbose=False)

            important = set(self.config["classes"].get(mode, []))
            frame_h, frame_w = cropped.shape[:2]
            frame_area = frame_h * frame_w

            raw_items: list[dict] = []
            grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label_raw = result.names.get(cls_id, str(cls_id))
                    label = self._normalize_label(label_raw, mode)

                    if label not in important:
                        continue

                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    bbox = (x1, y1, x2, y2)
                    x1, y1, x2, y2 = bbox
                    x_center_norm = ((x1 + x2) / 2.0) / float(max(1, frame_w))
                    y_center_norm = ((y1 + y2) / 2.0) / float(max(1, frame_h))
                    zone = self.zone_from_centroid_score(x_center_norm, conf, direction_margin)
                    direction = self.direction_from_zone(zone)
                    risk = self._risk_from_bbox(bbox, frame_area)

                    item = {
                        "label": label,
                        "direction": direction,
                        "confidence": conf,
                        "bbox": bbox,
                        "risk_level": risk,
                        "x_center_norm": max(0.0, min(1.0, x_center_norm)),
                        "y_center_norm": max(0.0, min(1.0, y_center_norm)),
                        "zone": zone,
                        "timestamp_ms": timestamp_ms,
                        "frame_id": frame_id,
                    }
                    raw_items.append(item)
                    grouped[(label, direction)].append(item)

            print(f"[VisionRaw] frame_id={frame_id} ts={timestamp_ms} objects={len(raw_items)}")

            events: list[DetectionEvent] = []
            debug_detections: list[dict] = []
            for (label, direction), items in grouped.items():
                count = len(items)
                chosen = max(items, key=lambda x: (x["confidence"], x["bbox"][3] - x["bbox"][1]))
                risk_level = "close" if any(x["risk_level"] == "close" for x in items) else (
                    "approaching" if any(x["risk_level"] == "approaching" for x in items) else "normal"
                )

                events.append(
                    DetectionEvent(
                        label=label,
                        direction=direction,
                        confidence=chosen["confidence"],
                        bbox=chosen["bbox"],
                        count=count,
                        is_multiple=count > 1,
                        risk_level=risk_level,
                    )
                )
                debug_detections.append(
                    {
                        "label": str(label).upper(),
                        "x_center_norm": round(float(chosen.get("x_center_norm", 0.5)), 4),
                        "y_center_norm": round(float(chosen.get("y_center_norm", 0.5)), 4),
                        "zone": str(chosen.get("zone", "FRONT")).upper(),
                        "confidence": round(float(chosen["confidence"]), 3),
                        "timestamp_ms": int(chosen.get("timestamp_ms", timestamp_ms)),
                        "frame_id": int(chosen.get("frame_id", frame_id)),
                        "bbox": chosen["bbox"],
                        "direction": direction,
                    }
                )

            print(f"[VisionFiltered] frame_id={frame_id} ts={timestamp_ms} objects={len(debug_detections)}")
            return events, debug_detections
        except Exception as exc:
            print(f"[Detection] Detection error: {exc}")
            traceback.print_exc()
            return [], []

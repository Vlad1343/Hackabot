#!/usr/bin/env python3
"""
video_overlay.py

Overlay YOLO-style annotations from JSON/CSV onto a prerecorded video.
Supports both point annotations (x,y) and bbox annotations (x1,y1,x2,y2 or x,y,w,h).

Run:
  python video_overlay.py --video demo.mp4 --coords coordinates.json --output output_video.mp4
  python video_overlay.py --video demo.mp4 --coords coordinates.csv --output output_video.mp4 --show-direction
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

CLASS_COLORS = {
    "traffic_light_red": (0, 0, 255),
    "traffic_light_yellow": (0, 255, 255),
    "traffic_light_green": (0, 200, 0),
    "car": (255, 120, 0),
    "pedestrian": (255, 0, 255),
    "bicycle": (255, 255, 0),
    "bus": (0, 165, 255),
}


def class_color(label: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(str(label).strip().lower(), (0, 255, 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay simulated YOLO detections on video")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--coords", required=True, help="Coordinates file (.json or .csv)")
    parser.add_argument("--output", default="output_video.mp4", help="Output annotated video path")
    parser.add_argument("--default-box-width", type=int, default=80, help="Used when only x,y exists")
    parser.add_argument("--default-box-height", type=int, default=80, help="Used when only x,y exists")
    parser.add_argument("--show-direction", action="store_true", help="Append left/center/right to label")
    return parser.parse_args()


def validate_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")


def parse_int(value, default: int = -1) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def normalize_item(item: dict) -> dict:
    return {
        "class": str(item.get("class", "object")),
        "x": parse_int(item.get("x", -1)),
        "y": parse_int(item.get("y", -1)),
        "w": parse_int(item.get("w", -1)),
        "h": parse_int(item.get("h", -1)),
        "x1": parse_int(item.get("x1", -1)),
        "y1": parse_int(item.get("y1", -1)),
        "x2": parse_int(item.get("x2", -1)),
        "y2": parse_int(item.get("y2", -1)),
    }


def load_json(path: Path) -> Dict[str, List[dict]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object with frame keys like frame_1")

    frames: Dict[str, List[dict]] = {}
    for frame_key, items in data.items():
        if not isinstance(items, list):
            raise ValueError(f"Frame '{frame_key}' must contain a list")
        frames[frame_key] = [normalize_item(item) for item in items if isinstance(item, dict)]
    return frames


def load_csv(path: Path) -> Dict[str, List[dict]]:
    frames: Dict[str, List[dict]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"frame", "class", "x", "y"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSV must contain at least: frame,class,x,y")

        for row in reader:
            frame_num = str(row.get("frame", "")).strip()
            if not frame_num.isdigit():
                continue
            key = f"frame_{int(frame_num)}"
            frames.setdefault(key, []).append(normalize_item(row))
    return frames


def load_coords(path: Path) -> Dict[str, List[dict]]:
    if path.suffix.lower() == ".csv":
        return load_csv(path)
    return load_json(path)


def clamp(n: int, low: int, high: int) -> int:
    return max(low, min(high, n))


def direction_label(cx: int, width: int) -> str:
    if width <= 0:
        return "center"
    ratio = cx / float(width)
    if ratio < 0.33:
        return "left"
    if ratio > 0.66:
        return "right"
    return "center"


def box_from_item(item: dict, frame_w: int, frame_h: int, default_w: int, default_h: int) -> Tuple[int, int, int, int] | None:
    x1 = int(item.get("x1", -1))
    y1 = int(item.get("y1", -1))
    x2 = int(item.get("x2", -1))
    y2 = int(item.get("y2", -1))

    # Preferred format: explicit corners.
    if min(x1, y1, x2, y2) >= 0 and x2 > x1 and y2 > y1:
        return (
            clamp(x1, 0, frame_w - 1),
            clamp(y1, 0, frame_h - 1),
            clamp(x2, 0, frame_w - 1),
            clamp(y2, 0, frame_h - 1),
        )

    # Alternative format: center + size.
    cx = int(item.get("x", -1))
    cy = int(item.get("y", -1))
    w = int(item.get("w", -1))
    h = int(item.get("h", -1))
    if cx >= 0 and cy >= 0 and w > 1 and h > 1:
        x1 = cx - w // 2
        y1 = cy - h // 2
        x2 = cx + w // 2
        y2 = cy + h // 2
        return (
            clamp(x1, 0, frame_w - 1),
            clamp(y1, 0, frame_h - 1),
            clamp(x2, 0, frame_w - 1),
            clamp(y2, 0, frame_h - 1),
        )

    # Backward compatibility: point only.
    if cx >= 0 and cy >= 0:
        half_w = max(1, default_w // 2)
        half_h = max(1, default_h // 2)
        return (
            clamp(cx - half_w, 0, frame_w - 1),
            clamp(cy - half_h, 0, frame_h - 1),
            clamp(cx + half_w, 0, frame_w - 1),
            clamp(cy + half_h, 0, frame_h - 1),
        )

    return None


def main() -> int:
    args = parse_args()
    video_path = Path(args.video)
    coords_path = Path(args.coords)
    output_path = Path(args.output)

    try:
        validate_file(video_path, "Video")
        validate_file(coords_path, "Coordinates file")
    except Exception as exc:
        print(f"[Error] {exc}")
        return 1

    try:
        coords = load_coords(coords_path)
    except Exception as exc:
        print(f"[Error] Failed to read coordinates: {exc}")
        return 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Error] Unable to open video: {video_path}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"[Error] Unable to create output video: {output_path}")
        cap.release()
        return 1

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_idx += 1
        key = f"frame_{frame_idx}"
        items = coords.get(key, [])

        debug_items = []
        for item in items:
            cls = str(item.get("class", "object"))
            box = box_from_item(item, width, height, args.default_box_width, args.default_box_height)
            if box is None:
                print(f"[Warn] frame={frame_idx} invalid item skipped class={cls}")
                continue

            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                print(f"[Warn] frame={frame_idx} invalid box skipped class={cls}")
                continue

            cx = (x1 + x2) // 2
            direction = direction_label(cx, width)
            label = cls if not args.show_direction else f"{cls} {direction}"
            color = class_color(cls)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
            debug_items.append({"class": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "direction": direction})

        print(f"[Frame {frame_idx}] objects={debug_items}")
        writer.write(frame)

    cap.release()
    writer.release()

    print(f"[Done] Output video saved: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

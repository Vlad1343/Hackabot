#!/usr/bin/env python3
"""
mouse_tracker.py

Manual bounding-box tracker for prerecorded videos.

Run:
  python mouse_tracker.py --video demo.mp4 --output coordinates.json --classes car,pedestrian,bicycle,traffic_light
  python mouse_tracker.py --video demo.mp4 --output coordinates.csv --classes person,chair,table,door,wall

Controls:
  Left click + drag : draw bounding box for current class
  1..9              : switch class from --classes list
  u                 : undo last box in current frame
  c                 : clear all boxes in current frame
  n or SPACE        : save current frame and move to next
  q                 : save and quit
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual bbox tracker for video frames")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", default="coordinates.json", help="Output file (.json or .csv)")
    parser.add_argument(
        "--classes",
        default="car,pedestrian,bicycle,traffic_light",
        help="Comma-separated class labels. Use keys 1..9 to switch class.",
    )
    parser.add_argument("--min-box-size", type=int, default=8, help="Minimum width/height to keep drawn box")
    return parser.parse_args()


def validate_video(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Video path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Video path is not a file: {path}")


def norm_box(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int, int, int]:
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def make_record(cls: str, x1: int, y1: int, x2: int, y2: int) -> dict:
    x1, y1, x2, y2 = norm_box(x1, y1, x2, y2)
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w // 2
    cy = y1 + h // 2
    return {
        "class": cls,
        "x": int(cx),
        "y": int(cy),
        "w": int(w),
        "h": int(h),
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
    }


def save_json(output_path: Path, records: Dict[str, List[dict]]) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def save_csv(output_path: Path, records: Dict[str, List[dict]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "class", "x", "y", "w", "h", "x1", "y1", "x2", "y2"])
        for frame_key, items in records.items():
            frame_num = frame_key.replace("frame_", "")
            for item in items:
                writer.writerow(
                    [
                        frame_num,
                        item.get("class", "object"),
                        item.get("x", -1),
                        item.get("y", -1),
                        item.get("w", -1),
                        item.get("h", -1),
                        item.get("x1", -1),
                        item.get("y1", -1),
                        item.get("x2", -1),
                        item.get("y2", -1),
                    ]
                )


def draw_annotations(img, items: List[dict]) -> None:
    for i, item in enumerate(items, start=1):
        x1 = int(item.get("x1", item.get("x", 0) - 20))
        y1 = int(item.get("y1", item.get("y", 0) - 20))
        x2 = int(item.get("x2", item.get("x", 0) + 20))
        y2 = int(item.get("y2", item.get("y", 0) + 20))
        label = f"{i}:{item.get('class', 'object')}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    video_path = Path(args.video)
    output_path = Path(args.output)

    try:
        validate_video(video_path)
    except Exception as exc:
        print(f"[Error] Invalid video path: {exc}")
        return 1

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not classes:
        print("[Error] No classes provided. Use --classes class1,class2")
        return 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Error] Unable to open video: {video_path}")
        return 1

    frame_records: Dict[str, List[dict]] = {}
    current_boxes: List[dict] = []
    current_class_idx = 0
    frame_idx = 0
    current_frame = None

    drawing = False
    drag_start = (0, 0)
    drag_end = (0, 0)

    window_name = "Mouse Tracker (BBox)"
    cv2.namedWindow(window_name)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal drawing, drag_start, drag_end, current_boxes

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            drag_start = (x, y)
            drag_end = (x, y)
            return

        if event == cv2.EVENT_MOUSEMOVE and drawing:
            drag_end = (x, y)
            return

        if event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            drag_end = (x, y)
            x1, y1, x2, y2 = norm_box(drag_start[0], drag_start[1], drag_end[0], drag_end[1])
            if (x2 - x1) < args.min_box_size or (y2 - y1) < args.min_box_size:
                print(f"[Skip] frame={frame_idx + 1} small box ignored ({x2-x1}x{y2-y1})")
                return

            rec = make_record(classes[current_class_idx], x1, y1, x2, y2)
            current_boxes.append(rec)
            print(
                "[Box] "
                f"frame={frame_idx + 1} class={rec['class']} "
                f"x1={rec['x1']} y1={rec['y1']} x2={rec['x2']} y2={rec['y2']}"
            )

    cv2.setMouseCallback(window_name, on_mouse)

    print("[Info] Started bbox tracker")
    print(f"[Info] Classes: {classes}")
    print("[Info] Controls: drag-box | 1..9 class | u undo | c clear-frame | n/SPACE next | q save+quit")

    while True:
        if current_frame is None:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[Info] End of video reached")
                break
            current_frame = frame
            current_boxes = []

        display = current_frame.copy()
        draw_annotations(display, current_boxes)

        if drawing:
            x1, y1, x2, y2 = norm_box(drag_start[0], drag_start[1], drag_end[0], drag_end[1])
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 200, 0), 2)

        hud_line1 = f"frame={frame_idx + 1} class={classes[current_class_idx]} objects={len(current_boxes)}"
        hud_line2 = "keys: 1..9 class | u undo | c clear | n/SPACE next | q quit"
        cv2.putText(display, hud_line1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(display, hud_line2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            frame_records[f"frame_{frame_idx + 1}"] = list(current_boxes)
            print("[Info] Quit requested. Saving annotations...")
            break

        if key in (ord("n"), 32):
            frame_records[f"frame_{frame_idx + 1}"] = list(current_boxes)
            print(f"[Frame] Saved frame {frame_idx + 1} with {len(current_boxes)} object(s)")
            frame_idx += 1
            current_frame = None
            continue

        if key == ord("u"):
            if current_boxes:
                removed = current_boxes.pop()
                print(f"[Undo] Removed {removed['class']} from frame {frame_idx + 1}")
            else:
                print("[Undo] Nothing to remove")
            continue

        if key == ord("c"):
            current_boxes.clear()
            print(f"[Clear] Cleared annotations for frame {frame_idx + 1}")
            continue

        if ord("1") <= key <= ord("9"):
            idx = key - ord("1")
            if idx < len(classes):
                current_class_idx = idx
                print(f"[Class] Current class -> {classes[current_class_idx]}")

    cap.release()
    cv2.destroyAllWindows()

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".csv":
            save_csv(output_path, frame_records)
        else:
            save_json(output_path, frame_records)
        print(f"[Done] Saved annotations to: {output_path.resolve()}")
    except Exception as exc:
        print(f"[Error] Failed to save output: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

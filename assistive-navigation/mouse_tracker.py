#!/usr/bin/env python3
"""
mouse_tracker.py

Manual bounding-box tracker for prerecorded videos.

Run:
  python mouse_tracker.py --video demo.mp4 --output coordinates.json --classes car,pedestrian,bicycle,traffic_light,bus
  python mouse_tracker.py --video demo.mp4 --output coordinates.csv --classes person,chair,table,door,wall

Controls:
  Left click + drag : draw bounding box for current class (when clicking empty area)
  Click box         : select existing box
  Drag selected box : move box
  Drag corner       : resize selected box
  DELETE/BACKSPACE  : delete selected box
  1..9              : switch class from --classes list
  u                 : undo last box in current frame
  c                 : clear all boxes in current frame
  n or SPACE        : save current frame and move to next (with --frame-step)
  q                 : save and quit
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def expand_classes(raw: list[str]) -> list[str]:
    expanded: list[str] = []
    for item in raw:
        key = item.strip().lower()
        if not key:
            continue
        if key == "traffic_light":
            expanded.extend(["traffic_light_red", "traffic_light_yellow", "traffic_light_green"])
        else:
            expanded.append(key)
    # preserve order while removing duplicates
    seen = set()
    out = []
    for c in expanded:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual bbox tracker for video frames")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", default="coordinates.json", help="Output file (.json or .csv)")
    parser.add_argument(
        "--classes",
        default="car,pedestrian,bicycle,traffic_light,bus",
        help="Comma-separated class labels. Use keys 1..9 to switch class.",
    )
    parser.add_argument("--min-box-size", type=int, default=8, help="Minimum width/height to keep drawn box")
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="How many frames to jump after saving current frame (1 = every frame, 5 = every 5th frame)",
    )
    parser.add_argument(
        "--auto-track",
        action="store_true",
        help="Auto-propagate boxes to next frames using OpenCV tracker; you then only fix errors.",
    )
    parser.add_argument(
        "--tracker-type",
        default="csrt",
        choices=["csrt", "kcf", "mil"],
        help="Tracker backend for --auto-track",
    )
    parser.add_argument(
        "--max-shift-ratio",
        type=float,
        default=0.35,
        help="Max allowed center shift per frame as ratio of last box size (lower = less drift)",
    )
    parser.add_argument(
        "--max-area-change",
        type=float,
        default=2.5,
        help="Max allowed area growth/shrink factor per frame for auto-track (e.g. 2.5)",
    )
    parser.add_argument(
        "--pred-alpha",
        type=float,
        default=0.65,
        help="Blend for measured center vs predicted center (0..1)",
    )
    parser.add_argument(
        "--pred-beta",
        type=float,
        default=0.50,
        help="Blend for measured size vs predicted size (0..1)",
    )
    parser.add_argument(
        "--vel-gamma",
        type=float,
        default=0.60,
        help="Velocity smoothing factor (0..1), higher = smoother/slower change",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
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
        color = class_color(item.get("class", "object"))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def persist_records(output_path: Path, frame_records: Dict[str, List[dict]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        save_csv(output_path, frame_records)
    else:
        save_json(output_path, frame_records)


def load_existing_records(output_path: Path) -> Dict[str, List[dict]]:
    if not output_path.exists():
        return {}
    if output_path.suffix.lower() == ".csv":
        records: Dict[str, List[dict]] = {}
        with output_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_s = str(row.get("frame", "")).strip()
                if not frame_s.isdigit():
                    continue
                key = f"frame_{int(frame_s)}"
                records.setdefault(key, []).append(
                    {
                        "class": str(row.get("class", "object")),
                        "x": int(float(row.get("x", -1))),
                        "y": int(float(row.get("y", -1))),
                        "w": int(float(row.get("w", -1))),
                        "h": int(float(row.get("h", -1))),
                        "x1": int(float(row.get("x1", -1))),
                        "y1": int(float(row.get("y1", -1))),
                        "x2": int(float(row.get("x2", -1))),
                        "y2": int(float(row.get("y2", -1))),
                    }
                )
        return records
    with output_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    return {}


def latest_saved_frame(records: Dict[str, List[dict]]) -> int:
    mx = 0
    for k in records.keys():
        if not k.startswith("frame_"):
            continue
        n = k.replace("frame_", "").strip()
        if n.isdigit():
            mx = max(mx, int(n))
    return mx


def _point_in_box(x: int, y: int, box: dict) -> bool:
    x1 = int(box.get("x1", 0))
    y1 = int(box.get("y1", 0))
    x2 = int(box.get("x2", 0))
    y2 = int(box.get("y2", 0))
    return x1 <= x <= x2 and y1 <= y <= y2


def _box_area(box: dict) -> int:
    return max(0, int(box.get("x2", 0)) - int(box.get("x1", 0))) * max(0, int(box.get("y2", 0)) - int(box.get("y1", 0)))


def _find_top_box_index(items: List[dict], x: int, y: int) -> int:
    matches = []
    for i, box in enumerate(items):
        if _point_in_box(x, y, box):
            matches.append((i, _box_area(box)))
    if not matches:
        return -1
    # prefer smaller area if boxes overlap (more precise selection)
    matches.sort(key=lambda t: t[1])
    return matches[0][0]


def _handle_hit(box: dict, x: int, y: int, handle_size: int = 10) -> str:
    x1 = int(box.get("x1", 0))
    y1 = int(box.get("y1", 0))
    x2 = int(box.get("x2", 0))
    y2 = int(box.get("y2", 0))
    corners = {
        "tl": (x1, y1),
        "tr": (x2, y1),
        "bl": (x1, y2),
        "br": (x2, y2),
    }
    for name, (cx, cy) in corners.items():
        if abs(x - cx) <= handle_size and abs(y - cy) <= handle_size:
            return name
    return ""


def _tracker_ctor(name: str):
    lname = name.lower()
    constructors = []
    if lname == "csrt":
        constructors = ["TrackerCSRT_create", "legacy.TrackerCSRT_create"]
    elif lname == "kcf":
        constructors = ["TrackerKCF_create", "legacy.TrackerKCF_create"]
    else:
        constructors = ["TrackerMIL_create", "legacy.TrackerMIL_create"]

    for ctor_name in constructors:
        try:
            if "." in ctor_name:
                parent, child = ctor_name.split(".", 1)
                obj = getattr(cv2, parent, None)
                if obj is not None and hasattr(obj, child):
                    return getattr(obj, child)
            else:
                if hasattr(cv2, ctor_name):
                    return getattr(cv2, ctor_name)
        except Exception:
            continue
    return None


def init_trackers(frame, boxes: List[dict], tracker_type: str):
    ctor = _tracker_ctor(tracker_type)
    chosen = tracker_type
    if ctor is None:
        # Auto-fallback so --auto-track still works on limited OpenCV builds.
        for alt in ("kcf", "mil"):
            ctor = _tracker_ctor(alt)
            if ctor is not None:
                chosen = alt
                print(f"[Track] Tracker '{tracker_type}' unavailable; falling back to '{alt}'")
                break
    if ctor is None:
        print(f"[Track] No supported tracker backend available (tried: {tracker_type}, kcf, mil)")
        return []

    trackers = []
    for box in boxes:
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        try:
            tr = ctor()
            init_result = tr.init(frame, (x1, y1, w, h))
            # OpenCV trackers are inconsistent across versions:
            # some return bool, some return None on success.
            ok = True if init_result is None else bool(init_result)
        except Exception:
            ok = False
            tr = None
        if ok and tr is not None:
            trackers.append(
                {
                    "tracker": tr,
                    "class": box.get("class", "object"),
                    "missed": 0,
                    "last_box": make_record(str(box.get("class", "object")), x1, y1, x2, y2),
                    "velocity": (0.0, 0.0),  # center delta per frame (smoothed)
                }
            )
        else:
            print(f"[Track] init failed for class={box.get('class', 'object')} box=({x1},{y1},{x2},{y2})")
    print(f"[Track] Initialized {len(trackers)} tracker(s) with backend '{chosen}'")
    return trackers


def predict_boxes_from_trackers(
    frame,
    trackers: List[dict],
    max_shift_ratio: float,
    max_area_change: float,
    pred_alpha: float,
    pred_beta: float,
    vel_gamma: float,
) -> List[dict]:
    predicted: List[dict] = []
    alive = []
    max_missed = 5
    alpha = max(0.0, min(1.0, float(pred_alpha)))
    beta = max(0.0, min(1.0, float(pred_beta)))
    gamma = max(0.0, min(1.0, float(vel_gamma)))
    for t in trackers:
        tr = t["tracker"]
        cls = t["class"]
        try:
            ok, bb = tr.update(frame)
        except Exception:
            ok = False
            bb = None
        if ok and bb is not None:
            x, y, w, h = bb
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            if x2 > x1 and y2 > y1:
                rec = make_record(str(cls), x1, y1, x2, y2)
                last_box = t.get("last_box")
                # Drift guard: reject sudden side jumps/size explosions between adjacent frames.
                if isinstance(last_box, dict):
                    lx1, ly1, lx2, ly2 = int(last_box["x1"]), int(last_box["y1"]), int(last_box["x2"]), int(last_box["y2"])
                    lw, lh = max(1, lx2 - lx1), max(1, ly2 - ly1)
                    lcx, lcy = lx1 + lw / 2.0, ly1 + lh / 2.0
                    ncx, ncy = rec["x1"] + rec["w"] / 2.0, rec["y1"] + rec["h"] / 2.0
                    vx, vy = t.get("velocity", (0.0, 0.0))
                    exp_cx, exp_cy = lcx + float(vx), lcy + float(vy)
                    # Predict + fuse (center, size) using previous motion.
                    pred_w, pred_h = lw, lh
                    fused_cx = alpha * ncx + (1.0 - alpha) * exp_cx
                    fused_cy = alpha * ncy + (1.0 - alpha) * exp_cy
                    fw = int(round(beta * rec["w"] + (1.0 - beta) * pred_w))
                    fh = int(round(beta * rec["h"] + (1.0 - beta) * pred_h))
                    fw, fh = max(1, fw), max(1, fh)
                    rec = make_record(
                        str(cls),
                        int(round(fused_cx - fw / 2.0)),
                        int(round(fused_cy - fh / 2.0)),
                        int(round(fused_cx + fw / 2.0)),
                        int(round(fused_cy + fh / 2.0)),
                    )
                    ncx, ncy = rec["x1"] + rec["w"] / 2.0, rec["y1"] + rec["h"] / 2.0
                    dx, dy = abs(ncx - lcx), abs(ncy - lcy)
                    max_shift_px = max(6.0, float(max_shift_ratio) * max(lw, lh))
                    la = max(1.0, float(lw * lh))
                    na = max(1.0, float(rec["w"] * rec["h"]))
                    area_factor = max(na / la, la / na)
                    if dx > max_shift_px or dy > (max_shift_px * 0.9) or area_factor > float(max_area_change):
                        ok = False
                if ok:
                    if isinstance(last_box, dict):
                        lcx = int(last_box["x1"]) + int(last_box["w"]) / 2.0
                        lcy = int(last_box["y1"]) + int(last_box["h"]) / 2.0
                        ncx = int(rec["x1"]) + int(rec["w"]) / 2.0
                        ncy = int(rec["y1"]) + int(rec["h"]) / 2.0
                        prev_vx, prev_vy = t.get("velocity", (0.0, 0.0))
                        # Smooth velocity from recent box movement.
                        t["velocity"] = (
                            (gamma * float(prev_vx)) + ((1.0 - gamma) * float(ncx - lcx)),
                            (gamma * float(prev_vy)) + ((1.0 - gamma) * float(ncy - lcy)),
                        )
                    t["last_box"] = rec
                    t["missed"] = 0
                    predicted.append(rec)
                    alive.append(t)
                    continue

        # Keep tracker briefly alive on temporary misses (motion blur/occlusion).
        t["missed"] = int(t.get("missed", 0)) + 1
        if t["missed"] <= max_missed and t.get("last_box") is not None:
            lb = dict(t["last_box"])
            # On temporary miss, extrapolate using motion velocity.
            vx, vy = t.get("velocity", (0.0, 0.0))
            if lb.get("w", 0) > 0 and lb.get("h", 0) > 0:
                x1 = int(lb["x1"] + vx)
                y1 = int(lb["y1"] + vy)
                x2 = int(lb["x2"] + vx)
                y2 = int(lb["y2"] + vy)
                lb = make_record(str(lb.get("class", cls)), x1, y1, x2, y2)
                t["last_box"] = lb
            predicted.append(lb)
            alive.append(t)
    trackers[:] = alive
    return predicted


def main() -> int:
    args = parse_args()
    video_path = Path(args.video)
    output_path = Path(args.output)

    try:
        validate_video(video_path)
    except Exception as exc:
        print(f"[Error] Invalid video path: {exc}")
        return 1

    classes = expand_classes([c.strip() for c in args.classes.split(",") if c.strip()])
    if not classes:
        print("[Error] No classes provided. Use --classes class1,class2")
        return 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Error] Unable to open video: {video_path}")
        return 1

    # Keep memory pressure lower on long annotation runs.
    try:
        cv2.setNumThreads(1)
        if hasattr(cv2, "ocl") and hasattr(cv2.ocl, "setUseOpenCL"):
            cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    frame_records: Dict[str, List[dict]] = {}
    current_boxes: List[dict] = []
    current_class_idx = 0
    frame_idx = 0
    current_frame = None

    if args.resume:
        try:
            frame_records = load_existing_records(output_path)
            last = latest_saved_frame(frame_records)
            if last > 0:
                frame_idx = last
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                print(f"[Resume] Loaded {last} frame(s); continuing from frame {last + 1}")
        except Exception as exc:
            print(f"[Resume] Could not load existing output, starting fresh: {exc}")

    drawing = False
    drag_start = (0, 0)
    drag_end = (0, 0)
    selected_idx = -1
    edit_mode = ""
    move_offset = (0, 0)
    resize_anchor = (0, 0)

    window_name = "Mouse Tracker (BBox)"
    cv2.namedWindow(window_name)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal drawing, drag_start, drag_end, current_boxes, selected_idx, edit_mode, move_offset, resize_anchor

        if event == cv2.EVENT_LBUTTONDOWN:
            hit_idx = _find_top_box_index(current_boxes, x, y)
            if hit_idx >= 0:
                selected_idx = hit_idx
                box = current_boxes[selected_idx]
                handle = _handle_hit(box, x, y)
                if handle:
                    edit_mode = f"resize:{handle}"
                    x1 = int(box["x1"])
                    y1 = int(box["y1"])
                    x2 = int(box["x2"])
                    y2 = int(box["y2"])
                    if handle == "tl":
                        resize_anchor = (x2, y2)
                    elif handle == "tr":
                        resize_anchor = (x1, y2)
                    elif handle == "bl":
                        resize_anchor = (x2, y1)
                    else:
                        resize_anchor = (x1, y1)
                else:
                    edit_mode = "move"
                    x1 = int(box["x1"])
                    y1 = int(box["y1"])
                    move_offset = (x - x1, y - y1)
                return

            selected_idx = -1
            edit_mode = "draw"
            drawing = True
            drag_start = (x, y)
            drag_end = (x, y)
            return

        if event == cv2.EVENT_MOUSEMOVE:
            if edit_mode == "draw" and drawing:
                drag_end = (x, y)
                return
            if selected_idx >= 0 and selected_idx < len(current_boxes):
                box = current_boxes[selected_idx]
                cls = box.get("class", "object")
                if edit_mode == "move":
                    w = max(1, int(box["x2"]) - int(box["x1"]))
                    h = max(1, int(box["y2"]) - int(box["y1"]))
                    nx1 = x - move_offset[0]
                    ny1 = y - move_offset[1]
                    current_boxes[selected_idx] = make_record(str(cls), nx1, ny1, nx1 + w, ny1 + h)
                    return
                if edit_mode.startswith("resize:"):
                    ax, ay = resize_anchor
                    current_boxes[selected_idx] = make_record(str(cls), ax, ay, x, y)
            return

        if event == cv2.EVENT_LBUTTONUP:
            if edit_mode == "draw" and drawing:
                drawing = False
                drag_end = (x, y)
                x1, y1, x2, y2 = norm_box(drag_start[0], drag_start[1], drag_end[0], drag_end[1])
                if (x2 - x1) == 0 and (y2 - y1) == 0:
                    # Plain click on empty area: ignore silently.
                    edit_mode = ""
                    return
                if (x2 - x1) < args.min_box_size or (y2 - y1) < args.min_box_size:
                    print(f"[Skip] frame={frame_idx + 1} small box ignored ({x2-x1}x{y2-y1})")
                    edit_mode = ""
                    return

                rec = make_record(classes[current_class_idx], x1, y1, x2, y2)
                current_boxes.append(rec)
                selected_idx = len(current_boxes) - 1
                print(
                    "[Box] "
                    f"frame={frame_idx + 1} class={rec['class']} "
                    f"x1={rec['x1']} y1={rec['y1']} x2={rec['x2']} y2={rec['y2']}"
                )
            elif selected_idx >= 0 and selected_idx < len(current_boxes):
                b = current_boxes[selected_idx]
                if int(b["x2"]) - int(b["x1"]) < args.min_box_size or int(b["y2"]) - int(b["y1"]) < args.min_box_size:
                    removed = current_boxes.pop(selected_idx)
                    print(f"[Edit] Removed too-small box class={removed['class']}")
                    selected_idx = -1
            edit_mode = ""

    cv2.setMouseCallback(window_name, on_mouse)

    print("[Info] Started bbox tracker")
    print(f"[Info] Classes: {classes}")
    frame_step = max(1, int(args.frame_step))
    auto_track = bool(args.auto_track)
    tracker_type = str(args.tracker_type).lower()
    trackers: List[dict] = []
    if auto_track and frame_step > 1:
        print("[Track] Auto-track with frame-step > 1 enabled (faster, slightly less precise)")
    print(f"[Info] Frame step: {frame_step}")
    if auto_track:
        print(
            f"[Info] Auto-track: enabled (tracker={tracker_type}, "
            f"max_shift_ratio={args.max_shift_ratio}, max_area_change={args.max_area_change}, "
            f"pred_alpha={args.pred_alpha}, pred_beta={args.pred_beta}, vel_gamma={args.vel_gamma})"
        )
    print("[Info] Controls: drag-box | click/select/move/resize | DEL delete-selected | 1..9 class | u undo | c clear-frame | n/SPACE next | q save+quit")

    while True:
        if current_frame is None:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[Info] End of video reached")
                break
            current_frame = frame
            current_boxes = []
            if auto_track and trackers:
                current_boxes = predict_boxes_from_trackers(
                    current_frame,
                    trackers,
                    max_shift_ratio=float(args.max_shift_ratio),
                    max_area_change=float(args.max_area_change),
                    pred_alpha=float(args.pred_alpha),
                    pred_beta=float(args.pred_beta),
                    vel_gamma=float(args.vel_gamma),
                )
                if current_boxes:
                    print(f"[Track] frame={frame_idx + 1} predicted {len(current_boxes)} box(es)")

        display = current_frame.copy()
        draw_annotations(display, current_boxes)

        if selected_idx >= 0 and selected_idx < len(current_boxes):
            sb = current_boxes[selected_idx]
            sx1, sy1, sx2, sy2 = int(sb["x1"]), int(sb["y1"]), int(sb["x2"]), int(sb["y2"])
            cv2.rectangle(display, (sx1, sy1), (sx2, sy2), (255, 255, 255), 2)
            for hx, hy in [(sx1, sy1), (sx2, sy1), (sx1, sy2), (sx2, sy2)]:
                cv2.rectangle(display, (hx - 4, hy - 4), (hx + 4, hy + 4), (255, 255, 255), -1)

        if drawing:
            x1, y1, x2, y2 = norm_box(drag_start[0], drag_start[1], drag_end[0], drag_end[1])
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 200, 0), 2)

        hud_line1 = f"frame={frame_idx + 1} class={classes[current_class_idx]} objects={len(current_boxes)}"
        hud_line2 = "keys: DEL delete-selected | 1..9 class | u undo | c clear | n/SPACE next | q quit"
        cv2.putText(display, hud_line1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(display, hud_line2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF

        # Delete selected box (supports Delete/Backspace variants across platforms).
        if key in (8, 127):
            if 0 <= selected_idx < len(current_boxes):
                removed = current_boxes.pop(selected_idx)
                print(f"[Delete] Removed selected {removed['class']} from frame {frame_idx + 1}")
                selected_idx = -1
            else:
                print("[Delete] No selected box")
            continue

        if key == ord("q"):
            frame_records[f"frame_{frame_idx + 1}"] = list(current_boxes)
            try:
                persist_records(output_path, frame_records)
                print("[Autosave] Progress saved")
            except Exception as exc:
                print(f"[Autosave] Save failed: {exc}")
            print("[Info] Quit requested. Saving annotations...")
            break

        if key in (ord("n"), 32):
            frame_records[f"frame_{frame_idx + 1}"] = list(current_boxes)
            print(f"[Frame] Saved frame {frame_idx + 1} with {len(current_boxes)} object(s)")
            try:
                persist_records(output_path, frame_records)
            except Exception as exc:
                print(f"[Autosave] Save failed at frame {frame_idx + 1}: {exc}")
            if auto_track:
                trackers = init_trackers(current_frame, current_boxes, tracker_type)
            # Skip ahead for faster annotation sessions.
            if frame_step > 1:
                skipped = 0
                for _ in range(frame_step - 1):
                    ok_skip, skip_frame = cap.read()
                    if not ok_skip:
                        break
                    if auto_track and trackers and skip_frame is not None:
                        # Advance tracker states across skipped frames.
                        _ = predict_boxes_from_trackers(
                            skip_frame,
                            trackers,
                            max_shift_ratio=float(args.max_shift_ratio),
                            max_area_change=float(args.max_area_change),
                            pred_alpha=float(args.pred_alpha),
                            pred_beta=float(args.pred_beta),
                            vel_gamma=float(args.vel_gamma),
                        )
                    skipped += 1
                if skipped < (frame_step - 1):
                    print("[Info] End of video reached during frame skip")
            frame_idx += frame_step
            current_frame = None
            if frame_idx % 30 == 0:
                gc.collect()
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
        persist_records(output_path, frame_records)
        print(f"[Done] Saved annotations to: {output_path.resolve()}")
    except Exception as exc:
        print(f"[Error] Failed to save output: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

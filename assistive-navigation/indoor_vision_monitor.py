#!/usr/bin/env python3
"""
Indoor vision monitor:
- YOLO indoor detections on screen
- Pre-generated indoor audio announcements
- Optional Pico serial safety phase override for beeps and speech suppression
"""
from __future__ import annotations

import argparse
import asyncio
import re
import time
import traceback

import cv2

try:
    import serial
except Exception:
    serial = None

from audio_controller import AudioController
from camera_provider import NetworkCameraProvider, VideoFileCameraProvider, draw_debug_overlay
from config import load_config
from detection import DetectionEngine
from phase_audio import PhaseToneEngine

try:
    from depth_estimator import MiDaSDepthEstimator
except Exception:
    MiDaSDepthEstimator = None

DROIDCAM_PRIMARY = "http://10.205.48.81:4747/video"
DROIDCAM_FALLBACK = "http://10.205.48.81:4747/mjpegfeed"

PATTERN_3 = re.compile(r"L:(\d),C:(\d),R:(\d)")
PATTERN_1 = re.compile(r"C:(\d)")
PATTERN_DISTANCE = re.compile(r"Distance:\s*([0-9]+(?:\.[0-9]+)?)\s*cm", re.IGNORECASE)
PATTERN_DISTANCE_EQ = re.compile(r"distance\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*cm", re.IGNORECASE)
PATTERN_DIST_C = re.compile(r"dist\s*C:\s*(None|[0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
PATTERN_STATE_C = re.compile(r"'C'\s*:\s*([0-2])")
PATTERN_NO_READING = re.compile(r"No reading", re.IGNORECASE)


def event_to_audio_key(label: str, direction: str, is_multiple: bool) -> tuple[str, str]:
    if is_multiple:
        return f"several_{label}_{direction}", f"several:{label}:{direction}"
    return f"{label}_{direction}", f"{label}:{direction}"


def _best_event_per_label(events):
    by_label = {}
    for ev in events:
        cur = by_label.get(ev.label)
        if cur is None:
            by_label[ev.label] = ev
            continue
        cur_score = (
            int(getattr(cur, "is_multiple", False)),
            int(getattr(cur, "count", 1)),
            float(getattr(cur, "confidence", 0.0)),
        )
        new_score = (
            int(getattr(ev, "is_multiple", False)),
            int(getattr(ev, "count", 1)),
            float(getattr(ev, "confidence", 0.0)),
        )
        if new_score > cur_score:
            by_label[ev.label] = ev
    return by_label


def _select_mixed_events(events, label_cursor: int, max_items: int = 2):
    by_label = _best_event_per_label(events)
    priority_order = ["person", "chair", "table"]
    labels = [l for l in priority_order if l in by_label]
    if not labels:
        return [], label_cursor
    if len(labels) > 1:
        start = label_cursor % len(labels)
        labels = labels[start:] + labels[:start]
        label_cursor = (label_cursor + 1) % len(labels)
    selected = [by_label[l] for l in labels[: max(1, max_items)]]
    return selected, label_cursor


def _flip_direction(direction: str) -> str:
    if direction == "left":
        return "right"
    if direction == "right":
        return "left"
    return direction


def _mirror_debug_detections_for_display(dets: list[dict], frame_width: int) -> list[dict]:
    out: list[dict] = []
    for d in dets:
        x1, y1, x2, y2 = d.get("bbox", (0, 0, 0, 0))
        nx1 = max(0, int(frame_width - x2))
        nx2 = max(0, int(frame_width - x1))
        nd = dict(d)
        nd["bbox"] = (nx1, int(y1), nx2, int(y2))
        nd["direction"] = _flip_direction(str(d.get("direction", "ahead")))
        out.append(nd)
    return out


def _rotate_frame(frame, rotate_mode: str):
    mode = str(rotate_mode).lower()
    if mode == "cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if mode == "ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mode == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def _direction_from_x(x_norm: float, left_threshold: float, right_threshold: float) -> str:
    x = max(0.0, min(1.0, float(x_norm)))
    if x < left_threshold:
        return "left"
    if x > right_threshold:
        return "right"
    return "ahead"


def _x_center_norm_from_det(det: dict, frame_width: int) -> float:
    if "x_center_norm" in det:
        try:
            return max(0.0, min(1.0, float(det["x_center_norm"])))
        except Exception:
            pass
    x1, _, x2, _ = det.get("bbox", (0, 0, 0, 0))
    return max(0.0, min(1.0, ((float(x1) + float(x2)) / 2.0) / float(max(1, frame_width))))


def _distance_to_phase(distance_cm: float | None, warning_cm: float, danger_cm: float) -> str:
    if distance_cm is None:
        return "SAFE"
    if distance_cm <= danger_cm:
        return "DANGER"
    if distance_cm <= warning_cm:
        return "WARNING"
    return "SAFE"


def _phase_from_serial_line(line: str, warning_cm: float, danger_cm: float) -> str | None:
    s = line.strip()
    if not s:
        return None
    if s == "NO_SIGNAL":
        return "SAFE"

    m3 = PATTERN_3.fullmatch(s)
    if m3:
        vals = [int(m3.group(1)), int(m3.group(2)), int(m3.group(3))]
        if 2 in vals:
            return "DANGER"
        if 1 in vals:
            return "WARNING"
        return "SAFE"

    m1 = PATTERN_1.fullmatch(s)
    if m1:
        c = int(m1.group(1))
        if c == 2:
            return "DANGER"
        if c == 1:
            return "WARNING"
        return "SAFE"

    md = PATTERN_DISTANCE.search(s)
    if md:
        return _distance_to_phase(float(md.group(1)), warning_cm=warning_cm, danger_cm=danger_cm)
    md2 = PATTERN_DISTANCE_EQ.search(s)
    if md2:
        return _distance_to_phase(float(md2.group(1)), warning_cm=warning_cm, danger_cm=danger_cm)
    md3 = PATTERN_DIST_C.search(s)
    if md3:
        token = str(md3.group(1)).strip().lower()
        if token == "none":
            return "SAFE"
        return _distance_to_phase(float(token), warning_cm=warning_cm, danger_cm=danger_cm)
    ms = PATTERN_STATE_C.search(s)
    if ms:
        c = int(ms.group(1))
        if c == 2:
            return "DANGER"
        if c == 1:
            return "WARNING"
        return "SAFE"

    if PATTERN_NO_READING.search(s):
        return "SAFE"

    return None


def _distance_from_serial_line(line: str) -> float | None:
    s = line.strip()
    md = PATTERN_DISTANCE.search(s)
    if md:
        return float(md.group(1))
    md2 = PATTERN_DISTANCE_EQ.search(s)
    if md2:
        return float(md2.group(1))
    md3 = PATTERN_DIST_C.search(s)
    if md3:
        token = str(md3.group(1)).strip().lower()
        if token == "none":
            return None
        return float(token)
    return None


def _majority_phase(phases: list[str]) -> str:
    if not phases:
        return "SAFE"
    counts = {"SAFE": 0, "WARNING": 0, "DANGER": 0}
    for p in phases:
        if p in counts:
            counts[p] += 1
    return max(counts.keys(), key=lambda k: (counts[k], {"SAFE": 0, "WARNING": 1, "DANGER": 2}[k]))


async def run(args: argparse.Namespace) -> None:
    config = load_config()
    detection = DetectionEngine(config)
    audio = AudioController(config)
    depth = None
    if MiDaSDepthEstimator is not None:
        depth = MiDaSDepthEstimator(
            far_threshold=float(args.depth_far_threshold),
            medium_threshold=float(args.depth_medium_threshold),
            enabled=bool(args.enable_depth),
        )
    elif args.enable_depth:
        print("[DEPTH] depth_estimator.py not found; depth disabled")
    tone = PhaseToneEngine() if args.depth_beeps else None

    camera = (
        VideoFileCameraProvider(args.video, loop=True)
        if args.video
        else NetworkCameraProvider(source=args.stream, fallback_source=args.stream_fallback)
    )

    pico_ser = None
    pico_reconnect_at = 0.0
    if args.pico_port:
        if serial is None:
            print("[PICO] pyserial not installed; Pico safety override disabled")
        else:
            try:
                pico_ser = serial.Serial(args.pico_port, args.pico_baud, timeout=0.05)
                print(f"[PICO] serial connected: {args.pico_port} @ {args.pico_baud}")
            except Exception as exc:
                print(f"[PICO] failed to open {args.pico_port}: {exc}")
                pico_ser = None
                pico_reconnect_at = time.monotonic() + 1.0

    await audio.start()
    await camera.start()

    detection_interval = float(args.detection_interval or config["app"]["detection_interval_sec"])
    detection_timeout_sec = float(args.detection_timeout)
    depth_interval_sec = max(0.05, float(args.depth_interval))
    depth_timeout_sec = max(0.05, float(args.depth_timeout))
    camera_read_timeout_sec = float(args.camera_read_timeout)
    max_fps = max(5.0, float(args.max_fps))
    frame_period = 1.0 / max_fps
    show_overlay = not bool(args.no_overlay)
    mirror = bool(args.mirror)
    input_flip = bool(args.unmirror_input)
    rotate_input = str(args.rotate_input).lower()
    left_threshold = float(args.left_threshold)
    right_threshold = float(args.right_threshold)

    if not (0.0 < left_threshold < right_threshold < 1.0):
        raise ValueError("--left-threshold and --right-threshold must satisfy: 0 < left < right < 1")

    print(
        "[IndoorVision] Controls: q=quit, f=toggle input flip, m=toggle display mirror, r=rotate input | "
        f"input_flip={input_flip} display_mirror={mirror} rotate_input={rotate_input} "
        f"zones=({left_threshold:.2f},{right_threshold:.2f})"
    )

    try:
        try:
            cv2.setNumThreads(1)
            if hasattr(cv2, "ocl") and hasattr(cv2.ocl, "setUseOpenCL"):
                cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass

        last_detection_submit = 0.0
        last_debug_dets: list[dict] = []
        detect_task: asyncio.Task | None = None
        depth_task: asyncio.Task | None = None
        last_depth_submit = 0.0
        last_depth_state = "FAR"
        last_depth_value = 1.0
        label_cursor = 0
        last_frame_time = 0.0

        pico_phase_hist: list[str] = []
        last_pico_phase = "SAFE"
        last_pico_rx_t = 0.0
        last_pico_distance: float | None = None

        while True:
            now = asyncio.get_running_loop().time()

            if args.pico_port and serial is not None and pico_ser is None and time.monotonic() >= pico_reconnect_at:
                try:
                    pico_ser = serial.Serial(args.pico_port, args.pico_baud, timeout=0.05)
                    print(f"[PICO] serial reconnected: {args.pico_port} @ {args.pico_baud}")
                except Exception:
                    pico_reconnect_at = time.monotonic() + 1.0

            if pico_ser is not None:
                try:
                    while True:
                        raw = pico_ser.readline()
                        if not raw:
                            break
                        line = raw.decode("utf-8", "ignore").strip()
                        phase = _phase_from_serial_line(
                            line,
                            warning_cm=float(args.pico_warning_cm),
                            danger_cm=float(args.pico_danger_cm),
                        )
                        dist = _distance_from_serial_line(line)
                        if dist is not None:
                            last_pico_distance = dist
                        elif PATTERN_NO_READING.search(line):
                            last_pico_distance = None
                        if phase is None:
                            continue
                        pico_phase_hist.append(phase)
                        if len(pico_phase_hist) > max(1, int(args.pico_phase_window)):
                            pico_phase_hist = pico_phase_hist[-int(args.pico_phase_window) :]
                        last_pico_phase = _majority_phase(pico_phase_hist)
                        last_pico_rx_t = time.monotonic()
                        if args.pico_debug:
                            d_txt = "-" if last_pico_distance is None else f"{last_pico_distance:.1f}cm"
                            print(f"[PICO] {line} -> {last_pico_phase} dist={d_txt}")
                    if args.pico_timeout_sec > 0 and (time.monotonic() - last_pico_rx_t) > float(args.pico_timeout_sec):
                        last_pico_phase = "SAFE"
                except Exception as exc:
                    print(f"[PICO] serial read error: {exc}")
                    try:
                        pico_ser.close()
                    except Exception:
                        pass
                    pico_ser = None
                    pico_reconnect_at = time.monotonic() + 1.0

            try:
                raw_frame = await asyncio.wait_for(camera.read(), timeout=max(0.2, camera_read_timeout_sec))
            except asyncio.TimeoutError:
                print("[CAMERA] read timeout; waiting for next frame")
                if tone is not None:
                    phase = last_pico_phase if pico_ser is not None else {"FAR": "SAFE", "MEDIUM": "WARNING", "CLOSE": "DANGER"}.get(
                        last_depth_state, "SAFE"
                    )
                    tone.play_if_due(phase)
                await asyncio.sleep(0.02)
                continue

            if raw_frame is None:
                if tone is not None:
                    phase = last_pico_phase if pico_ser is not None else {"FAR": "SAFE", "MEDIUM": "WARNING", "CLOSE": "DANGER"}.get(
                        last_depth_state, "SAFE"
                    )
                    tone.play_if_due(phase)
                await asyncio.sleep(0.02)
                continue

            if input_flip:
                raw_frame = cv2.flip(raw_frame, 1)
            raw_frame = _rotate_frame(raw_frame, rotate_input)

            processing_frame = raw_frame
            display_frame = cv2.flip(raw_frame, 1) if mirror else raw_frame.copy()

            if detect_task is not None and detect_task.done():
                try:
                    events, debug_dets = detect_task.result()
                except Exception as exc:
                    print(f"[IndoorVision] detection task error: {exc}")
                    events, debug_dets = [], []
                detect_task = None

                valid_labels = {"person", "chair", "table"}
                events = [e for e in events if e.label in valid_labels]

                event_x_map: dict[tuple[str, tuple[int, int, int, int]], float] = {}
                for d in debug_dets:
                    event_x_map[(str(d.get("label", "")).lower(), tuple(d.get("bbox", (0, 0, 0, 0))))] = _x_center_norm_from_det(
                        d, frame_width=processing_frame.shape[1]
                    )

                for ev in events:
                    x_norm = event_x_map.get((ev.label, tuple(ev.bbox)))
                    if x_norm is None:
                        x1, _, x2, _ = ev.bbox
                        x_norm = ((x1 + x2) / 2.0) / float(max(1, processing_frame.shape[1]))
                    ev.direction = _direction_from_x(x_norm, left_threshold, right_threshold)

                for d in debug_dets:
                    x_norm = _x_center_norm_from_det(d, frame_width=processing_frame.shape[1])
                    d["x_center_norm"] = round(float(x_norm), 4)
                    d["direction"] = _direction_from_x(x_norm, left_threshold, right_threshold)
                    d["zone"] = {"left": "LEFT", "right": "RIGHT", "ahead": "FRONT"}[d["direction"]]

                if mirror:
                    for ev in events:
                        ev.direction = _flip_direction(ev.direction)
                    last_debug_dets = _mirror_debug_detections_for_display(debug_dets, frame_width=display_frame.shape[1])
                else:
                    last_debug_dets = debug_dets

                selected_events, label_cursor = _select_mixed_events(events, label_cursor=label_cursor, max_items=2)

                # Safety override: suppress all object voice announcements in DANGER mode.
                if last_pico_phase == "DANGER":
                    audio.drop_pending(mode="indoor")
                    selected_events = []
                    if args.pico_debug:
                        print("[SAFETY] DANGER active -> object announcements suppressed")

                if selected_events:
                    if audio.queue.qsize() > 4:
                        audio.drop_pending(mode="indoor")
                    for ev in selected_events:
                        key, cooldown_key = event_to_audio_key(ev.label, ev.direction, ev.is_multiple)
                        audio.enqueue(
                            key=key,
                            mode="indoor",
                            priority=2,
                            cooldown_key=cooldown_key,
                            cooldown_sec=float(config["cooldowns"]["default_sec"]),
                            reason=f"vision_mix:{ev.label}:{ev.direction}:count={ev.count}",
                        )

            if depth_task is not None and depth_task.done():
                try:
                    dres = depth_task.result()
                except Exception as exc:
                    print(f"[DEPTH] failed: {exc}")
                    dres = None
                depth_task = None
                if dres is not None:
                    last_depth_value = float(dres.value)
                    last_depth_state = str(dres.state)
                    print(f"[DEPTH] value={last_depth_value:.2f} state={last_depth_state}")

            if depth is not None and depth.is_ready() and depth_task is None and (now - last_depth_submit) >= depth_interval_sec:
                last_depth_submit = now
                frame_for_depth = processing_frame.copy()
                depth_task = asyncio.create_task(
                    asyncio.wait_for(
                        asyncio.to_thread(depth.estimate, frame_for_depth),
                        timeout=depth_timeout_sec,
                    )
                )

            if detect_task is None and (now - last_detection_submit) >= detection_interval:
                last_detection_submit = now
                frame_for_detect = processing_frame.copy()
                detect_task = asyncio.create_task(
                    asyncio.wait_for(
                        asyncio.to_thread(detection.detect, frame_for_detect, "indoor", is_mirrored=False),
                        timeout=max(0.05, detection_timeout_sec),
                    )
                )

            if show_overlay:
                out = draw_debug_overlay(display_frame, last_debug_dets)
                cv2.putText(
                    out,
                    f"DEPTH {last_depth_state} ({last_depth_value:.2f})",
                    (10, out.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if pico_ser is not None:
                    color = (0, 200, 0)
                    if last_pico_phase == "WARNING":
                        color = (0, 255, 255)
                    elif last_pico_phase == "DANGER":
                        color = (0, 0, 255)
                    d_txt = "-" if last_pico_distance is None else f"{last_pico_distance:.1f}cm"
                    cv2.putText(
                        out,
                        f"PICO {last_pico_phase} {d_txt}",
                        (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow("HACKABOT Indoor Vision Monitor", out)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("f"):
                    input_flip = not input_flip
                    print(f"[IndoorVision] input_flip={input_flip}")
                if key == ord("m"):
                    mirror = not mirror
                    print(f"[IndoorVision] display_mirror={mirror}")
                if key == ord("r"):
                    order = ["none", "cw", "ccw", "180"]
                    idx = order.index(rotate_input) if rotate_input in order else 0
                    rotate_input = order[(idx + 1) % len(order)]
                    print(f"[IndoorVision] rotate_input={rotate_input}")

            if tone is not None:
                if pico_ser is not None:
                    phase = last_pico_phase
                else:
                    phase = {"FAR": "SAFE", "MEDIUM": "WARNING", "CLOSE": "DANGER"}.get(last_depth_state, "SAFE")
                tone.play_if_due(phase)

            elapsed = time.monotonic() - last_frame_time if last_frame_time else 0.0
            sleep_for = max(0.0, frame_period - elapsed)
            last_frame_time = time.monotonic()
            await asyncio.sleep(min(0.02, sleep_for))

    except KeyboardInterrupt:
        print("[IndoorVision] Stopped by user")
    except Exception as exc:
        print(f"[IndoorVision] Runtime error: {exc}")
        traceback.print_exc()
    finally:
        try:
            if "detect_task" in locals() and detect_task is not None and not detect_task.done():
                detect_task.cancel()
                try:
                    await detect_task
                except Exception:
                    pass
        except Exception:
            pass
        await camera.stop()
        await audio.stop()
        if pico_ser is not None:
            try:
                pico_ser.close()
            except Exception:
                pass
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Indoor YOLO monitor with pre-recorded audio")
    p.add_argument("--stream", type=str, default=DROIDCAM_PRIMARY, help="DroidCam stream URL")
    p.add_argument("--stream-fallback", type=str, default=DROIDCAM_FALLBACK, help="DroidCam fallback stream URL")
    p.add_argument("--video", type=str, default="", help="Optional prerecorded video path")

    p.add_argument("--enable-depth", action="store_true", help="Enable MiDaS depth side-signal")
    p.add_argument("--depth-beeps", action="store_true", help="Enable laptop phase beeps")
    p.add_argument("--depth-interval", type=float, default=0.18, help="Depth update period in seconds")
    p.add_argument("--depth-timeout", type=float, default=0.25, help="Depth task timeout in seconds")
    p.add_argument("--depth-far-threshold", type=float, default=0.70, help="Depth FAR threshold (normalized)")
    p.add_argument("--depth-medium-threshold", type=float, default=0.40, help="Depth MEDIUM threshold (normalized)")

    p.add_argument("--pico-port", type=str, default="", help="Optional Pico serial port, e.g. /dev/cu.usbmodem2101")
    p.add_argument("--pico-baud", type=int, default=115200, help="Pico serial baudrate")
    p.add_argument("--pico-warning-cm", type=float, default=120.0, help="WARNING threshold for 'Distance: X cm'")
    p.add_argument("--pico-danger-cm", type=float, default=50.0, help="DANGER threshold for 'Distance: X cm'")
    p.add_argument("--pico-phase-window", type=int, default=3, help="Majority window for serial phase")
    p.add_argument("--pico-timeout-sec", type=float, default=1.5, help="Fallback to SAFE if no serial packets")
    p.add_argument("--pico-debug", action="store_true", help="Print serial phase debug")

    p.add_argument("--detection-interval", type=float, default=0.0, help="Override detection interval (0 = config)")
    p.add_argument("--detection-timeout", type=float, default=0.35, help="YOLO timeout per detection task (sec)")
    p.add_argument("--camera-read-timeout", type=float, default=1.2, help="Camera read timeout (sec)")
    p.add_argument("--max-fps", type=float, default=30.0, help="Max preview FPS")
    p.add_argument("--no-overlay", action="store_true", help="Disable OpenCV window")
    p.add_argument("--mirror", action="store_true", help="Mirror display only (selfie-style view)")
    p.add_argument(
        "--rotate-input",
        choices=["none", "cw", "ccw", "180"],
        default="cw",
        help="Rotate incoming camera frames before YOLO/display",
    )
    p.add_argument("--left-threshold", type=float, default=0.40, help="Left zone boundary in normalized x (0..1)")
    p.add_argument("--right-threshold", type=float, default=0.60, help="Right zone boundary in normalized x (0..1)")
    p.add_argument(
        "--unmirror-input",
        action="store_true",
        help="Flip incoming stream before detection/display",
    )
    return p


if __name__ == "__main__":
    parser = build_parser()
    asyncio.run(run(parser.parse_args()))

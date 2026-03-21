#!/usr/bin/env python3
"""
Indoor vision-only monitor:
- YOLO indoor detections on screen
- Pre-generated indoor audio announcements (ElevenLabs-generated WAVs)
- No sensors, no OLED, no buzzer
"""
from __future__ import annotations

import argparse
import asyncio
import time
import traceback

import cv2

from audio_controller import AudioController
from camera_provider import NetworkCameraProvider, VideoFileCameraProvider, draw_debug_overlay
from config import load_config
from detection import DetectionEngine

DROIDCAM_PRIMARY = "http://10.205.48.81:4747/video"
DROIDCAM_FALLBACK = "http://10.205.48.81:4747/mjpegfeed"


def event_to_audio_key(label: str, direction: str, is_multiple: bool) -> tuple[str, str]:
    # detection.py returns direction in: left | ahead | right
    if is_multiple:
        return f"several_{label}_{direction}", f"several:{label}:{direction}"
    return f"{label}_{direction}", f"{label}:{direction}"


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


async def run(args: argparse.Namespace) -> None:
    config = load_config()
    detection = DetectionEngine(config)
    audio = AudioController(config)

    camera = (
        VideoFileCameraProvider(args.video, loop=True)
        if args.video
        else NetworkCameraProvider(source=args.stream, fallback_source=args.stream_fallback)
    )

    await audio.start()
    await camera.start()

    detection_interval = float(args.detection_interval or config["app"]["detection_interval_sec"])
    detection_timeout_sec = float(args.detection_timeout)
    camera_read_timeout_sec = float(args.camera_read_timeout)
    max_fps = max(5.0, float(args.max_fps))
    frame_period = 1.0 / max_fps
    show_overlay = True
    mirror = bool(args.mirror)

    try:
        last_detection_submit = 0.0
        last_debug_dets: list[dict] = []
        detect_task: asyncio.Task | None = None
        last_frame_time = 0.0
        while True:
            # Keep camera read bounded; avoid long stalls on unstable network streams.
            try:
                raw_frame = await asyncio.wait_for(camera.read(), timeout=max(0.2, camera_read_timeout_sec))
            except asyncio.TimeoutError:
                print("[CAMERA] read timeout; waiting for next frame")
                await asyncio.sleep(0.02)
                continue
            if raw_frame is None:
                await asyncio.sleep(0.02)
                continue

            # Inference always runs on raw frame.
            processing_frame = raw_frame
            display_frame = cv2.flip(raw_frame, 1) if mirror else raw_frame.copy()

            now = asyncio.get_running_loop().time()
            # Collect finished detection result (if any).
            if detect_task is not None and detect_task.done():
                try:
                    events, debug_dets = detect_task.result()
                except Exception as exc:
                    print(f"[IndoorVision] detection task error: {exc}")
                    events, debug_dets = [], []
                detect_task = None

                valid_labels = {"person", "chair", "table"}
                events = [e for e in events if e.label in valid_labels]
                if mirror:
                    for ev in events:
                        ev.direction = _flip_direction(ev.direction)
                    last_debug_dets = _mirror_debug_detections_for_display(debug_dets, frame_width=display_frame.shape[1])
                else:
                    last_debug_dets = debug_dets

                for ev in events:
                    key, cooldown_key = event_to_audio_key(ev.label, ev.direction, ev.is_multiple)
                    audio.enqueue(
                        key=key,
                        mode="indoor",
                        priority=2,
                        cooldown_key=cooldown_key,
                        cooldown_sec=float(config["cooldowns"]["default_sec"]),
                        reason=f"vision:{ev.label}:{ev.direction}:count={ev.count}",
                    )

            # Submit new detection only when previous finished (latest-frame-only, no backlog).
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
                cv2.imshow("HACKABOT Indoor Vision Monitor", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Keep UI/camera loop bounded (default max 30 FPS).
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
        await camera.stop()
        await audio.stop()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Indoor YOLO monitor with pre-recorded audio (no sensors/OLED)")
    p.add_argument("--stream", type=str, default=DROIDCAM_PRIMARY, help="DroidCam stream URL")
    p.add_argument("--stream-fallback", type=str, default=DROIDCAM_FALLBACK, help="DroidCam fallback stream URL")
    p.add_argument("--video", type=str, default="", help="Optional prerecorded video path")
    p.add_argument("--detection-interval", type=float, default=0.0, help="Override detection interval (0 = config)")
    p.add_argument("--detection-timeout", type=float, default=0.35, help="YOLO timeout per detection task (sec)")
    p.add_argument("--camera-read-timeout", type=float, default=1.2, help="Camera read timeout (sec)")
    p.add_argument("--max-fps", type=float, default=30.0, help="Max preview FPS")
    p.add_argument("--mirror", action="store_true", help="Mirror display/input (for selfie-style testing)")
    return p


if __name__ == "__main__":
    parser = build_parser()
    asyncio.run(run(parser.parse_args()))

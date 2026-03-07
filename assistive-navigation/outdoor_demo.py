"""
HACKABOT Outdoor Demo.
Detects cars, bicycles, pedestrians, traffic lights; simulates navigation.
Uses pre-recorded audio; prioritizes warnings; cooldowns per object/direction.
Run: python outdoor_demo.py [--camera 0] [--show-overlay] [--keyboard-sensor] [--simulate-obstacles] [--simulate-navigation]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import traceback

import cv2

from announcement_gate import StableAnnouncementGate
from audio_controller import AudioController
from camera_provider import VideoFileCameraProvider, WebcamCameraProvider, draw_debug_overlay
from config import load_config
from detection import DetectionEngine
from distance_sensor import KeyboardDistanceSensor
from navigation_simulator import NavigationSimulator
from obstacle_simulator import ObstacleSimulator


def event_to_audio_key(
    label: str, direction: str, is_multiple: bool, risk_level: str, config: dict
) -> tuple[str, int, str, float]:
    """Maps detection event to audio key, priority (0=highest), cooldown_key, cooldown_sec."""
    cooldowns = config["cooldowns"]
    if label == "traffic_light":
        return "traffic_light_detected", 2, "traffic_light_detected", cooldowns["several_sec"]
    if risk_level == "close":
        return f"{label}_{direction}", 2, f"{label}:{direction}", cooldowns["default_sec"]
    if is_multiple:
        return f"several_{label}_{direction}", 2, f"several:{label}:{direction}", cooldowns["several_sec"]
    return f"{label}_{direction}", 2, f"{label}:{direction}", cooldowns["default_sec"]


def select_primary_event(events: list, direction_margin: float):
    if not events:
        return None
    # Priority: confidence, bbox area, center closeness.
    ranked = sorted(
        events,
        key=lambda ev: (
            -float(getattr(ev, "confidence", 0.0)),
            -float(max(0, ev.bbox[2] - ev.bbox[0]) * max(0, ev.bbox[3] - ev.bbox[1])),
            abs((((ev.bbox[0] + ev.bbox[2]) / 2.0) / 640.0) - 0.5),
        ),
    )
    return ranked[0]


async def run(args: argparse.Namespace) -> None:
    config = load_config()
    detection = DetectionEngine(config)
    audio = AudioController(config)

    camera = VideoFileCameraProvider(args.video, loop=True) if args.video else WebcamCameraProvider(device_index=args.camera)
    sensor = KeyboardDistanceSensor() if args.keyboard_sensor else None
    obstacle_sim = ObstacleSimulator(enabled=args.simulate_obstacles)
    nav_sim = NavigationSimulator(interval_sec=args.nav_interval, enabled=args.simulate_navigation)
    stability = config.get("stability", {})
    gate = StableAnnouncementGate(
        min_consistent_frames=int(stability.get("min_consistent_frames", 3)),
        history_size=int(stability.get("history_size", 5)),
        min_confidence=float(stability.get("min_confidence", 0.45)),
        repeat_same_direction_sec=float(stability.get("outdoor_repeat_same_direction_sec", stability.get("repeat_same_direction_sec", 10.0))),
        track_lost_reset_sec=float(stability.get("track_lost_reset_sec", 4.0)),
        key_mode="label_direction",
    )

    await audio.start()
    await camera.start()
    if sensor:
        await sensor.start()
    await obstacle_sim.start()
    await nav_sim.start()

    detection_interval = float(args.detection_interval or config["app"]["detection_interval_sec"])
    show_overlay = bool(args.show_overlay or config["app"].get("show_overlay", False))
    vision_cfg = config.get("vision", {})
    max_event_age_ms = int(vision_cfg.get("max_event_age_ms", 1500))
    process_timeout_ms = int(vision_cfg.get("max_processing_ms", 220))
    direction_margin = float(vision_cfg.get("direction_margin", 0.15))
    # Processing frame must stay non-mirrored for direction correctness.
    flip_for_processing = False
    camera_cfg = config.get("camera", {})
    if getattr(args, "mirror_input", None) is not None:
        flip_for_display = bool(args.mirror_input)
    else:
        base_mirror = bool(camera_cfg.get("flip_for_display", config["app"].get("mirror_input_outdoor", config["app"].get("mirror_input", True))))
        flip_for_display = base_mirror and not bool(args.no_flip)

    try:
        last_detection = 0.0
        last_debug_dets: list[dict] = []
        while True:
            frame = await camera.read()
            if frame is None:
                await asyncio.sleep(0.03)
                continue
            processing_frame = frame if not flip_for_processing else cv2.flip(frame, 1)
            user_frame = cv2.flip(frame, 1) if flip_for_display else frame.copy()
            if flip_for_processing:
                print("[Vision] FRAME MIRRORING VIOLATION: inference frame must be non-mirrored")
            if show_overlay and not flip_for_display:
                print("[Vision] FRAME MIRRORING VIOLATION: user display expected mirrored in UI mode")

            now = asyncio.get_running_loop().time()
            if now - last_detection >= detection_interval:
                last_detection = now
                try:
                    events, last_debug_dets = await asyncio.wait_for(
                        asyncio.to_thread(detection.detect, processing_frame, "outdoor", is_mirrored=False),
                        timeout=max(0.05, process_timeout_ms / 1000.0),
                    )
                except asyncio.TimeoutError:
                    print("[Vision] detection skipped: processing timeout")
                    events, last_debug_dets = [], []
                now_ms = int(time.time() * 1000)
                stale_cutoff_ms = now_ms - max_event_age_ms
                schema_events = [
                    {
                        "label": str(d["label"]).upper(),
                        "x_center_norm": float(d["x_center_norm"]),
                        "y_center_norm": float(d["y_center_norm"]),
                        "zone": str(d["zone"]).upper(),
                        "confidence": float(d["confidence"]),
                        "timestamp_ms": int(d["timestamp_ms"]),
                        "frame_id": int(d["frame_id"]),
                    }
                    for d in last_debug_dets
                    if int(d.get("timestamp_ms", 0)) >= stale_cutoff_ms
                ]
                print(
                    "[VisionSchema] "
                    + json.dumps(
                        {
                            "frame_id": int(schema_events[0]["frame_id"]) if schema_events else -1,
                            "timestamp_ms": int(schema_events[0]["timestamp_ms"]) if schema_events else 0,
                            "objects": schema_events,
                        },
                        separators=(",", ":"),
                    )
                )
                primary = select_primary_event(events, direction_margin=direction_margin)
                if primary is not None:
                    zone = {"left": "LEFT", "right": "RIGHT", "ahead": "FRONT"}.get(primary.direction, "FRONT")
                    matched = next((d for d in schema_events if d["label"] == str(primary.label).upper() and d["zone"] == zone), None)
                    if matched is not None:
                        requires_stability = primary.label != "traffic_light"
                        if (not requires_stability) or gate.allow(primary, now):
                            key, priority, cooldown_key, cooldown_sec = event_to_audio_key(
                                primary.label, primary.direction, primary.is_multiple, primary.risk_level, config
                            )
                            audio.enqueue(
                                key=key,
                                mode="outdoor",
                                priority=priority,
                                cooldown_key=cooldown_key,
                                cooldown_sec=cooldown_sec,
                                reason=f"det:{primary.label}:{primary.direction}:count={primary.count}:risk={primary.risk_level}",
                            )
                            print(
                                "[FusionDecision] "
                                + json.dumps(
                                    {
                                        "frame_id": int(matched.get("frame_id", -1)),
                                        "timestamp_ms": int(matched.get("timestamp_ms", 0)),
                                        "reason": f"det:{primary.label}:{primary.direction}",
                                        "audio_key": key,
                                    },
                                    separators=(",", ":"),
                                )
                            )

            if sensor:
                sensor_ev = await sensor.read()
                if sensor_ev is not None:
                    key = "car_ahead"
                    audio.enqueue(
                        key=key,
                        mode="outdoor",
                        priority=1,
                        cooldown_key=f"sensor:{sensor_ev.level}",
                        cooldown_sec=config["cooldowns"]["default_sec"],
                        reason=f"sensor:{sensor_ev.level}",
                    )

            obstacle_ev = await obstacle_sim.read()
            if obstacle_ev:
                key = "car_ahead"
                audio.enqueue(
                    key=key,
                    mode="outdoor",
                    priority=1,
                    cooldown_key=f"sim:{obstacle_ev.level}",
                    cooldown_sec=config["cooldowns"]["default_sec"],
                    reason=f"obstacle_sim:{obstacle_ev.level}",
                )

            nav_ev = await nav_sim.read()
            if nav_ev:
                audio.enqueue(
                    key=nav_ev.key,
                    mode="outdoor",
                    priority=3,
                    cooldown_key=f"nav:{nav_ev.key}",
                    cooldown_sec=config["cooldowns"]["navigation_sec"],
                    reason="navigation",
                )

            if show_overlay:
                raw_out = draw_debug_overlay(processing_frame.copy(), last_debug_dets)
                user_out = draw_debug_overlay(user_frame, last_debug_dets)
                cv2.imshow("HACKABOT Outdoor RAW (Not Mirrored)", raw_out)
                cv2.imshow("HACKABOT Outdoor USER VIEW", user_out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            await asyncio.sleep(0.01)
    except KeyboardInterrupt:
        print("[OutdoorDemo] Stopped by user")
    except Exception as exc:
        print(f"[OutdoorDemo] Runtime error: {exc}")
        traceback.print_exc()
    finally:
        await nav_sim.stop()
        await obstacle_sim.stop()
        if sensor:
            await sensor.stop()
        await camera.stop()
        await audio.stop()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HACKABOT outdoor demo")
    p.add_argument("--camera", type=int, default=0, help="Webcam index")
    p.add_argument("--video", type=str, default="", help="Optional prerecorded video path")
    p.add_argument("--show-overlay", action="store_true", help="Show OpenCV overlay")
    p.add_argument("--no-flip", action="store_true", help="Disable mirrored USER VIEW window")
    p.add_argument("--mirror-input", dest="mirror_input", action="store_true", default=None, help="Mirror USER VIEW window (inference stays non-mirrored)")
    p.add_argument("--no-mirror-input", dest="mirror_input", action="store_false", help="Do not mirror USER VIEW window")
    p.add_argument("--keyboard-sensor", action="store_true", help="Enable keyboard distance sensor (O/C)")
    p.add_argument("--simulate-obstacles", action="store_true", help="Enable synthetic obstacle events")
    p.add_argument("--simulate-navigation", action="store_true", help="Enable synthetic nav messages")
    p.add_argument("--nav-interval", type=float, default=6.0, help="Navigation simulator interval (sec)")
    p.add_argument("--detection-interval", type=float, default=0.0, help="Override detection interval (0 = use config)")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run(args))

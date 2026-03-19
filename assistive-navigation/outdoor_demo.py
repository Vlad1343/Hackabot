"""
HACKABOT Outdoor Demo.
Detects cars, bicycles, pedestrians, traffic lights; simulates navigation.
Uses pre-recorded audio; prioritizes warnings; cooldowns per object/direction.
Run: python outdoor_demo.py [--camera 0] [--show-overlay] [--keyboard-sensor] [--simulate-obstacles] [--simulate-navigation]
"""
from __future__ import annotations

import argparse
import asyncio
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
    if getattr(args, "mirror_input", None) is not None:
        flip_camera = bool(args.mirror_input)
    else:
        base_mirror = bool(config["app"].get("mirror_input_outdoor", config["app"].get("mirror_input", True)))
        flip_camera = base_mirror and not bool(args.no_flip)

    try:
        last_detection = 0.0
        last_debug_dets: list[dict] = []
        while True:
            frame = await camera.read()
            if frame is None:
                await asyncio.sleep(0.03)
                continue
            if flip_camera:
                frame = cv2.flip(frame, 1)

            now = asyncio.get_running_loop().time()
            if now - last_detection >= detection_interval:
                last_detection = now
                events, last_debug_dets = detection.detect(frame, mode="outdoor")
                for ev in events:
                    requires_stability = ev.label != "traffic_light"
                    if requires_stability and not gate.allow(ev, now):
                        continue
                    key, priority, cooldown_key, cooldown_sec = event_to_audio_key(
                        ev.label, ev.direction, ev.is_multiple, ev.risk_level, config
                    )
                    audio.enqueue(
                        key=key,
                        mode="outdoor",
                        priority=priority,
                        cooldown_key=cooldown_key,
                        cooldown_sec=cooldown_sec,
                        reason=f"det:{ev.label}:{ev.direction}:count={ev.count}:risk={ev.risk_level}",
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
                out = draw_debug_overlay(frame, last_debug_dets)
                cv2.imshow("HACKABOT Outdoor Demo", out)
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
    p.add_argument("--no-flip", action="store_true", help="Disable default horizontal camera flip")
    p.add_argument("--mirror-input", dest="mirror_input", action="store_true", default=None, help="Force mirrored camera input")
    p.add_argument("--no-mirror-input", dest="mirror_input", action="store_false", help="Force non-mirrored camera input")
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

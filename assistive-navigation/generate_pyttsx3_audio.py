"""
HACKABOT Offline Audio Generation (pyttsx3).
Generates WAV files for low-latency demo; no API required.
Files under audio/indoors and audio/outdoors matching playback logic exactly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyttsx3

BASE_DIR = Path(__file__).resolve().parent / "audio"
INDOORS = BASE_DIR / "indoors"
OUTDOORS = BASE_DIR / "outdoors"


def phrase_catalog() -> dict[str, dict[str, str]]:
    """Exact keys matching audio_controller key resolution (object_direction, several_object_direction, object_close, object_approaching)."""
    indoor = {
        "person_ahead": "Person ahead",
        "person_left": "Person on your left",
        "person_right": "Person on your right",
        "several_person_ahead": "Several people ahead",
        "several_person_left": "Several people on your left",
        "several_person_right": "Several people on your right",
        "person_close": "Person very close",
        "chair_ahead": "Chair ahead",
        "chair_left": "Chair on your left",
        "chair_right": "Chair on your right",
        "several_chair_ahead": "Several chairs ahead",
        "several_chair_left": "Several chairs on your left",
        "several_chair_right": "Several chairs on your right",
        "chair_close": "Chair very close",
        "table_ahead": "Table ahead",
        "table_left": "Table on your left",
        "table_right": "Table on your right",
        "several_table_ahead": "Several tables ahead",
        "several_table_left": "Several tables on your left",
        "several_table_right": "Several tables on your right",
        "table_close": "Table very close",
        "door_ahead": "Door ahead",
        "door_left": "Door on your left",
        "door_right": "Door on your right",
        "several_door_ahead": "Several doors ahead",
        "several_door_left": "Several doors on your left",
        "several_door_right": "Several doors on your right",
        "door_close": "Door very close",
        "wall_ahead": "Wall ahead",
        "wall_close": "Wall very close",
        "path_clear": "Path is clear",
        "turn_slightly_left": "Turn slightly left",
        "turn_slightly_right": "Turn slightly right",
    }
    outdoor = {
        "car_ahead": "Car ahead",
        "car_left": "Car on your left",
        "car_right": "Car on your right",
        "several_car_ahead": "Several cars ahead",
        "several_car_left": "Several cars on your left",
        "several_car_right": "Several cars on your right",
        "car_close": "Car very close",
        "bicycle_ahead": "Bicycle ahead",
        "bicycle_left": "Bicycle on your left",
        "bicycle_right": "Bicycle on your right",
        "several_bicycle_ahead": "Several bicycles ahead",
        "several_bicycle_left": "Several bicycles on your left",
        "several_bicycle_right": "Several bicycles on your right",
        "bicycle_close": "Bicycle very close",
        "pedestrian_ahead": "Pedestrian ahead",
        "pedestrian_left": "Pedestrian on your left",
        "pedestrian_right": "Pedestrian on your right",
        "several_pedestrian_ahead": "Several pedestrians ahead",
        "several_pedestrian_left": "Several pedestrians on your left",
        "several_pedestrian_right": "Several pedestrians on your right",
        "pedestrian_close": "Pedestrian very close",
        "traffic_light_detected": "Traffic light detected",
        "red_light": "Red light",
        "green_light": "Green light",
        "safe_to_cross": "Safe to cross",
        "path_clear": "Path is clear",
        "turn_slightly_left": "Turn slightly left",
        "turn_slightly_right": "Turn slightly right",
    }
    return {"indoor": indoor, "outdoor": outdoor}


def generate(engine: pyttsx3.Engine, out_dir: Path, items: dict[str, str], overwrite: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, text in items.items():
        target = out_dir / f"{key}.wav"
        if target.exists() and not overwrite:
            print(f"[Skip] {target.name}")
            continue
        print(f"[Write] {target.name} -> {text}")
        engine.save_to_file(text, str(target))
    engine.runAndWait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline WAV files with pyttsx3")
    parser.add_argument("--mode", choices=["indoor", "outdoor", "all"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rate", type=int, default=165)
    parser.add_argument("--volume", type=float, default=1.0)
    args = parser.parse_args()

    engine = pyttsx3.init()
    engine.setProperty("rate", args.rate)
    engine.setProperty("volume", max(0.0, min(1.0, args.volume)))

    catalog = phrase_catalog()
    if args.mode in ("indoor", "all"):
        generate(engine, INDOORS, catalog["indoor"], args.overwrite)
    if args.mode in ("outdoor", "all"):
        generate(engine, OUTDOORS, catalog["outdoor"], args.overwrite)


if __name__ == "__main__":
    main()

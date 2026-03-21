from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(RuntimeError):
    pass


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path = "config.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path

    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as exc:
        raise ConfigError(f"Failed to load config from {path}: {exc}") from exc

    defaults = {
        "app": {
            "default_mode": "indoor",
            "detection_interval_sec": 0.7,
            "crop_top_ratio": 0.08,
            "crop_bottom_ratio": 0.12,
            "input_size": 416,
            "show_overlay": True,
            "mirror_input": True,
            "mirror_input_indoor": True,
            "mirror_input_outdoor": False,
        },
        "cooldowns": {
            "default_sec": 3.0,
            "indoor_direction_sec": 1.8,
            "close_sec": 2.2,
            "navigation_sec": 5.5,
            "several_sec": 4.0,
            "min_gap_sec": 1.2,
        },
        "detection": {
            "confidence_threshold": 0.30,
            "close_area_ratio": 0.22,
            "approaching_area_ratio": 0.12,
            "left_boundary": 0.33,
            "right_boundary": 0.66,
            "left_boundary_indoor": 0.33,
            "right_boundary_indoor": 0.66,
            "left_boundary_outdoor": 0.33,
            "right_boundary_outdoor": 0.66,
        },
        "direction": {
            "left_threshold": 0.33,
            "right_threshold": 0.66,
        },
        "vision": {
            "max_event_age_ms": 1500,
            "smoothing_window": 3,
            "direction_margin": 0.15,
            "latest_frame_only": True,
        },
        "runtime": {
            "drop_backlog": True,
        },
        "stability": {
            "min_consistent_frames": 3,
            "history_size": 5,
            "min_confidence": 0.45,
            "repeat_same_direction_sec": 2.5,
            "indoor_repeat_same_direction_sec": 1.8,
            "outdoor_repeat_same_direction_sec": 5.0,
            "track_lost_reset_sec": 4.0,
        },
        "audio": {
            "enabled": True,
            "folder": "audio/indoors",
            "multi_object_threshold": 2,
            "cooldown_ms": 1000,
            "confidence_threshold": 0.3,
            "base_dir": "audio",
            "indoors_dir": "indoors",
            "outdoors_dir": "outdoors",
            "extension": ".wav",
            "file_overrides": {},
            "navigation_keys": ["path_clear", "turn_slightly_left", "turn_slightly_right"],
        },
        "model": {"weights_path": "yolov8n.pt"},
        "class_aliases": {},
        "classes": {"indoor": ["person", "chair", "table", "door", "wall"], "outdoor": ["car", "bicycle", "pedestrian", "traffic_light"]},
    }

    merged = _merge_dicts(defaults, config)
    merged["_config_path"] = str(path)
    merged["_project_dir"] = str(Path(__file__).resolve().parent)
    return merged


def resolve_path(*parts: str) -> Path:
    return Path(__file__).resolve().parent.joinpath(*parts)

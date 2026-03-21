# HACKABOT Assistive Navigation

Hackathon-ready indoor/outdoor assistive navigation demo with YOLOv8 detection, prioritized audio alerts, cooldown suppression, simulators, and hardware-ready interfaces.

## Quick Start

```bash
cd assistive-navigation
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Generate audio (required before first run; use pyttsx3 for offline demo):

```bash
python generate_pyttsx3_audio.py --mode all --overwrite
```

YOLOv8 nano weights (`yolov8n.pt`) are downloaded automatically on first detection run.

## Run Indoor Demo

```bash
python indoor_demo.py --show-overlay --keyboard-sensor --simulate-navigation
```

## Run Outdoor Demo

```bash
python outdoor_demo.py --show-overlay --keyboard-sensor --simulate-navigation
```

## Unified Entry Point

```bash
python main.py --mode indoor --show-overlay
python main.py --mode outdoor --video /path/to/video.mp4 --show-overlay
```

Recommended quick commands:

```bash
./run_indoor.sh
./run_outdoor.sh
```

## Keyboard Sensor Controls

When `--keyboard-sensor` is enabled:
- `o` + Enter: obstacle approaching
- `c` + Enter: obstacle very close
- `q` + Enter: stop keyboard sensor thread

## Audio Generation

Offline generation (fast demo fallback):

```bash
python generate_pyttsx3_audio.py --mode all --overwrite
```

ElevenLabs generation:

```bash
# .env must include ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID
python generate_elevenlabs_audio.py --mode all --overwrite
```

## Notes

- Audio events are logged to `assistive-navigation/logs/audio_events.csv`.
- Missing audio files are reported but do not crash the pipeline.
- Replace `PiCameraProviderStub` and `DistanceSensor` implementations for hardware without changing core demo logic.

## Two-Pico Embedded MVP (Hackathon Path)

Added standalone files in `assistive-navigation/`:
- `pico_a_sensor_node.py` (MicroPython): HC-SR04 -> classify -> nRF24 TX
- `pico_b_relay_node.py` (MicroPython): nRF24 RX -> USB serial relay
- `laptop_relay_monitor.py` (Python): serial -> text/voice output

### Run Order

1. Flash `pico_a_sensor_node.py` to Pico A (`main.py`) and wire HC-SR04 (+ voltage divider on ECHO).
2. Flash `pico_b_relay_node.py` to Pico B (`main.py`) and wire nRF24.
3. On laptop:

```bash
cd assistive-navigation
source ../.venv/bin/activate
pip install -r requirements.txt
python laptop_relay_monitor.py --port /dev/tty.usbmodemXXXX --voice
```

If you start with one sensor, set `USE_THREE_SENSORS = False` in `pico_a_sensor_node.py`.

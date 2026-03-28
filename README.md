# Stepwise

> **🥈 2nd Place at Hack-A-Bot Creative Challenge** <br>
> – Real-Time Assistive Navigation Project <br>
> – Accessibility-Focused AI System for Safer Mobility (sponsored by **Arm** and **EEESoc**)

---

Stepwise is a **real-time assistive navigation system** designed for **visually impaired users**, combining computer vision, embedded sensing, and low-latency wireless communication into a unified **safety-first pipeline** for indoor and outdoor environments.

---

## How It Works

Stepwise combines two independent inputs into a single decision:

- **Camera** (via phone): captures the environment and runs **YOLOv8** to detect objects and their direction (left, ahead, right)
- **Ultrasonic sensor** (via Pico): measures distance to obstacles and determines risk level (**SAFE**, **WARNING**, **DANGER**)

These signals are processed together on the laptop:

- **Distance** is treated as safety-critical
- Object detection provides **context only**
- When there is a conflict, **proximity always takes priority**

The final output is converted into audio feedback:

- **Voice instructions** for objects and direction
- **Beep patterns** for immediate danger levels

---

## Stepwise System Flow

1. Camera stream and ultrasonic sensing run in parallel
2. YOLOv8 detects objects and estimates direction (left, ahead, right)
3. Ultrasonic pipeline classifies proximity (SAFE, WARNING, DANGER)
4. Decision layer prioritises safety state over visual semantics
5. Audio engine emits controlled, non-overlapping feedback (voice + tones)
6. Logging layer maintains visibility for debugging and system stability

---

## Core Features

- **Real-time assistive perception** for indoor and outdoor mobility
- **YOLOv8-based object detection** with directional awareness
- **Ultrasonic hazard detection** with immediate risk classification
- **Sensor fusion** with strict safety-first prioritisation
- **Anti-spam audio system** (cooldowns, de-duplication, event gating)
- **Robust camera streaming** with automatic recovery
- **Modular architecture** for vision, sensing, and feedback layers
- **Wireless fail-safe design** with packet validation and fallback logic

---

## System Architecture

### Architecture Modes

| Mode | Data Path | Purpose |
|---|---|---|
| Demo runtime (current) | HC-SR04 -> Pico (sensor + classification) -> USB Serial -> Laptop -> Audio + Visual Output | Fast setup for controlled demos and development |
| Target wireless runtime | HC-SR04 -> Pico 1 -> nRF24L01 link -> Pico 2 -> USB Serial -> Laptop feedback system | Field-oriented reliability and wireless decoupling |

### Two-Pico Node Responsibilities

| Node | Responsibility | Why it matters |
|---|---|---|
| Pico 1 (sensor node) | Real-time sensing and immediate risk classification | Keeps proximity decisions low-latency at the edge |
| Pico 2 (gateway node) | Wireless reliability checks, packet validation, stable forwarding to laptop | Prevents noisy RF conditions from destabilising user feedback |

### End-to-End Safety Pipeline

```text
Sensor Plane   : HC-SR04 -> Pico 1 classification
Transport Plane: nRF24L01 link -> Pico 2 validation -> USB Serial
Compute Plane  : Laptop fusion engine (vision + risk priority)
Output Plane   : Voice guidance + danger beeps + visual overlay
```

---

## Vision + Safety Fusion

| Input Stream | Processing | Output Signal |
|---|---|---|
| Phone camera (DroidCam) | YOLOv8 inference | Directional semantic events |
| Ultrasonic sensor | Risk classification | Safety state (SAFE/WARNING/DANGER) |

Fusion rule: **safety state always has priority** when semantic and proximity signals conflict.

Final output: **prioritised audio feedback** (voice + tones).

---

## Hardware Stack

- Raspberry Pi Pico (dual-node architecture)
- HC-SR04 ultrasonic distance sensor
- nRF24L01 wireless modules with external antennas
- Custom-made 3D-printed phone case + integrated mounting enclosure
- Voltage level shifting for safe GPIO interfacing
- Optional power stabilisation capacitors for RF reliability

---

## Software Stack

- Python-based orchestration and runtime control
- OpenCV video ingestion and frame handling
- Ultralytics YOLOv8 for real-time object detection
- Audio engine (voice prompts + tone-based alerts)
- Serial communication bridge (Pico telemetry)
- Wireless diagnostics and RF monitoring

---

## Audio & Feedback Logic

The system enforces a **strict single-channel output model**:

- **No overlapping audio events**
- **Cooldown-based repetition control**
- **Priority order**: DANGER > WARNING > SAFE

### Output Types

- Directional voice cues (object + position)
- Safety tones mapped to risk level
- Event suppression during high-frequency detection bursts

---

## Reliability Engineering

Stepwise is designed for **unstable real-world conditions**:

- **Automatic camera reconnection** on stream failure
- **Serial port recovery** for Pico re-enumeration
- **RF packet validation** (MAX_RT, NO_SIGNAL)
- **Temporal smoothing** of sensor noise
- **Debounce logic** for unstable ultrasonic readings
- **Logging layer** for traceability of system state

---

## Indoor and Outdoor Behaviour

- **Indoor**: close-range obstacle avoidance (people, furniture, walls)
- **Outdoor**: dynamic hazards (traffic, crossings, moving objects)
- **Core rule**: proximity overrides semantics

---

## Project Gallery

<p align="center">
	<img src="photos/1.png" alt="Stepwise Prototype 1" width="400" />
</p>

<p align="center">
	<img src="photos/2.png" alt="Stepwise Prototype 2" width="400" />
</p>

<p align="center">
	<img src="photos/3.png" alt="Stepwise Prototype 3" width="1000" />
</p>

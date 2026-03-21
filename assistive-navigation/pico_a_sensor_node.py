"""
Pico A (MicroPython): Ultrasonic sensor node -> nRF24 TX.

Purpose:
- Read 1 or 3 HC-SR04 sensors
- Classify SAFE/WARNING/DANGER per direction
- Transmit compact packet to Pico B over nRF24

Requires on Pico filesystem:
- nrf24l01.py (MicroPython nRF24 driver)
"""

from machine import Pin, SPI, time_pulse_us
import utime

try:
    from nrf24l01 import NRF24L01
except Exception:
    print("ERROR: missing nrf24l01.py on Pico A")
    raise

# -----------------------------
# Config (edit for your wiring)
# -----------------------------
USE_THREE_SENSORS = True

# HC-SR04 pins (BCM-like GP numbers for Pico)
TRIG_FRONT = 2
ECHO_FRONT = 3
TRIG_LEFT = 4
ECHO_LEFT = 5
TRIG_RIGHT = 6
ECHO_RIGHT = 7

# Distance thresholds (cm)
DANGER_CM = 30.0
SAFE_CM = 100.0

# Timing
LOOP_MS = 100  # ~10Hz
ECHO_TIMEOUT_US = 30000

# nRF24 pins/settings
SPI_ID = 0
SCK_PIN = 18
MOSI_PIN = 19
MISO_PIN = 16
CE_PIN = 20
CSN_PIN = 17
CHANNEL = 108

PIPE_TX = b"\xe1\xf0\xf0\xf0\xf0"
PIPE_RX = b"\xd2\xf0\xf0\xf0\xf0"


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class HCSR04:
    def __init__(self, trig_pin: int, echo_pin: int):
        self.trig = Pin(trig_pin, Pin.OUT)
        self.echo = Pin(echo_pin, Pin.IN)
        self.trig.value(0)

    def read_cm_once(self):
        self.trig.value(0)
        utime.sleep_us(2)
        self.trig.value(1)
        utime.sleep_us(10)
        self.trig.value(0)
        pulse = time_pulse_us(self.echo, 1, ECHO_TIMEOUT_US)
        if pulse < 0:
            return None
        d = (pulse * 0.0343) / 2.0
        if d <= 0 or d > 500:
            return None
        return d

    def read_cm_filtered(self):
        vals = []
        for _ in range(3):
            v = self.read_cm_once()
            if v is not None:
                vals.append(v)
            utime.sleep_ms(20)
        if not vals:
            return None
        vals.sort()
        if len(vals) >= 3:
            return vals[1]  # median of 3
        return sum(vals) / len(vals)


def classify(distance_cm):
    if distance_cm is None:
        return 0  # SAFE fallback
    if distance_cm <= DANGER_CM:
        return 2  # DANGER
    if distance_cm <= SAFE_CM:
        return 1  # WARNING
    return 0


def build_packet(states):
    # Compact packet for reliability: L:x,C:y,R:z
    if USE_THREE_SENSORS:
        return "L:{},C:{},R:{}".format(states["L"], states["C"], states["R"])
    return "C:{}".format(states["C"])


def main():
    front = HCSR04(TRIG_FRONT, ECHO_FRONT)
    left = HCSR04(TRIG_LEFT, ECHO_LEFT) if USE_THREE_SENSORS else None
    right = HCSR04(TRIG_RIGHT, ECHO_RIGHT) if USE_THREE_SENSORS else None

    spi = SPI(
        SPI_ID,
        baudrate=4000000,
        polarity=0,
        phase=0,
        sck=Pin(SCK_PIN),
        mosi=Pin(MOSI_PIN),
        miso=Pin(MISO_PIN),
    )
    nrf = NRF24L01(spi, Pin(CSN_PIN), Pin(CE_PIN), channel=CHANNEL, payload_size=32)
    nrf.open_tx_pipe(PIPE_TX)
    nrf.open_rx_pipe(1, PIPE_RX)
    nrf.stop_listening()

    print("Pico A started (sensor TX)")
    while True:
        d_front = front.read_cm_filtered()
        states = {"L": 0, "C": classify(d_front), "R": 0}

        if USE_THREE_SENSORS:
            d_left = left.read_cm_filtered() if left else None
            d_right = right.read_cm_filtered() if right else None
            states["L"] = classify(d_left)
            states["R"] = classify(d_right)
            print("dist L/C/R:", d_left, d_front, d_right, "state", states)
        else:
            print("dist C:", d_front, "state", states)

        packet = build_packet(states)
        try:
            nrf.send(packet.encode("utf-8"))
            print("tx:", packet)
        except Exception as exc:
            print("tx_error:", exc)

        utime.sleep_ms(LOOP_MS)


main()


"""
Pico HC-SR04 quick distance test (MicroPython).

How to use:
1) Copy this file to Raspberry Pi Pico as main.py (or run manually from REPL).
2) Wire HC-SR04:
   - VCC -> 5V
   - GND -> GND
   - TRIG -> Pico GP2  (default below)
   - ECHO -> Pico GP3  (default below) THROUGH VOLTAGE DIVIDER to 3.3V
3) Open serial console and read printed distances.
"""

from machine import Pin, time_pulse_us
import utime

# Change these pins if your wiring is different.
TRIG_PIN = 2
ECHO_PIN = 3

# Thresholds for simple state classification.
DANGER_CM = 30.0
SAFE_CM = 100.0

# Echo timeout (microseconds). 30000us ~= ~5m max pulse window.
ECHO_TIMEOUT_US = 30000

trig = Pin(TRIG_PIN, Pin.OUT)
echo = Pin(ECHO_PIN, Pin.IN)
trig.value(0)


def read_distance_cm():
    # 10us trigger pulse
    trig.value(0)
    utime.sleep_us(2)
    trig.value(1)
    utime.sleep_us(10)
    trig.value(0)

    pulse = time_pulse_us(echo, 1, ECHO_TIMEOUT_US)
    if pulse < 0:
        return None

    # Distance = speed_of_sound * time / 2
    return (pulse * 0.0343) / 2.0


def classify(d_cm):
    if d_cm is None:
        return "NO_READING"
    if d_cm < DANGER_CM:
        return "DANGER"
    if d_cm < SAFE_CM:
        return "WARNING"
    return "SAFE"


def median3(a, b, c):
    vals = [a, b, c]
    vals.sort()
    return vals[1]


print("Pico HC-SR04 quick check started")
print("TRIG=GP{}, ECHO=GP{}".format(TRIG_PIN, ECHO_PIN))

while True:
    samples = []
    for _ in range(3):
        d = read_distance_cm()
        if d is not None and 1.0 <= d <= 500.0:
            samples.append(d)
        utime.sleep_ms(20)

    if len(samples) >= 3:
        d_filtered = median3(samples[0], samples[1], samples[2])
    elif len(samples) > 0:
        d_filtered = sum(samples) / len(samples)
    else:
        d_filtered = None

    state = classify(d_filtered)
    if d_filtered is None:
        print("distance=None state={}".format(state))
    else:
        print("distance={:.1f}cm state={}".format(d_filtered, state))

    utime.sleep_ms(200)


"""
Pico HC-SR04 quick distance test (MicroPython).

Serial output intentionally matches Arduino-like format so laptop tools can parse:
- "Distance: <value> cm"
- "No reading"
"""

from machine import Pin, time_pulse_us
import utime

# Change these pins if your wiring is different.
# Default matches your previously working Arduino sketch:
#   TRIG_PIN=3, ECHO_PIN=2
TRIG_PIN = 3
ECHO_PIN = 2

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


print("HC-SR04 test starting...")
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

    if d_filtered is None:
        print("No reading")
    else:
        print("Distance: {:.1f} cm".format(d_filtered))

    utime.sleep_ms(500)

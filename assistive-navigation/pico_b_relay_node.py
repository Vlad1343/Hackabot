"""
Pico B (MicroPython): nRF24 RX -> USB serial relay.

Purpose:
- Receive packets from Pico A over nRF24
- Relay them to laptop via USB serial using print()

Requires on Pico filesystem:
- nrf24l01.py
"""

from machine import Pin, SPI
import utime

try:
    from nrf24l01 import NRF24L01
except Exception:
    print("ERROR: missing nrf24l01.py on Pico B")
    raise

# -----------------------------
# Config (edit for your wiring)
# -----------------------------
SPI_ID = 0
SCK_PIN = 18
MOSI_PIN = 19
MISO_PIN = 16
CE_PIN = 20
CSN_PIN = 17
CHANNEL = 108

PIPE_TX = b"\xd2\xf0\xf0\xf0\xf0"
PIPE_RX = b"\xe1\xf0\xf0\xf0\xf0"

NO_SIGNAL_TIMEOUT_MS = 1500


def main():
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
    nrf.start_listening()

    last_rx = utime.ticks_ms()
    print("Pico B started (relay RX)")

    while True:
        got = False
        try:
            if nrf.any():
                raw = nrf.recv()
                msg = raw.decode("utf-8", "ignore").strip("\x00").strip()
                if msg:
                    print(msg)  # USB serial output to laptop
                    last_rx = utime.ticks_ms()
                    got = True
        except Exception as exc:
            print("rx_error:", exc)

        if not got and utime.ticks_diff(utime.ticks_ms(), last_rx) > NO_SIGNAL_TIMEOUT_MS:
            print("NO_SIGNAL")
            last_rx = utime.ticks_ms()

        utime.sleep_ms(20)


main()


"""
Single hardware mapping file for indoor embedded mode.
Edit this file only when switching from simulation to real hardware.
"""

HARDWARE_CONFIG = {
    "backend": {
        "sensor": "mock",      # mock | hc_sr04
        "radio": "mock",       # mock | nrf24
        "display": "console",  # console | ssd1306
        "buzzer": "none",      # none | console | gpio
    },
    "sensors": {
        "model": "HC-SR04",
        "count": 1,
        "max_range_cm": 400.0,
        "raw_samples_per_tick": 3,
        "ema_alpha": 0.4,

        # Reliability controls
        "echo_timeout_us": 30000,
        "read_timeout_sec": 0.04,
        "read_retries": 2,

        # Pin placeholders (set real GPIO numbers later)
        "front": {"trig": "GPIO_TRIG_FRONT", "echo": "GPIO_ECHO_FRONT"},
        "left": {"trig": "GPIO_TRIG_LEFT", "echo": "GPIO_ECHO_LEFT"},
        "right": {"trig": "GPIO_TRIG_RIGHT", "echo": "GPIO_ECHO_RIGHT"},
    },
    "radio": {
        "model": "nRF24L01+",
        "payload_size": 32,
        "channel": 76,

        # SPI mapping placeholders (set real values later)
        "spi_bus": "SPI_BUS",
        "sck_pin": "GPIO_SPI_SCK",
        "mosi_pin": "GPIO_SPI_MOSI",
        "miso_pin": "GPIO_SPI_MISO",
        "ce_pin": "GPIO_CE",
        "csn_pin": "GPIO_CSN",

        # Pipe placeholders
        "tx_pipe": "0xE8E8F0F0E1",
        "rx_pipe": "0xE8E8F0F0D2",

        # RF tuning and reliability
        "pa_level": 1,
        "spi_baudrate": 1000000,
        "send_timeout_sec": 0.05,
        "send_retries": 2,
        "recv_timeout_sec": 0.02,
    },
    "display": {
        "model": "SSD1306",
        "width": 128,
        "height": 64,

        # I2C mapping placeholders (set real values later)
        "i2c_bus": "I2C_BUS",
        "scl_pin": "GPIO_I2C_SCL",
        "sda_pin": "GPIO_I2C_SDA",
        "i2c_address": "0x3C",
        "i2c_freq": 400000,
    },
    "buzzer": {
        "model": "GPIO_BUZZER",
        "pin": "GPIO_BUZZER_PIN",
        "active_high": True,
    },
    "thresholds_cm": {
        "danger": 30.0,
        "safe": 100.0,
    },
    "runtime": {
        # Unified scheduler rates
        "sensor_hz": 10.0,
        "radio_hz": 10.0,
        "ui_hz": 10.0,

        # Stability & fail-safe
        "state_hysteresis_cm": 5.0,
        "state_debounce_ticks": 2,
        "rx_no_signal_timeout_sec": 1.0,

        # Structured debug
        "debug_enabled": False,
        "drop_backlog": True,
    },
    "yolo": {
        "enabled": True,
        "backend": "udp",  # mock | udp
        "camera_stream_url": "http://ESP32_CAM_IP:81/stream",
        "udp_host": "0.0.0.0",
        "udp_port": 5005,
        "mock_rate_hz": 2.0,
        "confidence_threshold": 0.3,
        "temporal_smoothing_frames": 3,
    },
    "camera": {
        # Processing must stay geometry-true (no flip before inference).
        "flip_for_processing": False,
        # UI preview can be mirrored for user comfort.
        "flip_for_display": True,
    },
    "direction": {
        # Global direction convention: LEFT < 0.33, FRONT 0.33..0.66, RIGHT > 0.66
        "left_threshold": 0.33,
        "right_threshold": 0.66,
    },
    "vision": {
        "max_event_age_ms": 1500,
        "smoothing_window": 3,
        "direction_margin": 0.15,
        "latest_frame_only": True,
    },
    "audio": {
        "enabled": True,
        "folder": "audio/indoors",
        "multi_object_threshold": 2,
        "cooldown_ms": 1000,
        "confidence_threshold": 0.3,
    },
}

# MicroPython nRF24L01 driver (minimal, fixed payload)
from micropython import const
import utime

# Commands
_R_REGISTER = const(0x00)
_W_REGISTER = const(0x20)
_R_RX_PAYLOAD = const(0x61)
_W_TX_PAYLOAD = const(0xA0)
_FLUSH_TX = const(0xE1)
_FLUSH_RX = const(0xE2)
_NOP = const(0xFF)

# Registers
_CONFIG = const(0x00)
_EN_AA = const(0x01)
_EN_RXADDR = const(0x02)
_SETUP_AW = const(0x03)
_SETUP_RETR = const(0x04)
_RF_CH = const(0x05)
_RF_SETUP = const(0x06)
_STATUS = const(0x07)
_RX_ADDR_P0 = const(0x0A)
_RX_ADDR_P1 = const(0x0B)
_TX_ADDR = const(0x10)
_RX_PW_P0 = const(0x11)
_RX_PW_P1 = const(0x12)
_FIFO_STATUS = const(0x17)

# Bit masks
_MASK_RX_DR = const(1 << 6)
_MASK_TX_DS = const(1 << 5)
_MASK_MAX_RT = const(1 << 4)
_PWR_UP = const(1 << 1)
_PRIM_RX = const(1 << 0)


class NRF24L01:
    def __init__(self, spi, csn, ce, channel=76, payload_size=32):
        if payload_size < 1 or payload_size > 32:
            raise ValueError("payload_size must be 1..32")
        self.spi = spi
        self.csn = csn
        self.ce = ce
        self.payload_size = payload_size

        self.csn.init(self.csn.OUT, value=1)
        self.ce.init(self.ce.OUT, value=0)

        utime.sleep_ms(5)

        # Basic radio setup
        self.reg_write(_SETUP_AW, 0x03)      # 5-byte address
        self.reg_write(_RF_CH, channel & 0x7F)
        self.reg_write(_RF_SETUP, 0x06)      # 1Mbps, 0dBm
        self.reg_write(_SETUP_RETR, 0x3F)    # auto retry
        self.reg_write(_EN_AA, 0x03)         # auto-ack P0,P1
        self.reg_write(_EN_RXADDR, 0x03)     # enable P0,P1
        self.reg_write(_RX_PW_P0, payload_size)
        self.reg_write(_RX_PW_P1, payload_size)
        self.reg_write(_STATUS, _MASK_RX_DR | _MASK_TX_DS | _MASK_MAX_RT)

        # Power up (PTX by default)
        cfg = self.reg_read(_CONFIG)
        cfg |= _PWR_UP
        cfg &= ~_PRIM_RX
        self.reg_write(_CONFIG, cfg)
        utime.sleep_ms(5)

    def _csn(self, v):
        self.csn(v)

    def _ce(self, v):
        self.ce(v)

    def reg_read(self, reg):
        self._csn(0)
        self.spi.write(bytearray([_R_REGISTER | (reg & 0x1F)]))
        val = self.spi.read(1)[0]
        self._csn(1)
        return val

    def reg_write(self, reg, val):
        self._csn(0)
        self.spi.write(bytearray([_W_REGISTER | (reg & 0x1F), val & 0xFF]))
        self._csn(1)

    def reg_read_bytes(self, reg, n=5):
        self._csn(0)
        self.spi.write(bytearray([_R_REGISTER | (reg & 0x1F)]))
        data = self.spi.read(n)
        self._csn(1)
        return data

    def reg_write_bytes(self, reg, buf):
        self._csn(0)
        self.spi.write(bytearray([_W_REGISTER | (reg & 0x1F)]))
        self.spi.write(buf)
        self._csn(1)

    def flush_rx(self):
        self._csn(0)
        self.spi.write(bytearray([_FLUSH_RX]))
        self._csn(1)

    def flush_tx(self):
        self._csn(0)
        self.spi.write(bytearray([_FLUSH_TX]))
        self._csn(1)

    def open_tx_pipe(self, address):
        if len(address) != 5:
            raise ValueError("TX address must be 5 bytes")
        self.reg_write_bytes(_TX_ADDR, address)
        self.reg_write_bytes(_RX_ADDR_P0, address)  # for auto-ack
        self.reg_write(_RX_PW_P0, self.payload_size)

    def open_rx_pipe(self, pipe_id, address):
        if pipe_id not in (0, 1):
            raise ValueError("Only pipe 0 or 1 supported in this minimal driver")
        if len(address) != 5:
            raise ValueError("RX address must be 5 bytes")
        if pipe_id == 0:
            self.reg_write_bytes(_RX_ADDR_P0, address)
            self.reg_write(_RX_PW_P0, self.payload_size)
        else:
            self.reg_write_bytes(_RX_ADDR_P1, address)
            self.reg_write(_RX_PW_P1, self.payload_size)

        en = self.reg_read(_EN_RXADDR)
        en |= (1 << pipe_id)
        self.reg_write(_EN_RXADDR, en)

    def start_listening(self):
        cfg = self.reg_read(_CONFIG)
        cfg |= (_PWR_UP | _PRIM_RX)
        self.reg_write(_CONFIG, cfg)
        self.reg_write(_STATUS, _MASK_RX_DR | _MASK_TX_DS | _MASK_MAX_RT)
        self.flush_rx()
        self.flush_tx()
        self._ce(1)
        utime.sleep_us(130)

    def stop_listening(self):
        self._ce(0)
        cfg = self.reg_read(_CONFIG)
        cfg = (cfg | _PWR_UP) & (~_PRIM_RX)
        self.reg_write(_CONFIG, cfg)
        utime.sleep_us(130)

    def any(self):
        fifo = self.reg_read(_FIFO_STATUS)
        rx_empty = fifo & 0x01
        return rx_empty == 0

    def recv(self):
        self._csn(0)
        self.spi.write(bytearray([_R_RX_PAYLOAD]))
        data = self.spi.read(self.payload_size)
        self._csn(1)
        self.reg_write(_STATUS, _MASK_RX_DR)
        return data

    def send(self, buf, timeout_ms=200):
        if isinstance(buf, str):
            buf = buf.encode("utf-8")
        if len(buf) > self.payload_size:
            raise ValueError("payload too large")

        self.stop_listening()
        self.flush_tx()

        payload = bytes(buf) + b"\x00" * (self.payload_size - len(buf))
        self._csn(0)
        self.spi.write(bytearray([_W_TX_PAYLOAD]))
        self.spi.write(payload)
        self._csn(1)

        self._ce(1)
        utime.sleep_us(20)
        self._ce(0)

        start = utime.ticks_ms()
        while True:
            status = self.reg_read(_STATUS)
            if status & _MASK_TX_DS:
                self.reg_write(_STATUS, _MASK_TX_DS)
                return
            if status & _MASK_MAX_RT:
                self.reg_write(_STATUS, _MASK_MAX_RT)
                self.flush_tx()
                raise OSError("nRF24 send failed (MAX_RT)")
            if utime.ticks_diff(utime.ticks_ms(), start) > timeout_ms:
                self.flush_tx()
                raise OSError("nRF24 send timeout")


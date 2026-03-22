"""
HACKABOT Camera Provider.
Abstract interface for camera input; webcam mock for development;
stub for Pi camera. Frames captured asynchronously.
"""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


class CameraError(RuntimeError):
    pass


class CameraProvider(ABC):
    @abstractmethod
    async def start(self) -> None:
        ...

    @abstractmethod
    async def read(self) -> Optional[np.ndarray]:
        ...

    @abstractmethod
    async def stop(self) -> None:
        ...


class CameraSource:
    """Network stream camera source (DroidCam MJPEG/HTTP)."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None
        self._last_reconnect_attempt = 0.0
        self._last_frame_log = 0.0

    def is_open(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def reconnect(self) -> None:
        now = time.monotonic()
        if now - self._last_reconnect_attempt < 2.0:
            return
        self._last_reconnect_attempt = now
        print("[CAMERA] reconnecting...")
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cap = cv2.VideoCapture(self.source)
        if self.cap.isOpened():
            print("[CAMERA] connected to DroidCam stream")

    def read(self) -> Optional[np.ndarray]:
        if not self.is_open():
            self.reconnect()
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            print("[CAMERA] frame dropped")
            self.reconnect()
            return None
        now = time.monotonic()
        if now - self._last_frame_log >= 2.0:
            self._last_frame_log = now
            print("[CAMERA] frame received")
        return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class NetworkCameraProvider(CameraProvider):
    """DroidCam network stream provider with auto-reconnect."""

    def __init__(self, source: str, fallback_source: Optional[str] = None) -> None:
        self.source = source
        self.fallback_source = fallback_source
        self.camera = CameraSource(source=source)
        self._read_task: Optional[asyncio.Task] = None
        self._read_started_at = 0.0

    async def start(self) -> None:
        if cv2 is None:
            raise CameraError("opencv-python is not installed")
        self.camera.reconnect()
        if not self.camera.is_open():
            if self.fallback_source:
                print(f"[CAMERA] primary stream unavailable, trying fallback: {self.fallback_source}")
                self.camera = CameraSource(source=self.fallback_source)
                self.camera.reconnect()
            if not self.camera.is_open():
                print(
                    "[CAMERA] stream unavailable at startup; continuing in retry mode: "
                    f"{self.source}"
                    + (f" / {self.fallback_source}" if self.fallback_source else "")
                )
        self._read_task = None
        self._read_started_at = 0.0

    async def read(self) -> Optional[np.ndarray]:
        now = time.monotonic()
        # Keep at most one in-flight blocking read to avoid thread pileups/freezes.
        if self._read_task is None:
            self._read_started_at = now
            self._read_task = asyncio.create_task(asyncio.to_thread(self.camera.read))
            return None

        if not self._read_task.done():
            # If capture read blocks too long, drop task reference and force reconnect.
            if now - self._read_started_at > 1.2:
                print("[CAMERA] read stalled; forcing reconnect")
                self.camera.reconnect()
                self._read_task = None
            return None

        task = self._read_task
        self._read_task = None
        try:
            return task.result()
        except Exception:
            return None

    async def stop(self) -> None:
        self._read_task = None
        self.camera.close()


class VideoFileCameraProvider(CameraProvider):
    """Pre-recorded video for demo; optional loop."""

    def __init__(self, video_path: str | Path, loop: bool = True) -> None:
        self.video_path = str(video_path)
        self.loop = loop
        self.cap: Optional[cv2.VideoCapture] = None

    async def start(self) -> None:
        if cv2 is None:
            raise CameraError("opencv-python is not installed")
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise CameraError(f"Unable to open video file: {self.video_path}")

    async def read(self) -> Optional[np.ndarray]:
        if self.cap is None:
            return None
        ok, frame = await asyncio.to_thread(self.cap.read)
        if ok:
            return frame

        if not self.loop:
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = await asyncio.to_thread(self.cap.read)
        return frame if ok else None

    async def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class PiCameraProviderStub(CameraProvider):
    """Placeholder for Raspberry Pi camera; replace implementation without changing pipeline."""

    async def start(self) -> None:
        raise CameraError(
            "PiCameraProviderStub is a placeholder. Replace with Raspberry Pi camera implementation."
        )

    async def read(self) -> Optional[np.ndarray]:
        return None

    async def stop(self) -> None:
        return


def draw_debug_overlay(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    if cv2 is None:
        return frame
    output = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['label']} {det['direction']} {det['confidence']:.2f}"
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(output, label, (x1, max(12, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
    return output


def resize_frame(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    if cv2 is None:
        return frame
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

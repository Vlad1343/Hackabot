"""
HACKABOT Camera Provider.
Abstract interface for camera input; webcam mock for development;
stub for Pi camera. Frames captured asynchronously.
"""
from __future__ import annotations

import asyncio
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


class WebcamCameraProvider(CameraProvider):
    """Webcam for development on laptop."""

    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480) -> None:
        self.device_index = device_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None

    async def start(self) -> None:
        if cv2 is None:
            raise CameraError("opencv-python is not installed")
        self.cap = cv2.VideoCapture(self.device_index)
        if not self.cap.isOpened():
            raise CameraError(f"Unable to open webcam index {self.device_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    async def read(self) -> Optional[np.ndarray]:
        if self.cap is None:
            return None
        ok, frame = await asyncio.to_thread(self.cap.read)
        if not ok:
            return None
        return frame

    async def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


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

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from src.models.schemas import FrameData

logger = logging.getLogger(__name__)


class FrameGrabber:
    """Wraps OpenCV VideoCapture for frame acquisition."""

    def __init__(self, source: int | str = 0) -> None:
        self._source = source
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        self._frame_id = 0
        logger.info("FrameGrabber opened source: %s", source)

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def frame_size(self) -> tuple[int, int]:
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def read(self) -> FrameData | None:
        ret, frame = self._cap.read()
        if not ret:
            return None
        self._frame_id += 1
        return FrameData(
            frame=frame,
            timestamp=time.time(),
            frame_id=self._frame_id,
        )

    def release(self) -> None:
        self._cap.release()
        logger.info("FrameGrabber released source: %s", self._source)

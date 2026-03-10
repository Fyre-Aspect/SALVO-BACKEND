from __future__ import annotations

import logging

import numpy as np
from ultralytics import YOLO

from src.models.schemas import BoundingBox, Detection, FrameData

logger = logging.getLogger(__name__)

PERSON_CLASS_ID = 0  # COCO class 0 = person


class PersonDetector:
    """YOLO-based person detection."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self._model = YOLO(model_path)
        self._confidence_threshold = confidence_threshold
        self._device = device
        logger.info(
            "PersonDetector loaded model=%s conf=%.2f device=%s",
            model_path,
            confidence_threshold,
            device,
        )

    def detect(self, frame_data: FrameData) -> list[Detection]:
        results = self._model.predict(
            frame_data.frame,
            conf=self._confidence_threshold,
            classes=[PERSON_CLASS_ID],
            device=self._device,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append(
                    Detection(
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        confidence=conf,
                        class_id=PERSON_CLASS_ID,
                        frame_id=frame_data.frame_id,
                        timestamp=frame_data.timestamp,
                    )
                )

        return detections

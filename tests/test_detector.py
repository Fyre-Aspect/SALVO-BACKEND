"""Tests for PersonDetector (mocked YOLO)."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.models.schemas import FrameData
from src.vision.detector import PersonDetector


def _make_frame(frame_id=1):
    return FrameData(
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
        timestamp=1000.0,
        frame_id=frame_id,
    )


@patch("src.vision.detector.YOLO")
def test_detect_returns_detections(mock_yolo_cls):
    # Set up mock
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    # Mock a YOLO result with one detected person
    mock_box = MagicMock()
    mock_box.xyxy = [MagicMock()]
    mock_box.xyxy[0].tolist.return_value = [100.0, 50.0, 200.0, 300.0]
    mock_box.conf = [MagicMock()]
    mock_box.conf[0].__float__ = lambda self: 0.85

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_model.predict.return_value = [mock_result]

    detector = PersonDetector("yolov8n.pt", confidence_threshold=0.5)
    detections = detector.detect(_make_frame())

    assert len(detections) == 1
    assert detections[0].confidence == 0.85
    assert detections[0].bbox.x1 == 100.0
    assert detections[0].class_id == 0


@patch("src.vision.detector.YOLO")
def test_detect_empty_frame(mock_yolo_cls):
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    mock_result = MagicMock()
    mock_result.boxes = []
    mock_model.predict.return_value = [mock_result]

    detector = PersonDetector("yolov8n.pt")
    detections = detector.detect(_make_frame())

    assert len(detections) == 0

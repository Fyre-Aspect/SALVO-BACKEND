"""Integration test for the full pipeline with mocked camera and YOLO."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.models.schemas import BoundingBox, Detection, FrameData


def _make_frame(frame_id=1):
    return FrameData(
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
        timestamp=1000.0 + frame_id * 0.066,
        frame_id=frame_id,
    )


def _make_detection(cx, cy, frame_id=1, ts=1000.0):
    return Detection(
        bbox=BoundingBox(cx - 25, cy - 50, cx + 25, cy + 50),
        confidence=0.9,
        class_id=0,
        frame_id=frame_id,
        timestamp=ts,
    )


def test_pipeline_single_frame_flow():
    """Test that a single frame flows through detection → tracking → distress."""
    from src.distress.analyzer import DistressAnalyzer, DistressConfig
    from src.tracking.tracker import PersonTracker

    tracker = PersonTracker()
    analyzer = DistressAnalyzer(DistressConfig(min_track_seconds=0.0))

    detections = [_make_detection(320, 240, frame_id=1, ts=1.0)]
    tracked = tracker.update(detections)

    assert len(tracked) == 1
    assert tracked[0].track_id == 0

    events = analyzer.analyze(tracked, frame_id=1)
    # With only 1 frame of history, scores should be minimal
    # (features need multiple frames to compute meaningful scores)
    assert isinstance(events, list)


def test_pipeline_multi_frame_tracking():
    """Verify person ID persists across frames."""
    from src.tracking.tracker import PersonTracker

    tracker = PersonTracker()

    # Frame 1
    t1 = tracker.update([_make_detection(100, 200, frame_id=1, ts=1.0)])
    id1 = t1[0].track_id

    # Frame 2 — person moves slightly
    t2 = tracker.update([_make_detection(105, 202, frame_id=2, ts=1.066)])
    id2 = t2[0].track_id

    assert id1 == id2  # Same person

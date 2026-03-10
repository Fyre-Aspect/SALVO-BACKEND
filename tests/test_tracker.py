"""Tests for PersonTracker."""

from src.models.schemas import BoundingBox, Detection
from src.tracking.tracker import PersonTracker


def _make_detection(cx, cy, w=50, h=100, frame_id=1, ts=1000.0):
    return Detection(
        bbox=BoundingBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
        confidence=0.9,
        class_id=0,
        frame_id=frame_id,
        timestamp=ts,
    )


def test_register_new_tracks():
    tracker = PersonTracker()
    dets = [_make_detection(100, 200), _make_detection(300, 200)]
    tracked = tracker.update(dets)

    assert len(tracked) == 2
    ids = {t.track_id for t in tracked}
    assert len(ids) == 2  # Unique IDs


def test_track_persistence():
    tracker = PersonTracker()

    # Frame 1: person at (100, 200)
    tracked = tracker.update([_make_detection(100, 200, ts=1.0)])
    assert len(tracked) == 1
    first_id = tracked[0].track_id

    # Frame 2: person moved slightly to (110, 205)
    tracked = tracker.update([_make_detection(110, 205, ts=1.1)])
    assert len(tracked) == 1
    assert tracked[0].track_id == first_id  # Same ID


def test_track_disappears_after_max():
    tracker = PersonTracker(max_disappeared=3)

    tracker.update([_make_detection(100, 200)])
    assert len(tracker.tracks) == 1

    # Empty frames
    for _ in range(4):
        tracker.update([])

    assert len(tracker.tracks) == 0  # Track removed


def test_new_far_detection_creates_new_track():
    tracker = PersonTracker(max_distance=50)

    tracker.update([_make_detection(100, 200)])
    # Detection far away — should create a new track, not match
    tracked = tracker.update([_make_detection(100, 200), _make_detection(500, 500)])

    assert len(tracked) == 2


def test_empty_detections():
    tracker = PersonTracker(max_disappeared=5)
    tracked = tracker.update([])
    assert len(tracked) == 0

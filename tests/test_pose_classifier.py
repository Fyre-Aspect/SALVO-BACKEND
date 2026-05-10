"""Unit tests for the gesture classifier rules.

We bypass the YOLO model and call the rule engine directly with synthetic
17-keypoint arrays so the rules are tested deterministically.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from src.distress.pose_classifier import (
    GestureConfig,
    PoseGestureClassifier,
)
from src.models.schemas import GestureType


def _kp(positions: dict[int, tuple[float, float]], conf: float = 0.9) -> np.ndarray:
    """Build a (17,3) keypoint array; missing indices are zero-confidence."""
    arr = np.zeros((17, 3), dtype=np.float32)
    for idx, (x, y) in positions.items():
        arr[idx] = [x, y, conf]
    return arr


def _make_classifier(min_hold: int = 1) -> PoseGestureClassifier:
    with patch("src.distress.pose_classifier.YOLO", return_value=MagicMock()):
        return PoseGestureClassifier(
            GestureConfig(min_hold_frames=min_hold)
        )


def test_both_arms_above_head_fires_sos():
    c = _make_classifier(min_hold=1)
    # Image coords: y grows downward. Head/wrists at y=50, shoulders at y=200.
    kps = _kp({
        0: (320, 50),    # nose
        5: (280, 200),   # L shoulder
        6: (360, 200),   # R shoulder
        9: (260, 30),    # L wrist (above head)
        10: (380, 30),   # R wrist (above head)
    })
    result = c._classify_keypoints(track_id=1, kp=kps)
    assert result.gesture == GestureType.SOS_BOTH_ARMS


def test_crossed_arms_overhead_takes_priority():
    c = _make_classifier(min_hold=1)
    # Wrists crossed past midline (left wrist on right side, vice versa).
    kps = _kp({
        0: (320, 60),
        5: (280, 200),  # L shoulder x=280
        6: (360, 200),  # R shoulder x=360 -> midline 320
        9: (370, 30),   # L wrist on RIGHT side of midline
        10: (270, 30),  # R wrist on LEFT side of midline
    })
    result = c._classify_keypoints(track_id=2, kp=kps)
    assert result.gesture == GestureType.CROSSED_ARMS_OVERHEAD


def test_walking_keypoints_do_not_fire():
    c = _make_classifier(min_hold=1)
    # Standing/walking: wrists below shoulders, hands at hip level.
    kps = _kp({
        0: (320, 80),
        5: (280, 200),
        6: (360, 200),
        9: (270, 350),   # wrists at hip level
        10: (370, 350),
    })
    result = c._classify_keypoints(track_id=3, kp=kps)
    assert result.gesture == GestureType.NONE


def test_single_arm_wave_requires_oscillation():
    c = _make_classifier(min_hold=1)
    # First call: one arm up, no history of waving yet.
    kps_static = _kp({
        0: (320, 80),
        5: (280, 200),
        6: (360, 200),
        9: (270, 30),    # L wrist above head, R wrist absent
    })
    res1 = c._classify_keypoints(track_id=4, kp=kps_static)
    # No oscillation history — should not fire SOS_SINGLE_ARM
    assert res1.gesture == GestureType.NONE

    # Now build oscillation history by sending varying wrist X values
    for x in [200, 350, 200, 350, 200, 350]:
        kps_wave = _kp({
            0: (320, 80),
            5: (280, 200),
            6: (360, 200),
            9: (x, 30),
        })
        res = c._classify_keypoints(track_id=4, kp=kps_wave)
    assert res.gesture == GestureType.SOS_SINGLE_ARM


def test_min_hold_frames_suppresses_one_off_arm_raise():
    c = _make_classifier(min_hold=4)
    kps = _kp({
        0: (320, 50),
        5: (280, 200),
        6: (360, 200),
        9: (260, 30),
        10: (380, 30),
    })
    # First 3 frames: streak not yet hit, gesture suppressed
    for _ in range(3):
        res = c._classify_keypoints(track_id=5, kp=kps)
        assert res.gesture == GestureType.NONE
    # 4th frame: streak met, gesture fires
    res = c._classify_keypoints(track_id=5, kp=kps)
    assert res.gesture == GestureType.SOS_BOTH_ARMS


def test_drowning_posture_head_below_shoulders_with_motion():
    c = _make_classifier(min_hold=1)
    # Build wrist motion history with HIGH amplitude (>60 px) so it clears
    # the production drowning_motion_px threshold (60). Walking arm-swing
    # is well below this and must NOT trigger the rule.
    for x in [200, 290, 200, 290, 200, 290]:
        kps_motion = _kp({
            0: (320, 250),    # head clearly BELOW shoulders (250 > 200+15)
            5: (280, 200),
            6: (360, 200),
            9: (x, 220),
            10: (x + 100, 220),
        })
        res = c._classify_keypoints(track_id=6, kp=kps_motion)
    assert res.gesture == GestureType.DROWNING_POSTURE


def test_drowning_does_not_fire_when_nose_keypoint_is_missing():
    """Distant pedestrians often have an undetected nose. Rule must require
    POSITIVE evidence of head-below-shoulders, not absence of detection."""
    c = _make_classifier(min_hold=1)
    for x in [200, 290, 200, 290, 200, 290]:
        kps = _kp({
            # nose (idx 0) intentionally omitted -> low confidence
            5: (280, 200),
            6: (360, 200),
            9: (x, 220),
            10: (x + 100, 220),
        })
        res = c._classify_keypoints(track_id=7, kp=kps)
    assert res.gesture == GestureType.NONE


def test_drowning_does_not_fire_on_walking_arm_swing():
    """Walking arm-swing produces ~30-50 px lateral wrist motion. The
    drowning rule's 60 px threshold must reject it."""
    c = _make_classifier(min_hold=1)
    for x in [300, 340, 300, 340, 300, 340]:  # ~40 px swing
        kps = _kp({
            0: (320, 250),     # head below shoulders (worst case)
            5: (280, 200),
            6: (360, 200),
            9: (x, 220),
            10: (x + 100, 220),
        })
        res = c._classify_keypoints(track_id=8, kp=kps)
    assert res.gesture == GestureType.NONE
